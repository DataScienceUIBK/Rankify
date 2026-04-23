"""
Unit tests for Rankify rerankers: DuoT5, RankLLaMA, and DeAR.

These tests mock the underlying model objects so they run fast without
requiring any GPU or network access.
"""

import copy
import math
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from rankify.dataset.dataset import Answer, Context, Document, Question


# ---------------------------------------------------------------------------
# Helpers – thin stubs for the heavy HuggingFace objects
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(token_false_id=0, token_true_id=1):
    """Return a callable mock that behaves like a T5Tokenizer."""
    tok = MagicMock()
    tok.convert_tokens_to_ids.side_effect = lambda t: (
        token_false_id if "false" in t else token_true_id
    )

    # When called as tok(prompts, ...) return a dict-like object with .to()
    def _call_side_effect(prompts, **kwargs):
        batch = len(prompts) if isinstance(prompts, list) else 1
        ids = torch.zeros(batch, 4, dtype=torch.long)
        mask = torch.ones(batch, 4, dtype=torch.long)
        result = MagicMock()
        result.__getitem__ = lambda self, k: ids if k == "input_ids" else mask
        result.items.return_value = [("input_ids", ids), ("attention_mask", mask)]
        result.to = lambda device: result
        return result

    tok.side_effect = _call_side_effect
    return tok


def _make_mock_model(vocab_size=32128, true_id=1, false_id=0, doc0_wins=True):
    """
    Return a mock T5ForConditionalGeneration.

    If *doc0_wins* is True the model always predicts that Document0 is more
    relevant (logit for token ``true_id`` is larger).  Otherwise Document1 wins.
    """
    model = MagicMock()
    model.config.decoder_start_token_id = 0

    # Encoder stub
    encoder = MagicMock()
    encoder.return_value = MagicMock()
    model.get_encoder.return_value = encoder

    # prepare_inputs_for_generation → just pass through
    model.prepare_inputs_for_generation.side_effect = (
        lambda decode_ids, **kw: {"input_ids": decode_ids}
    )

    # Forward pass – produce logits that favour doc0 or doc1
    def _forward(**kwargs):
        batch = kwargs["input_ids"].size(0)
        logits = torch.zeros(batch, 1, vocab_size)
        if doc0_wins:
            logits[:, 0, true_id] = 10.0   # doc0 > doc1
            logits[:, 0, false_id] = -10.0
        else:
            logits[:, 0, true_id] = -10.0  # doc1 > doc0
            logits[:, 0, false_id] = 10.0
        out = MagicMock()
        out.logits = logits
        # Support both outputs[0] and out.logits
        out.__getitem__ = lambda self, i: logits
        return out

    model.side_effect = _forward
    model.eval.return_value = model
    model.to.return_value = model
    return model


# ---------------------------------------------------------------------------
# Patch context manager that sets up the entire HuggingFace import chain
# ---------------------------------------------------------------------------

def _patch_hf(doc0_wins=True):
    """
    Returns a list of patches that replace T5 loading with in-memory stubs.
    """
    mock_tok = _make_mock_tokenizer(token_false_id=0, token_true_id=1)
    mock_mdl = _make_mock_model(doc0_wins=doc0_wins)

    patches = [
        patch(
            "rankify.models.duot5.T5Tokenizer.from_pretrained",
            return_value=mock_tok,
        ),
        patch(
            "rankify.models.duot5.T5ForConditionalGeneration.from_pretrained",
            return_value=mock_mdl,
        ),
    ]
    return patches, mock_tok, mock_mdl


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestDuoT5GetOutputTokens(unittest.TestCase):
    """Tests for the static token-lookup helper."""

    def test_known_model_returns_correct_tokens(self):
        from rankify.models.duot5 import DuoT5

        false_tok, true_tok = DuoT5._get_output_tokens(
            "castorini/duot5-base-msmarco"
        )
        self.assertEqual(false_tok, "▁false")
        self.assertEqual(true_tok, "▁true")

    def test_unknown_model_falls_back_to_defaults(self):
        from rankify.models.duot5 import DuoT5

        false_tok, true_tok = DuoT5._get_output_tokens(
            "some-unknown-model-xyz"
        )
        self.assertEqual(false_tok, "▁false")
        self.assertEqual(true_tok, "▁true")

    def test_explicit_tokens_override_lookup(self):
        from rankify.models.duot5 import DuoT5

        false_tok, true_tok = DuoT5._get_output_tokens(
            "castorini/duot5-base-msmarco",
            token_false="no",
            token_true="yes",
        )
        self.assertEqual(false_tok, "no")
        self.assertEqual(true_tok, "yes")


class TestDuoT5ScoreMatrix(unittest.TestCase):
    """
    Tests for the pairwise score-matrix computation.

    The matrix is computed purely by the mock model; actual T5 weights are
    never loaded.
    """

    def _build_reranker(self, doc0_wins=True):
        from rankify.models.duot5 import DuoT5

        patches, tok, mdl = _patch_hf(doc0_wins=doc0_wins)
        for p in patches:
            p.start()
        reranker = DuoT5.__new__(DuoT5)
        reranker._device = torch.device("cpu")
        reranker._context_size = 512
        reranker.batch_size = 32
        reranker.use_amp = False
        reranker.inputs_template = (
            "Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"
        )
        reranker.tokenizer = tok
        reranker.model = mdl
        reranker.token_false_id = 0
        reranker.token_true_id = 1
        for p in patches:
            p.stop()
        return reranker

    def test_score_matrix_shape(self):
        reranker = self._build_reranker()
        docs = ["doc A", "doc B", "doc C"]
        matrix = reranker._get_pairwise_scores("query", docs, max_length=32)
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)

    def test_diagonal_is_zero(self):
        reranker = self._build_reranker()
        docs = ["doc A", "doc B", "doc C"]
        matrix = reranker._get_pairwise_scores("query", docs, max_length=32)
        for i in range(3):
            self.assertEqual(matrix[i][i], 0.0)

    def test_doc0_wins_gives_high_off_diagonal(self):
        """When doc0_wins=True every off-diagonal score should be high (close to 0 in log-space)."""
        reranker = self._build_reranker(doc0_wins=True)
        docs = ["doc A", "doc B", "doc C"]
        matrix = reranker._get_pairwise_scores("query", docs, max_length=32)
        for i in range(3):
            for j in range(3):
                if i != j:
                    # log_softmax([10, −10]) ≈ 0 for the winning token → close to 0
                    self.assertGreater(matrix[i][j], -1.0)

    def test_doc1_wins_gives_low_off_diagonal(self):
        """When doc0_wins=False the off-diagonal scores should be low (very negative)."""
        reranker = self._build_reranker(doc0_wins=False)
        docs = ["doc A", "doc B", "doc C"]
        matrix = reranker._get_pairwise_scores("query", docs, max_length=32)
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertLess(matrix[i][j], -5.0)


class TestDuoT5Rank(unittest.TestCase):
    """End-to-end tests for the public ``rank`` method."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _make_document():
        question = Question("When did Thomas Edison invent the light bulb?")
        answers = Answer(["1879"])
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=1),
            Context(text="Thomas Edison tried to invent a device for cars but failed", id=2),
            Context(text="Coffee is good for diet", id=3),
            Context(text="Thomas Edison invented the light bulb in 1879", id=4),
            Context(text="Thomas Edison worked with electricity", id=5),
        ]
        return Document(question=question, answers=answers, contexts=contexts)

    def _build_reranker_via_new(self, doc0_wins=True):
        """
        Instantiate DuoT5 while bypassing the real HF loading.
        We use __new__ + manual attribute assignment to avoid loading
        any weights from disk.
        """
        from rankify.models.duot5 import DuoT5

        patches, tok, mdl = _patch_hf(doc0_wins=doc0_wins)
        for p in patches:
            p.start()

        reranker = DuoT5.__new__(DuoT5)
        reranker._device = torch.device("cpu")
        reranker._context_size = 512
        reranker.batch_size = 32
        reranker.use_amp = False
        reranker.inputs_template = (
            "Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"
        )
        reranker.tokenizer = tok
        reranker.model = mdl
        reranker.token_false_id = 0
        reranker.token_true_id = 1

        for p in patches:
            p.stop()
        return reranker

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_rank_returns_all_contexts(self):
        reranker = self._build_reranker_via_new()
        doc = self._make_document()
        reranker.rank([doc])
        self.assertEqual(len(doc.reorder_contexts), len(doc.contexts))

    def test_rank_preserves_context_objects(self):
        """reorder_contexts must contain the same texts as the original contexts."""
        reranker = self._build_reranker_via_new()
        doc = self._make_document()
        original_texts = {ctx.text for ctx in doc.contexts}
        reranker.rank([doc])
        reranked_texts = {ctx.text for ctx in doc.reorder_contexts}
        self.assertEqual(original_texts, reranked_texts)

    def test_rank_assigns_scores(self):
        """Every context in reorder_contexts should have a numeric score."""
        reranker = self._build_reranker_via_new()
        doc = self._make_document()
        reranker.rank([doc])
        for ctx in doc.reorder_contexts:
            self.assertIsInstance(ctx.score, (int, float))

    def test_rank_scores_are_descending(self):
        """reorder_contexts must be sorted highest-score first."""
        reranker = self._build_reranker_via_new()
        doc = self._make_document()
        reranker.rank([doc])
        scores = [ctx.score for ctx in doc.reorder_contexts]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_single_context(self):
        """A document with only one context should not crash."""
        reranker = self._build_reranker_via_new()
        question = Question("Test question?")
        answers = Answer(["test answer"])
        contexts = [Context(text="Only one passage here.", id=0)]
        doc = Document(question=question, answers=answers, contexts=contexts)
        reranker.rank([doc])
        self.assertEqual(len(doc.reorder_contexts), 1)

    def test_rank_does_not_mutate_original_contexts(self):
        """rank() must not alter the original ``document.contexts`` list."""
        reranker = self._build_reranker_via_new()
        doc = self._make_document()
        original_texts = [ctx.text for ctx in doc.contexts]
        reranker.rank([doc])
        self.assertEqual([ctx.text for ctx in doc.contexts], original_texts)

    def test_rank_multiple_documents(self):
        """rank() should process a list of multiple documents correctly."""
        reranker = self._build_reranker_via_new()
        docs = [self._make_document(), self._make_document()]
        reranker.rank(docs)
        for doc in docs:
            self.assertEqual(len(doc.reorder_contexts), len(doc.contexts))


class TestDuoT5Registration(unittest.TestCase):
    """Verify DuoT5 is correctly wired into the method/model registries."""

    def test_duot5_in_hf_pre_defind_models(self):
        from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS

        self.assertIn("duot5", HF_PRE_DEFIND_MODELS)
        self.assertIn("duot5-base-msmarco", HF_PRE_DEFIND_MODELS["duot5"])
        self.assertEqual(
            HF_PRE_DEFIND_MODELS["duot5"]["duot5-base-msmarco"],
            "castorini/duot5-base-msmarco",
        )

    def test_duot5_prediction_tokens_registered(self):
        from rankify.utils.pre_defind_models import PREDICTION_TOKENS

        self.assertIn("castorini/duot5-base-msmarco", PREDICTION_TOKENS)
        self.assertEqual(
            PREDICTION_TOKENS["castorini/duot5-base-msmarco"],
            ["▁false", "▁true"],
        )

    def test_duot5_in_method_map(self):
        try:
            from rankify.utils.pre_defined_methods import METHOD_MAP
        except ImportError as exc:
            self.skipTest(f"Optional dependency not installed: {exc}")
        self.assertIn("duot5", METHOD_MAP)

    def test_method_map_points_to_duot5_class(self):
        from rankify.models.duot5 import DuoT5
        try:
            from rankify.utils.pre_defined_methods import METHOD_MAP
        except ImportError as exc:
            self.skipTest(f"Optional dependency not installed: {exc}")
        self.assertIs(METHOD_MAP["duot5"], DuoT5)


# ---------------------------------------------------------------------------
# Helpers for cross-encoder style rerankers (RankLLaMA & DeAR)
# ---------------------------------------------------------------------------

def _make_cross_encoder_tokenizer():
    """Return a mock tokenizer for cross-encoder style models."""
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"

    def _call_side_effect(q_batch, d_batch, **kwargs):
        batch = len(q_batch) if isinstance(q_batch, list) else 1
        ids = torch.zeros(batch, 8, dtype=torch.long)
        mask = torch.ones(batch, 8, dtype=torch.long)
        result = MagicMock()
        result.__iter__ = lambda self: iter([])
        result.items.return_value = [("input_ids", ids), ("attention_mask", mask)]
        result.to = lambda device: result
        result.__getitem__ = lambda self, k: ids if k == "input_ids" else mask
        return result

    tok.side_effect = _call_side_effect
    return tok


def _make_cross_encoder_model(score_value=1.0):
    """Return a mock AutoModelForSequenceClassification."""
    model = MagicMock()

    def _forward(**kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is not None:
            batch = input_ids.shape[0]
        else:
            batch = 1
        out = MagicMock()
        out.logits = torch.full((batch, 1), score_value)
        return out

    model.side_effect = _forward
    model.eval.return_value = model
    model.to.return_value = model
    model.device = torch.device("cpu")
    return model


# ---------------------------------------------------------------------------
# RankLLaMA tests
# ---------------------------------------------------------------------------

class TestRankLLaMAReranker(unittest.TestCase):
    """Tests for the RankLLaMAReranker using mocked PEFT and HuggingFace objects."""

    @staticmethod
    def _make_document():
        question = Question("When did Thomas Edison invent the light bulb?")
        answers = Answer(["1879"])
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=1),
            Context(text="Thomas Edison invented the light bulb in 1879", id=2),
            Context(text="Coffee is good for diet", id=3),
        ]
        return Document(question=question, answers=answers, contexts=contexts)

    def _build_reranker(self, score_value=1.0):
        """Build a RankLLaMAReranker with all heavy I/O mocked out."""
        from rankify.models.rankllama_reranker import RankLLaMAReranker

        mock_tok = _make_cross_encoder_tokenizer()
        mock_mdl = _make_cross_encoder_model(score_value)

        reranker = RankLLaMAReranker.__new__(RankLLaMAReranker)
        reranker.method = "rankllama"
        reranker.model_name = "castorini/rankllama-v1-7b-lora-passage"
        reranker.device = "cpu"
        reranker.batch_size = 8
        reranker.max_length = 512
        reranker.tokenizer = mock_tok
        reranker.model = mock_mdl
        return reranker

    def test_rank_returns_all_contexts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        self.assertEqual(len(doc.reorder_contexts), len(doc.contexts))

    def test_rank_preserves_context_texts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        original_texts = {ctx.text for ctx in doc.contexts}
        reranker.rank([doc])
        self.assertEqual(original_texts, {ctx.text for ctx in doc.reorder_contexts})

    def test_rank_assigns_scores(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        for ctx in doc.reorder_contexts:
            self.assertIsInstance(ctx.score, (int, float))

    def test_rank_scores_descending(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        scores = [ctx.score for ctx in doc.reorder_contexts]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_does_not_mutate_original_contexts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        original_texts = [ctx.text for ctx in doc.contexts]
        reranker.rank([doc])
        self.assertEqual([ctx.text for ctx in doc.contexts], original_texts)

    def test_rank_includes_title_in_doc_string(self):
        """Contexts with a title should produce 'document: <title> <text>' strings."""
        reranker = self._build_reranker()
        question = Question("What is a camelid?")
        contexts = [
            Context(text="The llama is a domesticated South American camelid.", title="Llama", id=0),
        ]
        doc = Document(question=question, answers=Answer([""]), contexts=contexts)

        captured = []

        def _capturing_score(q_batch, d_batch):
            captured.extend(d_batch)
            return [1.0] * len(q_batch)

        reranker._score_batched = _capturing_score
        reranker.rank([doc])

        self.assertEqual(len(captured), 1)
        self.assertIn("Llama", captured[0])
        self.assertIn("camelid", captured[0])

    def test_rankllama_in_hf_pre_defind_models(self):
        from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS

        self.assertIn("rankllama", HF_PRE_DEFIND_MODELS)
        self.assertIn(
            "rankllama-v1-7b-lora-passage",
            HF_PRE_DEFIND_MODELS["rankllama"],
        )

    def test_rankllama_in_method_map(self):
        try:
            from rankify.utils.pre_defined_methods import METHOD_MAP
        except ImportError as exc:
            self.skipTest(f"Optional dependency not installed: {exc}")
        self.assertIn("rankllama", METHOD_MAP)

    def test_rankllama_raises_without_peft(self):
        """RankLLaMAReranker.__init__ should raise ImportError when peft is absent."""
        import rankify.models.rankllama_reranker as module

        orig = module.PEFT_AVAILABLE
        module.PEFT_AVAILABLE = False
        try:
            with self.assertRaises(ImportError):
                module.RankLLaMAReranker(
                    method="rankllama",
                    model_name="castorini/rankllama-v1-7b-lora-passage",
                )
        finally:
            module.PEFT_AVAILABLE = orig


# ---------------------------------------------------------------------------
# DeAR Reranker tests
# ---------------------------------------------------------------------------

class TestDeARReranker(unittest.TestCase):
    """Tests for the DeARReranker using mocked HuggingFace objects."""

    @staticmethod
    def _make_document():
        question = Question("When did Thomas Edison invent the light bulb?")
        answers = Answer(["1879"])
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=1),
            Context(text="Thomas Edison invented the light bulb in 1879", id=2),
            Context(text="Coffee is good for diet", id=3),
        ]
        return Document(question=question, answers=answers, contexts=contexts)

    def _build_reranker(self, score_value=1.0):
        """Build a DeARReranker with all heavy I/O mocked out."""
        from rankify.models.dear_reranker import DeARReranker

        mock_tok = _make_cross_encoder_tokenizer()
        mock_mdl = _make_cross_encoder_model(score_value)

        reranker = DeARReranker.__new__(DeARReranker)
        reranker.method = "dear_reranker"
        reranker.model_name = "abdoelsayed/dear-3b-reranker-ce-v1"
        reranker.device = "cpu"
        reranker.batch_size = 32
        reranker.max_length = 228
        reranker.tokenizer = mock_tok
        reranker.model = mock_mdl
        return reranker

    def test_rank_returns_all_contexts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        self.assertEqual(len(doc.reorder_contexts), len(doc.contexts))

    def test_rank_preserves_context_texts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        original_texts = {ctx.text for ctx in doc.contexts}
        reranker.rank([doc])
        self.assertEqual(original_texts, {ctx.text for ctx in doc.reorder_contexts})

    def test_rank_assigns_scores(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        for ctx in doc.reorder_contexts:
            self.assertIsInstance(ctx.score, (int, float))

    def test_rank_scores_descending(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        reranker.rank([doc])
        scores = [ctx.score for ctx in doc.reorder_contexts]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_does_not_mutate_original_contexts(self):
        reranker = self._build_reranker()
        doc = self._make_document()
        original_texts = [ctx.text for ctx in doc.contexts]
        reranker.rank([doc])
        self.assertEqual([ctx.text for ctx in doc.contexts], original_texts)

    def test_query_format(self):
        """Document strings sent to the model must start with 'document:'."""
        reranker = self._build_reranker()
        doc = self._make_document()

        captured_docs = []

        def _capturing_score(q_batch, d_batch):
            captured_docs.extend(d_batch)
            return [1.0] * len(q_batch)

        reranker._score_batched = _capturing_score
        reranker.rank([doc])

        for d in captured_docs:
            self.assertTrue(
                d.startswith("document:"),
                f"Expected 'document:' prefix, got: {d!r}",
            )

    def test_rank_multiple_documents(self):
        reranker = self._build_reranker()
        docs = [self._make_document(), self._make_document()]
        reranker.rank(docs)
        for doc in docs:
            self.assertEqual(len(doc.reorder_contexts), len(doc.contexts))

    def test_dear_in_hf_pre_defind_models(self):
        from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS

        self.assertIn("dear_reranker", HF_PRE_DEFIND_MODELS)
        self.assertIn(
            "dear-3b-reranker-ce-v1",
            HF_PRE_DEFIND_MODELS["dear_reranker"],
        )
        self.assertEqual(
            HF_PRE_DEFIND_MODELS["dear_reranker"]["dear-3b-reranker-ce-v1"],
            "abdoelsayed/dear-3b-reranker-ce-v1",
        )

    def test_dear_in_method_map(self):
        try:
            from rankify.utils.pre_defined_methods import METHOD_MAP
        except ImportError as exc:
            self.skipTest(f"Optional dependency not installed: {exc}")
        self.assertIn("dear_reranker", METHOD_MAP)

    def test_method_map_points_to_dear_class(self):
        from rankify.models.dear_reranker import DeARReranker
        try:
            from rankify.utils.pre_defined_methods import METHOD_MAP
        except ImportError as exc:
            self.skipTest(f"Optional dependency not installed: {exc}")
        self.assertIs(METHOD_MAP["dear_reranker"], DeARReranker)


if __name__ == "__main__":
    unittest.main()
