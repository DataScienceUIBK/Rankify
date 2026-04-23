import logging
from math import ceil
from typing import List, Optional

import copy
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from rankify.dataset.dataset import Document
from rankify.models.base import BaseRanking
from rankify.utils.pre_defind_models import PREDICTION_TOKENS

logger = logging.getLogger(__name__)


class DuoT5(BaseRanking):
    """
    Implements **DuoT5 Reranking**, a **pairwise sequence-to-sequence ranking approach**
    using **T5** to compare pairs of documents for relevance assessment.

    DuoT5 ranks passages by generating pairwise relevance predictions ("true"/"false")
    for all query-document-document triples. For each pair of documents, it predicts
    whether Document0 is more relevant than Document1. The final ranking is derived by
    aggregating pairwise preference probabilities into a per-document score.

    References:
        - **Pradeep et al. (2021)**: *The Expando-Mono-Duo Design Pattern for Text Ranking
          with Pretrained Sequence-to-Sequence Models*.
          [Paper](https://arxiv.org/abs/2101.05667)

    Attributes:
        method (str, optional): The **name of the reranking method**.
        model_name (str, optional): The **name of the pre-trained DuoT5 model**.
        _device (torch.device): The **device (CPU/GPU)** on which the model runs.
        _context_size (int): The **maximum sequence length** for encoding.
        batch_size (int): The **batch size** used for pairwise inference.
        use_amp (bool): Whether to use **Automatic Mixed Precision** for faster inference.
        tokenizer (T5Tokenizer): The **T5 tokenizer** for processing inputs.
        model (T5ForConditionalGeneration): The **pretrained T5 model** for ranking.
        token_true_id (int): **Token ID** corresponding to the **"true"** label.
        token_false_id (int): **Token ID** corresponding to the **"false"** label.

    Examples:
        **Basic Usage:**
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("When did Thomas Edison invent the light bulb?")
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=0),
            Context(text="Thomas Edison tried to invent a device for cars but failed", id=1),
            Context(text="Thomas Edison invented the light bulb in 1879", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize DuoT5 Reranker
        model = Reranking(method='duot5', model_name='duot5-base-msmarco')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """

    def __init__(self, method=None, model_name=None, **kwargs):
        """
        Initializes **DuoT5** for pairwise reranking tasks.

        Args:
            method (str, optional): The **reranking method name**.
            model_name (str, optional): The **name of the pretrained DuoT5 model**.
            kwargs (dict): Additional parameters:
                - device (str, optional): Device (`"cuda"`, `"cpu"`). Default: auto-detect.
                - context_size (int, optional): Max sequence length (default: ``512``).
                - batch_size (int, optional): Batch size for pairwise inference (default: ``32``).
                - use_amp (bool, optional): Use AMP on CUDA (default: auto-detect).
        """
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device) if isinstance(device, str) else device
        self._context_size = kwargs.get("context_size", 512)
        self.batch_size = kwargs.get("batch_size", 32)
        self.use_amp = kwargs.get("use_amp", self._device.type == "cuda")

        # Input template: predict whether Document0 is more relevant than Document1
        self.inputs_template = (
            "Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"
        )

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        dtype = (
            torch.float16
            if self._device.type == "cuda" and self.use_amp
            else torch.float32
        )
        self.model = (
            T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            .to(self._device)
            .eval()
        )

        token_false, token_true = self._get_output_tokens(model_name)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        logger.info(
            f"DuoT5 initialised — true token ID: {self.token_true_id}, "
            f"false token ID: {self.token_false_id}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks contexts within each document using **DuoT5** pairwise comparisons.

        For every document, all ordered pairs of contexts are compared and the
        probability that context *i* is more relevant than context *j* is computed.
        Each context receives an aggregate score equal to the sum of its pairwise
        preference probabilities and contexts are sorted in descending order of
        that score.

        Args:
            documents (List[Document]): A list of **Document** instances to rerank.

        Returns:
            List[Document]: Documents with updated **``reorder_contexts``** after reranking.
        """
        for doc in tqdm(documents, desc="Reranking Documents"):
            query = doc.question.question
            doc_texts = [ctx.text for ctx in doc.contexts]
            n = len(doc_texts)

            if n <= 1:
                doc.reorder_contexts = copy.deepcopy(doc.contexts)
                continue

            score_matrix = self._get_pairwise_scores(
                query, doc_texts, max_length=self._context_size
            )

            # Aggregate: score(i) = Σ P(doc_i > doc_j)  for all j ≠ i
            agg_scores = [
                sum(score_matrix[i][j] for j in range(n) if j != i)
                for i in range(n)
            ]

            contexts = copy.deepcopy(doc.contexts)
            for ctx, score in zip(contexts, agg_scores):
                ctx.score = score

            doc.reorder_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)

        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _get_pairwise_scores(
        self,
        query: str,
        docs: List[str],
        max_length: int = 512,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute a pairwise preference score matrix for all ordered document pairs.

        ``score_matrix[i][j]`` is the model probability that document *i* is more
        relevant than document *j* (i.e. the log-softmax probability of the "true"
        token when document *i* is placed in the Document0 slot).

        Args:
            query (str): The query string.
            docs (List[str]): List of document texts to compare.
            max_length (int, optional): Maximum tokenisation length (default: ``512``).
            batch_size (int, optional): Override the instance ``batch_size``.

        Returns:
            List[List[float]]: ``n × n`` score matrix (diagonal entries are ``0.0``).
        """
        n = len(docs)
        score_matrix = [[0.0] * n for _ in range(n)]

        if batch_size is None:
            batch_size = self.batch_size

        # All ordered pairs (i, j) with i ≠ j
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            prompts = [
                self.inputs_template.format(
                    query=query, doc0=docs[i], doc1=docs[j]
                )
                for i, j in batch_pairs
            ]

            tokenized = self.tokenizer(
                prompts,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
                return_attention_mask=True,
            ).to(self._device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, batch_logits = self._greedy_decode(
                    model=self.model,
                    input_ids=tokenized["input_ids"],
                    length=1,
                    attention_mask=tokenized["attention_mask"],
                    return_last_logits=True,
                )

            # Extract true/false logits and convert to log-probabilities
            pair_logits = batch_logits[:, [self.token_false_id, self.token_true_id]]
            log_probs = torch.log_softmax(pair_logits, dim=-1)
            probs = log_probs[:, 1].tolist()  # P(doc_i > doc_j)

            for (i, j), prob in zip(batch_pairs, probs):
                score_matrix[i][j] = prob

        return score_matrix

    @torch.inference_mode()
    def _greedy_decode(
        self,
        model,
        input_ids: torch.Tensor,
        length: int,
        attention_mask: torch.Tensor = None,
        return_last_logits: bool = True,
    ):
        """
        Single-step greedy decode — shared with MonoT5.

        Returns the decoder token sequence and, optionally, the last-step logits.
        """
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)

        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)
        next_token_logits = outputs.logits[:, -1, :]
        decode_ids = torch.cat(
            [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
        )
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_output_tokens(
        model_name: str,
        token_false: str = "auto",
        token_true: str = "auto",
    ):
        """
        Retrieve the true/false prediction tokens for the given DuoT5 model.

        Falls back to the default MonoT5 tokens when the model name is not
        explicitly listed in ``PREDICTION_TOKENS``.

        Args:
            model_name (str): The model name or HuggingFace identifier.
            token_false (str, optional): Override for the "false" token.
            token_true (str, optional): Override for the "true" token.

        Returns:
            Tuple[str, str]: ``(token_false, token_true)``.
        """
        if token_false == "auto" and model_name in PREDICTION_TOKENS:
            token_false = PREDICTION_TOKENS[model_name][0]
        if token_true == "auto" and model_name in PREDICTION_TOKENS:
            token_true = PREDICTION_TOKENS[model_name][1]
        if token_false == "auto" or token_true == "auto":
            token_false, token_true = PREDICTION_TOKENS["default"]
            logger.warning(
                f"Model {model_name} not found in PREDICTION_TOKENS. "
                "Using default tokens."
            )
        return token_false, token_true
