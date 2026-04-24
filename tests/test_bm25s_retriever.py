"""
Unit tests for BM25SRetriever – pure-Python BM25 retriever using ``bm25s``.

These tests run without any heavy dependencies (no Java, no Pyserini, no GPU).
They exercise:
  - Building an index from JSONL corpus
  - Building an index from TSV corpus
  - Loading a persisted index
  - Correctness of returned contexts (scores, has_answer flag, title/text)
  - Edge-cases (no results, empty answers list)
  - Unified ``Retriever(method='bm25s', ...)`` interface
"""

import json
import os
import tempfile
import unittest

from rankify.dataset.dataset import Answer, Context, Document, Question
from rankify.retrievers.bm25s_retriever import BM25SRetriever, _has_answers
from rankify.retrievers.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORPUS = [
    {"id": "1", "title": "Anarchism", "text": "Anarchism is a political philosophy that advocates self-governed societies."},
    {"id": "2", "title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"id": "3", "title": "Python", "text": "Python is a high-level, general-purpose programming language."},
    {"id": "4", "title": "BM25", "text": "BM25 is a bag-of-words retrieval function used to rank documents."},
    {"id": "5", "title": "Biology", "text": "Biology is the scientific study of life and living organisms."},
]


def _write_jsonl(path: str, docs=_CORPUS):
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc) + "\n")


def _write_tsv(path: str, docs=_CORPUS):
    """Write PSGs-style TSV: id TAB text TAB title."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id\ttext\ttitle\n")
        for doc in docs:
            fh.write(f"{doc['id']}\t{doc['text']}\t{doc['title']}\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHasAnswers(unittest.TestCase):
    def test_present(self):
        self.assertTrue(_has_answers("Paris is great", ["Paris"]))

    def test_case_insensitive(self):
        self.assertTrue(_has_answers("PARIS is great", ["paris"]))

    def test_absent(self):
        self.assertFalse(_has_answers("London is great", ["Paris"]))

    def test_empty_answers(self):
        self.assertFalse(_has_answers("Paris is great", []))


class TestBM25SRetrieverJSONL(unittest.TestCase):
    """Build index from JSONL corpus and run queries."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.corpus_path = os.path.join(self.tmpdir, "corpus.jsonl")
        _write_jsonl(self.corpus_path)
        self.index_folder = os.path.join(self.tmpdir, "index")
        self.retriever = BM25SRetriever(
            n_docs=3,
            corpus_path=self.corpus_path,
            index_folder=self.index_folder,
        )

    def _query(self, question: str, answers=None):
        doc = Document(
            question=Question(question=question),
            answers=Answer(answers=answers or []),
        )
        return self.retriever.retrieve([doc])[0]

    def test_returns_n_docs(self):
        result = self._query("French capital")
        self.assertEqual(len(result.contexts), 3)

    def test_contexts_are_context_objects(self):
        result = self._query("philosophy")
        for ctx in result.contexts:
            self.assertIsInstance(ctx, Context)

    def test_top_result_is_relevant(self):
        result = self._query("capital of France")
        self.assertEqual(result.contexts[0].title, "Paris")

    def test_has_answer_flag_true(self):
        result = self._query("capital city", answers=["Paris"])
        top = result.contexts[0]
        self.assertTrue(top.has_answer)

    def test_has_answer_flag_false(self):
        result = self._query("programming language", answers=["Java"])
        # Java does not appear in any doc text
        for ctx in result.contexts:
            self.assertFalse(ctx.has_answer)

    def test_context_fields_populated(self):
        result = self._query("political philosophy")
        ctx = result.contexts[0]
        self.assertIsInstance(ctx.id, str)
        self.assertIsInstance(ctx.title, str)
        self.assertIsInstance(ctx.text, str)
        self.assertIsInstance(ctx.score, float)

    def test_scores_descending(self):
        result = self._query("biology living organisms")
        scores = [ctx.score for ctx in result.contexts]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestBM25SRetrieverTSV(unittest.TestCase):
    """Build index from TSV corpus."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.corpus_path = os.path.join(self.tmpdir, "corpus.tsv")
        _write_tsv(self.corpus_path)
        self.index_folder = os.path.join(self.tmpdir, "index")
        self.retriever = BM25SRetriever(
            n_docs=2,
            corpus_path=self.corpus_path,
            index_folder=self.index_folder,
        )

    def test_top_result_tsv(self):
        doc = Document(
            question=Question(question="capital France"),
            answers=Answer(answers=["Paris"]),
        )
        result = self.retriever.retrieve([doc])[0]
        self.assertEqual(result.contexts[0].title, "Paris")
        self.assertTrue(result.contexts[0].has_answer)


class TestBM25SIndexPersistence(unittest.TestCase):
    """Index built on first run should be loadable without corpus_path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        corpus_path = os.path.join(self.tmpdir, "corpus.jsonl")
        _write_jsonl(corpus_path)
        self.index_folder = os.path.join(self.tmpdir, "index")
        # Build the index
        BM25SRetriever(n_docs=2, corpus_path=corpus_path, index_folder=self.index_folder)

    def test_load_without_corpus_path(self):
        """Loading without corpus_path should succeed because the index already exists."""
        loaded = BM25SRetriever(n_docs=2, index_folder=self.index_folder)
        doc = Document(
            question=Question(question="programming language"),
            answers=Answer(answers=[]),
        )
        result = loaded.retrieve([doc])[0]
        self.assertEqual(len(result.contexts), 2)
        self.assertEqual(result.contexts[0].title, "Python")

    def test_missing_index_no_corpus_raises(self):
        """Without a pre-built index and without corpus_path, expect FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            BM25SRetriever(n_docs=2, index_folder=os.path.join(self.tmpdir, "nonexistent"))


class TestBM25SRetrieverBatchQueries(unittest.TestCase):
    """Multiple queries in a single call."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        corpus_path = os.path.join(self.tmpdir, "corpus.jsonl")
        _write_jsonl(corpus_path)
        index_folder = os.path.join(self.tmpdir, "index")
        self.retriever = BM25SRetriever(n_docs=2, corpus_path=corpus_path, index_folder=index_folder)

    def test_batch_retrieval(self):
        docs = [
            Document(question=Question(question="anarchism politics"), answers=Answer(answers=[])),
            Document(question=Question(question="Paris France"),       answers=Answer(answers=["Paris"])),
            Document(question=Question(question="python programming"),  answers=Answer(answers=[])),
        ]
        results = self.retriever.retrieve(docs)
        self.assertEqual(len(results), 3)
        for doc in results:
            self.assertEqual(len(doc.contexts), 2)


class TestRetrieverUnifiedInterface(unittest.TestCase):
    """Ensure ``Retriever(method='bm25s', ...)`` works end-to-end."""

    def test_bm25s_in_supported_methods(self):
        self.assertIn("bm25s", Retriever.supported_methods())

    def test_unified_retriever(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, "corpus.jsonl")
            _write_jsonl(corpus_path)
            index_folder = os.path.join(tmpdir, "index")

            retriever = Retriever(
                method="bm25s",
                n_docs=2,
                index_folder=index_folder,
                corpus_path=corpus_path,
            )
            docs = [Document(
                question=Question(question="capital France"),
                answers=Answer(answers=["Paris"]),
            )]
            results = retriever.retrieve(docs)
            self.assertEqual(len(results[0].contexts), 2)
            self.assertEqual(results[0].contexts[0].title, "Paris")


if __name__ == "__main__":
    unittest.main()
