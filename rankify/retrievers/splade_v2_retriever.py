"""
SPLADE-v2 learned sparse retriever using Pyserini's LuceneImpactSearcher.

References:
    - Formal et al. (2022): "From Distillation to Hard Negative Sampling:
      Making Sparse Neural IR Models More Effective."
      https://arxiv.org/abs/2205.04733
    - Formal et al. (2021): "SPLADE: Sparse Lexical and Expansion Model for
      First Stage Ranking."  https://arxiv.org/abs/2107.05720
"""

import json
from typing import List, Optional
from tqdm import tqdm

try:
    from pyserini.search.lucene import LuceneImpactSearcher
    from pyserini.encode import SpladeQueryEncoder
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context


class SpladeV2Retriever(BaseRetriever):
    """
    SPLADE-v2 learned sparse retriever using Pyserini's LuceneImpactSearcher.

    SPLADE learns sparse, high-dimensional representations by predicting
    importance weights over the vocabulary for each token position, enabling
    efficient inverted-index retrieval with neural relevance signals.

    References:
        - Formal et al. (2022): "From Distillation to Hard Negative Sampling"
          https://arxiv.org/abs/2205.04733
        - Prebuilt index: ``msmarco-v1-passage.splade-pp-ed``

    Args:
        index_type (str): Prebuilt index alias.  One of:

            * ``'splade-pp-ed'`` – SPLADE++ EnsembleDistil (default)
            * ``'splade-pp-sd'`` – SPLADE++ SelfDistil

            Pass ``index_folder`` to use a custom Lucene impact index.
        index_folder (str, optional): Path to a custom Lucene impact-index dir.
        corpus_path (str, optional): JSONL file for passage text lookup
            (fields: ``id``, ``text``, ``title``).
        model_name (str): SPLADE query encoder checkpoint.
        device (str): ``'cpu'`` or ``'cuda'``.

    Example:
        ```python
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(
            method="splade-v2",
            n_docs=10,
            index_type="splade-pp-ed",
            corpus_path="msmarco_passages.jsonl",
        )
        ```
    """

    PREBUILT_INDEX_MAP = {
        "splade-pp-ed": "msmarco-v1-passage.splade-pp-ed",
        "splade-pp-sd": "msmarco-v1-passage.splade-pp-sd",
        # Convenience aliases
        "msmarco": "msmarco-v1-passage.splade-pp-ed",
        "msmarco-splade-pp-ed": "msmarco-v1-passage.splade-pp-ed",
        "msmarco-splade-pp-sd": "msmarco-v1-passage.splade-pp-sd",
    }

    DEFAULT_MODEL_MAP = {
        "splade-pp-ed": "naver/splade-cocondenser-ensembledistil",
        "splade-pp-sd": "naver/splade-cocondenser-selfdistil",
        "msmarco": "naver/splade-cocondenser-ensembledistil",
        "msmarco-splade-pp-ed": "naver/splade-cocondenser-ensembledistil",
        "msmarco-splade-pp-sd": "naver/splade-cocondenser-selfdistil",
    }

    def __init__(
        self,
        index_type: str = "splade-pp-ed",
        index_folder: Optional[str] = None,
        corpus_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not PYSERINI_AVAILABLE:
            raise ImportError(
                "Pyserini is required for SpladeV2Retriever. "
                "Install with: pip install pyserini"
            )
        self.index_type = index_type
        self.index_folder = index_folder
        self.corpus_path = corpus_path
        self.device = device
        # Choose a sensible default model for the requested index variant
        self.model_name = model_name or self.DEFAULT_MODEL_MAP.get(
            index_type, "naver/splade-cocondenser-ensembledistil"
        )

        self.passage_store: dict = {}
        if corpus_path:
            self._load_corpus(corpus_path)

        self.searcher = self._initialize_searcher()

    def _load_corpus(self, corpus_path: str) -> None:
        """Load passage texts from a JSONL file into a dict keyed by id."""
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pid = str(obj.get("id", obj.get("docid", "")))
                self.passage_store[pid] = {
                    "text": obj.get("text", obj.get("contents", "")),
                    "title": obj.get("title", ""),
                }
        print(f"Loaded {len(self.passage_store)} passages.")

    def _initialize_searcher(self) -> "LuceneImpactSearcher":
        """Build and return a LuceneImpactSearcher with a SPLADE query encoder."""
        query_encoder = SpladeQueryEncoder(self.model_name, device=self.device)
        if self.index_folder:
            return LuceneImpactSearcher(self.index_folder, query_encoder)
        prebuilt = self.PREBUILT_INDEX_MAP.get(self.index_type)
        if prebuilt is None:
            raise ValueError(
                f"Unknown index_type '{self.index_type}'. "
                f"Supported values: {list(self.PREBUILT_INDEX_MAP.keys())}, "
                "or pass index_folder."
            )
        return LuceneImpactSearcher.from_prebuilt_index(prebuilt, query_encoder)

    def _get_passage(self, docid: str):
        """Return (text, title) for *docid*, falling back to Lucene stored field."""
        passage = self.passage_store.get(docid)
        if passage is not None:
            return passage["text"], passage["title"]
        try:
            raw = self.searcher.doc(docid)
            if raw:
                parsed = json.loads(raw.raw())
                text = parsed.get("contents", parsed.get("text", ""))
                title = parsed.get("title", "")
                return text, title
        except Exception:
            pass
        return "Text not available", ""

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve passages for each document using the SPLADE sparse index.

        Args:
            documents (List[Document]): Documents whose ``question`` field
                contains the query.

        Returns:
            List[Document]: Documents with populated ``contexts``.
        """
        for document in tqdm(documents, desc="SPLADE-v2 retrieving"):
            query = document.question.question
            hits = self.searcher.search(query, k=self.n_docs)
            contexts: List[Context] = []
            for hit in hits:
                docid = str(hit.docid)
                score = float(hit.score)
                text, title = self._get_passage(docid)

                has_ans = False
                if document.answers and document.answers.answers:
                    try:
                        from pyserini.eval.evaluate_dpr_retrieval import (
                            has_answers,
                            SimpleTokenizer,
                        )
                        has_ans = has_answers(
                            text, document.answers.answers, SimpleTokenizer()
                        )
                    except Exception:
                        pass

                contexts.append(
                    Context(
                        id=docid,
                        title=title,
                        text=text,
                        score=score,
                        has_answer=has_ans,
                    )
                )
            document.contexts = contexts
        return documents
