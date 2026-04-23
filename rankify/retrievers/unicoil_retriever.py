"""
UniCOIL learned sparse retriever using Pyserini's LuceneImpactSearcher.

References:
    - Lin & Ma (2021): "A Few Brief Notes on DeepImpact, COIL, and a Conceptual
      Framework for Information Retrieval Techniques."
      https://arxiv.org/abs/2106.14807
"""

import json
from typing import List, Optional
from tqdm import tqdm

try:
    from pyserini.search.lucene import LuceneImpactSearcher
    from pyserini.encode import UniCoilQueryEncoder
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context


class UniCOILRetriever(BaseRetriever):
    """
    UniCOIL learned sparse retriever using Pyserini's LuceneImpactSearcher.

    UniCOIL (Unified COmpact Index with Learned sparse representation) learns
    term weights via a BERT model with a linear projection to a scalar weight
    per token position.

    References:
        - Lin & Ma (2021): "A Few Brief Notes on DeepImpact, COIL, and a Conceptual
          Framework for Information Retrieval Techniques."
          https://arxiv.org/abs/2106.14807
        - Model: castorini/unicoil-msmarco-passage

    Args:
        index_type (str): ``'msmarco'`` to use the Pyserini prebuilt MS MARCO unicoil
            index, or ``'msmarco-noexp'`` for the no-expansion variant.
            Supply ``index_folder`` for a custom Lucene impact index.
        index_folder (str, optional): Path to a custom Lucene impact-index directory.
        corpus_path (str, optional): JSONL file for passage text lookup
            (fields: ``id``, ``text``, ``title``).  Required when using a custom
            ``index_folder`` because the Lucene index may not store raw text.
        model_name (str): UniCOIL query encoder checkpoint.
        device (str): ``'cpu'`` or ``'cuda'``.

    Example:
        ```python
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(
            method="unicoil",
            n_docs=10,
            index_type="msmarco",
            corpus_path="msmarco_passages.jsonl",
        )
        ```
    """

    PREBUILT_INDEX_MAP = {
        "msmarco": "msmarco-v1-passage.unicoil",
        "msmarco-noexp": "msmarco-v1-passage.unicoil-noexp",
    }

    def __init__(
        self,
        index_type: str = "msmarco",
        index_folder: Optional[str] = None,
        corpus_path: Optional[str] = None,
        model_name: str = "castorini/unicoil-msmarco-passage",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not PYSERINI_AVAILABLE:
            raise ImportError(
                "Pyserini is required for UniCOILRetriever. "
                "Install with: pip install pyserini"
            )
        self.index_type = index_type
        self.index_folder = index_folder
        self.corpus_path = corpus_path
        self.model_name = model_name
        self.device = device

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
        """Build and return a LuceneImpactSearcher with a UniCOIL query encoder."""
        query_encoder = UniCoilQueryEncoder(self.model_name, device=self.device)
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
        Retrieve passages for each document using the UniCOIL sparse index.

        Args:
            documents (List[Document]): Documents whose ``question`` field
                contains the query.

        Returns:
            List[Document]: Documents with populated ``contexts``.
        """
        for document in tqdm(documents, desc="UniCOIL retrieving"):
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
