"""
API-based embedding retrievers (OpenAI, Cohere, Voyage AI).

These retrievers embed a JSONL corpus via an external embedding API, build a
FAISS nearest-neighbour index, then answer queries by embedding the query text
and returning the top-k most similar passages.

Document embeddings are cached as ``.npy`` files so the corpus is only
embedded once per (corpus, model) combination.
"""

import json
import os
from typing import List, Optional
import numpy as np
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import openai as _openai_module
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cohere as _cohere_module
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import voyageai as _voyage_module
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context

# Default model names per provider
_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "cohere": "embed-english-v3.0",
    "voyage": "voyage-3",
}

# Embedding batch sizes per provider (API rate-limit friendly)
_BATCH_SIZES = {
    "openai": 128,
    "cohere": 96,
    "voyage": 128,
}


class APIEmbeddingRetriever(BaseRetriever):
    """
    Dense retriever backed by an external embedding API and a FAISS index.

    Supported providers:

    * ``'openai'``  – uses ``openai.OpenAI`` client
    * ``'cohere'``  – uses ``cohere.Client``
    * ``'voyage'``  – uses ``voyageai.Client``

    The corpus is embedded once and cached as a ``.npy`` file under
    ``cache_dir``.  Subsequent runs load the cache automatically.

    Args:
        provider (str): One of ``'openai'``, ``'cohere'``, or ``'voyage'``.
        api_key (str): API key for the chosen provider.
        corpus_path (str): Path to a JSONL corpus file.  Each line must
            contain ``id``, ``text``, and optionally ``title``.
        model_name (str, optional): Embedding model name.  Defaults are
            ``text-embedding-3-small`` (OpenAI), ``embed-english-v3.0``
            (Cohere), and ``voyage-3`` (Voyage).
        cache_dir (str): Directory for cached embeddings.
        embed_batch_size (int, optional): Override the default API batch size.

    Example:
        ```python
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(
            method="openai-embedding",
            n_docs=10,
            provider="openai",
            api_key="sk-...",
            corpus_path="corpus.jsonl",
        )
        ```
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        corpus_path: str,
        model_name: Optional[str] = None,
        cache_dir: str = "./cache",
        embed_batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for APIEmbeddingRetriever. "
                "Install with: pip install faiss-cpu  (or faiss-gpu)"
            )

        self.provider = provider.lower()
        self.api_key = api_key
        self.corpus_path = corpus_path
        self.model_name = model_name or _DEFAULT_MODELS.get(self.provider, "")
        self.cache_dir = cache_dir
        self.embed_batch_size = embed_batch_size or _BATCH_SIZES.get(self.provider, 128)

        self._validate_provider()

        # Load corpus texts
        self.doc_ids, self.doc_texts, self.doc_titles = self._load_corpus()
        self._docid_to_idx = {did: i for i, did in enumerate(self.doc_ids)}

        # Build or load FAISS index
        self.doc_emb = self._load_or_build_embeddings()
        self.index = self._build_faiss_index(self.doc_emb)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_provider(self) -> None:
        if self.provider == "openai" and not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        if self.provider == "cohere" and not COHERE_AVAILABLE:
            raise ImportError(
                "cohere package is required. Install with: pip install cohere"
            )
        if self.provider == "voyage" and not VOYAGE_AVAILABLE:
            raise ImportError(
                "voyageai package is required. Install with: pip install voyageai"
            )
        if self.provider not in _DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported: {list(_DEFAULT_MODELS.keys())}"
            )

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        doc_ids, doc_texts, doc_titles = [], [], []
        with open(self.corpus_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_ids.append(str(obj.get("id", obj.get("docid", len(doc_ids)))))
                doc_texts.append(obj.get("text", obj.get("contents", "")))
                doc_titles.append(obj.get("title", ""))
        return doc_ids, doc_texts, doc_titles

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_openai(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        client = _openai_module.OpenAI(api_key=self.api_key)
        all_emb = []
        for i in tqdm(range(0, len(texts), self.embed_batch_size), desc="OpenAI embed"):
            batch = texts[i : i + self.embed_batch_size]
            response = client.embeddings.create(input=batch, model=self.model_name)
            batch_emb = [item.embedding for item in response.data]
            all_emb.extend(batch_emb)
        return np.array(all_emb, dtype=np.float32)

    def _embed_cohere(self, texts: List[str], input_type: str = "search_document") -> np.ndarray:
        client = _cohere_module.Client(self.api_key)
        all_emb = []
        for i in tqdm(range(0, len(texts), self.embed_batch_size), desc="Cohere embed"):
            batch = texts[i : i + self.embed_batch_size]
            response = client.embed(
                texts=batch, model=self.model_name, input_type=input_type
            )
            all_emb.extend(response.embeddings)
        return np.array(all_emb, dtype=np.float32)

    def _embed_voyage(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        client = _voyage_module.Client(api_key=self.api_key)
        all_emb = []
        for i in tqdm(range(0, len(texts), self.embed_batch_size), desc="Voyage embed"):
            batch = texts[i : i + self.embed_batch_size]
            response = client.embed(batch, model=self.model_name, input_type=input_type)
            all_emb.extend(response.embeddings)
        return np.array(all_emb, dtype=np.float32)

    def _embed(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        if self.provider == "openai":
            return self._embed_openai(texts, input_type)
        if self.provider == "cohere":
            cohere_type = "search_document" if input_type == "document" else "search_query"
            return self._embed_cohere(texts, cohere_type)
        if self.provider == "voyage":
            return self._embed_voyage(texts, input_type)
        raise ValueError(f"Unknown provider '{self.provider}'")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self) -> str:
        corpus_id = os.path.splitext(os.path.basename(self.corpus_path))[0]
        folder = os.path.join(
            self.cache_dir, "api_emb", self.provider, self.model_name.replace("/", "_"), corpus_id
        )
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "doc_emb.npy")

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache = self._cache_path()
        if os.path.isfile(cache):
            print(f"Loading cached embeddings from {cache}")
            return np.load(cache)
        print(f"Embedding {len(self.doc_texts)} documents via {self.provider} ({self.model_name})…")
        emb = self._embed(self.doc_texts, input_type="document")
        np.save(cache, emb)
        print(f"Saved embeddings to {cache}")
        return emb

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _build_faiss_index(self, emb: np.ndarray) -> "faiss.Index":
        dim = emb.shape[1]
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb_norm = (emb / norms).astype(np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(emb_norm)
        return index

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _initialize_searcher(self):
        return None  # FAISS index is built in __init__

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve passages for each document by embedding the query and
        searching the FAISS index.

        Args:
            documents (List[Document]): Documents with query in ``question``.

        Returns:
            List[Document]: Documents with populated ``contexts``.
        """
        queries = [d.question.question for d in documents]
        query_emb = self._embed(queries, input_type="query")

        # L2-normalize for inner-product cosine search
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        query_emb_norm = (query_emb / norms).astype(np.float32)

        scores_matrix, indices_matrix = self.index.search(query_emb_norm, self.n_docs)

        for document, scores_row, idx_row in zip(documents, scores_matrix, indices_matrix):
            contexts: List[Context] = []
            for score, idx in zip(scores_row, idx_row):
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                doc_id = self.doc_ids[idx]
                text = self.doc_texts[idx]
                title = self.doc_titles[idx]

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
                        id=doc_id,
                        title=title,
                        text=text,
                        score=float(score),
                        has_answer=has_ans,
                    )
                )
            document.contexts = contexts
        return documents
