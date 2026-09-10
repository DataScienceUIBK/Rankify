# bm25s_retriever.py
"""
BM25S Retriever - a pure-Python BM25 retriever backed by the ``bm25s`` library.

Unlike the Pyserini-based :class:`BM25Retriever`, this retriever has no Java
dependency and is considerably lighter (no JVM, no Lucene).

Usage::

    from rankify.retrievers import BM25SRetriever
    from rankify.dataset.dataset import Document, Question, Answer

    retriever = BM25SRetriever(
        n_docs=10,
        corpus_path="/path/to/corpus.jsonl",   # build index on first run
        index_folder="/path/to/save/index",    # persist index here
    )
    docs = retriever.retrieve([Document(question=Question("What is BM25?"),
                                        answers=Answer([]))])

The corpus file can be:

* **JSONL** (one JSON object per line): each line must contain ``id``,
  ``title``, and ``text`` (or ``contents``) fields.
* **TSV** (tab-separated): columns are ``id\\ttext\\ttitle`` (the same layout
  used by the ``psgs_w100.tsv`` Wikipedia dump).

Once built, the index is saved to ``<index_folder>/bm25s_index/`` so subsequent
calls skip the expensive indexing step.
"""

import os
import json
from typing import List, Optional, Tuple

from tqdm import tqdm

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context


# ---------------------------------------------------------------------------
# Small helper – avoids importing pyserini
# ---------------------------------------------------------------------------

def _has_answers(text: str, answers: List[str]) -> bool:
    """Return *True* if any answer string appears (case-insensitively) in *text*."""
    text_lower = text.lower()
    return any(ans.lower() in text_lower for ans in answers)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class BM25SRetriever(BaseRetriever):
    """
    BM25 retriever using `bm25s <https://github.com/xhluca/bm25s>`_ – a pure
    Python implementation with **no Java / JVM dependency**.

    Parameters
    ----------
    index_type:
        Logical corpus name used to locate a cached index (``"wiki"`` or
        ``"msmarco"``).  Ignored when *index_folder* is given explicitly.
    index_folder:
        Directory used to persist (or load) the bm25s index.  A sub-directory
        ``bm25s_index`` is created inside this path.  If *None* the location is
        derived from *index_type* via the :class:`~rankify.retrievers.index_manager.IndexManager`.
    corpus_path:
        Path to the raw corpus file (JSONL or TSV).  Required when no
        pre-built index exists yet.
    stopwords:
        Stopword list passed to ``bm25s.tokenize``.  Use ``"en"`` for English
        (default) or ``None`` / ``""`` to disable.
    stemmer_lang:
        ISO language code for the PyStemmer ``Stemmer.Stemmer`` (e.g.
        ``"english"``).  Requires the optional ``PyStemmer`` package.  Set to
        ``None`` (default) to skip stemming.
    n_docs:
        Number of documents to return per query.
    batch_size / threads:
        Inherited from :class:`BaseRetriever`; unused by this implementation
        but kept for API compatibility.
    """

    def __init__(
        self,
        index_type: str = "wiki",
        index_folder: str = None,
        corpus_path: str = None,
        stopwords: str = "en",
        stemmer_lang: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_type = index_type
        self.corpus_path = corpus_path
        self.stopwords = stopwords or None   # bm25s treats "" same as None
        self.stemmer_lang = stemmer_lang

        # Resolve the directory where the index will live
        if index_folder:
            self.index_folder = index_folder
        else:
            from .index_manager import IndexManager
            im = IndexManager()
            try:
                self.index_folder = im.get_index_path("bm25s", index_type)
            except (ValueError, KeyError):
                # bm25s not registered in IndexManager yet – fall back to cache dir
                cache_dir = os.environ.get("RERANKING_CACHE_DIR", "./cache")
                self.index_folder = os.path.join(cache_dir, "index", f"bm25s_{index_type}")

        self.stemmer = self._init_stemmer()
        # _initialize_searcher is called here; it either loads or builds the index
        self.searcher = self._initialize_searcher()

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _initialize_searcher(self):
        """Load an existing bm25s index or build one from *corpus_path*."""
        import bm25s  # lazy import keeps package optional at import time

        index_path = os.path.join(self.index_folder, "bm25s_index")

        if os.path.isdir(index_path) and os.listdir(index_path):
            print(f"Loading BM25S index from {index_path} …")
            retriever = bm25s.BM25.load(index_path, load_corpus=True)
            return retriever

        # Index does not exist yet – build it
        if not self.corpus_path:
            raise FileNotFoundError(
                f"No pre-built BM25S index found at '{index_path}'. "
                "Please provide 'corpus_path' to build the index."
            )

        return self._build_index(index_path)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve the top-*n_docs* contexts for every document in *documents*."""
        import bm25s  # lazy import

        queries = [doc.question.question for doc in documents]
        print(f"Retrieving {len(documents)} document(s) with BM25S …")

        query_tokens = bm25s.tokenize(
            queries,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            show_progress=False,
        )

        results, scores = self.searcher.retrieve(query_tokens, k=self.n_docs)

        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            contexts: List[Context] = []
            num_hits = results.shape[1]
            for j in range(num_hits):
                doc_data = results[i, j]
                score = float(scores[i, j])

                doc_id = str(doc_data.get("id", ""))
                title = doc_data.get("title", "")
                text = doc_data.get("text", "")

                answers = document.answers.answers if document.answers else []
                context = Context(
                    id=doc_id,
                    title=title,
                    text=text,
                    score=score,
                    has_answer=_has_answers(text, answers),
                )
                contexts.append(context)

            document.contexts = contexts

        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_stemmer(self):
        """Return a PyStemmer stemmer, or *None* if unavailable / not requested."""
        if not self.stemmer_lang:
            return None
        try:
            import Stemmer  # PyStemmer (optional)
            return Stemmer.Stemmer(self.stemmer_lang)
        except ImportError:
            print(
                "Warning: PyStemmer is not installed; stemming is disabled. "
                "Install it with: pip install PyStemmer"
            )
            return None

    def _build_index(self, index_path: str):
        """Build a bm25s index from *self.corpus_path* and save it."""
        import bm25s  # lazy import

        corpus, corpus_texts = self._load_corpus(self.corpus_path)

        print(f"Tokenizing {len(corpus)} documents …")
        corpus_tokens = bm25s.tokenize(
            corpus_texts,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            show_progress=True,
        )

        print("Indexing corpus …")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        os.makedirs(index_path, exist_ok=True)
        print(f"Saving BM25S index to {index_path} …")
        retriever.save(index_path)

        return retriever

    def _load_corpus(self, corpus_path: str) -> Tuple[List[dict], List[str]]:
        """
        Load the corpus from a JSONL or TSV file.

        Returns
        -------
        corpus:
            List of ``{"id": …, "title": …, "text": …}`` dicts.
        corpus_texts:
            List of strings (``"<title>\\n<text>"``) used for tokenisation.
        """
        corpus: List[dict] = []

        lower = corpus_path.lower()
        if lower.endswith(".jsonl") or lower.endswith(".json"):
            corpus = self._load_jsonl(corpus_path)
        elif lower.endswith(".tsv"):
            corpus = self._load_tsv(corpus_path)
        else:
            # Try JSONL first, fall back to TSV heuristic
            try:
                corpus = self._load_jsonl(corpus_path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                corpus = self._load_tsv(corpus_path)

        if not corpus:
            raise ValueError(f"Corpus loaded from '{corpus_path}' is empty.")

        corpus_texts = [
            f"{doc['title']}\n{doc['text']}" if doc.get("title") else doc["text"]
            for doc in corpus
        ]
        return corpus, corpus_texts

    @staticmethod
    def _load_jsonl(corpus_path: str) -> List[dict]:
        corpus = []
        with open(corpus_path, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, desc="Loading corpus (JSONL)"):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = str(doc.get("id") or doc.get("docid") or "")
                title = doc.get("title", "")
                text = doc.get("text") or doc.get("contents", "")
                corpus.append({"id": doc_id, "title": title, "text": text})
        return corpus

    @staticmethod
    def _load_tsv(corpus_path: str) -> List[dict]:
        """
        Load TSV corpus.  Supports both ``id\\ttext\\ttitle`` (psgs_w100 layout)
        and ``id\\ttitle\\ttext`` layouts; the header row is used to detect the
        column order when present.
        """
        corpus = []
        title_col, text_col = 2, 1  # psgs_w100 default: id | text | title

        with open(corpus_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(tqdm(fh, desc="Loading corpus (TSV)")):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if i == 0 and parts[0].lower() in ("id", "docid"):
                    # Detect column order from header
                    lower_parts = [p.lower() for p in parts]
                    if "title" in lower_parts and "text" in lower_parts:
                        title_col = lower_parts.index("title")
                        text_col = lower_parts.index("text")
                    continue  # skip header row

                if len(parts) < 2:
                    continue

                doc_id = parts[0]
                text = parts[text_col] if len(parts) > text_col else ""
                title = parts[title_col] if len(parts) > title_col else ""
                corpus.append({"id": doc_id, "title": title, "text": text})
        return corpus
