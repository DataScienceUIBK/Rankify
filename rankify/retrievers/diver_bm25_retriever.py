"""
DiverBM25Retriever.
Uses Gensim's LuceneBM25Model to provide a keyword-based retrieval baseline.
"""

import os
import numpy as np
import json
from typing import List
from tqdm import tqdm

from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
from pyserini import analysis
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context

# Helper function from diver's original code
def get_scores(
        query_ids: List[str], 
        doc_ids: List[str], 
        scores: List[List[float]],
        num_hits: int = 1000):

    assert len(scores) == len(query_ids)
    
    emb_scores = {}
    for query_id, row in zip(query_ids, scores):
        cur = {str(doc_id): float(s) for doc_id, s in zip(doc_ids, row)}
        ranked = sorted(cur.items(), key=lambda x: x[1], reverse=True)[:num_hits]
        emb_scores[str(query_id)] = {doc_id: sc for doc_id, sc in ranked}

    return emb_scores

# To normalize BM25's similarity scoring

def minmax_normalize(scores, eps=1e-8):
    scores = np.asarray(scores, dtype=np.float32)
    min_s = float(scores.min())
    max_s = float(scores.max())
    if max_s - min_s < eps:
        return [0.0] * len(scores)
    return ((scores - min_s) / (max_s - min_s + eps)).tolist()



class DiverBM25Retriever(BaseRetriever):
    """
    Diver-style BM25 retriever using Gensim's LuceneBM25Model.
    """

    def __init__(
        self,
        corpus_path: str,
        corpus_format: str = "jsonl",
        text_field: str = "text",
        title_field: str = "title",
        id_field: str = "id",
        normalize_scores: bool = False,
        k1: float = 0.9,
        b: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if corpus_path is None:
            raise ValueError("DiverBM25Retriever requires `corpus_path`")
        
        self.corpus_path = corpus_path
        self.corpus_format = corpus_format.lower()
        self.text_field = text_field
        self.title_field = title_field
        self.id_field = id_field
        self.normalize_scores = normalize_scores
        self.k1 = k1
        self.b = b

        self.tokenizer_simple = SimpleTokenizer()
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

        self.dictionary = None
        self.doc_ids, self.doc_texts, self.doc_titles = self._load_corpus()
        self._docid_to_idx = {str(did): i for i, did in enumerate(self.doc_ids)}

        self.model, self.bm25_index = self._initialize_searcher()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")

        doc_ids: List[str] = []
        doc_texts: List[str] = []
        doc_titles: List[str] = []

        if self.corpus_format == "jsonl":
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    doc_id = str(obj.get(self.id_field, len(doc_ids)))
                    text = obj.get(self.text_field, "") or ""
                    title = obj.get(self.title_field, "") or ""
                    doc_ids.append(doc_id)
                    doc_texts.append(text)
                    doc_titles.append(title)
        else:
            raise ValueError("Corpus format must be 'jsonl'")

        return doc_ids, doc_texts, doc_titles

    def _initialize_searcher(self):
        corpus_analyzed = [self.analyzer.analyze(x) for x in self.doc_texts]
        self.dictionary = Dictionary(corpus_analyzed)
        model = LuceneBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
        bm25_corpus = model[list(map(self.dictionary.doc2bow, corpus_analyzed))]
        bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus_analyzed), num_terms=len(self.dictionary),
                                            normalize_queries=False, normalize_documents=False)
        return model, bm25_index

    def retrieve(self, documents: List[Document]) -> List[Document]:
        queries = [d.question.question for d in documents]
        query_ids = [str(d.id) if d.id is not None else str(i) for i, d in enumerate(documents)]

        all_scores_list = []
        for query in tqdm(queries, desc="BM25 retrieval"):
            query_analyzed = self.analyzer.analyze(query)
            bm25_query = self.model[self.dictionary.doc2bow(query_analyzed)]
            scores = self.bm25_index[bm25_query].tolist()
            if self.normalize_scores:
                scores = minmax_normalize(scores)
            all_scores_list.append(scores)

        results = get_scores(
            query_ids=query_ids,
            doc_ids=self.doc_ids,
            scores=all_scores_list,
            num_hits=self.n_docs,
        )

        for d, query_id in zip(documents, query_ids):
            ctxs: List[Context] = []
            for doc_id, score in results[query_id].items():
                idx = self._docid_to_idx.get(str(doc_id))
                if idx is None:
                    continue

                text = self.doc_texts[idx]
                title = self.doc_titles[idx] if idx < len(self.doc_titles) else ""

                ctxs.append(
                    Context(
                        id=str(doc_id),
                        title=title,
                        text=text,
                        score=float(score),
                        has_answer=has_answers(text, d.answers.answers, self.tokenizer_simple),
                    )
                )
            d.contexts = ctxs

        return documents

