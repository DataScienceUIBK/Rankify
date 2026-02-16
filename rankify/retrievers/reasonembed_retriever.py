"""
ReasonEmbed Retriever.
- SentenceTransformer-based
- Supports embedding models:
    - qwen3-4B,
    - qwen3-8b,
    - llama3.1-8B,
with specialised reasoning-instruction prompting.
"""

import os
import json
import numpy as np
import torch
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context


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


# ReasonEmbedRetriever Class
class ReasonEmbedRetriever(BaseRetriever):
    """
    A dense retriever that leverages ReasonEmbed models for reasoning-heavy retrieval tasks.
    
    Attributes:
        model_id (str): Identifier for the embedding backbone (e.g. 'qwen3-8b', 'llama-8b').
        encode_batch_size (int): Batch size for embedding generation.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        corpus_path: str,
        model_id: str,      
        corpus_format: str = "jsonl",          
        text_field: str = "text",             
        title_field: str = "title",           
        id_field: str = "id",                
        cache_dir: str = "./cache",
        encode_batch_size: int = 8, 
        
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if corpus_path is None:
            raise ValueError("ReasonEmbedRetriever requires `corpus_path`")

        if model_id is None:
            raise ValueError("ReasonEmbedRetriever requires `model_id`")
        
        self.corpus_path = corpus_path
        self.corpus_format = corpus_format.lower()
        self.text_field = text_field
        self.title_field = title_field
        self.id_field = id_field
        self.tokenizer_simple = SimpleTokenizer()

        self.model_id = model_id
        self.checkpoint = checkpoint

        self.cache_dir = cache_dir
        self.encode_batch_size = encode_batch_size
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load corpus
        self.doc_ids, self.doc_texts, self.doc_titles = self._load_corpus()
        self._docid_to_idx = {str(did): i for i, did in enumerate(self.doc_ids)}

        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()

        self.doc_emb = self._load_or_build_doc_embeddings()

    def _load_model(self):
        # ReasonEmbed SentenceTransformer models
        if self.model_id == "qwen3-8b":
            self.model = SentenceTransformer(self.checkpoint or "hanhainebula/reason-embed-qwen3-8b-0928", trust_remote_code=True, device=self.device)
            self.model = self.model.to(self.device)
        elif self.model_id == "qwen3-4b":
            self.model = SentenceTransformer(self.checkpoint or "hanhainebula/reason-embed-qwen3-4b-0928", trust_remote_code=True, device=self.device)
            self.model = self.model.to(self.device)
        elif self.model_id == "llama-8b":
            self.model = SentenceTransformer(self.checkpoint or "hanhainebula/reason-embed-llama-3.1-8b-0928", trust_remote_code=True, device=self.device)
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"The model {self.model_id} is not supported")
        
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

    def _corpus_id(self):
        return os.path.basename(os.path.dirname(self.corpus_path))

    def _cache_file(self):
        corpus_id = self._corpus_id()
        folder = os.path.join(self.cache_dir, "doc_emb", corpus_id, self.model_id)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "0.npy")
    
    def _load_or_build_doc_embeddings(self):
        cache_file = self._cache_file()
        if os.path.isfile(cache_file):
            doc_emb = np.load(cache_file, allow_pickle=True)
            return doc_emb

        docs = self.doc_texts
        
        # Encoding logic based on model_id
        if self.model_id in ["qwen3-4b", "qwen3-8b", "llama-8b"]:
            doc_emb = self.model.encode(docs, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
        else:
            raise ValueError(f"Encoding not implemented for {self.model_id}")

        np.save(cache_file, doc_emb)

        assert len(self.doc_ids) == doc_emb.shape[0], (
            f"Embedding mismatch: {len(self.doc_ids)} docs vs {doc_emb.shape[0]} embeddings"
        )

        return doc_emb
    
    def _initialize_searcher(self):
        return None

    def _encode_queries(self, queries: List[str]):
        # ReasonEmbed models require a specific 'Instruct' prefix to trigger
        # reasoning capabilities during the embedding process.
        if self.model_id in ["qwen3-4b", "qwen3-8b", "llama-8b"]:
            task = "Given a search query, retrieve relevant passages that answer the query."
            instruct_queries = [f"Instruct: {task}\nQuery: {q}" for q in queries]
            return self.model.encode(instruct_queries, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
        raise ValueError(f"Encoding not implemented for {self.model_id}")

    def retrieve(self, documents: List[Document]):
        queries = [d.question.question for d in documents]
        query_ids = [str(d.id) if d.id is not None else str(i) for i, d in enumerate(documents)]

        query_emb = self._encode_queries(queries)
        
        # Ensure doc_emb is loaded
        if self.doc_emb is None:
            self.doc_emb = self._load_or_build_doc_embeddings()
                
        # Use torchmetrics for cosine similarity
        scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(self.doc_emb)).tolist()

        results = get_scores(
            query_ids=query_ids,
            doc_ids=self.doc_ids,
            scores=scores,
            num_hits=self.n_docs,
        )

        # Fill contexts
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
