"""
ReasonIR Retriever.
- SentenceTransformer-based
Implements the reasonir/ReasonIR-8B model. 
This retriever uses instruction-tuned SentenceTransformers to perform reasoning-aware document retrieval.
"""

import os
import json
from typing import List, Optional

import numpy as np
import torch
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

# ReasonIRRetriever Class
class ReasonIRRetriever(BaseRetriever):
    """
    Rankify retriever wrapper for ReasonIR (uses model.encode with instructions)
    - caches doc embeddings in cache_dir/doc_emb/corpus_id/reasonir/0.npy
    - encodes queries
    - cosine similarity + top-k
    """

    def __init__(
        self,
        corpus_path: str,
        checkpoint: str = "reasonir/ReasonIR-8B",
        task: str = "retrieval",
        query_instruction: str = "Represent this question for retrieving relevant passages: ",
        doc_instruction: str = "Represent this passage for retrieval: ",
        cache_dir: str = "./cache",
        batch_size: int = 4,
        query_max_length: int = 32768,
        doc_max_length: int = 32768,
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        device: Optional[str] = None,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if corpus_path is None:
            raise ValueError("ReasonirRetriever requires `corpus_path`")
        
        self.corpus_path = corpus_path
        self.corpus_format = "jsonl"
        self.checkpoint = checkpoint
        self.task = task
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction

        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.normalize = normalize

        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.simple_tokenizer = SimpleTokenizer()

        # Load corpus
        self.doc_ids, self.doc_texts, self.doc_titles = self._load_corpus()

        self._docid_to_idx = {str(did): i for i, did in enumerate(self.doc_ids)}

        self.model = SentenceTransformer(self.checkpoint, trust_remote_code=True, device=self.device)
        self.tokenizer = None

        # Cache / load doc embeddings
        self.doc_emb = self._load_or_build_doc_embeddings()

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

        folder = os.path.join(
            self.cache_dir,
            "doc_emb",
            corpus_id,
            "reasonir",
        )
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "0.npy")

    @torch.no_grad()
    def _encode(self, texts: List[str], instruction: str, max_length: int):
        emb = self.model.encode(
            texts,
            instruction=instruction,
            batch_size=self.batch_size,
            max_length=max_length,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return emb

    def _load_or_build_doc_embeddings(self):
        cache_file = self._cache_file()
        if os.path.isfile(cache_file):
            return np.load(cache_file, allow_pickle=True)

        doc_instr = self.doc_instruction.format(task=self.task) if "{task}" in self.doc_instruction else self.doc_instruction
        doc_emb = self._encode(self.doc_texts, instruction=doc_instr, max_length=self.doc_max_length)
        np.save(cache_file, doc_emb)
        return doc_emb
    
    def _initialize_searcher(self):
    # Retrievers don't use an external searcher
        return None

    def retrieve(self, documents: List[Document]):
        # Queries + ids
        queries = [d.question.question for d in documents]
        query_ids = [str(d.id) if d.id is not None else str(i) for i, d in enumerate(documents)]

        q_instr = self.query_instruction.format(task=self.task) if "{task}" in self.query_instruction else self.query_instruction
        query_emb = self._encode(queries, instruction=q_instr, max_length=self.query_max_length)

        # Similarity
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
                        has_answer=has_answers(text, d.answers.answers, self.simple_tokenizer),
                    )
                )

            d.contexts = ctxs

        return documents
