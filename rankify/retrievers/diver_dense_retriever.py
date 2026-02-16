"""
DiverDenseRetriever implementing a variety of dense retriever models from diver's code:
    - https://github.com/AQ-MedAI/Diver/blob/main/Retriever/retrievers.py

Supports:
    - SentenceTransformers (bge, sbert, nomic, instructor)
    - HF AutoModels (sf, qwen, e5, rader, contriever, m2)
    - GritLM (grit)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context

# Optional imports for specific models
try:
    from gritlm import GritLM
    GRITLM_AVAILABLE = True
except ImportError:
    GRITLM_AVAILABLE = False


# Helper Functions from the original diver code 

def add_instruct_concatenate(texts, task, instruction):
    return [instruction.format(task=task) + t for t in texts]

def add_instruct_list(texts, task, instruction):
    return [[instruction.format(task=task), t] for t in texts]

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

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


# Main DiverDenseRetriever Class
class DiverDenseRetriever(BaseRetriever):
    """
    Comprehensive Diver-style retriever supporting:
    - SentenceTransformers (bge, sbert, nomic, instructor)
    - HF AutoModels (sf, qwen, e5, rader, contriever, m2)
    - GritLM (grit)
    """

    def __init__(
        self,
        corpus_path: str,
        model_id: str,      
        corpus_format: str = "jsonl",          
        text_field: str = "text",             
        title_field: str = "title",           
        id_field: str = "id",                

        task: str = "retrieval",
        instruction_query: str = "Represent this question for retrieving relevant passages: ",
        instruction_document: str = "Represent this passage for retrieval: ",
        doc_max_length: int = 2048, 
        query_max_length: int = 2048,

        cache_dir: str = "./cache",
        long_context: bool = False,
        encode_batch_size: int = 8,
        
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if corpus_path is None:
            raise ValueError("DiverDenseRetriever requires `corpus_path`")

        if model_id is None:
            raise ValueError("DiverDenseRetriever requires `model_id`")
        
        self.corpus_path = corpus_path
        self.corpus_format = corpus_format.lower()
        self.text_field = text_field
        self.title_field = title_field
        self.id_field = id_field

        self.model_id = model_id
        self.checkpoint = checkpoint
        self.task = task
        self.instruction_query = instruction_query
        self.instruction_document = instruction_document

        self.cache_dir = cache_dir
        self.long_context = long_context
        self.doc_max_length = doc_max_length
        self.query_max_length = query_max_length
        self.encode_batch_size = encode_batch_size
        if self.model_id == "grit":
            self.encode_batch_size = 1  # GritLM is sensitive to batching, force to 1
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer_simple = SimpleTokenizer()

        # Load corpus
        self.doc_ids, self.doc_texts, self.doc_titles = self._load_corpus()
        self._docid_to_idx = {str(did): i for i, did in enumerate(self.doc_ids)}

        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()

        self.doc_emb = self._load_or_build_doc_embeddings()

    def _load_model(self):
        # SentenceTransformer models
        if self.model_id == "bge":
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        elif self.model_id == "sbert":
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        elif self.model_id == "contriever_st":
            self.model = SentenceTransformer('nishimoto/contriever-sentencetransformer')
        elif self.model_id == "nomic":
            self.model = SentenceTransformer(self.checkpoint or "nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        elif self.model_id == "inst-l":
            self.model = SentenceTransformer("hkunlp/instructor-large")
            self.model.max_seq_length = self.doc_max_length
        elif self.model_id == "inst-xl":
            self.model = SentenceTransformer("hkunlp/instructor-xl")
            self.model.max_seq_length = self.doc_max_length
        
        # HF AutoModel models (sf, e5, rader)
        elif self.model_id in ["sf", "e5", "rader"]:
            checkpoint = self.checkpoint
            if not checkpoint:
                if self.model_id == 'sf': checkpoint = 'Salesforce/SFR-Embedding-Mistral'
                elif self.model_id == 'e5': checkpoint = 'intfloat/e5-mistral-7b-instruct'
                elif self.model_id == 'rader': checkpoint = 'Raderspace/RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical'
            
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True).eval()
            
        # M2
        elif self.model_id == "m2":
            checkpoint = self.checkpoint or "togethercomputer/m2-bert-80M-32k-retrieval"
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=32768)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, trust_remote_code=True).eval()
            # M2 usually runs on CPU/GPU via HF, ensure device
            self.model.to(self.device)

        # Contriever (Original HF)
        elif self.model_id == "contriever":
            checkpoint = self.checkpoint or 'facebook/contriever-msmarco'
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModel.from_pretrained(checkpoint).to(self.device).eval()

        # GritLM
        elif self.model_id == "grit":
            if not GRITLM_AVAILABLE:
                raise ImportError("gritlm is required for 'grit' model. Please install it.")
            checkpoint = self.checkpoint or 'GritLM/GritLM-7B'
            self.model = GritLM(checkpoint, torch_dtype="auto", mode="embedding")

        else:
            raise ValueError(f"The model {self.model_id} is not supported")
        
        # Handle device placement for models that don't auto-map
        if hasattr(self.model, "to") and self.model_id not in ["grit", "sf", "e5", "rader"]: 
            self.model = self.model.to(self.device)

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
        # match Diver folder naming so you can reuse caches
        batch_size = self.encode_batch_size
        corpus_id = self._corpus_id()
        folder = os.path.join(self.cache_dir, "doc_emb", corpus_id, self.model_id, self.task, f"long_{self.long_context}_{batch_size}")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "0.npy")

    def _encode_hf_auto_model(self, texts, max_length, instruction=None):
        # For sf, qwen, e5, rader
        if instruction:
            texts = add_instruct_concatenate(texts, self.task, instruction)
            
        all_embeddings = []
        batch_size = self.encode_batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                if self.model_id == 'rader':
                     embeddings = outputs.last_hidden_state[:, -1, :] # Last token
                else:
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def _encode_m2(self, texts, max_length, instruction=None):
        if instruction:
            texts = add_instruct_concatenate(texts, self.task, instruction)
        
        all_embeddings = []
        batch_size = self.encode_batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding M2"):
            batch_texts = texts[i:i+batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            # Move to device if model is on device
            if next(self.model.parameters()).is_cuda:
                 batch_dict = {k: v.to(next(self.model.parameters()).device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = outputs['sentence_embedding']
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def _encode_contriever(self, texts, max_length):
        # Original HF Contriever
        all_embeddings = []
        batch_size = self.encode_batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Contriever"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def _load_or_build_doc_embeddings(self):
        cache_file = self._cache_file()
        if os.path.isfile(cache_file):
            doc_emb = np.load(cache_file, allow_pickle=True)
            return doc_emb

        docs = self.doc_texts
        
        # Encoding logic based on model_id
        if self.model_id in ["bge", "sbert", "contriever_st", "nomic"]:
            doc_emb = self.model.encode(docs, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
            
        elif self.model_id in ["inst-l", "inst-xl"]:
            docs = add_instruct_list(docs, task=self.task, instruction=self.instruction_document)
            doc_emb = self.model.encode(docs, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
            
        elif self.model_id == "e5":  # e5 paper uses strict "passage: {document}" for encoding
            docs = [f"passage: {d}" for d in docs]
            doc_emb = self._encode_hf_auto_model(docs, self.doc_max_length, instruction=None)

        elif self.model_id in ["sf", "rader"]:
            if self.model_id == 'rader':
                 docs = [f"document: {t[:8192]}" for t in docs]
            doc_emb = self._encode_hf_auto_model(docs, self.doc_max_length, instruction=None)
            
        elif self.model_id == "m2":
            doc_emb = self._encode_m2(docs, self.doc_max_length, instruction=None)
            
        elif self.model_id == "contriever":
            doc_emb = self._encode_contriever(docs, self.doc_max_length)
            
        elif self.model_id == "grit":
            doc_emb = self.model.encode(docs, instruction=self.instruction_document, batch_size=self.encode_batch_size, max_length=self.doc_max_length)
            
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
        if self.model_id in ["bge", "sbert", "contriever_st", "nomic"]:
            if self.model_id == "bge":
                queries = add_instruct_concatenate(texts=queries, task=self.task, instruction=self.instruction_query)
            return self.model.encode(queries, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
            
        elif self.model_id == "e5":
            queries = [f"query: {q}" for q in queries]    # e5 paper requires strict "query: {query}" when encoding
            return self._encode_hf_auto_model(queries, self.query_max_length, instruction=None)
        
        elif self.model_id in ["inst-l", "inst-xl"]:
            queries = add_instruct_list(texts=queries, task=self.task, instruction=self.instruction_query)
            return self.model.encode(queries, show_progress_bar=True, batch_size=self.encode_batch_size, normalize_embeddings=True)
            
        elif self.model_id in ["sf", "rader"]:
            if self.model_id == 'rader':
                queries = [f"query: {q}" for q in queries]
            return self._encode_hf_auto_model(queries, self.query_max_length, instruction=self.instruction_query if self.model_id != 'rader' else None)
            
        elif self.model_id == "m2":
            return self._encode_m2(queries, self.query_max_length, instruction=self.instruction_query)
            
        elif self.model_id == "contriever":
            return self._encode_contriever(queries, self.query_max_length)
            
        elif self.model_id == "grit":
            q_instr = self.instruction_query.format(task=self.task) if "{task}" in self.instruction_query else self.instruction_query
            return self.model.encode(queries, instruction=q_instr, batch_size=self.encode_batch_size, max_length=self.query_max_length)
        
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