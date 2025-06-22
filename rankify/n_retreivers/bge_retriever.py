import os
import requests
import h5py
import pickle
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from typing import List
import zipfile
from urllib.parse import urlparse
import shutil
import gzip

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context


class BGERetriever(BaseRetriever):
    """
    BGE retriever implementation using precomputed embeddings and FAISS indexing.
    
    Implements BGE (Beijing Academy of Artificial Intelligence General Embedding) model
    for dense passage retrieval with efficient FAISS-based search.
    """
    
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", index_type: str = "wiki", 
                 index_folder: str = None, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self.index_type = index_type
        self.index_folder = index_folder
        self.device = device
        self.tokenizer_simple = SimpleTokenizer()
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Setup paths and download if needed
        if index_folder:
            self.index_path = index_folder
            self.passage_path = os.path.join(self.index_folder, 'passages.tsv')
        else:
            self.index_path = os.path.join(self.index_manager.cache_dir, "index", f"bge_index_{index_type}")
            self._ensure_index_and_passages_downloaded()
            self.passage_path = self._get_passage_path()
        
        # Load components
        self.doc_ids = self._load_document_ids()
        self.doc_texts = self._load_tsv()
        self.index = self._initialize_searcher()
        
        # Load model and tokenizer
        self.model_hf = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _initialize_searcher(self):
        """Initialize FAISS index for BGE retrieval."""
        return self._build_faiss_index()
    
    def _ensure_index_and_passages_downloaded(self):
        """Download and extract BGE index and passages if needed."""
        os.makedirs(self.index_path, exist_ok=True)
        
        if self.index_type not in self.index_manager.index_configs.get("bge", {}):
            raise ValueError(f"Unsupported BGE index type: {self.index_type}")
        
        config = self.index_manager.index_configs["bge"][self.index_type]
        
        # Check if required files exist
        required_files = [
            os.path.join(self.index_path, "bge_doc_ids.pkl"),
            os.path.join(self.index_path, "bge_embeddings.h5")
        ]
        
        if not all(os.path.exists(f) for f in required_files):
            urls = config.get("urls")
            if isinstance(urls, list):
                # Multi-part download
                for url in urls:
                    filename = self._extract_filename_from_url(url)
                    local_path = os.path.join(self.index_path, filename)
                    if not os.path.exists(local_path):
                        self._download_file(url, local_path)
                self._extract_multi_part_archive()
            else:
                # Single file download
                zip_path = os.path.join(self.index_path, "index.zip")
                if not os.path.exists(zip_path):
                    self._download_file(urls, zip_path)
                self._extract_zip_files()
        
        # Download passages if needed
        passages_url = config.get("passages_url")
        if passages_url:
            passage_filename = self._extract_filename_from_url(passages_url)
            passage_path = os.path.join(self.index_manager.cache_dir, passage_filename)
            if not os.path.exists(passage_path):
                self._download_file(passages_url, passage_path)
    
    def _get_passage_path(self):
        """Get path to passages file."""
        if self.index_type not in self.index_manager.index_configs.get("bge", {}):
            raise ValueError(f"Unsupported BGE index type: {self.index_type}")
        
        config = self.index_manager.index_configs["bge"][self.index_type]
        passages_url = config.get("passages_url")
        if passages_url:
            passage_filename = self._extract_filename_from_url(passages_url)
            return os.path.join(self.index_manager.cache_dir, passage_filename)
        return None
    
    def _extract_filename_from_url(self, url):
        """Extract filename from URL."""
        parsed_url = urlparse(url)
        return os.path.basename(parsed_url.path).split('?')[0]
    
    def _download_file(self, url: str, save_path: str):
        """Download file from URL with progress bar."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), 
                            desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)
    
    def _extract_zip_files(self):
        """Extract ZIP files in the index folder."""
        zip_files = [f for f in os.listdir(self.index_path) if f.endswith(".zip")]
        
        for zip_file in zip_files:
            zip_path = os.path.join(self.index_path, zip_file)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                    
                    extracted_path = os.path.join(self.index_path, filename)
                    with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)
            
            print(f"Extracted {zip_file}")
            os.remove(zip_path)
    
    def _extract_multi_part_archive(self):
        """Extract multi-part tar.gz archives."""
        parts = sorted([f for f in os.listdir(self.index_path) if f.startswith("bgb_index.tar.")],
                      key=lambda x: x.split('.')[-1])
        
        if not parts:
            return
        
        combined_path = os.path.join(self.index_path, "combined.tar.gz")
        
        # Combine parts
        with open(combined_path, "wb") as combined:
            for part in parts:
                with open(os.path.join(self.index_path, part), "rb") as part_file:
                    shutil.copyfileobj(part_file, combined)
        
        try:
            # Decompress and extract
            tar_file = combined_path.replace(".gz", "")
            with gzip.open(combined_path, "rb") as f_in, open(tar_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            shutil.unpack_archive(tar_file, self.index_path)
            os.remove(tar_file)
            
        except Exception as e:
            raise RuntimeError(f"Error extracting multi-part archive: {e}")
        finally:
            # Cleanup
            if os.path.exists(combined_path):
                os.remove(combined_path)
            for part in parts:
                part_path = os.path.join(self.index_path, part)
                if os.path.exists(part_path):
                    os.remove(part_path)
    
    def _build_faiss_index(self):
        """Build or load FAISS index."""
        print("Handling FAISS index...")
        
        index_path = os.path.join(self.index_path, "faiss_index.bin")
        embeddings_path = os.path.join(self.index_path, "bge_embeddings.h5")
        
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}...")
            index = faiss.read_index(index_path)
        else:
            print(f"Building FAISS index from embeddings at {embeddings_path}...")
            with h5py.File(embeddings_path, "r") as f:
                embeddings = f["embeddings"][:].astype(np.float32)
            
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)
            
            # Add embeddings in chunks
            chunk_size = 50000
            for start in tqdm(range(0, embeddings.shape[0], chunk_size), 
                            desc="Adding embeddings to FAISS index"):
                end = min(start + chunk_size, embeddings.shape[0])
                chunk = embeddings[start:end]
                index.add(chunk)
            
            print(f"Saving FAISS index to {index_path}...")
            faiss.write_index(index, index_path)
        
        print(f"FAISS index loaded with {index.ntotal} embeddings.")
        return index
    
    def _load_document_ids(self):
        """Load document IDs from pickle file."""
        doc_ids_path = os.path.join(self.index_path, "bge_doc_ids.pkl")
        print(f"Loading document IDs from {doc_ids_path}...")
        
        doc_ids = []
        with open(doc_ids_path, "rb") as f:
            while True:
                try:
                    doc_ids.extend(pickle.load(f))
                except EOFError:
                    break
        
        print(f"Loaded {len(doc_ids)} document IDs.")
        return doc_ids
    
    def _load_tsv(self):
        """Load document texts from TSV file."""
        if not self.passage_path or not os.path.exists(self.passage_path):
            print("Warning: Passage file not found, using empty corpus")
            return {}
        
        doc_texts = {}
        with open(self.passage_path, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                try:
                    doc_id, passage, title = line.strip().split("\t")
                    doc_texts[doc_id] = {"text": passage, "title": title}
                except ValueError:
                    continue  # Skip malformed lines
        
        print(f"Loaded {len(doc_texts)} passages.")
        return doc_texts
    
    def _encode_queries(self, queries: List[str]):
        """Encode queries into dense embeddings."""
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), self.batch_size), desc="Encoding queries"):
                batch = queries[i:i + self.batch_size]
                tokenized = self.tokenizer(batch, padding=True, truncation=True, 
                                         return_tensors="pt").to(self.device)
                model_output = self.model_hf(**tokenized)
                embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using BGE."""
        queries = [doc.question.question for doc in documents]
        print(f"Retrieving {len(documents)} documents with BGE...")
        
        # Encode queries
        query_embeddings = self._encode_queries(queries)
        
        # Check dimension compatibility
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(f"Dimension mismatch: queries={query_embeddings.shape[1]}, "
                           f"index={self.index.d}")
        
        # Batch FAISS search
        all_distances = []
        all_indices = []
        
        for start_idx in tqdm(range(0, len(query_embeddings), self.batch_size), 
                            desc="FAISS search"):
            end_idx = min(start_idx + self.batch_size, len(query_embeddings))
            batch_embeddings = query_embeddings[start_idx:end_idx]
            
            distances, indices = self.index.search(batch_embeddings, self.n_docs)
            all_distances.append(distances)
            all_indices.append(indices)
        
        # Combine results
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        
        # Process results
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            contexts = []
            for dist, idx in zip(all_distances[i], all_indices[i]):
                try:
                    context = self._create_context_from_result(idx, dist, document)
                    if context:
                        contexts.append(context)
                except Exception as e:
                    print(f"Error processing result {idx}: {e}")
            
            document.contexts = contexts
        
        return documents
    
    def _create_context_from_result(self, idx: int, score: float, document: Document) -> Context:
        """Create Context object from search result."""
        doc_id = self.doc_ids[idx]
        doc_data = self.doc_texts.get(str(doc_id), {
            "text": "Text not found", 
            "title": "No Title"
        })
        
        return Context(
            id=doc_id,
            title=doc_data["title"],
            text=doc_data["text"],
            score=float(score),
            has_answer=has_answers(doc_data["text"], document.answers.answers, self.tokenizer_simple)
        )