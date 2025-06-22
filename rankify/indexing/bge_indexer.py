import shutil
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_tsv
from tqdm import tqdm
import faiss
import pickle
import h5py
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)

class BGEIndexer(BaseIndexer):
    """
    Corrected BGE Indexer that builds dense FAISS-based indexes with proper normalization.

    Key improvements:
    - L2 normalization of embeddings
    - Proper CLS token extraction for BGE models
    - Cosine similarity via normalized embeddings
    - Better error handling and validation
    """

    def __init__(self,
                 corpus_path,
                 encoder_name="BAAI/bge-large-en-v1.5",
                 output_dir="rankify_indices",
                 index_dir=None,
                 chunk_size=50000,
                 threads=32,
                 index_type="wiki",
                 batch_size=32,
                 retriever_name="bge",
                 device="cuda"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type, retriever_name)
        self.encoder_name = encoder_name
        self.device = device
        self.batch_size = batch_size
        if not index_dir:
            self.index_dir = self.output_dir / f"bge_index_{index_type}"
        else:
            self.index_dir = Path(index_dir)

    def _normalize_embeddings(self, embeddings):
        """L2 normalize embeddings for cosine similarity."""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _extract_bge_embeddings(self, texts, tokenizer, model):
        """
        Extract embeddings using BGE model with proper CLS token handling.
        """
        # Tokenize with proper parameters for BGE
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # BGE uses CLS token (first token) representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
        return embeddings.cpu().numpy()

    def build_faiss_index(self):
        """Build FAISS index with normalized embeddings for cosine similarity."""
        embeddings_path = self.index_dir / "bge_embeddings.h5"
        logging.info(f"Loading embeddings from {embeddings_path}...")
        
        with h5py.File(embeddings_path, "r") as f:
            embeddings = f["embeddings"][:].astype(np.float32)

        # Normalize embeddings for cosine similarity
        logging.info("Normalizing embeddings for cosine similarity...")
        embeddings = self._normalize_embeddings(embeddings)

        embedding_dim = embeddings.shape[1]
        logging.info(f"Building FAISS index with embedding dimension {embedding_dim}...")
        
        # Use IndexFlatIP with normalized embeddings = cosine similarity
        index = faiss.IndexFlatIP(embedding_dim)

        # Add embeddings in batches for memory efficiency
        chunk_size = 50000
        for start in tqdm(range(0, embeddings.shape[0], chunk_size), desc="Adding embeddings to FAISS index"):
            end = min(start + chunk_size, embeddings.shape[0])
            index.add(embeddings[start:end])

        index_path = self.index_dir / "faiss_index.bin"
        logging.info(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, str(index_path))

    def _merge_embedding_chunks(self):
        """Merge embedding chunks with proper validation."""
        chunk_files = list(self.index_dir.glob("embeddings_chunk_*.pkl"))
        logging.info(f"Found {len(chunk_files)} embedding chunk files to merge.")

        if not chunk_files:
            raise ValueError("No embedding chunk files found to merge.")

        all_embeddings = []
        all_ids = []
        dims = []

        for file in tqdm(chunk_files, desc="Loading embedding chunks"):
            try:
                with open(file, "rb") as fin:
                    chunk = pickle.load(fin)
                
                if not isinstance(chunk, list):
                    raise ValueError(f"Unexpected format in {file.name}: {type(chunk)}")

                for entry in chunk:
                    embedding = entry.get("embedding")
                    doc_id = entry.get("id")
                    
                    if embedding is None or doc_id is None:
                        logging.warning(f"Missing keys in chunk entry from {file.name}")
                        continue
                        
                    # Ensure embedding is numpy array
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    all_embeddings.append(embedding)
                    all_ids.append(doc_id)
                    dims.append(len(embedding))
                    
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
                continue

        if not all_embeddings:
            raise ValueError("No valid embeddings found in chunk files.")

        if len(set(dims)) != 1:
            raise ValueError(f"Embedding dimension mismatch: found dimensions {set(dims)}")

        # Stack embeddings and normalize
        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        logging.info(f"Loaded {len(all_embeddings)} embeddings with dimension {all_embeddings.shape[1]}")

        # Save merged embeddings
        merged_embedding_path = self.index_dir / "bge_embeddings.h5"
        with h5py.File(merged_embedding_path, "w") as hf:
            hf.create_dataset("embeddings", data=all_embeddings)

        # Save merged IDs
        merged_ids_path = self.index_dir / "bge_doc_ids.pkl"
        with open(merged_ids_path, "wb") as f:
            pickle.dump(all_ids, f)

        # Clean up chunk files
        for file in chunk_files:
            file.unlink()

        logging.info(f"Merged embeddings saved to {merged_embedding_path}")
        logging.info(f"Merged IDs saved to {merged_ids_path}")

        return merged_embedding_path, merged_ids_path

    def _save_dense_embeddings(self):
        """Save dense embeddings with proper BGE handling."""
        logging.info(f"Loading BGE model: {self.encoder_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        model = AutoModel.from_pretrained(self.encoder_name).to(self.device)
        model.eval()

        chunk_dir = self.index_dir
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for chunk_lines, start_idx in tqdm(self._generate_chunks(f, self.chunk_size), desc="Embedding chunks"):
                chunk_data = self._bge_embed_chunk(chunk_lines, tokenizer, model)

                chunk_path = chunk_dir / f"embeddings_chunk_{start_idx}.pkl"
                with open(chunk_path, "wb") as fout:
                    pickle.dump(chunk_data, fout)

                chunk_paths.append(chunk_path)
                
        logging.info(f"Saved {len(chunk_paths)} embedding chunk files.")
        return chunk_paths

    def _generate_chunks(self, file_obj, chunk_size):
        """Generate chunks from file with line counting."""
        chunk = []
        start_idx = 0
        
        for i, line in enumerate(file_obj):
            chunk.append(line.strip())
            
            if len(chunk) >= chunk_size:
                yield chunk, start_idx
                chunk = []
                start_idx = i + 1
                
        if chunk:
            yield chunk, start_idx

    def _bge_embed_chunk(self, chunk_lines, tokenizer, model):
        """
        Corrected BGE embedding function with proper text handling.
        """
        chunk_data = []
        texts = []
        doc_ids = []
        
        # Parse documents from chunk
        for line in chunk_lines:
            if not line.strip():
                continue
                
            try:
                doc = json.loads(line)
                text = doc.get("text", doc.get("contents", "")).strip()
                doc_id = doc.get("id", f"doc_{len(doc_ids)}")
                
                if text:  # Only process non-empty texts
                    texts.append(text)
                    doc_ids.append(doc_id)
                    
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON line: {line[:100]}...")
                continue

        if not texts:
            return []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = doc_ids[i:i + self.batch_size]
            
            # Extract embeddings using corrected method
            embeddings = self._extract_bge_embeddings(batch_texts, tokenizer, model)
            
            # Store embeddings with IDs
            for doc_id, embedding in zip(batch_ids, embeddings):
                chunk_data.append({
                    "id": doc_id,
                    "embedding": embedding
                })
                
        return chunk_data

    def _save_corpus(self):
        """Convert the corpus to TSV format for indexing."""
        return to_tsv(self.corpus_path, self.index_dir, self.chunk_size, self.threads)

    def build_index(self):
        """Build the BGE index from the corpus."""
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        self._save_corpus()
        self._save_dense_embeddings()
        self._merge_embedding_chunks()
        self.build_faiss_index()

        logging.info(f"BGE index stored at {self.index_dir}")

    def load_index(self) -> tuple:
        """Load the FAISS index and document IDs from the index directory."""
        faiss_index_path = self.index_dir / "faiss_index.bin"
        merged_ids_path = self.index_dir / "bge_doc_ids.pkl"

        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {faiss_index_path}")
        if not merged_ids_path.exists():
            raise FileNotFoundError(f"Merged IDs file not found at {merged_ids_path}")

        logging.info(f"Loading FAISS index from {faiss_index_path}...")
        index = faiss.read_index(str(faiss_index_path))

        logging.info(f"Loading document IDs from {merged_ids_path}...")
        with open(merged_ids_path, "rb") as f:
            doc_ids = pickle.load(f)

        logging.info(f"Loaded FAISS index and {len(doc_ids)} document IDs from {self.index_dir}")

        return index, doc_ids