# rankify/indexing/ance_indexer.py - FIXED VERSION
import shutil
import logging
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_pyserini_jsonl_dense , to_tsv

logging.basicConfig(level=logging.INFO)

class ANCEIndexer(BaseIndexer):
    """
    FIXED ANCE Indexer using proper ANCE models.
    
    Key fixes:
    - Uses single model for both indexing and retrieval (same as ANCE paper)
    - Defaults to proper castorini/ance-msmarco-passage model
    - Consistent with ANCERetriever
    
    Supported models:
    - castorini/ance-msmarco-passage (recommended - works for both queries and passages)
    - castorini/ance-msmarco-doc-maxp
    - castorini/ance-msmarco-doc-firstp
    
    Args:
        corpus_path (str): Path to the corpus file.
        encoder_name (str): ANCE model name (default: castorini/ance-msmarco-passage).
        output_dir (str): Directory to save the index.
        chunk_size (int): Size of chunks to process the corpus.
        threads (int): Number of threads to use for processing.
        index_type (str): Type of index to build (default: "custom").
        batch_size (int): Batch size for encoding passages.
        device (str): Device to use for encoding ("cpu" or "cuda").
    """

    def __init__(self,
                 corpus_path,
                 encoder_name="castorini/ance-dpr-context-multi",  # ✅ FIXED: Same default as retriever
                 output_dir="ance_indices",
                 chunk_size=100,
                 threads=32,
                 index_type="custom",
                 retriever_name="ance",
                 batch_size=16,
                 device="cuda"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type, retriever_name)
        self.encoder_name = encoder_name
        self.index_dir = self.output_dir / f"ance_index_{index_type}"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        logging.info(f"🚀 ANCE Indexer initialized")
        logging.info(f"  📚 Corpus: {corpus_path}")
        logging.info(f"  🤖 Model: {encoder_name}")
        logging.info(f"  💾 Output: {self.index_dir}")
        logging.info(f"  🖥️ Device: {self.device}")

    def _save_dense_corpus(self):
        """Convert the corpus to dense-compatible format."""
        return to_pyserini_jsonl_dense(self.corpus_path, self.output_dir, self.chunk_size, self.threads)

    def _load_ance_model(self):
        """Load ANCE model and tokenizer."""
        logging.info(f"Loading ANCE model: {self.encoder_name}")
        
        try:
            # Try to import AnceEncoder (from newer transformers)
            from transformers import AnceEncoder
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            model = AnceEncoder.from_pretrained(self.encoder_name)
            logging.info("✅ Loaded using AnceEncoder")
        except ImportError:
            # Fallback to AutoModel if AnceEncoder not available
            logging.warning("AnceEncoder not available, using AutoModel")
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            model = AutoModel.from_pretrained(self.encoder_name)
        
        model = model.to(self.device)
        model.eval()
        
        return tokenizer, model

    def _encode_batch(self, texts, tokenizer, model):
        """Encode a batch of texts using ANCE."""
        # Prepare inputs
        inputs = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Encode with ANCE
        with torch.no_grad():
            if hasattr(model, 'encode'):
                # Some ANCE models have an encode method
                embeddings = model.encode(inputs["input_ids"])
            else:
                # Fallback to standard forward pass
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use [CLS] token representation
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    embeddings = outputs[0][:, 0, :]  # First token of first output
        
        return embeddings.detach().cpu().numpy()

    def _create_id_mapping(self, doc_ids):
        """Create mapping between FAISS sequential IDs and original document IDs."""
        logging.info("Creating ID mapping for FAISS index...")
        
        faiss_to_docid = {i: str(doc_id) for i, doc_id in enumerate(doc_ids)}
        docid_to_faiss = {str(doc_id): i for i, doc_id in enumerate(doc_ids)}
        
        logging.info(f"Created ID mapping for {len(faiss_to_docid)} documents")
        return faiss_to_docid, docid_to_faiss

    def _save_id_mapping(self, faiss_to_docid, docid_to_faiss):
        """Save ID mappings to files."""
        # Save FAISS index to original docid mapping
        faiss_to_docid_file = self.index_dir / "faiss_to_docid.json"
        with open(faiss_to_docid_file, 'w', encoding='utf-8') as f:
            json.dump(faiss_to_docid, f, indent=2)
        
        # Save original docid to FAISS index mapping  
        docid_to_faiss_file = self.index_dir / "docid_to_faiss.json"
        with open(docid_to_faiss_file, 'w', encoding='utf-8') as f:
            json.dump(docid_to_faiss, f, indent=2)
            
        logging.info(f"ID mappings saved to {self.index_dir}")

    def _save_corpus_metadata(self, corpus_file, doc_ids):
        """Save corpus metadata for retrieval."""
        logging.info("Saving corpus metadata...")
        
        corpus_metadata = {}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                doc = json.loads(line.strip())
                doc_id = doc_ids[line_idx] if line_idx < len(doc_ids) else f"doc_{line_idx}"
                
                # Extract title and contents
                contents = doc.get('contents', '') or doc.get('text', '')
                title = doc.get('title', '')
                
                # If no explicit title, try to extract from contents
                if not title and contents:
                    lines = contents.split('\n')
                    title = lines[0] if lines else "No Title"
                
                corpus_metadata[str(doc_id)] = {
                    'title': title,
                    'contents': contents,
                    'text': contents
                }
        
        # Save corpus metadata
        corpus_metadata_file = self.index_dir / "corpus_metadata.json"
        with open(corpus_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_metadata, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Corpus metadata saved to {corpus_metadata_file}")

    def _save_model_config(self):
        """Save model configuration for retrieval."""
        model_config = {
            "encoder_name": self.encoder_name,
            "model_type": "ance",
            "index_type": self.index_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "embedding_dim": 768,  # Standard ANCE dimension
            "similarity": "inner_product"  # ANCE uses inner product
        }
        
        config_file = self.index_dir / "model_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2)
        
        logging.info(f"Model config saved to {config_file}")

    def build_index(self):
        """
        Build the ANCE dense index using FAISS.
        
        Steps:
        1. Convert corpus to proper format
        2. Load ANCE model and tokenizer
        3. Encode all documents in batches
        4. Build FAISS index
        5. Save mappings and metadata
        """
        # Step 1: Prepare corpus
        corpus_path = self._save_dense_corpus()
        
        # Create output directory
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)
        
        # Step 2: Load ANCE model
        tokenizer, model = self._load_ance_model()
        
        # Step 3: Encode documents
        logging.info("🔍 Encoding documents with ANCE...")
        embeddings = []
        doc_ids = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            batch_texts = []
            batch_ids = []
            
            for line_idx, line in enumerate(tqdm(f, desc="Processing documents")):
                doc = json.loads(line.strip())
                doc_id = doc.get('docid') or doc.get('id', f"doc_{line_idx}")
                text = doc.get('contents', '') or doc.get('text', '')
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                
                # Process batch when full
                if len(batch_texts) >= self.batch_size:
                    batch_embeddings = self._encode_batch(batch_texts, tokenizer, model)
                    embeddings.extend(batch_embeddings)
                    doc_ids.extend(batch_ids)
                    
                    batch_texts = []
                    batch_ids = []
            
            # Process remaining documents
            if batch_texts:
                batch_embeddings = self._encode_batch(batch_texts, tokenizer, model)
                embeddings.extend(batch_embeddings)
                doc_ids.extend(batch_ids)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        logging.info(f"📊 Encoded {len(embeddings)} documents with {embeddings.shape[1]}D embeddings")
        
        # Step 4: Build FAISS index
        logging.info("🏗️ Building FAISS index...")
        
        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product for ANCE)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Save FAISS index
        faiss_index_file = self.index_dir / "index"
        faiss.write_index(index, str(faiss_index_file))
        
        # Create docid file (required by some FAISS searchers)
        docid_file = self.index_dir / "docid"
        with open(docid_file, 'w', encoding='utf-8') as f:
            for doc_id in doc_ids:
                f.write(f"{doc_id}\n")
        
        # Step 5: Save mappings and metadata
        faiss_to_docid, docid_to_faiss = self._create_id_mapping(doc_ids)
        self._save_id_mapping(faiss_to_docid, docid_to_faiss)
        self._save_corpus_metadata(corpus_path, doc_ids)
        self._save_model_config()
        
        # Save title map (inherited from base class)
        self._save_title_map()
        
        # Clean up temporary corpus file
        if corpus_path.exists():
            corpus_path.unlink()
        
        logging.info(f"✅ ANCE indexing complete! Index stored at {self.index_dir}")
        logging.info(f"📁 Index files created:")
        logging.info(f"  - index.faiss (FAISS index)")
        logging.info(f"  - docid (document IDs)")
        logging.info(f"  - faiss_to_docid.json (ID mapping)")
        logging.info(f"  - docid_to_faiss.json (reverse ID mapping)")
        logging.info(f"  - corpus_metadata.json (document content)")
        logging.info(f"  - model_config.json (model configuration)")
        logging.info(f"  - title_map.json (document titles)")
        
        return self.index_dir

    def load_index(self):
        """Load the ANCE index from the specified directory."""
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist or is empty.")
        
        # Check for required files
        required_files = [
            "index.faiss",
            "docid",
            "faiss_to_docid.json",
            "corpus_metadata.json",
            "model_config.json"
        ]
        
        for file_name in required_files:
            file_path = self.index_dir / file_name
            if not file_path.exists():
                logging.warning(f"Missing index file: {file_path}")
        
        logging.info(f"✅ ANCE index loaded from {self.index_dir}")
        return self.index_dir