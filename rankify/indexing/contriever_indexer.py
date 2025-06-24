import shutil
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_contriever_embedding_chunks, to_pyserini_jsonl_dense,to_tsv
from rankify.utils.retrievers.contriever.index import Indexer

logging.basicConfig(level=logging.INFO)

class ContrieverIndexer(BaseIndexer):
    """
    Corrected Contriever Indexer that efficiently builds dense FAISS-based indexes.

    Key improvements:
    - Memory-efficient batch processing
    - Proper error handling
    - Streamlined embedding indexing
    - Better resource management
    """

    def __init__(self,
                 corpus_path,
                 encoder_name="facebook/contriever",
                 output_dir="rankify_indices",
                 chunk_size=5000000,
                 threads=32,
                 index_type="wiki",
                 batch_size=1000000,
                 retriever_name="contriever",
                 device="cuda",
                 embedding_batch_size=32):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type, retriever_name)
        self.encoder_name = encoder_name
        self.device = device
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size  # For embedding generation
        self.index_dir = self.output_dir / f"contriever_index_{index_type}"
        
        # Validate parameters
        if not Path(corpus_path).exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    def _save_dense_embeddings(self) -> List[Path]:
        """
        Convert the corpus into Contriever embeddings stored in shard .pkl files.
        Returns list of paths to embedding files.
        """
        logging.info("Generating Contriever embeddings...")
        
        try:
            embedding_files = to_contriever_embedding_chunks(
                self.corpus_path,
                self.index_dir,
                self.chunk_size,
                self.encoder_name,
                self.embedding_batch_size,  # Use smaller batch for embeddings
                self.device
            )
            
            if not embedding_files:
                raise ValueError("No embedding files were generated")
                
            logging.info(f"Generated {len(embedding_files)} embedding files")
            return embedding_files
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

    def _save_corpus(self) -> str:
        """Convert the corpus to dense-compatible Pyserini format."""
        logging.info("Converting corpus to Pyserini JSONL format...")
        return to_tsv(self.corpus_path, self.index_dir, self.chunk_size, self.threads)

    def build_index(self):
        """Build the Contriever index with improved error handling and efficiency."""
        try:
            # Clean and create index directory
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True)

            # Convert corpus
            self._save_corpus()

            # Generate embeddings
            embedding_files = self._save_dense_embeddings()

            # Create index
            logging.info("Creating Contriever index...")
            index = Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)

            # Index embeddings efficiently
            self._index_encoded_data_efficient(index, embedding_files)

            # Serialize index
            logging.info("Serializing index...")
            index.serialize(self.index_dir)

            # Save title map
            self._save_title_map()

            # Clean up embedding files to save space
            self._cleanup_embedding_files(embedding_files)

            logging.info(f"✅ Contriever index successfully stored at {self.index_dir}")
            
        except Exception as e:
            logging.error(f"Error building index: {e}")
            # Clean up on failure
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            raise

    def _index_encoded_data_efficient(self, index: Indexer, embedding_files: List[Path]):
        """
        Efficiently loads and indexes encoded passage embeddings into FAISS.
        
        This version processes files one at a time and batches within each file,
        avoiding memory issues with large corpora.
        """
        total_indexed = 0
        
        for file_path in tqdm(embedding_files, desc="Indexing embedding files"):
            try:
                logging.info(f"Processing embedding file: {file_path}")
                
                # Load embeddings from file
                with open(file_path, "rb") as fin:
                    ids, embeddings = pickle.load(fin)
                
                if len(ids) != len(embeddings):
                    raise ValueError(f"Mismatch between IDs ({len(ids)}) and embeddings ({len(embeddings)}) in {file_path}")
                
                # Process this file's embeddings in batches
                num_embeddings = len(embeddings)
                for start_idx in tqdm(range(0, num_embeddings, self.batch_size), 
                                    desc=f"Batching {file_path.name}", leave=False):
                    end_idx = min(start_idx + self.batch_size, num_embeddings)
                    
                    batch_ids = ids[start_idx:end_idx]
                    batch_embeddings = embeddings[start_idx:end_idx]
                    
                    # Validate embeddings
                    if not isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = np.array(batch_embeddings)
                    
                    if batch_embeddings.dtype != np.float32:
                        batch_embeddings = batch_embeddings.astype(np.float32)
                    
                    # Add to index
                    index.index_data(batch_ids, batch_embeddings)
                    total_indexed += len(batch_ids)
                
                logging.info(f"Processed {num_embeddings} embeddings from {file_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                raise
        
        logging.info(f"✅ Successfully indexed {total_indexed} embeddings")

    def _cleanup_embedding_files(self, embedding_files: List[Path]):
        """Clean up embedding files after indexing to save disk space."""
        try:
            for file_path in embedding_files:
                if file_path.exists():
                    file_path.unlink()
            logging.info(f"Cleaned up {len(embedding_files)} embedding files")
        except Exception as e:
            logging.warning(f"Error cleaning up embedding files: {e}")

    def load_index(self) -> Indexer:
        """Load the Contriever index from the specified directory."""
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist")
        
        if not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} is empty")

        try:
            index = Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)
            index.deserialize_from(self.index_dir)
            logging.info(f"✅ Successfully loaded index from {self.index_dir}")
            return index
            
        except Exception as e:
            logging.error(f"Error loading index from {self.index_dir}: {e}")
            raise

    def validate_index(self) -> bool:
        """Validate that the index was built correctly."""
        try:
            index = self.load_index()
            
            # Check if index has embeddings
            if hasattr(index, 'index') and hasattr(index.index, 'ntotal'):
                num_vectors = index.index.ntotal
                logging.info(f"Index contains {num_vectors} vectors")
                return num_vectors > 0
            
            return True
            
        except Exception as e:
            logging.error(f"Index validation failed: {e}")
            return False

    def get_index_stats(self) -> dict:
        """Get statistics about the built index."""
        try:
            index = self.load_index()
            stats = {
                "index_type": "contriever",
                "vector_dimension": 768,
                "index_directory": str(self.index_dir),
                "encoder_model": self.encoder_name
            }
            
            if hasattr(index, 'index') and hasattr(index.index, 'ntotal'):
                stats["num_vectors"] = index.index.ntotal
            
            # Check index directory size
            total_size = sum(f.stat().st_size for f in self.index_dir.rglob('*') if f.is_file())
            stats["index_size_mb"] = total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting index stats: {e}")
            return {"error": str(e)}