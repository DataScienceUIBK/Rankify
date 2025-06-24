import shutil
import logging
import json
import csv
from pathlib import Path

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_tsv
from rankify.utils.retrievers.colbert.colbert import Indexer
from rankify.utils.retrievers.colbert.colbert.data import Collection
from rankify.utils.retrievers.colbert.colbert.infra import ColBERTConfig
from rankify.utils.retrievers.colbert.colbert.search.index_loader import IndexLoader

logging.basicConfig(level=logging.INFO)
import os
 
os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "true"

class ColBERTIndexer(BaseIndexer):
    """
    ColBERT Indexer that properly handles string IDs by converting them to sequential integers.
    
    This indexer:
    1. Converts string IDs to sequential integers (required by ColBERT)
    2. Maintains ID mappings for retrieval
    3. Creates proper TSV header format
    4. Generates all necessary files for the retriever
    """
    
    def __init__(self,
                 corpus_path,
                 encoder_name="colbert-ir/colbertv2.0",
                 output_dir="rankify_indices",
                 chunk_size=100,
                 threads=32,
                 index_type="wiki",
                 batch_size=64,
                 retriever_name="colbert",
                 device="cuda"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type, retriever_name)
        self.encoder_name = encoder_name
        self.index_dir = self.output_dir / f"colbert_index_{index_type}"
        self.device = device
        self.batch_size = batch_size
        self.id_mapping_file = self.index_dir / "id_mapping.json"
        self.passages_file = self.index_dir / "passages.tsv"  # For retriever compatibility

    def _create_colbert_tsv_from_jsonl(self) -> tuple[str, dict, dict]:
        """
        Create ColBERT-compatible TSV directly from JSONL corpus.
        
        Returns:
            tuple: (tsv_path, original_to_sequential_mapping, sequential_to_original_mapping)
        """
        logging.info("Creating ColBERT TSV directly from JSONL corpus...")
        
        colbert_tsv_path = self.index_dir / "collection.tsv"  # ColBERT collection file
        passages_tsv_path = self.passages_file  # For retriever
        
        original_to_sequential = {}
        sequential_to_original = {}
        
        # Create both files simultaneously
        with open(self.corpus_path, 'r', encoding='utf-8') as infile, \
             open(colbert_tsv_path, 'w', encoding='utf-8', newline='') as collection_file, \
             open(passages_tsv_path, 'w', encoding='utf-8', newline='') as passages_file:
            
            collection_writer = csv.writer(collection_file, delimiter='\t')
            passages_writer = csv.writer(passages_file, delimiter='\t')
            
            # Write headers
            collection_writer.writerow(['id', 'text', 'title'])  # ColBERT format
            passages_writer.writerow(['id', 'text', 'title'])    # Retriever format
            
            sequential_id = 1  # Start from 1 to match line indices
            processed_docs = 0
            
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    
                    # Extract fields with fallbacks
                    original_id = doc.get("id", f"doc_{line_idx}")
                    text = doc.get("contents", doc.get("text", doc.get("passage", ""))).strip()
                    title = doc.get("title", "").strip()
                    
                    if not text:  # Skip empty documents
                        logging.warning(f"Skipping document {original_id} with empty text")
                        continue
                    
                    # Clean text and title for TSV (remove problematic characters)
                    text = self._clean_tsv_field(text)
                    title = self._clean_tsv_field(title)
                    
                    # Create mappings
                    original_to_sequential[original_id] = sequential_id
                    sequential_to_original[sequential_id] = original_id
                    
                    # Write to both files
                    collection_writer.writerow([sequential_id, text, title])  # For ColBERT
                    passages_writer.writerow([original_id, text, title])      # For retriever (keeps original IDs)
                    
                    sequential_id += 1
                    processed_docs += 1
                    
                    if processed_docs % 10000 == 0:
                        logging.info(f"Processed {processed_docs} documents...")
                        
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON at line {line_idx}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing document at line {line_idx}: {e}")
                    continue
        
        logging.info(f"✅ Created ColBERT TSV with {processed_docs} documents")
        logging.info(f"   Collection file: {colbert_tsv_path}")
        logging.info(f"   Passages file: {passages_tsv_path}")
        
        return str(colbert_tsv_path), original_to_sequential, sequential_to_original

    def _clean_tsv_field(self, field: str) -> str:
        """Clean field for TSV format by removing problematic characters."""
        if not field:
            return ""
        
        # Replace tabs, newlines, and carriage returns with spaces
        field = field.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        
        # Remove multiple spaces
        field = ' '.join(field.split())
        
        # Truncate if too long (prevent memory issues)
        if len(field) > 10000:
            field = field[:10000] + "..."
            
        return field

    def _save_id_mappings(self, original_to_sequential: dict, sequential_to_original: dict):
        """Save ID mappings for retriever use."""
        mappings = {
            "original_to_sequential": original_to_sequential,
            "sequential_to_original": {str(k): v for k, v in sequential_to_original.items()},
            "total_documents": len(original_to_sequential),
            "encoder_name": self.encoder_name,
            "index_type": self.index_type
        }
        
        with open(self.id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Saved ID mappings to {self.id_mapping_file}")
        logging.info(f"   Mapped {len(original_to_sequential)} original IDs to sequential IDs")

    def _verify_colbert_tsv(self, tsv_path: str):
        """Verify that the TSV format is correct for ColBERT."""
        logging.info(f"Verifying ColBERT TSV format: {tsv_path}")
        
        issues_found = []
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("TSV file is empty")
        
        # Check header
        header = lines[0].strip().split('\t')
        expected_header = ['id', 'text', 'title']
        if header != expected_header:
            issues_found.append(f"Header mismatch: expected {expected_header}, got {header}")
        
        # Check first 10 data lines
        for i in range(1, min(11, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                issues_found.append(f"Line {i}: Insufficient columns: {len(parts)}")
                continue
            
            pid = parts[0]
            text = parts[1]
            
            # Verify PID is integer and matches expected sequence
            try:
                pid_int = int(pid)
                expected_pid = i  # Line index should match PID for ColBERT
                if pid_int != expected_pid:
                    issues_found.append(f"Line {i}: PID {pid} doesn't match expected {expected_pid}")
            except ValueError:
                issues_found.append(f"Line {i}: PID '{pid}' is not an integer")
            
            # Check for empty text
            if not text.strip():
                issues_found.append(f"Line {i}: Empty text field")
        
        if issues_found:
            logging.error("❌ TSV verification failed:")
            for issue in issues_found:
                logging.error(f"   {issue}")
            raise ValueError(f"TSV format issues found: {len(issues_found)} problems")
        
        logging.info(f"✅ TSV verification passed. Total lines: {len(lines)}")

    def build_index(self):
        """Build the ColBERT index with proper string ID handling."""
        logging.info("Building ColBERT index with string ID support...")

        # Clean up existing index
        if self.index_dir.exists():
            logging.info(f"Removing existing index: {self.index_dir}")
            shutil.rmtree(self.index_dir)
        
        self.index_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create ColBERT-compatible TSV and ID mappings
            collection_path, original_to_sequential, sequential_to_original = self._create_colbert_tsv_from_jsonl()
            
            # Save ID mappings for retriever
            self._save_id_mappings(original_to_sequential, sequential_to_original)
            
            # Verify TSV format
            self._verify_colbert_tsv(collection_path)
            
        except Exception as e:
            logging.error(f"Failed to create ColBERT TSV: {e}")
            raise

        # Configure ColBERT
        config = ColBERTConfig(
            root=str(self.output_dir),
            index_root=str(self.output_dir),
            index_name=f"colbert_index_{self.index_type}",
            index_path=str(self.index_dir),
            checkpoint=self.encoder_name,
            collection=collection_path,
            doc_maxlen=180,        # Max document length
            nbits=2,               # Compression bits
            avoid_fork_if_possible=True,
            kmeans_niters=4,       # K-means iterations
            nranks=1,              # Single GPU/process
            bsize=self.batch_size,
            index_bsize=self.batch_size,
        )

        # Create indexer and build index
        logging.info(f"Initializing ColBERT indexer with model: {self.encoder_name}")
        indexer = Indexer(checkpoint=self.encoder_name, config=config, verbose=2)
        
        try:
            logging.info("Starting ColBERT indexing process...")
            index_path = indexer.index(
                name=config.index_name, 
                collection=Collection.cast(collection_path), 
                overwrite=True,
            )
            
            logging.info(f"✅ ColBERT index successfully built!")
            logging.info(f"   Index path: {index_path}")
            logging.info(f"   Collection: {collection_path}")
            logging.info(f"   Passages file: {self.passages_file}")
            logging.info(f"   ID mappings: {self.id_mapping_file}")
            
            # Save title map for compatibility
            self._save_title_map()
            
            return index_path
            
        except Exception as e:
            logging.error(f"❌ ColBERT indexing failed: {e}")
            
            # Provide debugging info
            self._debug_indexing_failure(collection_path)
            raise

    def _debug_indexing_failure(self, collection_path: str):
        """Debug helper for indexing failures."""
        logging.info("=== DEBUGGING INDEXING FAILURE ===")
        
        # Check collection file
        if not os.path.exists(collection_path):
            logging.error(f"Collection file missing: {collection_path}")
            return
        
        # Check file size and format
        file_size = os.path.getsize(collection_path) / (1024 * 1024)  # MB
        logging.info(f"Collection file size: {file_size:.2f} MB")
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logging.info(f"Collection lines: {len(lines)}")
        
        if len(lines) > 0:
            logging.info(f"First line (header): {lines[0].strip()}")
        if len(lines) > 1:
            logging.info(f"Second line (data): {lines[1].strip()}")
        
        # Check for common issues
        if len(lines) < 2:
            logging.error("Collection has no data lines")
        elif len(lines) > 1000000:
            logging.warning(f"Large collection ({len(lines)} lines) - may need more memory")

    def load_index(self):
        """Load the ColBERT index."""
        logging.info(f"Loading ColBERT index from {self.index_dir}...")
        
        try:
            # Check if index files exist
            index_files = list(self.index_dir.glob("*.pt"))  # PyTorch files
            if not index_files:
                raise FileNotFoundError(f"No index files found in {self.index_dir}")
            
            # Load using IndexLoader
            index = IndexLoader(index_path=str(self.index_dir), use_gpu=(self.device != "cpu"))
            logging.info(f"✅ Successfully loaded ColBERT index")
            return index
            
        except Exception as e:
            logging.error(f"❌ Failed to load ColBERT index: {e}")
            return None

    def load_id_mappings(self):
        """Load ID mappings for retrieval."""
        if not self.id_mapping_file.exists():
            logging.warning(f"ID mapping file not found: {self.id_mapping_file}")
            return {}, {}
        
        try:
            with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            original_to_sequential = mappings.get("original_to_sequential", {})
            sequential_to_original = {int(k): v for k, v in mappings.get("sequential_to_original", {}).items()}
            
            logging.info(f"Loaded ID mappings: {len(original_to_sequential)} entries")
            return original_to_sequential, sequential_to_original
            
        except Exception as e:
            logging.error(f"Error loading ID mappings: {e}")
            return {}, {}

    def get_original_id(self, sequential_id: int) -> str:
        """Convert sequential ID back to original ID."""
        _, sequential_to_original = self.load_id_mappings()
        return sequential_to_original.get(sequential_id, str(sequential_id))

    def get_sequential_id(self, original_id: str) -> int:
        """Convert original ID to sequential ID."""
        original_to_sequential, _ = self.load_id_mappings()
        return original_to_sequential.get(original_id, -1)