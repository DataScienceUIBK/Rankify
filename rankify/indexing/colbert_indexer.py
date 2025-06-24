import shutil
import logging
import json
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
    FIXED ColBERT Indexer that handles string IDs properly.
    
    Key fixes:
    - Converts string IDs to sequential integers for ColBERT compatibility
    - Maintains ID mapping for retrieval
    - Adds proper TSV header format
    - No modifications to ColBERT utils required
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

    def _generate_colbert_passages_tsv(self) -> str:
        """
        FIXED: Convert the corpus to a ColBERT-compatible TSV format.
        
        ColBERT expects:
        - Line 0 (header): 'id' as the first column
        - Line N (data): PID = N (must match line index)
        
        Example:
        Line 0: id\ttext\ttitle
        Line 1: 1\tSome text\tSome title
        Line 2: 2\tMore text\tAnother title
        
        This method converts string IDs to sequential integers starting from 1.
        """
        logging.info("Converting corpus to ColBERT-compatible TSV format...")
        
        # First generate the standard TSV
        original_tsv_path = to_tsv(self.corpus_path, self.index_dir, self.chunk_size, self.threads)
        
        # Create ColBERT-compatible TSV with sequential integer IDs
        colbert_tsv_path = self.index_dir / "colbert_passages.tsv"
        id_mapping = {}  # Maps original_id -> sequential_integer
        reverse_mapping = {}  # Maps sequential_integer -> original_id
        
        with open(original_tsv_path, 'r', encoding='utf-8') as infile, \
             open(colbert_tsv_path, 'w', encoding='utf-8') as outfile:
            
            # Write header for ColBERT compatibility
            outfile.write("id\ttext\ttitle\n")
            
            sequential_id = 1  # FIXED: Start from 1 to match line_idx
            
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                
                # Parse TSV line
                parts = line.split('\t')
                if len(parts) < 2:
                    logging.warning(f"Skipping malformed line {line_idx}: {line}")
                    continue
                
                original_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                title = parts[2] if len(parts) > 2 else ""
                
                # Create mappings
                id_mapping[original_id] = sequential_id
                reverse_mapping[sequential_id] = original_id
                
                # Write with sequential integer ID that matches line index
                outfile.write(f"{sequential_id}\t{text}\t{title}\n")
                sequential_id += 1
        
        # Save ID mappings for retrieval
        self._save_id_mappings(id_mapping, reverse_mapping)
        
        logging.info(f"Converted {sequential_id} passages to ColBERT format: {colbert_tsv_path}")
        return str(colbert_tsv_path)

    def _save_id_mappings(self, id_mapping: dict, reverse_mapping: dict):
        """Save ID mappings for later retrieval use."""
        mappings = {
            "original_to_sequential": id_mapping,
            "sequential_to_original": {str(k): v for k, v in reverse_mapping.items()}  # JSON keys must be strings
        }
        
        with open(self.id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2)
        
        logging.info(f"Saved ID mappings to {self.id_mapping_file}")

    def load_id_mappings(self):
        """Load ID mappings for retrieval."""
        if not self.id_mapping_file.exists():
            logging.warning(f"ID mapping file not found: {self.id_mapping_file}")
            return {}, {}
        
        with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        original_to_sequential = mappings.get("original_to_sequential", {})
        sequential_to_original = {int(k): v for k, v in mappings.get("sequential_to_original", {}).items()}
        
        return original_to_sequential, sequential_to_original

    def _generate_alternative_tsv_format(self) -> str:
        """
        Alternative method: Create TSV directly from JSONL with proper format.
        Use this if the standard to_tsv doesn't work with your data.
        """
        logging.info("Creating ColBERT TSV directly from JSONL...")
        
        colbert_tsv_path = self.index_dir / "colbert_passages.tsv"
        id_mapping = {}
        reverse_mapping = {}
        
        with open(self.corpus_path, 'r', encoding='utf-8') as infile, \
             open(colbert_tsv_path, 'w', encoding='utf-8') as outfile:
            
            # Write header
            outfile.write("id\ttext\ttitle\n")
            
            sequential_id = 1  # FIXED: Start from 1 to match line_idx
            
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    
                    # Extract fields
                    original_id = doc.get("id", f"doc_{line_idx}")
                    text = doc.get("contents", doc.get("text", "")).strip()
                    title = doc.get("title", "").strip()
                    
                    if not text:  # Skip empty documents
                        continue
                    
                    # Clean text and title for TSV (remove tabs and newlines)
                    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    title = title.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    
                    # Create mappings
                    id_mapping[original_id] = sequential_id
                    reverse_mapping[sequential_id] = original_id
                    
                    # Write TSV line
                    outfile.write(f"{sequential_id}\t{text}\t{title}\n")
                    sequential_id += 1
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON at line {line_idx}: {e}")
                    continue
        
        # Save mappings
        self._save_id_mappings(id_mapping, reverse_mapping)
        
        logging.info(f"Created ColBERT TSV with {sequential_id} passages: {colbert_tsv_path}")
        return str(colbert_tsv_path)

    def build_index(self):
        """
        Build the ColBERT index from the corpus with proper ID handling.
        """
        logging.info("Start Building ColBERT index...")

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        try:
            # Try standard conversion first
            collection_path = self._generate_colbert_passages_tsv()
        except Exception as e:
            logging.warning(f"Standard TSV conversion failed: {e}")
            logging.info("Trying alternative direct JSONL conversion...")
            collection_path = self._generate_alternative_tsv_format()

        # Verify the TSV format
        self._verify_tsv_format(collection_path)

        config = ColBERTConfig(
            root=str(self.output_dir),
            index_root=str(self.output_dir),
            index_name=f"colbert_index_{self.index_type}",
            index_path=str(self.index_dir),
            checkpoint=self.encoder_name,
            collection=str(collection_path),
            doc_maxlen=180,
            nbits=2,
            avoid_fork_if_possible = True,
            kmeans_niters=4,
            nranks=1,
            bsize=self.batch_size,
            index_bsize=self.batch_size,
        )

        indexer = Indexer(checkpoint=self.encoder_name, config=config, verbose=0)
        
        try:
            index_path = indexer.index(
                name=config.index_name, 
                collection=Collection.cast(str(collection_path)), 
                overwrite=True,
            )
            logging.info(f"✅ ColBERT index built and saved to {index_path}")
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            # Debug the collection format
            self._debug_collection_format(collection_path)
            raise

    def _verify_tsv_format(self, tsv_path: str):
        """Verify that the TSV format is correct for ColBERT."""
        logging.info(f"Verifying TSV format: {tsv_path}")
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("TSV file is empty")
        
        # Check header
        header = lines[0].strip().split('\t')
        logging.info(f"Header: {header}")
        
        # Check first few data lines
        for i in range(1, min(6, len(lines))):
            parts = lines[i].strip().split('\t')
            if len(parts) >= 2:
                pid = parts[0]
                # Verify PID format - should match line index
                try:
                    pid_int = int(pid)
                    expected_pid = i  # Line index should match PID
                    if pid_int == expected_pid:
                        logging.info(f"Line {i}: PID={pid} (✓ matches line index)")
                    else:
                        logging.error(f"Line {i}: PID={pid} doesn't match line index {i}")
                        raise ValueError(f"PID {pid} doesn't match expected line index {i}")
                except ValueError:
                    logging.error(f"Line {i}: Invalid PID format: {pid}")
                    raise ValueError(f"Invalid PID format at line {i}: {pid}")
            else:
                logging.warning(f"Line {i}: Insufficient columns: {parts}")
        
        logging.info(f"✅ TSV format verification passed. Total lines: {len(lines)}")

    def _debug_collection_format(self, collection_path: str):
        """Debug helper to understand collection format issues."""
        logging.info("=== DEBUGGING COLLECTION FORMAT ===")
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logging.info(f"Total lines: {len(lines)}")
        logging.info(f"First 5 lines:")
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split('\t')
            logging.info(f"  Line {i}: {len(parts)} parts - {parts}")
        
        # Check for problematic IDs
        problematic_lines = []
        for i, line in enumerate(lines[1:11], 1):  # Skip header, check next 10
            parts = line.strip().split('\t')
            if parts:
                pid = parts[0]
                try:
                    int(pid)
                except ValueError:
                    problematic_lines.append((i, pid))
        
        if problematic_lines:
            logging.error(f"Found {len(problematic_lines)} problematic PIDs:")
            for line_num, pid in problematic_lines:
                logging.error(f"  Line {line_num}: {pid}")

    def load_index(self):
        """
        Load the ColBERT index from the specified directory.
        :return: IndexLoader instance if successful, None otherwise.
        """
        logging.info(f"Loading index from {self.index_dir}...")
        try:
            index = IndexLoader(index_path=self.index_dir, use_gpu=self.device != "cpu")
            return index
        except Exception as e:
            logging.error(f"Failed to load ColBERT index: {e}")
            return None

    def get_original_id(self, sequential_id: int) -> str:
        """Convert sequential ID back to original ID."""
        _, sequential_to_original = self.load_id_mappings()
        return sequential_to_original.get(sequential_id, str(sequential_id))

    def get_sequential_id(self, original_id: str) -> int:
        """Convert original ID to sequential ID."""
        original_to_sequential, _ = self.load_id_mappings()
        return original_to_sequential.get(original_id, -1)