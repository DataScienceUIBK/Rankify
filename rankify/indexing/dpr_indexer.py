import shutil
import subprocess
import logging
import json

import torch

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_pyserini_jsonl_dense

logging.basicConfig(level=logging.INFO)

class DPRIndexer(BaseIndexer):
    """
    DPR Indexer that builds dense FAISS-based indexes using Pyserini.

    Supports indexing using models like DPR, ANCE, or BPR via HuggingFace.

    Args:
        corpus_path (str): Path to the corpus file.
        encoder_name (str): HuggingFace model name for the DPR encoder.
        output_dir (str): Directory to save the index.
        chunk_size (int): Size of chunks to process the corpus.
        threads (int): Number of threads to use for processing.
        index_type (str): Type of index to build (default: "wiki").
        batch_size (int): Batch size for encoding passages.
        device (str): Device to use for encoding ("cpu" or "cuda").
    """

    def __init__(self,
                 corpus_path,
                 encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
                 output_dir="rankify_indices",
                 chunk_size=100,
                 threads=32,
                 index_type="wiki",
                 retriever_name="dpr",
                 batch_size=16,
                 device="cuda"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type)
        self.encoder_name = encoder_name
        self.index_dir = self.output_dir / f"dpr_index_{index_type}"
        self.device = device
        self.batch_size = batch_size

    def _save_dense_corpus(self):
        """
        Convert the corpus to dense-compatible Pyserini format.
        Each passage must have a `text` and optionally a `title`.
        """
        return to_pyserini_jsonl_dense(self.corpus_path, self.output_dir, self.chunk_size, self.threads)

    def _create_id_mapping(self, corpus_file):
        """
        Create mapping between FAISS sequential IDs and original document IDs.
        
        Args:
            corpus_file (Path): Path to the corpus JSONL file
            
        Returns:
            dict: Mapping from FAISS index (0, 1, 2, ...) to original doc IDs
        """
        logging.info("Creating ID mapping for FAISS index...")
        
        faiss_to_docid = {}
        docid_to_faiss = {}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for faiss_idx, line in enumerate(f):
                doc = json.loads(line.strip())
                original_docid = doc.get('docid') or doc.get('id', str(faiss_idx))
                
                faiss_to_docid[faiss_idx] = str(original_docid)
                docid_to_faiss[str(original_docid)] = faiss_idx
        
        logging.info(f"Created ID mapping for {len(faiss_to_docid)} documents")
        return faiss_to_docid, docid_to_faiss

    def _save_id_mapping(self, faiss_to_docid, docid_to_faiss):
        """
        Save ID mappings to files.
        
        Args:
            faiss_to_docid (dict): Mapping from FAISS index to original doc ID
            docid_to_faiss (dict): Mapping from original doc ID to FAISS index
        """
        # Save FAISS index to original docid mapping
        faiss_to_docid_file = self.index_dir / "faiss_to_docid.json"
        with open(faiss_to_docid_file, 'w', encoding='utf-8') as f:
            json.dump(faiss_to_docid, f, indent=2)
        
        # Save original docid to FAISS index mapping  
        docid_to_faiss_file = self.index_dir / "docid_to_faiss.json"
        with open(docid_to_faiss_file, 'w', encoding='utf-8') as f:
            json.dump(docid_to_faiss, f, indent=2)
            
        logging.info(f"ID mappings saved to {self.index_dir}")

    def _save_corpus_with_metadata(self, corpus_file):
        """
        Save corpus with metadata for easy retrieval.
        
        Args:
            corpus_file (Path): Path to the corpus JSONL file
        """
        logging.info("Saving corpus metadata...")
        
        corpus_metadata = {}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                docid = doc.get('docid') or doc.get('id')
                
                # Extract title and contents
                contents = doc.get('contents', '')
                title = doc.get('title', '')
                
                # If no explicit title, try to extract from contents
                if not title and contents:
                    lines = contents.split('\n')
                    title = lines[0] if lines else "No Title"
                
                corpus_metadata[str(docid)] = {
                    'title': title,
                    'contents': contents,
                    'text': doc.get('text', contents)  # Some formats use 'text' instead of 'contents'
                }
        
        # Save corpus metadata
        corpus_metadata_file = self.index_dir / "corpus_metadata.json"
        with open(corpus_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_metadata, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Corpus metadata saved to {corpus_metadata_file}")

    def build_index(self):
        """
        Build the DPR dense index using FAISS.
        Steps:
        1. Converts corpus to Pyserini JSONL format.
        2. Creates ID mappings between FAISS and original document IDs.
        3. Runs DPR indexing command using the HuggingFace encoder.
        4. Saves all necessary mapping files.
        """
        corpus_path = self._save_dense_corpus()

        temp_corpus_dir = self.output_dir / "temp_corpus"

        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)

        temp_corpus_dir.mkdir(parents=True)

        # Move corpus into temporary directory
        dense_file = temp_corpus_dir / "corpus.jsonl"
        corpus_path.rename(dense_file)

        # Create ID mappings before indexing
        faiss_to_docid, docid_to_faiss = self._create_id_mapping(dense_file)

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)
        
        print("Start indexing ............................")
        cmd = [
            "python", "-m", "pyserini.encode",

            "input",
            "--corpus", str(dense_file),

            "output",
            "--embeddings", str(self.index_dir),

            "encoder",
            "--encoder", self.encoder_name,
            "--batch-size", str(self.batch_size),
            "--max-length", "512",
        ]

        if self.device == "cuda":
            if torch.cuda.is_available():
                cmd.extend(["--device", "cuda"])
            else:
                logging.warning("CUDA is not available. Using CPU for encoding.")
                cmd.extend(["--device", "cpu"])
        else:
            cmd.extend(["--device", self.device])

        logging.info(f"Encoding dense vectors with DPR: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Now index the dense vectors into FAISS
        index_cmd = [
            "python", "-m", "pyserini.index.faiss",
            "--input", str(self.index_dir),
            "--output", str(self.index_dir),
        ]

        logging.info(f"Building FAISS index: {' '.join(index_cmd)}")
        subprocess.run(index_cmd, check=True)

        # Save ID mappings
        self._save_id_mapping(faiss_to_docid, docid_to_faiss)
        
        # Save corpus metadata for retrieval
        self._save_corpus_with_metadata(dense_file)

        # Save the original title map (inherited from base class)
        self._save_title_map()

        # Move corpus.jsonl to final location
        if dense_file.exists():
            dest_file = self.index_dir / "corpus.jsonl"
            if dest_file.exists():
                dest_file.unlink()
            shutil.move(str(dense_file), str(dest_file))

        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)

        logging.info(f"Dense indexing complete. Index stored at {self.index_dir}")
        logging.info(f"Index files created:")
        logging.info(f"  - FAISS index files")
        logging.info(f"  - corpus.jsonl")
        logging.info(f"  - faiss_to_docid.json")
        logging.info(f"  - docid_to_faiss.json") 
        logging.info(f"  - corpus_metadata.json")

    def load_index(self):
        """
        Load the DPR index from the specified directory.

        This method checks if the index directory exists and is not empty,
        and raises an error if the index is locked or not found.
        """
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist or is empty.")
        
        # Check for required mapping files
        required_files = [
            "faiss_to_docid.json",
            "docid_to_faiss.json", 
            "corpus_metadata.json"
        ]
        
        for file_name in required_files:
            file_path = self.index_dir / file_name
            if not file_path.exists():
                logging.warning(f"Missing mapping file: {file_path}")
        
        logging.info(f"Index loaded from {self.index_dir}")