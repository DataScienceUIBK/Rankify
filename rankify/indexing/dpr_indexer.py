import shutil
import subprocess
import logging

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

    def build_index(self):
        """
        Build the DPR dense index using FAISS.
        Steps:
        1. Converts corpus to Pyserini JSONL format.
        2. Runs DPR indexing command using the HuggingFace encoder.
        """
        corpus_path = self._save_dense_corpus()

        temp_corpus_dir = self.output_dir / "temp_corpus"

        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)

        temp_corpus_dir.mkdir(parents=True)

        # Move corpus into temporary directory
        dense_file = temp_corpus_dir / "corpus.jsonl"
        corpus_path.rename(dense_file)

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

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

        self._save_title_map()

        if dense_file.exists():
            dest_file = self.index_dir / "corpus.jsonl"
            if dest_file.exists():
                dest_file.unlink()
            shutil.move(str(dense_file), str(dest_file))

        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)

        logging.info(f"Dense indexing complete. Index stored at {self.index_dir}")

    def load_index(self):
        """
        Load the DPR index from the specified directory.

        This method checks if the index directory exists and is not empty,
        and raises an error if the index is locked or not found.
        """
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist or is empty.")
        logging.info(f"Index loaded from {self.index_dir}")