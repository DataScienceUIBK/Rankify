import shutil
import pickle
import logging

import numpy as np

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_contriever_embedding_chunks, to_pyserini_jsonl_dense
from rankify.utils.retrievers.contriever.index import Indexer

logging.basicConfig(level=logging.INFO)

class ContrieverIndexer(BaseIndexer):
    """
    Contriever Indexer that builds dense FAISS-based indexes using precomputed embeddings.

    Args:
        corpus_path (str): Path to the corpus file (.jsonl).
        encoder_name (str): HuggingFace model name for Contriever.
        output_dir (str): Directory to save the index.
        chunk_size (int): Number of lines per chunk during processing.
        threads (int): Unused here (kept for consistency).
        index_type (str): Type of index (e.g., "wiki").
        batch_size (int): Batch size for encoding.
        device (str): Device to use (i.e. "cuda" or "cpu").
    """

    def __init__(self,
                 corpus_path,
                 encoder_name="facebook/contriever",
                 output_dir="rankify_indices",
                 chunk_size=100,
                 threads=32,
                 index_type="wiki",
                 batch_size=1000000,
                 device="cuda"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type)
        self.encoder_name = encoder_name
        self.device = device
        self.batch_size = batch_size
        self.index_dir = self.output_dir / f"contriever_index_{index_type}"

    def _save_dense_embeddings(self):
        """
        Convert the corpus into Contriever embeddings stored in shard .pkl files.
        """
        return to_contriever_embedding_chunks(
            self.corpus_path,
            self.index_dir,
            self.chunk_size,
            self.encoder_name,
            self.batch_size,
            self.device
        )

    def _save_corpus(self):
        """
        Convert the corpus to dense-compatible Pyserini format.
        """
        return to_pyserini_jsonl_dense(self.corpus_path, self.index_dir, self.chunk_size, self.threads)

    def build_index(self):
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        self._save_corpus()

        embedding_files = self._save_dense_embeddings()  # returns list of .pkl shards

        index = Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)

        self._index_encoded_data(index, embedding_files, self.batch_size)

        index.serialize(self.index_dir)

        self._save_title_map()

        logging.info(f"Contriever index stored at {self.index_dir}")

    def _index_encoded_data(self, index, embedding_files, indexing_batch_size=1000000):
        """
        Loads and indexes **encoded passage embeddings** into FAISS.

        Args:
            index (Indexer): The FAISS **index** to store the embeddings.
            embedding_files (List[str]): List of **files containing precomputed embeddings**.
            indexing_batch_size (int, optional): **Batch size for indexing** (default: `1000000`).
        """
        all_ids = []
        all_embeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            all_embeddings = np.vstack((all_embeddings, embeddings)) if all_embeddings.size else embeddings
            all_ids.extend(ids)
            while all_embeddings.shape[0] > indexing_batch_size:
                all_embeddings, all_ids = self.add_embeddings(index, all_embeddings, all_ids, indexing_batch_size)

        while all_embeddings.shape[0] > 0:
            all_embeddings, all_ids = self.add_embeddings(index, all_embeddings, all_ids, indexing_batch_size)

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        """
        Adds a batch of embeddings to the index.
        Args:
            index (Indexer): The FAISS index to store the embeddings.
            embeddings (np.ndarray): Array of embeddings to add.
            ids (list): List of IDs corresponding to the embeddings.
            indexing_batch_size (int): Batch size for indexing.
        """
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def load_index(self):
        """
        Loads the Contriever index from the specified directory.
        :return:
        """
        index = Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)

        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} is missing or empty.")

        index.deserialize_from(self.index_dir)

        logging.info(f"Loaded index from {self.index_dir}")
        return index
