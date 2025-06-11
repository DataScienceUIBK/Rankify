import shutil
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_tsv
from tqdm import tqdm
import faiss
import pickle
import h5py
import logging
from rankify.retrievers.utils import generate_chunks, bge_embed_chunk

logging.basicConfig(level=logging.INFO)

class BGEIndexer(BaseIndexer):
    """
    BGE Indexer that builds dense FAISS-based indexes using precomputed embeddings.

    Args:
        corpus_path (str): Path to the corpus file (.jsonl).
        encoder_name (str): HuggingFace model name for BGE.
        output_dir (str): Directory to save the index.
        chunk_size (int): Number of lines per chunk during processing.
        threads (int): Unused here (kept for consistency).
        index_type (str): Type of index (e.g., "wiki").
        batch_size (int): Batch size for encoding.
        retriever_name (str): Name of the retriever (default: "bge").
        device (str): Device to use (i.e. "cuda" or "cpu").
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
            self.index_dir = index_dir

    def build_faiss_index(self):
        embeddings_path = self.index_dir / "bge_embeddings.h5"
        logging.info(f"Loading embeddings from {embeddings_path}...")
        with h5py.File(embeddings_path, "r") as f:
            embeddings = f["embeddings"][:].astype(np.float32)

        embedding_dim = embeddings.shape[1]
        logging.info(f"Building FAISS index with embedding dimension {embedding_dim}...")
        index = faiss.IndexFlatIP(embedding_dim)

        chunk_size = 50000  # Add embeddings in batches for memory efficiency
        for start in tqdm(range(0, embeddings.shape[0], chunk_size), desc="Adding embeddings to FAISS index"):
            end = min(start + chunk_size, embeddings.shape[0])
            index.add(embeddings[start:end])

        index_path = self.index_dir / "faiss_index.bin"
        logging.info(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, str(index_path))



    def _merge_embedding_chunks(self):
        import pickle
        import numpy as np
        import h5py
        from tqdm import tqdm
        import logging

        chunk_files = list(self.index_dir.glob("embeddings_chunk_*.pkl"))
        logging.info(f"Found {len(chunk_files)} embedding chunk files to merge.")

        all_embeddings = []
        all_ids = []
        dims = []

        for file in tqdm(chunk_files, desc="Loading embedding chunks"):
            with open(file, "rb") as fin:
                chunk = pickle.load(fin)
            if not isinstance(chunk, list):
                raise ValueError(f"Unexpected format in {file.name}: {type(chunk)}")

            for entry in chunk:
                embedding = entry.get("embedding")
                doc_id = entry.get("id")
                if embedding is None or doc_id is None:
                    raise ValueError(f"Missing keys in chunk entry from {file.name}")
                all_embeddings.append(embedding)
                all_ids.append(doc_id)
                dims.append(len(embedding))

        if len(set(dims)) != 1:
            raise ValueError(f"Embedding dimension mismatch in chunks: found dimensions {set(dims)}")

        all_embeddings = np.vstack(all_embeddings).astype(np.float32)

        merged_embedding_path = self.index_dir / "bge_embeddings.h5"
        with h5py.File(merged_embedding_path, "w") as hf:
            hf.create_dataset("embeddings", data=all_embeddings)

        merged_ids_path = self.index_dir / "bge_doc_ids.pkl"
        with open(merged_ids_path, "wb") as f:
            pickle.dump(all_ids, f)

        #remove chunk files after merging
        for file in chunk_files:
            file.unlink()

        logging.info(f"Merged embeddings saved to {merged_embedding_path}")
        logging.info(f"Merged IDs saved to {merged_ids_path}")

        return merged_embedding_path, merged_ids_path



    def _save_dense_embeddings(self):
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        model = AutoModel.from_pretrained(self.encoder_name).to(self.device)
        model.eval()

        chunk_dir = self.index_dir
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for chunk_lines, start_idx in tqdm(generate_chunks(f, self.chunk_size), desc="Embedding chunks"):
                chunk_data = bge_embed_chunk(chunk_lines, tokenizer, model, self.device)

                chunk_path = chunk_dir / f"embeddings_chunk_{start_idx}.pkl"
                with open(chunk_path, "wb") as fout:
                    pickle.dump(chunk_data, fout)

                chunk_paths.append(chunk_path)
        logging.info(f"Saved {len(chunk_paths)} embedding chunk files.")
        return chunk_paths


    def _save_corpus(self):
        """
        Convert the corpus to TSV format for indexing.
        """
        return to_tsv(self.corpus_path, self.index_dir, self.chunk_size, self.threads)

    def build_index(self):
        """
        Build the BGE index from the corpus.
        :return:
        """
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        self._save_corpus()
        self._save_dense_embeddings()
        self._merge_embedding_chunks()
        #self._save_title_map()

        self.build_faiss_index()

        logging.info(f"BGE index stored at {self.index_dir}")

    def load_index(self) -> tuple:
        """
        Load the FAISS index and document IDs from the index directory.
        :return: tuple of (faiss_index, doc_ids)
        """
        faiss_index_path = self.index_dir / "faiss_index.bin"
        merged_ids_path = self.index_dir / "bge_doc_ids.pkl"
        #embeddings_path = self.index_dir / "bge_embeddings.h5"

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

