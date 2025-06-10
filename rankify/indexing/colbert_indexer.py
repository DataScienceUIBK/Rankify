import shutil
import logging

from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_tsv
from rankify.utils.retrievers.colbert.colbert import Indexer
from rankify.utils.retrievers.colbert.colbert.data import Collection
from rankify.utils.retrievers.colbert.colbert.infra import ColBERTConfig
from rankify.utils.retrievers.colbert.colbert.search.index_loader import IndexLoader

logging.basicConfig(level=logging.INFO)

class ColBERTIndexer(BaseIndexer):
    """
    ColBERT Indexer for creating and loading ColBERT indices.
    This class handles the preparation of the corpus, building the index, and loading the index.
    It uses the ColBERT framework for efficient indexing and retrieval.
    Args:
        corpus_path (str): Path to the corpus file.
        encoder_name (str): Name of the ColBERT encoder model.
        output_dir (str): Directory where the index will be saved.
        chunk_size (int): Size of chunks to process the corpus.
        threads (int): Number of threads to use for processing.
        index_type (str): Type of index to create (e.g., "wiki").
        batch_size (int): Batch size for indexing.
        device (str): Device to use for indexing ("cuda" or "cpu").
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

    def _generate_colbert_passages_tsv(self) -> str:
        """
        Convert the corpus to a TSV format suitable for ColBERT indexing.
        Each passage must have an id, a `text` or `contents` and optionally a `title`.
        The output is a TSV file with the format:
        <doc_id>\t<text>\t<title>
        :return: Path to the converted corpus file in TSV format.
        """
        #corpus_path = to_pyserini_jsonl_dense(self.corpus_path, self.index_dir, self.chunk_size, self.threads)
        passages_path =  to_tsv(self.corpus_path, self.index_dir, self.chunk_size, self.threads)
        return passages_path


    def build_index(self):
        """
        Build the ColBERT index from the corpus.
        """
        logging.info("Start Building ColBERT index...")

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        collection_path = self._generate_colbert_passages_tsv()

        config = ColBERTConfig(
            root=str(self.output_dir),
            index_root=str(self.output_dir),
            index_name=f"colbert_index_{self.index_type}",
            index_path=str(self.index_dir),
            checkpoint=self.encoder_name,
            collection=str(collection_path),
            doc_maxlen=180,
            nbits=2,
            kmeans_niters=4,
            nranks=1,
            bsize=self.batch_size,
            index_bsize=self.batch_size,
        )

        indexer = Indexer(checkpoint=self.encoder_name, config=config, verbose=0)
        index_path = indexer.index(name=config.index_name, collection=Collection.cast(str(collection_path)), overwrite=True)

        logging.info(f"ColBERT index built and saved to {index_path}")

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


