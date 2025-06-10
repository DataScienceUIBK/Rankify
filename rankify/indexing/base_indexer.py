from abc import ABC, abstractmethod
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
import json

class BaseIndexer(ABC):
    """
    Abstract base class for indexers. This class provides a template for building and loading indices.
    """
    def __init__(self, corpus_path, output_dir="rankify_indices", chunk_size=512, threads=32, index_type="wiki",
                 retriever_name="base"):
        self.index_dir = None
        self.corpus_path = corpus_path
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.threads = threads
        self.doc_chunks = []
        self.index_type = index_type
        self.retriever_name = retriever_name

    @abstractmethod
    def build_index(self):
        """
            Build index from corpus. This method should be implemented in subclasses. 
        """
        pass

    @abstractmethod
    def load_index(self):
        """
            Load index from disk. This method should be implemented in subclasses. 
        """
        pass

    def cleanup_lock_file(self, index_dir=None):
        if index_dir is None:
            index_dir = self.output_dir / "bm25_index"
        lock_file = index_dir / "write.lock"
        if lock_file.exists():
            print(f"Remove Lock-File {lock_file}")
            lock_file.unlink()

    def _save_title_map(self):
        title_map = {}
        logging.info("Saving title map...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    doc_id = doc.get("id") or f"doc{i}"
                    title = doc.get("title") or doc.get("contents") or doc.get("text") or "No Title"
                    title_map[doc_id] = title[:100]
                except json.JSONDecodeError:
                    continue

        title_map_path = self.index_dir / "title_map.json"
        with open(title_map_path, "w", encoding="utf-8") as f:
            json.dump(title_map, f, ensure_ascii=False, indent=2)