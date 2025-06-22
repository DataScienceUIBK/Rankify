import os
import requests
import zipfile
from tqdm import tqdm
from typing import List
from urllib.parse import urlparse

from rankify.utils.retrievers.colbert.colbert.infra import Run, RunConfig, ColBERTConfig
from rankify.utils.retrievers.colbert.colbert import Searcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT retriever implementation using late interaction for efficient passage ranking.
    
    Implements ColBERT (Contextualized Late Interaction over BERT) which enables
    scalable document retrieval through token-wise interactions and compressed representations.
    
    References:
        - Khattab, O. & Zaharia, M. (2020): ColBERT: Efficient and Effective Passage Search 
          via Contextualized Late Interaction over BERT. https://arxiv.org/abs/2004.12832
    """
    
    def __init__(self, model: str = "colbert-ir/colbertv2.0", index_type: str = "wiki", 
                 index_folder: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.index_type = index_type
        self.index_folder = index_folder
        self.tokenizer = SimpleTokenizer()
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Setup paths and download if needed
        if index_folder:
            self.index_path = index_folder
            self.passages_file = os.path.join(self.index_folder, "passages.tsv")
        else:
            self.index_path = os.path.join(self.index_manager.cache_dir, "index", "colbert")
            self._ensure_index_and_passages_downloaded()
            self.passages_file = self._get_passages_file_path()
        
        # Initialize searcher
        self.searcher = self._initialize_searcher()
        
        # Load passages
        self.passages = self._load_passages()
    
    def _initialize_searcher(self):
        """Initialize ColBERT searcher."""
        with Run().context(RunConfig(nranks=1, experiment="colbert")):
            # Determine the correct index root path
            if self.index_folder:
                index_root = self.index_folder
            else:
                index_root = os.path.join(self.index_path, self.index_type)
            
            config = ColBERTConfig(
                root=index_root,
                collection=self.passages_file
            )
            
            return Searcher(index=index_root, config=config)
    
    def _ensure_index_and_passages_downloaded(self):
        """Download and extract ColBERT index and passages if needed."""
        os.makedirs(self.index_path, exist_ok=True)
        
        if self.index_type not in self.index_manager.index_configs.get("colbert", {}):
            raise ValueError(f"Unsupported ColBERT index type: {self.index_type}")
        
        config = self.index_manager.index_configs["colbert"][self.index_type]
        
        # Check if index exists
        index_dir = os.path.join(self.index_path, self.index_type)
        if not os.path.exists(index_dir):
            urls = config.get("urls")
            if isinstance(urls, list):
                # Multi-part download
                for url in urls:
                    filename = self._extract_filename_from_url(url)
                    local_path = os.path.join(self.index_path, filename)
                    if not os.path.exists(local_path):
                        self._download_file(url, local_path)
                self._extract_multi_part_zip()
            else:
                # Single file download
                zip_path = os.path.join(self.index_path, "index.zip")
                if not os.path.exists(zip_path):
                    self._download_file(urls, zip_path)
                self._extract_zip_files()
        
        # Download passages if needed
        passages_url = config.get("passages_url")
        if passages_url:
            passages_filename = self._extract_filename_from_url(passages_url)
            passages_path = os.path.join(self.index_manager.cache_dir, passages_filename)
            if not os.path.exists(passages_path):
                self._download_file(passages_url, passages_path)
    
    def _get_passages_file_path(self):
        """Get path to passages file."""
        if self.index_type not in self.index_manager.index_configs.get("colbert", {}):
            raise ValueError(f"Unsupported ColBERT index type: {self.index_type}")
        
        config = self.index_manager.index_configs["colbert"][self.index_type]
        passages_url = config.get("passages_url")
        if passages_url:
            passages_filename = self._extract_filename_from_url(passages_url)
            return os.path.join(self.index_manager.cache_dir, passages_filename)
        return None
    
    def _extract_filename_from_url(self, url):
        """Extract filename from URL, ignoring query parameters."""
        parsed_url = urlparse(url)
        return os.path.basename(parsed_url.path).split('?')[0]
    
    def _download_file(self, url: str, save_path: str):
        """Download file from URL with progress bar."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), 
                            desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)
    
    def _extract_zip_files(self):
        """Extract ZIP files in the index folder."""
        zip_files = [f for f in os.listdir(self.index_path) if f.endswith(".zip")]
        
        for zip_file in zip_files:
            zip_path = os.path.join(self.index_path, zip_file)
            
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.index_path)
            
            os.remove(zip_path)
    
    def _extract_multi_part_zip(self):
        """Extract multi-part ZIP files by combining and decompressing them."""
        # Find all parts (handle both .001, .002 format and other naming patterns)
        all_files = os.listdir(self.index_path)
        zip_parts = []
        
        # Look for wiki.zip.001, wiki.zip.002, etc.
        for f in all_files:
            if f.startswith("wiki.zip.") and f.split('.')[-1].isdigit():
                zip_parts.append(f)
        
        if not zip_parts:
            print("No multi-part ZIP files found")
            return
        
        # Sort by part number
        zip_parts.sort(key=lambda x: int(x.split('.')[-1]))
        
        combined_zip_path = os.path.join(self.index_path, "combined.zip")
        
        print(f"Combining {len(zip_parts)} parts into {combined_zip_path}...")
        
        # Combine all parts
        with open(combined_zip_path, "wb") as combined:
            for part in zip_parts:
                part_path = os.path.join(self.index_path, part)
                print(f"Adding {part} to combined archive...")
                with open(part_path, "rb") as part_file:
                    # Use shutil.copyfileobj for better memory efficiency
                    import shutil
                    shutil.copyfileobj(part_file, combined)
        
        # Extract the combined ZIP
        print(f"Extracting combined ZIP file: {combined_zip_path}...")
        try:
            with zipfile.ZipFile(combined_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.index_path)
            print("Extraction completed successfully")
        except zipfile.BadZipFile as e:
            print(f"Error: Combined file is not a valid ZIP: {e}")
            raise
        finally:
            # Clean up: remove combined file and parts
            if os.path.exists(combined_zip_path):
                os.remove(combined_zip_path)
            for part in zip_parts:
                part_path = os.path.join(self.index_path, part)
                if os.path.exists(part_path):
                    os.remove(part_path)
    
    def _load_passages(self):
        """Load passages from TSV file."""
        if not self.passages_file or not os.path.exists(self.passages_file):
            print("Warning: Passages file not found, using empty passages dict")
            return {}
        
        passages = {}
        print(f"Loading passages from {self.passages_file}...")
        
        with open(self.passages_file, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        pid, text, title = parts[0], parts[1], parts[2]
                        passages[pid] = {"text": text, "title": title}
                    else:
                        print(f"Warning: Malformed line {line_num} in passages file")
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(passages)} passages.")
        return passages
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using ColBERT."""
        print(f"Retrieving {len(documents)} documents with ColBERT...")
        
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            query = document.question.question
            
            try:
                # ColBERT search returns (document_ids, ranks, scores)
                results = self.searcher.search(query, k=self.n_docs)
                document_ids, ranks, scores = results
                
                contexts = []
                for pid, rank, score in zip(document_ids, ranks, scores):
                    try:
                        context = self._create_context_from_result(str(pid), score, document)
                        if context:
                            contexts.append(context)
                    except Exception as e:
                        print(f"Error processing result {pid}: {e}")
                
                document.contexts = contexts
                
            except Exception as e:
                print(f"Error searching for document {i}: {e}")
                document.contexts = []
        
        return documents
    
    def _create_context_from_result(self, pid: str, score: float, document: Document) -> Context:
        """Create Context object from ColBERT search result."""
        if pid in self.passages:
            passage = self.passages[pid]
            return Context(
                id=pid,
                title=passage["title"],
                text=passage["text"],
                score=float(score),
                has_answer=has_answers(passage["text"], document.answers.answers, self.tokenizer)
            )
        else:
            print(f"Warning: Passage {pid} not found in passages dict")
            return Context(
                id=pid,
                title="Passage not found",
                text=f"Passage {pid} not found in corpus",
                score=float(score),
                has_answer=False
            )