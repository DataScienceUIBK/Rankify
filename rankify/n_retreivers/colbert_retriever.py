import os
import json
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
    ColBERT retriever with backward compatibility for prebuilt indices.
    
    Supports two modes:
    1. Prebuilt indices (wiki, msmarco) - Original format with passages.tsv
    2. Custom indices - New format with collection.tsv + ID mappings
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
        
        # Initialize flag (will be properly set in _setup_paths)
        self.is_custom_index = False
        
        # Setup paths and determine index type
        self._setup_paths()
        
        # Load ID mappings (only for custom indices)
        self.id_mappings = self._load_id_mappings()
        
        # Initialize searcher
        self.searcher = self._initialize_searcher()
        
        # Load passages
        self.passages = self._load_passages()
    
    def _setup_paths(self):
        """Setup file paths based on whether it's a custom or prebuilt index."""
        if self.index_folder:
            # Check if it's actually a custom index by looking for custom files
            collection_file = os.path.join(self.index_folder, "collection.tsv")
            id_mapping_file = os.path.join(self.index_folder, "id_mapping.json")
            
            if os.path.exists(collection_file) and os.path.exists(id_mapping_file):
                # True custom index
                self.is_custom_index = True
                self.index_path = self.index_folder
                self.passages_file = os.path.join(self.index_folder, "passages.tsv")
                self.collection_file = collection_file
                self.id_mapping_file = id_mapping_file
                print(f"Using custom index: {self.index_folder}")
            else:
                # Prebuilt index with index_folder specified
                self.is_custom_index = False
                self.index_path = self.index_folder
                passages_file = os.path.join(self.index_folder, "passages.tsv")
                if os.path.exists(passages_file):
                    self.passages_file = passages_file
                else:
                    # Try to find passages file in parent directories
                    self._ensure_index_and_passages_downloaded()
                    self.passages_file = self._get_passages_file_path()
                self.collection_file = self.passages_file  # Same file for prebuilt indices
                self.id_mapping_file = None
                print(f"Using prebuilt index at: {self.index_folder}")
        else:
            # Standard prebuilt index
            self.is_custom_index = False
            self.index_path = os.path.join(self.index_manager.cache_dir, "index", "colbert")
            self._ensure_index_and_passages_downloaded()
            self.passages_file = self._get_passages_file_path()
            self.collection_file = self.passages_file  # Same file for prebuilt indices
            self.id_mapping_file = None
            print(f"Using prebuilt index: {self.index_type}")
    
    def _load_id_mappings(self):
        """Load ID mappings for custom indices only."""
        if not self.is_custom_index or not self.id_mapping_file:
            return None
            
        if not os.path.exists(self.id_mapping_file):
            print(f"Warning: ID mapping file not found: {self.id_mapping_file}")
            return None
        
        try:
            with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            # Convert string keys back to integers for sequential_to_original
            sequential_to_original = {int(k): v for k, v in mappings.get("sequential_to_original", {}).items()}
            original_to_sequential = mappings.get("original_to_sequential", {})
            
            print(f"✅ Loaded ID mappings: {len(original_to_sequential)} entries")
            return {
                "sequential_to_original": sequential_to_original,
                "original_to_sequential": original_to_sequential
            }
            
        except Exception as e:
            print(f"Warning: Error loading ID mappings: {e}")
            return None
    
    def _initialize_searcher(self):
        """Initialize ColBERT searcher with backward compatibility."""
        with Run().context(RunConfig(nranks=1, experiment="colbert")):
            
            if self.is_custom_index:
                # Custom index - use our format
                index_root = self.index_folder
                collection_path = self.collection_file
                
                # Verify collection file exists
                if not os.path.exists(collection_path):
                    raise FileNotFoundError(f"ColBERT collection file not found: {collection_path}")
                    
                print(f"Initializing custom ColBERT searcher...")
                print(f"  Index root: {index_root}")
                print(f"  Collection: {collection_path}")
                
            else:
                # Prebuilt index - use original format
                if self.index_folder:
                    index_root = self.index_folder
                else:
                    index_root = os.path.join(self.index_path, self.index_type)
                collection_path = self.passages_file
                
                # Verify passages file exists (this is what prebuilt indices use)
                if not os.path.exists(collection_path):
                    raise FileNotFoundError(f"Passages file not found: {collection_path}")
                
                print(f"Initializing prebuilt ColBERT searcher...")
                print(f"  Index root: {index_root}")
                print(f"  Collection: {collection_path}")
            
            config = ColBERTConfig(
                root=index_root,
                collection=collection_path
            )
            
            return Searcher(index=index_root, config=config)
    
    def _ensure_index_and_passages_downloaded(self):
        """Download and extract ColBERT index and passages if needed (prebuilt indices only)."""
        if self.is_custom_index or self.index_folder:
            return  # Skip for custom indices or when index_folder is provided
            
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
        """Get path to passages file for prebuilt indices."""
        if self.index_folder and not self.is_custom_index:
            # Direct path provided for prebuilt index
            passages_path = os.path.join(self.index_folder, "passages.tsv")
            if os.path.exists(passages_path):
                return passages_path
        
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
        """Load passages with format detection."""
        if not self.passages_file or not os.path.exists(self.passages_file):
            print("Warning: Passages file not found, using empty passages dict")
            return {}
        
        passages = {}
        print(f"Loading passages from {self.passages_file}...")
        
        # Detect if the first line has integer or string IDs
        with open(self.passages_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning
            
            # Check if header exists
            has_header = first_line.startswith('id\t') or first_line.lower().startswith('id\t')
            
            if has_header:
                next(f)  # Skip header
            
            # Process lines
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pid = parts[0]  # Keep as string (works for both int and string IDs)
                        text = parts[1] if len(parts) > 1 else ""
                        title = parts[2] if len(parts) > 2 else ""
                        
                        passages[pid] = {"text": text, "title": title}
                    else:
                        print(f"Warning: Malformed line {line_num} in passages file")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(passages)} passages")
        return passages
    
    def _convert_result_id(self, result_id: int) -> str:
        """Convert ColBERT result ID to the appropriate format."""
        if self.is_custom_index and self.id_mappings:
            # Custom index: convert sequential ID back to original string ID
            return self.id_mappings["sequential_to_original"].get(result_id, str(result_id))
        else:
            # Prebuilt index: IDs are already in correct format
            return str(result_id)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using ColBERT."""
        print(f"Retrieving {len(documents)} documents with ColBERT...")
        print(f"Mode: {'Custom index' if self.is_custom_index else f'Prebuilt index ({self.index_type})'}")
        
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            query = document.question.question
            
            try:
                # ColBERT search returns (document_ids, ranks, scores)
                results = self.searcher.search(query, k=self.n_docs)
                document_ids, ranks, scores = results
                
                contexts = []
                for result_id, rank, score in zip(document_ids, ranks, scores):
                    try:
                        context = self._create_context_from_result(result_id, score, document)
                        if context:
                            contexts.append(context)
                    except Exception as e:
                        print(f"Error processing result {result_id}: {e}")
                
                document.contexts = contexts
                
            except Exception as e:
                print(f"Error searching for document {i}: {e}")
                document.contexts = []
        
        return documents
    
    def _create_context_from_result(self, result_id: int, score: float, document: Document) -> Context:
        """Create Context object from ColBERT search result."""
        # Convert result ID to appropriate format
        passage_id = self._convert_result_id(result_id)
        
        # Find passage by ID
        if passage_id in self.passages:
            passage = self.passages[passage_id]
            return Context(
                id=passage_id,
                title=passage["title"],
                text=passage["text"],
                score=float(score),
                has_answer=has_answers(passage["text"], document.answers.answers, self.tokenizer)
            )
        else:
            print(f"Warning: Passage {passage_id} (result_id: {result_id}) not found in passages dict")
            return Context(
                id=passage_id,
                title="Passage not found",
                text=f"Passage {passage_id} not found in corpus",
                score=float(score),
                has_answer=False
            )