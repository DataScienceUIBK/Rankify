import os
import requests
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Union
import pickle
import glob
import tarfile
import zipfile
from urllib.parse import urlparse

from rankify.utils.retrievers.contriever.index import Indexer
from rankify.utils.retrievers.contriever.normalize_text import normalize
from rankify.utils.retrievers.contriever.data import load_passages
from rankify.utils.retrievers.contriever.contriever import load_retriever
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ContrieverRetriever(BaseRetriever):
    """
    FIXED Contriever retriever implementation with proper string ID handling.
    
    Key fixes:
    - Handles both string and integer document IDs
    - Robust ID mapping and lookup
    - Better error handling for missing documents
    """
    
    def __init__(self, model: str = "facebook/contriever-msmarco", index_type: str = "wiki", 
                 index_folder: str = None, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model
        self.index_type = index_type
        self.index_folder = index_folder
        self.device = device
        self.tokenizer_simple = SimpleTokenizer()
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Setup paths
        if index_folder:
            self.index_path = index_folder
            self.passage_path = os.path.join(self.index_folder, "passages.tsv")
        else:
            self.embeddings_dir = os.path.join(self.index_manager.cache_dir, "index", "contriever_embeddings")
            self.index_path = os.path.join(self.embeddings_dir, self.index_type)
            self._ensure_index_and_passages_downloaded()
            self.passage_path = self._get_passage_path()
        
        # Load components
        self.index = self._load_index()
        print(f"Loading passages from: {self.passage_path}")
        self.passages = load_passages(self.passage_path)
        
        # FIXED: Handle both string and integer IDs properly
        self.passage_id_map = self._create_passage_id_map(self.passages)
        
        # Debug: Print some sample IDs to understand the format
        sample_ids = list(self.passage_id_map.keys())[:5]
        print(f"Sample passage IDs: {sample_ids}")
        print(f"Total passages loaded: {len(self.passage_id_map)}")
        
        # Load model
        self.model, self.tokenizer, _ = load_retriever(self.model_path)
        self.model = self.model.to(self.device).eval()
    
    def _create_passage_id_map(self, passages: List[Dict]) -> Dict[Union[str, int], Dict]:
        """
        FIXED: Create passage ID mapping that handles both string and integer IDs.
        
        Args:
            passages: List of passage dictionaries
            
        Returns:
            Dictionary mapping IDs (strings or integers) to passage data
        """
        passage_map = {}
        
        for passage in passages:
            original_id = passage["id"]
            
            # Try to convert to int, but keep as string if it fails
            try:
                # First try direct integer conversion
                int_id = int(original_id)
                passage_map[int_id] = passage
            except (ValueError, TypeError):
                # If that fails, keep as string
                passage_map[str(original_id)] = passage
                
                # Also try extracting numbers from string IDs for compatibility
                try:
                    # Extract numeric part if it's a string like "doc123" or similar
                    numeric_part = ''.join(filter(str.isdigit, str(original_id)))
                    if numeric_part:
                        int_id = int(numeric_part)
                        # Store both string and integer versions for flexibility
                        if int_id not in passage_map:
                            passage_map[int_id] = passage
                except:
                    pass  # If extraction fails, just use string
        
        return passage_map
    
    def _initialize_searcher(self):
        """Initialize searcher - not needed for Contriever as it uses direct FAISS."""
        return None
    
    def _ensure_index_and_passages_downloaded(self):
        """Download and extract Contriever index and passages if needed."""
        if self.index_type not in self.index_manager.index_configs.get("contriever", {}):
            raise ValueError(f"Unsupported Contriever index type: {self.index_type}")
        
        config = self.index_manager.index_configs["contriever"][self.index_type]
        
        # Handle embeddings download and extraction
        embeddings_url = config.get("url")
        if embeddings_url and not os.path.exists(self.index_path):
            os.makedirs(self.index_path, exist_ok=True)
            
            archive_name = self._clean_filename(embeddings_url)
            archive_path = os.path.join(self.index_path, archive_name)
            
            print(f"Downloading embeddings for '{self.index_type}'...")
            self._download_file(embeddings_url, archive_path)
            
            # Extract based on file type
            if archive_name.endswith(".tar"):
                print(f"Extracting TAR archive for '{self.index_type}'...")
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(path=self.index_path)
            elif archive_name.endswith(".zip"):
                print(f"Extracting ZIP archive for '{self.index_type}'...")
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(self.index_path)
            else:
                raise ValueError(f"Unsupported archive format: {archive_name}")
            
            os.remove(archive_path)  # Clean up
        
        # Handle passages download
        passages_url = config.get("passages_url")
        if passages_url:
            passage_filename = self._clean_filename(passages_url)
            passage_path = os.path.join(self.index_manager.cache_dir, passage_filename)
            
            if not os.path.exists(passage_path):
                print(f"Downloading passages '{passage_filename}'...")
                self._download_file(passages_url, passage_path)
    
    def _get_passage_path(self):
        """Get path to passages file."""
        if self.index_type not in self.index_manager.index_configs.get("contriever", {}):
            raise ValueError(f"Unsupported Contriever index type: {self.index_type}")
        
        config = self.index_manager.index_configs["contriever"][self.index_type]
        passages_url = config.get("passages_url")
        if passages_url:
            passage_filename = self._clean_filename(passages_url)
            return os.path.join(self.index_manager.cache_dir, passage_filename)
        return None
    
    def _clean_filename(self, url):
        """Extract filename from URL, removing query parameters."""
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
    
    def _load_index(self) -> Indexer:
        """Load or build the FAISS index using Contriever utilities."""
        # Get vector size from config
        if self.index_type in self.index_manager.index_configs.get("contriever", {}):
            config = self.index_manager.index_configs["contriever"][self.index_type]
            vector_size = config.get("vector_size", 768)
        else:
            vector_size = 768
        
        index = Indexer(vector_sz=vector_size, n_subquantizers=0, n_bits=8)
        
        # Determine index folder
        if self.index_folder:
            index_folder = self.index_folder
        else:
            index_folder = self.index_path
            if self.index_type == "wiki":
                index_folder = os.path.join(index_folder, "wikipedia_embeddings")
        
        index_path = os.path.join(index_folder, "index.faiss")
        
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_folder}...")
            index.deserialize_from(index_folder)
        else:
            print(f"Building FAISS index from embeddings in {index_folder}...")
            # Look for both .pkl files and passages_XX files (Contriever format)
            embeddings_files = glob.glob(os.path.join(index_folder, "*.pkl"))
            if not embeddings_files:
                # Try Contriever's naming convention: passages_XX
                embeddings_files = glob.glob(os.path.join(index_folder, "passages_*"))
            
            embeddings_files = sorted(embeddings_files)
            
            if not embeddings_files:
                raise FileNotFoundError(f"No embedding files found in {index_folder}. "
                                      f"Looking for .pkl files or passages_* files.")
            
            print(f"Found {len(embeddings_files)} embedding files: {[os.path.basename(f) for f in embeddings_files[:5]]}...")
            self._index_encoded_data(index, embeddings_files)
            index.serialize(index_folder)
        
        return index
    
    def _debug_file_format(self, file_path):
        """Debug helper to understand file format."""
        try:
            with open(file_path, "rb") as f:
                first_bytes = f.read(100)
                print(f"First 100 bytes of {file_path}: {first_bytes}")
                
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    def _index_encoded_data(self, index, embedding_files, indexing_batch_size=1000000):
        """Load and index encoded passage embeddings into FAISS."""
        allids = []
        allembeddings = np.array([])
        
        # Debug first file to understand format
        if embedding_files:
            print(f"Debugging first file format...")
            self._debug_file_format(embedding_files[0])
        
        for i, file_path in enumerate(embedding_files):
            print(f"Loading embeddings from {file_path}")
            try:
                with open(file_path, "rb") as fin:
                    ids, embeddings = pickle.load(fin)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading pickle file {file_path}: {e}")
                print(f"Trying alternative loading method...")
                try:
                    # Try numpy format
                    data = np.load(file_path, allow_pickle=True)
                    if isinstance(data, tuple) and len(data) == 2:
                        ids, embeddings = data
                    else:
                        print(f"Unexpected data format in {file_path}")
                        continue
                except Exception as e2:
                    print(f"Failed to load {file_path} with alternative method: {e2}")
                    continue
            except Exception as e:
                print(f"Unexpected error loading {file_path}: {e}")
                continue
            
            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            
            # Process in batches to manage memory
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self._add_embeddings(index, allembeddings, allids, indexing_batch_size)
        
        # Process remaining embeddings
        while allembeddings.shape[0] > 0:
            allembeddings, allids = self._add_embeddings(index, allembeddings, allids, indexing_batch_size)
        
        print("Data indexing completed.")
    
    def _add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        """Add a batch of embeddings to the index."""
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids
    
    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed queries using the Contriever model with text normalization."""
        self.model.eval()
        embeddings = []
        batch_queries = []
        
        with torch.no_grad():
            for i, query in enumerate(queries):
                # Apply Contriever-specific preprocessing
                query = query.lower()
                query = normalize(query)
                batch_queries.append(query)
                
                if len(batch_queries) == self.batch_size or i == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_queries, 
                        return_tensors="pt", 
                        max_length=512, 
                        padding=True, 
                        truncation=True
                    )
                    
                    # Move to device
                    for k, v in encoded_batch.items():
                        encoded_batch[k] = v.to(self.device)
                    
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_queries = []
        
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.numpy()
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using Contriever."""
        print(f"Retrieving {len(documents)} documents with Contriever...")
        
        # Prepare queries (remove question marks as in original)
        queries = [doc.question.question.replace("?", "") for doc in documents]
        
        # Embed queries
        query_embeddings = self._embed_queries(queries)
        
        # Search using FAISS index
        top_ids_and_scores = self.index.search_knn(
            query_embeddings, 
            self.n_docs, 
            index_batch_size=self.batch_size
        )
        
        # Process results
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            top_ids, scores = top_ids_and_scores[i]
            contexts = []
            
            for doc_id, score in zip(top_ids, scores):
                try:
                    context = self._create_context_from_result(doc_id, score, document)
                    if context:
                        contexts.append(context)
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Error processing result {doc_id}: {e}")
                    continue
            
            document.contexts = contexts
        
        return documents
    
    def _create_context_from_result(self, doc_id, score: float, document: Document) -> Context:
        """
        FIXED: Create Context object from Contriever search result with robust ID handling.
        """
        # Try multiple strategies to find the passage
        passage = None
        original_doc_id = doc_id
        
        # Strategy 1: Direct lookup (string or int)
        if doc_id in self.passage_id_map:
            passage = self.passage_id_map[doc_id]
        
        # Strategy 2: Try as string if it's currently an int
        elif str(doc_id) in self.passage_id_map:
            passage = self.passage_id_map[str(doc_id)]
        
        # Strategy 3: Try as int if it's currently a string  
        elif isinstance(doc_id, str):
            try:
                int_doc_id = int(doc_id)
                if int_doc_id in self.passage_id_map:
                    passage = self.passage_id_map[int_doc_id]
            except ValueError:
                pass
        
        # Strategy 4: Handle "doc" prefix removal
        if passage is None:
            try:
                clean_id = str(doc_id).replace("doc", "")
                if clean_id in self.passage_id_map:
                    passage = self.passage_id_map[clean_id]
                elif clean_id.isdigit() and int(clean_id) in self.passage_id_map:
                    passage = self.passage_id_map[int(clean_id)]
            except (ValueError, AttributeError):
                pass
        
        # Strategy 5: Extract numeric part from string IDs
        if passage is None and isinstance(doc_id, str):
            try:
                numeric_part = ''.join(filter(str.isdigit, str(doc_id)))
                if numeric_part:
                    numeric_id = int(numeric_part)
                    if numeric_id in self.passage_id_map:
                        passage = self.passage_id_map[numeric_id]
            except ValueError:
                pass
        
        if passage is None:
            print(f"Warning: Document {original_doc_id} not found in passage map")
            print(f"Available ID types: {type(list(self.passage_id_map.keys())[0]) if self.passage_id_map else 'None'}")
            return None
        
        # Get text content (handle different field names)
        text_content = passage.get("contents", passage.get("text", ""))
        
        return Context(
            id=original_doc_id,  # Keep original ID format
            title=passage.get("title", ""),
            text=text_content,
            score=float(score),
            has_answer=has_answers(
                text_content, 
                document.answers.answers, 
                self.tokenizer_simple, 
                regex=False
            )
        )