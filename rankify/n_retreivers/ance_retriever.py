# rankify/retrievers/ance_retriever.py - FIXED VERSION following DenseRetriever pattern exactly
import json
import os
import requests
import numpy as np
import torch
import faiss
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from pyserini.search.faiss import FaissSearcher

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context

class ANCERetriever(BaseRetriever):
    """
    ANCE retriever implementation following the exact same pattern as DenseRetriever.
    
    Supports both prebuilt and custom indices using FaissSearcher like DPR.
    
    Args:
        index_type (str): Type of prebuilt index ("wiki", "msmarco"). 
        index_folder (str, optional): Path to custom ANCE index folder.
        encoder_name (str, optional): ANCE model name (auto-detected if not provided).
        method (str): Method variant ("ance", "ance-multi", "ance-msmarco").
        n_docs (int): Number of documents to retrieve per query.
        batch_size (int): Number of queries to process in a batch.
        threads (int): Number of parallel threads for processing.
        device (str): Device to use for encoding ("cpu" or "cuda").
    """
    
    def __init__(self, index_type: str = "wiki", index_folder: str = None,
                 encoder_name: str = None, device: str = "cuda", method: str = "ance-multi", **kwargs):
        super().__init__(**kwargs)
        
        self.method = method
        self.index_type = index_type
        self.index_folder = index_folder
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer_simple = SimpleTokenizer()
        self.batch_size = 128
        # Initialize index manager (following DenseRetriever pattern exactly)
        self.index_manager = IndexManager()
        
        # Get query encoder (following DenseRetriever logic exactly)
        if index_folder:
            # For custom indexes, get encoder from method and default index_type
            if method in self.index_manager.index_configs:
                # Use wiki as default if the provided index_type is a path
                default_index_type = "wiki" if (index_type.startswith("./") or 
                                               index_type.startswith("/") or 
                                               os.path.exists(index_type)) else index_type
                if default_index_type in self.index_manager.index_configs[method]:
                    self.query_encoder = self.index_manager.index_configs[method][default_index_type]["encoder"]
                else:
                    # Fallback to first available encoder for this method
                    available_types = list(self.index_manager.index_configs[method].keys())
                    self.query_encoder = self.index_manager.index_configs[method][available_types[0]]["encoder"]
                    print(f"Warning: Using default encoder {self.query_encoder} for custom index")
            else:
                raise ValueError(f"Unsupported method: {method}")
        else:
            # Get query encoder for standard indexes
            if method not in self.index_manager.index_configs:
                raise ValueError(f"Unsupported method: {method}")
            if index_type not in self.index_manager.index_configs[method]:
                raise ValueError(f"Unsupported index_type '{index_type}' for method '{method}'")
            self.query_encoder = self.index_manager.index_configs[method][index_type]["encoder"]
        
        # Initialize variables (following DenseRetriever pattern exactly)
        self.id_mapping = None
        self.corpus = {}
        
        # Handle custom vs prebuilt indexes (following DenseRetriever pattern exactly)
        if index_folder:
            self.index_path = index_folder
            self.id_mapping = self.index_manager.load_id_mapping(index_folder)
            self.corpus = self.index_manager.load_corpus(index_folder)
            print(f"Loaded corpus with {len(self.corpus)} documents")
            if self.id_mapping:
                print(f"Loaded ID mapping with {len(self.id_mapping)} entries")
        else:
            # FIXED: Use the actual method name (ance-multi) to get the correct prebuilt index
            self.index_path = self.index_manager.get_index_path(self.method, index_type)
            print(f"🔍 Method '{self.method}' + Index type '{index_type}' -> Index path: '{self.index_path}'")
        
        # Initialize searcher (following DenseRetriever pattern exactly)
        self.searcher = self._initialize_searcher()
        if self.searcher is None:
            raise RuntimeError("Failed to initialize searcher")
        
        print(f"✅ ANCE Retriever initialized")
        print(f"  📚 Index: {self.index_path}")
        print(f"  🏷️ Type: {index_type}")
        print(f"  🔧 Method: {self.method}")
        print(f"  🤖 Model: {self.query_encoder}")
        print(f"  📊 Corpus: {len(self.corpus)} documents")
        if self.id_mapping:
            print(f"  🗂️ Mappings: {len(self.id_mapping)} entries")
        print(f"  🖥️ Device: {self.device}")

    def _initialize_searcher(self):
        """
        Initialize FAISS searcher following DenseRetriever pattern exactly.
        This uses FaissSearcher for both prebuilt and custom indices.
        """
        try:
            # Case 1: Custom index folder provided
            if self.index_folder:
                print(f"Initializing FaissSearcher with custom index folder: {self.index_folder}")
                if not os.path.exists(self.index_folder):
                    raise ValueError(f"Custom index folder '{self.index_folder}' does not exist.")
                return FaissSearcher(self.index_folder, self.query_encoder)
            
            # Case 2: Standard indexes (prebuilt or downloaded)
            print(f"Index path: {self.index_path}")
            
            # Case 2a: URL - need to download first
            if self.index_path.startswith("http"):
                print(f"Downloading index from URL: {self.index_path}")
                local_dir = self.index_manager.download_and_extract_index(self.index_path)
                print(f"Downloaded index to: {local_dir}")
                return FaissSearcher(local_dir, self.query_encoder)
            
            # Case 2b: Local file path (already downloaded)
            elif (os.path.exists(self.index_path) or 
                  self.index_path.startswith('/') or 
                  self.index_path.startswith('./')):
                print(f"Using local index path: {self.index_path}")
                if not os.path.exists(self.index_path):
                    raise ValueError(f"Local index path '{self.index_path}' does not exist.")
                return FaissSearcher(self.index_path, self.query_encoder)
            
            # Case 2c: Prebuilt index name (like "wikipedia-dpr-100w.ance-multi")
            else:
                print(f"Using prebuilt index: {self.index_path}")
                # IMPORTANT: For ance-multi, this should be a prebuilt identifier like "wikipedia-dpr-100w.ance-multi"
                if self.method == "ance-multi" and self.index_path.startswith("/"):
                    raise ValueError(f"ERROR: Method 'ance-multi' should use prebuilt index, but got local path: {self.index_path}. "
                                   f"Check IndexManager configuration for method='{self.method}', index_type='{self.index_type}'")
                return FaissSearcher.from_prebuilt_index(self.index_path, self.query_encoder)
                
        except Exception as e:
            print(f"ERROR initializing searcher:")
            print(f"  - Method: {self.method}")
            print(f"  - Index type: {self.index_type}")
            print(f"  - Index folder: {self.index_folder}")
            print(f"  - Index path: {self.index_path}")
            print(f"  - Query encoder: {self.query_encoder}")
            print(f"  - Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using ANCE (following DenseRetriever pattern exactly)."""
        queries = [doc.question.question for doc in documents]
        qids = [str(i) for i in range(len(queries))]
        
        print(f"🔍 Retrieving {len(documents)} documents with {self.method}...")
        
        # Perform batch search (following DenseRetriever pattern exactly)
        batch_results = self._batch_search(queries, qids)
        
        # Process results (following DenseRetriever pattern exactly)
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            contexts = []
            hits = batch_results.get(str(i), [])
            
            for hit in hits:
                try:
                    context = self._create_context_from_hit(hit, document)
                    if context:
                        contexts.append(context)
                except Exception as e:
                    print(f"Error processing document ID {hit.docid}: {e}")
            
            document.contexts = contexts
        
        print(f"✅ Retrieved contexts for {len(documents)} documents")
        return documents

    def _batch_search(self, queries: List[str], qids: List[str]) -> dict:
        """Perform batch search using FAISS searcher (following DenseRetriever pattern exactly)."""
        if self.searcher is None:
            raise RuntimeError("Searcher not initialized properly")
            
        if not hasattr(self.searcher, 'batch_search'):
            raise AttributeError(f"Searcher {type(self.searcher)} does not have batch_search method")
            
        batch_results = {}
        batch_qids, batch_queries = [], []
        
        for idx, (qid, query) in enumerate(tqdm(zip(qids, queries), desc="Batch search", total=len(qids))):
            query = self._preprocess_query(query)
            batch_qids.append(qid)
            batch_queries.append(query)
            
            if (idx + 1) % self.batch_size == 0 or idx == len(qids) - 1:
                try:
                    results = self.searcher.batch_search(batch_queries, batch_qids, 
                                                       self.n_docs, self.threads)
                    batch_results.update(results)
                except Exception as e:
                    print(f"Batch search failed:")
                    print(f"  - Searcher type: {type(self.searcher)}")
                    print(f"  - Queries: {batch_queries}")
                    print(f"  - Error: {e}")
                    raise e
                
                batch_qids.clear()
                batch_queries.clear()
        
        return batch_results

    def _create_context_from_hit(self, hit, document: Document) -> Context:
        """Create Context object from search hit (following DenseRetriever pattern exactly)."""
        doc_id = hit.docid
        text = ""
        title = ""
        
        try:
            # Strategy 1: Use loaded corpus (custom)
            if self.corpus:
                # Custom corpus format
                doc_data = self.corpus.get(str(hit.docid), {})
                if doc_data:
                    text = doc_data.get("contents", "")
                    title = doc_data.get("title", "")
                    # Use ID mapping if available
                    if self.id_mapping:
                        doc_id = self.id_mapping.get(int(hit.docid), hit.docid)
                else:
                    text = f"Document {hit.docid} not found in custom corpus"
                    title = f"Document {hit.docid}"
            
            # Strategy 2: Try to get document from searcher (prebuilt)
            elif hasattr(self.searcher, 'doc'):
                try:
                    doc = self.searcher.doc(hit.docid)
                    raw_content = json.loads(doc.raw())
                    content = raw_content.get("contents", "")
                    
                    if '\n' in content:
                        lines = content.split('\n', 1)
                        title = lines[0].strip()
                        text = lines[1].strip() if len(lines) > 1 else ""
                    else:
                        title = "No Title"
                        text = content
                        
                except Exception as e:
                    print(f"Error retrieving document {hit.docid} from searcher: {e}")
                    text = f"Error retrieving document {hit.docid}"
                    title = f"Document {hit.docid}"
            
            # Strategy 3: Fallback
            else:
                print(f"Warning: No way to retrieve content for document {hit.docid}")
                text = f"Document {hit.docid}"
                title = f"Document {hit.docid}"
                
        except Exception as e:
            print(f"Error processing hit {hit.docid}: {e}")
            text = f"Error: {str(e)}"
            title = f"Document {hit.docid}"
        
        # Check if document contains answer (following DenseRetriever pattern exactly)
        has_answer = has_answers(text, document.answers.answers, self.tokenizer_simple)
        
        return Context(
            id=doc_id,
            title=title,
            text=text,
            score=hit.score,
            has_answer=has_answer
        )

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to handle token length limits (following DenseRetriever pattern)."""
        # For ANCE, we don't need to load the tokenizer here since FaissSearcher handles encoding
        # Just return the query as-is, following the same pattern as DenseRetriever
        return query.strip()