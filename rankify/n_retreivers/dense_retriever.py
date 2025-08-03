# dense_retriever.py
import json
import os
import requests
from typing import List
from pyserini.search.faiss import FaissSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context

class DenseRetriever(BaseRetriever):
    """
    Dense retriever implementation supporting DPR, ANCE, and BPR models.
    
    Uses FAISS indexes for efficient dense vector retrieval.
    """
    
    MSMARCO_CORPUS_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true"
    
    def __init__(self, method: str = "dpr-multi", index_type: str = "wiki", 
                 index_folder: str = None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.index_type = index_type
        self.index_folder = index_folder
        self.tokenizer = SimpleTokenizer()
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Handle custom index folder case
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
        
        self.question_tokenizer = AutoTokenizer.from_pretrained(self.query_encoder)
        
        # Initialize variables
        self.id_mapping = None
        self.corpus = {}
        
        # Handle custom vs prebuilt indexes
        if index_folder:
            self.index_path = index_folder
            self.id_mapping = self.index_manager.load_id_mapping(index_folder)
            self.corpus = self.index_manager.load_corpus(index_folder)
            print(f"Loaded corpus with {len(self.corpus)} documents")
            if self.id_mapping:
                print(f"Loaded ID mapping with {len(self.id_mapping)} entries")
        else:
            self.index_path = self.index_manager.get_index_path(method, index_type)
        
        # Initialize searcher
        self.searcher = self._initialize_searcher()
        if self.searcher is None:
            raise RuntimeError("Failed to initialize searcher")
        
        # Load corpus for MSMARCO if needed
        if index_type == "msmarco" and not index_folder:
            self._load_msmarco_corpus()
    
    def _initialize_searcher(self):
        """
        Initialize FAISS searcher for all cases.
        This is PURE DPR - no LuceneSearcher anywhere!
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
            
            # Case 2c: Prebuilt index name (like "wikipedia-dpr-100w.dpr-multi")
            else:
                print(f"Using prebuilt index: {self.index_path}")
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
    
    def _load_msmarco_corpus(self):
        """Load MSMARCO corpus into memory."""
        corpus_file = os.path.join(self.index_manager.cache_dir, "msmarco-passage-corpus.tsv")
        
        if not os.path.exists(corpus_file):
            print("Downloading MSMARCO corpus...")
            self._download_file(self.MSMARCO_CORPUS_URL, corpus_file)
        
        print("Loading MSMARCO corpus...")
        self.corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    doc_id, text, title = parts[0], parts[1], parts[2]
                    self.corpus[doc_id] = {"text": text, "title": title}
    
    def _download_file(self, url: str, save_path: str):
        """Download a file from URL."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), 
                            desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using dense retrieval."""
        queries = [doc.question.question for doc in documents]
        qids = [str(i) for i in range(len(queries))]
        
        print(f"Retrieving {len(documents)} documents with {self.method}...")
        
        # Perform batch search
        batch_results = self._batch_search(queries, qids)
        
        # Process results
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
            #break
        
        return documents
    
    def _create_context_from_hit(self, hit, document) -> Context:
        """Create Context object from search hit."""
        doc_id = hit.docid
        text = ""
        title = ""
        
        try:
            # Strategy 1: Use loaded corpus (MSMARCO or custom)
            if self.corpus:
                if self.index_type == "msmarco":
                    # MSMARCO corpus format
                    doc_data = self.corpus.get(str(hit.docid), {})
                    if doc_data:
                        text = doc_data.get("text", "")
                        title = doc_data.get("title", "")
                    else:
                        text = f"Document {hit.docid} not found in MSMARCO corpus"
                        title = f"Document {hit.docid}"
                else:
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
            
            # Strategy 2: Try to get document from searcher
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
        
        return Context(
            id=doc_id,
            title=title,
            text=text,
            score=hit.score,
            has_answer=has_answers(text, document.answers.answers, self.tokenizer)
        )
    
    def _batch_search(self, queries: List[str], qids: List[str]) -> dict:
        """Perform batch search using FAISS searcher."""
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
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to handle token length limits."""
        try:
            tokenized = self.question_tokenizer(query, add_special_tokens=True)
            
            if len(tokenized["input_ids"]) > 511:
                truncated = self.question_tokenizer.decode(
                    tokenized["input_ids"][:511], skip_special_tokens=True
                )
                return truncated
            return query
        except Exception as e:
            print(f"Error preprocessing query '{query}': {e}")
            return query