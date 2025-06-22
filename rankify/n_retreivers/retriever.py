# rankify/retrievers/retriever.py - UPDATED VERSION
from typing import List, Dict, Type
from rankify.dataset.dataset import Document
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .ance_retriever import ANCERetriever  # UPDATED IMPORT
from .bge_retriever import BGERetriever
from .colbert_retriever import ColBERTRetriever
from .contriever_retriever import ContrieverRetriever
from .online_retriever import OnlineRetriever
from .hyde_retriever import HydeRetriever

# Method mapping - UPDATED WITH PROPER ANCE SUPPORT
METHOD_MAP: Dict[str, Type[BaseRetriever]] = {
    "bm25": BM25Retriever,
    "dpr-multi": DenseRetriever,
    "dpr-single": DenseRetriever,
    "ance-multi": ANCERetriever,   # UPDATED: Now uses ANCERetriever with prebuilt indices
    "bpr-single": DenseRetriever,
    "bge": BGERetriever, 
    "colbert": ColBERTRetriever, 
    "contriever": ContrieverRetriever, 
    "online": OnlineRetriever, 
    "hyde": HydeRetriever, 
}

class Retriever:
    """
    Unified retriever interface for the rankify framework.
    
    Provides a simple interface to access different retrieval methods
    (BM25, DPR, ANCE, BPR) with consistent parameters.
    
    Example:
        ```python
        # Initialize with BM25
        retriever = Retriever(method="bm25", n_docs=10, index_type="wiki")
        
        # Initialize with DPR
        retriever = Retriever(method="dpr-multi", n_docs=5, index_type="msmarco")
        
        # Initialize with ANCE (UPDATED - now works with index_type)
        retriever = Retriever(method="ance", n_docs=10, index_type="wiki")
        
        # Initialize with ANCE-Multi (uses prebuilt Wikipedia indices)
        retriever = Retriever(method="ance-multi", n_docs=10, index_type="wiki")
        
        # Initialize with custom index folder (works for all methods)
        retriever = Retriever(method="ance", n_docs=10, index_folder="/path/to/index")
        
        # Retrieve documents
        retrieved_documents = retriever.retrieve(documents)
        ```
    """
    
    def __init__(self, method: str, n_docs: int = 10, index_type: str = "wiki", 
                 index_folder: str = None, encoder_name: str = None, **kwargs):
        """
        Initialize the retriever.
        
        Args:
            method (str): Retrieval method ('bm25', 'dpr-multi', 'dpr-single', 'ance', 'ance-multi', 'bpr-single', etc.)
            n_docs (int): Number of documents to retrieve per query
            index_type (str): Index type ('wiki', 'msmarco') - ignored if index_folder is provided
            index_folder (str): Path to custom index folder (optional)
            encoder_name (str): Model name for encoding (method-specific)
            **kwargs: Additional parameters passed to the specific retriever
        """
        self.method = method.lower()
        self.n_docs = n_docs
        self.index_type = index_type.lower()
        self.index_folder = index_folder
        self.encoder_name = encoder_name
        self.kwargs = kwargs
        
        # Initialize the specific retriever
        self.retriever = self._initialize_retriever()
    
    def _initialize_retriever(self) -> BaseRetriever:
        """Initialize the specific retriever based on the method."""
        if self.method not in METHOD_MAP:
            supported_methods = ", ".join(METHOD_MAP.keys())
            raise ValueError(f"Unsupported method '{self.method}'. "
                           f"Supported methods: {supported_methods}")
        
        retriever_class = METHOD_MAP[self.method]
        
        # Prepare initialization parameters
        init_params = {
            "n_docs": self.n_docs,
            **self.kwargs
        }
        
        # UPDATED: Handle all ANCE variants the same way (with index_type support)
        if self.method in ["ance", "ance-msmarco", "ance-multi"]:
            init_params["index_type"] = self.index_type
            
            # Add index_folder if provided
            if self.index_folder:
                init_params["index_folder"] = self.index_folder
            
            # Add encoder_name if provided
            if self.encoder_name:
                init_params["encoder_name"] = self.encoder_name
                
        # Handle other retrieval methods (same as before)
        else:
            init_params["index_type"] = self.index_type
            
            # Add index_folder if provided
            if self.index_folder:
                init_params["index_folder"] = self.index_folder
            
            # Add method parameter for dense retrievers
            if self.method in ["dpr-multi", "dpr-single", "bpr-single"]:
                init_params["method"] = self.method
        
        return retriever_class(**init_params)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve relevant contexts for the given documents.
        
        Args:
            documents (List[Document]): List of documents containing queries
            
        Returns:
            List[Document]: Documents updated with retrieved contexts
        """
        return self.retriever.retrieve(documents)
    
    @classmethod
    def supported_methods(cls) -> List[str]:
        """Get list of supported retrieval methods."""
        return list(METHOD_MAP.keys())
    
    def __repr__(self) -> str:
        index_info = f"index_folder='{self.index_folder}'" if self.index_folder else f"index_type='{self.index_type}'"
        return (f"Retriever(method='{self.method}', n_docs={self.n_docs}, "
                f"{index_info})")