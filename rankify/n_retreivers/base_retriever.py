# base_retriever.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from rankify.dataset.dataset import Document

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods in the rankify framework.
    
    This class defines the common interface that all retrievers must implement,
    ensuring consistency across different retrieval methods (BM25, DPR, etc.).
    """
    
    def __init__(self, n_docs: int = 10, batch_size: int = 36, threads: int = 30, **kwargs):
        """
        Initialize the base retriever.
        
        Args:
            n_docs (int): Number of documents to retrieve per query
            batch_size (int): Number of queries to process in a batch
            threads (int): Number of parallel threads for retrieval
            **kwargs: Additional parameters specific to each retriever
        """
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.threads = threads
        self.config = kwargs
        
    @abstractmethod
    def _initialize_searcher(self) -> Any:
        """Initialize the specific searcher for this retriever type."""
        pass
    
    @abstractmethod
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve relevant contexts for the given documents.
        
        Args:
            documents (List[Document]): List of documents containing queries
            
        Returns:
            List[Document]: Documents updated with retrieved contexts
        """
        pass
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text. Can be overridden by specific retrievers.
        
        Args:
            query (str): Raw query text
            
        Returns:
            str: Preprocessed query text
        """
        return query.strip()