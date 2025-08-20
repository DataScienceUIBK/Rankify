from abc import ABC, abstractmethod
from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.models.base_rag_model import BaseRAGModel

class BaseRAGMethod(ABC):
    """
    **Base RAG Method** for Retrieval-Augmented Generation (RAG) techniques.

    This abstract base class defines the blueprint for implementing RAG methods in Rankify.
    Each RAG method (e.g., zero-shot, chain-of-thought, Fusion-in-Decoder) should inherit from this class
    and implement the logic for answering questions using a provided RAG model.

    Attributes:
        model (BaseRAGModel): The RAG model instance used for generation.

    Methods:
        answer_questions(documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
            Abstract method to answer questions based on a list of documents and optional custom prompt.

    Notes:
        - Extend this class to implement new RAG techniques or strategies.
        - The `answer_questions` method must be implemented by all subclasses.
        - This class enables modularity and extensibility for different retrieval-augmented generation approaches.
    """
    def __init__(self, model: BaseRAGModel, **kwargs):
        """
        Initialize the BaseRAGMethod.

        Args:
            model (BaseRAGModel): The RAG model instance used for generation.
            **kwargs: Additional configuration parameters for the RAG method.
        """
        self.model = model

    @abstractmethod
    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Abstract method to answer questions based on a list of documents.
    
        Args:
            documents (List[Document]): List of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the answering logic.
    
        Returns:
            List[str]: List of generated answers, one per document.
    
        Notes:
            - Must be implemented by subclasses to define the RAG technique's answering logic.
            - Enables flexible integration of different prompting or generation strategies.
        """
        pass