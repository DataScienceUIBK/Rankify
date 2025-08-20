from abc import ABC, abstractmethod
from typing import List

class BaseRAGModel(ABC):
    """
    **Base RAG Model** for Retrieval-Augmented Generation (RAG).

    This is an abstract base class for implementing LLM endpoints in rankify. 
    It defines the interface for generating responses and optional embedding generation.

    Methods:
        generate(prompt: str, **kwargs) -> str:
            Abstract method to generate a response based on the given prompt.
        embed(text: str, **kwargs) -> List[float]:
            Optional method to generate embeddings for the given text.

    Notes:
        - This class serves as a blueprint for RAG models like `OpenAIModel` and `HuggingFaceModel`.
        - The `embed` method is optional and can be implemented if needed.
        - This class needs to be extended to include new LLM endpoints in Rankify.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the given prompt."""
        pass

    def embed(self, text: str, **kwargs) -> List[float]:
        """Optional: Generate embeddings for the given text."""
        raise NotImplementedError("Embedding is not required for this implementation.")