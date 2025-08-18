from abc import ABC, abstractmethod
from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.models.base_rag_model import BaseRAGModel

class BaseRAGMethod(ABC):
    def __init__(self, model: BaseRAGModel, **kwargs):
        self.model = model

    @abstractmethod
    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Abstract method to answer a question based on a list of documents.
        """
        pass