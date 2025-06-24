from abc import ABC, abstractmethod
from typing import List
from rankify.dataset.dataset import Document

class BaseRAGMethod(ABC):
    @abstractmethod
    def answer_questions(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Abstract method to answer a question based on a list of documents.
        """
        pass