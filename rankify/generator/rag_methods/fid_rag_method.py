from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel

from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

class FiDRAGMethod(BaseRAGMethod):
    """
    **FiD RAG Method** for Open-Domain Question Answering.

    This class implements a retrieval-augmented generation (RAG) method using the 
    Fusion-in-Decoder (FiD) approach. The FiD model aggregates information from 
    multiple retrieved passages to generate context-aware answers.

    References:
        - **Izacard & Grave** *Leveraging Passage Retrieval with Generative Models for Open-Domain QA*  
          [Paper](https://arxiv.org/abs/2007.01282)

    Attributes:
        model (BaseRAGModel): The underlying FiD model used for text generation.

    Methods:
        answer_questions(documents: List[Document], **kwargs) -> List[str]:
            Generates answers for a list of documents using the FiD model.

    Notes:
        - The FiD model combines multiple passages to generate better responses.
    
    """
    def __init__(self, model: BaseRAGModel):
        self.model = model

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the model.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            Lists[str]: An answer based on the given documents and question.
        """

        # Generate the answer using the model
        answer = self.model.generate(documents, **kwargs)
        
        return answer