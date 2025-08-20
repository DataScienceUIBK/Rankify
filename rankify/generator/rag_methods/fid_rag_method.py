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

    See Also:
        - `FiDModel`: Class for FiDModel, containing the FiD specific logic.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.generator.generator import Generator

        # Define question and answer
        question = Question("What is the capital of France?")
        answers = Answer([""])
        contexts = [
            Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
            Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
        ]

        # Construct document
        doc = Document(question=question, answers=answers, contexts=contexts)

        # Initialize Generator (e.g., Meta Llama)
        generator = Generator(method="fid", model_name='nq_reader_base', backend="fid")

        # Generate answer
        generated_answers = generator.generate([doc])
        print(generated_answers)  # Output: ["Paris"]
        ```
    
    Notes:
        - This class was created to keep the unified interface of RAG methods.
        - Since FiD is a specific RAG technique that relies on the full transformer architecture,
          the logic is included in the model, see 
    
    """
    def __init__(self, model: BaseRAGModel, **kwargs):
        super().__init__(model=model)

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the FiDModel.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            Lists[str]: An answer based on the given documents and question.
        """

        # Generate the answer using the model
        answer = self.model.generate(documents, **kwargs)
        
        return answer