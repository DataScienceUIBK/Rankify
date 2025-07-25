from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from tqdm.auto import tqdm 

class ZeroShotRAG(BaseRAGMethod):
    """
    **Zero-Shot RAG** for Open-Domain Question Answering.

    This class implements a zero-shot retrieval-augmented generation (RAG) method, 
    where the model generates answers directly from the provided contexts without 
    requiring additional fine-tuning.

    Attributes:
        model (BaseRAGModel): The underlying model used for text generation.
    """
    def __init__(self, model: BaseRAGModel, **kwargs):
        """
        Initialize the ZeroShotRAG method.

        Args:
            model (BaseRAGModel): A model instance for text generation.
            kwargs: Additional arguments for customization.
        """
        self.model = model

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the model in a zero-shot manner.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            List[str]: A list of answers.
        """
        answers = []

        for document in tqdm(documents, desc="Answering questions", unit="q"):
            # Extract question and contexts from the document
            question = document.question.question
            
            # Construct the prompt
            prompt = self.model.prompt_generator.generate_user_prompt(question, None, custom_prompt)
            
            # Generate the answer using the model
            answer = self.model.generate(prompt, **kwargs)

            # Append the answer to the list
            answers.append(answer)

        return answers