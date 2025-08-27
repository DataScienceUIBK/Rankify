from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from tqdm.auto import tqdm 

class ZeroShotRAG(BaseRAGMethod):
    """
    **Zero-Shot RAG** for Open-Domain Question Answering.

    This class implements zero-shot genration, where the model generates answers directly 
    without any type of context. This serves as a baseline for comparison with RAG methods.

    Methods:
        answer_questions(documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
            Answers questions for a list of documents using the model in a zero-shot manner.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.generator.generator import Generator

        # Sample question and contexts
        question = Question("What is the capital of France?")
        answers=Answer('')
        contexts = [
            Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
            Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
        ]

        # Create a Document
        doc = Document(question=question, answers= answers, contexts=contexts)

        # Initialize Generator (e.g., Meta Llama, with huggingface backend)
        generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface")

        # Generate answer
        generated_answers = generator.generate([doc])
        print(generated_answers)  # Output: ["Paris"]
        ```
            
    Notes:
        - Suitable for baseline comparison with RAG methods.
        - Uses the model's prompt generator to construct prompts from question.
    """
    def __init__(self, model: BaseRAGModel, **kwargs):
        """
        Initialize the ZeroShotRAG method.

        Args:
            model (BaseRAGModel): The underlying model used for text generation.
        """
        super().__init__(model=model)

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the model in a zero-shot manner.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the model's generate method.

        Returns:
            List[str]: A list of answers.

        Notes:
            - Constructs prompts using only the question.
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