from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

from tqdm.auto import tqdm 

class ChainOfThoughtRAG(BaseRAGMethod):
    """
    **Chain-of-Thought RAG** for Open-Domain Question Answering.

    This class implements a chain-of-thought retrieval-augmented generation (RAG) method, 
    where the model generates answers by reasoning step-by-step using the provided contexts.

    Attributes:
        model (BaseRAGModel): The underlying model used for text generation.

    References:
        - **Wei et al. **Chain-of-thought prompting elicits reasoning in large language models**  
          [Paper](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
        
    Methods:
        answer_questions(documents: List[Document], **kwargs) -> List[str]:
            Generates answers for a list of documents using chain-of-thought reasoning.

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
        generator = Generator(method="chain-of-thought-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface")

        # Generate answer
        generated_answers = generator.generate([doc])
        print(generated_answers)  # Output: ["Paris"]
        ```
            
    Notes:
        - This method uses chain-of-thought reasoning to generate more detailed and logical answers.
        - It is particularly useful for complex questions that require multi-step reasoning.

    """
    def __init__(self, model: BaseRAGModel, **kwargs):
        super().__init__(model=model)

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using chain-of-thought reasoning.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the model's generate method.

        Returns:
            List[str]: Answers generated for each document, with chain-of-thought reasoning.

        Notes:
            - Uses the model's prompt generator to build chain-of-thought prompts from question and contexts.
            - Suitable for complex questions requiring multi-step inference.
        """
        answers = []

        for document in tqdm(documents, desc="Answering questions", unit="q"):
            # Extract question and contexts from the document
            question = document.question.question
            contexts = [context.text for context in document.contexts]

            # Construct the prompt
            prompt = self.model.prompt_generator.generate_user_prompt(question, contexts, custom_prompt)
            
            # Generate the answer using the model
            answer = self.model.generate(prompt=prompt, **kwargs)
            
            # Append the answer to the list
            answers.append(answer)
        return answers