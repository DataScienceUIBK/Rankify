from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel

from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from tqdm.auto import tqdm 

class BasicRAG(BaseRAGMethod):
    """
    **BasicRAG (Naive RAG) Method** for Retrieval-Augmented Generation.

    Implements a simple RAG technique that answers questions by concatenating retrieved contexts and passing them,
    along with the question, to the underlying model. This is the most straightforward RAG approach.

    Attributes:
        model (BaseRAGModel): The RAG model instance used for generation.

    References:
        - **Lewis et al. **Retrieval-augmented generation for knowledge-intensive nlp tasks**  
          [Paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
    
    Notes:
        - This method does not apply advanced reasoning or fusion techniques.
        - Suitable as a baseline for comparison with more sophisticated RAG methods.
    """
    def __init__(self, model: BaseRAGModel):
        """
        Initialize the BasicRAG method.

        Args:
            model (BaseRAGModel): The RAG model instance used for generation.
        """
        super().__init__(model=model)

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the model.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the model's generate method.

        Returns:
            List[str]: Answers generated for each document.

        Notes:
            - Concatenates all context passages and passes them with the question to the model.
            - Uses the model's prompt generator to construct prompts.
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