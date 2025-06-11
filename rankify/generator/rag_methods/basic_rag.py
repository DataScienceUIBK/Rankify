from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel

from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

class BasicRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, **kwargs):
        self.model = model

    def answer_questions(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Answer question for a list of documents using the model.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            str: An answer based on the given documents and question.
        """
        answers = []

        for document in documents:
            # Extract question and contexts from the document
            question = document.question.question
            contexts = [context.text for context in document.contexts]

            # Construct the prompt
            prompt = self.model.prompt_generator.generate_user_prompt(question, contexts)
            # Generate the answer using the model
            answer = self.model.generate(prompt=prompt, **kwargs)
            
            # Append the answer to the list
            answers.append(answer)

            document.answers = answer

        return answers