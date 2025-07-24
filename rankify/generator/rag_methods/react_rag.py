from typing import List, Optional
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
import re

class ReActRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, retriever, max_steps: int = 3, **kwargs):
        self.model = model
        self.retriever = retriever  # Pass a retriever instance
        self.max_steps = max_steps

    def answer_questions(self, documents: List[Document], custom_prompt: Optional[str] = None, **kwargs) -> List[str]:
        answers = []
        for document in documents:
            question = document.question.question
            contexts = [context.text for context in document.contexts]
            history = []
            for step in range(self.max_steps):
                prompt = self.model.prompt_generator.generate_user_prompt(
                    question, contexts, custom_prompt=custom_prompt
                )
                # Add history to the prompt
                if history:
                    prompt += "\n" + "\n".join(history)
                output = self.model.generate(prompt=prompt, **kwargs)
                # Parse for Search action
                search_match = re.search(r"Search\[(.*?)\]", output)
                if search_match:
                    query = search_match.group(1)
                    # Use retriever to get new context
                    retrieved_docs = self.retriever.retrieve([Document(question=document.question, answers=document.answers, contexts=[])])
                    # For simplicity, just take the first retrieved context
                    if retrieved_docs and retrieved_docs[0].contexts:
                        obs = f"Observation: {retrieved_docs[0].contexts[0].text}"
                        history.append(f"Search[{query}]\n{obs}")
                        contexts.append(retrieved_docs[0].contexts[0].text)
                        continue  # Continue reasoning
                # If no Search action, assume answer is given
                answers.append(output)
                break
            else:
                # If max_steps reached without answer, return last output
                answers.append(output)
        return answers