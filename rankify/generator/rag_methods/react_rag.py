from typing import List, Optional
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
import re

class ReActRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, retriever, max_steps: int = 20, max_contexts_per_search: int = 3, use_internal_knowledge: bool = True, **kwargs):
        self.model = model
        self.retriever = retriever  # Pass a retriever instance
        self.max_steps = max_steps
        self.max_contexts_per_search = max_contexts_per_search
        self.use_internal_knowledge = use_internal_knowledge

    def answer_questions(self, documents: List[Document], custom_prompt: Optional[str] = None, **kwargs) -> List[str]:
        answers = []
        for document in documents:
            question = document.question.question
            contexts = [context.text for context in document.contexts]
            history = []
            final_answer = None
            for step in range(self.max_steps):
                prompt = self.model.prompt_generator.generate_user_prompt(
                    question, contexts, custom_prompt=custom_prompt
                )
                # Add history to the prompt
                if history:
                    prompt += "\n" + "\n".join(history)
                output = self.model.generate(prompt=prompt, **kwargs)
                # Check for Final Answer
                final_match = re.search(r"Final Answer:\s*(.*)", output)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    break
                # Parse for Search action
                search_match = re.search(r"Search\[(.*?)\]", output)
                if search_match:
                    query = search_match.group(1)
                    # Use retriever to get new contexts (limit by max_contexts_per_search)
                    retrieved_docs = self.retriever.retrieve([Document(question=document.question, answers=document.answers, contexts=[])])
                    if retrieved_docs and retrieved_docs[0].contexts:
                        obs_texts = []
                        for ctx in retrieved_docs[0].contexts[:self.max_contexts_per_search]:
                            obs_texts.append(ctx.text)
                            contexts.append(ctx.text)
                        obs = "Observation: " + " ".join(obs_texts)
                        history.append(f"Search[{query}]\n{obs}")
                else:
                    # If neither Search nor Final Answer, add reasoning to history
                    history.append(output)
            # If final answer found, use it
            if final_answer is not None:
                answers.append(final_answer)
            else:
                # Fallback to internal knowledge if flag is set
                if self.use_internal_knowledge:
                    internal_prompt = (
                        "You are a knowledgeable assistant. Answer the following question using only your internal knowledge.\n"
                        f"Question: {question}\n"
                        "Answer:"
                    )
                    internal_output = self.model.generate(prompt=internal_prompt, **kwargs)
                    answers.append(internal_output)
                else:
                    # Else answer with last output
                    answers.append(output)
        return answers