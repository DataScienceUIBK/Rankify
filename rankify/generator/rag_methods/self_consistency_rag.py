from typing import List, Optional
from collections import Counter
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document, Answer
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

class SelfConsistencyRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, num_samples: int = 5, reranker=None, **kwargs):
        self.model = model
        self.num_samples = num_samples
        self.reranker = reranker  # Optional: pass a reranker instance

    def answer_questions(self, documents: List[Document], custom_prompt: Optional[str] = None, **kwargs) -> List[str]:
        """
        For each document, generate multiple answers and aggregate by majority vote or reranker.
        """
        answers = []
        for document in documents:
            question = document.question.question
            contexts = [context.text for context in document.contexts]
            prompt = self.model.prompt_generator.generate_user_prompt(question, contexts, custom_prompt)

            # Generate multiple answers in one call
            sample_answers = self.model.generate(
                prompt=prompt,
                do_sample=True,
                num_return_sequences=self.num_samples,
                **kwargs
            )

            # Ensure sample_answers is a list
            if isinstance(sample_answers, str):
                sample_answers = [sample_answers]

            print(f"Generated {len(sample_answers)} samples: {sample_answers}")

            # Use reranker if provided, otherwise majority vote
            if self.reranker:
                rerank_docs = []
                for ans in sample_answers:
                    doc = Document(question=document.question, answers=Answer(answers=[ans]), contexts=document.contexts)
                    rerank_docs.append(doc)
                reranked = self.reranker.rank(rerank_docs)
                best_answer = reranked[0].answers.answers
                if isinstance(best_answer, list):
                    best_answer = best_answer[0]
                answers.append(best_answer)
            else:
                # Majority vote (with normalization)
                def normalize(text):
                    return text.lower().strip()
                normalized = [normalize(ans) for ans in sample_answers]
                most_common, _ = Counter(normalized).most_common(1)[0]
                answers.append(most_common)
        return answers