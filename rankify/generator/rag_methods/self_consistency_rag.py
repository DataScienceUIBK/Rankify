from typing import List
from collections import Counter
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

class SelfConsistencyRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, num_samples: int = 5, **kwargs):
        self.model = model
        self.num_samples = num_samples

    def answer_questions(self, documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
        """
        For each document, generate multiple answers and aggregate by majority vote.
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
            # Majority vote
            most_common, _ = Counter([ans.strip() for ans in sample_answers]).most_common(1)[0]
            answers.append(most_common)
        return answers