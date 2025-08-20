from typing import List, Optional
from collections import Counter
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document, Answer
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from tqdm.auto import tqdm 

class SelfConsistencyRAG(BaseRAGMethod):
    """
    **SelfConsistencyRAG Method** for Retrieval-Augmented Generation.

    Implements the self-consistency technique for multi-step question answering.
    For each question, the model generates multiple answers with a Chain-of-Thought style prompt
    and aggregates them using majority vote or an optional reranker, improving reliability
    and robustness of the final answer.

    Attributes:
        model (BaseRAGModel): The RAG model instance used for generation.
        num_samples (int): Number of answer samples to generate per question (default: 5).
        reranker: Optional reranker instance for ranking generated answers.

    Methods:
        __init__(model, num_samples=5, reranker=None, **kwargs):
            Initializes the SelfConsistencyRAG method with the provided model, sample count, and optional reranker.
        answer_questions(documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
            Generates multiple answers for each document and aggregates by majority vote or reranker.

    Notes:
        - Generates multiple answers per question using the underlying model's sampling capabilities.
        - If a reranker is provided, uses it to select the best answer; otherwise, applies majority voting.
        - Suitable for reducing randomness and improving answer reliability in generative QA.

    References:
        - Wang et al. *Self-consistency improves chain of thought reasoning in language models*  
          [Paper](https://arxiv.org/abs/2203.11171)
    """
    def __init__(self, model: BaseRAGModel, num_samples: int = 5, reranker=None):
        """
        Initialize the SelfConsistencyRAG method.

        Args:
            model (BaseRAGModel): The RAG model instance used for generation.
            num_samples (int, optional): Number of answer samples to generate per question (default: 5).
            reranker (optional): Reranker instance for ranking generated answers.
            **kwargs: Additional configuration parameters (unused).
        """
        super().__init__(model=model)
        self.num_samples = num_samples
        self.reranker = reranker  # Optional: pass a reranker instance

    def answer_questions(self, documents: List[Document], custom_prompt: Optional[str] = None, **kwargs) -> List[str]:
        """
        For each document, generate multiple answers and aggregate by majority vote or reranker.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the model's generate method.

        Returns:
            List[str]: Aggregated answers for each document.

        Notes:
            - Uses the model's prompt generator to construct Chain-of-Thought style prompt.
            - Generates multiple answers per question using sampling and aggregates results.
            - If a reranker is provided, selects the best answer using reranking; otherwise, applies majority voting.
        """
        answers = []
        for document in tqdm(documents, desc="Answering questions", unit="q"):
            question = document.question.question
            contexts = [context.text for context in document.contexts]
            prompt = self.model.prompt_generator.generate_user_prompt(question, contexts, custom_prompt)

            # Generate multiple answers in one call
            sample_answers = self.model.generate(
                prompt=prompt,
                #do_sample=True,
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
                    return text.strip()
                normalized = [normalize(ans) for ans in sample_answers]
                most_common, _ = Counter(normalized).most_common(1)[0]
                answers.append(most_common)
        return answers