"""
TART: Task-Aware Reranker with Instructions.

References:
    - Asai et al. (2022): "Task-Aware Retrieval with Instructions"
      https://arxiv.org/abs/2211.09260
    - Model: facebook/tart-full-flan-t5-xl
"""

import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context


class TARTReranker(BaseRanking):
    """
    TART: Task-Aware Reranker with Instructions.

    Uses a cross-encoder that takes an optional task instruction prepended to
    the query, enabling zero-shot reranking on diverse tasks.  The default
    model is ``facebook/tart-full-flan-t5-xl`` which is a T5-xl model
    fine-tuned with instruction-following for cross-encoding.

    The relevance score is computed as ``softmax(logits)[:, 1]`` – the
    probability that a passage is relevant to the (instruction, query) pair.

    References:
        - Asai et al. (2022): https://arxiv.org/abs/2211.09260

    Args:
        method (str, optional): Reranking method name.
        model_name (str): HuggingFace model ID or path.
        api_key (str, optional): Unused; kept for API compatibility.
        instruction (str): Task instruction prepended to the query.
        batch_size (int): Inference batch size.
        device (str, optional): ``'cpu'`` or ``'cuda'``.  Auto-detected if
            not supplied.
        max_length (int): Tokenisation max length.

    Example:
        ```python
        from rankify.models.reranking import Reranking

        model = Reranking(method='tart', model_name='tart-full-flan-t5-xl')
        model.rank(documents)
        ```
    """

    DEFAULT_INSTRUCTION = "Retrieve a passage that answers the question."

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        instruction: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
        **kwargs,
    ):
        self.method = method or "tart"
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve model name alias
        resolved = model_name
        if model_name is not None:
            from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
            tart_models = HF_PRE_DEFIND_MODELS.get("tart", {})
            resolved = tart_models.get(model_name, model_name)

        if resolved is None:
            resolved = "facebook/tart-full-flan-t5-xl"

        self.model_name = resolved
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading TART model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _build_query(self, query: str) -> str:
        """Prepend instruction to the query as TART expects."""
        return f"{self.instruction} [SEP] {query}"

    @torch.no_grad()
    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank passages using TART's instruction-aware cross-encoder.

        Args:
            documents (List[Document]): Documents to rerank.

        Returns:
            List[Document]: Documents with populated ``reorder_contexts``.
        """
        for document in tqdm(documents, desc="TART reranking"):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            ctx_copy = copy.deepcopy(document.contexts)
            query_with_instruction = self._build_query(document.question.question)
            doc_texts = [ctx.text for ctx in ctx_copy]

            all_scores: List[float] = []
            for i in range(0, len(doc_texts), self.batch_size):
                batch_docs = doc_texts[i : i + self.batch_size]
                # Replicate query for each document in the batch
                batch_queries = [query_with_instruction] * len(batch_docs)
                features = self.tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**features).logits
                # Probability of class 1 (relevant)
                probs = F.softmax(logits, dim=1)
                batch_scores = probs[:, 1].cpu().tolist()
                all_scores.extend(batch_scores)

            for score, ctx in zip(all_scores, ctx_copy):
                ctx.score = score

            ctx_copy.sort(key=lambda x: x.score, reverse=True)
            document.reorder_contexts = ctx_copy

        return documents
