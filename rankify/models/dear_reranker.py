import copy
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rankify.dataset.dataset import Context, Document
from rankify.models.base import BaseRanking


class DeARReranker(BaseRanking):
    """
    Implements **DeAR (Dual-Stage Document Reranking)**, a family of
    efficient pointwise rerankers based on **LLaMA-3.2** and trained with
    **Binary Cross-Entropy** (BCE) or **RankNet** loss via knowledge
    distillation from a large teacher model.

    The model scores query–document pairs using the prompt format:

    .. code-block::

        query: <query>   [SEP]   document: <document>

    Multiple DeAR variants are supported (3B-CE, 3B-RankNet, 8B-CE, LoRA).

    References:
        - **Abdallah et al. (2025)**: *DeAR: Dual-Stage Document Reranking
          with Reasoning Agents via LLM Distillation*.
          [Paper](https://arxiv.org/abs/2508.16998)

    Attributes:
        method (str): The name of the reranking method.
        model_name (str): HuggingFace model identifier.
        device (str): Computation device (``"cuda"`` or ``"cpu"``).
        tokenizer (AutoTokenizer): Tokenizer for the DeAR model.
        model (AutoModelForSequenceClassification): The DeAR reranking model.
        batch_size (int): Batch size for inference.
        max_length (int): Maximum tokenisation length (default 228 per paper).

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.reranking import Reranking

        question = Question("When did Thomas Edison invent the light bulb?")
        answers = Answer(["1879"])
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=1),
            Context(text="Thomas Edison invented the light bulb in 1879", id=2),
            Context(text="Coffee is good for diet", id=3),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        model = Reranking(method='dear_reranker', model_name='dear-3b-reranker-ce-v1')
        model.rank([document])

        for ctx in document.reorder_contexts:
            print(ctx.text)
        ```
    """

    def __init__(
        self,
        method: str = None,
        model_name: str = None,
        api_key: str = None,
        **kwargs,
    ):
        """
        Initialises **DeARReranker**.

        Args:
            method (str, optional): Reranking method name.
            model_name (str): HuggingFace model identifier
                (e.g. ``"abdoelsayed/dear-3b-reranker-ce-v1"``).
            api_key (str, optional): Unused; present for framework consistency.
            **kwargs:
                - device (str): ``"cuda"`` or ``"cpu"``. Default: auto-detect.
                - batch_size (int): Inference batch size. Default: ``32``.
                - max_length (int): Max tokenisation length. Default: ``228``.
                - dtype: Torch dtype. Default: ``bfloat16`` on CUDA, ``float32`` on CPU.
        """
        self.method = method
        self.model_name = model_name

        device_str = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = device_str
        self.batch_size = kwargs.get("batch_size", 32)
        # Paper trains at max_length=228; expose as a tunable kwarg
        self.max_length = kwargs.get("max_length", 228)

        # Dtype: bfloat16 on GPU (matches paper), float32 on CPU
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif device_str == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device_str == "cuda" else None,
        )
        if device_str != "cuda":
            self.model = self.model.to(device_str)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks contexts within each document using **DeAR** relevance scores.

        Args:
            documents (List[Document]): Documents whose contexts to rerank.

        Returns:
            List[Document]: Documents with updated ``reorder_contexts``.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            contexts = copy.deepcopy(document.contexts)

            query_texts = [f"query: {query}"] * len(contexts)
            doc_texts = [f"document: {ctx.text}" for ctx in contexts]

            scores = self._score_batched(query_texts, doc_texts)

            for ctx, score in zip(contexts, scores):
                ctx.score = score

            document.reorder_contexts = sorted(
                contexts, key=lambda x: x.score, reverse=True
            )

        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batched(
        self,
        query_texts: List[str],
        doc_texts: List[str],
    ) -> List[float]:
        """
        Compute relevance scores for pre-formatted ``(query, document)`` pairs.

        Args:
            query_texts: Already-formatted query strings (``"query: …"``).
            doc_texts: Already-formatted document strings (``"document: …"``).

        Returns:
            List of float scores, one per pair.
        """
        scores: List[float] = []
        for start in range(0, len(query_texts), self.batch_size):
            q_batch = query_texts[start : start + self.batch_size]
            d_batch = doc_texts[start : start + self.batch_size]

            tokenized = self.tokenizer(
                q_batch,
                d_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            tokenized = {
                k: v.to(self.model.device) for k, v in tokenized.items()
            }

            logits = self.model(**tokenized).logits  # (batch, 1)
            batch_scores = logits.squeeze(-1).cpu().tolist()

            if isinstance(batch_scores, float):
                scores.append(batch_scores)
            else:
                scores.extend(batch_scores)

        return scores
