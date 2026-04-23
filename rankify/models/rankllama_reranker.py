import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from rankify.dataset.dataset import Context, Document
from rankify.models.base import BaseRanking

try:
    from peft import PeftConfig, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RankLLaMAReranker(BaseRanking):
    """
    Implements **RankLLaMA**, a **LLaMA-2-7B** model fine-tuned with **LoRA**
    for passage reranking via pointwise relevance scoring.

    The model is loaded using the **PEFT** library (LoRA adapter merged into the
    base model), and scores query–passage pairs using the prompt format:

    .. code-block::

        query: <query>   [SEP]   document: <title> <passage>

    References:
        - **Ma et al. (2023)**: *Fine-Tuning LLaMA for Multi-Stage Text Retrieval*.
          [Paper](https://arxiv.org/abs/2310.08319)

    Attributes:
        method (str): The name of the reranking method.
        model_name (str): The HuggingFace PEFT adapter name/path.
        device (str): Computation device (``"cuda"`` or ``"cpu"``).
        tokenizer (AutoTokenizer): Tokenizer loaded from the base LLaMA-2-7B checkpoint.
        model (AutoModelForSequenceClassification): Merged LoRA + base model.
        batch_size (int): Batch size for inference.
        max_length (int): Maximum tokenisation length.

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

        model = Reranking(method='rankllama', model_name='rankllama-v1-7b-lora-passage')
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
        Initialises **RankLLaMAReranker**.

        Args:
            method (str, optional): Reranking method name.
            model_name (str): HuggingFace PEFT adapter identifier
                (e.g. ``"castorini/rankllama-v1-7b-lora-passage"``).
            api_key (str, optional): Unused; present for framework consistency.
            **kwargs:
                - device (str): ``"cuda"`` or ``"cpu"``. Default: auto-detect.
                - batch_size (int): Inference batch size. Default: ``8``.
                - max_length (int): Max tokenisation length. Default: ``512``.
                - tokenizer_name (str): Override the base tokenizer to load.
                  Default: resolved automatically from the PEFT config.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "The `peft` package is required for RankLLaMAReranker. "
                "Install it with: pip install peft"
            )

        self.method = method
        self.model_name = model_name
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = kwargs.get("batch_size", 8)
        self.max_length = kwargs.get("max_length", 512)

        # Load PEFT config to discover the base model name
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Tokenizer: use the base LLaMA-2 tokenizer (or an explicit override)
        tokenizer_name = kwargs.get("tokenizer_name", base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model + merge LoRA weights
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            torch_dtype=dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, model_name)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks contexts within each document using **RankLLaMA** scores.

        Args:
            documents (List[Document]): Documents whose contexts to rerank.

        Returns:
            List[Document]: Documents with updated ``reorder_contexts``.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            contexts = copy.deepcopy(document.contexts)

            # Build query strings and document strings
            query_texts = [f"query: {query}"] * len(contexts)
            doc_texts = [
                f"document: {ctx.title + ' ' if ctx.title else ''}{ctx.text}"
                for ctx in contexts
            ]

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
        Compute relevance scores for ``(query, document)`` string pairs.

        Args:
            query_texts: Already-formatted query strings.
            doc_texts: Already-formatted document strings.

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
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            logits = self.model(**tokenized).logits  # (batch, 1)
            batch_scores = logits.squeeze(-1).cpu().tolist()

            if isinstance(batch_scores, float):
                scores.append(batch_scores)
            else:
                scores.extend(batch_scores)

        return scores
