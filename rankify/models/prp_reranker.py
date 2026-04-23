"""
Pairwise Ranking Prompting (PRP) reranker.

References:
    - Qin et al. (2023): "Large Language Models are Effective Text Rankers
      with Pairwise Ranking Prompting"
      https://arxiv.org/abs/2306.17563
"""

import copy
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context


class PRPReranker(BaseRanking):
    """
    Pairwise Ranking Prompting (PRP) reranker.

    For each query, compares document pairs by prompting an LLM to decide
    which of two passages (A or B) is more relevant.  The final ranking is
    determined by the number of pairwise wins each document accumulates.

    Two comparison modes are available:

    * ``'allpairs'`` – compare every ordered pair (O(N²) LLM calls).
    * ``'bubblesort'`` – O(N log N) bubble-sort passes; faster for large sets.

    For **local** models (``method='prp'``) the winner is decided by
    comparing the next-token logit probability of the tokens ``"A"`` and
    ``"B"``.  For **API** models (``method='prp-api'``) the response text
    is parsed for the letter ``"A"`` or ``"B"``.

    References:
        - Qin et al. (2023): https://arxiv.org/abs/2306.17563

    Args:
        method (str): ``'prp'`` for local HuggingFace models,
            ``'prp-api'`` for API-based LLMs.
        model_name (str): HuggingFace model ID (local) or API model name.
        api_key (str, optional): API key when ``method='prp-api'``.
        mode (str): ``'allpairs'`` (default) or ``'bubblesort'``.
        max_pairs (int): Maximum pairwise comparisons per query (allpairs only).
            Defaults to 100.
        api_endpoint (str, optional): OpenAI-compatible API base URL.
        device (str, optional): ``'cpu'`` or ``'cuda'``.  Auto-detected when
            not supplied.

    Example:
        ```python
        from rankify.models.reranking import Reranking

        model = Reranking(method='prp', model_name='llamav3.1-8b')
        model.rank(documents)
        ```
    """

    PROMPT_TEMPLATE = (
        "Given the following query: {query}\n\n"
        "Document A: {doc_a}\n\n"
        "Document B: {doc_b}\n\n"
        "Which document is more relevant to the query? Answer with 'A' or 'B'."
    )

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        mode: str = "allpairs",
        max_pairs: int = 100,
        api_endpoint: str = "https://api.openai.com/v1",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "prp"
        self.model_name = model_name
        self.api_key = api_key
        self.mode = mode
        self.max_pairs = max_pairs
        self.api_endpoint = api_endpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._token_id_a: Optional[int] = None
        self._token_id_b: Optional[int] = None

        if self.method == "prp":
            self._load_local_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        if self.model_name is None:
            raise ValueError("model_name must be provided for method='prp'")
        print(f"Loading PRP model: {self.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Resolve token IDs for "A" and "B" (first token of each)
        self._token_id_a = self._tokenizer.encode("A", add_special_tokens=False)[0]
        self._token_id_b = self._tokenizer.encode("B", add_special_tokens=False)[0]

    # ------------------------------------------------------------------
    # Pairwise comparison primitives
    # ------------------------------------------------------------------

    def _compare_local(self, query: str, doc_a: str, doc_b: str) -> bool:
        """Return True if doc_a is preferred over doc_b (local model)."""
        prompt = self.PROMPT_TEMPLATE.format(query=query, doc_a=doc_a, doc_b=doc_b)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        last_logits = logits[0, -1]  # shape: (vocab_size,)
        score_a = last_logits[self._token_id_a].item()
        score_b = last_logits[self._token_id_b].item()
        return score_a >= score_b

    def _compare_api(self, query: str, doc_a: str, doc_b: str) -> bool:
        """Return True if doc_a is preferred over doc_b (API model)."""
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package is required for method='prp-api'. "
                "Install with: pip install openai"
            )
        prompt = self.PROMPT_TEMPLATE.format(query=query, doc_a=doc_a, doc_b=doc_b)
        client = _openai.OpenAI(api_key=self.api_key, base_url=self.api_endpoint)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("A")

    def _compare(self, query: str, doc_a: str, doc_b: str) -> bool:
        """Dispatch to local or API comparison."""
        if self.method == "prp-api":
            return self._compare_api(query, doc_a, doc_b)
        return self._compare_local(query, doc_a, doc_b)

    # ------------------------------------------------------------------
    # Ranking modes
    # ------------------------------------------------------------------

    def _rank_allpairs(self, query: str, contexts: List[Context]) -> List[Context]:
        """All-pairs tournament ranking."""
        n = len(contexts)
        wins = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        # Limit comparisons if max_pairs is set
        if self.max_pairs and len(pairs) > self.max_pairs:
            import random
            pairs = random.sample(pairs, self.max_pairs)
        for i, j in pairs:
            if self._compare(query, contexts[i].text, contexts[j].text):
                wins[i] += 1
        ranked = sorted(range(n), key=lambda k: wins[k], reverse=True)
        return [contexts[k] for k in ranked]

    def _rank_bubblesort(self, query: str, contexts: List[Context]) -> List[Context]:
        """Bubble-sort based ranking (O(N log N) comparisons)."""
        docs = list(contexts)
        n = len(docs)
        for i in range(n):
            for j in range(0, n - i - 1):
                if not self._compare(query, docs[j].text, docs[j + 1].text):
                    docs[j], docs[j + 1] = docs[j + 1], docs[j]
        return docs

    # ------------------------------------------------------------------
    # BaseRanking interface
    # ------------------------------------------------------------------

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using pairwise comparisons.

        Args:
            documents (List[Document]): Documents to rerank.

        Returns:
            List[Document]: Documents with populated ``reorder_contexts``.
        """
        for document in tqdm(documents, desc="PRP reranking"):
            if not document.contexts:
                document.reorder_contexts = []
                continue
            ctx_copy = copy.deepcopy(document.contexts)
            query = document.question.question

            if self.mode == "bubblesort":
                ranked = self._rank_bubblesort(query, ctx_copy)
            else:
                ranked = self._rank_allpairs(query, ctx_copy)

            # Assign synthetic scores based on rank position
            for rank_pos, ctx in enumerate(ranked):
                ctx.score = float(len(ranked) - rank_pos)

            document.reorder_contexts = ranked
        return documents
