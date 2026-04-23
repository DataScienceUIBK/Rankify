"""
RankGemma: Gemma-based listwise passage reranker.

Extends RankGPT with Gemma-specific model defaults and configuration.
Uses the same sliding-window permutation ranking approach as RankGPT.

References:
    - RankGPT: Sun et al. (2023): "Is ChatGPT Good at Search?"
      https://arxiv.org/abs/2304.09542
    - Gemma: https://huggingface.co/google/gemma-2-2b-it
"""

from typing import Optional

from rankify.models.rankgpt import RankGPT


class RankGemmaReranker(RankGPT):
    """
    Gemma-based listwise passage reranker.

    A thin wrapper around :class:`~rankify.models.rankgpt.RankGPT` that
    provides Gemma-specific model aliases and sane defaults.

    Supported ``model_name`` aliases:

    * ``'gemma-2-2b'``  → ``google/gemma-2-2b-it``
    * ``'gemma-2-9b'``  → ``google/gemma-2-9b-it``
    * ``'gemma-2-27b'`` → ``google/gemma-2-27b-it``

    Any raw HuggingFace model ID is also accepted (e.g.
    ``'google/gemma-2-2b-it'``).

    Args:
        method (str, optional): Reranking method name.
        model_name (str, optional): Model alias or full HuggingFace ID.
        api_key (str, optional): Unused for local models; kept for interface
            compatibility.
        **kwargs: Additional keyword arguments forwarded to RankGPT
            (e.g. ``window_size``, ``step``).

    Example:
        ```python
        from rankify.models.reranking import Reranking

        model = Reranking(method='rankgemma', model_name='gemma-2-2b')
        model.rank(documents)
        ```
    """

    DEFAULT_MODELS = {
        "gemma-2-2b": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
    }

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        resolved = self.DEFAULT_MODELS.get(model_name, model_name)
        super().__init__(
            method=method or "rankgemma",
            model_name=resolved,
            api_key=api_key,
            **kwargs,
        )
