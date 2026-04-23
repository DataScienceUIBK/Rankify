"""
RankMistral: Mistral-based listwise passage reranker.

Extends RankGPT with Mistral-specific model defaults and configuration.
Uses the same sliding-window permutation ranking approach as RankGPT.

References:
    - RankGPT: Sun et al. (2023): "Is ChatGPT Good at Search?"
      https://arxiv.org/abs/2304.09542
    - Mistral: https://mistral.ai
"""

from typing import Optional

from rankify.models.rankgpt import RankGPT


class RankMistralReranker(RankGPT):
    """
    Mistral-based listwise passage reranker.

    A thin wrapper around :class:`~rankify.models.rankgpt.RankGPT` that
    provides Mistral-specific model aliases and sane defaults.

    Supported ``model_name`` aliases:

    * ``'mistral-7b'``       → ``mistralai/Mistral-7B-Instruct-v0.3``
    * ``'mistral-7b-v0.2'``  → ``mistralai/Mistral-7B-Instruct-v0.2``
    * ``'mixtral-8x7b'``     → ``mistralai/Mixtral-8x7B-Instruct-v0.1``

    Any raw HuggingFace model ID is also accepted.

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

        model = Reranking(method='rankmistral', model_name='mistral-7b')
        model.rank(documents)
        ```
    """

    DEFAULT_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
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
            method=method or "rankmistral",
            model_name=resolved,
            api_key=api_key,
            **kwargs,
        )
