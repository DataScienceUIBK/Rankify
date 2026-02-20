import torch
from rankify.generator.models.fid_model import FiDModel
from rankify.generator.models.litellm_model import LitellmModel
from rankify.generator.models.openai_model import OpenAIModel
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.models.vllm_model import VLLMModel
from rankify.generator.prompt_generator import PromptGenerator
from rankify.utils.generator.huggingface_models.model_utils import load_model, load_tokenizer

def model_factory(model_name: str, backend: str, method: str, stop_at_period=False, **kwargs) -> BaseRAGModel:
    """
    Factory function to instantiate and return the appropriate RAG model based on the backend.

    Args:
        model_name (str): Name of the model to use (e.g., 'meta-llama/Meta-Llama-3.1-8B', 'nq_reader_base').
        backend (str): Backend type ('openai', 'huggingface', 'fid', 'litellm', 'vllm').
        method (str): RAG method or prompt strategy (e.g., 'zero-shot', 'fid').
        stop_at_period (bool, optional): If True, truncate output at first period (default: False).
        **kwargs: Additional model-specific parameters, such as API keys or generation settings.

    Returns:
        BaseRAGModel: An instance of the selected model class, ready for text generation.

    Raises:
        ValueError: If an unsupported backend is specified.

    Notes:
        - Automatically initializes the required prompt generator.
        - For 'openai', expects 'api_keys' in kwargs.
        - For 'huggingface', loads tokenizer and model locally.
        - For 'fid', instantiates Fusion-in-Decoder model.
        - For 'litellm', expects 'api_key' in kwargs.
        - For 'vllm', passes all kwargs to VLLMModel.

    Example:
        ```python
        model = model_factory(
            model_name="meta-llama/Meta-Llama-3.1-8B",
            backend="huggingface",
            method="zero-shot",
            torch_dtype=torch.float16
        )
        ```
    """
    prompt_generator = PromptGenerator(model_type=model_name, method=method)
    if backend == "openai":
        return OpenAIModel(model_name, kwargs["api_keys"], prompt_generator,
                           base_url=kwargs.get("base_url", None))
    elif backend == "huggingface":
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name, **kwargs) 
        return HuggingFaceModel(model_name, tokenizer, model, prompt_generator, stop_at_period=stop_at_period)
    elif backend == "fid":
        return FiDModel(method, model_name, **kwargs)
    elif backend == "litellm":
        return LitellmModel(model_name, kwargs["api_key"], prompt_generator)
    elif backend == "vllm":
        return VLLMModel(model_name, prompt_generator, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")