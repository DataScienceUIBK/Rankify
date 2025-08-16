import torch
from rankify.generator.models.fid_model import FiDModel
from rankify.generator.models.litellm_model import LitellmModel
from rankify.generator.models.openai_model import OpenAIModel
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.models.vllm_model import VLLMModel
from rankify.generator.prompt_generator import PromptGenerator

from transformers import AutoTokenizer, AutoModelForCausalLM

from rankify.utils.generator.huggingface_models.model_utils import load_model, load_tokenizer

def model_factory(model_name: str, backend: str, method: str, use_litellm=False, stop_at_period=False, **kwargs) -> BaseRAGModel:
    prompt_generator = PromptGenerator(model_type=model_name, method=method)
    if backend == "openai":
        return OpenAIModel(model_name, kwargs["api_keys"], prompt_generator)
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