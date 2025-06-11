from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator

class HuggingFaceModel(BaseRAGModel):
    """
    **Hugging Face Model** for Retrieval-Augmented Generation (RAG).

    This class integrates Hugging Face's pretrained models for text generation in a RAG pipeline. 
    It uses the Hugging Face Transformers library for tokenization and model inference.

    Attributes:
        model_name (str): Name of the Hugging Face model.
        tokenizer: Tokenizer instance for encoding input text.
        model: Pretrained Hugging Face model for text generation.
        prompt_generator (PromptGenerator): Instance for generating prompts.

    Notes:
        - This model uses Hugging Face's Transformers library for text generation.
        - Default generation parameters like `max_length` and `temperature` can be overridden.
    """

    def __init__(self, model_name: str, tokenizer, model, prompt_generator: PromptGenerator):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_generator = prompt_generator

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Hugging Face's model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # ensure inputs are on the same device as the model
        # define generation parameters defaults TODO: should be excluded into config
        kwargs.setdefault("max_new_tokens", 32)
        kwargs.setdefault("temperature", 0.3)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("num_return_sequences", 1)
        
        outputs = self.model.generate(**inputs, **kwargs)
        #return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_ids = inputs["input_ids"]
        return self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:])[0]