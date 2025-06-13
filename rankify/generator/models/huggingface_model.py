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
        """Generate a response using Hugging Face's model and return only the answer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs.setdefault("max_new_tokens", 32)
        kwargs.setdefault("temperature", 0.3)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("num_return_sequences", 1)

        outputs = self.model.generate(**inputs, **kwargs)
        # Decode the full generated sequence
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated text to get only the answer
        answer = generated_text[len(prompt):].strip()
        # Optionally, take only the first line if the answer is multi-line
        answer = answer.split("\n")[0].strip()
        return answer