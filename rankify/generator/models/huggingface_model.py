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
        stop_at_period (bool): If True, cuts generated answer at the first period.

    Notes:
        - This model uses Hugging Face's Transformers library for text generation.
        - Default generation parameters like `max_length` and `temperature` can be overridden.
    """

    def __init__(self, model_name: str, tokenizer, model, prompt_generator: PromptGenerator, stop_at_period: bool = False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_generator = prompt_generator
        self.stop_at_period = stop_at_period

    def generate(self, prompt: str, **kwargs):
        """
        Generates a response using the Hugging Face model and returns the answer(s).

        Args:
            prompt (str): The input prompt for generation.
            **kwargs: Optional generation parameters, such as:
                - max_new_tokens (int): Maximum number of new tokens to generate (default: 64).
                - do_sample (bool): Whether to use sampling (default: True).
                - num_return_sequences (int): Number of answers to generate (default: 1).
                - eos_token_id (int): End-of-sequence token ID (default: tokenizer.eos_token_id).
                - pad_token_id (int): Padding token ID (default: tokenizer.eos_token_id).
                - temperature (float): Sampling temperature (default: 0.1).
                - top_p (float): Nucleus sampling parameter (default: 1.0).

        Returns:
            str or List[str]: The generated answer(s). If `num_return_sequences` > 1, returns a list of answers.

        Notes:
            - The answer is post-processed to remove the prompt and extra whitespace.
            - If `stop_at_period` is True, the answer is truncated at the first period.
            - All generation parameters can be overridden via `kwargs`.

        Example:
            ```python
            answer = model.generate("What is the capital of France?", max_new_tokens=32)
            ```
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        kwargs.setdefault("max_new_tokens", 64)
        kwargs.setdefault("do_sample", True)
        kwargs.setdefault("num_return_sequences", 1)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("top_p", 1.0)
        
        outputs = self.model.generate(**inputs, **kwargs)

        def clean_answer(text):
            answer = text[len(prompt):].strip()
            answer = answer.split("\n")[0].strip()
            if self.stop_at_period:
                idx = answer.find(".")
                if idx != -1:
                    return answer[:idx+1].strip()
            return answer

        if kwargs.get("num_return_sequences", 1) > 1:
            return [clean_answer(self.tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
        else:
            return clean_answer(self.tokenizer.decode(outputs[0], skip_special_tokens=True))