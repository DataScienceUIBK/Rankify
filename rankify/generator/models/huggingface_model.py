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

    def generate(self, prompt: str, **kwargs):
        """Generate a response using Hugging Face's model and return only the answer(s)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        kwargs.setdefault("max_new_tokens", 64)
        kwargs.setdefault("do_sample", True)
        kwargs.setdefault("num_return_sequences", 1)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("top_p", 1.0)
        print(f"Generating with prompt: {prompt}")

        outputs = self.model.generate(**inputs, **kwargs)

        def stop_at_first_period(text):
           idx = text.find(".")
           if idx != -1:
               return text[:idx+1].strip()
           return text.strip()


    # If multiple sequences requested, decode all
        if kwargs.get("num_return_sequences", 1) > 1:
            answers = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                answer = generated_text[len(prompt):].strip()
                answer = answer.split("\n")[0].strip()
                answer = stop_at_first_period(answer)
                answers.append(answer)
            return answers
        else:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()
            answer = answer.split("\n")[0].strip()
            answer = stop_at_first_period(answer)
            return answer