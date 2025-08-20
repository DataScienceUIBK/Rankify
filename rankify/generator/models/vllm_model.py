from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator
from vllm import LLM

class VLLMModel(BaseRAGModel):
    """
    **vLLM Model** for Retrieval-Augmented Generation (RAG).

    This class integrates vLLM's API for text generation in a RAG pipeline. 
    It uses the vLLM library to generate responses based on input prompts.

    Attributes:
        model_name (str): Name of the vLLM model.
        prompt_generator (PromptGenerator): Instance for generating prompts.
        client (LLM): Client for interacting with the vLLM library.

    Notes:
        - This model uses vLLM's library for text generation.
        - It supports additional parameters like `max_tokens` and `temperature`.
        - **Specialty:** vLLM requires a `sampling_params` dictionary to control decoding and sampling behavior, 
          allowing fine-grained control over generation (e.g., `max_tokens`, `temperature`, `top_p`, etc.).
    """
    def __init__(self, model_name: str, prompt_generator: PromptGenerator, **kwargs):
        """
        Initialize the VLLMModel with the vLLM client.

        :param model_name: Name of the vLLM model.
        :param prompt_generator: Instance of PromptGenerator for generating prompts.
        :param device: Device to run the model on (default: "cuda").
        """
        self.model_name = model_name
        self.prompt_generator = prompt_generator
        self.llm = LLM(model=model_name, **kwargs)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using vLLM's API.

        Args:
            prompt (str): The input prompt for the model.
            sampling_params (dict, required in kwargs): Dictionary of sampling parameters for vLLM generation, such as:
                - max_tokens (int): Maximum number of tokens to generate.
                - temperature (float): Sampling temperature.
                - top_p (float): Nucleus sampling parameter.
                - other vLLM-supported generation options.
            **kwargs: Additional parameters for the vLLM API call.

        Returns:
            str: The generated response as a string.

        Notes:
            - `sampling_params` controls the decoding and sampling behavior of the vLLM model.
            - Typical keys include `max_tokens`, `temperature`, `top_p`, etc.
            - See vLLM documentation for all supported sampling parameters.

        Example:
            ```python
            answer = model.generate(
                "What is the capital of France?",
                sampling_params={"max_tokens": 64, "temperature": 0.7, "top_p": 0.9}
            )
        ```
        """
        # Call the vLLM API using the LLM client
        response = self.llm.generate(prompt, kwargs["sampling_params"])
        
        if response and response[0].outputs:
            # Extract the generated text from the response
            generated_text = response[0].outputs[0].text.strip()
        else:
            generated_text = ""

        return generated_text