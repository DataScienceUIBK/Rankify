from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator
from rankify.utils.api.litellmclient import LitellmClient

class LitellmModel(BaseRAGModel):
    """
    **LiteLLM Model** for Retrieval-Augmented Generation (RAG).

    This class integrates LiteLLM's API for text generation in a RAG pipeline. 
    It uses the LiteLLM API to generate responses based on input prompts.

    Attributes:
        model_name (str): Name of the LiteLLM model.
        prompt_generator (PromptGenerator): Instance for generating prompts.
        client (LitellmClient): Client for interacting with the LiteLLM API.

    Notes:
        - This model uses LiteLLM's API for text generation.
        - It supports additional parameters like `max_tokens` and `temperature`.
    """
    def __init__(self, model_name: str, api_key: str, prompt_generator: PromptGenerator):
        """
        Initialize the LitellmModel with the LitellmClient.

        :param model_name: Name of the LiteLLM model.
        :param api_key: API key for LiteLLM.
        :param prompt_generator: Instance of PromptGenerator for generating prompts.
        """
        self.model_name = model_name
        self.prompt_generator = prompt_generator
        
        self.client = LitellmClient(keys=api_key)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using LiteLLM's API.

        Args:
            prompt (str): The input prompt for the model.
            **kwargs: Additional parameters for the LiteLLM API call, such as:
                - model (str): Model name to use (default: self.model_name).
                - max_tokens (int): Maximum number of tokens to generate (default: 128).
                - temperature (float): Sampling temperature (default: 0.7).
    
        Returns:
            str: The generated response as a string.
    
        Notes:
            - Default parameters: model=self.model_name, max_tokens=128, temperature=0.7.
            - All generation parameters can be overridden via `kwargs`.
    
        Example:
            ```python
            answer = model.generate("What is the capital of France?", max_tokens=64, temperature=0.5)
            ```
        """

        # Set default parameters for the LiteLLM API call
        kwargs.setdefault("model", self.model_name)
        kwargs.setdefault("max_tokens", 128)
        kwargs.setdefault("temperature", 0.7)

        # Call the LiteLLM API using the LitellmClient
        response = self.client.chat(messages=[{"role": "user", "content": prompt}], return_text=True, **kwargs)
        return response