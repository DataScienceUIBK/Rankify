from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator
from vllm import LLM, SamplingParams

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

        :param prompt: The input prompt for the model.
        :param kwargs: Additional parameters for the vLLM API call.
        :return: The generated response as a string.
        """
        # Todo: use this later -> Generate the prompt using the prompt generator
        # full_prompt = self.prompt_generator.generate_prompt(prompt)

        # Call the vLLM API using the LLM client
        response = self.llm.generate(prompt, kwargs["sampling_params"])
        
        return response