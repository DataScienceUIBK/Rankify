from typing import List, Optional

from rankify.generator.prompt_template import DEFAULT_PROMPTS, PromptTemplate

class PromptGenerator:
    """
    **PromptGenerator** for Retrieval-Augmented Generation (RAG).

    This class manages prompt construction for different RAG methods and model types in Rankify.
    It selects and formats prompt templates based on the specified method and model, enabling flexible and consistent prompt generation.

    Attributes:
        method (str): The RAG method or strategy (e.g., "basic-rag", "chain-of-thought-rag").
        prompt_template (PromptTemplate): The prompt template used for formatting prompts.
        model_type (str): The type of model (e.g., "huggingface", "openai").

    Methods:
        generate_user_prompt(question, contexts, custom_prompt=None) -> str:
            Generates a user prompt by formatting the question and contexts with the selected template.
        _select_template(method) -> PromptTemplate:
            Selects the appropriate prompt template for the given method.

    Notes:
        - Supports custom prompt templates via the `custom_prompt` argument.
        - If no contexts are provided, generates prompts with only the question.
        - Ensures consistent prompt formatting across different RAG methods and models.
        - Automatically selects a default template if none is specified.
    """
    def __init__(self, method: str, model_type: str, prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the PromptGenerator.

        Args:
            method (str): The RAG method or strategy (e.g., "basic-rag", "chain-of-thought-rag").
            model_type (str): The type of model (e.g., "huggingface", "openai").
            prompt_template (PromptTemplate, optional): Custom prompt template to use. If None, selects based on method.

        Notes:
            - If no custom template is provided, selects a default template for the specified method.
            - Stores the method, model type, and selected prompt template for prompt generation.
        """
        self.method = method
        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = self._select_template(method)

    def _select_template(self, method: str) -> PromptTemplate:
        """
        Selects the appropriate prompt template for the given RAG method.

        Args:
            method (str): The RAG method or strategy.

        Returns:
            PromptTemplate: The selected prompt template.

        Notes:
            - If the method is not recognized, defaults to BASIC_RAG template.
        """
        method = method.lower()
        if method in PromptTemplate._value2member_map_:
            return PromptTemplate(method)
        return PromptTemplate.BASIC_RAG

    def generate_user_prompt(self, question: str, contexts: List[str], custom_prompt: Optional[str] = None) -> str:
        """
        Generates a user prompt by formatting the question and contexts with the selected template.
    
        Args:
            question (str): The question to be answered.
            contexts (List[str]): List of context passages to include in the prompt.
            custom_prompt (str, optional): Custom prompt template string. If provided, overrides the default template.
    
        Returns:
            str: The formatted prompt string.
    
        Notes:
            - If no contexts are provided, generates a prompt with only the question.
            - Custom prompt templates can use `{question}` and `{contexts}` placeholders.
            - Ensures consistent prompt formatting for all RAG methods and models.
        """
        if contexts is None:
            contexts = []
        context_str = "\n".join(contexts)
        if custom_prompt:
            return custom_prompt.format(question=question, contexts=context_str)
        template = DEFAULT_PROMPTS[self.prompt_template]
        return template.format(question=question, contexts=context_str)