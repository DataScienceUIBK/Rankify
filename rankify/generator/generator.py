from rankify.generator.models.model_factory import model_factory
from rankify.generator.rag_methods.basic_rag import BasicRAG
from rankify.generator.rag_methods.chain_of_thought_rag import ChainOfThoughtRAG
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.rag_methods.fid_rag_method import FiDRAGMethod
from rankify.generator.rag_methods.in_context_ralm_rag import InContextRALMRAG
from rankify.utils.generator.generator_models import RAG_METHODS


class Generator:
    """
    **Generator Interface** for Retrieval-Augmented Generation (RAG) techniques.

    This class serves as a unified wrapper for all RAG methods and underlying model endpoints in Rankify.
    It dynamically initializes the appropriate model and RAG method, using the model_factory method,
    based on the provided `method`, `model_name`, and `backend`.

    Attributes:
        rag_method (BaseRAGMethod): The initialized RAG method instance, combining the selected model and technique.

    Raises:
        ValueError: If the specified `method` is not available in `RAG_METHODS`.

    Example:
        ```python
        from rankify.generator.generator import Generator
        from rankify.dataset.dataset import Document, Question

        # Example: Fusion-in-Decoder (FiD)
        generator = Generator(method="fid", model_name="nq_reader_base", backend="fid")
        documents = [Document(question=Question("Who wrote Hamlet?"))]
        generated_answers = generator.generate(documents)
        print(generated_answers[0])  # Output: "William Shakespeare wrote Hamlet in the early 1600s."

        # Example: Chain-of-Thought RAG
        generator = Generator(method="chain-of-thought-rag", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", backend="huggingface")
        documents = [Document(question=Question("What is the capital of France?"))]
        print(generator.generate(documents)[0])  # Output: "Paris"
        ```

    Notes:
        - The generator puts together both the model endpoint (e.g., OpenAI, HuggingFace, FiD, vLLM) and the RAG method (e.g., basic, chain-of-thought, self-consistency).
        - All configuration (model, backend, method, and additional kwargs) is passed to the appropriate factory and method class.
        - Use `generate()` to obtain answers for a batch of documents, calling `answer_questions()` of the specified RAG method.
    """

    def __init__(self, method: str, model_name: str, backend:str = "huggingface", **kwargs):
        """
        Initializes the selected RAG method and model.

        Args:
            method (str): The RAG technique (e.g., "fid", "chain-of-thought-rag", "self-consistency-rag").
            model_name (str): The specific model name (e.g., "nq_reader_base", "meta-llama/Meta-Llama-3.1-8B-Instruct").
            backend (str): The model backend ("huggingface", "openai", "fid", "vllm", etc.).
            **kwargs: Additional parameters for model and method initialization.

        Raises:
            ValueError: If the specified `method` is not available in `RAG_METHODS`.
        """
        if method not in RAG_METHODS:
            raise ValueError(f"Generator method {method} is not supported.")
        
        # Initialize the generator model based on the method
        model = model_factory(model_name=model_name, backend=backend, method=method, **kwargs)
        
        # get the class for the specified method
        rag_method_class = RAG_METHODS[method]

        # Initialize the generator with the model and any additional parameters
        self.rag_method = rag_method_class(model, **kwargs)


    def generate(self, documents, custom_prompt=None,**kwargs):
        """
        Generates answers for a batch of input documents using the selected RAG method.

        Args:
            documents (List[Document]): A list of `Document` objects containing queries and retrieved contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the RAG method's answer logic or model generation.

        Returns:
            List[str]: A list of generated answers, one for each document.

        """
        return self.rag_method.answer_questions(documents, custom_prompt, **kwargs)
