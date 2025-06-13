from rankify.generator.models.model_factory import model_factory
from rankify.generator.rag_methods.basic_rag import BasicRAG
from rankify.generator.rag_methods.chain_of_thought_rag import ChainOfThoughtRAG
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.rag_methods.fid_rag_method import FiDRAGMethod
from rankify.generator.rag_methods.in_context_ralm_rag import InContextRALMRAG
from rankify.utils.generator.generator_models import RAG_METHODS


class Generator:
    """
    **Generator Interface** for handling different **text generation models**.

    This class serves as a **wrapper** for multiple **answer generation models** (e.g., **FiD, GPT, T5, RALM**).
    It dynamically initializes the appropriate model based on the provided `method`.


    Attributes:
        generator (BaseGenerator): The initialized generator instance based on the selected method.

    Raises:
        ValueError: If the specified `method` is not in `GENERATOR_MODELS`.

    Example:
        ```python
        from rankify.generator.generator import Generator
        from rankify.dataset.dataset import Document, Question

        generator = Generator(method="fid", model_name="nq_reader_base")
        documents = [Document(question=Question("Who discovered gravity?"))]
        generated_answers = generator.generate(documents)

        print(generated_answers[0])
        # Output: "Isaac Newton discovered gravity in the late 17th century."
        ```

        **Using In-Context Retrieval-Augmented Language Model (RALM) Generator**:
        ```python
        generator = Generator(method="in-context-ralm", model_name="gpt-4-ralm")
        documents = [Document(question=Question("What is deep learning?"))]
        print(generator.generate(documents)[0])
        # Output: "Deep learning is a subset of machine learning that uses neural networks..."
        ```
    """

    def __init__(self, method: str, model_name: str, backend:str = "huggingface", **kwargs):
        """
        Initializes the selected **generator model**.

        Args:
            method (str): The generator type (`"fid"`, `"in-context-ralm"`, etc.).
            model_name (str): The specific model name (e.g., `"nq_reader_base"`, `"gpt-4-ralm"`).
            kwargs (dict): Additional parameters for model initialization.

        Raises:
            ValueError: If the specified `method` is not available in `GENERATOR_MODELS`.
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
        Generates answers based on the **input documents**.

        Args:
            documents (List[Document]): A list of `Document` objects containing queries and retrieved contexts.

        Returns:
            List[str]: A list of generated **answers**, one for each document.

        Example:
            ```python
            generator = Generator(method="fid", model_name="nq_reader_base")
            documents = [Document(question=Question("Who wrote Hamlet?"))]
            print(generator.generate(documents))
            # Output: ['William Shakespeare wrote Hamlet in the early 1600s.']
            ```
        """
        return self.rag_method.answer_questions(documents, custom_prompt, **kwargs)
