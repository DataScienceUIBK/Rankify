
#retriever.py
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.utils.pre_defined_methods_retrievers import METHOD_MAP

from typing import List

class Retriever:
    """
    Implements a **retriever interface** that selects relevant **documents** based on predefined retrieval methods.

    The class provides an interface for **various retrieval methods** (e.g., BM25, Dense Retrieval),
    allowing **customization of retrieval parameters**.

    Attributes:
        method (str): The **name of the retrieval method** (must be in `METHOD_MAP`).
        n_docs (int): The **number of documents** to retrieve per query (default: `10`).
        kwargs (dict): Additional retrieval parameters.

    Raises:
        ValueError: If the **specified retrieval method** is **not supported** in `METHOD_MAP`.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.retrieval import Retriever

        # Define a query
        question = Question("What are the benefits of artificial intelligence?")
        answers = Answer(["AI improves efficiency and automates tasks."])
        contexts = [
            Context(text="Artificial intelligence enhances automation.", id=0),
            Context(text="AI is used in healthcare and finance.", id=1),
            Context(text="Machine learning is a subset of AI.", id=2),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        # Initialize Retriever (using BM25 or Dense Retrieval)
        retriever = Retriever(method="bm25", n_docs=5)
        retrieved_documents = retriever.retrieve([document])

        # Print retrieved documents
        print("Retrieved Documents:")
        for doc in retrieved_documents:
            for ctx in doc.contexts:
                print(ctx.text)
        ```
    """
    def __init__(self, method: str = None, n_docs: int = 10, **kwargs):
        """
        Initializes the Retriever instance.

        Args:
            method (str, optional): The **name of the retrieval method** 
                (must be in `METHOD_MAP`, default: `None`).
            n_docs (int, optional): The **number of top documents to retrieve** (default: `10`).
            kwargs (dict): Additional parameters for retrieval.

        Raises:
            ValueError: If the retrieval `method` is **not found** in `METHOD_MAP`.
        """
        self.method = method
        self.n_docs = n_docs
        self.kwargs = kwargs
        self.retriever = self.initialize()

    def initialize(self):
        """
        Initializes the retrieval method.

        Returns:
            object: The **retriever instance** corresponding to `self.method`.

        Raises:
            ValueError: If the **retrieval method is not supported**.
        """
        if self.method in METHOD_MAP:
            return METHOD_MAP[self.method](n_docs=self.n_docs, **self.kwargs)
        else:
            raise ValueError(f"Retrieving method {self.method} is not supported.")

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **relevant documents** for a given set of **queries**.

        Args:
            documents (List[Document]): A list of **Document** instances containing queries.

        Returns:
            List[Document]: A **list of retrieved documents**, each containing:
                - The **original query** (`Question`)
                - **Retrieved contexts** (`Context`)
                - (Optional) Corresponding **answers** (`Answer`).
        """
        return self.retriever.retrieve(documents)
