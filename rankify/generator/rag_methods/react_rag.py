from typing import List, Optional
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
import re

class ReActRAG(BaseRAGMethod):
    """
    **ReActRAG (Reason+Act RAG) Method** for Retrieval-Augmented Generation.

    Implements the ReAct technique, combining reasoning and action steps for open-domain question answering.
    The model alternates between generating reasoning steps and issuing search actions to a retriever, iteratively building up context and history until a final answer is produced.

    Attributes:
        model (BaseRAGModel): The RAG model instance used for generation.
        retriever: An external retriever instance for fetching new contexts.
        max_steps (int): Maximum number of reasoning/search steps per question (default: 20).
        max_contexts_per_search (int): Maximum number of contexts to add per search action (default: 3).
        use_internal_knowledge (bool): Whether to fallback to internal knowledge if no answer is found (default: True).

    Methods:
        __init__(model, retriever, max_steps=20, max_contexts_per_search=3, use_internal_knowledge=True, **kwargs):
            Initializes the ReActRAG method with the provided model and retriever.
        answer_questions(documents: List[Document], custom_prompt=None, **kwargs) -> List[str]:
            Answers questions for a list of documents using iterative reasoning and retrieval.

    Notes:
        - The method parses model outputs for "Search[query]" actions and "Final Answer:" statements.
        - Retrieved contexts are appended to the prompt and history for subsequent steps.
        - If no final answer is found, optionally falls back to internal knowledge or last output.
        - Suitable for complex questions requiring multi-step reasoning and dynamic retrieval.

    Example:
        ```python
        import torch
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.generator.generator import Generator
        from rankify.n_retreivers.retriever import Retriever
        question = Question("Who won the FIFA World Cup after Germany in 2014?")
        answers = Answer("")

        contexts = [
            Context(id=1, title="2014 FIFA World Cup", text="Germany won the FIFA World Cup in 2014, held in Brazil.", score=0.9),
            Context(id=2, title="FIFA World Cup History", text="The FIFA World Cup is held every four years, with different countries winning each time.", score=0.8),
            Context(id=3, title="World Cup Winners", text="The winners of the FIFA World Cup are celebrated globally.", score=0.7),
        ]

        docs = [Document(question=question, answers=answers, contexts=contexts)]

        # Initialize retriever (example: BM25, can be any retriever compatible with your pipeline)
        retriever = Retriever(method='bm25', n_docs=5, index_type='wiki')
        retrieved_documents = retriever.retrieve(docs)
        for i, doc in enumerate(retrieved_documents):
             print(f"\nDocument {i+1}:")
             print(doc)

        # Initialize Generator for ReAct RAG
        generator = Generator(
            method="react-rag",
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            backend="huggingface",
            torch_dtype=torch.float16,
            retriever=retriever,
            stop_at_period=True
        )

        # Generate answer
        generated_answers = generator.generate(test_docs)
        print(generated_answers)  # Expected output: ["France"]
        ```
    
    References:
        - Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models*  
          [Paper](https://arxiv.org/abs/2210.03629)
    """

    def __init__(self, model: BaseRAGModel, retriever, max_steps: int = 20, max_contexts_per_search: int = 3, use_internal_knowledge: bool = True, **kwargs):
        """
        Initialize the ReActRAG method.

        Args:
            model (BaseRAGModel): The RAG model instance used for generation.
            retriever: An external retriever instance for fetching new contexts.
            max_steps (int, optional): Maximum number of reasoning/search steps per question (default: 20).
            max_contexts_per_search (int, optional): Maximum number of contexts to add per search action (default: 3).
            use_internal_knowledge (bool, optional): Whether to fallback to internal knowledge if no answer is found (default: True).
        """
        super().__init__(model=model)
        self.retriever = retriever  # Pass a retriever instance
        self.max_steps = max_steps
        self.max_contexts_per_search = max_contexts_per_search
        self.use_internal_knowledge = use_internal_knowledge

    def answer_questions(self, documents: List[Document], custom_prompt: Optional[str] = None, **kwargs) -> List[str]:
        """
        Answer questions for a list of documents using the ReAct reasoning and retrieval loop.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.
            custom_prompt (str, optional): Custom prompt to override default prompt generation.
            **kwargs: Additional parameters for the model's generate method.

        Returns:
            List[str]: Answers generated for each document, using iterative reasoning and retrieval.

        Notes:
            - Iteratively builds up history and context by parsing model outputs for reasoning and search actions.
            - Stops when a "Final Answer:" is obtained or max_steps is reached.
            - If no final answer is found, optionally falls back to internal knowledge (if self.use_internal_knowledge=True) or last output.
        """
        answers = []
        for document in documents:
            question = document.question.question
            print("answering question: " + document.question.question)
            contexts = [context.text for context in document.contexts]
            history = []
            final_answer = None
            for step in range(self.max_steps):
                prompt = self.model.prompt_generator.generate_user_prompt(
                    question, contexts, custom_prompt=custom_prompt
                )
                # Add history to the prompt
                if history:
                    prompt += "\n" + "\n".join(history)
                output = self.model.generate(prompt=prompt, **kwargs)
                # Check for Final Answer
                final_match = re.search(r"Final Answer:\s*(.*)", output)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    break
                # Parse for Search action
                search_match = re.search(r"Search\[(.*?)\]", output)
                if search_match:
                    query = search_match.group(1)
                    # Use retriever to get new contexts (limit by max_contexts_per_search)
                    retrieved_docs = self.retriever.retrieve([Document(question=document.question, answers=document.answers, contexts=[])])
                    if retrieved_docs and retrieved_docs[0].contexts:
                        obs_texts = []
                        for ctx in retrieved_docs[0].contexts[:self.max_contexts_per_search]:
                            obs_texts.append(ctx.text)
                            contexts.append(ctx.text)
                        obs = "Observation: " + " ".join(obs_texts)
                        history.append(f"Search[{query}]\n{obs}")
                else:
                    # If neither Search nor Final Answer, add reasoning to history
                    history.append(output)
            # If final answer found, use it
            if final_answer is not None:
                answers.append(final_answer)
            else:
                # Fallback to internal knowledge if flag is set
                if self.use_internal_knowledge:
                    internal_prompt = (
                        "You are a knowledgeable assistant. Answer the following question using only your internal knowledge.\n"
                        f"Question: {question}\n"
                        "Answer:"
                    )
                    internal_output = self.model.generate(prompt=internal_prompt, **kwargs)
                    answers.append(internal_output)
                else:
                    # Else answer with last output
                    answers.append(output)
        return answers