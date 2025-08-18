import os
import json
from typing import List
import torch
from tqdm import tqdm
from rankify.dataset.dataset import Document
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from transformers import AutoConfig

class InContextRALMRAG(BaseRAGMethod):
    """
    **In-Context Retrieval-Augmented Language Model (RALM) Generator**.


    This class implements **In-Context Retrieval-Augmented Language Models (RALM)** for **context-aware text generation**.
    It **integrates retrieved passages** into the input prompt for **few-shot learning**, improving response quality.

    Attributes:
        device (str): Device used for inference (`'cuda'` or `'cpu'`).
        model_name (str): Name of the pre-trained RALM model.
        cache_dir (str): Directory to store downloaded models.
        num_docs (int): Number of retrieved contexts to include in the prompt (default: `1`).
        max_length (int): Maximum number of tokens the model can process.
        max_tokens_to_generate (int): Maximum number of tokens the model generates in response.

    References:
        - **Ram et al.** *In-Context Retrieval-Augmented Language Models*  
          [Paper](https://arxiv.org/abs/2302.00083)

    See Also:
        - `BaseGenerator`: Parent class for InContextRALMGenerator.
        - `Few-Shot Learning`: This model **leverages retrieved passages** for in-context QA.

    Example:
        ```python
        from rankify.generator.generator import Generator
        from rankify.dataset.dataset import Document, Question, Answer, Context

        generator = Generator(method="in-context-ralm", model_name="meta-llama/Llama-3.1-8B")
        documents = [
            Document(
                question=Question("Who discovered gravity?"),
                answers=Answer(["Isaac Newton"]),
                contexts=[
                    Context(title="Gravity", text="Isaac Newton formulated the laws of gravity.", score=1.0)
                ]
            )
        ]

        generated_answers = generator.generate(documents)
        print(generated_answers[0])  # 'Isaac Newton discovered gravity in the late 17th century.'
        ```

    Notes:
        - **RALM dynamically integrates retrieved passages** to guide response generation.
        - **Optimized for retrieval-augmented question answering** (RAG).
        - Supports **meta-llama models (Llama-3.1-8B, Llama-2, etc.)**.
    """

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, model:BaseRAGModel, **kwargs):
        """
        Initializes the RALM Generator.

        Args:
            method (str): The generator type (`"in-context-ralm"`).
            model_name (str): The name of the pre-trained RALM model (e.g., `"meta-llama/Llama-3.1-8B"`).
            **kwargs: Additional parameters for model configuration.

        Example:
            ```python
            generator = InContextRALMGenerator(method="in-context-ralm", model_name="meta-llama/Llama-3.1-8B")
            ```
        """
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = self.CACHE_DIR
        self.num_docs = kwargs.get("num_docs", 1)  # Default: 1 supporting document



        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(model.model_name)

        # Set generation parameters
        self.max_length = self.config.n_positions if hasattr(self.config, "n_positions") else self.config.max_position_embeddings
        self.max_tokens_to_generate = kwargs.get("max_tokens_to_generate", 10)

    def _build_qa_prompt(self, example):
        """
        Constructs the **QA prompt** for in-context learning.

        Args:
            example (dict): A dictionary containing question and retrieved contexts.

        Returns:
            str: The constructed prompt with retrieved passages.

        Example:
            ```python
            example = {
                "question": "Who invented the light bulb?",
                "ctxs": [{"title": "Edison", "text": "Thomas Edison patented the light bulb."}]
            }
            prompt = generator._build_qa_prompt(example)
            ```
        """
        question = example["question"]
        question = question[0].lower() + question[1:] if not question.endswith("?") else question

        if self.num_docs == 0:
            return f"Answer these questions:\nQ: {question}\nA:"
        
        docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:self.num_docs]])
        return f"{docs_text}\n\nBased on these texts, answer these questions in the shortest, most precise way possible. I need a factual answer since I want to compare your answer.:\nQ: {question}\nA:"

    def _prepare_dataloader(self, documents: list[Document]):
        """
        Converts `Document` objects into a dataset **formatted for RALM**.

        Args:
            documents (List[Document]): A list of documents with **queries and retrieved contexts**.

        Returns:
            list: A list of dictionaries formatted for in-context learning.

        Example:
            ```python
            dataset = generator._prepare_dataloader(documents)
            ```
        """
        examples = []
        for doc in documents:
            example = {
                "question": doc.question.question,
                "answers": doc.answers.answers,
                "ctxs": [{"title": ctx.title, "text": ctx.text, "score": ctx.score} for ctx in doc.contexts]
            }
            examples.append(example)

        return examples

    def answer_questions(self, documents: List[Document], custom_prompt=None) -> List[str]:
        """
        Generates answers for **a list of documents** using RALM.

        Args:
            documents (List[Document]): A list of documents with **queries and retrieved contexts**.

        Returns:
            List[str]: A list of generated answers.

        Example:
            ```python
            generator = InContextRALMGenerator(method="in-context-ralm", model_name="meta-llama/Llama-3.1-8B")
            documents = [Document(question=Question("Who wrote Hamlet?"))]
            print(generator.generate(documents))  # ['William Shakespeare wrote Hamlet in the early 1600s.']
            ```
        """
        eval_dataset = self._prepare_dataloader(documents)

        results = []
        for example in tqdm(eval_dataset, desc="Answering questions", unit="q"):
            prompt = self._build_qa_prompt(example)

            tokenized_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized_input.input_ids.to(self.device)
            attention_mask = tokenized_input.attention_mask.to(self.device)  # Extract attention mask

            if input_ids.shape[-1] > self.max_length - self.max_tokens_to_generate:
                input_ids = input_ids[..., -(self.max_length - self.max_tokens_to_generate):]
                attention_mask = attention_mask[..., -(self.max_length - self.max_tokens_to_generate):]

            with torch.no_grad():
                outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=self.max_tokens_to_generate,  pad_token_id=self.tokenizer.pad_token_id )

            # Extract generated text
            generation_str = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            answer = generation_str[len(prompt):].split("\n")[0]

            results.append(answer)

        return results

