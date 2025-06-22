import numpy as np
from typing import List
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from .contriever_retriever import ContrieverRetriever
from rankify.utils.retrievers.hyde import Promptor, OpenAIGenerator
from rankify.dataset.dataset import Document, Context


class HydeRetriever(BaseRetriever):
    """
    Hypothetical Document Embedding (HyDE) Retriever implementation.
    
    HyDE enhances document retrieval by generating hypothetical documents using an LLM
    and averaging their embeddings with the query embedding for improved search performance.
    
    The retrieval process:
    1. Generate hypothetical documents using an LLM based on the query
    2. Encode both the original query and generated documents
    3. Average the embeddings to obtain a refined query representation
    4. Retrieve top documents using the averaged embedding
    
    References:
        - Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. (2022): 
          Precise Zero-Shot Dense Retrieval without Relevance Labels.
          https://arxiv.org/abs/2212.10496
    """
    
    def __init__(self, model: str = "facebook/contriever-msmarco", index_type: str = "wiki", 
                 device: str = "cuda", api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.index_type = index_type
        self.device = device
        self.api_key = api_key
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Get configuration
        config = self._get_config()
        
        # HyDE-specific parameters
        self.task = kwargs.get("task", config.get("task", "web search"))
        self.llm_model = kwargs.get("llm_model", config.get("llm_model", "gpt-3.5-turbo-0125"))
        self.base_url = kwargs.get("base_url", config.get("base_url", None))
        self.num_generated_docs = kwargs.get("num_generated_docs", config.get("num_generated_docs", 1))
        self.max_token_generated_docs = kwargs.get("max_token_generated_docs", config.get("max_token_generated_docs", 512))
        self.temperature = kwargs.get("temperature", config.get("temperature", 0.7))
        
        # Validate API key if required
        if config.get("requires_api_key", True) and not self.api_key:
            raise ValueError("API key is required for HyDE retriever with LLM generation")
        
        # Initialize components
        self.tokenizer = SimpleTokenizer()
        self.promptor = Promptor(self.task)
        self.generator = OpenAIGenerator(
            model_name=self.llm_model,
            api_key=self.api_key,
            base_url=self.base_url,
            n=self.num_generated_docs,
            max_tokens=self.max_token_generated_docs,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['\n\n\n'],
            wait_till_success=False
        )
        
        # Initialize the base retriever (Contriever by default)
        base_model = kwargs.get("base_model", config.get("base_model", model))
        base_index_type = kwargs.get("base_index_type", config.get("base_index_type", index_type))
        
        self.contriever = ContrieverRetriever(
            model=base_model,
            n_docs=self.n_docs,
            batch_size=self.batch_size,
            device=device,
            index_type=base_index_type
        )
        
        # Set searcher reference
        self.searcher = self.contriever.index
    
    def _get_config(self):
        """Get configuration for HyDE retriever."""
        hyde_configs = self.index_manager.index_configs.get("hyde", {})
        return hyde_configs.get(self.index_type, {
            "task": "web search",
            "llm_model": "gpt-3.5-turbo-0125",
            "num_generated_docs": 1,
            "max_token_generated_docs": 512,
            "temperature": 0.7,
            "requires_api_key": True
        })
    
    def _initialize_searcher(self):
        """Initialize searcher - handled through ContrieverRetriever."""
        return None
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve contexts using HyDE-based query expansion.
        
        The process involves:
        1. Generating hypothetical documents using an LLM
        2. Encoding both the original query and generated documents
        3. Averaging the embeddings to obtain a refined query representation
        4. Retrieving top documents using FAISS-based search
        
        Args:
            documents: List of Document objects containing queries
            
        Returns:
            List of Document objects with retrieved contexts
        """
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="docs")):
            contexts = []
            
            # Build prompt and generate hypothetical documents
            prompt = self.promptor.build_prompt(document.question.question.replace("?", ""))
            hypothesis_documents = self.generator.generate(prompt)
            
            # Embed query and hypothetical documents
            all_emb_c = []
            for c in [prompt] + hypothesis_documents:
                c_emb = self.contriever._embed_queries([c])
                all_emb_c.append(c_emb.squeeze(0))
            
            # Average embeddings
            all_emb_c = np.array(all_emb_c)
            avg_emb_c = np.mean(all_emb_c, axis=0)
            hype_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
            
            # Search using averaged embedding
            top_ids_and_scores = self.contriever.index.search_knn(
                hype_vector, 
                self.n_docs, 
                index_batch_size=self.batch_size
            )
            doc_ids, scores = top_ids_and_scores[0]
            
            # Create contexts from results
            for doc_id, score in zip(doc_ids, scores):
                try:
                    passage = self.contriever.passage_id_map[int(doc_id)]
                    context = Context(
                        id=int(doc_id),
                        title=passage["title"],
                        text=passage.get("text", passage.get("contents", "")),
                        score=score,
                        has_answer=has_answers(
                            passage.get("text", passage.get("contents", "")), 
                            document.answers.answers, 
                            self.tokenizer, 
                            regex=False
                        )
                    )
                    contexts.append(context)
                except (IndexError, KeyError):
                    # Log or handle the error, and continue with the next passage
                    continue
            
            document.contexts = contexts
            
        return documents