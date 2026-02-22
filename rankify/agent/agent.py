"""
RankifyAgent - Conversational AI Assistant for Model Selection.

Provides both programmatic and conversational interfaces for 
intelligent model recommendation in Rankify.
"""

import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from rankify.agent.model_registry import (
    ModelMetadata,
    TaskType,
    RETRIEVER_REGISTRY,
    RERANKER_REGISTRY,
    RAG_METHOD_REGISTRY,
)
from rankify.agent.recommender import (
    RankifyRecommender,
    RecommendationResult,
    PipelineConfig,
)


@dataclass
class AgentResponse:
    """Response from the agent."""
    message: str
    recommendation: Optional[RecommendationResult] = None
    code_snippet: Optional[str] = None
    

class RankifyAgent:
    """
    Conversational AI agent for Rankify model selection.
    
    Supports multiple LLM backends:
    - Azure OpenAI
    - OpenAI
    - LiteLLM (100+ providers)
    - Local LLMs via transformers
    
    Example:
        ```python
        agent = RankifyAgent(backend="azure")
        response = agent.chat("I need to build a QA system with no GPU")
        print(response.message)
        print(response.recommendation)
        ```
    """
    
    SYSTEM_PROMPT = """You are RankifyAgent, an expert AI assistant for the Rankify framework.
Rankify is a comprehensive Python toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation (RAG).

Your job is to help users select the best models for their use case. You have access to:
- **Sparse Retrievers**: BM25 (fast, no GPU, exact match)
- **Dense Retrievers**: DPR, ANCE, BGE, ColBERT, Contriever
- **Diver Dense Retrievers** (method="diver-dense"): Many bi-encoder and LLM-based variants selectable via model_id
- **Reasoning-Augmented Retrievers** (SOTA on BRIGHT benchmark): ReasonIR-8B, ReasonEmbed, BGE-Reasoner-Embed
- **Online Retriever**: Web search via APIs (real-time data)
- **HyDE**: Hypothetical Document Embedding for complex queries
- 23 reranking methods (MonoT5, FlashRank, RankGPT, InRanker, ColBERT, API rerankers, etc.)
- 7 RAG methods (Basic RAG, Chain-of-Thought, Self-Consistency, ReAct, FiD, etc.)

**Diver Dense Retriever Guide (method="diver-dense"):**
Valid model_ids (must have corpus_path):
- `bge` → BAAI/bge-large-en-v1.5
- `sbert` → sentence-transformers/all-mpnet-base-v2
- `nomic` → nomic-ai/nomic-embed-text-v1
- `diver` → AQ-MedAI/Diver-Retriever-4B (flagship diverse evidence model)
- `inst-l` → hkunlp/instructor-large
- `inst-xl` → hkunlp/instructor-xl
- `e5` → intfloat/e5-mistral-7b-instruct (LLM-based)
- `sf` → Salesforce/SFR-Embedding-Mistral (LLM-based)
- `rader` → Raderspace/RaDeR_Qwen_25_7B (reasoning-augmented)
- `grit` → GritLM/GritLM-7B (generative representation)
- `m2` → togethercomputer/m2-bert-80M-32k-retrieval (long context, 32k)
- `contriever` → facebook/contriever-msmarco

Example: `Retriever(method="diver-dense", model_id="diver", corpus_path="data/corpus.jsonl", n_docs=10)`

**Reasoning-Augmented Retrievers:**
- `Retriever(method="reasonir", corpus_path=...)` → reasonir/ReasonIR-8B (SOTA BRIGHT benchmark)
- `Retriever(method="reason-embed", model_id="qwen3-8b"|"qwen3-4b"|"llama-8b", corpus_path=...)`
- `Retriever(method="bge-reasoner-embed", corpus_path=...)` → BAAI/bge-reasoner-embed-qwen3-8b

When helping users, consider:
1. Their task type (QA, search, summarization, conversational, reasoning-intensive)
2. Hardware constraints (GPU availability, memory)
3. Latency requirements
4. Whether they can use APIs or need local models
5. Language requirements

Provide clear recommendations with explanations. Include code snippets when helpful.
Be concise but thorough. Ask clarifying questions if needed."""

    def __init__(
        self,
        backend: str = "azure",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize RankifyAgent.
        
        Args:
            backend: LLM backend - "azure", "openai", "litellm", "local"
            model_name: Model name (optional, uses defaults per backend)
            api_key: API key (optional, uses environment variables)
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self.recommender = RankifyRecommender()
        self.conversation_history: List[Dict[str, str]] = []
        
        # Set up the LLM client
        self._setup_client()
    
    def _setup_client(self):
        """Set up the LLM client based on backend."""
        if self.backend == "azure":
            self._setup_azure()
        elif self.backend == "openai":
            self._setup_openai()
        elif self.backend == "litellm":
            self._setup_litellm()
        elif self.backend == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _setup_azure(self):
        """Set up Azure OpenAI client."""
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = AzureOpenAI(
            api_key=self.api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_API_VERSION", "2024-05-01-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        self.model_name = self.model_name or os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        self._call_llm = self._call_azure
    
    def _setup_openai(self):
        """Set up OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.model_name = self.model_name or "gpt-4o-mini"
        self._call_llm = self._call_openai
    
    def _setup_litellm(self):
        """Set up LiteLLM for 100+ providers."""
        try:
            import litellm
        except ImportError:
            raise ImportError("Please install litellm: pip install litellm")
        
        self.litellm = litellm
        self.model_name = self.model_name or "gpt-4o-mini"
        self._call_llm = self._call_litellm
    
    def _setup_local(self):
        """Set up local LLM via transformers."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.model_name = self.model_name or "microsoft/Phi-3-mini-4k-instruct"
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            **self.kwargs
        )
        self._call_llm = self._call_local
    
    def _call_azure(self, messages: List[Dict[str, str]]) -> str:
        """Call Azure OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    def _call_litellm(self, messages: List[Dict[str, str]]) -> str:
        """Call LiteLLM."""
        response = self.litellm.completion(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    def _call_local(self, messages: List[Dict[str, str]]) -> str:
        """Call local LLM."""
        # Format messages for local model
        prompt = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in messages
        ])
        prompt += "\nASSISTANT:"
        
        output = self.pipeline(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
        )
        return output[0]["generated_text"].split("ASSISTANT:")[-1].strip()
    
    def chat(self, user_message: str) -> AgentResponse:
        """
        Have a conversation with the agent.
        
        Args:
            user_message: User's message/question
            
        Returns:
            AgentResponse with message, optional recommendation, and code snippet
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Detect if this is a recommendation request
        intent = self._detect_intent(user_message)
        
        # If asking for recommendation, get one first
        recommendation = None
        context = ""
        if intent.get("wants_recommendation"):
            recommendation = self._get_recommendation_from_intent(intent)
            context = self._format_recommendation_context(recommendation)
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT + context}
        ] + self.conversation_history
        
        # Get LLM response
        response_text = self._call_llm(messages)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        # Generate code snippet if recommendation was made
        code_snippet = None
        if recommendation:
            code_snippet = self._generate_code_snippet(recommendation)
        
        return AgentResponse(
            message=response_text,
            recommendation=recommendation,
            code_snippet=code_snippet,
        )
    
    def recommend(
        self,
        task: str = "qa",
        gpu: bool = True,
        api_allowed: bool = True,
        include_rag: bool = False,
    ) -> RecommendationResult:
        """
        Get a direct recommendation without conversation.
        
        Args:
            task: Task type - "qa", "search", "summarization"
            gpu: Whether GPU is available
            api_allowed: Whether API-based models are allowed
            include_rag: Whether to include RAG method
            
        Returns:
            RecommendationResult with recommended models
        """
        constraints = {"gpu": gpu}
        if not api_allowed:
            constraints["no_api"] = True
        
        return self.recommender.recommend(
            task=task,
            constraints=constraints,
            include_rag=include_rag,
        )
    
    def _detect_intent(self, message: str) -> Dict[str, Any]:
        """Detect user intent from message."""
        message_lower = message.lower()
        
        intent = {
            "wants_recommendation": False,
            "task": None,
            "constraints": {}
        }
        
        # Check for recommendation keywords
        recommendation_keywords = [
            "recommend", "suggest", "best", "which", "what should",
            "help me choose", "need a", "looking for", "want to build"
        ]
        if any(kw in message_lower for kw in recommendation_keywords):
            intent["wants_recommendation"] = True
        
        # Detect task type
        if any(w in message_lower for w in ["qa", "question", "answer", "reading"]):
            intent["task"] = "qa"
        elif any(w in message_lower for w in ["search", "find", "retriev", "lookup"]):
            intent["task"] = "search"
        elif any(w in message_lower for w in ["summar", "condense", "tldr"]):
            intent["task"] = "summarization"
        elif any(w in message_lower for w in ["chat", "convers", "dialog"]):
            intent["task"] = "conversational"
        else:
            intent["task"] = "qa"  # Default
        
        # Detect constraints
        if any(w in message_lower for w in ["no gpu", "cpu only", "without gpu"]):
            intent["constraints"]["gpu"] = False
        elif "gpu" in message_lower:
            intent["constraints"]["gpu"] = True
        
        if any(w in message_lower for w in ["fast", "low latency", "quick", "speed"]):
            intent["constraints"]["prefer_speed"] = True
        
        if any(w in message_lower for w in ["accurate", "best quality", "highest quality"]):
            intent["constraints"]["prefer_accuracy"] = True
        
        if any(w in message_lower for w in ["local", "no api", "offline", "privacy"]):
            intent["constraints"]["no_api"] = True
        
        if any(w in message_lower for w in ["production", "deploy", "scale"]):
            intent["constraints"]["prefer_speed"] = True
        
        return intent
    
    def _get_recommendation_from_intent(
        self, 
        intent: Dict[str, Any]
    ) -> RecommendationResult:
        """Get recommendation based on detected intent."""
        return self.recommender.recommend(
            task=intent.get("task", "qa"),
            constraints=intent.get("constraints", {}),
            include_reranker=True,
            include_rag=True,
        )
    
    def _format_recommendation_context(
        self, 
        recommendation: RecommendationResult
    ) -> str:
        """Format recommendation as context for LLM."""
        parts = ["\n\n[RECOMMENDATION CONTEXT]"]
        
        if recommendation.retriever:
            parts.append(f"Best Retriever: {recommendation.retriever.name}")
            parts.append(f"  Method: {recommendation.retriever.method}")
        
        if recommendation.reranker:
            parts.append(f"Best Reranker: {recommendation.reranker.name}")
            parts.append(f"  Method: {recommendation.reranker.method}")
        
        if recommendation.rag_method:
            parts.append(f"Best RAG Method: {recommendation.rag_method.name}")
        
        return "\n".join(parts)
    
    def _generate_code_snippet(
        self, 
        recommendation: RecommendationResult
    ) -> str:
        """Generate code snippet from recommendation."""
        lines = ["from rankify.retrievers.retriever import Retriever"]
        
        if recommendation.reranker:
            lines.append("from rankify.models.reranking import Reranking")
        
        if recommendation.rag_method:
            lines.append("from rankify.generator.generator import Generator")
        
        lines.append("")
        
        # Retriever
        if recommendation.retriever:
            r = recommendation.retriever
            lines.append(f"# Retriever: {r.name}")
            if r.method == "bm25":
                lines.append(f'retriever = Retriever(method="{r.method}", n_docs=100)')
            else:
                lines.append(f'retriever = Retriever(method="{r.method}", n_docs=100)')
        
        # Reranker
        if recommendation.reranker:
            rr = recommendation.reranker
            lines.append(f"\n# Reranker: {rr.name}")
            if rr.model_path:
                lines.append(f'reranker = Reranking(method="{rr.method}", model_name="{rr.model_path}")')
            else:
                lines.append(f'reranker = Reranking(method="{rr.method}")')
        
        # RAG
        if recommendation.rag_method:
            rag = recommendation.rag_method
            lines.append(f"\n# RAG Method: {rag.name}")
            lines.append(f'generator = Generator(method="{rag.method}", model_name="gpt-4o-mini", backend="openai")')
        
        # Usage
        lines.append("\n# Usage")
        lines.append("documents = retriever.retrieve(documents)")
        if recommendation.reranker:
            lines.append("documents = reranker.rank(documents)")
        if recommendation.rag_method:
            lines.append("answers = generator.generate(documents)")
        
        return "\n".join(lines)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_available_models(self, model_type: str = "all") -> Dict[str, List[str]]:
        """
        Get list of available models.
        
        Args:
            model_type: "retriever", "reranker", "rag", or "all"
            
        Returns:
            Dict of model names by type
        """
        result = {}
        
        if model_type in ["retriever", "all"]:
            result["retrievers"] = list(RETRIEVER_REGISTRY.keys())
        
        if model_type in ["reranker", "all"]:
            result["rerankers"] = list(RERANKER_REGISTRY.keys())
        
        if model_type in ["rag", "all"]:
            result["rag_methods"] = list(RAG_METHOD_REGISTRY.keys())
        
        return result
