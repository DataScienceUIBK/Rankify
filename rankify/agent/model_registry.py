"""
Model Registry for RankifyAgent.

Contains metadata about all available retrievers, rerankers, and RAG methods
to enable intelligent model recommendation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Supported task types."""
    QUESTION_ANSWERING = "qa"
    SEARCH = "search"
    SUMMARIZATION = "summarization"
    CONVERSATIONAL = "conversational"
    DOMAIN_SPECIFIC = "domain_specific"


class Speed(Enum):
    """Model speed categories."""
    VERY_FAST = "very_fast"  # < 10ms per query
    FAST = "fast"            # < 50ms per query
    MEDIUM = "medium"        # < 200ms per query
    SLOW = "slow"            # < 1s per query
    VERY_SLOW = "very_slow"  # > 1s per query


class Accuracy(Enum):
    """Model accuracy categories."""
    BASIC = "basic"
    GOOD = "good"
    VERY_GOOD = "very_good"
    EXCELLENT = "excellent"
    STATE_OF_THE_ART = "sota"


@dataclass
class ModelMetadata:
    """Metadata for a single model."""
    name: str
    method: str
    description: str
    speed: Speed
    accuracy: Accuracy
    gpu_required: bool
    memory_mb: int
    best_for: List[str]
    languages: List[str] = field(default_factory=lambda: ["en"])
    api_required: bool = False
    api_provider: Optional[str] = None
    model_path: Optional[str] = None
    
    def matches_constraints(self, constraints: Dict[str, Any]) -> bool:
        """Check if model matches user constraints."""
        if constraints.get("gpu") is False and self.gpu_required:
            return False
        if constraints.get("max_memory_mb") and self.memory_mb > constraints["max_memory_mb"]:
            return False
        if constraints.get("api_only") is True and not self.api_required:
            return False
        if constraints.get("no_api") is True and self.api_required:
            return False
        if constraints.get("language") and constraints["language"] not in self.languages:
            return False
        return True
    
    def score_for_task(self, task: TaskType) -> float:
        """Score model suitability for a task (0-1)."""
        task_keywords = {
            TaskType.QUESTION_ANSWERING: ["qa", "question", "answer", "reading comprehension"],
            TaskType.SEARCH: ["search", "retrieval", "ranking", "information retrieval"],
            TaskType.SUMMARIZATION: ["summarization", "long context", "multi-document"],
            TaskType.CONVERSATIONAL: ["conversational", "dialogue", "chat"],
            TaskType.DOMAIN_SPECIFIC: ["domain", "specialized", "technical"],
        }
        keywords = task_keywords.get(task, [])
        matches = sum(1 for k in keywords if k in " ".join(self.best_for).lower())
        return min(1.0, matches / max(1, len(keywords)) + 0.3)  # Base score of 0.3


# =============================================================================
# RETRIEVER REGISTRY
# =============================================================================

RETRIEVER_REGISTRY: Dict[str, ModelMetadata] = {
    "bm25": ModelMetadata(
        name="BM25",
        method="bm25",
        description="Classic sparse retrieval using BM25 algorithm. Fast, no GPU needed.",
        speed=Speed.VERY_FAST,
        accuracy=Accuracy.GOOD,
        gpu_required=False,
        memory_mb=500,
        best_for=["keyword search", "exact match", "large corpus", "low latency"],
        languages=["en", "multilingual"],
    ),
    "dpr-multi": ModelMetadata(
        name="DPR (Multi-Encoder)",
        method="dpr",
        description="Dense Passage Retrieval with separate query/passage encoders.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=2000,
        best_for=["semantic search", "qa", "open-domain qa"],
        model_path="facebook-dpr-ctx_encoder-multiset-base",
    ),
    "dpr-single": ModelMetadata(
        name="DPR (Single-Encoder)",
        method="dpr",
        description="DPR with single encoder for both query and passage.",
        speed=Speed.FAST,
        accuracy=Accuracy.GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["semantic search", "faster inference"],
        model_path="facebook-dpr-question_encoder-single-nq-base",
    ),
    "ance": ModelMetadata(
        name="ANCE",
        method="ance",
        description="Approximate Nearest Neighbor Negative Contrastive Estimation.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=2500,
        best_for=["high accuracy retrieval", "qa", "passage ranking"],
        model_path="castorini/ance-msmarco-passage",
    ),
    "bge": ModelMetadata(
        name="BGE",
        method="bge",
        description="BAAI General Embedding - state-of-the-art dense retriever.",
        speed=Speed.FAST,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=1500,
        best_for=["semantic search", "qa", "high accuracy", "multilingual"],
        languages=["en", "zh", "multilingual"],
        model_path="BAAI/bge-base-en-v1.5",
    ),
    "colbert": ModelMetadata(
        name="ColBERT",
        method="colbert",
        description="Contextualized Late Interaction over BERT - token-level matching.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=4000,
        best_for=["high precision", "passage retrieval", "fine-grained matching"],
        model_path="colbert-ir/colbertv2.0",
    ),
    "contriever": ModelMetadata(
        name="Contriever",
        method="contriever",
        description="Unsupervised dense retriever, good zero-shot performance.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1800,
        best_for=["zero-shot retrieval", "domain transfer", "no training data"],
        model_path="facebook/contriever-msmarco",
    ),
    "hyde": ModelMetadata(
        name="HyDE (Hypothetical Document Embeddings)",
        method="hyde",
        description="Generates hypothetical documents to improve retrieval.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=3000,
        api_required=True,
        best_for=["complex queries", "sparse training data", "query expansion"],
    ),
    "online": ModelMetadata(
        name="Online Retriever",
        method="online",
        description="Web search-based retrieval using search APIs.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=False,
        memory_mb=100,
        api_required=True,
        api_provider="serper",
        best_for=["real-time data", "current events", "web search"],
    ),

    # === DIVER DENSE RETRIEVERS ===
    # method="diver-dense", use model_path as the model_id argument
    # Valid model_ids: bge, sbert, contriever_st, nomic, diver, inst-l, inst-xl, sf, e5, rader, m2, contriever, grit
    "diver-bge": ModelMetadata(
        name="Diver (BGE Large)",
        method="diver-dense",
        description="BAAI/bge-large-en-v1.5 via the Diver dense retrieval framework.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=3000,
        best_for=["semantic search", "high accuracy", "BEIR benchmarks"],
        model_path="bge",
    ),
    "diver-sbert": ModelMetadata(
        name="Diver (SBERT all-mpnet-base-v2)",
        method="diver-dense",
        description="sentence-transformers/all-mpnet-base-v2 via the Diver framework.",
        speed=Speed.FAST,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["sentence similarity", "semantic search"],
        model_path="sbert",
    ),
    "diver-nomic": ModelMetadata(
        name="Diver (Nomic Embed)",
        method="diver-dense",
        description="nomic-ai/nomic-embed-text-v1 via the Diver framework.",
        speed=Speed.FAST,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["long context", "semantic search", "document retrieval"],
        model_path="nomic",
    ),
    "diver-e5": ModelMetadata(
        name="Diver (E5-Mistral-7B)",
        method="diver-dense",
        description="intfloat/e5-mistral-7b-instruct — instruction-tuned LLM encoder in the Diver framework.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=16000,
        best_for=["zero-shot retrieval", "instruction following", "complex queries"],
        model_path="e5",
    ),
    "diver-sf": ModelMetadata(
        name="Diver (SFR-Embedding-Mistral)",
        method="diver-dense",
        description="Salesforce/SFR-Embedding-Mistral — Salesforce Mistral-based bi-encoder in the Diver framework.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=16000,
        best_for=["high accuracy retrieval", "complex queries", "BEIR benchmarks"],
        model_path="sf",
    ),
    "diver-rader": ModelMetadata(
        name="Diver (RaDeR)",
        method="diver-dense",
        description="Raderspace/RaDeR_Qwen_25_7B — reasoning-aware dense retriever in the Diver framework.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=16000,
        best_for=["reasoning-intensive queries", "multi-hop QA", "math-related retrieval"],
        model_path="rader",
    ),
    "diver-grit": ModelMetadata(
        name="Diver (GritLM-7B)",
        method="diver-dense",
        description="GritLM/GritLM-7B — generative representation model in the Diver framework.",
        speed=Speed.VERY_SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=16000,
        best_for=["generative retrieval", "LLM-quality embeddings", "long context"],
        model_path="grit",
    ),
    "diver-model": ModelMetadata(
        name="Diver Retriever-4B",
        method="diver-dense",
        description="AQ-MedAI/Diver-Retriever-4B — the flagship Diver diverse-evidence retrieval model.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=8000,
        best_for=["diverse evidence retrieval", "BEIR benchmarks", "medical QA"],
        model_path="diver",
    ),
    "diver-inst-l": ModelMetadata(
        name="Diver (Instructor-Large)",
        method="diver-dense",
        description="hkunlp/instructor-large — instruction-following encoder in the Diver framework.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=3000,
        best_for=["instruction following", "domain-specific retrieval"],
        model_path="inst-l",
    ),
    "diver-m2": ModelMetadata(
        name="Diver (M2-BERT-32K)",
        method="diver-dense",
        description="togethercomputer/m2-bert-80M-32k-retrieval — long-context retrieval in the Diver framework.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=2000,
        best_for=["long-context retrieval", "32k sequence length"],
        model_path="m2",
    ),

    # === REASONING-AUGMENTED RETRIEVERS ===
    "reasonir": ModelMetadata(
        name="ReasonIR-8B",
        method="reasonir",
        description="reasonir/ReasonIR-8B — SOTA reasoning-intensive retriever on the BRIGHT benchmark. No model_id needed.",
        speed=Speed.VERY_SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=16000,
        best_for=["reasoning-intensive queries", "BRIGHT benchmark", "complex multi-hop QA", "science queries"],
    ),
    "reason-embed-qwen3-8b": ModelMetadata(
        name="ReasonEmbed Qwen3-8B",
        method="reason-embed",
        description="hanhainebula/reason-embed-qwen3-8b-0928 — Qwen3-8B for reasoning retrieval. Use model_id='qwen3-8b'.",
        speed=Speed.VERY_SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=16000,
        best_for=["reasoning-intensive retrieval", "complex queries"],
        model_path="qwen3-8b",
    ),
    "reason-embed-qwen3-4b": ModelMetadata(
        name="ReasonEmbed Qwen3-4B",
        method="reason-embed",
        description="hanhainebula/reason-embed-qwen3-4b-0928 — balanced Qwen3-4B for reasoning retrieval. Use model_id='qwen3-4b'.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=8000,
        best_for=["reasoning retrieval", "balanced accuracy/speed"],
        model_path="qwen3-4b",
    ),
    "reason-embed-llama-8b": ModelMetadata(
        name="ReasonEmbed LLaMA-3.1-8B",
        method="reason-embed",
        description="hanhainebula/reason-embed-llama-3.1-8b-0928 — LLaMA-3.1-8B for reasoning retrieval. Use model_id='llama-8b'.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=16000,
        best_for=["reasoning retrieval", "open-source LLaMA backbone"],
        model_path="llama-8b",
    ),
    "bge-reasoner-embed": ModelMetadata(
        name="BGE Reasoner Embed (Qwen3-8B)",
        method="bge-reasoner-embed",
        description="BAAI/bge-reasoner-embed-qwen3-8b-0923 — BGE reasoning-augmented retriever. No model_id needed.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=16000,
        best_for=["reasoning-augmented retrieval", "BEIR benchmarks", "complex queries"],
    ),
}




# =============================================================================
# RERANKER REGISTRY
# =============================================================================

RERANKER_REGISTRY: Dict[str, ModelMetadata] = {
    # === POINTWISE RERANKERS ===
    "monot5-base-msmarco": ModelMetadata(
        name="MonoT5 Base",
        method="monot5",
        description="T5-based pointwise reranker, trained on MS MARCO.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=900,
        best_for=["qa", "general reranking", "passage ranking"],
        model_path="castorini/monot5-base-msmarco",
    ),
    "monot5-3b-msmarco": ModelMetadata(
        name="MonoT5 3B",
        method="monot5",
        description="Large MonoT5 reranker for highest accuracy.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=12000,
        best_for=["highest accuracy", "offline processing"],
        model_path="castorini/monot5-3b-msmarco",
    ),
    "monobert": ModelMetadata(
        name="MonoBERT",
        method="monobert",
        description="BERT-based pointwise reranker.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["qa", "document ranking"],
        model_path="castorini/monobert-large-msmarco",
    ),
    "flashrank-minilm": ModelMetadata(
        name="FlashRank MiniLM",
        method="flashrank",
        description="Ultra-fast ONNX-based reranker, CPU-friendly.",
        speed=Speed.VERY_FAST,
        accuracy=Accuracy.GOOD,
        gpu_required=False,
        memory_mb=50,
        best_for=["low latency", "cpu deployment", "edge devices", "production"],
        model_path="ms-marco-MiniLM-L-12-v2",
    ),
    "flashrank-tinybert": ModelMetadata(
        name="FlashRank TinyBERT",
        method="flashrank",
        description="Smallest FlashRank model, fastest inference.",
        speed=Speed.VERY_FAST,
        accuracy=Accuracy.BASIC,
        gpu_required=False,
        memory_mb=20,
        best_for=["ultra-low latency", "mobile", "embedded"],
        model_path="ms-marco-TinyBERT-L-2-v2",
    ),
    "upr-t5-base": ModelMetadata(
        name="UPR (T5 Base)",
        method="upr",
        description="Unsupervised Passage Reranker using T5.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.GOOD,
        gpu_required=True,
        memory_mb=900,
        best_for=["zero-shot reranking", "no training data"],
        model_path="t5-base",
    ),
    
    # === PAIRWISE RERANKERS ===
    "rankgpt-gpt4": ModelMetadata(
        name="RankGPT (GPT-4)",
        method="rankgpt-api",
        description="LLM-based listwise reranking using GPT-4.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=False,
        memory_mb=100,
        api_required=True,
        api_provider="openai",
        best_for=["highest accuracy", "complex queries", "reasoning-based ranking"],
        model_path="gpt-4o",
    ),
    "rankgpt-llama": ModelMetadata(
        name="RankGPT (LLaMA)",
        method="rankgpt",
        description="Local LLM-based listwise reranking using LLaMA.",
        speed=Speed.VERY_SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=16000,
        best_for=["high accuracy", "privacy", "no api costs"],
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
    "inranker-base": ModelMetadata(
        name="InRanker Base",
        method="inranker",
        description="In-context reranking with instruction-tuned models.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["instruction following", "zero-shot"],
        model_path="unicamp-dl/InRanker-base",
    ),
    "echorank": ModelMetadata(
        name="EchoRank",
        method="echorank",
        description="Pairwise reranking with echo-based verification.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=2000,
        best_for=["verification", "fact checking"],
    ),
    
    # === LISTWISE RERANKERS ===
    "rankt5-base": ModelMetadata(
        name="RankT5 Base",
        method="rankt5",
        description="T5-based listwise reranker.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=1500,
        best_for=["listwise ranking", "multi-document"],
        model_path="Soyoung97/RankT5-base",
    ),
    "listt5-base": ModelMetadata(
        name="ListT5 Base",
        method="listt5",
        description="List-aware T5 reranker.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=1500,
        best_for=["list ranking", "document sets"],
        model_path="Soyoung97/ListT5-base",
    ),
    "colbert-reranker": ModelMetadata(
        name="ColBERT Reranker",
        method="colbert_ranker",
        description="ColBERT-based reranking for fine-grained matching.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=True,
        memory_mb=3000,
        best_for=["fine-grained matching", "high precision"],
    ),
    "transformer-reranker": ModelMetadata(
        name="Transformer Reranker",
        method="transformer_ranker",
        description="General transformer-based cross-encoder reranker.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1500,
        best_for=["general reranking", "cross-encoder"],
    ),
    
    # === API-BASED RERANKERS ===
    "cohere-rerank": ModelMetadata(
        name="Cohere Reranker",
        method="apiranker",
        description="Cohere's reranking API - fast and accurate.",
        speed=Speed.FAST,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=False,
        memory_mb=50,
        api_required=True,
        api_provider="cohere",
        best_for=["production", "easy deployment", "no gpu"],
        model_path="cohere",
    ),
    "jina-rerank": ModelMetadata(
        name="Jina Reranker",
        method="apiranker",
        description="Jina AI's reranking API.",
        speed=Speed.FAST,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=False,
        memory_mb=50,
        api_required=True,
        api_provider="jina",
        best_for=["production", "multilingual"],
        model_path="jina",
    ),
    "voyage-rerank": ModelMetadata(
        name="Voyage Reranker",
        method="apiranker",
        description="Voyage AI's reranking API.",
        speed=Speed.FAST,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=False,
        memory_mb=50,
        api_required=True,
        api_provider="voyage",
        best_for=["production", "high quality"],
        model_path="voyage",
    ),
    
    # === SPECIALIZED RERANKERS ===
    "splade-reranker": ModelMetadata(
        name="SPLADE Reranker",
        method="splade_reranker",
        description="Sparse lexical and dense hybrid reranker.",
        speed=Speed.FAST,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=True,
        memory_mb=1200,
        best_for=["hybrid retrieval", "keyword + semantic"],
    ),
    "sentence-transformer-reranker": ModelMetadata(
        name="Sentence Transformer Reranker",
        method="sentence_transformer_reranker",
        description="Reranking with sentence transformer embeddings.",
        speed=Speed.FAST,
        accuracy=Accuracy.GOOD,
        gpu_required=True,
        memory_mb=1000,
        best_for=["semantic similarity", "embedding-based"],
    ),
    "llm2vec-reranker": ModelMetadata(
        name="LLM2Vec Reranker",
        method="llm2vec",
        description="LLM embeddings converted for reranking.",
        speed=Speed.SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=8000,
        best_for=["llm-based", "high accuracy"],
    ),
    "twolar": ModelMetadata(
        name="TWOLAR",
        method="twolar",
        description="Two-stage list-aware reranking.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=2000,
        best_for=["list-aware", "multi-stage"],
    ),
}


# =============================================================================
# RAG METHOD REGISTRY
# =============================================================================

RAG_METHOD_REGISTRY: Dict[str, ModelMetadata] = {
    "zero-shot": ModelMetadata(
        name="Zero-Shot RAG",
        method="zero-shot",
        description="Direct answer generation without examples.",
        speed=Speed.FAST,
        accuracy=Accuracy.GOOD,
        gpu_required=False,  # API-based
        memory_mb=100,
        best_for=["simple qa", "fast responses"],
    ),
    "basic-rag": ModelMetadata(
        name="Basic RAG",
        method="basic-rag",
        description="Standard RAG with context injection.",
        speed=Speed.FAST,
        accuracy=Accuracy.VERY_GOOD,
        gpu_required=False,
        memory_mb=100,
        best_for=["general qa", "document qa"],
    ),
    "chain-of-thought-rag": ModelMetadata(
        name="Chain-of-Thought RAG",
        method="chain-of-thought-rag",
        description="RAG with step-by-step reasoning.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=False,
        memory_mb=100,
        best_for=["complex questions", "multi-step reasoning", "math"],
    ),
    "self-consistency-rag": ModelMetadata(
        name="Self-Consistency RAG",
        method="self-consistency-rag",
        description="Multiple generations with majority voting.",
        speed=Speed.SLOW,
        accuracy=Accuracy.STATE_OF_THE_ART,
        gpu_required=False,
        memory_mb=100,
        best_for=["highest accuracy", "critical applications", "verification"],
    ),
    "react-rag": ModelMetadata(
        name="ReAct RAG",
        method="react-rag",
        description="Reasoning and acting interleaved.",
        speed=Speed.VERY_SLOW,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=False,
        memory_mb=100,
        best_for=["multi-hop qa", "tool use", "complex reasoning"],
    ),
    "fid-rag": ModelMetadata(
        name="Fusion-in-Decoder RAG",
        method="fid-rag",
        description="All contexts fused in decoder at once.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=True,
        memory_mb=4000,
        best_for=["multi-document", "information synthesis"],
        model_path="google/fid-nq-base",
    ),
    "in-context-ralm": ModelMetadata(
        name="In-Context RALM",
        method="in-context-ralm",
        description="In-context retrieval-augmented language modeling.",
        speed=Speed.MEDIUM,
        accuracy=Accuracy.EXCELLENT,
        gpu_required=False,
        memory_mb=100,
        best_for=["few-shot learning", "context adaptation"],
    ),
}


# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

def get_all_retrievers() -> Dict[str, ModelMetadata]:
    """Get all available retrievers."""
    return RETRIEVER_REGISTRY


def get_all_rerankers() -> Dict[str, ModelMetadata]:
    """Get all available rerankers."""
    return RERANKER_REGISTRY


def get_all_rag_methods() -> Dict[str, ModelMetadata]:
    """Get all available RAG methods."""
    return RAG_METHOD_REGISTRY


def get_model(model_type: str, name: str) -> Optional[ModelMetadata]:
    """Get a specific model by type and name."""
    registries = {
        "retriever": RETRIEVER_REGISTRY,
        "reranker": RERANKER_REGISTRY,
        "rag": RAG_METHOD_REGISTRY,
    }
    registry = registries.get(model_type)
    if registry:
        return registry.get(name)
    return None


def filter_models(
    model_type: str,
    constraints: Dict[str, Any],
    task: Optional[TaskType] = None,
) -> List[ModelMetadata]:
    """Filter models by constraints and optionally score for task."""
    registries = {
        "retriever": RETRIEVER_REGISTRY,
        "reranker": RERANKER_REGISTRY,
        "rag": RAG_METHOD_REGISTRY,
    }
    registry = registries.get(model_type, {})
    
    results = []
    for name, model in registry.items():
        if model.matches_constraints(constraints):
            results.append(model)
    
    # Sort by task suitability if task provided
    if task:
        results.sort(key=lambda m: m.score_for_task(task), reverse=True)
    
    return results
