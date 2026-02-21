"""
Rankify - A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and RAG

Simple usage:
    >>> from rankify import pipeline
    >>> rag = pipeline("rag")
    >>> answers = rag("What is machine learning?", documents)
"""

import os
from pathlib import Path

# Set up cache directory
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "rankify")
os.environ.setdefault("RERANKING_CACHE_DIR", DEFAULT_CACHE_DIR)
try:
    Path(os.environ["RERANKING_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass

# Main pipeline interface
from rankify.pipeline import (
    Pipeline,
    pipeline,
    search_pipeline,
    rerank_pipeline,
    rag_pipeline,
    PipelineResult,
)

# Version
__version__ = "0.1.5"

__all__ = [
    # Pipeline (main interface)
    "Pipeline",
    "pipeline",
    "search_pipeline",
    "rerank_pipeline",
    "rag_pipeline",
    "PipelineResult",
    # Version
    "__version__",
]