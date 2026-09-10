# rankify/retrievers/__init__.py

from .retriever import Retriever
from .base_retriever import BaseRetriever
from .bm25s_retriever import BM25SRetriever

try:
    from .bm25_retriever import BM25Retriever
except ImportError:
    BM25Retriever = None  # type: ignore[assignment,misc]

try:
    from .dense_retriever import DenseRetriever
except ImportError:
    DenseRetriever = None  # type: ignore[assignment,misc]

try:
    from .ance_retriever import ANCERetriever
except ImportError:
    ANCERetriever = None  # type: ignore[assignment,misc]

try:
    from .bge_retriever import BGERetriever
except ImportError:
    BGERetriever = None  # type: ignore[assignment,misc]

try:
    from .colbert_retriever import ColBERTRetriever
except ImportError:
    ColBERTRetriever = None  # type: ignore[assignment,misc]

try:
    from .contriever_retriever import ContrieverRetriever
except ImportError:
    ContrieverRetriever = None  # type: ignore[assignment,misc]

try:
    from .online_retriever import OnlineRetriever
except ImportError:
    OnlineRetriever = None  # type: ignore[assignment,misc]

try:
    from .hyde_retriever import HydeRetriever
except ImportError:
    HydeRetriever = None  # type: ignore[assignment,misc]

try:
    from .diver_dense_retriever import DiverDenseRetriever
except ImportError:
    DiverDenseRetriever = None  # type: ignore[assignment,misc]

try:
    from .diver_bm25_retriever import DiverBM25Retriever
except ImportError:
    DiverBM25Retriever = None  # type: ignore[assignment,misc]

try:
    from .reasonir_retriever import ReasonIRRetriever
except ImportError:
    ReasonIRRetriever = None  # type: ignore[assignment,misc]

try:
    from .reasonembed_retriever import ReasonEmbedRetriever
except ImportError:
    ReasonEmbedRetriever = None  # type: ignore[assignment,misc]

try:
    from .bge_reasoner_retriever import BgeReasonerRetriever
except ImportError:
    BgeReasonerRetriever = None  # type: ignore[assignment,misc]


__all__ = [
    "Retriever",
    "BaseRetriever",
    "BM25SRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "ANCERetriever",
    "BGERetriever",
    "ColBERTRetriever",
    "ContrieverRetriever",
    "OnlineRetriever",
    "HydeRetriever",
    "DiverDenseRetriever",
    "DiverBM25Retriever",
    "ReasonIRRetriever",
    "ReasonEmbedRetriever",
    "BgeReasonerRetriever",
]