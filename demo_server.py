"""
Rankify Demo Server
===================
A FastAPI server that accepts retriever, reranker, and generator as per-request
parameters. This allows the demo UI to switch models without restarting the server.

Usage:
    python demo_server.py --port 8000

    # Or with default models pre-loaded for speed:
    python demo_server.py --port 8000 --default-retriever bm25 --default-reranker flashrank

Endpoints:
    GET  /health          - Server health check
    POST /pipeline        - Run retrieve / rerank / rag based on mode param
    GET  /models          - List all supported models
"""

import os
import time
import json
import logging
import argparse
import traceback
from typing import Optional, List, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise SystemExit("FastAPI not found. Install: pip install fastapi uvicorn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Component Cache ───────────────────────────────────────────────────────────
# We cache initialized components to avoid reloading heavy models on each request.
_retriever_cache: Dict[str, Any] = {}
_reranker_cache:  Dict[str, Any] = {}
_generator_cache: Dict[str, Any] = {}


def get_retriever(method: str, n_docs: int = 100, index_type: str = "wiki"):
    key = f"{method}|{index_type}"
    if key not in _retriever_cache:
        logger.info(f"Loading retriever: {method} [{index_type}]")
        from rankify.retrievers.retriever import Retriever
        _retriever_cache[key] = Retriever(method=method, n_docs=n_docs, index_type=index_type)
        logger.info(f"Retriever loaded: {key}")
    return _retriever_cache[key]


def get_reranker(method: str, model_name: str):
    key = f"{method}|{model_name}"
    if key not in _reranker_cache:
        logger.info(f"Loading reranker: {method} / {model_name}")
        from rankify.models.reranking import Reranking
        _reranker_cache[key] = Reranking(method=method, model_name=model_name)
        logger.info(f"Reranker loaded: {key}")
    return _reranker_cache[key]


def get_generator(method: str, model_name: str, backend: str):
    key = f"{method}|{model_name}|{backend}"
    if key not in _generator_cache:
        logger.info(f"Loading generator: {method} / {model_name}")
        from rankify.generator.generator import Generator
        _generator_cache[key] = Generator(method=method, model_name=model_name, backend=backend)
        logger.info(f"Generator loaded: {key}")
    return _generator_cache[key]


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class PipelineRequest(BaseModel):
    query: str = Field(..., description="User query")
    mode: str = Field("rag", description="Pipeline mode: retrieve | rerank | rag")
    retriever: str = Field("bm25", description="Retriever method")
    rerankerCategory: str = Field("flashrank", description="Reranker category/method")
    rerankerModel: str = Field("ms-marco-MiniLM-L-12-v2", description="Specific reranker model")
    generator: str = Field("openai", description="Generator backend")
    dataSource: str = Field("wiki", description="Index type: wiki | msmarco | custom")
    n_docs: int = Field(10, description="Number of documents to retrieve")
    n_contexts: int = Field(5, description="Number of top contexts for generation")


class DocumentOut(BaseModel):
    id: str
    text: str
    title: Optional[str] = None
    score: Optional[float] = None


class PipelineResponse(BaseModel):
    query: str
    mode: str
    retrieved_docs: List[DocumentOut] = []
    reranked_docs:  List[DocumentOut] = []
    answer: Optional[str] = None
    retriever_latency_ms: float = 0
    reranker_latency_ms:  float = 0
    generator_latency_ms: float = 0
    error: Optional[str] = None


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Rankify Demo API",
    description="Dynamic retrieval/reranking/RAG API for the Rankify demo UI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cached_retrievers": list(_retriever_cache.keys()),
        "cached_rerankers":  list(_reranker_cache.keys()),
        "cached_generators": list(_generator_cache.keys()),
    }


@app.get("/models")
async def list_models():
    """Return all supported models (mirrors models.ts on the frontend)."""
    return {
        "retrievers": ["bm25", "dpr", "ance", "contriever", "colbert", "bge"],
        "reranker_categories": {
            "flashrank": ["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2",
                          "ms-marco-MultiBERT-L-12", "rank-T5-flan", "ce-esci-MiniLM-L12-v2"],
            "transformer_ranker": ["bge-reranker-base", "bge-reranker-large",
                                   "bge-reranker-v2-m3", "mxbai-rerank-base", "mxbai-rerank-large",
                                   "ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2"],
            "monot5": ["monot5-base-msmarco", "monot5-large-msmarco", "monot5-base-msmarco-10k"],
            "colbert_ranker": ["colbertv2.0", "jina-colbert-v1-en"],
            "monobert": ["monobert-large"],
        },
        "generators": ["openai", "claude", "llama-3", "mistral", "litellm"],
    }


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(req: PipelineRequest):
    """
    Run the full Rankify pipeline based on the requested mode.
    Every parameter (retriever, reranker, generator) is chosen per-request.
    """
    from rankify.dataset.dataset import Document, Question, Answer, Context

    resp = PipelineResponse(query=req.query, mode=req.mode)
    index_type = "msmarco" if req.dataSource == "msmarco" else "wiki"

    try:
        # ── Step 1: Retrieval ──────────────────────────────────────────────────
        t0 = time.time()
        retriever = get_retriever(req.retriever, n_docs=req.n_docs, index_type=index_type)
        doc = Document(question=Question(req.query), answers=Answer([]), contexts=[])
        results = retriever.retrieve([doc])
        retrieved = results[0].contexts[:req.n_docs]
        resp.retriever_latency_ms = round((time.time() - t0) * 1000, 1)

        resp.retrieved_docs = [
            DocumentOut(
                id=str(ctx.id or i),
                text=str(ctx.text or "")[:600],
                title=str(ctx.title) if hasattr(ctx, "title") and ctx.title else None,
                score=float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else None,
            )
            for i, ctx in enumerate(retrieved)
        ]

        if req.mode == "retrieve":
            return resp

        # ── Step 2: Reranking ──────────────────────────────────────────────────
        t1 = time.time()
        reranker = get_reranker(req.rerankerCategory, req.rerankerModel)

        doc2 = Document(
            question=Question(req.query),
            answers=Answer([]),
            contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                              title=ctx.title if hasattr(ctx, "title") else "")
                      for i, ctx in enumerate(retrieved)],
        )
        reranked_results = reranker.rank([doc2])
        reranked = (reranked_results[0].reorder_contexts or reranked_results[0].contexts)[:req.n_contexts]
        resp.reranker_latency_ms = round((time.time() - t1) * 1000, 1)

        resp.reranked_docs = [
            DocumentOut(
                id=str(ctx.id or i),
                text=str(ctx.text or "")[:600],
                title=str(ctx.title) if hasattr(ctx, "title") and ctx.title else None,
                score=float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else None,
            )
            for i, ctx in enumerate(reranked)
        ]

        if req.mode == "rerank":
            return resp

        # ── Step 3: Generation ────────────────────────────────────────────────
        t2 = time.time()

        # Map UI generator value → Rankify generator method + backend
        GENERATOR_MAP = {
            "openai":   ("basic-rag", "gpt-4o-mini",        "openai"),
            "claude":   ("basic-rag", "claude-3-5-sonnet",   "anthropic"),
            "llama-3":  ("basic-rag", "meta-llama/Meta-Llama-3.1-8B-Instruct", "vllm"),
            "mistral":  ("basic-rag", "mistralai/Mistral-7B-Instruct-v0.3",    "vllm"),
            "litellm":  ("basic-rag", "gpt-4o-mini",        "litellm"),
        }
        method, model_name, backend = GENERATOR_MAP.get(req.generator, ("basic-rag", "gpt-4o-mini", "openai"))

        gen_doc = Document(
            question=Question(req.query),
            answers=Answer([]),
            contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                              title=ctx.title if hasattr(ctx, "title") else "")
                      for i, ctx in enumerate(reranked)],
        )

        generator = get_generator(method=method, model_name=model_name, backend=backend)
        answers = generator.generate([gen_doc])
        resp.answer = answers[0] if answers else "No answer generated."
        resp.generator_latency_ms = round((time.time() - t2) * 1000, 1)

    except Exception as e:
        logger.error(f"Pipeline error: {traceback.format_exc()}")
        resp.error = str(e)

    return resp


@app.post("/pipeline/stream")
async def pipeline_stream(req: PipelineRequest):
    """
    Same as /pipeline but streams the answer token-by-token using SSE.
    Used by the 'rag' mode for real-time generation display.
    """
    async def generate():
        import asyncio
        # First run the non-streaming pipeline to get docs
        from rankify.dataset.dataset import Document, Question, Answer, Context

        index_type = "msmarco" if req.dataSource == "msmarco" else "wiki"

        try:
            # Retrieval
            retriever = get_retriever(req.retriever, n_docs=req.n_docs, index_type=index_type)
            doc = Document(question=Question(req.query), answers=Answer([]), contexts=[])
            results = retriever.retrieve([doc])
            retrieved = results[0].contexts[:req.n_docs]

            retrieved_payload = json.dumps({
                "type": "retrieved",
                "docs": [
                    {"id": str(ctx.id or i), "text": str(ctx.text or "")[:600],
                     "score": float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else 0.0}
                    for i, ctx in enumerate(retrieved)
                ]
            })
            yield f"data: {retrieved_payload}\n\n"

            if req.mode == "retrieve":
                yield "data: {\"type\":\"done\"}\n\n"
                return

            # Reranking
            reranker = get_reranker(req.rerankerCategory, req.rerankerModel)
            doc2 = Document(
                question=Question(req.query),
                answers=Answer([]),
                contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                                  title=ctx.title if hasattr(ctx, "title") else "")
                          for i, ctx in enumerate(retrieved)],
            )
            reranked_results = reranker.rank([doc2])
            reranked = (reranked_results[0].reorder_contexts or reranked_results[0].contexts)[:req.n_contexts]

            reranked_payload = json.dumps({
                "type": "reranked",
                "docs": [
                    {"id": str(ctx.id or i), "text": str(ctx.text or "")[:600],
                     "score": float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else 0.0}
                    for i, ctx in enumerate(reranked)
                ]
            })
            yield f"data: {reranked_payload}\n\n"

            if req.mode == "rerank":
                yield "data: {\"type\":\"done\"}\n\n"
                return

            # Generation
            GENERATOR_MAP = {
                "openai":  ("basic-rag", "gpt-4o-mini", "openai"),
                "claude":  ("basic-rag", "claude-3-5-sonnet", "anthropic"),
                "llama-3": ("basic-rag", "meta-llama/Meta-Llama-3.1-8B-Instruct", "vllm"),
                "mistral": ("basic-rag", "mistralai/Mistral-7B-Instruct-v0.3", "vllm"),
                "litellm": ("basic-rag", "gpt-4o-mini", "litellm"),
            }
            method, model_name, backend = GENERATOR_MAP.get(req.generator, ("basic-rag", "gpt-4o-mini", "openai"))

            gen_doc = Document(
                question=Question(req.query),
                answers=Answer([]),
                contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                                  title=ctx.title if hasattr(ctx, "title") else "")
                          for i, ctx in enumerate(reranked)],
            )

            generator = get_generator(method=method, model_name=model_name, backend=backend)
            answers = generator.generate([gen_doc])
            answer = answers[0] if answers else "No answer generated."

            # Stream the answer word by word
            words = answer.split()
            for i, word in enumerate(words):
                token_payload = json.dumps({"type": "token", "content": word + (" " if i < len(words) - 1 else "")})
                yield f"data: {token_payload}\n\n"
                await asyncio.sleep(0.02)

            yield "data: {\"type\":\"done\"}\n\n"

        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rankify Demo Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting Rankify Demo Server on http://{args.host}:{args.port}")
    logger.info("Docs available at /docs")
    uvicorn.run("demo_server:app", host=args.host, port=args.port, reload=args.reload)
