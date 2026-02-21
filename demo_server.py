"""
Rankify Demo Server v2
======================
Dynamic FastAPI server - all models configurable per-request.
Supports Azure OpenAI, all Rankify RAG methods, and streaming.

Usage:
    python demo_server.py --port 8000

Environment (set in .env or export):
    RANKIFY_AZURE_ENDPOINT    - Azure OpenAI endpoint
    RANKIFY_AZURE_KEY         - Azure OpenAI key
    RANKIFY_AZURE_DEPLOYMENT  - Deployment name (e.g. gpt-4o-2)
    RANKIFY_AZURE_API_VERSION - API version
"""

import os
import time
import json
import logging
import argparse
import traceback
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "demo-web", ".env.local"))
load_dotenv()

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise SystemExit("Install: pip install fastapi uvicorn python-dotenv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Azure / OpenAI env ────────────────────────────────────────────────────────
AZURE_ENDPOINT   = os.getenv("RANKIFY_AZURE_ENDPOINT", "")
AZURE_KEY        = os.getenv("RANKIFY_AZURE_KEY", "")
AZURE_DEPLOYMENT = os.getenv("RANKIFY_AZURE_DEPLOYMENT", "gpt-4o-2")
AZURE_API_VER    = os.getenv("RANKIFY_AZURE_API_VERSION", "2025-01-01-preview")

# ─── Component Cache ───────────────────────────────────────────────────────────
_retriever_cache: Dict[str, Any] = {}
_reranker_cache:  Dict[str, Any] = {}
_generator_cache: Dict[str, Any] = {}
_agent_cache:     Dict[str, Any] = {}


def get_retriever(method: str, n_docs: int = 10, index_type: str = "wiki"):
    key = f"{method}|{index_type}|{n_docs}"
    if key not in _retriever_cache:
        logger.info(f"Loading retriever: {method} [{index_type}]")
        from rankify.retrievers.retriever import Retriever
        _retriever_cache[key] = Retriever(method=method, n_docs=n_docs, index_type=index_type)
    return _retriever_cache[key]


def get_reranker(method: str, model_name: str):
    if method == "none":
        return None
    key = f"{method}|{model_name}"
    if key not in _reranker_cache:
        logger.info(f"Loading reranker: {method} / {model_name}")
        from rankify.models.reranking import Reranking
        _reranker_cache[key] = Reranking(method=method, model_name=model_name)
    return _reranker_cache[key]


def get_generator(rag_method: str, model_name: str, backend: str, **kwargs):
    key = f"{rag_method}|{model_name}|{backend}"
    if key not in _generator_cache:
        logger.info(f"Loading generator: {rag_method} / {model_name} / {backend}")
        from rankify.generator.generator import Generator
        _generator_cache[key] = Generator(method=rag_method, model_name=model_name, backend=backend, **kwargs)
    return _generator_cache[key]


def _resolve_generator_config(generator_id: str, rag_method_id: str) -> dict:
    """Map UI generator + RAG method IDs to Rankify Generator params.

    Rankify's model_factory (openai backend) expects:
      - api_keys: list of API key strings
      - (optional) base_url passed to OpenAIModel → OpenaiClient

    For Azure OpenAI the openai SDK supports:
      - AzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)
    We pass the Azure endpoint as base_url in the format that the openai
    SDK accepts when using standard OpenAI() with a custom base_url:
        https://<resource>.openai.azure.com/openai/deployments/<deployment>
    And set api_version via default_query.
    """
    # Azure base_url: route to the specific deployment
    azure_base = None
    if AZURE_ENDPOINT and AZURE_DEPLOYMENT:
        azure_base = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}"

    configs = {
        "azure": dict(
            rag_method="basic-rag", backend="openai",
            model_name=AZURE_DEPLOYMENT,
            api_keys=[AZURE_KEY],
            base_url=azure_base,
        ),
        "azure-cot": dict(
            rag_method="chain-of-thought-rag", backend="openai",
            model_name=AZURE_DEPLOYMENT,
            api_keys=[AZURE_KEY],
            base_url=azure_base,
        ),
        "azure-self-consistency": dict(
            rag_method="self-consistency-rag", backend="openai",
            model_name=AZURE_DEPLOYMENT,
            api_keys=[AZURE_KEY],
            base_url=azure_base,
        ),
        "openai": dict(
            rag_method="basic-rag", backend="openai",
            model_name="gpt-4o-mini",
            api_keys=[os.getenv("OPENAI_API_KEY", "")],
        ),
        "claude": dict(
            rag_method="basic-rag", backend="anthropic",
            model_name="claude-3-5-sonnet-20240620",
        ),
        "llama-3": dict(
            rag_method="basic-rag", backend="vllm",
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        "mistral": dict(
            rag_method="basic-rag", backend="vllm",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
        ),
        "fid": dict(
            rag_method="fid", backend="fid",
            model_name="nq_reader_base",
        ),
        "zero-shot": dict(
            rag_method="zero-shot", backend="openai",
            model_name=AZURE_DEPLOYMENT,
            api_keys=[AZURE_KEY],
            base_url=azure_base,
        ),
    }
    base = configs.get(generator_id, configs["azure"]).copy()
    # Allow RAG method override from UI (e.g. chain-of-thought on non-Azure model)
    if rag_method_id and rag_method_id not in ("auto", ""):
        base["rag_method"] = rag_method_id
    return base



# ─── Models ────────────────────────────────────────────────────────────────────
class PipelineRequest(BaseModel):
    query: str
    mode: str = "rag"            # retrieve | rerank | rag
    retriever: str = "bm25"
    rerankerCategory: str = "flashrank"
    rerankerModel: str = "ms-marco-MiniLM-L-12-v2"
    generator: str = "azure"     # azure | openai | claude | llama-3 | fid | zero-shot
    ragMethod: str = "auto"      # auto | basic-rag | chain-of-thought-rag | self-consistency-rag
    dataSource: str = "wiki"
    n_docs: int = 100
    n_contexts: int = 10


class DocOut(BaseModel):
    id: str
    text: str
    title: Optional[str] = None
    score: Optional[float] = None
    rank_delta: Optional[int] = None

class PipelineResponse(BaseModel):
    query: str
    mode: str
    retrieved_docs: List[DocOut] = []
    reranked_docs:  List[DocOut] = []
    answer: Optional[str] = None
    retriever_latency_ms: float = 0
    reranker_latency_ms:  float = 0
    generator_latency_ms: float = 0
    rag_method: Optional[str] = None
    error: Optional[str] = None

class AgentRequest(BaseModel):
    message: str
    session_id: str

class ArenaPipeline(BaseModel):
    retriever: str = "bm25"
    rerankerCategory: str = "none"
    rerankerModel: str = ""
    generator: str = "azure"
    ragMethod: str = "basic-rag"

class ArenaRequest(BaseModel):
    dataset: str = "nq"
    n_docs: int = 10
    n_queries: int = 5
    pipeline_a: ArenaPipeline
    pipeline_b: ArenaPipeline


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Rankify Demo API v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "azure_configured": bool(AZURE_KEY),
        "azure_deployment": AZURE_DEPLOYMENT,
        "cached_retrievers": list(_retriever_cache.keys()),
        "cached_rerankers":  list(_reranker_cache.keys()),
        "cached_generators": list(_generator_cache.keys()),
    }


@app.get("/models")
async def list_models():
    return {
        "retrievers": ["bm25", "dpr", "ance", "contriever", "colbert", "bge"],
        "generators": ["azure", "azure-cot", "azure-self-consistency", "openai", "claude", "llama-3", "mistral", "fid", "zero-shot"],
        "rag_methods": ["basic-rag", "chain-of-thought-rag", "self-consistency-rag", "zero-shot", "fid"],
    }


def _run_retrieve(req: PipelineRequest, from_rankify):
    """Run retrieval and return (retrieved_docs, latency_ms)."""
    Document, Question, Answer = from_rankify
    t0 = time.time()
    idx = "msmarco" if req.dataSource == "msmarco" else "wiki"
    retriever = get_retriever(req.retriever, n_docs=req.n_docs, index_type=idx)
    doc = Document(question=Question(req.query), answers=Answer([]), contexts=[])
    results = retriever.retrieve([doc])
    retrieved = results[0].contexts[:req.n_docs]
    latency = round((time.time() - t0) * 1000, 1)
    out = [
        DocOut(
            id=str(ctx.id or i),
            text=str(ctx.text or "")[:800],
            title=str(ctx.title) if hasattr(ctx, "title") and ctx.title else None,
            score=float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else None,
        )
        for i, ctx in enumerate(retrieved)
    ]
    return retrieved, out, latency


def _run_rerank(req: PipelineRequest, retrieved, from_rankify):
    """Run reranking and return (reranked_contexts, docs_out, latency_ms)."""
    Document, Question, Answer, Context = from_rankify
    if not retrieved:
        return [], [], 0.0
    t0 = time.time()
    reranker = get_reranker(req.rerankerCategory, req.rerankerModel)
    if reranker is None:
        return retrieved[:req.n_contexts], [], 0.0
    
    retrieved_ids = [str(ctx.id or i) for i, ctx in enumerate(retrieved)]

    doc2 = Document(
        question=Question(req.query),
        answers=Answer([]),
        contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                          title=ctx.title if hasattr(ctx, "title") else "")
                  for i, ctx in enumerate(retrieved)],
    )
    res = reranker.rank([doc2])
    reranked = (res[0].reorder_contexts or res[0].contexts)[:req.n_contexts]
    latency = round((time.time() - t0) * 1000, 1)
    
    out = []
    for i, ctx in enumerate(reranked):
        ctx_id = str(ctx.id or i)
        try:
            old_rank = retrieved_ids.index(ctx_id)
            delta = old_rank - i
        except ValueError:
            delta = 0
            
        out.append(DocOut(
            id=ctx_id,
            text=str(ctx.text or "")[:800],
            title=str(ctx.title) if hasattr(ctx, "title") and ctx.title else None,
            score=float(ctx.score) if hasattr(ctx, "score") and ctx.score is not None else None,
            rank_delta=delta
        ))
    return reranked, out, latency


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(req: PipelineRequest):
    from rankify.dataset.dataset import Document, Question, Answer, Context
    resp = PipelineResponse(query=req.query, mode=req.mode)

    try:
        # 1. Retrieve
        retrieved, resp.retrieved_docs, resp.retriever_latency_ms = \
            _run_retrieve(req, (Document, Question, Answer))

        if req.mode == "retrieve":
            return resp

        # 2. Rerank
        reranked, resp.reranked_docs, resp.reranker_latency_ms = \
            _run_rerank(req, retrieved, (Document, Question, Answer, Context))

        if req.mode == "rerank":
            return resp

        # 3. Generate
        t2 = time.time()
        gen_cfg = _resolve_generator_config(req.generator, req.ragMethod)
        rag_method = gen_cfg.pop("rag_method")
        backend    = gen_cfg.pop("backend")
        model_name = gen_cfg.pop("model_name")

        gen_doc = Document(
            question=Question(req.query),
            answers=Answer([]),
            contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                              title=ctx.title if hasattr(ctx, "title") else "")
                      for i, ctx in enumerate(reranked or retrieved[:req.n_contexts])],
        )
        generator = get_generator(rag_method, model_name, backend, **gen_cfg)
        answers = generator.generate([gen_doc])
        resp.answer = answers[0] if answers else "No answer generated."
        resp.generator_latency_ms = round((time.time() - t2) * 1000, 1)
        resp.rag_method = rag_method

    except Exception as e:
        logger.error(traceback.format_exc())
        resp.error = str(e)

    return resp


@app.post("/pipeline/stream")
async def pipeline_stream(req: PipelineRequest):
    """SSE streaming endpoint for real-time RAG responses."""
    async def gen():
        import asyncio
        from rankify.dataset.dataset import Document, Question, Answer, Context

        try:
            # Retrieve
            retrieved, retrieved_out, r_lat = _run_retrieve(req, (Document, Question, Answer))
            yield f"data: {json.dumps({'type':'retrieved','docs':[d.model_dump() for d in retrieved_out],'latency_ms':r_lat})}\n\n"

            if req.mode == "retrieve":
                yield 'data: {"type":"done"}\n\n'; return

            # Rerank
            reranked, reranked_out, rr_lat = _run_rerank(req, retrieved, (Document, Question, Answer, Context))
            yield f"data: {json.dumps({'type':'reranked','docs':[d.model_dump() for d in reranked_out],'latency_ms':rr_lat})}\n\n"

            if req.mode == "rerank":
                yield 'data: {"type":"done"}\n\n'; return

            # Generate
            gen_cfg = _resolve_generator_config(req.generator, req.ragMethod)
            rag_method = gen_cfg.pop("rag_method")
            backend    = gen_cfg.pop("backend")
            model_name = gen_cfg.pop("model_name")

            # Create generation payload for Retrived
            t_gen_start = time.time()
            gen_doc_ret = Document(
                question=Question(req.query),
                answers=Answer([]),
                contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                                  title=ctx.title if hasattr(ctx, "title") else "")
                          for i, ctx in enumerate(retrieved[:req.n_contexts])],
            )
            
            generator = get_generator(rag_method, model_name, backend, **gen_cfg)

            is_reranked = len(reranked_out) > 0
            if is_reranked:
                gen_doc_rr = Document(
                    question=Question(req.query),
                    answers=Answer([]),
                    contexts=[Context(text=ctx.text, id=str(ctx.id or i),
                                      title=ctx.title if hasattr(ctx, "title") else "")
                              for i, ctx in enumerate(reranked[:req.n_contexts])],
                )
                answers = generator.generate([gen_doc_ret, gen_doc_rr])
                ans_ret = answers[0] if answers else "No answer generated."
                ans_rr = answers[1] if len(answers) > 1 else ans_ret
            else:
                answers = generator.generate([gen_doc_ret])
                ans_ret = answers[0] if answers else "No answer generated."
                ans_rr = ""

            gen_latency_ms = round((time.time() - t_gen_start) * 1000, 1)

            # Stream tokens
            words_ret = ans_ret.split()
            words_rr = ans_rr.split()
            max_len = max(len(words_ret), len(words_rr))
            
            for i in range(max_len):
                if i < len(words_ret):
                    token_ret = words_ret[i] + (" " if i < len(words_ret) - 1 else "")
                    yield f"data: {json.dumps({'type': 'token_retrieved' if is_reranked else 'token', 'content':token_ret, 'method':rag_method})}\n\n"
                if is_reranked and i < len(words_rr):
                    token_rr = words_rr[i] + (" " if i < len(words_rr) - 1 else "")
                    yield f"data: {json.dumps({'type':'token_reranked', 'content':token_rr, 'method':rag_method})}\n\n"
                await asyncio.sleep(0.015)

            yield f"data: {json.dumps({'type':'metrics', 'generator_latency_ms': gen_latency_ms})}\n\n"
            yield 'data: {"type":"done"}\n\n'

        except Exception as e:
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/agent/chat")
async def agent_chat_stream(req: AgentRequest):
    """SSE endpoint for chatting with RankifyAgent."""
    async def gen():
        import asyncio
        from rankify.agent.agent import RankifyAgent
        try:
            if req.session_id not in _agent_cache:
                logger.info(f"Initializing RankifyAgent for session {req.session_id}")
                backend = "azure" if AZURE_KEY else "openai"
                _agent_cache[req.session_id] = RankifyAgent(backend=backend)
            
            agent = _agent_cache[req.session_id]
            # Synchronous call under the hood
            resp = agent.chat(req.message)
            
            # Stream the message word by word for a nice effect
            words = resp.message.split()
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.02)
                
            # Stream the recommendation / code snippet at the end if it exists
            if resp.recommendation or resp.code_snippet:
                rec_data = {
                    "code_snippet": resp.code_snippet,
                    "retriever": resp.recommendation.retriever.name if resp.recommendation and resp.recommendation.retriever else None,
                    "reranker": resp.recommendation.reranker.name if resp.recommendation and resp.recommendation.reranker else None,
                    "rag_method": resp.recommendation.rag_method.name if resp.recommendation and resp.recommendation.rag_method else None,
                }
                yield f"data: {json.dumps({'type': 'recommendation', 'data': rec_data})}\n\n"
            
            yield 'data: {"type": "done"}\n\n'
            
        except Exception as e:
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/arena/run")
async def arena_run(req: ArenaRequest):
    """Compare two pipelines on a dataset."""
    try:
        from rankify.dataset.dataset import Dataset
        from rankify.metrics.metrics import Metrics
        
        # We need a generic way to fetch queries. Dataset.load_dataset_qa can be used if we download it.
        # But Dataset.download() loads the whole dataset. 
        # We will use the Dataset class to download / get the documents.
        logger.info(f"Arena: Running benchmark on {req.dataset}")
        ds = Dataset(retriever="bm25", dataset_name=req.dataset, n_docs=req.n_docs)
        documents = ds.download(force_download=False)
        
        import random
        # Select N random documents to evaluate
        eval_docs = random.sample(documents, min(req.n_queries, len(documents)))
        
        def evaluate_pipeline(pipeline_cfg: ArenaPipeline, docs):
            import copy
            docs_copy = copy.deepcopy(docs)
            
            # Retrieval
            retriever = get_retriever(pipeline_cfg.retriever, n_docs=req.n_docs, index_type=req.dataset)
            t0 = time.time()
            ret_results = retriever.retrieve(docs_copy)
            ret_latency = (time.time() - t0) * 1000 / len(docs_copy)
            
            # Reranking
            rr_latency = 0
            reranker = get_reranker(pipeline_cfg.rerankerCategory, pipeline_cfg.rerankerModel)
            if reranker:
                t1 = time.time()
                ret_results = reranker.rank(ret_results)
                rr_latency = (time.time() - t1) * 1000 / len(docs_copy)
            
            # Evaluation
            metrics = Metrics(ret_results)
            use_rr = reranker is not None
            ret_metrics = metrics.calculate_retrieval_metrics(ks=[10], use_reordered=use_rr)
            
            # Gen Evaluation (we mock generation evaluation by F1/EM on just the retrieved contexts for speed)
            # A full Generation eval takes too long for an interactive demo (requires hitting the LLM API 5 times)
            # Wait, interactive demo will timeout. We will just evaluate Retrieval MRR@10 and F1 of contexts
            em_score = 0
            for doc in ret_results:
                golden = doc.answers.answers
                ctxs = doc.reorder_contexts if use_rr and doc.reorder_contexts else doc.contexts
                pred_text = " ".join([c.text for c in ctxs[:3]])
                normalized_pred = pred_text.lower()
                for ans in golden:
                    if ans.lower() in normalized_pred:
                        em_score += 1
                        break
                        
            return {
                "mrr_10": ret_metrics.get("top_10", 0),
                "context_em": (em_score / len(docs_copy)) * 100,
                "latency_ms": ret_latency + rr_latency
            }

        res_a = evaluate_pipeline(req.pipeline_a, eval_docs)
        res_b = evaluate_pipeline(req.pipeline_b, eval_docs)
        
        return {
            "num_queries": len(eval_docs),
            "pipeline_a": res_a,
            "pipeline_b": res_b
        }
        
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    logger.info(f"Starting on http://{args.host}:{args.port}")
    logger.info(f"Azure: {'configured ✓' if AZURE_KEY else 'NOT configured'} ({AZURE_DEPLOYMENT})")
    uvicorn.run("demo_server:app", host=args.host, port=args.port, reload=args.reload)
