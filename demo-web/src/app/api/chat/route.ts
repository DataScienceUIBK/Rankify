// Next.js App Router API Route: /api/chat
// Proxies the UI chat request to the Python demo_server.py
// Supports both streaming (SSE) and non-streaming responses.

export const runtime = 'nodejs';

const PYTHON_API = process.env.RANKIFY_API_URL || 'http://localhost:8000';

export async function POST(req: Request) {
    let body: Record<string, unknown>;
    try {
        body = await req.json();
    } catch {
        return new Response(JSON.stringify({ error: 'Invalid JSON' }), { status: 400 });
    }

    const { messages, configuration } = body as {
        messages?: { role: string; content: string }[];
        configuration?: Record<string, string>;
    };

    if (!messages || messages.length === 0) {
        return new Response(JSON.stringify({ error: 'No messages provided' }), { status: 400 });
    }

    // The LAST user message is the query. We do NOT reuse the whole history —
    // each retrieval is independent. This is the correct behavior.
    const query = messages[messages.length - 1]?.content?.trim() ?? '';
    if (!query) {
        return new Response(JSON.stringify({ error: 'Empty query' }), { status: 400 });
    }

    const {
        pipelineMode = 'rag',
        dataSource = 'wiki',
        retriever = 'bm25',
        rerankerCategory = 'flashrank',
        rerankerModel = 'ms-marco-MiniLM-L-12-v2',
        generator = 'azure',
    } = configuration ?? {};

    const payload = {
        query,
        mode: pipelineMode,
        retriever,
        rerankerCategory,
        rerankerModel,
        generator,
        dataSource: dataSource === 'wiki' ? 'wiki' : 'msmarco',
        n_docs: 100,      // Always retrieve 100 candidates; reranker picks best n_contexts
        n_contexts: 10,
    };

    // NOTE: No per-request health check here — it adds latency AND the AbortSignal
    // can interfere with downstream requests. The /api/health-check route is polled
    // independently by the UI every 30 seconds.

    if (pipelineMode === 'rag') {
        return handleStreamingPipeline(payload);
    } else {
        return handleBlockingPipeline(payload, pipelineMode);
    }
}

// ─── Streaming handler (for RAG mode) ─────────────────────────────────────────
async function handleStreamingPipeline(payload: Record<string, unknown>) {
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
        async start(controller) {
            const enqueue = (data: string) => controller.enqueue(encoder.encode(data));

            try {
                const res = await fetch(`${PYTHON_API}/pipeline/stream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(180_000),
                });

                if (!res.ok || !res.body) {
                    const errText = await res.text().catch(() => `HTTP ${res.status}`);
                    enqueue(`0:"⚠️ Server error: ${escapeJson(errText)}"\n`);
                    return;
                }

                const reader = res.body.getReader();
                const dec = new TextDecoder();

                let retrievedDocs: unknown[] = [];
                let rerankedDocs: unknown[] = [];
                let ragMethod: string | undefined;

                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += dec.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    // Keep last partial line in buffer
                    buffer = lines.pop() ?? '';

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const raw = line.slice(6).trim();
                        if (!raw) continue;

                        let event: Record<string, unknown>;
                        try { event = JSON.parse(raw); } catch { continue; }

                        if (event.type === 'retrieved') {
                            retrievedDocs = event.docs as unknown[];
                            const meta = JSON.stringify({ pipelineMeta: { retrievedDocs, rerankedDocs } });
                            enqueue(`2:${JSON.stringify([meta])}\n`);

                        } else if (event.type === 'reranked') {
                            rerankedDocs = event.docs as unknown[];
                            const meta = JSON.stringify({ pipelineMeta: { retrievedDocs, rerankedDocs } });
                            enqueue(`2:${JSON.stringify([meta])}\n`);

                        } else if (event.type === 'token') {
                            ragMethod = (event.method as string) ?? ragMethod;
                            const token = (event.content as string) ?? '';
                            enqueue(`0:"${escapeJson(token)}"\n`);

                        } else if (event.type === 'error') {
                            const errMsg = String(event.message ?? 'Unknown server error');
                            // Send final pipeline metadata (in case we have partial docs)
                            const meta = JSON.stringify({ pipelineMeta: { retrievedDocs, rerankedDocs, ragMethod } });
                            enqueue(`2:${JSON.stringify([meta])}\n`);
                            enqueue(`0:"\\n\\n⚠️ Error: ${escapeJson(errMsg)}"\n`);
                        }
                        // 'done' event — just stop loop naturally
                    }
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                const hint = msg.toLowerCase().includes('fetch') || msg.toLowerCase().includes('connect')
                    ? '\\n\\nMake sure the Python server is running: `python demo_server.py --port 8000`'
                    : '';
                enqueue(`0:"⚠️ ${escapeJson(msg)}${hint}"\n`);
            } finally {
                controller.close();
            }
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-vercel-ai-data-stream': 'v1',
            'Cache-Control': 'no-cache',
        },
    });
}

// ─── Blocking handler (for retrieve/rerank modes) ─────────────────────────────
async function handleBlockingPipeline(payload: Record<string, unknown>, mode: string) {
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
        async start(controller) {
            const enqueue = (data: string) => controller.enqueue(encoder.encode(data));

            try {
                const res = await fetch(`${PYTHON_API}/pipeline`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(90_000),
                });

                if (!res.ok) {
                    const errText = await res.text().catch(() => `HTTP ${res.status}`);
                    enqueue(`0:"⚠️ Server error: ${escapeJson(errText)}"\n`);
                    return;
                }

                const data = await res.json() as {
                    retrieved_docs?: unknown[];
                    reranked_docs?: unknown[];
                    retriever_latency_ms?: number;
                    reranker_latency_ms?: number;
                    error?: string;
                };

                if (data.error) throw new Error(data.error);

                // Send pipeline metadata annotation first
                const meta = JSON.stringify({
                    pipelineMeta: {
                        retrievedDocs: data.retrieved_docs ?? [],
                        rerankedDocs: data.reranked_docs ?? [],
                    }
                });
                enqueue(`2:${JSON.stringify([meta])}\n`);

                // Build descriptive summary
                const numDocs = (data.retrieved_docs ?? []).length;
                const numReranked = (data.reranked_docs ?? []).length;
                const retriever = String(payload.retriever ?? 'bm25');
                const ds = String(payload.dataSource ?? 'wiki');
                const rrCat = String(payload.rerankerCategory ?? '');
                const rrModel = String(payload.rerankerModel ?? '');
                const rMs = data.retriever_latency_ms ?? 0;
                const rrMs = data.reranker_latency_ms ?? 0;

                let summary: string;
                if (mode === 'retrieve') {
                    summary = numDocs > 0
                        ? `Retrieved **${numDocs} documents** using **${retriever}** from **${ds}**.\n\nExpand the ⬆️ panel above to view results.`
                        : `No documents found for this query using **${retriever}** on **${ds}**. Try a different query or retriever.`;
                } else {
                    summary = numDocs > 0
                        ? `Retrieved **${numDocs} documents** then reranked to **${numReranked} top passages** using **${rrCat}** (${rrModel}).\n\nRetrieval: ${rMs.toFixed(0)}ms · Reranking: ${rrMs.toFixed(0)}ms`
                        : `No documents retrieved for this query. Try a different query or retriever.`;
                }

                // Stream summary word by word
                const words = summary.split(' ');
                for (let i = 0; i < words.length; i++) {
                    const token = words[i] + (i < words.length - 1 ? ' ' : '');
                    enqueue(`0:"${escapeJson(token)}"\n`);
                    await sleep(12);
                }

            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                const hint = msg.toLowerCase().includes('fetch') || msg.toLowerCase().includes('connect')
                    ? ' Make sure the Python server is running: python demo_server.py --port 8000'
                    : '';
                enqueue(`0:"⚠️ Error: ${escapeJson(msg + hint)}"\n`);
            } finally {
                controller.close();
            }
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-vercel-ai-data-stream': 'v1',
            'Cache-Control': 'no-cache',
        },
    });
}

// ─── Helpers ───────────────────────────────────────────────────────────────────
/** Escape a string for embedding inside a JSON double-quoted string. */
function escapeJson(s: string): string {
    return s
        .replace(/\\/g, '\\\\')
        .replace(/"/g, '\\"')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r');
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }
