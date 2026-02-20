// Next.js App Router API Route: /api/chat
// Proxies the UI chat request to the Python demo_server.py
// Supports both streaming (SSE) and non-streaming responses.

// Use Node.js runtime (not edge) because we need to make server-to-server fetch calls
export const runtime = 'nodejs';

const PYTHON_API = process.env.RANKIFY_API_URL || 'http://localhost:8000';

export async function POST(req: Request) {
    const body = await req.json();
    const { messages, configuration } = body;

    if (!messages || messages.length === 0) {
        return new Response(JSON.stringify({ error: 'No messages provided' }), { status: 400 });
    }

    // Get the latest user message
    const query = messages[messages.length - 1]?.content ?? '';
    if (!query.trim()) {
        return new Response(JSON.stringify({ error: 'Empty query' }), { status: 400 });
    }

    const {
        pipelineMode = 'rag',
        dataSource = 'wiki',
        retriever = 'bm25',
        rerankerCategory = 'flashrank',
        rerankerModel = 'ms-marco-MiniLM-L-12-v2',
        generator = 'openai',
    } = configuration ?? {};

    const payload = {
        query,
        mode: pipelineMode,
        retriever,
        rerankerCategory,
        rerankerModel,
        generator,
        dataSource: dataSource === 'wiki' ? 'wiki' : 'msmarco',
        n_docs: 10,
        n_contexts: 5,
    };

    // ── Check if Python server is reachable ──────────────────────────────────────
    try {
        const healthRes = await fetch(`${PYTHON_API}/health`, {
            signal: AbortSignal.timeout(3000),
        });
        if (!healthRes.ok) throw new Error('Server not healthy');
    } catch {
        // Python server unavailable → return a clear error through the AI stream format
        return streamText(
            `⚠️ **Rankify Python server is not running.**\n\n` +
            `Please start it with:\n\`\`\`\npython demo_server.py --port 8000\n\`\`\`\n\n` +
            `Make sure Rankify is installed: \`pip install "rankify[all]"\``
        );
    }

    // ── Use SSE streaming endpoint for full RAG, regular POST for retrieve/rerank ─
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
            try {
                const res = await fetch(`${PYTHON_API}/pipeline/stream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(120_000),
                });

                if (!res.ok || !res.body) {
                    throw new Error(`Server returned ${res.status}`);
                }

                const reader = res.body.getReader();
                const dec = new TextDecoder();

                let retrievedDocs: unknown[] = [];
                let rerankedDocs: unknown[] = [];
                let answerSoFar = '';
                let headerSent = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = dec.decode(value, { stream: true });
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const raw = line.slice(6).trim();
                        if (!raw) continue;

                        let event: Record<string, unknown>;
                        try { event = JSON.parse(raw); } catch { continue; }

                        if (event.type === 'retrieved') {
                            retrievedDocs = event.docs as unknown[];
                            // Send pipeline metadata as a special annotation chunk
                            const meta = JSON.stringify({ pipelineMeta: { retrievedDocs, rerankedDocs } });
                            controller.enqueue(encoder.encode(`2:${JSON.stringify([meta])}\n`));

                        } else if (event.type === 'reranked') {
                            rerankedDocs = event.docs as unknown[];
                            const meta = JSON.stringify({ pipelineMeta: { retrievedDocs, rerankedDocs } });
                            controller.enqueue(encoder.encode(`2:${JSON.stringify([meta])}\n`));

                        } else if (event.type === 'token') {
                            const token = (event.content as string) ?? '';
                            answerSoFar += token;
                            // Write in Vercel AI SDK data-stream format: `0:"word "`
                            if (!headerSent) { headerSent = true; }
                            const escaped = token.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
                            controller.enqueue(encoder.encode(`0:"${escaped}"\n`));

                        } else if (event.type === 'done') {
                            break;

                        } else if (event.type === 'error') {
                            const err = event.message ?? 'Unknown error';
                            const escaped = String(err).replace(/"/g, '\\"');
                            controller.enqueue(encoder.encode(`0:"\\n\\n⚠️ Error: ${escaped}"\n`));
                        }
                    }
                }
            } catch (err) {
                const msg = `⚠️ Failed to connect to Rankify server: ${err}`;
                const escaped = msg.replace(/"/g, '\\"');
                controller.enqueue(encoder.encode(`0:"${escaped}"\n`));
            } finally {
                controller.close();
            }
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-vercel-ai-data-stream': 'v1',
        },
    });
}

// ─── Blocking handler (for retrieve/rerank modes) ─────────────────────────────
async function handleBlockingPipeline(payload: Record<string, unknown>, mode: string) {
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
        async start(controller) {
            try {
                const res = await fetch(`${PYTHON_API}/pipeline`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(60_000),
                });

                const data = await res.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                // Send pipeline metadata annotation
                const meta = JSON.stringify({
                    pipelineMeta: {
                        retrievedDocs: data.retrieved_docs ?? [],
                        rerankedDocs: data.reranked_docs ?? [],
                    }
                });
                controller.enqueue(encoder.encode(`2:${JSON.stringify([meta])}\n`));

                // Build a descriptive text summary as the "answer"
                const numDocs = (data.retrieved_docs ?? []).length;
                const numReranked = (data.reranked_docs ?? []).length;

                let summary = '';
                if (mode === 'retrieve') {
                    summary = `Retrieved **${numDocs} documents** using **${payload.retriever}** from **${payload.dataSource}**.\n\n`;
                    summary += `Top results shown above. ⬆️ Click "Documents Retrieved" to expand.`;
                } else {
                    summary = `Retrieved **${numDocs} documents** then reranked to **${numReranked} top passages** using **${payload.rerankerCategory}** (${payload.rerankerModel}).\n\n`;
                    summary += `Retrieval: ${data.retriever_latency_ms}ms · Reranking: ${data.reranker_latency_ms}ms`;
                }

                // Stream the summary as tokens
                const words = summary.split(' ');
                for (let i = 0; i < words.length; i++) {
                    const word = words[i] + (i < words.length - 1 ? ' ' : '');
                    const escaped = word.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
                    controller.enqueue(encoder.encode(`0:"${escaped}"\n`));
                    await sleep(15);
                }

            } catch (err) {
                const msg = `⚠️ Error: ${err}`;
                controller.enqueue(encoder.encode(`0:"${msg.replace(/"/g, '\\"')}"\n`));
            } finally {
                controller.close();
            }
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-vercel-ai-data-stream': 'v1',
        },
    });
}

// ─── Helpers ───────────────────────────────────────────────────────────────────
function streamText(text: string): Response {
    const encoder = new TextEncoder();
    const words = text.split(' ');
    const stream = new ReadableStream({
        async start(controller) {
            for (let i = 0; i < words.length; i++) {
                const word = words[i] + (i < words.length - 1 ? ' ' : '');
                const escaped = word.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
                controller.enqueue(encoder.encode(`0:"${escaped}"\n`));
                await sleep(20);
            }
            controller.close();
        }
    });
    return new Response(stream, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8', 'x-vercel-ai-data-stream': 'v1' }
    });
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }
