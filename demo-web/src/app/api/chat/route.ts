export const runtime = 'edge';

export async function POST(req: Request) {
    // Parse the incoming request from the Vercel AI useChat hook
    const { messages, configuration } = await req.json();
    const latestMessage = messages[messages.length - 1];

    // In a real application, this would securely call the Rankify python backend
    // e.g: const pythonApi = process.env.VITE_API_URL || 'http://localhost:8000';

    const responseText = `Based on the provided context, the **${configuration.reranker}** reranker and **${configuration.retriever}** retriever combination against the **${configuration.dataSource}** dataset yields the following results for your query: "${latestMessage.content}".\n\nCross-encoders significantly improve retrieval augmented generation by providing a deeper, interaction-based relevance assessment between the query and the documents. Unlike bi-encoders used in initial retrieval, cross-encoders process the query and document simultaneously through the transformer layers, leading to much higher accuracy and better grounding for the final LLM generation phase.\n\n*Source: "Understanding Cross-Encoders vs Bi-Encoders in Search"*`;

    const encoder = new TextEncoder();

    // Custom ReadableStream to simulate streaming tokens back to the UI
    const stream = new ReadableStream({
        async start(controller) {
            const words = responseText.split(' ');

            for (let i = 0; i < words.length; i++) {
                // AI SDK 3.0+ Data Stream format: `0:"word "`
                const word = words[i] + (i === words.length - 1 ? '' : ' ');
                const payload = `0:"${word.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"\n`;

                controller.enqueue(encoder.encode(payload));

                // Artificial latency to mirror LLM generation
                await new Promise(r => setTimeout(r, 40));
            }
            controller.close();
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-vercel-ai-data-stream': 'v1'
        }
    });
}
