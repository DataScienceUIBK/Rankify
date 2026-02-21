export const runtime = 'nodejs';

const PYTHON_API = process.env.RANKIFY_API_URL || 'http://localhost:8000';

export async function POST(req: Request) {
    try {
        const body = await req.json();

        const res = await fetch(`${PYTHON_API}/api/agent/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const errText = await res.text().catch(() => `HTTP ${res.status}`);
            return new Response(`data: {"type": "error", "message": ${JSON.stringify(errText)}}\n\n`, {
                status: 200,
                headers: { 'Content-Type': 'text/event-stream' }
            });
        }

        return new Response(res.body, {
            status: 200,
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        });
    } catch (error: any) {
        return new Response(`data: {"type": "error", "message": ${JSON.stringify(error.message)}}\n\n`, {
            status: 200,
            headers: { 'Content-Type': 'text/event-stream' }
        });
    }
}
