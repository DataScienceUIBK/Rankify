export const runtime = 'nodejs';

const PYTHON_API = process.env.RANKIFY_API_URL || 'http://localhost:8000';

export async function POST(req: Request) {
    try {
        const body = await req.json();

        const res = await fetch(`${PYTHON_API}/api/arena/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const errText = await res.text().catch(() => `HTTP ${res.status}`);
            return new Response(JSON.stringify({ detail: errText }), { status: res.status });
        }

        const data = await res.json();
        return new Response(JSON.stringify(data), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error: any) {
        return new Response(JSON.stringify({ detail: error.message }), { status: 500 });
    }
}
