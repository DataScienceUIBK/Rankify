// Health-check route: proxies to Python backend
export const runtime = 'nodejs';

const PYTHON_API = process.env.RANKIFY_API_URL || 'http://localhost:8000';

export async function GET() {
    try {
        const res = await fetch(`${PYTHON_API}/health`, {
            signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
            const data = await res.json();
            return Response.json({ status: 'online', ...data });
        }
        return Response.json({ status: 'offline' }, { status: 502 });
    } catch {
        return Response.json({ status: 'offline', error: 'Cannot reach Python server' }, { status: 503 });
    }
}
