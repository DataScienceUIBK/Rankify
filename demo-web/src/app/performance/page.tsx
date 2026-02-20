import Link from "next/link";

// ─── Data (from LaTeX tables) ──────────────────────────────────────────────────
const RETRIEVER_NQ = [
    { name: "BM25", top1: 23.46, top5: null, top10: 56.32, top20: 64.8, top50: null, top100: 79.7 },
    { name: "DPR", top1: 44.57, top5: 67.76, top10: 74.52, top20: 79.5, top50: 84.4, top100: 86.81 },
    { name: "Contriever", top1: 38.81, top5: 65.65, top10: 73.91, top20: 79.6, top50: 84.88, top100: 88.01 },
    { name: "ANCE", top1: 50.80, top5: 71.86, top10: 78.12, top20: 82.52, top50: 86.23, top100: 88.28 },
    { name: "ColBERT", top1: 42.99, top5: 68.78, top10: 76.12, top20: 82.1, top50: 86.15, top100: 88.59 },
    { name: "BGE", top1: 48.03, top5: 72.22, top10: 78.50, top20: 82.66, top50: 86.90, top100: 89.45 },
];

const POINTWISE = [
    { method: "FlashRank TinyBERT", dl19: 67.68, dl20: 60.65, covid: 61.48, scifact: 64.08, news: 37.53, robust: 41.37 },
    { method: "FlashRank MiniLM", dl19: 70.40, dl20: 65.60, covid: 69.66, scifact: 59.14, news: 41.44, robust: 46.09 },
    { method: "MonoT5 Base-10k", dl19: 71.38, dl20: 66.31, covid: 74.61, scifact: 73.39, news: 46.09, robust: 51.69 },
    { method: "MonoT5 3B", dl19: 71.83, dl20: 68.89, covid: 80.71, scifact: 76.57, news: 48.49, robust: 56.71 },
    { method: "InRanker 3B", dl19: 72.71, dl20: 67.09, covid: 81.75, scifact: 78.31, news: 49.63, robust: 62.47 },
    { method: "mxbai-rerank-base", dl19: 72.49, dl20: 67.15, covid: 84.00, scifact: 72.33, news: 51.92, robust: 55.59 },
    { method: "bge-reranker-v2-m3", dl19: 72.19, dl20: 66.98, covid: 74.79, scifact: 73.48, news: 45.84, robust: 48.44 },
    { method: "MonoBERT 340M", dl19: 70.50, dl20: 67.28, covid: 70.01, scifact: 71.36, news: 44.62, robust: 49.35 },
    { method: "Cohere Rerank-v2", dl19: 73.22, dl20: 67.08, covid: 81.81, scifact: 74.44, news: 47.59, robust: 50.78 },
    { method: "TWOLAR-XL", dl19: 73.51, dl20: 70.84, covid: 82.70, scifact: 76.5, news: 50.8, robust: 57.9 },
];

const LISTWISE = [
    { method: "RankGPT Llama-3.2-1B", dl19: 47.1, dl20: 44.9, covid: 59.2, scifact: 67.4, news: 38.7, robust: 38.1 },
    { method: "RankGPT Llama 3.1 8B", dl19: 58.4, dl20: 59.6, covid: 69.6, scifact: 69.8, news: 43.9, robust: 49.5 },
    { method: "RankGPT GPT-3.5", dl19: 65.8, dl20: 62.9, covid: 76.6, scifact: 70.4, news: 48.8, robust: 50.6 },
    { method: "RankGPT GPT-4", dl19: 75.5, dl20: 70.5, covid: 85.5, scifact: 74.9, news: 52.8, robust: 57.5 },
    { method: "ListT5 Base", dl19: 71.8, dl20: 68.1, covid: 78.3, scifact: 74.1, news: 48.5, robust: 52.1 },
    { method: "ListT5 3B", dl19: 71.8, dl20: 69.1, covid: 84.7, scifact: 77.0, news: 53.2, robust: 57.8 },
    { method: "LiT5-Distill-XL", dl19: 72.4, dl20: 72.4, covid: 72.9, scifact: 71.8, news: 46.5, robust: 53.7 },
    { method: "Zephyr 7B", dl19: 74.2, dl20: 70.2, covid: 80.7, scifact: 75.1, news: 48.9, robust: 54.2 },
];

const BRIGHT = [
    { model: "BM25", type: "Sparse", avg: 14.5, bio: 18.9, earth: 27.2, econ: 14.9, psy: 12.5, leet: 24.4, color: "#94a3b8" },
    { model: "Contriever", type: "Dense (<1B)", avg: 11.1, bio: 9.2, earth: 13.6, econ: 10.5, psy: 12.1, leet: 24.5, color: "#6366f1" },
    { model: "SBERT", type: "Dense (<1B)", avg: 14.9, bio: 15.1, earth: 20.4, econ: 16.6, psy: 22.7, leet: 26.4, color: "#8b5cf6" },
    { model: "Nomic", type: "Dense (<1B)", avg: 14.2, bio: 16.2, earth: 19.2, econ: 16.9, psy: 19.0, leet: 25.2, color: "#a78bfa" },
    { model: "E5-Mistral", type: "LLM-Dense (>1B)", avg: 17.9, bio: 18.6, earth: 26.0, econ: 15.5, psy: 15.8, leet: 28.7, color: "#0ea5e9" },
    { model: "GTE-Qwen1.5", type: "LLM-Dense (>1B)", avg: 22.5, bio: 30.6, earth: 36.4, econ: 17.8, psy: 24.6, leet: 25.5, color: "#10b981" },
    { model: "GTE-Qwen2", type: "LLM-Dense (>1B)", avg: 23.3, bio: 31.8, earth: 40.7, econ: 16.2, psy: 26.6, leet: 31.1, color: "#059669" },
    { model: "ReasonIR", type: "Reasoning", avg: 24.1, bio: 25.4, earth: 27.8, econ: 20.3, psy: 29.7, leet: 33.2, color: "#f59e0b" },
    { model: "RaDeR", type: "Reasoning", avg: 23.5, bio: 23.4, earth: 26.1, econ: 17.3, psy: 25.5, leet: 31.6, color: "#ef4444" },
    { model: "Diver", type: "Reasoning", avg: 29.4, bio: 42.0, earth: 46.6, econ: 21.8, psy: 35.1, leet: 37.4, color: "#dc2626" },
];

const COLS = ["DL19", "DL20", "COVID", "SciFact", "News", "Robust04"] as const;

function ColorBar({ value, max, color }: { value: number; max: number; color: string }) {
    return (
        <div className="flex items-center gap-2">
            <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${(value / max) * 100}%`, backgroundColor: color }} />
            </div>
            <span className="text-xs font-bold text-slate-700 w-9 text-right">{value.toFixed(1)}</span>
        </div>
    );
}

function SectionHeader({ icon, title, sub }: { icon: string; title: string; sub?: string }) {
    return (
        <div className="mb-6">
            <div className="flex items-center gap-2 mb-1">
                <span className="text-xl">{icon}</span>
                <h2 className="text-2xl font-black text-slate-900">{title}</h2>
            </div>
            {sub && <p className="text-slate-500 text-sm ml-9">{sub}</p>}
        </div>
    );
}

export default function PerformancePage() {
    const typeColors: Record<string, string> = {
        "Sparse": "bg-slate-100 text-slate-600",
        "Dense (<1B)": "bg-indigo-50 text-indigo-700",
        "LLM-Dense (>1B)": "bg-sky-50 text-sky-700",
        "Reasoning": "bg-amber-50 text-amber-700",
    };

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Hero */}
            <div className="bg-gradient-to-r from-slate-900 to-indigo-950 text-white py-14 px-6">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center gap-2 mb-2">
                        <Link href="/" className="text-indigo-400 text-sm hover:underline">← Home</Link>
                    </div>
                    <h1 className="text-4xl font-black mb-2">📊 Benchmarks</h1>
                    <p className="text-slate-300 max-w-xl">
                        Comprehensive evaluation of retrievers, pointwise, listwise, and pairwise rerankers across TREC DL, BEIR, and BRIGHT benchmarks.
                    </p>
                    <Link href="/chat" className="inline-block mt-4 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold transition-colors">
                        Try the Models Live →
                    </Link>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 pt-12 space-y-16">

                {/* ── Retriever Performance ── */}
                <section>
                    <SectionHeader icon="🔍" title="Retriever Comparison" sub="Top-K accuracy on NQ. Rankify uses Pyserini as indexing backend, reproducing published results." />
                    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-slate-50 border-b border-slate-200">
                                        <th className="text-left px-4 py-3 font-semibold text-slate-700 w-32">Retriever</th>
                                        {["Top-1", "Top-5", "Top-10", "Top-20", "Top-50", "Top-100"].map(h => (
                                            <th key={h} className="text-center px-3 py-3 font-semibold text-slate-500 text-xs">{h}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {RETRIEVER_NQ.map((r, i) => (
                                        <tr key={r.name} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                                            <td className="px-4 py-3 font-semibold text-slate-800 text-xs">{r.name}</td>
                                            {[r.top1, r.top5, r.top10, r.top20, r.top50, r.top100].map((v, j) => (
                                                <td key={j} className="px-3 py-3 text-center text-xs">
                                                    {v != null ? (
                                                        <span className={`inline-block px-2 py-0.5 rounded font-semibold ${v >= 85 ? "bg-emerald-100 text-emerald-800" : v >= 75 ? "bg-indigo-50 text-indigo-700" : v >= 60 ? "bg-slate-100 text-slate-700" : "text-slate-500"}`}>
                                                            {v.toFixed(1)}
                                                        </span>
                                                    ) : <span className="text-slate-300">—</span>}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Bar chart for Top-20 */}
                    <div className="mt-6 bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
                        <h3 className="text-sm font-bold text-slate-600 mb-4">Top-20 Accuracy (NQ) — Visual comparison</h3>
                        {[...RETRIEVER_NQ].sort((a, b) => b.top20 - a.top20).map((r, i) => {
                            const colors = ["#6366f1", "#8b5cf6", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"];
                            return (
                                <div key={r.name} className="mb-2 flex items-center gap-3">
                                    <span className="w-20 text-xs text-slate-600 text-right font-medium shrink-0">{r.name}</span>
                                    <ColorBar value={r.top20} max={90} color={colors[i % colors.length]} />
                                </div>
                            );
                        })}
                    </div>
                </section>

                {/* ── Pointwise Rerankers ── */}
                <section>
                    <SectionHeader icon="📌" title="Pointwise Rerankers (nDCG@10)" sub="First-stage BM25 top-100 reranked. Key datasets from TREC DL and BEIR." />
                    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-slate-50 border-b border-slate-200">
                                        <th className="text-left px-4 py-3 font-semibold text-slate-700">Model</th>
                                        {COLS.map(c => <th key={c} className="text-center px-2 py-3 font-semibold text-slate-500 text-xs">{c}</th>)}
                                    </tr>
                                </thead>
                                <tbody>
                                    {POINTWISE.map((r, i) => (
                                        <tr key={r.method} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                                            <td className="px-4 py-2.5 font-medium text-slate-800 text-xs">{r.method}</td>
                                            {[r.dl19, r.dl20, r.covid, r.scifact, r.news, r.robust].map((v, j) => {
                                                const col = j === 0 ? POINTWISE.map(x => x.dl19) : j === 1 ? POINTWISE.map(x => x.dl20) : j === 2 ? POINTWISE.map(x => x.covid) : j === 3 ? POINTWISE.map(x => x.scifact) : j === 4 ? POINTWISE.map(x => x.news) : POINTWISE.map(x => x.robust);
                                                const isTop = v === Math.max(...col);
                                                return <td key={j} className="px-2 py-2.5 text-center text-xs">
                                                    <span className={`inline-block px-1.5 py-0.5 rounded font-semibold ${isTop ? "bg-indigo-600 text-white" : v >= 75 ? "bg-indigo-50 text-indigo-700" : v >= 65 ? "bg-slate-100 text-slate-700" : "text-slate-500"}`}>{v.toFixed(1)}</span>
                                                </td>;
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                {/* ── Listwise Rerankers ── */}
                <section>
                    <SectionHeader icon="📋" title="Listwise Rerankers (nDCG@10)" sub="LLM-based listwise reranking. GPT-4 achieves best overall with 75.5 on DL19." />
                    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-slate-50 border-b border-slate-200">
                                        <th className="text-left px-4 py-3 font-semibold text-slate-700">Model</th>
                                        {COLS.map(c => <th key={c} className="text-center px-2 py-3 font-semibold text-slate-500 text-xs">{c}</th>)}
                                    </tr>
                                </thead>
                                <tbody>
                                    {LISTWISE.map((r, i) => (
                                        <tr key={r.method} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                                            <td className="px-4 py-2.5 font-medium text-slate-800 text-xs">{r.method}</td>
                                            {[r.dl19, r.dl20, r.covid, r.scifact, r.news, r.robust].map((v, j) => {
                                                const col = j === 0 ? LISTWISE.map(x => x.dl19) : j === 1 ? LISTWISE.map(x => x.dl20) : j === 2 ? LISTWISE.map(x => x.covid) : j === 3 ? LISTWISE.map(x => x.scifact) : j === 4 ? LISTWISE.map(x => x.news) : LISTWISE.map(x => x.robust);
                                                const isTop = v === Math.max(...col);
                                                return <td key={j} className="px-2 py-2.5 text-center text-xs">
                                                    <span className={`inline-block px-1.5 py-0.5 rounded font-semibold ${isTop ? "bg-violet-600 text-white" : v >= 75 ? "bg-violet-50 text-violet-700" : v >= 65 ? "bg-slate-100 text-slate-700" : "text-slate-500"}`}>{v.toFixed(1)}</span>
                                                </td>;
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                {/* ── BRIGHT ── */}
                <section>
                    <SectionHeader icon="🌟" title="BRIGHT Benchmark (nDCG@10)" sub="Reasoning-intensive retrieval across 12 tasks. Avg across all tasks." />
                    {/* Model type legend */}
                    <div className="flex flex-wrap gap-2 mb-4">
                        {Object.entries(typeColors).map(([t, cls]) => (
                            <span key={t} className={`px-2 py-1 rounded text-xs font-semibold ${cls}`}>{t}</span>
                        ))}
                    </div>

                    {/* Column chart */}
                    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 mb-6">
                        <h3 className="text-xs font-bold text-slate-500 mb-4 uppercase tracking-wider">Average nDCG@10 (higher = better)</h3>
                        <div className="flex items-end gap-3 h-48">
                            {BRIGHT.map(m => (
                                <div key={m.model} className="flex-1 flex flex-col items-center gap-1">
                                    <span className="text-[10px] font-bold text-slate-700">{m.avg}</span>
                                    <div className="w-full rounded-t-md transition-all" style={{ height: `${(m.avg / 35) * 160}px`, backgroundColor: m.color }} />
                                    <span className="text-[9px] text-slate-500 text-center leading-tight mt-1 rotate-[-20deg] origin-top-left whitespace-nowrap">{m.model}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Full BRIGHT table */}
                    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-slate-50 border-b border-slate-200">
                                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-700">Model</th>
                                        <th className="text-left px-2 py-3 text-xs font-semibold text-slate-500">Type</th>
                                        <th className="text-center px-2 py-3 text-xs font-semibold text-indigo-600">Avg</th>
                                        {["Bio", "Earth", "Econ", "Psy", "Leet"].map(h => <th key={h} className="text-center px-2 py-3 text-xs font-semibold text-slate-400">{h}</th>)}
                                    </tr>
                                </thead>
                                <tbody>
                                    {[...BRIGHT].sort((a, b) => b.avg - a.avg).map((r, i) => (
                                        <tr key={r.model} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                                            <td className="px-4 py-2.5 font-semibold text-slate-800 text-xs">{r.model}</td>
                                            <td className="px-2 py-2.5"><span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${typeColors[r.type] ?? ""}`}>{r.type}</span></td>
                                            <td className="px-2 py-2.5 text-center">
                                                <span className={`inline-block px-2 py-0.5 rounded font-bold text-xs ${r.avg === Math.max(...BRIGHT.map(x => x.avg)) ? "bg-indigo-600 text-white" : r.avg >= 20 ? "bg-indigo-50 text-indigo-700" : "text-slate-600"}`}>{r.avg}</span>
                                            </td>
                                            {[r.bio, r.earth, r.econ, r.psy, r.leet].map((v, j) => (
                                                <td key={j} className="px-2 py-2.5 text-center text-xs text-slate-600">{v}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                {/* ── CTA ── */}
                <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-2xl p-8 text-white text-center">
                    <h3 className="text-2xl font-black mb-2">Run these models yourself</h3>
                    <p className="text-indigo-200 text-sm mb-4">Interactive demo with real-time retrieval, reranking and generation.</p>
                    <Link href="/chat" className="inline-block px-6 py-2.5 bg-white text-indigo-700 font-bold rounded-lg hover:bg-indigo-50 transition-colors">
                        Open RAG Console →
                    </Link>
                </div>
            </div>
        </div>
    );
}
