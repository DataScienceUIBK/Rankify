"use client";
import { useState, useMemo } from "react";
import RERANKERS_MAP from "./rerankers.json";
import { Play, Database, Search, ListTree, Loader2, Trophy, ArrowRight, CheckCircle2 } from "lucide-react";

const DATASETS = [
    { value: "dl19", label: "TREC Deep Learning 19 (dl19)" },
    { value: "dl20", label: "TREC Deep Learning 20 (dl20)" },
    { value: "beir-covid", label: "TREC-COVID (beir-covid)" },
    { value: "beir-nfc", label: "NFCorpus (beir-nfc)" },
    { value: "beir-touche", label: "Touché-2020 (beir-touche)" },
    { value: "beir-dbpedia", label: "DBPedia (beir-dbpedia)" },
    { value: "beir-scifact", label: "SciFact (beir-scifact)" },
    { value: "beir-signal", label: "Signal-1M (beir-signal)" },
    { value: "beir-news", label: "TREC-News (beir-news)" },
    { value: "beir-robust04", label: "Robust04 (beir-robust04)" },
    { value: "beir-arguana", label: "ArguAna (beir-arguana)" },
    { value: "beir-fever", label: "FEVER (beir-fever)" },
    { value: "beir-fiqa", label: "FiQA (beir-fiqa)" },
    { value: "beir-quora", label: "Quora (beir-quora)" },
    { value: "beir-scidocs", label: "SciDocs (beir-scidocs)" }
];

const METHODS = ["none", ...Object.keys(RERANKERS_MAP)];

function Sel({ value, onChange, opts, label, icon: Icon }: { value: string; onChange: (v: string) => void; opts: { value: string; label: string }[]; label: string; icon: any; }) {
    return (
        <div className="flex flex-col gap-1.5 flex-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                <Icon className="w-3 h-3" /> {label}
            </label>
            <select value={value} onChange={e => onChange(e.target.value)}
                className="h-10 px-3 text-sm border border-slate-200 rounded-lg bg-slate-50 text-slate-700 outline-none focus:border-indigo-400 focus:bg-white cursor-pointer w-full transition-colors">
                {opts.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
        </div>
    );
}

function MetricBar({ label, value, max, format = "number" }: { label: string; value: number; max: number; format?: "percent" | "number" | "ms" }) {
    const pct = Math.min(100, Math.max(0, (value / max) * 100));
    const displayValue = format === "percent" ? `${value.toFixed(1)}%` : format === "ms" ? `${value.toFixed(0)}ms` : value.toFixed(2);

    return (
        <div className="flex flex-col gap-1">
            <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-500">{label}</span>
                <span className="text-slate-800">{displayValue}</span>
            </div>
            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-500 rounded-full transition-all duration-1000 ease-out" style={{ width: `${pct}%` }} />
            </div>
        </div>
    );
}

export default function ArenaPage() {
    const [dataset, setDataset] = useState("beir-covid");
    const [queries, setQueries] = useState(5);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [pipeA, setPipeA] = useState({ retriever: "bm25", method: "none", model: "none" });
    const [pipeB, setPipeB] = useState({ retriever: "bm25", method: "flashrank", model: "ms-marco-MiniLM-L-12-v2" });

    const [results, setResults] = useState<any>(null);

    const handleRun = async () => {
        setLoading(true);
        setError(null);
        setResults(null);

        // Helper to map UI selections to backend schema
        const mapPipe = (p: typeof pipeA) => ({
            retriever: p.retriever,
            rerankerCategory: p.method,
            rerankerModel: p.model === "none" ? "" : p.model,
            generator: "azure",
            ragMethod: "basic-rag"
        });

        try {
            const res = await fetch("/api/arena/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset,
                    n_docs: 20, // top 20 for speed
                    n_queries: queries,
                    pipeline_a: mapPipe(pipeA),
                    pipeline_b: mapPipe(pipeB)
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to run benchmark");
            }

            const data = await res.json();
            setResults(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-[calc(100vh-56px)] bg-slate-50 p-6 md:p-12 font-sans">
            <div className="max-w-5xl mx-auto flex flex-col gap-8">

                {/* Header */}
                <div className="flex flex-col gap-2text-center md:text-left">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold w-fit mb-2">
                        <Trophy className="w-3.5 h-3.5" /> BEIR Evaluation
                    </div>
                    <h1 className="text-3xl font-black text-slate-900 tracking-tight">BEIR Dataset Evaluation Arena</h1>
                    <p className="text-slate-500 max-w-2xl">
                        Compare two reranking methods on standard BEIR datasets. Real-time evaluation of NDCG@10, MRR@10, and Latency using the Rankify evaluation engine without saving results to logs.
                    </p>
                </div>

                {/* Configuration Panel */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col md:flex-row">
                    {/* Common Settings */}
                    <div className="p-6 border-b md:border-b-0 md:border-r border-slate-100 bg-slate-50/50 min-w-[250px] flex flex-col gap-6">
                        <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
                            <Database className="w-4 h-4 text-indigo-500" /> Evaluation Dataset
                        </div>
                        <div className="flex flex-col gap-4">
                            <Sel value={dataset} onChange={setDataset} opts={DATASETS} label="Dataset" icon={Database} />
                            <div className="flex flex-col gap-1.5">
                                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">Number of Queries</label>
                                <input type="number" min="1" max="50" value={queries} onChange={e => setQueries(Number(e.target.value))}
                                    className="h-10 px-3 text-sm border border-slate-200 rounded-lg bg-white outline-none focus:border-indigo-400" />
                            </div>
                        </div>

                        <div className="mt-auto pt-4">
                            <button
                                onClick={handleRun}
                                disabled={loading}
                                className="w-full flex items-center justify-center gap-2 h-12 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-bold rounded-xl shadow-sm transition-all active:scale-95">
                                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                                {loading ? "Running Benchmark..." : "Run Evaluation"}
                            </button>
                        </div>
                    </div>

                    {/* Pipelines */}
                    <div className="flex-1 flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-slate-100">
                        {/* Pipeline A */}
                        <div className="flex-1 p-6 flex flex-col gap-6">
                            <div className="flex items-center gap-2 text-sm font-bold text-slate-800 border-b border-slate-100 pb-3">
                                <div className="w-6 h-6 rounded bg-slate-100 text-slate-600 flex items-center justify-center text-xs">A</div>
                                Pipeline A (Baseline)
                            </div>
                            <div className="flex flex-col gap-4">
                                <Sel value={pipeA.retriever} onChange={() => { }} opts={[{ value: "bm25", label: "BM25 (Sparse Base)" }]} label="Retriever (Fixed by BEIR)" icon={Search} />
                                <Sel
                                    value={pipeA.method}
                                    onChange={v => setPipeA(p => ({ ...p, method: v, model: v === "none" ? "none" : (RERANKERS_MAP[v as keyof typeof RERANKERS_MAP]?.[0] || "") }))}
                                    opts={METHODS.map(m => ({ value: m, label: m === "none" ? "None (Base BM25 Only)" : m }))}
                                    label="Reranking Method"
                                    icon={ListTree}
                                />
                                {pipeA.method !== "none" && (
                                    <Sel
                                        value={pipeA.model}
                                        onChange={v => setPipeA(p => ({ ...p, model: v }))}
                                        opts={(RERANKERS_MAP[pipeA.method as keyof typeof RERANKERS_MAP] || []).map(m => ({ value: m, label: m }))}
                                        label="Model Variant"
                                        icon={Database}
                                    />
                                )}
                            </div>
                        </div>

                        {/* Pipeline B */}
                        <div className="flex-1 p-6 flex flex-col gap-6 bg-indigo-50/10">
                            <div className="flex items-center gap-2 text-sm font-bold text-indigo-900 border-b border-indigo-100 pb-3">
                                <div className="w-6 h-6 rounded bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs">B</div>
                                Pipeline B (Challenger)
                            </div>
                            <div className="flex flex-col gap-4">
                                <Sel value={pipeB.retriever} onChange={() => { }} opts={[{ value: "bm25", label: "BM25 (Sparse Base)" }]} label="Retriever (Fixed by BEIR)" icon={Search} />
                                <Sel
                                    value={pipeB.method}
                                    onChange={v => setPipeB(p => ({ ...p, method: v, model: v === "none" ? "none" : (RERANKERS_MAP[v as keyof typeof RERANKERS_MAP]?.[0] || "") }))}
                                    opts={METHODS.map(m => ({ value: m, label: m === "none" ? "None (Base BM25 Only)" : m }))}
                                    label="Reranking Method"
                                    icon={ListTree}
                                />
                                {pipeB.method !== "none" && (
                                    <Sel
                                        value={pipeB.model}
                                        onChange={v => setPipeB(p => ({ ...p, model: v }))}
                                        opts={(RERANKERS_MAP[pipeB.method as keyof typeof RERANKERS_MAP] || []).map(m => ({ value: m, label: m }))}
                                        label="Model Variant"
                                        icon={Database}
                                    />
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {error && (
                    <div className="p-4 bg-red-50 text-red-600 rounded-xl border border-red-200 text-sm font-medium">
                        Failed to run benchmark: {error}. Check server logs.
                    </div>
                )}

                {/* Results Panel */}
                {results && !loading && (
                    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                            <h2 className="text-xl font-bold text-slate-800">Results ({results.num_queries} queries evaluated)</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Result Card A */}
                            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col gap-6 relative overflow-hidden">
                                {results.pipeline_a.mrr_10 > results.pipeline_b.mrr_10 && <div className="absolute top-0 right-0 w-16 h-16 bg-emerald-500 transform rotate-45 translate-x-8 -translate-y-8" />}

                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-600 flex items-center justify-center font-bold text-lg">A</div>
                                    <div>
                                        <div className="font-bold text-slate-800">Pipeline A</div>
                                        <div className="text-xs text-slate-500">BM25 {pipeA.method !== "none" ? `+ ${pipeA.method} (${pipeA.model})` : "(No Reranker)"}</div>
                                    </div>
                                </div>
                                <div className="flex flex-col gap-4 mt-2">
                                    <MetricBar label="NDCG@10 (Ranking Quality)" value={results.pipeline_a.ndcg_10} max={100} format="percent" />
                                    <MetricBar label="MRR@10 (Top Relevance)" value={results.pipeline_a.mrr_10} max={100} format="percent" />
                                    <MetricBar label="Avg End-to-End Latency" value={results.pipeline_a.latency_ms} max={3000} format="ms" />
                                </div>
                            </div>

                            {/* Result Card B */}
                            <div className="bg-white p-6 rounded-2xl border border-indigo-200 shadow-md flex flex-col gap-6 relative overflow-hidden ring-1 ring-indigo-500 ring-opacity-20">
                                {results.pipeline_b.mrr_10 > results.pipeline_a.mrr_10 && <div className="absolute top-0 right-0 w-16 h-16 bg-emerald-500 transform rotate-45 translate-x-8 -translate-y-8 flex items-end justify-center pb-1"><Trophy className="w-4 h-4 text-white -rotate-45" /></div>}

                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-lg">B</div>
                                    <div>
                                        <div className="font-bold text-slate-800">Pipeline B</div>
                                        <div className="text-xs text-slate-500">BM25 {pipeB.method !== "none" ? `+ ${pipeB.method} (${pipeB.model})` : "(No Reranker)"}</div>
                                    </div>
                                </div>
                                <div className="flex flex-col gap-4 mt-2">
                                    <MetricBar label="NDCG@10 (Ranking Quality)" value={results.pipeline_b.ndcg_10} max={100} format="percent" />
                                    <MetricBar label="MRR@10 (Top Relevance)" value={results.pipeline_b.mrr_10} max={100} format="percent" />
                                    <MetricBar label="Avg End-to-End Latency" value={results.pipeline_b.latency_ms} max={3000} format="ms" />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}
