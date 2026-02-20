import Link from "next/link";

// ─── Data ──────────────────────────────────────────────────────────────────────
const DATASET_CATEGORIES = [
  { name: "Question Answering", count: 17, color: "bg-indigo-500", datasets: ["NQ", "TriviaQA", "WebQ", "SQuAD", "NarrativeQA", "MSMARCO-QA", "PopQA", "SIQA", "Fermi", "WikiQA", "AmbigQA", "CommonsenseQA", "PIQA", "BoolQ", "StrategyQA", "CuratedTREC", "WikiPassageQA"] },
  { name: "Multi-Hop QA", count: 4, color: "bg-violet-500", datasets: ["2WikiMultiHopQA", "Bamboogle", "Musique", "HotpotQA"] },
  { name: "Temporal QA", count: 2, color: "bg-sky-500", datasets: ["ArchivalQA", "ChroniclingQA"] },
  { name: "Long-Form QA", count: 2, color: "bg-teal-500", datasets: ["ELI5", "ASQA"] },
  { name: "Multiple-Choice", count: 6, color: "bg-emerald-500", datasets: ["MMLU", "TruthfulQA", "HellaSwag", "ARC", "OpenBookQA", "QuaRTz"] },
  { name: "Entity Linking", count: 3, color: "bg-amber-500", datasets: ["WNED", "AIDA CoNLL-YAGO", "EntityQuestions"] },
  { name: "Slot Filling", count: 2, color: "bg-orange-500", datasets: ["Zero-shot RE", "T-REx"] },
  { name: "Dialog Generation", count: 1, color: "bg-rose-500", datasets: ["WOW"] },
  { name: "Fact Verification", count: 1, color: "bg-red-500", datasets: ["FEVER"] },
  { name: "Summarization", count: 1, color: "bg-pink-500", datasets: ["WikiAsp"] },
  { name: "IR Benchmarks", count: 2, color: "bg-slate-500", datasets: ["BEIR", "DL19/20"] },
  { name: "Specialized", count: 1, color: "bg-purple-500", datasets: ["DomainRAG"] },
];

const RETRIEVER_STATS = [
  { name: "BM25", nq: 63.0, tqa: 76.4, webq: 62.3, color: "#6366f1" },
  { name: "DPR", nq: 79.5, tqa: 79.7, webq: 75.1, color: "#8b5cf6" },
  { name: "Contriever", nq: 79.6, tqa: 80.3, webq: 75.9, color: "#0ea5e9" },
  { name: "ANCE", nq: 82.5, tqa: 80.0, webq: 76.6, color: "#10b981" },
  { name: "ColBERT", nq: 82.1, tqa: 82.2, webq: 76.6, color: "#f59e0b" },
  { name: "BGE", nq: 78.5, tqa: 79.8, webq: 72.5, color: "#ef4444" },
];

const RERANKER_HIGHLIGHTS = [
  { name: "MonoT5 3B", dl19: 71.83, dl20: 68.89, covid: 80.71 },
  { name: "InRanker 3B", dl19: 72.71, dl20: 67.09, covid: 81.75 },
  { name: "RankGPT GPT-4", dl19: 75.5, dl20: 70.5, covid: 85.5 },
  { name: "TWOLAR-XL", dl19: 73.51, dl20: 70.84, covid: 82.70 },
  { name: "Zephyr 7B", dl19: 74.2, dl20: 70.2, covid: 80.7 },
  { name: "LiT5-Distill-XL", dl19: 72.4, dl20: 72.4, covid: 72.9 },
  { name: "Cohere Rerank-v2", dl19: 73.22, dl20: 67.08, covid: 81.81 },
  { name: "mxbai-rerank-base", dl19: 72.49, dl20: 67.15, covid: 84.0 },
];

const BRIGHT_LEADERS = [
  { name: "Diver", avg: 29.4, color: "#6366f1" },
  { name: "RaDeR", avg: 23.5, color: "#8b5cf6" },
  { name: "ReasonIR", avg: 24.1, color: "#0ea5e9" },
  { name: "GTE-Qwen2", avg: 23.3, color: "#10b981" },
  { name: "GTE-Qwen1.5", avg: 22.5, color: "#f59e0b" },
  { name: "BM25", avg: 14.5, color: "#94a3b8" },
];

// ─── Components ────────────────────────────────────────────────────────────────
function StatCard({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div className="flex flex-col items-center p-6 bg-white rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow text-center">
      <div className="text-4xl font-black text-indigo-600 mb-1">{value}</div>
      <div className="text-sm font-semibold text-slate-700">{label}</div>
      {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
    </div>
  );
}

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${(value / max) * 100}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-semibold text-slate-600 w-9 text-right">{value.toFixed(1)}</span>
    </div>
  );
}

export default function LandingPage() {
  const totalDatasets = DATASET_CATEGORIES.reduce((a, c) => a + c.count, 0);
  return (
    <div className="min-h-screen bg-slate-50">

      {/* ── Hero ── */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white pt-24 pb-20">
        <div className="absolute inset-0 opacity-20"
          style={{ backgroundImage: "radial-gradient(circle at 25% 60%, #6366f1 0%, transparent 50%), radial-gradient(circle at 75% 30%, #8b5cf6 0%, transparent 50%)" }} />
        <div className="relative max-w-5xl mx-auto px-6 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-xs font-semibold mb-6">
            🔬 Open-Source Research Tool
          </div>
          <h1 className="text-5xl md:text-6xl font-black tracking-tight mb-4">
            <span className="bg-gradient-to-r from-white to-indigo-300 bg-clip-text text-transparent">Rankify</span>
          </h1>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto mb-8 leading-relaxed">
            A unified Python toolkit for <strong className="text-white">retrieval</strong>, <strong className="text-white">reranking</strong>, and <strong className="text-white">retrieval-augmented generation</strong> — with 42 datasets, 6 retrievers, and 140+ reranker models.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link href="/chat"
              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-semibold text-white transition-all shadow-lg shadow-indigo-900/40">
              🚀 Open Live Demo
            </Link>
            <Link href="/performance"
              className="px-6 py-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded-xl font-semibold text-white transition-all">
              📊 View Benchmarks
            </Link>
            <a href="https://github.com/DataScienceUIBK/Rankify" target="_blank"
              className="px-6 py-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded-xl font-semibold text-white transition-all">
              ⭐ GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ── Stats row ── */}
      <section className="max-w-5xl mx-auto px-6 -mt-8 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard value="42" label="Datasets" sub="Across 12 NLP task categories" />
          <StatCard value="15" label="Retrievers" sub="BM25, DPR, ANCE, ColBERT, Diver, ReasonIR…" />
          <StatCard value="140+" label="Reranker Models" sub="Pointwise, Listwise, Pairwise" />
          <StatCard value="7" label="RAG Methods" sub="Basic, CoT, Self-Consistency, FiD, ReAct…" />
        </div>
      </section>

      {/* ── Dataset overview ── */}
      <section className="max-w-5xl mx-auto px-6 pt-16 pb-8">
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-2xl font-bold text-slate-900">Dataset Ecosystem</h2>
          <span className="px-2 py-0.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-full border border-indigo-100">{totalDatasets} total</span>
        </div>
        <p className="text-slate-500 text-sm mb-8">Spanning 12 NLP task categories — from opening-domain QA to IR benchmarking.</p>

        {/* Category bubbles */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-10">
          {DATASET_CATEGORIES.map(cat => (
            <div key={cat.name} className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between mb-2">
                <div className={`w-3 h-3 rounded-full ${cat.color}`} />
                <span className="text-2xl font-black text-slate-800">{cat.count}</span>
              </div>
              <div className="text-xs font-semibold text-slate-700 mb-1">{cat.name}</div>
              <div className="text-[10px] text-slate-400 line-clamp-2">{cat.datasets.slice(0, 3).join(", ")}{cat.datasets.length > 3 ? ` +${cat.datasets.length - 3}` : ""}</div>
            </div>
          ))}
        </div>

        {/* Category bar chart */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <h3 className="text-sm font-bold text-slate-700 mb-4">Dataset Count by Category</h3>
          <div className="space-y-2">
            {DATASET_CATEGORIES.sort((a, b) => b.count - a.count).map(cat => (
              <div key={cat.name} className="flex items-center gap-3">
                <div className="w-32 text-xs text-slate-600 text-right shrink-0">{cat.name}</div>
                <div className="flex-1 h-6 bg-slate-50 rounded-md overflow-hidden">
                  <div className={`h-full rounded-md flex items-center px-2 text-white text-xs font-bold transition-all ${cat.color}`}
                    style={{ width: `${(cat.count / 17) * 100}%` }}>
                    {cat.count}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Retriever Performance Preview ── */}
      <section className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-bold text-slate-900 mb-1">Retriever Performance (Top-20/100)</h2>
        <p className="text-slate-500 text-sm mb-6">Accuracy on NQ, TriviaQA, WebQ. Hover over bars for details.</p>
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="grid grid-cols-3 gap-4 text-center text-xs font-bold text-slate-400 mb-4 pl-20">
            <div>NQ (Top-20)</div><div>TriviaQA (Top-20)</div><div>WebQ (Top-20)</div>
          </div>
          {RETRIEVER_STATS.map(r => (
            <div key={r.name} className="mb-3">
              <div className="flex items-center gap-3 mb-1">
                <span className="w-20 text-xs font-bold text-slate-700 text-right shrink-0">{r.name}</span>
                <div className="flex-1 grid grid-cols-3 gap-2">
                  {[r.nq, r.tqa, r.webq].map((v, i) => (
                    <MiniBar key={i} value={v} max={90} color={r.color} />
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Reranker Performance Preview ── */}
      <section className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-bold text-slate-900 mb-1">Top Reranker Models (nDCG@10)</h2>
        <p className="text-slate-500 text-sm mb-6">TREC DL19, DL20 and BEIR Covid datasets. First-stage BM25 top-100.</p>
        <div className="grid md:grid-cols-3 gap-4">
          {(["DL19", "DL20", "COVID-19"] as const).map((dataset, di) => (
            <div key={dataset} className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5">
              <h3 className="text-sm font-bold text-slate-700 mb-3">{dataset}</h3>
              {RERANKER_HIGHLIGHTS
                .sort((a, b) => [b.dl19, b.dl20, b.covid][di] - [a.dl19, a.dl20, a.covid][di])
                .map((r, i) => {
                  const score = [r.dl19, r.dl20, r.covid][di];
                  return (
                    <div key={r.name} className="mb-2">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-slate-600 font-medium">{r.name}</span>
                        <span className="font-bold text-slate-800">{score}</span>
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full rounded-full bg-indigo-500" style={{ width: `${(score / 90) * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
            </div>
          ))}
        </div>
      </section>

      {/* ── BRIGHT Benchmark ── */}
      <section className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-bold text-slate-900 mb-1">BRIGHT Benchmark</h2>
        <p className="text-slate-500 text-sm mb-6">Reasoning-intensive retrieval — average nDCG@10 across 12 tasks.</p>
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="grid grid-cols-6 gap-4 items-end justify-center">
            {BRIGHT_LEADERS.map(m => (
              <div key={m.name} className="flex flex-col items-center gap-1">
                <span className="text-xs font-bold text-slate-700">{m.avg}</span>
                <div className="w-full rounded-t-lg" style={{ height: `${(m.avg / 35) * 120}px`, backgroundColor: m.color }} />
                <span className="text-[10px] text-slate-500 text-center leading-tight">{m.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── RAG Methods ── */}
      <section className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-bold text-slate-900 mb-6">Supported RAG Methods</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { icon: "⚡", name: "Basic RAG", desc: "Naive retrieval-augmented generation — retrieve, concatenate, generate." },
            { icon: "🧠", name: "Chain-of-Thought", desc: "Prompts the model to reason step-by-step before generating the final answer." },
            { icon: "🔄", name: "Self-Consistency", desc: "Generates multiple answers and aggregates via majority voting for robustness." },
            { icon: "🗂️", name: "Fusion-in-Decoder (FiD)", desc: "Encodes each retrieved passage independently then fuses during decoding." },
            { icon: "🤔", name: "ReAct RAG", desc: "Interleaves reasoning and acting (search) in an agentic loop." },
            { icon: "🎯", name: "Zero-Shot", desc: "No context — evaluates the model's parametric knowledge alone." },
          ].map(m => (
            <div key={m.name} className="p-5 bg-white rounded-xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-2xl mb-2">{m.icon}</div>
              <div className="font-semibold text-slate-800 mb-1 text-sm">{m.name}</div>
              <div className="text-xs text-slate-500 leading-relaxed">{m.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="max-w-5xl mx-auto px-6 py-12">
        <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-3xl p-10 text-center text-white">
          <h2 className="text-3xl font-black mb-2">Ready to explore?</h2>
          <p className="text-indigo-200 mb-6 text-sm">Run the full RAG pipeline interactively with your own queries.</p>
          <Link href="/chat"
            className="inline-block px-8 py-3 bg-white text-indigo-700 font-bold rounded-xl hover:bg-indigo-50 transition-colors shadow-lg">
            Open RAG Console →
          </Link>
        </div>
      </section>

      <footer className="border-t border-slate-100 py-8 text-center text-xs text-slate-400">
        Rankify · DataScienceUIBK · Open-Source Research
      </footer>
    </div>
  );
}
