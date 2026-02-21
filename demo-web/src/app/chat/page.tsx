"use client";

import {
    useState, useRef, useEffect, useCallback, useReducer
} from "react";
import {
    Send, Search, ListTree, Bot, Database, Upload,
    Sparkles, FileText, X, ChevronDown, ChevronUp,
    Layers, Loader2, ArrowRight, Plus, MessageSquare,
    Trash2, BrainCircuit, ChevronRight, AlertCircle, CheckCircle2
} from "lucide-react";

// ─── Inline Button ─────────────────────────────────────────────────────────────
function Btn({ children, disabled, onClick, className = "", type = "button" }: {
    children: React.ReactNode; disabled?: boolean; onClick?: () => void; className?: string; type?: "button" | "submit";
}) {
    return (
        <button type={type} disabled={disabled} onClick={onClick}
            className={`inline-flex items-center justify-center gap-1 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none ${className}`}>
            {children}
        </button>
    );
}

// ─── Model Definitions ─────────────────────────────────────────────────────────
const RETRIEVERS = [
    { value: "bm25", label: "BM25 (Sparse)", group: "Sparse" },
    { value: "dpr-multi", label: "DPR Multi", group: "Dense" },
    { value: "dpr-single", label: "DPR Single", group: "Dense" },
    { value: "ance-multi", label: "ANCE Multi", group: "Dense" },
    { value: "bpr-single", label: "BPR Single", group: "Dense" },
    { value: "bge", label: "BGE v1.5", group: "Dense" },
    { value: "colbert", label: "ColBERTv2", group: "Dense" },
    { value: "contriever", label: "Contriever", group: "Dense" },
    { value: "online", label: "Online (Web)", group: "Online" },
    { value: "hyde", label: "HyDE (Hypothetical)", group: "Advanced" },
    { value: "diver-dense", label: "Diver Dense", group: "Advanced" },
    { value: "diver-bm25", label: "Diver BM25", group: "Advanced" },
    { value: "reasonir", label: "ReasonIR", group: "Reasoning" },
    { value: "reason-embed", label: "Reason-Embed", group: "Reasoning" },
    { value: "bge-reasoner-embed", label: "BGE Reasoner Embed", group: "Reasoning" },
];

const RERANKER_CATEGORIES = [
    { value: "none", label: "None", models: [] },
    { value: "flashrank", label: "FlashRank", models: ["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2", "ms-marco-MultiBERT-L-12", "rank-T5-flan", "ce-esci-MiniLM-L12-v2"] },
    { value: "transformer_ranker", label: "Cross-Encoder", models: ["bge-reranker-base", "bge-reranker-large", "bge-reranker-v2-m3", "mxbai-rerank-xsmall", "mxbai-rerank-base", "mxbai-rerank-large", "jina-reranker-tiny", "jina-reranker-turbo", "ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2", "ms-marco-electra-base"] },
    { value: "monot5", label: "MonoT5", models: ["monot5-base-msmarco", "monot5-large-msmarco", "monot5-base-msmarco-10k", "monot5-large-msmarco-10k", "monot5-3b-msmarco-10k", "monot5-base-med-msmarco"] },
    { value: "rankgpt", label: "RankGPT (Local)", models: ["Llama-3.2-1B", "Llama-3.2-3B", "llamav3.1-8b", "llamav3.1-70b", "Qwen2.5-7B", "Mistral-7B-Instruct-v0.3"] },
    { value: "rankgpt-api", label: "RankGPT (API)", models: ["gpt-3.5", "gpt-4", "gpt-4-mini", "claude-3-5", "llamav3.1-8b", "llamav3.1-70b"] },
    { value: "monobert", label: "MonoBERT", models: ["monobert-large"] },
    { value: "colbert_ranker", label: "ColBERT Reranker", models: ["colbertv2.0", "jina-colbert-v1-en", "mxbai-colbert-large-v1"] },
    { value: "upr", label: "UPR", models: ["t5-small", "t5-base", "t5-large", "t0-3b", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "flan-t5-xl"] },
    { value: "inranker", label: "InRanker", models: ["inranker-small", "inranker-base", "inranker-3b"] },
    { value: "rankt5", label: "RankT5", models: ["rankt5-base", "rankt5-large", "rankt5-3b"] },
    { value: "listt5", label: "ListT5", models: ["listt5-base", "listt5-3b"] },
    { value: "lit5dist", label: "LiT5-Distill", models: ["LiT5-Distill-base", "LiT5-Distill-large", "LiT5-Distill-xl", "LiT5-Distill-base-v2", "LiT5-Distill-large-v2", "LiT5-Distill-xl-v2"] },
    { value: "lit5score", label: "LiT5-Score", models: ["LiT5-Score-base", "LiT5-Score-large", "LiT5-Score-xl"] },
    { value: "vicuna_reranker", label: "Vicuna Reranker", models: ["rank_vicuna_7b_v1", "rank_vicuna_7b_v1_fp16"] },
    { value: "zephyr_reranker", label: "Zephyr Reranker", models: ["rank_zephyr_7b_v1_full"] },
    { value: "incontext_reranker", label: "InContext Reranker", models: ["llamav3.1-8b", "llamav3.1-70b", "Mistral-7B-Instruct-v0.2"] },
    { value: "llm_layerwise_ranker", label: "BGE Layerwise", models: ["bge-reranker-v2-gemma", "bge-reranker-v2-minicpm-layerwise", "bge-reranker-v2.5-gemma2-lightweight", "bge-multilingual-gemma2"] },
    { value: "sentence_transformer_reranker", label: "Sentence Transformer", models: ["gtr-t5-base", "gtr-t5-large", "gtr-t5-xl", "gtr-t5-xxl", "all-MiniLM-L6-v2"] },
    { value: "apiranker", label: "API Reranker", models: ["cohere", "jina", "voyage", "mixedbread.ai"] },
    { value: "twolar", label: "TWOLAR", models: ["twolar-xl", "twolar-large"] },
    { value: "echorank", label: "EchoRank", models: ["flan-t5-large", "flan-t5-xl"] },
    { value: "blender_reranker", label: "PairRM (Blender)", models: ["PairRM"] },
    { value: "splade_reranker", label: "SPLADE Reranker", models: ["splade-cocondenser"] },
];

const GENERATORS = [
    { value: "azure", label: "GPT-4o (Azure)", desc: "Your Azure OpenAI deployment" },
    { value: "azure-cot", label: "GPT-4o Chain-of-Thought", desc: "Step-by-step reasoning" },
    { value: "azure-self-consistency", label: "GPT-4o Self-Consistency", desc: "Multi-sample + vote" },
    { value: "llama-3", label: "Llama 3.1 8B", desc: "Local vLLM" },
    { value: "mistral", label: "Mistral 7B", desc: "Local vLLM" },
    { value: "fid", label: "FiD (nq_reader_base)", desc: "Fusion-in-Decoder" },
    { value: "zero-shot", label: "Zero-Shot GPT-4o", desc: "No context, parametric only" },
];

// ─── Types ──────────────────────────────────────────────────────────────────────
type PipelineMode = "retrieve" | "rerank" | "rag";
interface PDoc { id: string; text: string; title?: string; score?: number; }
interface ChatMsg {
    id: string; role: "user" | "assistant"; content: string;
    pipelineData?: { retrievedDocs?: PDoc[]; rerankedDocs?: PDoc[]; ragMethod?: string; };
}
interface Cfg {
    pipelineMode: PipelineMode; dataSource: string;
    retriever: string; rerankerCategory: string; rerankerModel: string; generator: string;
}
interface Session { id: string; title: string; updatedAt: number; messages: ChatMsg[]; config: Cfg; }

function gid() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }
function defCfg(): Cfg { return { pipelineMode: "rag", dataSource: "wiki", retriever: "bm25", rerankerCategory: "flashrank", rerankerModel: "ms-marco-MiniLM-L-12-v2", generator: "azure" }; }
function newSess(cfg?: Partial<Cfg>): Session { return { id: gid(), title: "New Pipeline", updatedAt: Date.now(), messages: [], config: { ...defCfg(), ...cfg } }; }

type SA =
    | { type: "LOAD"; s: Session[]; id: string }
    | { type: "CREATE"; s: Session }
    | { type: "SELECT"; id: string }
    | { type: "DELETE"; id: string }
    | { type: "CFG"; id: string; c: Partial<Cfg> }
    | { type: "MSG"; id: string; m: ChatMsg }
    | { type: "UPD"; id: string; content: string; pd?: ChatMsg["pipelineData"] };

function reducer(state: { sessions: Session[]; activeId: string }, a: SA) {
    switch (a.type) {
        case "LOAD": return { sessions: a.s, activeId: a.id };
        case "CREATE": return { sessions: [a.s, ...state.sessions], activeId: a.s.id };
        case "SELECT": return { ...state, activeId: a.id };
        case "DELETE": {
            const next = state.sessions.filter(s => s.id !== a.id);
            const ns = next.length > 0 ? next : [newSess()];
            return { sessions: ns, activeId: state.activeId === a.id ? ns[0].id : state.activeId };
        }
        case "CFG": return { ...state, sessions: state.sessions.map(s => s.id === a.id ? { ...s, config: { ...s.config, ...a.c }, updatedAt: Date.now() } : s) };
        case "MSG": return {
            ...state, sessions: state.sessions.map(s => {
                if (s.id !== a.id) return s;
                const msgs = [...s.messages, a.m];
                const title = s.messages.length === 0 && a.m.role === "user" ? a.m.content.slice(0, 50) : s.title;
                return { ...s, messages: msgs, title, updatedAt: Date.now() };
            })
        };
        case "UPD": return {
            ...state, sessions: state.sessions.map(s => {
                if (s.id !== a.id) return s;
                const msgs = [...s.messages];
                for (let i = msgs.length - 1; i >= 0; i--) { if (msgs[i].role === "assistant") { msgs[i] = { ...msgs[i], content: a.content, pipelineData: a.pd ?? msgs[i].pipelineData }; break; } }
                return { ...s, messages: msgs };
            })
        };
        default: return state;
    }
}

// ─── Sub-components ────────────────────────────────────────────────────────────
function Sel({ value, onChange, opts, w = "w-auto" }: { value: string; onChange: (v: string) => void; opts: { value: string; label: string }[]; w?: string; }) {
    return (
        <select value={value} onChange={e => onChange(e.target.value)}
            className={`h-8 px-2 text-xs border border-slate-200 rounded-md bg-slate-50 text-slate-700 outline-none focus:border-indigo-400 cursor-pointer ${w}`}>
            {opts.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
    );
}

function DocCard({ doc, rank, isReranked }: { doc: PDoc; rank: number; isReranked?: boolean }) {
    return (
        <div className={`flex gap-2 p-2.5 rounded-lg border text-xs ${isReranked ? "bg-indigo-50/60 border-indigo-100" : "bg-slate-50 border-slate-100"}`}>
            <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${isReranked ? "bg-indigo-100 text-indigo-700" : "bg-slate-200 text-slate-600"}`}>{rank}</div>
            <div className="flex-1 min-w-0">
                {doc.title && <div className="font-semibold text-slate-700 truncate mb-0.5">{doc.title}</div>}
                <p className="text-slate-600 line-clamp-3 leading-relaxed">{doc.text}</p>
                <div className="flex gap-2 mt-1 text-[10px] text-slate-400">
                    <span>ID: {doc.id}</span>
                    {doc.score != null && <span className={`font-semibold ${isReranked ? "text-indigo-600" : "text-slate-500"}`}>score: {doc.score.toFixed(4)}</span>}
                </div>
            </div>
        </div>
    );
}

function PipelineViz({ msg, cfg }: { msg: ChatMsg; cfg: Cfg }) {
    const [showR, setShowR] = useState(false);
    const [showRR, setShowRR] = useState(false);
    const pd = msg.pipelineData;
    if (!pd) return null;
    const rLabel = RETRIEVERS.find(r => r.value === cfg.retriever)?.label ?? cfg.retriever;
    const rrCat = RERANKER_CATEGORIES.find(c => c.value === cfg.rerankerCategory);
    const genLabel = GENERATORS.find(g => g.value === cfg.generator)?.label ?? cfg.generator;

    return (
        <div className="flex flex-col gap-2 w-full">
            <div className="flex flex-col md:flex-row gap-4 w-full">
                {pd.retrievedDocs && pd.retrievedDocs.length > 0 && (
                    <div className="flex-1 rounded-xl border border-slate-200 bg-white overflow-hidden flex flex-col">
                        <button onClick={() => setShowR(v => !v)} className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-slate-50 transition-colors">
                            <div className="flex items-center gap-2">
                                <span className="w-6 h-6 rounded-full bg-emerald-100 flex items-center justify-center"><Search className="w-3 h-3 text-emerald-600" /></span>
                                <span className="text-xs font-semibold text-slate-700">Retrieved</span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">{rLabel}</span>
                                <span className="text-[11px] text-slate-400">{pd.retrievedDocs.length} docs</span>
                            </div>
                            {showR ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
                        </button>
                        {showR && <div className="px-3 pb-3 space-y-1.5 border-t border-slate-100 pt-2 flex-1 overflow-y-auto max-h-72">{pd.retrievedDocs.map((d, i) => <DocCard key={d.id} doc={d} rank={i + 1} />)}</div>}
                    </div>
                )}

                {pd.rerankedDocs && pd.rerankedDocs.length > 0 && (
                    <div className="flex-1 rounded-xl border border-indigo-200 bg-white overflow-hidden flex flex-col">
                        <button onClick={() => setShowRR(v => !v)} className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-indigo-50/30 transition-colors">
                            <div className="flex items-center gap-2">
                                <span className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center"><ListTree className="w-3 h-3 text-indigo-600" /></span>
                                <span className="text-xs font-semibold text-slate-700">Reranked</span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">{rrCat?.label} · {cfg.rerankerModel}</span>
                            </div>
                            {showRR ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
                        </button>
                        {showRR && <div className="px-3 pb-3 space-y-1.5 border-t border-indigo-100 pt-2 flex-1 overflow-y-auto max-h-72">{pd.rerankedDocs.map((d, i) => <DocCard key={d.id} doc={d} rank={i + 1} isReranked />)}</div>}
                    </div>
                )}
            </div>

            {cfg.pipelineMode === "rag" && <>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-slate-200 rotate-90 md:rotate-0" /></div>
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-violet-200 bg-violet-50/40">
                    <span className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center"><Bot className="w-3 h-3 text-violet-600" /></span>
                    <span className="text-xs font-semibold text-slate-700">Generator</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 border border-violet-100">{genLabel}</span>
                    {pd.ragMethod && <span className="text-[10px] text-slate-400">via {pd.ragMethod}</span>}
                </div>
            </>}
        </div>
    );
}

// ─── New-Pipeline Modal ───────────────────────────────────────────────────────
function NewPipelineModal({ onClose, onCreate }: {
    onClose: () => void;
    onCreate: (mode: PipelineMode) => void;
}) {
    const modes: { mode: PipelineMode; icon: string; title: string; desc: string; color: string }[] = [
        { mode: "retrieve", icon: "🔍", title: "Retrieve", color: "border-emerald-300 hover:bg-emerald-50", desc: "Search and retrieve relevant documents only. Best for exploring what's in the corpus." },
        { mode: "rerank", icon: "🔀", title: "Retrieve + Rerank", color: "border-indigo-300 hover:bg-indigo-50", desc: "Retrieve candidates then re-score them with a cross-encoder for higher precision." },
        { mode: "rag", icon: "✨", title: "Full RAG Pipeline", color: "border-violet-300 hover:bg-violet-50", desc: "Retrieve → Rerank → Generate. Get a synthesised answer grounded in real documents." },
    ];
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
            <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />
            <div className="relative bg-white rounded-2xl shadow-2xl w-[480px] max-w-[94vw] p-6 z-10" onClick={e => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-bold text-slate-800">Choose a Pipeline Mode</h2>
                    <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-700">
                        <X className="w-4 h-4" />
                    </button>
                </div>
                <div className="space-y-3">
                    {modes.map(m => (
                        <button key={m.mode} onClick={() => { onCreate(m.mode); onClose(); }}
                            className={`w-full text-left flex items-start gap-3 p-4 rounded-xl border-2 transition-all ${m.color}`}>
                            <span className="text-2xl mt-0.5">{m.icon}</span>
                            <div>
                                <div className="font-semibold text-slate-800 text-sm">{m.title}</div>
                                <div className="text-xs text-slate-500 mt-0.5 leading-relaxed">{m.desc}</div>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}

function SidebarPanel({ sessions, activeId, onCreate, onSelect, onDelete }: {
    sessions: Session[]; activeId: string;
    onCreate: () => void; onSelect: (id: string) => void; onDelete: (id: string) => void;
}) {
    function ago(ts: number) { const m = Math.floor((Date.now() - ts) / 60000); if (m < 1) return "now"; if (m < 60) return `${m}m`; const h = Math.floor(m / 60); return h < 24 ? `${h}h` : `${Math.floor(h / 24)}d`; }
    const mc: Record<PipelineMode, string> = { retrieve: "bg-emerald-900/60 text-emerald-300", rerank: "bg-purple-900/60 text-purple-300", rag: "bg-indigo-900/60 text-indigo-300" };
    const ml: Record<PipelineMode, string> = { retrieve: "R", rerank: "R+R", rag: "RAG" };
    return (
        <div className="flex flex-col h-full bg-[#0f1117]">
            <div className="p-4 border-b border-white/10 flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0"><BrainCircuit className="w-4 h-4 text-white" /></div>
                <span className="font-semibold text-white text-sm">Rankify Console</span>
            </div>
            <div className="p-3">
                <button onClick={onCreate} className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-white/70 hover:text-white hover:bg-white/10 transition-all border border-white/10">
                    <Plus className="w-4 h-4" /> New Pipeline
                    <ChevronDown className="w-3.5 h-3.5 ml-auto opacity-50" />
                </button>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
                {sessions.map(s => (
                    <div key={s.id}
                        className={`group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all text-sm ${s.id === activeId ? "bg-white/10 text-white" : "text-white/60 hover:text-white/90 hover:bg-white/5"}`}
                        onClick={() => onSelect(s.id)}>
                        <MessageSquare className="w-4 h-4 shrink-0 opacity-70" />
                        <div className="flex-1 min-w-0">
                            <div className="truncate font-medium text-[13px]">{s.title}</div>
                            <div className="flex items-center gap-1.5 mt-0.5">
                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${mc[s.config.pipelineMode]}`}>{ml[s.config.pipelineMode]}</span>
                                <span className="text-[11px] text-white/30">{ago(s.updatedAt)}</span>
                            </div>
                        </div>
                        <button className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 text-white/30 shrink-0" onClick={e => { e.stopPropagation(); onDelete(s.id); }}>
                            <Trash2 className="w-3.5 h-3.5" />
                        </button>
                        {s.id === activeId && <ChevronRight className="w-3.5 h-3.5 shrink-0 text-white/40" />}
                    </div>
                ))}
            </div>
            <div className="p-3 border-t border-white/10 text-xs text-white/25 text-center">Rankify · Open-Source RAG Toolkit</div>
        </div>
    );
}

// ─── Main Chat Page ────────────────────────────────────────────────────────────
export default function ChatPage() {
    const SK = "rankify_v4", AK = "rankify_act_v4";
    const [state, dispatch] = useReducer(reducer, { sessions: [], activeId: "" });
    const [hydrated, setHydrated] = useState(false);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [serverOk, setServerOk] = useState<boolean | null>(null);
    const [showModal, setShowModal] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const fileRef = useRef<HTMLInputElement>(null);
    // ↓ refs to avoid stale closures in async callbacks
    const stateRef = useRef(state);
    useEffect(() => { stateRef.current = state; }, [state]);

    useEffect(() => {
        try {
            const raw = localStorage.getItem(SK);
            const rawId = localStorage.getItem(AK);
            const loaded: Session[] = raw ? JSON.parse(raw) : [];
            if (loaded.length === 0) { const s = newSess(); dispatch({ type: "LOAD", s: [s], id: s.id }); }
            else { dispatch({ type: "LOAD", s: loaded, id: rawId && loaded.find(s => s.id === rawId) ? rawId : loaded[0].id }); }
        } catch { const s = newSess(); dispatch({ type: "LOAD", s: [s], id: s.id }); }
        setHydrated(true);
    }, []);

    useEffect(() => { if (!hydrated) return; localStorage.setItem(SK, JSON.stringify(state.sessions)); localStorage.setItem(AK, state.activeId); }, [state, hydrated]);
    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [state.sessions, loading]);

    useEffect(() => {
        const check = async () => { try { const r = await fetch("/api/health-check", { signal: AbortSignal.timeout(3000) }); setServerOk(r.ok); } catch { setServerOk(false); } };
        check(); const t = setInterval(check, 30000); return () => clearInterval(t);
    }, []);

    // Read active session from ref (not stale closure)
    const getActive = () => stateRef.current.sessions.find(s => s.id === stateRef.current.activeId);
    const cfg = getActive()?.config ?? defCfg();

    const handleCfg = useCallback((c: Partial<Cfg>) => {
        dispatch({ type: "CFG", id: stateRef.current.activeId, c });
    }, []);

    const handleSubmit = useCallback(async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim() || loading) return;

        // Read fresh state from ref
        const activeId = stateRef.current.activeId;
        const activeCfg = getActive()?.config ?? defCfg();
        const prevMsgs = getActive()?.messages ?? [];

        const userMsg: ChatMsg = { id: gid(), role: "user", content: input.trim() };
        dispatch({ type: "MSG", id: activeId, m: userMsg });
        setInput("");
        setLoading(true);

        dispatch({ type: "MSG", id: activeId, m: { id: gid(), role: "assistant", content: "" } });

        let pd: ChatMsg["pipelineData"] = {};
        let answer = "";

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [...prevMsgs, userMsg],
                    configuration: activeCfg,
                }),
            });
            if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

            const reader = res.body.getReader();
            const dec = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += dec.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() ?? "";

                for (const line of lines) {
                    if (!line) continue;
                    if (line.startsWith("2:")) {
                        try {
                            const arr = JSON.parse(line.slice(2)) as string[];
                            const meta = JSON.parse(arr[0]) as { pipelineMeta: ChatMsg["pipelineData"] };
                            pd = { ...pd, ...meta.pipelineMeta };
                            dispatch({ type: "UPD", id: activeId, content: answer, pd });
                        } catch { }
                    }
                    if (line.startsWith("0:")) {
                        const match = line.slice(2).match(/^"([\s\S]*)"$/);
                        if (match) { answer += match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"'); dispatch({ type: "UPD", id: activeId, content: answer, pd }); }
                    }
                }
            }
        } catch (err) {
            answer = `⚠️ ${err instanceof Error ? err.message : String(err)}\n\nStart the server:\n\`python demo_server.py --port 8000\``;
            dispatch({ type: "UPD", id: activeId, content: answer, pd });
        } finally {
            setLoading(false);
        }
    }, [input, loading]);

    if (!hydrated) return <div className="flex h-screen items-center justify-center"><Loader2 className="w-6 h-6 animate-spin text-indigo-500" /></div>;

    const messages = getActive()?.messages ?? [];
    const selCat = RERANKER_CATEGORIES.find(c => c.value === cfg.rerankerCategory);
    const modeLabels: Record<PipelineMode, string> = { retrieve: "🔍 Retrieve", rerank: "🔀 Retrieve + Rerank", rag: "✨ Full RAG" };

    return (
        <div className="flex h-[calc(100vh-56px)] overflow-hidden bg-white">
            {/* New-Pipeline modal */}
            {showModal && (
                <NewPipelineModal
                    onClose={() => setShowModal(false)}
                    onCreate={(mode) => {
                        const currentCfg = getActive()?.config ?? defCfg();
                        dispatch({ type: "CREATE", s: newSess({ ...currentCfg, pipelineMode: mode }) });
                    }}
                />
            )}

            {/* Sidebar */}
            <aside className="w-[240px] shrink-0 hidden md:block">
                <SidebarPanel
                    sessions={state.sessions} activeId={state.activeId}
                    onCreate={() => setShowModal(true)}
                    onSelect={id => dispatch({ type: "SELECT", id })}
                    onDelete={id => dispatch({ type: "DELETE", id })} />
            </aside>

            {/* Main */}
            <div className="flex flex-col flex-1 min-w-0 h-full">

                {/* Config Bar */}
                <div className="shrink-0 border-b border-slate-100 bg-white z-10">
                    <div className="px-4 pt-3 pb-2 flex items-center justify-between gap-4 flex-wrap">
                        {/* Current mode badge + switch button */}
                        <div className="flex items-center gap-2">
                            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border ${cfg.pipelineMode === "retrieve" ? "bg-emerald-50 text-emerald-700 border-emerald-200" :
                                cfg.pipelineMode === "rerank" ? "bg-indigo-50 text-indigo-700 border-indigo-200" :
                                    "bg-violet-50 text-violet-700 border-violet-200"}`}>
                                {cfg.pipelineMode === "retrieve" ? "🔍" : cfg.pipelineMode === "rerank" ? "🔀" : "✨"}
                                {modeLabels[cfg.pipelineMode]}
                            </div>
                            <button onClick={() => setShowModal(true)}
                                className="px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-700 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors">
                                Switch mode
                            </button>
                        </div>
                        <div className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${serverOk === true ? "border-emerald-200 bg-emerald-50 text-emerald-700" :
                            serverOk === false ? "border-red-200 bg-red-50 text-red-700" :
                                "border-slate-200 bg-slate-50 text-slate-500"}`}>
                            {serverOk === true ? <CheckCircle2 className="w-3 h-3" /> : serverOk === false ? <AlertCircle className="w-3 h-3" /> : <Loader2 className="w-3 h-3 animate-spin" />}
                            {serverOk === true ? "Server Online" : serverOk === false ? "Server Offline" : "Checking…"}
                        </div>
                    </div>

                    {/* Selects */}
                    <div className="px-4 pb-3 flex flex-wrap gap-2 items-end">
                        <div className="flex flex-col gap-0.5">
                            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-0.5"><Database className="w-3 h-3" />Data</label>
                            <Sel value={cfg.dataSource} onChange={v => handleCfg({ dataSource: v })} opts={[{ value: "wiki", label: "Wikipedia" }, { value: "msmarco", label: "MS MARCO" }, { value: "custom", label: "Custom JSONL" }]} w="w-32" />
                        </div>

                        {cfg.dataSource === "custom" && (
                            <div className="flex flex-col gap-0.5">
                                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">JSONL</label>
                                {file ? (
                                    <div className="flex items-center gap-1 h-8 px-2 rounded border border-emerald-200 bg-emerald-50 text-xs text-emerald-700">
                                        <FileText className="w-3 h-3" /><span className="max-w-20 truncate">{file.name}</span>
                                        <button onClick={() => setFile(null)}><X className="w-3 h-3 hover:text-red-500" /></button>
                                    </div>
                                ) : (
                                    <button onClick={() => fileRef.current?.click()}
                                        className="h-8 px-2 border border-dashed border-slate-300 rounded text-xs text-slate-500 hover:border-indigo-400 hover:text-indigo-600 flex items-center gap-1">
                                        <Upload className="w-3 h-3" />Upload
                                    </button>
                                )}
                                <input ref={fileRef} type="file" accept=".jsonl,.json" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) setFile(f); }} />
                            </div>
                        )}

                        <div className="flex flex-col gap-0.5">
                            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-0.5"><Search className="w-3 h-3" />Retriever</label>
                            <Sel value={cfg.retriever} onChange={v => handleCfg({ retriever: v })} opts={RETRIEVERS.map(r => ({ value: r.value, label: r.label }))} w="w-48" />
                        </div>

                        {cfg.pipelineMode !== "retrieve" && <>
                            <div className="flex flex-col gap-0.5">
                                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-0.5"><ListTree className="w-3 h-3" />Reranker</label>
                                <Sel value={cfg.rerankerCategory}
                                    onChange={v => { const cat = RERANKER_CATEGORIES.find(c => c.value === v); handleCfg({ rerankerCategory: v, rerankerModel: cat?.models[0] ?? "" }); }}
                                    opts={RERANKER_CATEGORIES.map(c => ({ value: c.value, label: c.label }))} w="w-44" />
                            </div>
                            {selCat && selCat.models.length > 0 && (
                                <div className="flex flex-col gap-0.5">
                                    <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Model</label>
                                    <Sel value={cfg.rerankerModel} onChange={v => handleCfg({ rerankerModel: v })} opts={selCat.models.map(m => ({ value: m, label: m }))} w="w-52" />
                                </div>
                            )}
                        </>}

                        {cfg.pipelineMode === "rag" && (
                            <div className="flex flex-col gap-0.5">
                                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-0.5"><Bot className="w-3 h-3" />Generator</label>
                                <Sel value={cfg.generator} onChange={v => handleCfg({ generator: v })} opts={GENERATORS.map(g => ({ value: g.value, label: g.label }))} w="w-48" />
                            </div>
                        )}
                    </div>
                </div>

                {/* Chat */}
                <div className="flex-1 overflow-y-auto">
                    <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">
                        {messages.length === 0 ? (
                            <div className="flex flex-col items-center text-center mt-16 space-y-5">
                                <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center border border-indigo-100">
                                    <Sparkles className="w-8 h-8 text-indigo-500" />
                                </div>
                                <div>
                                    <h2 className="text-xl font-semibold text-slate-800 mb-1">
                                        {cfg.pipelineMode === "retrieve" ? "Retrieval Mode" : cfg.pipelineMode === "rerank" ? "Retrieve + Rerank" : "Full RAG Pipeline"}
                                    </h2>
                                    <p className="text-sm text-slate-500 max-w-sm mx-auto">
                                        {cfg.pipelineMode === "retrieve" ? `Retrieve real documents from ${cfg.dataSource === "wiki" ? "Wikipedia" : "MS MARCO"} using ${RETRIEVERS.find(r => r.value === cfg.retriever)?.label}.`
                                            : cfg.pipelineMode === "rerank" ? `Retrieve then rerank with ${RERANKER_CATEGORIES.find(c => c.value === cfg.rerankerCategory)?.label}.`
                                                : `Full pipeline: retrieve → rerank → generate with ${GENERATORS.find(g => g.value === cfg.generator)?.label}.`}
                                    </p>
                                </div>
                                {serverOk === false && (
                                    <div className="flex items-start gap-2 px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-800 max-w-xs text-left">
                                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                                        <div><div className="font-semibold text-sm">Server not running</div><code className="text-xs mt-1 block bg-amber-100 px-1 rounded">python demo_server.py</code></div>
                                    </div>
                                )}
                                <div className="flex flex-wrap gap-2 justify-center">
                                    {["What is dense retrieval?", "How does BM25 work?", "Compare cross-encoders vs bi-encoders", "What is retrieval-augmented generation?"].map(q => (
                                        <button key={q} onClick={() => setInput(q)}
                                            className="px-3 py-1.5 text-xs rounded-full border border-slate-200 text-slate-600 hover:border-indigo-300 hover:text-indigo-700 hover:bg-indigo-50 transition-all">{q}</button>
                                    ))}
                                </div>
                            </div>
                        ) : messages.map(m => (
                            <div key={m.id} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                                {m.role !== "user" && <div className="w-7 h-7 shrink-0 rounded-lg bg-indigo-100 flex items-center justify-center mt-1"><Layers className="w-4 h-4 text-indigo-600" /></div>}
                                <div className={`flex flex-col gap-2 ${m.role === "user" ? "items-end max-w-[75%]" : "items-start w-full"}`}>
                                    {m.role !== "user" && m.pipelineData && <div className="w-full"><PipelineViz msg={m} cfg={cfg} /></div>}
                                    {(m.content || (loading && m.role === "assistant")) && (
                                        <div className={`px-4 py-3 text-sm leading-relaxed rounded-2xl shadow-sm whitespace-pre-wrap ${m.role === "user" ? "bg-slate-900 text-white rounded-br-sm" : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"}`}>
                                            {m.content || <span className="flex gap-1"><span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" /><span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.1s]" /><span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.2s]" /></span>}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                        <div ref={bottomRef} />
                    </div>
                </div>

                {/* Input */}
                <div className="shrink-0 border-t border-slate-100 bg-white px-4 py-3">
                    <div className="max-w-3xl mx-auto">
                        <form onSubmit={handleSubmit}
                            className="flex items-end gap-2 bg-white border border-slate-300 shadow-sm focus-within:shadow-md focus-within:border-indigo-400 transition-all rounded-2xl p-2 pl-4">
                            <textarea value={input} onChange={e => setInput(e.target.value)}
                                placeholder={`Ask something — ${RETRIEVERS.find(r => r.value === cfg.retriever)?.label}${cfg.pipelineMode !== "retrieve" ? " → rerank" : ""}${cfg.pipelineMode === "rag" ? " → generate" : ""}`}
                                className="flex-1 max-h-28 min-h-[36px] bg-transparent resize-none outline-none text-sm pt-2 pb-1 placeholder:text-slate-400"
                                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }} />
                            <button type="submit" disabled={loading || !input.trim()} className="h-9 w-9 shrink-0 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 transition-colors flex items-center justify-center mb-0.5">
                                {loading ? <Loader2 className="w-4 h-4 text-white animate-spin" /> : <Send className="w-4 h-4 text-white ml-0.5" />}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}
