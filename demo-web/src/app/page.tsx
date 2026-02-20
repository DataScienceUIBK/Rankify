"use client";

import {
  useState, useRef, useEffect, useCallback, useReducer
} from "react";
import {
  Send, Search, ListTree, Bot, Database, Upload, Sparkles,
  FileJson, X, ChevronDown, ChevronUp, Layers,
  Loader2, FileText, ArrowRight, Plus, MessageSquare,
  Trash2, BrainCircuit, ChevronRight, AlertCircle, CheckCircle2
} from "lucide-react";
import { Button } from "./ui/button-local";

// ─── Inline type definitions (avoids cross-dir import issues) ──────────────────
type PipelineMode = "retrieve" | "rerank" | "rag";

interface RetrieverDef { value: string; label: string; description: string; }
interface ModelDef { value: string; label: string; }
interface RerankerCat { value: string; label: string; description: string; models: ModelDef[]; }
interface GeneratorDef { value: string; label: string; description: string; }

const RETRIEVERS: RetrieverDef[] = [
  { value: "bm25", label: "BM25 (Sparse)", description: "Lexical matching, no GPU needed" },
  { value: "dpr", label: "DPR", description: "Dense Passage Retrieval (bi-encoder)" },
  { value: "ance", label: "ANCE", description: "Approximate Nearest Neighbor" },
  { value: "contriever", label: "Contriever", description: "Unsupervised dense retriever" },
  { value: "colbert", label: "ColBERTv2", description: "Late interaction retrieval" },
  { value: "bge", label: "BGE-v1.5", description: "SOTA bi-encoder retriever" },
];

const RERANKER_CATEGORIES: RerankerCat[] = [
  { value: "none", label: "None", description: "Skip reranking", models: [] },
  {
    value: "flashrank", label: "FlashRank", description: "Ultra-fast ONNX reranker (recommended)", models: [
      { value: "ms-marco-TinyBERT-L-2-v2", label: "TinyBERT L2 v2" },
      { value: "ms-marco-MiniLM-L-12-v2", label: "MiniLM L12 v2" },
      { value: "ms-marco-MultiBERT-L-12", label: "MultiBERT L12" },
      { value: "rank-T5-flan", label: "Rank-T5-Flan" },
      { value: "ce-esci-MiniLM-L12-v2", label: "ce-esci MiniLM" },
    ]
  },
  {
    value: "transformer_ranker", label: "Cross-Encoder", description: "HuggingFace cross-encoders", models: [
      { value: "bge-reranker-base", label: "BGE Reranker Base" },
      { value: "bge-reranker-large", label: "BGE Reranker Large" },
      { value: "bge-reranker-v2-m3", label: "BGE Reranker v2-m3" },
      { value: "mxbai-rerank-xsmall", label: "mxbai xSmall" },
      { value: "mxbai-rerank-base", label: "mxbai Base" },
      { value: "mxbai-rerank-large", label: "mxbai Large" },
      { value: "ms-marco-MiniLM-L-6-v2", label: "MS-MARCO MiniLM-L-6" },
      { value: "ms-marco-MiniLM-L-12-v2", label: "MS-MARCO MiniLM-L-12" },
    ]
  },
  {
    value: "monot5", label: "MonoT5", description: "T5-based pointwise reranker", models: [
      { value: "monot5-base-msmarco", label: "MonoT5 Base" },
      { value: "monot5-large-msmarco", label: "MonoT5 Large" },
      { value: "monot5-base-msmarco-10k", label: "MonoT5 Base 10k" },
      { value: "monot5-3b-msmarco-10k", label: "MonoT5 3B" },
    ]
  },
  {
    value: "rankgpt", label: "RankGPT (Local LLM)", description: "LLM listwise reranker (vLLM)", models: [
      { value: "Llama-3.2-1B", label: "Llama 3.2 1B" },
      { value: "llamav3.1-8b", label: "Llama 3.1 8B" },
      { value: "Mistral-7B-Instruct-v0.3", label: "Mistral 7B v0.3" },
    ]
  },
  {
    value: "rankgpt-api", label: "RankGPT (API)", description: "Cloud LLM reranker", models: [
      { value: "gpt-3.5", label: "GPT-3.5 Turbo" },
      { value: "gpt-4", label: "GPT-4o" },
      { value: "gpt-4-mini", label: "GPT-4o mini" },
      { value: "claude-3-5", label: "Claude 3.5 Sonnet" },
    ]
  },
  {
    value: "monobert", label: "MonoBERT", description: "BERT-based pointwise reranker", models: [
      { value: "monobert-large", label: "MonoBERT Large" },
    ]
  },
  {
    value: "colbert_ranker", label: "ColBERT Reranker", description: "Late-interaction reranker", models: [
      { value: "colbertv2.0", label: "ColBERTv2.0" },
      { value: "jina-colbert-v1-en", label: "Jina ColBERT v1" },
    ]
  },
  {
    value: "upr", label: "UPR", description: "Unsupervised Passage Reranker", models: [
      { value: "t5-small", label: "T5 Small" },
      { value: "t5-base", label: "T5 Base" },
      { value: "t5-large", label: "T5 Large" },
      { value: "flan-t5-xl", label: "Flan-T5 XL" },
    ]
  },
  {
    value: "inranker", label: "InRanker", description: "T5-based reranker", models: [
      { value: "inranker-small", label: "InRanker Small" },
      { value: "inranker-base", label: "InRanker Base" },
      { value: "inranker-3b", label: "InRanker 3B" },
    ]
  },
];

const GENERATORS: GeneratorDef[] = [
  { value: "openai", label: "GPT-4o mini", description: "OpenAI API" },
  { value: "claude", label: "Claude 3.5 Sonnet", description: "Anthropic API" },
  { value: "llama-3", label: "Llama 3.1 8B", description: "Local vLLM" },
  { value: "mistral", label: "Mistral 7B", description: "Local vLLM" },
  { value: "litellm", label: "LiteLLM (any)", description: "Universal proxy" },
];

// ─── Session types ─────────────────────────────────────────────────────────────
interface PipelineDoc { id: string; text: string; title?: string; score?: number; }

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  pipelineData?: { retrievedDocs?: PipelineDoc[]; rerankedDocs?: PipelineDoc[]; };
}

interface SessionConfig {
  pipelineMode: PipelineMode;
  dataSource: string;
  retriever: string;
  rerankerCategory: string;
  rerankerModel: string;
  generator: string;
}

interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  config: SessionConfig;
}

function genId() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }

function defaultConfig(): SessionConfig {
  return {
    pipelineMode: "rag",
    dataSource: "wiki",
    retriever: "bm25",
    rerankerCategory: "flashrank",
    rerankerModel: "ms-marco-MiniLM-L-12-v2",
    generator: "openai",
  };
}

function newSession(cfg?: Partial<SessionConfig>): ChatSession {
  return {
    id: genId(),
    title: "New Pipeline",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    config: { ...defaultConfig(), ...cfg },
  };
}

// ─── Session reducer ───────────────────────────────────────────────────────────
type SessionAction =
  | { type: "CREATE"; session: ChatSession }
  | { type: "SELECT"; id: string }
  | { type: "DELETE"; id: string }
  | { type: "UPDATE_CONFIG"; id: string; config: Partial<SessionConfig> }
  | { type: "ADD_MSG"; id: string; message: ChatMessage }
  | { type: "UPDATE_LAST_ASSISTANT"; id: string; content: string; pipelineData?: ChatMessage["pipelineData"] }
  | { type: "LOAD"; sessions: ChatSession[]; activeId: string };

interface SessionState { sessions: ChatSession[]; activeId: string; }

function reducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "LOAD":
      return { sessions: action.sessions, activeId: action.activeId };
    case "CREATE":
      return { sessions: [action.session, ...state.sessions], activeId: action.session.id };
    case "SELECT":
      return { ...state, activeId: action.id };
    case "DELETE": {
      const next = state.sessions.filter(s => s.id !== action.id);
      const ns = next.length > 0 ? next : [newSession()];
      const activeId = state.activeId === action.id ? ns[0].id : state.activeId;
      return { sessions: ns, activeId };
    }
    case "UPDATE_CONFIG":
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id === action.id ? { ...s, config: { ...s.config, ...action.config }, updatedAt: Date.now() } : s
        ),
      };
    case "ADD_MSG":
      return {
        ...state,
        sessions: state.sessions.map(s => {
          if (s.id !== action.id) return s;
          const msgs = [...s.messages, action.message];
          const title = s.messages.length === 0 && action.message.role === "user"
            ? action.message.content.slice(0, 50)
            : s.title;
          return { ...s, messages: msgs, title, updatedAt: Date.now() };
        }),
      };
    case "UPDATE_LAST_ASSISTANT":
      return {
        ...state,
        sessions: state.sessions.map(s => {
          if (s.id !== action.id) return s;
          const msgs = [...s.messages];
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant") {
              msgs[i] = { ...msgs[i], content: action.content, pipelineData: action.pipelineData ?? msgs[i].pipelineData };
              break;
            }
          }
          return { ...s, messages: msgs };
        }),
      };
    default:
      return state;
  }
}

// ─── Simple Select component ───────────────────────────────────────────────────
function Select({ value, onChange, options, className = "" }: {
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string; description?: string }>;
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className={`h-8 text-xs px-2 rounded-md border border-slate-200 bg-slate-50 text-slate-700 outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200 cursor-pointer ${className}`}
    >
      {options.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

// ─── Document Card ─────────────────────────────────────────────────────────────
function DocCard({ doc, rank, isReranked }: { doc: PipelineDoc; rank: number; isReranked?: boolean }) {
  return (
    <div className={`flex gap-2 p-2.5 rounded-lg border text-xs ${isReranked ? "bg-indigo-50/60 border-indigo-100" : "bg-slate-50 border-slate-100"}`}>
      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5 ${isReranked ? "bg-indigo-100 text-indigo-700" : "bg-slate-200 text-slate-600"}`}>
        {rank}
      </div>
      <div className="flex-1 min-w-0">
        {doc.title && <div className="font-semibold text-slate-700 mb-0.5 truncate">{doc.title}</div>}
        <p className="text-slate-600 leading-relaxed line-clamp-3">{doc.text}</p>
        <div className="flex items-center gap-3 mt-1">
          <span className="text-[10px] text-slate-400">ID: {doc.id}</span>
          {doc.score != null && (
            <span className={`text-[10px] font-semibold ${isReranked ? "text-indigo-600" : "text-slate-500"}`}>
              score: {doc.score.toFixed(4)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Pipeline Visualization ────────────────────────────────────────────────────
function PipelineViz({ message, config }: { message: ChatMessage; config: SessionConfig }) {
  const [showDocs, setShowDocs] = useState(false);
  const [showReranked, setShowReranked] = useState(false);
  const pd = message.pipelineData;
  if (!pd) return null;

  const retrieverLabel = RETRIEVERS.find(r => r.value === config.retriever)?.label ?? config.retriever;
  const rerankerCat = RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory);
  const rerankerModel = rerankerCat?.models.find(m => m.value === config.rerankerModel)?.label ?? config.rerankerModel;
  const genLabel = GENERATORS.find(g => g.value === config.generator)?.label ?? config.generator;

  return (
    <div className="space-y-2 w-full">
      {/* Retrieved docs */}
      {pd.retrievedDocs && pd.retrievedDocs.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
          <button onClick={() => setShowDocs(v => !v)}
            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-slate-50 transition-colors">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-emerald-100 flex items-center justify-center">
                <Search className="w-3 h-3 text-emerald-600" />
              </span>
              <span className="text-xs font-semibold text-slate-700">Retrieved</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100 font-medium">{retrieverLabel}</span>
              <span className="text-[11px] text-slate-400">{pd.retrievedDocs.length} docs</span>
            </div>
            {showDocs ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
          </button>
          {showDocs && (
            <div className="px-3 pb-3 space-y-1.5 border-t border-slate-100 pt-2 max-h-80 overflow-y-auto">
              {pd.retrievedDocs.map((doc, i) => <DocCard key={doc.id} doc={doc} rank={i + 1} />)}
            </div>
          )}
        </div>
      )}

      {pd.rerankedDocs && pd.rerankedDocs.length > 0 && (
        <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-slate-200" /></div>
      )}

      {/* Reranked docs */}
      {pd.rerankedDocs && pd.rerankedDocs.length > 0 && (
        <div className="rounded-xl border border-indigo-200 bg-white overflow-hidden">
          <button onClick={() => setShowReranked(v => !v)}
            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-indigo-50/30 transition-colors">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center">
                <ListTree className="w-3 h-3 text-indigo-600" />
              </span>
              <span className="text-xs font-semibold text-slate-700">Reranked</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100 font-medium">
                {rerankerCat?.label} · {rerankerModel}
              </span>
            </div>
            {showReranked ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
          </button>
          {showReranked && (
            <div className="px-3 pb-3 space-y-1.5 border-t border-indigo-100 pt-2 max-h-80 overflow-y-auto">
              {pd.rerankedDocs.map((doc, i) => <DocCard key={doc.id} doc={doc} rank={i + 1} isReranked />)}
            </div>
          )}
        </div>
      )}

      {/* Generator chip */}
      {config.pipelineMode === "rag" && (
        <>
          <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-slate-200" /></div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-violet-200 bg-violet-50/40">
            <span className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center">
              <Bot className="w-3 h-3 text-violet-600" />
            </span>
            <span className="text-xs font-semibold text-slate-700">Generator</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 border border-violet-100 font-medium">{genLabel}</span>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Sidebar ───────────────────────────────────────────────────────────────────
function timeAgo(ts: number) {
  const m = Math.floor((Date.now() - ts) / 60000);
  if (m < 1) return "now"; if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60); if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

function SidebarPanel({ sessions, activeId, onCreate, onSelect, onDelete }: {
  sessions: ChatSession[]; activeId: string;
  onCreate: () => void; onSelect: (id: string) => void; onDelete: (id: string) => void;
}) {
  const modeMap: Record<PipelineMode, string> = { retrieve: "R", rerank: "R+R", rag: "RAG" };
  const modeColor: Record<PipelineMode, string> = {
    retrieve: "bg-emerald-900/60 text-emerald-300",
    rerank: "bg-purple-900/60 text-purple-300",
    rag: "bg-indigo-900/60 text-indigo-300",
  };
  return (
    <div className="flex flex-col h-full bg-[#0f1117]">
      <div className="p-4 border-b border-white/10 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
          <BrainCircuit className="w-4 h-4 text-white" />
        </div>
        <span className="font-semibold text-white text-sm tracking-tight">Rankify Console</span>
      </div>
      <div className="p-3">
        <button onClick={onCreate}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-white/70 hover:text-white hover:bg-white/10 transition-all border border-white/10 hover:border-white/20">
          <Plus className="w-4 h-4" /> New Pipeline
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
        {sessions.map(session => (
          <div key={session.id}
            className={`group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all text-sm ${session.id === activeId ? "bg-white/10 text-white" : "text-white/60 hover:text-white/90 hover:bg-white/5"
              }`}
            onClick={() => onSelect(session.id)}>
            <MessageSquare className="w-4 h-4 shrink-0 opacity-70" />
            <div className="flex-1 min-w-0">
              <div className="truncate font-medium text-[13px]">{session.title}</div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${modeColor[session.config.pipelineMode]}`}>
                  {modeMap[session.config.pipelineMode]}
                </span>
                <span className="text-[11px] text-white/30">{timeAgo(session.updatedAt)}</span>
              </div>
            </div>
            <button
              className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 text-white/30 transition-all shrink-0"
              onClick={e => { e.stopPropagation(); onDelete(session.id); }}>
              <Trash2 className="w-3.5 h-3.5" />
            </button>
            {session.id === activeId && <ChevronRight className="w-3.5 h-3.5 shrink-0 text-white/40" />}
          </div>
        ))}
      </div>
      <div className="p-3 border-t border-white/10">
        <div className="text-xs text-white/25 text-center">Rankify · Open-Source RAG Toolkit</div>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function ChatPage() {
  const STORAGE_KEY = "rankify_sessions_v3";
  const ACTIVE_KEY = "rankify_active_v3";

  const [state, dispatch] = useReducer(reducer, { sessions: [], activeId: "" });
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [serverStatus, setServerStatus] = useState<"unknown" | "online" | "offline">("unknown");
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Load from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const rawId = localStorage.getItem(ACTIVE_KEY);
      const loaded: ChatSession[] = raw ? JSON.parse(raw) : [];
      if (loaded.length === 0) {
        const s = newSession();
        dispatch({ type: "LOAD", sessions: [s], activeId: s.id });
      } else {
        const activeId = rawId && loaded.find(s => s.id === rawId) ? rawId : loaded[0].id;
        dispatch({ type: "LOAD", sessions: loaded, activeId });
      }
    } catch {
      const s = newSession();
      dispatch({ type: "LOAD", sessions: [s], activeId: s.id });
    }
    setHydrated(true);
  }, []);

  // Persist
  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
    localStorage.setItem(ACTIVE_KEY, state.activeId);
  }, [state, hydrated]);

  // Scroll to bottom
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [state.sessions, isLoading]);

  // Check server health
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch("/api/health-check", { signal: AbortSignal.timeout(3000) });
        setServerStatus(r.ok ? "online" : "offline");
      } catch { setServerStatus("offline"); }
    };
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, []);

  const activeSession = state.sessions.find(s => s.id === state.activeId);
  const config = activeSession?.config ?? defaultConfig();

  const handleConfigUpdate = useCallback((update: Partial<SessionConfig>) => {
    dispatch({ type: "UPDATE_CONFIG", id: state.activeId, config: update });
  }, [state.activeId]);

  // When pipeline mode changes → create a new session with the new mode
  const handleModeChange = useCallback((mode: PipelineMode) => {
    const s = newSession({ ...config, pipelineMode: mode });
    dispatch({ type: "CREATE", session: s });
  }, [config]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading || !state.activeId) return;

    const userMsg: ChatMessage = { id: genId(), role: "user", content: input.trim() };
    dispatch({ type: "ADD_MSG", id: state.activeId, message: userMsg });
    setInput("");
    setIsLoading(true);

    // Placeholder assistant message
    const asstId = genId();
    dispatch({ type: "ADD_MSG", id: state.activeId, message: { id: asstId, role: "assistant", content: "" } });

    let pipelineData: ChatMessage["pipelineData"] = {};
    let answerText = "";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...(activeSession?.messages ?? []), userMsg],
          configuration: config,
        }),
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line) continue;

          // Metadata annotation: `2:[...]`
          if (line.startsWith("2:")) {
            try {
              const arr = JSON.parse(line.slice(2)) as string[];
              const meta = JSON.parse(arr[0]) as { pipelineMeta: ChatMessage["pipelineData"] };
              pipelineData = { ...pipelineData, ...meta.pipelineMeta };
              dispatch({ type: "UPDATE_LAST_ASSISTANT", id: state.activeId, content: answerText, pipelineData });
            } catch { /* ignore parse errors */ }
          }

          // Text token: `0:"word "`
          if (line.startsWith("0:")) {
            const match = line.slice(2).match(/^"([\s\S]*)"$/);
            if (match) {
              answerText += match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"');
              dispatch({ type: "UPDATE_LAST_ASSISTANT", id: state.activeId, content: answerText, pipelineData });
            }
          }
        }
      }
    } catch (err) {
      answerText = `⚠️ Error: ${err instanceof Error ? err.message : String(err)}\n\nMake sure the Rankify Python server is running:\n\`python demo_server.py --port 8000\``;
      dispatch({ type: "UPDATE_LAST_ASSISTANT", id: state.activeId, content: answerText, pipelineData });
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, state.activeId, activeSession, config]);

  if (!hydrated) {
    return (
      <div className="flex h-screen items-center justify-center bg-white">
        <Loader2 className="w-6 h-6 animate-spin text-indigo-500" />
      </div>
    );
  }

  const messages = activeSession?.messages ?? [];
  const selectedCat = RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory);

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* ── Sidebar ───────────────────────────────────────────────────────── */}
      <aside className="w-[260px] shrink-0 hidden md:block">
        <SidebarPanel
          sessions={state.sessions}
          activeId={state.activeId}
          onCreate={() => dispatch({ type: "CREATE", session: newSession() })}
          onSelect={id => dispatch({ type: "SELECT", id })}
          onDelete={id => dispatch({ type: "DELETE", id })}
        />
      </aside>

      {/* ── Main ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 h-full min-w-0">

        {/* ── Config Bar ────────────────────────────────────────────────── */}
        <div className="shrink-0 border-b border-slate-100 bg-white/95 backdrop-blur-sm z-10">
          {/* Pipeline Mode Tabs */}
          <div className="px-4 pt-3 pb-2 flex items-center justify-between gap-4">
            <div className="flex gap-1 p-1 bg-slate-100 rounded-lg">
              {(["retrieve", "rerank", "rag"] as PipelineMode[]).map(mode => {
                const labels = { retrieve: "🔍 Retrieve", rerank: "🔀 Retrieve + Rerank", rag: "✨ Full RAG" };
                return (
                  <button key={mode} onClick={() => handleModeChange(mode)}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${config.pipelineMode === mode ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"
                      }`}>
                    {labels[mode]}
                  </button>
                );
              })}
            </div>

            {/* Server status indicator */}
            <div className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${serverStatus === "online" ? "border-emerald-200 bg-emerald-50 text-emerald-700" :
              serverStatus === "offline" ? "border-red-200 bg-red-50 text-red-700" :
                "border-slate-200 bg-slate-50 text-slate-500"
              }`}>
              {serverStatus === "online" ? <CheckCircle2 className="w-3 h-3" /> :
                serverStatus === "offline" ? <AlertCircle className="w-3 h-3" /> :
                  <Loader2 className="w-3 h-3 animate-spin" />}
              {serverStatus === "online" ? "Server Online" :
                serverStatus === "offline" ? "Server Offline" : "Checking..."}
            </div>
          </div>

          {/* Model Selectors */}
          <div className="px-4 pb-3 flex flex-wrap gap-3 items-end">
            {/* Data Source */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                <Database className="w-3 h-3" /> Data Source
              </label>
              <Select value={config.dataSource} onChange={v => handleConfigUpdate({ dataSource: v })}
                options={[{ value: "wiki", label: "Wikipedia" }, { value: "msmarco", label: "MS MARCO" }, { value: "custom", label: "Custom JSONL" }]} />
            </div>

            {config.dataSource === "custom" && (
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">JSONL File</label>
                {uploadedFile ? (
                  <div className="flex items-center gap-1.5 h-8 px-2.5 rounded-md border border-emerald-200 bg-emerald-50 text-xs text-emerald-700">
                    <FileText className="w-3.5 h-3.5" />
                    <span className="max-w-[100px] truncate">{uploadedFile.name}</span>
                    <button onClick={() => setUploadedFile(null)}><X className="w-3 h-3 hover:text-red-500" /></button>
                  </div>
                ) : (
                  <button onClick={() => fileRef.current?.click()}
                    className="h-8 px-3 border border-dashed border-slate-300 rounded-md text-xs text-slate-500 hover:border-indigo-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all flex items-center gap-1.5">
                    <Upload className="w-3.5 h-3.5" /> Upload .jsonl
                  </button>
                )}
                <input ref={fileRef} type="file" accept=".jsonl,.json" className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) setUploadedFile(f); }} />
              </div>
            )}

            {/* Retriever */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                <Search className="w-3 h-3" /> Retriever
              </label>
              <Select value={config.retriever} onChange={v => handleConfigUpdate({ retriever: v })}
                options={RETRIEVERS.map(r => ({ value: r.value, label: r.label }))} className="w-[150px]" />
            </div>

            {/* Reranker Category (rerank + rag mode) */}
            {config.pipelineMode !== "retrieve" && (
              <>
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                    <ListTree className="w-3 h-3" /> Reranker
                  </label>
                  <Select value={config.rerankerCategory}
                    onChange={v => {
                      const cat = RERANKER_CATEGORIES.find(c => c.value === v);
                      handleConfigUpdate({ rerankerCategory: v, rerankerModel: cat?.models[0]?.value ?? "" });
                    }}
                    options={RERANKER_CATEGORIES.map(c => ({ value: c.value, label: c.label }))}
                    className="w-[175px]" />
                </div>

                {selectedCat && selectedCat.models.length > 0 && (
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Model Variant</label>
                    <Select value={config.rerankerModel} onChange={v => handleConfigUpdate({ rerankerModel: v })}
                      options={selectedCat.models} className="w-[200px]" />
                  </div>
                )}
              </>
            )}

            {/* Generator (rag mode only) */}
            {config.pipelineMode === "rag" && (
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                  <Bot className="w-3 h-3" /> Generator
                </label>
                <Select value={config.generator} onChange={v => handleConfigUpdate({ generator: v })}
                  options={GENERATORS.map(g => ({ value: g.value, label: g.label }))} className="w-[150px]" />
              </div>
            )}
          </div>
        </div>

        {/* ── Chat Area ────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center mt-16 space-y-6">
                <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center border border-indigo-100">
                  <Sparkles className="w-8 h-8 text-indigo-500" />
                </div>
                <div className="space-y-2">
                  <h2 className="text-xl font-semibold text-slate-800">
                    {config.pipelineMode === "retrieve" ? "Retrieval Mode" :
                      config.pipelineMode === "rerank" ? "Retrieve + Rerank" : "Full RAG Pipeline"}
                  </h2>
                  <p className="text-sm text-slate-500 max-w-sm mx-auto">
                    {config.pipelineMode === "retrieve"
                      ? `Retrieve real documents from ${config.dataSource === "wiki" ? "Wikipedia" : "MS MARCO"} using ${RETRIEVERS.find(r => r.value === config.retriever)?.label}.`
                      : config.pipelineMode === "rerank"
                        ? `Retrieve then rerank passages using ${RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory)?.label}.`
                        : "Full pipeline: retrieve → rerank → generate with a real language model."}
                  </p>
                </div>
                {serverStatus === "offline" && (
                  <div className="flex items-start gap-2 px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-800 max-w-sm text-left">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                    <div>
                      <div className="font-semibold">Python server not detected</div>
                      <div className="text-xs mt-0.5 text-amber-700">Start with: <code className="font-mono bg-amber-100 px-1 rounded">python demo_server.py</code></div>
                    </div>
                  </div>
                )}
                <div className="flex flex-wrap gap-2 justify-center">
                  {["What is dense retrieval?", "How does BM25 work?", "Compare cross-encoders vs bi-encoders", "What is retrieval-augmented generation?"].map(q => (
                    <button key={q} onClick={() => setInput(q)}
                      className="px-3 py-1.5 text-xs rounded-full border border-slate-200 text-slate-600 hover:border-indigo-300 hover:text-indigo-700 hover:bg-indigo-50 transition-all">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map(m => (
                <div key={m.id} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  {m.role !== "user" && (
                    <div className="w-7 h-7 shrink-0 rounded-lg bg-indigo-100 flex items-center justify-center mt-1">
                      <Layers className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}
                  <div className={`flex flex-col gap-2 ${m.role === "user" ? "items-end max-w-[75%]" : "items-start w-full"}`}>
                    {m.role !== "user" && m.pipelineData && (
                      <div className="w-full">
                        <PipelineViz message={m} config={config} />
                      </div>
                    )}
                    {(m.content || (isLoading && m.role === "assistant")) && (
                      <div className={`px-4 py-3 text-sm leading-relaxed rounded-2xl shadow-sm whitespace-pre-wrap ${m.role === "user"
                        ? "bg-slate-900 text-white rounded-br-sm"
                        : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"
                        }`}>
                        {m.content || (
                          <span className="flex gap-1 py-0.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.1s]" />
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.2s]" />
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* ── Input Bar ─────────────────────────────────────────────────── */}
        <div className="shrink-0 border-t border-slate-100 bg-white px-4 py-4">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit}
              className="flex items-end gap-2 bg-white border border-slate-300 shadow-sm focus-within:shadow-md focus-within:border-indigo-400 transition-all rounded-2xl p-2 pl-4">
              <textarea value={input} onChange={e => setInput(e.target.value)}
                placeholder={`Search with ${RETRIEVERS.find(r => r.value === config.retriever)?.label}${config.pipelineMode !== "retrieve" ? " → rerank" : ""}${config.pipelineMode === "rag" ? " → generate" : ""}...`}
                className="flex-1 max-h-32 min-h-[40px] bg-transparent resize-none outline-none text-sm pt-2 pb-1 placeholder:text-slate-400"
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }} />
              <button type="submit" disabled={isLoading || !input.trim()}
                className="h-9 w-9 shrink-0 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 transition-colors flex items-center justify-center shadow-sm mb-0.5">
                {isLoading ? <Loader2 className="w-4 h-4 text-white animate-spin" /> : <Send className="w-4 h-4 text-white ml-0.5" />}
              </button>
            </form>
            <div className="flex items-center justify-center gap-2 mt-1.5 flex-wrap text-[11px] text-slate-400">
              <span>{RETRIEVERS.find(r => r.value === config.retriever)?.label}</span>
              {config.pipelineMode !== "retrieve" && <><span>·</span><span>{RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory)?.label} / {selectedCat?.models.find(m => m.value === config.rerankerModel)?.label}</span></>}
              {config.pipelineMode === "rag" && <><span>·</span><span>{GENERATORS.find(g => g.value === config.generator)?.label}</span></>}
              <span>·</span>
              <span>{config.dataSource === "wiki" ? "Wikipedia" : config.dataSource === "msmarco" ? "MS MARCO" : uploadedFile?.name ?? "Custom"}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
