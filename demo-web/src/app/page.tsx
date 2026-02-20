"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send, Search, ListTree, Bot, Database, Upload, Sparkles,
  FileJson, X, ChevronDown, ChevronUp, Layers,
  CheckCircle2, Circle, Loader2, FileText, ArrowRight
} from "lucide-react";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { ScrollArea } from "../components/ui/scroll-area";
import { Badge } from "../components/ui/badge";
import { Sidebar } from "../components/Sidebar";
import { useChatHistory, type ChatMessage, type ChatSession } from "./useChatHistory";
import { RETRIEVERS, RERANKER_CATEGORIES, GENERATORS } from "./models";

// ─── Types ──────────────────────────────────────────────────────────────────────
type PipelineMode = "retrieve" | "rerank" | "rag";

// ─── Mock pipeline data generator ──────────────────────────────────────────────
function generateMockPipelineData(query: string, config: ChatSession["config"]) {
  const docs = [
    { id: "1", text: `"Dense retrieval methods like ${config.retriever.toUpperCase()} show strong performance on open-domain QA benchmarks, particularly when combined with effective reranking strategies..."`, score: 0.923 },
    { id: "2", text: `"Passage retrieval for question answering requires balancing recall and precision. BM25 provides strong lexical overlap baselines while neural models capture semantic similarity..."`, score: 0.887 },
    { id: "3", text: `"Cross-encoder rerankers significantly improve the quality of retrieved passages by jointly encoding the query-document pair, enabling fine-grained relevance scoring..."`, score: 0.854 },
    { id: "4", text: `"The ${query.slice(0, 30)} topic has been extensively studied in information retrieval literature, with retrieval-augmented generation emerging as a key application..."`, score: 0.812 },
    { id: "5", text: `"Rankify provides a unified framework for retrieval, reranking, and generation tasks, enabling systematic comparison of different pipeline configurations..."`, score: 0.778 },
  ];

  const reranked = config.pipelineMode !== "retrieve" ? [
    { ...docs[2], score: 0.97 },
    { ...docs[0], score: 0.94 },
    { ...docs[3], score: 0.88 },
    { ...docs[1], score: 0.82 },
    { ...docs[4], score: 0.71 },
  ] : [];

  return { retrievedDocs: docs, rerankedDocs: reranked };
}

// ─── Pipeline Step Indicator ────────────────────────────────────────────────────
function PipelineStep({ label, icon, done, active }: { label: string; icon: React.ReactNode; done: boolean; active: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-slate-400">{icon}</span>
      <span className={`text-xs font-medium ${done ? "text-slate-700" : active ? "text-indigo-600" : "text-slate-300"}`}>{label}</span>
    </div>
  );
}

// ─── Document Card ──────────────────────────────────────────────────────────────
function DocCard({ doc, rank, isReranked }: { doc: { id: string; text: string; score: number }; rank: number; isReranked?: boolean }) {
  return (
    <div className={`flex gap-2 p-2.5 rounded-lg border text-xs ${isReranked ? "bg-indigo-50/60 border-indigo-100" : "bg-slate-50 border-slate-100"}`}>
      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5 ${isReranked ? "bg-indigo-100 text-indigo-700" : "bg-slate-200 text-slate-600"}`}>
        {rank}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-slate-600 leading-relaxed line-clamp-2 italic">{doc.text}</p>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] text-slate-400">ID: {doc.id}</span>
          <span className={`text-[10px] font-semibold ${isReranked ? "text-indigo-600" : "text-slate-500"}`}>
            score: {doc.score.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Pipeline Message Block ─────────────────────────────────────────────────────
function PipelineMessageBlock({ message, config }: { message: ChatMessage; config: ChatSession["config"] }) {
  const [showDocs, setShowDocs] = useState(false);
  const [showReranked, setShowReranked] = useState(false);

  const pd = message.pipelineData;
  if (!pd) return null;

  const retrieverLabel = RETRIEVERS.find(r => r.value === config.retriever)?.label ?? config.retriever;
  const rerankerCat = RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory);
  const rerankerModel = rerankerCat?.models.find(m => m.value === config.rerankerModel)?.label ?? config.rerankerModel;
  const generatorLabel = GENERATORS.find(g => g.value === config.generator)?.label ?? config.generator;

  return (
    <div className="space-y-2 w-full">
      {/* ── Step 1: Retrieved Documents ── */}
      {pd.retrievedDocs && pd.retrievedDocs.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm">
          <button
            onClick={() => setShowDocs(v => !v)}
            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-slate-50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-emerald-100 flex items-center justify-center">
                <Search className="w-3 h-3 text-emerald-600" />
              </div>
              <span className="text-xs font-semibold text-slate-700">Documents Retrieved</span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 bg-emerald-50 text-emerald-700 border-emerald-100">
                {retrieverLabel}
              </Badge>
              <span className="text-[11px] text-slate-400">{pd.retrievedDocs.length} docs</span>
            </div>
            {showDocs ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
          </button>
          {showDocs && (
            <div className="px-3 pb-3 space-y-1.5 border-t border-slate-100 pt-2">
              {pd.retrievedDocs.map((doc, i) => (
                <DocCard key={doc.id} doc={doc} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Connector Arrow ── */}
      {pd.rerankedDocs && pd.rerankedDocs.length > 0 && (
        <div className="flex justify-center">
          <ArrowRight className="w-4 h-4 text-slate-300" />
        </div>
      )}

      {/* ── Step 2: Reranked Documents ── */}
      {pd.rerankedDocs && pd.rerankedDocs.length > 0 && (
        <div className="rounded-xl border border-indigo-200 bg-white overflow-hidden shadow-sm">
          <button
            onClick={() => setShowReranked(v => !v)}
            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-indigo-50/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center">
                <ListTree className="w-3 h-3 text-indigo-600" />
              </div>
              <span className="text-xs font-semibold text-slate-700">Reranked</span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 bg-indigo-50 text-indigo-700 border-indigo-100">
                {rerankerCat?.label ?? config.rerankerCategory} · {rerankerModel}
              </Badge>
            </div>
            {showReranked ? <ChevronUp className="w-3.5 h-3.5 text-slate-400" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400" />}
          </button>
          {showReranked && (
            <div className="px-3 pb-3 space-y-1.5 border-t border-indigo-100 pt-2">
              {pd.rerankedDocs.map((doc, i) => (
                <DocCard key={doc.id} doc={doc} rank={i + 1} isReranked />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Step 3: Generator ── */}
      {config.pipelineMode === "rag" && (
        <>
          <div className="flex justify-center">
            <ArrowRight className="w-4 h-4 text-slate-300" />
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-violet-200 bg-violet-50/40">
            <div className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center">
              <Bot className="w-3 h-3 text-violet-600" />
            </div>
            <span className="text-xs font-semibold text-slate-700">Generator</span>
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 bg-violet-50 text-violet-700 border-violet-100">
              {generatorLabel}
            </Badge>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Config Panel ───────────────────────────────────────────────────────────────
function ConfigPanel({
  config,
  onUpdate,
  uploadedFile,
  onFileUpload,
  onFileClear,
}: {
  config: ChatSession["config"];
  onUpdate: (c: Partial<ChatSession["config"]>) => void;
  uploadedFile: File | null;
  onFileUpload: (f: File) => void;
  onFileClear: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const selectedCat = RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory);

  return (
    <div className="shrink-0 border-b border-slate-100 bg-white/95 backdrop-blur-sm z-10">
      {/* Pipeline Mode Tabs */}
      <div className="px-4 pt-3 pb-0">
        <div className="flex gap-1 p-1 bg-slate-100 rounded-lg w-fit max-w-full">
          {(["retrieve", "rerank", "rag"] as PipelineMode[]).map(mode => {
            const labels: Record<PipelineMode, string> = { retrieve: "🔍 Retrieve", rerank: "🔀 Retrieve + Rerank", rag: "✨ Full RAG" };
            return (
              <button
                key={mode}
                onClick={() => onUpdate({ pipelineMode: mode })}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${config.pipelineMode === mode
                  ? "bg-white text-slate-800 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
                  }`}
              >
                {labels[mode]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Model Selectors */}
      <div className="p-4 flex flex-wrap gap-3 items-end">
        {/* Data Source */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Database className="w-3 h-3" /> Data Source
          </label>
          <Select value={config.dataSource} onValueChange={v => onUpdate({ dataSource: v })}>
            <SelectTrigger className="h-8 text-xs w-[140px] bg-slate-50 border-slate-200">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="msmarco">MS MARCO</SelectItem>
              <SelectItem value="wiki">Wikipedia</SelectItem>
              <SelectItem value="custom">Custom JSONL</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Custom file upload (only when "custom" selected) */}
        {config.dataSource === "custom" && (
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
              <FileJson className="w-3 h-3" /> JSONL File
            </label>
            {uploadedFile ? (
              <div className="flex items-center gap-1.5 h-8 px-2.5 rounded-md border border-emerald-200 bg-emerald-50 text-xs text-emerald-700">
                <FileText className="w-3.5 h-3.5" />
                <span className="max-w-[100px] truncate">{uploadedFile.name}</span>
                <button onClick={onFileClear}><X className="w-3 h-3 text-emerald-600 hover:text-red-500" /></button>
              </div>
            ) : (
              <button
                onClick={() => fileRef.current?.click()}
                className="h-8 px-3 border border-dashed border-slate-300 rounded-md text-xs text-slate-500 hover:border-indigo-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all flex items-center gap-1.5"
              >
                <Upload className="w-3.5 h-3.5" /> Upload .jsonl
              </button>
            )}
            <input
              ref={fileRef}
              type="file"
              accept=".jsonl,.json"
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) onFileUpload(f); }}
            />
          </div>
        )}

        {/* Retriever */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Search className="w-3 h-3" /> Retriever
          </label>
          <Select value={config.retriever} onValueChange={v => onUpdate({ retriever: v })}>
            <SelectTrigger className="h-8 text-xs w-[160px] bg-slate-50 border-slate-200">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RETRIEVERS.map(r => (
                <SelectItem key={r.value} value={r.value}>
                  <div>
                    <div className="font-medium">{r.label}</div>
                    <div className="text-[10px] text-slate-400">{r.description}</div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Reranker Category — only shown in rerank / rag modes */}
        {config.pipelineMode !== "retrieve" && (
          <>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                <ListTree className="w-3 h-3" /> Reranker
              </label>
              <Select
                value={config.rerankerCategory}
                onValueChange={v => {
                  const cat = RERANKER_CATEGORIES.find(c => c.value === v);
                  onUpdate({
                    rerankerCategory: v,
                    rerankerModel: cat?.models[0]?.value ?? "",
                  });
                }}
              >
                <SelectTrigger className="h-8 text-xs w-[180px] bg-slate-50 border-slate-200">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-64">
                  {RERANKER_CATEGORIES.map(cat => (
                    <SelectItem key={cat.value} value={cat.value}>
                      <div>
                        <div className="font-medium">{cat.label}</div>
                        <div className="text-[10px] text-slate-400">{cat.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Reranker Model — shown when category has sub-models */}
            {selectedCat && selectedCat.models.length > 0 && (
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                  Model Variant
                </label>
                <Select value={config.rerankerModel} onValueChange={v => onUpdate({ rerankerModel: v })}>
                  <SelectTrigger className="h-8 text-xs w-[200px] bg-slate-50 border-slate-200">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-64">
                    {selectedCat.models.map(m => (
                      <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </>
        )}

        {/* Generator — only in rag mode */}
        {config.pipelineMode === "rag" && (
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
              <Bot className="w-3 h-3" /> Generator
            </label>
            <Select value={config.generator} onValueChange={v => onUpdate({ generator: v })}>
              <SelectTrigger className="h-8 text-xs w-[160px] bg-slate-50 border-slate-200">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GENERATORS.map(g => (
                  <SelectItem key={g.value} value={g.value}>
                    <div>
                      <div className="font-medium">{g.label}</div>
                      <div className="text-[10px] text-slate-400">{g.description}</div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────────────────
export default function ChatPage() {
  const {
    sessions, activeId, activeSession, hydrated,
    createSession, selectSession, deleteSession,
    updateSessionConfig, addMessage, updateLastAssistantMessage,
  } = useChatHistory();

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeSession?.messages.length, isLoading]);

  const config = activeSession?.config ?? {
    pipelineMode: "rag" as PipelineMode,
    dataSource: "msmarco",
    retriever: "bge",
    rerankerCategory: "flashrank",
    rerankerModel: "ms-marco-MiniLM-L-12-v2",
    generator: "openai",
  };

  const handleConfigUpdate = useCallback((update: Partial<ChatSession["config"]>) => {
    if (activeId) updateSessionConfig(activeId, update);
  }, [activeId, updateSessionConfig]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading || !activeId) return;

    const userMsg: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: "user",
      content: input.trim(),
    };
    addMessage(activeId, userMsg);
    setInput("");
    setIsLoading(true);

    // Placeholder assistant message
    const assistantId = Math.random().toString(36).slice(2);
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
    };
    addMessage(activeId, assistantMsg);

    try {
      const pipelineData = generateMockPipelineData(input.trim(), config);

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...(activeSession?.messages ?? []), userMsg],
          configuration: config,
        }),
      });

      if (!res.ok || !res.body) throw new Error("API error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        // Parse the ai-stream format: `0:"word "`
        const matches = chunk.matchAll(/0:"((?:[^"\\]|\\.)*)"/g);
        for (const m of matches) {
          fullText += m[1].replace(/\\n/g, "\n").replace(/\\"/g, "\"");
        }
        updateLastAssistantMessage(activeId, fullText, pipelineData);
      }
    } catch {
      updateLastAssistantMessage(activeId, "⚠️ Failed to connect to Rankify backend. Make sure the Python server is running on port 8000.");
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, activeId, activeSession, config, addMessage, updateLastAssistantMessage]);

  if (!hydrated) {
    return (
      <div className="flex h-screen items-center justify-center bg-white">
        <Loader2 className="w-6 h-6 animate-spin text-indigo-500" />
      </div>
    );
  }

  const messages = activeSession?.messages ?? [];

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* ── Sidebar ── */}
      {sidebarOpen && (
        <aside className="w-[260px] shrink-0 border-r border-slate-100 hidden md:flex flex-col">
          <Sidebar
            className="flex-1"
            sessions={sessions}
            activeId={activeId}
            onCreateSession={createSession}
            onSelectSession={selectSession}
            onDeleteSession={deleteSession}
          />
        </aside>
      )}

      {/* ── Main Content ── */}
      <div className="flex flex-col flex-1 h-full min-w-0">

        {/* Config Panel */}
        {activeSession && (
          <ConfigPanel
            config={config}
            onUpdate={handleConfigUpdate}
            uploadedFile={uploadedFile}
            onFileUpload={setUploadedFile}
            onFileClear={() => setUploadedFile(null)}
          />
        )}

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">

            {messages.length === 0 ? (
              /* ── Empty State ── */
              <div className="flex flex-col items-center justify-center text-center mt-20 space-y-6">
                <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center border border-indigo-100 shadow-sm">
                  <Sparkles className="w-8 h-8 text-indigo-500" />
                </div>
                <div className="space-y-2">
                  <h2 className="text-xl font-semibold text-slate-800">
                    {config.pipelineMode === "retrieve" ? "Retrieve Documents" :
                      config.pipelineMode === "rerank" ? "Retrieve + Rerank" :
                        "Full RAG Pipeline"}
                  </h2>
                  <p className="text-sm text-slate-500 max-w-sm mx-auto">
                    {config.pipelineMode === "retrieve"
                      ? `Enter a query to retrieve passages from ${config.dataSource === "msmarco" ? "MS MARCO" : config.dataSource === "wiki" ? "Wikipedia" : "your custom corpus"} using ${RETRIEVERS.find(r => r.value === config.retriever)?.label}.`
                      : config.pipelineMode === "rerank"
                        ? `Retrieve passages and rerank them using ${RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory)?.label}.`
                        : "Run the full Rankify pipeline: retrieve → rerank → generate an answer."}
                  </p>
                </div>

                {/* Quick-start chips */}
                <div className="flex flex-wrap gap-2 justify-center mt-2">
                  {["What is dense retrieval?", "How does BM25 work?", "Compare cross-encoders vs bi-encoders"].map(q => (
                    <button
                      key={q}
                      onClick={() => { setInput(q); }}
                      className="px-3 py-1.5 text-xs rounded-full border border-slate-200 text-slate-600 hover:border-indigo-300 hover:text-indigo-700 hover:bg-indigo-50 transition-all"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m) => (
                <div key={m.id} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  {m.role !== "user" && (
                    <div className="w-7 h-7 shrink-0 rounded-lg bg-indigo-100 flex items-center justify-center border border-indigo-100 mt-1">
                      <Layers className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}

                  <div className={`flex flex-col gap-2 ${m.role === "user" ? "items-end max-w-[75%]" : "items-start w-full"}`}>

                    {/* Pipeline display (assistant only) */}
                    {m.role !== "user" && m.pipelineData && (
                      <div className="w-full">
                        <PipelineMessageBlock message={m} config={config} />
                      </div>
                    )}

                    {/* Text bubble */}
                    {m.content && (
                      <div className={`px-4 py-3 text-[14px] leading-relaxed rounded-2xl shadow-sm ${m.role === "user"
                        ? "bg-slate-900 text-white rounded-br-sm"
                        : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"
                        }`}>
                        {m.content || (isLoading && m.role === "assistant"
                          ? <span className="flex gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.1s]" />
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.2s]" />
                          </span>
                          : null
                        )}
                      </div>
                    )}

                    {!m.content && m.role === "assistant" && isLoading && (
                      <div className="px-4 py-3 text-[14px] leading-relaxed rounded-2xl shadow-sm bg-white border border-slate-200 rounded-bl-sm">
                        <span className="flex gap-1 py-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.1s]" />
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.2s]" />
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* ── Input Bar ── */}
        <div className="shrink-0 border-t border-slate-100 bg-white px-4 py-4">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={handleSubmit}
              className="flex items-end gap-2 bg-white border border-slate-300 shadow-sm focus-within:shadow-md focus-within:border-indigo-400 transition-all rounded-2xl p-2 pl-4"
            >
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder={`Ask anything — using ${config.pipelineMode === "retrieve" ? "retrieval only" : config.pipelineMode === "rerank" ? "retrieval + reranking" : "full RAG pipeline"}...`}
                className="flex-1 max-h-32 min-h-[40px] bg-transparent resize-none outline-none text-sm pt-2 pb-1 placeholder:text-slate-400"
                onKeyDown={e => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
              />
              <Button
                type="submit"
                disabled={isLoading || !input.trim()}
                size="icon"
                className="h-9 w-9 shrink-0 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 transition-colors shadow-sm mb-0.5"
              >
                {isLoading
                  ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                  : <Send className="w-4 h-4 text-white ml-0.5" />
                }
              </Button>
            </form>

            {/* Active config summary pill */}
            <div className="flex items-center justify-center gap-2 mt-2 flex-wrap">
              <span className="text-[11px] text-slate-400 flex items-center gap-1">
                <Search className="w-3 h-3" />
                {RETRIEVERS.find(r => r.value === config.retriever)?.label}
              </span>
              {config.pipelineMode !== "retrieve" && (
                <>
                  <span className="text-slate-200">·</span>
                  <span className="text-[11px] text-slate-400 flex items-center gap-1">
                    <ListTree className="w-3 h-3" />
                    {RERANKER_CATEGORIES.find(c => c.value === config.rerankerCategory)?.label}
                  </span>
                </>
              )}
              {config.pipelineMode === "rag" && (
                <>
                  <span className="text-slate-200">·</span>
                  <span className="text-[11px] text-slate-400 flex items-center gap-1">
                    <Bot className="w-3 h-3" />
                    {GENERATORS.find(g => g.value === config.generator)?.label}
                  </span>
                </>
              )}
              <span className="text-slate-200">·</span>
              <span className="text-[11px] text-slate-400 flex items-center gap-1">
                <Database className="w-3 h-3" />
                {config.dataSource === "msmarco" ? "MS MARCO" : config.dataSource === "wiki" ? "Wikipedia" : uploadedFile?.name ?? "Custom"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
