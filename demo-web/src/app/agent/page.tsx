"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Code2, Sparkles, ServerCrash, Loader2, ArrowRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AgentRecommendation {
    code_snippet: string | null;
    retriever: string | null;
    reranker: string | null;
    rag_method: string | null;
}

interface ChatMsg {
    id: string;
    role: "user" | "assistant";
    content: string;
    recommendation?: AgentRecommendation | null;
}

export default function AgentPage() {
    const [messages, setMessages] = useState<ChatMsg[]>([
        {
            id: "1", role: "assistant", content: "Hi! I am RankifyAgent. Tell me about your requirements (e.g. \"I need a fast pipeline with no GPU for medical data\") and I'll recommend the optimal retrieval, reranking, and generation setup for you."
        }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [activeRec, setActiveRec] = useState<AgentRecommendation | null>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const sessionId = useRef(Math.random().toString(36).slice(2) + Date.now().toString(36));

    // Auto-scroll
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMsg = input.trim();
        setInput("");
        const msgId = Math.random().toString(36).slice(2);

        setMessages(prev => [...prev, { id: Date.now().toString(), role: "user", content: userMsg }]);
        setMessages(prev => [...prev, { id: msgId, role: "assistant", content: "" }]);
        setLoading(true);

        try {
            const res = await fetch("/api/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMsg, session_id: sessionId.current })
            });

            if (!res.body) throw new Error("No response body");

            const reader = res.body.getReader();
            const dec = new TextDecoder();
            let recData: AgentRecommendation | null = null;
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += dec.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() ?? '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const raw = line.slice(6).trim();
                    if (!raw) continue;

                    try {
                        const event = JSON.parse(raw);
                        if (event.type === "token") {
                            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: m.content + event.content } : m));
                        } else if (event.type === "recommendation") {
                            recData = event.data;
                            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, recommendation: recData } : m));
                            setActiveRec(recData); // auto-show the latest recommendation
                        } else if (event.type === "error") {
                            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: m.content + "\n\n⚠️ Error: " + event.message } : m));
                        }
                    } catch (e) {
                        // ignore parse errors for partial chunks
                    }
                }
            }
        } catch (err: any) {
            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: m.content + "\n\nConnection Error." } : m));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex h-[calc(100vh-56px)] overflow-hidden bg-white">
            {/* Chat Column */}
            <div className="flex-1 flex flex-col min-w-0 border-r border-slate-100">
                <div className="flex-1 overflow-y-auto p-4 md:p-8">
                    <div className="max-w-3xl mx-auto flex flex-col gap-6">
                        {messages.map(m => (
                            <div key={m.id} className={`flex gap-4 ${m.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                                <div className={`w-8 h-8 rounded-full flex shrink-0 items-center justify-center ${m.role === "user" ? "bg-slate-100 text-slate-600" : "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"}`}>
                                    {m.role === "user" ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
                                </div>
                                <div className={`flex flex-col gap-2 max-w-[80%] ${m.role === "user" ? "items-end" : "items-start"}`}>
                                    <div className={`p-4 rounded-2xl text-sm leading-relaxed ${m.role === "user" ? "bg-slate-100 text-slate-800 rounded-tr-sm" : "bg-white border border-slate-100 shadow-sm text-slate-700 rounded-tl-sm prose prose-sm max-w-none prose-slate"}`}>
                                        {m.content ? (
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                {m.content}
                                            </ReactMarkdown>
                                        ) : (
                                            <span className="animate-pulse text-slate-400">Thinking...</span>
                                        )}
                                    </div>
                                    {/* Action button to view recommendation if it exists */}
                                    {m.recommendation && (
                                        <button
                                            onClick={() => setActiveRec(m.recommendation!)}
                                            className="group flex items-center gap-2 px-3 py-2 bg-indigo-50 border border-indigo-100 rounded-lg text-xs font-semibold text-indigo-700 hover:bg-indigo-100 transition-colors">
                                            <Code2 className="w-4 h-4 text-indigo-500 group-hover:scale-110 transition-transform" />
                                            View Configuration & Code
                                            <ArrowRight className="w-3 h-3 ml-1 opacity-50" />
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                        <div ref={bottomRef} />
                    </div>
                </div>

                {/* Input Bar */}
                <div className="p-4 bg-white border-t border-slate-100">
                    <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-2 relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Describe your use case..."
                            className="flex-1 h-12 px-4 rounded-xl border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 transition-all text-sm"
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || loading}
                            className="h-12 w-12 flex items-center justify-center bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-sm"
                        >
                            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                        </button>
                    </form>
                </div>
            </div>

            {/* Config & Code Panel (Right) */}
            <div className={`w-[400px] shrink-0 bg-slate-50 border-l border-slate-200 flex flex-col transition-all duration-300 ${activeRec ? "translate-x-0" : "translate-x-full hidden"}`}>
                <div className="p-4 border-b border-slate-200 bg-white flex items-center justify-between">
                    <h2 className="font-bold text-slate-800 flex items-center gap-2">
                        <Bot className="w-5 h-5 text-indigo-500" /> Recommended Setup
                    </h2>
                    <button onClick={() => setActiveRec(null)} className="text-slate-400 hover:text-slate-700 text-xs">Close</button>
                </div>

                {activeRec && (
                    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-6">
                        {/* Summary Badges */}
                        <div className="flex flex-col gap-3">
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Pipeline Components</h3>
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center gap-3 p-3 bg-white border border-slate-200 rounded-xl shadow-sm">
                                    <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center text-emerald-600 text-sm font-bold shrink-0">1</div>
                                    <div className="min-w-0">
                                        <div className="text-[10px] uppercase font-bold text-slate-400">Retriever</div>
                                        <div className="text-sm font-semibold text-slate-800 truncate">{activeRec.retriever || "None"}</div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3 p-3 bg-white border border-slate-200 rounded-xl shadow-sm">
                                    <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center text-amber-600 text-sm font-bold shrink-0">2</div>
                                    <div className="min-w-0">
                                        <div className="text-[10px] uppercase font-bold text-slate-400">Reranker</div>
                                        <div className="text-sm font-semibold text-slate-800 truncate">{activeRec.reranker || "None"}</div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3 p-3 bg-white border border-slate-200 rounded-xl shadow-sm">
                                    <div className="w-8 h-8 rounded-lg bg-violet-100 flex items-center justify-center text-violet-600 text-sm font-bold shrink-0">3</div>
                                    <div className="min-w-0">
                                        <div className="text-[10px] uppercase font-bold text-slate-400">Generator & Method</div>
                                        <div className="text-sm font-semibold text-slate-800 truncate">{activeRec.rag_method || "None"}</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Code Snippet */}
                        {activeRec.code_snippet && (
                            <div className="flex flex-col gap-2">
                                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                                    <span>Python SDK Code</span>
                                    <button
                                        onClick={() => navigator.clipboard.writeText(activeRec.code_snippet!)}
                                        className="text-[10px] px-2 py-1 bg-indigo-100 text-indigo-700 rounded hover:bg-indigo-200 transition-colors">
                                        Copy
                                    </button>
                                </h3>
                                <div className="p-3 bg-[#0d1117] text-slate-300 font-mono text-[11px] rounded-xl overflow-x-auto border border-slate-800 shadow-inner">
                                    <pre><code>{activeRec.code_snippet}</code></pre>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
