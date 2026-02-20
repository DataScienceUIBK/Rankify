import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Send, Cpu, ChevronRight, FileText } from 'lucide-react';
import { GlassCard, Button, Badge } from '../components/ui';

export function RAG() {
    const [query, setQuery] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hello! I am the Rankify RAG Generator. Ask me anything, and I will retrieve context, perform re-ranking, and generate an answer based on the best available information.'
        }
    ]);

    const handleSend = (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: query }]);
        setQuery('');
        setIsGenerating(true);

        // Simulate RAG pipeline latency (Retrieval -> Reranking -> Generation)
        setTimeout(() => {
            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: 'Based on the retrieved context, Rankify provides a unified interface for retrieval augmented generation (RAG) by integrating state-of-the-art retrievers and cross-encoder re-rankers before feeding the context into an LLM generator. This ensures highly accurate and grounded responses.',
                    context: [
                        { id: '1', text: 'Rankify is a modular and efficient retrieval and RAG framework.' },
                        { id: '2', text: 'It supports multiple generator architectures like vLLM and OpenAI.' }
                    ]
                }
            ]);
            setIsGenerating(false);
        }, 2500);
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 min-h-[calc(100vh-100px)] flex flex-col">
            <header className="space-y-2 shrink-0">
                <h1 className="text-4xl font-bold flex items-center gap-3">
                    <MessageSquare className="w-10 h-10 text-pink-400" />
                    RAG Pipeline
                </h1>
                <p className="text-slate-400 text-lg">
                    End-to-End Retrieval Augmented Generation using Rankify Generator module.
                </p>
            </header>

            <div className="grid lg:grid-cols-4 gap-8 flex-1">
                {/* Chat Interface */}
                <div className="lg:col-span-3 flex flex-col h-full bg-[rgba(15,17,26,0.5)] border border-[#ffffff14] rounded-2xl overflow-hidden shadow-2xl">
                    {/* Chat Messages */}
                    <div className="flex-1 p-6 overflow-y-auto space-y-6">
                        <AnimatePresence>
                            {messages.map((msg, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-2' : ''}`}>
                                        <div className={`
                      p-4 rounded-2xl shadow-sm
                      ${msg.role === 'user'
                                                ? 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-tr-sm ml-4'
                                                : 'bg-[rgba(255,255,255,0.05)] border border-[#ffffff14] text-slate-200 rounded-tl-sm mr-4'
                                            }
                    `}>
                                            {msg.role === 'assistant' && idx === 0 && (
                                                <Cpu className="w-5 h-5 mb-2 text-pink-400" />
                                            )}

                                            <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>

                                            {msg.context && (
                                                <div className="mt-4 pt-4 border-t border-[#ffffff14]">
                                                    <p className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-2 flex items-center gap-1">
                                                        <FileText className="w-3 h-3" /> Grounded Context Used
                                                    </p>
                                                    <div className="space-y-2">
                                                        {msg.context.map((ctx, i) => (
                                                            <div key={i} className="text-sm bg-black/20 p-2 rounded text-slate-400 border border-white/5">
                                                                <span className="text-pink-400/70 mr-2">[{i + 1}]</span> {ctx.text}
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            ))}

                            {isGenerating && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex justify-start"
                                >
                                    <div className="bg-[rgba(255,255,255,0.05)] border border-[#ffffff14] p-4 rounded-2xl rounded-tl-sm w-24 flex items-center justify-center gap-2">
                                        <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce"></div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-[rgba(0,0,0,0.2)] border-t border-[#ffffff14]">
                        <form onSubmit={handleSend} className="relative max-w-4xl mx-auto flex items-end gap-2 text-white">
                            <div className="relative flex-1 bg-[rgba(255,255,255,0.03)] border border-[#ffffff14] rounded-2xl focus-within:border-pink-500/50 focus-within:bg-[rgba(255,255,255,0.05)] transition-all flex items-center p-2">
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Ask a question..."
                                    className="w-full bg-transparent border-none text-white px-4 py-2 focus:outline-none focus:ring-0 placeholder:text-slate-500"
                                    disabled={isGenerating}
                                />
                                <button
                                    type="submit"
                                    disabled={isGenerating || !query.trim()}
                                    className="p-3 bg-pink-500 hover:bg-pink-600 disabled:opacity-50 disabled:hover:bg-pink-500 rounded-xl text-white transition-colors"
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                {/* Info Panel */}
                <div className="lg:col-span-1 hidden lg:flex flex-col gap-4">
                    <GlassCard className="flex-1">
                        <h3 className="font-semibold text-lg mb-4 text-pink-400 flex items-center gap-2">
                            <Settings2 className="w-4 h-4" /> Pipeline Details
                        </h3>

                        <div className="space-y-6">
                            <div className="space-y-2">
                                <span className="text-xs text-slate-400 font-medium uppercase min-w-[40px]">Generator Backend</span>
                                <div className="flex items-center gap-2 bg-white/5 p-2 rounded-lg border border-white/5">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                    <span className="text-sm font-medium">vLLM Engine</span>
                                </div>
                            </div>

                            <div className="space-y-3 relative before:absolute before:inset-0 before:-left-3 before:w-px before:bg-white/10 before:my-2 ml-4">
                                <div className="relative">
                                    <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-blue-400"></div>
                                    <p className="text-sm text-slate-300">1. Retrieve (BGE)</p>
                                    <p className="text-xs text-slate-500">Top 100 documents</p>
                                </div>
                                <div className="relative">
                                    <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-purple-400"></div>
                                    <p className="text-sm text-slate-300">2. Rerank (FlashRank)</p>
                                    <p className="text-xs text-slate-500">Top 5 contexts selected</p>
                                </div>
                                <div className="relative">
                                    <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-pink-400"></div>
                                    <p className="text-sm text-slate-300">3. Generate (Llama-3)</p>
                                    <p className="text-xs text-slate-500">Streaming response</p>
                                </div>
                            </div>
                        </div>
                    </GlassCard>
                </div>
            </div>
        </div>
    );
}
