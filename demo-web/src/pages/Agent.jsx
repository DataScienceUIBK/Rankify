import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, Terminal, Code, Sparkles, Server } from 'lucide-react';
import { GlassCard, Button, Badge } from '../components/ui';

export function Agent() {
    const [task, setTask] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const [recommendation, setRecommendation] = useState(null);

    const handleRecommend = (e) => {
        e.preventDefault();
        if (!task.trim()) return;

        setIsThinking(true);
        setRecommendation(null);

        setTimeout(() => {
            setRecommendation({
                task: task,
                retriever: {
                    name: 'BGE-M3',
                    reason: 'Best overall performance for general QA tasks, supporting multi-linguality and dense retrieval natively.'
                },
                reranker: {
                    name: 'FlashRank',
                    reason: 'Ultra-fast inference suitable for production environments requiring low latency.'
                },
                code: `from rankify.agent import RankifyAgent, recommend

# Quick recommendation based on your task
result = recommend(task="${task}", gpu=True)

print(f"Recommended Retriever: {result.retriever.name}")
print(f"Recommended Reranker: {result.reranker.name}")

# Now you can initialize the pipeline
from rankify import pipeline
rag = pipeline(
    "rag", 
    retriever=result.retriever.name, 
    reranker=result.reranker.name
)`
            });
            setIsThinking(false);
        }, 2000);
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="space-y-2">
                <h1 className="text-4xl font-bold flex items-center gap-3">
                    <Bot className="w-10 h-10 text-emerald-400" />
                    Rankify<span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400">Agent</span>
                </h1>
                <p className="text-slate-400 text-lg">
                    AI-powered model selection and configuration for your specific use cases.
                </p>
            </header>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Interaction Panel */}
                <div className="space-y-6">
                    <GlassCard className="h-full">
                        <h3 className="text-xl font-semibold flex items-center gap-2 mb-6 text-emerald-400">
                            <Sparkles className="w-5 h-5" /> Describe Your Task
                        </h3>

                        <form onSubmit={handleRecommend} className="space-y-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-400 block">
                                    What are you trying to build?
                                </label>
                                <textarea
                                    value={task}
                                    onChange={(e) => setTask(e.target.value)}
                                    placeholder="e.g., I need to build a fast QA system over a large corpus of medical PDFs for production use..."
                                    className="w-full bg-[rgba(0,0,0,0.3)] border border-[#ffffff14] text-white rounded-xl px-4 py-3 h-32 resize-none focus:outline-none focus:border-emerald-500/50 transition-colors"
                                />
                            </div>

                            {/* Hardware constraints mock toggles */}
                            <div className="space-y-3">
                                <label className="text-sm font-medium text-slate-400 block">Infrastructure Constraints</label>
                                <div className="flex gap-4">
                                    <label className="flex items-center gap-2 cursor-pointer bg-white/5 p-3 rounded-lg border border-white/5 flex-1 justify-center hover:bg-white/10 transition">
                                        <input type="checkbox" defaultChecked className="accent-emerald-500 w-4 h-4" />
                                        <span className="text-sm flex items-center gap-2"><Server className="w-4 h-4 text-emerald-400" /> GPU Available</span>
                                    </label>
                                    <label className="flex items-center gap-2 cursor-pointer bg-white/5 p-3 rounded-lg border border-white/5 flex-1 justify-center hover:bg-white/10 transition">
                                        <input type="checkbox" defaultChecked className="accent-emerald-500 w-4 h-4" />
                                        <span className="text-sm">Latency Sensitive</span>
                                    </label>
                                </div>
                            </div>

                            <Button
                                type="submit"
                                disabled={isThinking || !task.trim()}
                                className="w-full py-4 text-lg bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 border-none shadow-[0_0_20px_rgba(16,185,129,0.2)]"
                            >
                                {isThinking ? (
                                    <span className="flex items-center gap-2">
                                        <Bot className="w-5 h-5 animate-pulse" /> Agent is analyzing...
                                    </span>
                                ) : (
                                    'Get Recommendations'
                                )}
                            </Button>
                        </form>
                    </GlassCard>
                </div>

                {/* Results Panel */}
                <div className="space-y-6">
                    <AnimatePresence mode="wait">
                        {!recommendation && !isThinking && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="h-full flex items-center justify-center p-8 bg-[rgba(15,17,26,0.3)] border border-dashed border-[#ffffff14] rounded-2xl"
                            >
                                <div className="text-center text-slate-500 space-y-4 max-w-sm mx-auto">
                                    <div className="w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center mx-auto ring-1 ring-white/5">
                                        <Bot className="w-10 h-10 opacity-50" />
                                    </div>
                                    <p>Describe your task and requirements, and the RankifyAgent will recommend the optimal retrieval pipeline configuration.</p>
                                </div>
                            </motion.div>
                        )}

                        {isThinking && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="h-full flex flex-col items-center justify-center gap-6 p-8"
                            >
                                <div className="relative">
                                    <div className="absolute inset-0 bg-emerald-500/20 blur-2xl rounded-full animate-pulse"></div>
                                    <Terminal className="w-16 h-16 text-emerald-400 relative z-10" />
                                </div>
                                <div className="space-y-2 text-center">
                                    <p className="text-emerald-400 font-mono tracking-wider">ANALYZING REQUIREMENTS...</p>
                                    <p className="text-slate-500 text-sm">Evaluating 40+ models against constraints</p>
                                </div>
                            </motion.div>
                        )}

                        {recommendation && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="space-y-4"
                            >
                                <GlassCard className="border-emerald-500/20 shadow-[0_0_30px_rgba(16,185,129,0.1)]">
                                    <h3 className="text-xl font-bold mb-6 text-emerald-400 flex items-center justify-between border-b border-[#ffffff14] pb-4">
                                        Recommendation Ready <Badge color="green">Success</Badge>
                                    </h3>

                                    <div className="space-y-6">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="bg-black/20 p-4 rounded-xl border border-white/5 space-y-2">
                                                <span className="text-xs text-slate-500 font-bold uppercase tracking-wider">Retriever</span>
                                                <div className="text-lg font-semibold text-white">{recommendation.retriever.name}</div>
                                                <p className="text-sm text-slate-400 leading-relaxed">{recommendation.retriever.reason}</p>
                                            </div>
                                            <div className="bg-black/20 p-4 rounded-xl border border-white/5 space-y-2">
                                                <span className="text-xs text-slate-500 font-bold uppercase tracking-wider">Reranker</span>
                                                <div className="text-lg font-semibold text-white">{recommendation.reranker.name}</div>
                                                <p className="text-sm text-slate-400 leading-relaxed">{recommendation.reranker.reason}</p>
                                            </div>
                                        </div>

                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-slate-400 text-sm">
                                                <span className="flex items-center gap-2"><Code className="w-4 h-4" /> Implementation</span>
                                                <button className="hover:text-white transition-colors">Copy Code</button>
                                            </div>
                                            <div className="bg-[#0f111a] p-4 rounded-xl border border-white/10 overflow-x-auto relative group">
                                                <div className="absolute top-2 right-2 flex gap-1.5 hidden group-hover:flex">
                                                    <div className="w-3 h-3 rounded-full bg-slate-600"></div>
                                                    <div className="w-3 h-3 rounded-full bg-slate-600"></div>
                                                    <div className="w-3 h-3 rounded-full bg-slate-600"></div>
                                                </div>
                                                <pre className="text-emerald-300 font-mono text-sm leading-relaxed">
                                                    <code>{recommendation.code}</code>
                                                </pre>
                                            </div>
                                        </div>
                                    </div>
                                </GlassCard>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
