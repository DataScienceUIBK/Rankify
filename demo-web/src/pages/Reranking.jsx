import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ListOrdered, ArrowDownUp, RefreshCw, Layers } from 'lucide-react';
import { GlassCard, Button, Badge } from '../components/ui';

// Unordered initial state
const initialDocs = [
    { id: '1', score: 0.62, text: 'Data science combines math and statistics to extract meaningful insights from data.' },
    { id: '2', score: 0.88, text: 'Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans.' },
    { id: '3', score: 0.76, text: 'Artificial Neural Networks form the core of deep learning algorithms and are inspired by the human brain.' },
    { id: '4', score: 0.95, text: 'Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models.' },
];

export function Reranking() {
    const [docs, setDocs] = useState(initialDocs);
    const [reranker, setReranker] = useState('flashrank');
    const [isReranking, setIsReranking] = useState(false);
    const [isDone, setIsDone] = useState(false);

    const handleRerank = () => {
        setIsReranking(true);
        // Simulate reranking delay
        setTimeout(() => {
            const sortedDocs = [...docs].sort((a, b) => b.score - a.score);
            setDocs(sortedDocs);
            setIsReranking(false);
            setIsDone(true);
        }, 2000);
    };

    const handleReset = () => {
        setDocs(initialDocs);
        setIsDone(false);
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="space-y-2">
                <h1 className="text-4xl font-bold flex items-center gap-3">
                    <ListOrdered className="w-10 h-10 text-purple-400" />
                    Cross-Encoder Re-Ranking
                </h1>
                <p className="text-slate-400 text-lg">
                    Refine and re-order initial retrieval results using powerful cross-encoder models.
                </p>
            </header>

            <div className="grid lg:grid-cols-3 gap-8">
                <div className="lg:col-span-1 space-y-6">
                    <GlassCard className="space-y-6">
                        <h3 className="text-xl font-semibold flex items-center gap-2 border-b border-[#ffffff14] pb-4">
                            <Layers className="w-5 h-5 text-purple-400" /> Reranker Settings
                        </h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-3">
                                    Select Reranking Model
                                </label>
                                <div className="space-y-2">
                                    {[
                                        { id: 'flashrank', name: 'FlashRank', desc: 'Ultra-fast, CPU optimized' },
                                        { id: 'rankzephyr', name: 'RankZephyr', desc: 'LLM-based listwise ranking' },
                                        { id: 'monot5', name: 'MonoT5', desc: 'Sequence-to-sequence ranking' }
                                    ].map(model => (
                                        <label key={model.id} className={`flex flex-col p-3 rounded-lg border cursor-pointer transition-all ${reranker === model.id ? 'bg-purple-500/10 border-purple-500' : 'border-[#ffffff14] hover:bg-white/5'}`}>
                                            <div className="flex items-center gap-3">
                                                <input
                                                    type="radio"
                                                    name="reranker"
                                                    value={model.id}
                                                    checked={reranker === model.id}
                                                    onChange={(e) => setReranker(e.target.value)}
                                                    className="hidden"
                                                />
                                                <div className={`w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${reranker === model.id ? 'border-purple-500' : 'border-slate-500'}`}>
                                                    {reranker === model.id && <div className="w-2 h-2 rounded-full bg-purple-500" />}
                                                </div>
                                                <span className={`font-medium ${reranker === model.id ? 'text-white' : 'text-slate-300'}`}>{model.name}</span>
                                            </div>
                                            <span className="text-xs text-slate-500 ml-7 mt-1">{model.desc}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            <div className="pt-4 border-t border-[#ffffff14] flex gap-3">
                                <Button
                                    onClick={handleRerank}
                                    disabled={isReranking || isDone}
                                    className="flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500"
                                >
                                    {isReranking ? (
                                        <RefreshCw className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <>Run Reranker <ArrowDownUp className="w-4 h-4 ml-2" /></>
                                    )}
                                </Button>
                                {isDone && (
                                    <Button variant="secondary" onClick={handleReset} className="px-4">
                                        Reset
                                    </Button>
                                )}
                            </div>
                        </div>
                    </GlassCard>
                </div>

                <div className="lg:col-span-2">
                    <GlassCard className="min-h-[500px]">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-medium text-slate-300">
                                {isDone ? 'Re-Ranked Documents' : 'Initial Retrieved Documents (Unordered)'}
                            </h3>
                            {isDone && <Badge color="purple">Processed in 843ms</Badge>}
                        </div>

                        <div className="space-y-3 relative">
                            {/* Overlay during reranking to show processing state */}
                            <AnimatePresence>
                                {isReranking && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="absolute inset-0 z-10 bg-[rgba(15,17,26,0.7)] backdrop-blur-sm rounded-xl flex flex-col items-center justify-center border border-purple-500/30"
                                    >
                                        <div className="relative">
                                            <ArrowDownUp className="w-12 h-12 text-purple-400 animate-bounce" />
                                            <div className="absolute inset-0 bg-purple-500/20 blur-xl rounded-full"></div>
                                        </div>
                                        <p className="mt-4 text-purple-300 font-medium">Applying cross-attention interactions...</p>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {docs.map((doc, idx) => (
                                <motion.div
                                    key={doc.id}
                                    layout
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{
                                        type: "spring",
                                        stiffness: 300,
                                        damping: 24,
                                        delay: isDone ? idx * 0.1 : 0
                                    }}
                                    className={`p-4 rounded-xl border transition-colors ${isDone ? 'bg-purple-500/5 border-purple-500/20' : 'bg-[#ffffff05] border-[#ffffff14]'}`}
                                >
                                    <div className="flex gap-4">
                                        <div className="w-8 h-8 rounded-lg bg-[rgba(0,0,0,0.3)] flex items-center justify-center text-slate-400 font-mono text-sm shrink-0">
                                            {isDone ? idx + 1 : '-'}
                                        </div>
                                        <div>
                                            <p className="text-slate-300 leading-relaxed mb-2">{doc.text}</p>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-mono px-2 py-1 rounded bg-[rgba(0,0,0,0.4)] text-slate-400">
                                                    ID: {doc.id}
                                                </span>
                                                {isDone && (
                                                    <motion.span
                                                        initial={{ opacity: 0, x: -10 }}
                                                        animate={{ opacity: 1, x: 0 }}
                                                        className="text-xs font-mono px-2 py-1 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                                    >
                                                        New Score: {(doc.score + (0.05 * Math.random())).toFixed(4)}
                                                    </motion.span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </GlassCard>
                </div>
            </div>
        </div>
    );
}
