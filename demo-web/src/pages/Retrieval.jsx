import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Database, Settings2, Loader2, FileText, CheckCircle2 } from 'lucide-react';
import { GlassCard, Button, Badge } from '../components/ui';

const mockResults = [
    { id: '1', score: 0.95, text: 'Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models.', has_answer: true },
    { id: '2', score: 0.88, text: 'Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans.', has_answer: false },
    { id: '3', score: 0.76, text: 'Artificial Neural Networks form the core of deep learning algorithms and are inspired by the human brain.', has_answer: false },
    { id: '4', score: 0.62, text: 'Data science combines math and statistics to extract meaningful insights from data.', has_answer: false },
];

export function Retrieval() {
    const [query, setQuery] = useState('');
    const [retriever, setRetriever] = useState('bge');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);

    const handleSearch = (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        // Simulate API call
        setTimeout(() => {
            setResults(mockResults);
            setLoading(false);
        }, 1500);
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="space-y-2">
                <h1 className="text-4xl font-bold flex items-center gap-3">
                    <Search className="w-10 h-10 text-blue-400" />
                    Document Retrieval
                </h1>
                <p className="text-slate-400 text-lg">
                    Query massive document corpora using state-of-the-art sparse and dense retrieval models.
                </p>
            </header>

            <div className="grid lg:grid-cols-3 gap-8">
                {/* Controls Panel */}
                <div className="lg:col-span-1 space-y-6">
                    <GlassCard className="space-y-6">
                        <h3 className="text-xl font-semibold flex items-center gap-2 border-b border-[#ffffff14] pb-4">
                            <Settings2 className="w-5 h-5 text-indigo-400" /> Configuration
                        </h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2 flex items-center gap-2">
                                    <Database className="w-4 h-4" /> Select Dataset
                                </label>
                                <select className="w-full bg-[rgba(0,0,0,0.3)] border border-[#ffffff14] text-white rounded-lg px-4 py-3 focus:outline-none focus:border-indigo-500 transition-colors cursor-pointer appearance-none">
                                    <option>Wikipedia (10k subset)</option>
                                    <option>MS MARCO</option>
                                    <option>Natural Questions</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2">
                                    Retriever Model
                                </label>
                                <div className="space-y-2">
                                    {[
                                        { id: 'bm25', name: 'BM25 (Sparse)' },
                                        { id: 'bge', name: 'BGE-M3 (Dense)' },
                                        { id: 'colbert', name: 'ColBERT (Late Inter.)' }
                                    ].map(model => (
                                        <label key={model.id} className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${retriever === model.id ? 'bg-indigo-500/10 border-indigo-500 text-white' : 'border-[#ffffff14] text-slate-400 hover:bg-white/5'}`}>
                                            <input
                                                type="radio"
                                                name="retriever"
                                                value={model.id}
                                                checked={retriever === model.id}
                                                onChange={(e) => setRetriever(e.target.value)}
                                                className="hidden"
                                            />
                                            <div className={`w-4 h-4 rounded-full border flex items-center justify-center ${retriever === model.id ? 'border-indigo-500' : 'border-slate-500'}`}>
                                                {retriever === model.id && <div className="w-2 h-2 rounded-full bg-indigo-500" />}
                                            </div>
                                            <span className="font-medium">{model.name}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </GlassCard>
                </div>

                {/* Search Results Panel */}
                <div className="lg:col-span-2 space-y-6">
                    <GlassCard>
                        <form onSubmit={handleSearch} className="flex gap-4 relative">
                            <div className="relative flex-1">
                                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Enter your natural language query..."
                                    className="w-full bg-[rgba(0,0,0,0.2)] border border-[#ffffff14] text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-indigo-500 transition-colors shadow-inner"
                                />
                            </div>
                            <Button type="submit" className="px-8 whitespace-nowrap" disabled={loading || !query}>
                                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Retrieve Data'}
                            </Button>
                        </form>
                    </GlassCard>

                    {/* Results Area */}
                    <div className="space-y-4">
                        {!results && !loading && (
                            <div className="flex flex-col items-center justify-center py-20 text-slate-500 border border-dashed border-[#ffffff14] rounded-2xl bg-[rgba(15,17,26,0.3)]">
                                <Search className="w-12 h-12 mb-4 opacity-50 block" />
                                <p>Configure settings and run a search to view retrieved contexts.</p>
                            </div>
                        )}

                        {loading && (
                            <div className="flex flex-col items-center justify-center py-20 space-y-4">
                                <div className="relative">
                                    <div className="w-16 h-16 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin"></div>
                                    <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full"></div>
                                </div>
                                <p className="text-indigo-400 font-medium animate-pulse text-lg">Scanning vector index...</p>
                            </div>
                        )}

                        {results && !loading && (
                            <motion.div
                                className="space-y-4 pt-4"
                                initial="hidden"
                                animate="visible"
                                variants={{
                                    hidden: { opacity: 0 },
                                    visible: {
                                        opacity: 1,
                                        transition: { staggerChildren: 0.1 }
                                    }
                                }}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="text-lg font-medium text-slate-300">Top {results.length} Contexts</h3>
                                    <Badge color="green">Latency: 142ms</Badge>
                                </div>

                                {results.map((result, idx) => (
                                    <motion.div
                                        key={result.id}
                                        variants={{
                                            hidden: { opacity: 0, y: 20 },
                                            visible: { opacity: 1, y: 0 }
                                        }}
                                    >
                                        <GlassCard hover className="p-5 flex gap-4 border-l-4 border-l-transparent hover:border-l-indigo-500 transition-all duration-300">
                                            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex flex-col items-center justify-center shrink-0">
                                                <span className="text-xs text-indigo-400 font-bold uppercase">Rank</span>
                                                <span className="text-xl font-black text-indigo-300">#{idx + 1}</span>
                                            </div>

                                            <div className="flex-1 space-y-3">
                                                <p className="text-slate-200 leading-relaxed group-hover:text-white transition-colors">{result.text}</p>

                                                <div className="flex items-center gap-4 text-sm mt-4 pt-3 border-t border-[#ffffff0a]">
                                                    <div className="flex items-center gap-2 w-32">
                                                        <span className="text-slate-400 text-xs font-medium uppercase min-w-[40px]">Score</span>
                                                        <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                                                            <div
                                                                className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
                                                                style={{ width: `${result.score * 100}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-indigo-300 font-mono text-xs">{result.score.toFixed(2)}</span>
                                                    </div>

                                                    {result.has_answer && (
                                                        <span className="flex items-center gap-1.5 text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded text-xs font-medium border border-emerald-400/20">
                                                            <CheckCircle2 className="w-3.5 h-3.5" /> Contains Answer
                                                        </span>
                                                    )}

                                                    <span className="flex items-center gap-1.5 text-slate-500 text-xs ml-auto">
                                                        <FileText className="w-3.5 h-3.5" /> doc_id: {result.id}
                                                    </span>
                                                </div>
                                            </div>
                                        </GlassCard>
                                    </motion.div>
                                ))}
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
