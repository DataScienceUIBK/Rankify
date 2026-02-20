import React from 'react';
import { motion } from 'framer-motion';
import { Search, Zap, Layers, Cpu, ArrowRight } from 'lucide-react';
import { GlassCard, Button } from '../components/ui';
import { useNavigate } from 'react-router-dom';

const features = [
    {
        icon: Search,
        title: 'Advanced Retrieval',
        desc: 'Support for BM25, exact entity matching, and dense representations like DPR, ColBERT, and BGE.',
        color: 'text-blue-400',
        bg: 'bg-blue-400/10'
    },
    {
        icon: Layers,
        title: 'State-of-the-Art Re-Ranking',
        desc: 'Out-of-the-box support for 24+ re-rankers including FlashRank, RankZephyr, BGE-Reranker, and MonoT5.',
        color: 'text-purple-400',
        bg: 'bg-purple-400/10'
    },
    {
        icon: Zap,
        title: 'RAG Endpoints',
        desc: 'Flexible generator architecture natively supporting OpenAI, vLLM, and LiteLLM backends.',
        color: 'text-pink-400',
        bg: 'bg-pink-400/10'
    }
];

export function Home() {
    const navigate = useNavigate();

    return (
        <div className="space-y-16 py-8">
            {/* Hero Section */}
            <section className="text-center space-y-8 max-w-4xl mx-auto mt-12">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 text-sm font-medium"
                >
                    <span className="flex h-2 w-2 rounded-full bg-indigo-400 animate-pulse"></span>
                    Now available for ACL Demo
                </motion.div>

                <motion.h1
                    className="text-6xl font-extrabold tracking-tight"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.1 }}
                >
                    The Ultimate Toolkit for{' '}
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
                        Retrieval & RAG
                    </span>
                </motion.h1>

                <motion.p
                    className="text-xl text-slate-400 leading-relaxed max-w-2xl mx-auto"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                >
                    A modular, efficient, and unified framework designed to push the boundaries of retrieval, re-ranking, and retrieval-augmented generation tasks.
                </motion.p>

                <motion.div
                    className="flex items-center justify-center gap-4 pt-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 }}
                >
                    <Button
                        className="px-8 py-4 text-lg rounded-xl shadow-[0_0_30px_rgba(99,102,241,0.3)] hover:shadow-[0_0_40px_rgba(99,102,241,0.5)] bg-gradient-to-r from-indigo-600 to-purple-600"
                        onClick={() => navigate('/retrieval')}
                    >
                        Start Interactive Demo
                        <ArrowRight className="w-5 h-5 ml-2" />
                    </Button>
                    <Button
                        variant="secondary"
                        className="px-8 py-4 text-lg rounded-xl hover:bg-white/10"
                        onClick={() => window.open('https://rankify.readthedocs.io/', '_blank')}
                    >
                        Documentation
                    </Button>
                </motion.div>
            </section>

            {/* Capabilities Grid */}
            <section className="grid md:grid-cols-3 gap-6 pt-16">
                {features.map((feature, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.4 + idx * 0.1 }}
                    >
                        <GlassCard className="h-full p-8 border border-[#ffffff14]">
                            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-6 ${feature.bg}`}>
                                <feature.icon className={`w-7 h-7 ${feature.color}`} />
                            </div>
                            <h3 className="text-2xl font-semibold mb-3">{feature.title}</h3>
                            <p className="text-slate-400 leading-relaxed">{feature.desc}</p>
                        </GlassCard>
                    </motion.div>
                ))}
            </section>

            {/* Integration Showcase (Visual flair) */}
            <motion.section
                className="pt-16 pb-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1, delay: 0.8 }}
            >
                <div className="relative rounded-3xl overflow-hidden border border-[#ffffff14] bg-[rgba(15,17,26,0.6)] p-8">
                    <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 to-purple-500/10"></div>

                    <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
                        <div className="flex-1 space-y-4">
                            <div className="inline-flex items-center gap-2 text-indigo-400 font-semibold uppercase tracking-wider text-sm">
                                <Cpu className="w-4 h-4" /> Powering AI Agents
                            </div>
                            <h2 className="text-4xl font-bold">RankifyAgent Built-In</h2>
                            <p className="text-slate-400 text-lg max-w-xl">
                                Let AI help you choose the best models for your use case. Ask our Agent for recommendations based on task semantics and infrastructure specs.
                            </p>
                            <Button variant="secondary" onClick={() => navigate('/agent')} className="mt-4">
                                Try RankifyAgent <Bot className="w-4 h-4 ml-2" />
                            </Button>
                        </div>

                        <div className="w-full md:w-1/3 aspect-square rounded-2xl bg-[#0f111a] border border-[#ffffff14] shadow-2xl overflow-hidden flex items-center justify-center relative group">
                            <div className="absolute inset-0 bg-grid-slate-900/[0.04] bg-[size:20px_20px]"></div>
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-indigo-500/30 rounded-full blur-3xl group-hover:bg-indigo-500/50 transition-colors duration-500"></div>
                            <Bot className="w-20 h-20 text-indigo-400 relative z-10 filter drop-shadow-[0_0_15px_rgba(99,102,241,0.5)] group-hover:scale-110 transition-transform duration-500" />
                        </div>
                    </div>
                </div>
            </motion.section>
        </div>
    );
}
