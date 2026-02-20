import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Database, ListTree, Cpu, ArrowRight } from 'lucide-react';

export function Home() {
    const navigate = useNavigate();

    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] w-full max-w-5xl mx-auto text-center px-4">
            {/* Header Badge */}
            <div className="mb-6 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-sm font-semibold text-slate-700 tracking-wide uppercase">
                <Database className="w-4 h-4" />
                AI FOR RESEARCH
            </div>

            {/* Main Title */}
            <h1 className="text-5xl md:text-7xl font-extrabold text-slate-900 tracking-tight leading-tight mb-6">
                Retrieval-Augmented <br className="hidden md:block" /> Generation Pipeline
            </h1>

            <p className="text-lg md:text-xl text-slate-500 max-w-2xl mx-auto mb-16 font-light">
                Configure, execute, and evaluate complete RAG pipelines with state-of-the-art retrievers and cross-encoder rerankers.
            </p>

            {/* Feature Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full mb-16">
                {/* Retrieve Card */}
                <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm text-left relative overflow-hidden group hover:shadow-md transition-shadow">
                    <div className="absolute top-6 right-6 text-5xl font-black text-slate-100 group-hover:text-slate-200 transition-colors pointer-events-none select-none">
                        01
                    </div>
                    <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center mb-6">
                        <Database className="w-5 h-5 text-slate-700" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">Retrieve</h3>
                    <p className="text-sm text-slate-500 leading-relaxed">
                        Start by selecting pre-indexed datasets like MS MARCO or Wikipedia, or upload your own custom JSONL corpus.
                    </p>
                </div>

                {/* Rerank Card */}
                <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm text-left relative overflow-hidden group hover:shadow-md transition-shadow">
                    <div className="absolute top-6 right-6 text-5xl font-black text-slate-100 group-hover:text-slate-200 transition-colors pointer-events-none select-none">
                        02
                    </div>
                    <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center mb-6">
                        <ListTree className="w-5 h-5 text-slate-700" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">Rerank</h3>
                    <p className="text-sm text-slate-500 leading-relaxed">
                        Use advanced cross-encoder models like FlashRank or MonoT5 to drastically improve context relevance.
                    </p>
                </div>

                {/* Generate Card */}
                <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm text-left relative overflow-hidden group hover:shadow-md transition-shadow">
                    <div className="absolute top-6 right-6 text-5xl font-black text-slate-100 group-hover:text-slate-200 transition-colors pointer-events-none select-none">
                        03
                    </div>
                    <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center mb-6">
                        <Cpu className="w-5 h-5 text-slate-700" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">Generate</h3>
                    <p className="text-sm text-slate-500 leading-relaxed">
                        Generate grounded answers using state-of-the-art LLMs, leveraging only the most highly-ranked context.
                    </p>
                </div>
            </div>

            {/* Action Area */}
            <div className="flex flex-col items-center gap-4">
                <span className="text-sm font-medium text-slate-500">
                    Ready to evaluate?
                </span>
                <button
                    onClick={() => navigate('/console')}
                    className="group inline-flex items-center justify-center gap-2 bg-slate-900 text-white font-semibold text-lg px-8 py-4 rounded-xl hover:bg-slate-800 transition-all focus-visible:ring-4 focus-visible:ring-slate-200 shadow-lg shadow-slate-200"
                >
                    Launch Console
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
            </div>
        </div>
    );
}
