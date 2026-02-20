import React, { useState } from 'react';
import { Card, Button, Input, Badge } from '../components/ui';
import { Search, Settings2, Database, Upload, Play, ListTree, Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react';

export function Console() {
    // Left Panel State (Sidebar)
    const [dataSource, setDataSource] = useState('msmarco'); // msmarco, wikipedia, custom
    const [customFile, setCustomFile] = useState(null);
    const [retriever, setRetriever] = useState('bge');
    const [reranker, setReranker] = useState('flashrank');
    const [generator, setGenerator] = useState('openai');
    const [query, setQuery] = useState('');

    // Main Area State
    const [isRunning, setIsRunning] = useState(false);
    const [pipelineState, setPipelineState] = useState('idle'); // idle, retrieving, reranking, generating, complete

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) setCustomFile(file);
    };

    const runPipeline = async () => {
        if (!query.trim()) return;

        setIsRunning(true);
        setPipelineState('retrieving');

        // Simulating the sequential pipeline
        setTimeout(() => {
            setPipelineState('reranking');

            setTimeout(() => {
                setPipelineState('generating');

                setTimeout(() => {
                    setPipelineState('complete');
                    setIsRunning(false);
                }, 2000);
            }, 1500);
        }, 1500);
    };

    return (
        <div className="flex h-[calc(100vh-56px)] w-full overflow-hidden bg-slate-50">

            {/* LEFT SIDEBAR: CONFIGURATION (Fixed Width) */}
            <aside className="w-[380px] flex-shrink-0 border-r border-slate-200 bg-white flex flex-col h-full z-10 shadow-[4px_0_24px_rgba(0,0,0,0.02)]">

                <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Settings2 className="w-5 h-5 text-slate-700" />
                        <h2 className="text-sm font-bold tracking-wide text-slate-900 uppercase">Engine Configuration</h2>
                    </div>
                    {pipelineState !== 'idle' && (
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-1 rounded-md border border-emerald-200">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                            </span>
                            Active
                        </div>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-5 pb-24 space-y-6 custom-scrollbar">

                    {/* Data Source Configuration */}
                    <div className="space-y-3">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">1. Data Source</h3>

                        <div className="flex rounded-md shadow-sm bg-slate-100 p-1">
                            {['msmarco', 'wikipedia', 'custom'].map((opt) => (
                                <button
                                    key={opt}
                                    onClick={() => setDataSource(opt)}
                                    className={`flex-1 py-1.5 px-2 text-xs font-semibold rounded transition-all ${dataSource === opt
                                            ? 'bg-white text-indigo-700 shadow border border-slate-200/60'
                                            : 'text-slate-500 hover:text-slate-700'
                                        }`}
                                >
                                    {opt === 'msmarco' ? 'MS MARCO' : opt === 'wikipedia' ? 'Wikipedia' : 'JSONL'}
                                </button>
                            ))}
                        </div>

                        {dataSource === 'custom' && (
                            <div className="mt-3 relative group">
                                <div className="absolute inset-0 bg-indigo-50/50 border border-indigo-200 border-dashed rounded-lg pointer-events-none transition-colors group-hover:bg-indigo-50"></div>
                                <label htmlFor="file-upload" className="relative flex flex-col items-center justify-center w-full py-6 cursor-pointer">
                                    <div className="flex flex-col items-center justify-center pt-2 pb-3">
                                        <Upload className="w-6 h-6 mb-2 text-indigo-500" />
                                        <p className="text-sm font-medium text-slate-700">
                                            {customFile ? customFile.name : 'Upload custom corpus'}
                                        </p>
                                        <p className="text-xs text-slate-500 mt-1">.jsonl files only</p>
                                    </div>
                                    <input id="file-upload" type="file" accept=".jsonl" className="hidden" onChange={handleFileUpload} />
                                </label>
                            </div>
                        )}
                        {dataSource !== 'custom' && (
                            <div className="mt-2 text-xs text-slate-500 bg-slate-50 border border-slate-100 rounded p-2 flex items-start gap-2">
                                <Database className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                                <span>Using pre-indexed standard benchmarking dataset.</span>
                            </div>
                        )}
                    </div>

                    <div className="h-px w-full bg-slate-100"></div>

                    {/* Model Configuration */}
                    <div className="space-y-4">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">2. Pipeline Architecture</h3>

                        <div className="space-y-1.5 focus-within:text-indigo-600 transition-colors">
                            <label className="text-xs font-semibold uppercase tracking-wider flex items-center gap-2">
                                <Search className="w-3.5 h-3.5" /> Primary Retriever
                            </label>
                            <select
                                value={retriever}
                                onChange={(e) => setRetriever(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-white p-2.5 text-slate-700 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-sm transition-all"
                            >
                                <optgroup label="Sparse Resolvers">
                                    <option value="bm25">BM25 (Standard)</option>
                                    <option value="splade">SPLADE (Neutral)</option>
                                </optgroup>
                                <optgroup label="Dense Encoders">
                                    <option value="bge">BGE-large-en-v1.5</option>
                                    <option value="e5">E5-large-v2</option>
                                    <option value="mxbai">mxbai-embed-large-v1</option>
                                </optgroup>
                                <optgroup label="Late Interaction">
                                    <option value="colbert">ColBERTv2</option>
                                </optgroup>
                            </select>
                        </div>

                        <div className="space-y-1.5 focus-within:text-indigo-600 transition-colors">
                            <label className="text-xs font-semibold uppercase tracking-wider flex items-center gap-2">
                                <ListTree className="w-3.5 h-3.5" /> Cross-Encoder Reranker
                            </label>
                            <select
                                value={reranker}
                                onChange={(e) => setReranker(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-white p-2.5 text-slate-700 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-sm transition-all"
                            >
                                <option value="none">-- Skip Reranking --</option>
                                <optgroup label="Dedicated Rerankers">
                                    <option value="flashrank">FlashRank (Fast)</option>
                                    <option value="bge-reranker">BGE Reranker Large</option>
                                    <option value="jina">Jina Reranker v1 Base</option>
                                </optgroup>
                                <optgroup label="Seq2Seq Models">
                                    <option value="monot5">MonoT5 (Document Relevance)</option>
                                    <option value="rankzephyr">RankZephyr (Zero-Shot)</option>
                                </optgroup>
                                <optgroup label="Commercial API">
                                    <option value="cohere">Cohere Rerank 3</option>
                                </optgroup>
                            </select>
                        </div>

                        <div className="space-y-1.5 focus-within:text-purple-600 transition-colors">
                            <label className="text-xs font-semibold uppercase tracking-wider flex items-center gap-2">
                                <Sparkles className="w-3.5 h-3.5" /> Generation Engine
                            </label>
                            <select
                                value={generator}
                                onChange={(e) => setGenerator(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-white p-2.5 text-slate-700 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 shadow-sm transition-all"
                            >
                                <optgroup label="Commercial APIs">
                                    <option value="openai">OpenAI (gpt-4o)</option>
                                    <option value="anthropic">Anthropic (claude-3-5-sonnet)</option>
                                    <option value="google">Google (gemini-1.5-pro)</option>
                                </optgroup>
                                <optgroup label="Open Weights (vLLM)">
                                    <option value="vllm_llama3">Meta Llama-3-8B-Instruct</option>
                                    <option value="vllm_mistral">Mistral-v0.3-Instruct</option>
                                    <option value="vllm_qwen">Qwen2-7B-Instruct</option>
                                </optgroup>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Query & Execution (Sticky Bottom) */}
                <div className="absolute bottom-0 left-0 w-[380px] bg-white border-t border-slate-200 p-5 shadow-[0_-10px_20px_rgba(0,0,0,0.03)] z-20">
                    <div className="space-y-3">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">3. Execution Query</label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Enter the search query or prompt to process across the pipeline..."
                            className="w-full h-24 text-sm border border-slate-300 rounded-md p-3 text-slate-900 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 resize-none shadow-inner"
                        />
                        <button
                            onClick={runPipeline}
                            disabled={isRunning || !query.trim()}
                            className="w-full bg-slate-900 hover:bg-indigo-600 text-white font-medium py-3 px-4 rounded-md flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-slate-900/10"
                        >
                            {isRunning ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Processing Step: {pipelineState.toUpperCase()}
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4 fill-current" />
                                    Execute Pipeline
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </aside>

            {/* MAIN CANVAS: RESULTS & VIEWPORT */}
            <main className="flex-1 flex flex-col min-w-0 bg-slate-50/50">

                {/* Top Half: Context Engineering (Retrieval/Reranking) */}
                <div className="flex-1 flex flex-col border-b border-slate-200 min-h-[50%] bg-white m-4 mb-2 rounded-xl shadow-sm border overflow-hidden">
                    <div className="bg-slate-50 border-b border-slate-100 px-5 py-3 flex items-center justify-between shrink-0">
                        <div className="flex items-center gap-3">
                            <Database className="w-4 h-4 text-slate-500" />
                            <h2 className="text-sm font-semibold text-slate-800">Context Pool (Engine Results)</h2>
                        </div>
                        <div className="flex items-center gap-2">
                            {pipelineState === 'retrieving' && <Badge variant="blue" className="animate-pulse">Retrieving from Dataset...</Badge>}
                            {pipelineState === 'reranking' && <Badge variant="purple" className="animate-pulse">Applying Cross-Encoder...</Badge>}
                            {(pipelineState === 'generating' || pipelineState === 'complete') && (
                                <Badge variant="green" className="border-emerald-200 bg-emerald-50 text-emerald-700">
                                    <CheckCircle2 className="w-3 h-3 mr-1" /> Context Locked
                                </Badge>
                            )}
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-0 bg-slate-50/30">
                        {pipelineState === 'idle' ? (
                            <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-3">
                                <ListTree className="w-10 h-10 text-slate-300" />
                                <p className="text-sm">Context pool empty. Run the pipeline to visualize document retrieval and reranking delta.</p>
                            </div>
                        ) : (
                            <div className="w-full">
                                {/* Simulated Document View - Wide Layout */}
                                <table className="w-full text-left border-collapse">
                                    <thead>
                                        <tr className="border-b border-slate-200 bg-white sticky top-0 z-10 shadow-sm">
                                            <th className="py-3 px-5 text-xs font-semibold text-slate-500 uppercase tracking-wider w-16">Rank</th>
                                            <th className="py-3 px-5 text-xs font-semibold text-slate-500 uppercase tracking-wider">Document Snippet</th>
                                            <th className="py-3 px-5 text-xs font-semibold text-slate-500 uppercase tracking-wider w-24 text-right">Ret. Score</th>
                                            <th className="py-3 px-5 text-xs font-semibold text-slate-500 uppercase tracking-wider w-24 text-right">Rrk. Score</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100">
                                        {[1, 2, 3, 4, 5].map((i) => (
                                            <tr key={i} className={`hover:bg-slate-50 transition-colors ${pipelineState === 'retrieving' ? 'opacity-50' : ''}`}>
                                                <td className="py-4 px-5 align-top">
                                                    <div className="w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-xs font-bold text-slate-600">
                                                        {i}
                                                    </div>
                                                </td>
                                                <td className="py-4 px-5">
                                                    <div className="space-y-2">
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">Doc ID: MS-{45920 + i}</span>
                                                        </div>
                                                        <p className="text-sm text-slate-700 leading-relaxed font-sans line-clamp-2">
                                                            This is a simulated retrieved document snippet demonstrating the wide view layout. In a real scenario, this would contain the actual text retrieved from MS MARCO or the uploaded JSONL file matching the given query input, allowing researchers to rapidly inspect context relevance over long paragraphs.
                                                        </p>
                                                    </div>
                                                </td>
                                                <td className="py-4 px-5 align-top text-right">
                                                    <span className="text-sm font-mono text-slate-600">{((6 - i) * 0.15 + 0.1).toFixed(3)}</span>
                                                </td>
                                                <td className="py-4 px-5 align-top text-right">
                                                    <span className="text-sm font-mono font-medium text-emerald-600">
                                                        {['generating', 'complete'].includes(pipelineState) ? ((6 - i) * 0.18 + 0.05).toFixed(3) : '-'}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>

                {/* Bottom Half: Generation Console */}
                <div className="flex-1 flex flex-col border-t border-slate-200 m-4 mt-2 rounded-xl shadow-sm border overflow-hidden bg-[#FAFAFA]">
                    <div className="bg-white border-b border-slate-200 px-5 py-3 flex items-center justify-between shrink-0">
                        <div className="flex items-center gap-3">
                            <Sparkles className="w-4 h-4 text-purple-600" />
                            <h2 className="text-sm font-semibold text-slate-800">Generation Console</h2>
                        </div>
                        {pipelineState === 'generating' && (
                            <span className="flex items-center gap-2 text-xs font-medium text-purple-600">
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                                </span>
                                LLM Inference...
                            </span>
                        )}
                    </div>

                    <div className="flex-1 overflow-y-auto p-6">
                        {['idle', 'retrieving', 'reranking'].includes(pipelineState) ? (
                            <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-3">
                                <AlertCircle className="w-8 h-8 text-slate-200" />
                                <p className="text-sm">Awaiting context assembly before LLM generation.</p>
                            </div>
                        ) : pipelineState === 'generating' ? (
                            <div className="space-y-4 max-w-4xl">
                                <div className="flex items-center gap-2 text-slate-500 font-medium mb-4 text-sm">
                                    Initializing generation stream connecting to backend...
                                </div>
                                <div className="space-y-3 animate-pulse">
                                    <div className="h-4 bg-slate-200 rounded w-full"></div>
                                    <div className="h-4 bg-slate-200 rounded w-11/12"></div>
                                    <div className="h-4 bg-slate-200 rounded w-4/5"></div>
                                    <div className="h-4 bg-slate-200 rounded w-full"></div>
                                </div>
                            </div>
                        ) : (
                            <div className="max-w-5xl prose prose-slate">
                                <p className="text-slate-800 leading-relaxed text-[15px]">
                                    <strong className="font-semibold text-slate-900 bg-indigo-50 px-2 py-0.5 rounded text-sm mr-2">System Output:</strong>
                                    Based on the retrieved and reranked context, cross-encoders significantly improve retrieval augmented generation by providing a deeper, interaction-based relevance assessment between the query and the documents. Unlike bi-encoders used in initial retrieval, cross-encoders process the query and document simultaneously through the transformer layers, leading to much higher accuracy and better grounding for the final LLM generation phase.
                                </p>

                                <div className="mt-8 p-4 bg-white border border-slate-200 rounded-lg shadow-sm">
                                    <h4 className="text-xs font-bold uppercase text-slate-500 tracking-wider mb-3 flex items-center gap-2">
                                        <Database className="w-3.5 h-3.5" /> Cited Sources
                                    </h4>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <div className="text-sm text-slate-600 bg-slate-50 border border-slate-100 p-2 rounded truncate transition-colors hover:border-indigo-200 hover:bg-indigo-50/30 cursor-pointer">
                                            <span className="font-medium text-indigo-600 mr-2">[1]</span>
                                            Understanding Cross-Encoders vs Bi-Encoders in Search
                                        </div>
                                        <div className="text-sm text-slate-600 bg-slate-50 border border-slate-100 p-2 rounded truncate transition-colors hover:border-indigo-200 hover:bg-indigo-50/30 cursor-pointer">
                                            <span className="font-medium text-indigo-600 mr-2">[2]</span>
                                            The Impact of Context Quality on LLM Hallucinations
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

            </main>
        </div>
    );
}

