import React, { useState } from 'react';
import { Card, Button, Input, Badge } from '../components/ui';
import { Search, Settings2, Database, Upload, Play, Server, ListTree, Sparkles, FileText, Bot } from 'lucide-react';

export function Console() {
    // Left Panel State (Setup)
    const [dataSource, setDataSource] = useState('msmarco'); // msmarco, wikipedia, custom
    const [customFile, setCustomFile] = useState(null);
    const [retriever, setRetriever] = useState('bge');
    const [reranker, setReranker] = useState('flashrank');
    const [generator, setGenerator] = useState('openai');
    const [query, setQuery] = useState('');

    // Middle & Right Panel State
    const [isRunning, setIsRunning] = useState(false);
    const [pipelineState, setPipelineState] = useState('idle'); // idle, retrieving, reranking, generating, complete
    const [results, setResults] = useState(null);

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) setCustomFile(file);
    };

    const runPipeline = async () => {
        if (!query.trim()) return;

        setIsRunning(true);
        setPipelineState('retrieving');

        // Simulating the sequential pipeline for now to demonstrate UI layout
        setTimeout(() => {
            setPipelineState('reranking');

            setTimeout(() => {
                setPipelineState('generating');

                setTimeout(() => {
                    setPipelineState('complete');
                    setIsRunning(false);
                }, 1500);
            }, 1200);
        }, 1200);
    };

    return (
        <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-120px)] w-full">

            {/* COLUMN 1: SETUP */}
            <div className="w-full lg:w-[350px] xl:w-[400px] flex flex-col gap-4 overflow-y-auto pr-2 pb-4">

                <Card className="flex flex-col gap-5 border-t-4 border-t-slate-800">
                    <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                        <Settings2 className="w-5 h-5 text-slate-700" />
                        <h2 className="text-sm font-bold tracking-wide text-slate-900 uppercase">1. Configuration</h2>
                    </div>

                    {/* Data Source */}
                    <div className="space-y-3">
                        <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Data Source</label>
                        <div className="grid grid-cols-3 gap-2">
                            {['msmarco', 'wikipedia', 'custom'].map((opt) => (
                                <button
                                    key={opt}
                                    onClick={() => setDataSource(opt)}
                                    className={`py-2 px-2 rounded-md text-xs font-medium border transition-all ${dataSource === opt
                                            ? 'bg-blue-50 border-blue-200 text-blue-700 shadow-sm'
                                            : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                                        }`}
                                >
                                    {opt === 'msmarco' ? 'MS MARCO' : opt === 'wikipedia' ? 'Wikipedia' : 'Upload JSONL'}
                                </button>
                            ))}
                        </div>

                        {dataSource === 'custom' && (
                            <div className="mt-3 p-4 border border-dashed border-slate-300 rounded-lg bg-slate-50 text-center flex flex-col items-center gap-2 transition-colors hover:border-blue-400">
                                <Upload className="w-6 h-6 text-slate-400" />
                                <span className="text-sm text-slate-600 font-medium">
                                    {customFile ? customFile.name : 'Upload .jsonl corpus'}
                                </span>
                                <input
                                    type="file"
                                    accept=".jsonl"
                                    className="hidden"
                                    id="file-upload"
                                    onChange={handleFileUpload}
                                />
                                <label htmlFor="file-upload" className="mt-1 text-xs text-blue-600 font-medium cursor-pointer hover:underline">
                                    {customFile ? 'Change File' : 'Browse Files'}
                                </label>
                            </div>
                        )}
                    </div>

                    {/* Models Setup */}
                    <div className="space-y-4 pt-2">
                        <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                                <Search className="w-3.5 h-3.5" /> Retriever Model
                            </label>
                            <select
                                value={retriever}
                                onChange={(e) => setRetriever(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-slate-50 p-2 text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            >
                                <option value="bm25">BM25 (Sparse)</option>
                                <option value="bge">BGE-large-en-v1.5 (Dense)</option>
                                <option value="colbert">ColBERTv2 (Late Interaction)</option>
                                <option value="splade">SPLADE (Sparse Dense)</option>
                            </select>
                        </div>

                        <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                                <ListTree className="w-3.5 h-3.5" /> Reranker Model
                            </label>
                            <select
                                value={reranker}
                                onChange={(e) => setReranker(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-slate-50 p-2 text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            >
                                <option value="flashrank">FlashRank</option>
                                <option value="monot5">MonoT5</option>
                                <option value="rankzephyr">RankZephyr</option>
                                <option value="jina">Jina Reranker</option>
                                <option value="cohere">Cohere Rerank</option>
                            </select>
                        </div>

                        <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                                <Bot className="w-3.5 h-3.5" /> Generator Model
                            </label>
                            <select
                                value={generator}
                                onChange={(e) => setGenerator(e.target.value)}
                                className="w-full text-sm border border-slate-200 rounded-md bg-slate-50 p-2 text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            >
                                <option value="openai">gpt-4o (OpenAI)</option>
                                <option value="anthropic">claude-3-5-sonnet (Anthropic)</option>
                                <option value="vllm_llama3">Llama-3-8B-Instruct (vLLM)</option>
                                <option value="vllm_mistral">Mistral-v0.3 (vLLM)</option>
                            </select>
                        </div>
                    </div>
                </Card>

                {/* Query Input */}
                <Card className="flex flex-col gap-4 border-t-4 border-t-blue-500">
                    <div className="flex flex-col gap-2">
                        <label className="text-sm font-bold tracking-wide text-slate-900 uppercase">2. Input Query</label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="e.g. What is the impact of cross-encoders on retrieval augmented generation?"
                            className="w-full h-32 text-sm border border-slate-200 rounded-lg p-3 text-slate-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none shadow-inner bg-slate-50/50"
                        />
                    </div>

                    <button
                        onClick={runPipeline}
                        disabled={isRunning || !query.trim()}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                    >
                        {isRunning ? (
                            <span className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Processing Pipeline...
                            </span>
                        ) : (
                            <>
                                <Play className="w-4 h-4 fill-current" />
                                Execute Pipeline
                            </>
                        )}
                    </button>
                </Card>

            </div>

            {/* COLUMN 2: CONTEXT POOL (RETRIEVAL & RERANKING) */}
            <Card className="flex-1 flex flex-col overflow-hidden !p-0 border-t-4 border-t-slate-300">
                <div className="bg-slate-50/80 border-b border-slate-200 p-4 shrink-0 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Database className="w-5 h-5 text-slate-500" />
                        <h2 className="text-sm font-bold tracking-wide text-slate-900 uppercase">3. Context Pool</h2>
                    </div>
                    {pipelineState !== 'idle' && (
                        <Badge variant={
                            pipelineState === 'retrieving' ? 'blue' :
                                pipelineState === 'reranking' ? 'purple' : 'green'
                        }>
                            {pipelineState === 'retrieving' ? 'Retrieving docs...' :
                                pipelineState === 'reranking' ? 'Reranking context...' : 'Context Locked'}
                        </Badge>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-4 bg-slate-50/30 flex flex-col items-center justify-center">
                    {/* Placeholder for when empty */}
                    {pipelineState === 'idle' && (
                        <div className="text-center text-slate-400 space-y-3">
                            <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-2">
                                <ListTree className="w-8 h-8 text-slate-300" />
                            </div>
                            <p className="text-sm font-medium">Pool is empty</p>
                            <p className="text-xs">Run the pipeline to see retrieved and reranked documents here.</p>
                        </div>
                    )}

                    {/* Placeholder for loading states - Will be replaced with actual document list delta UI later */}
                    {pipelineState !== 'idle' && (
                        <div className="w-full h-full flex flex-col gap-3">
                            {[1, 2, 3, 4, 5].map((i) => (
                                <div key={i} className={`p-4 rounded-lg border bg-white shadow-sm flex flex-col gap-2 transition-all duration-500
                                    ${pipelineState === 'retrieving' ? 'border-slate-200 opacity-60 animate-pulse' :
                                        pipelineState === 'reranking' ? 'border-purple-200 shadow-md' : 'border-slate-200'}
                                `}>
                                    <div className="flex justify-between items-center mb-1">
                                        <Badge variant="gray">Doc {i}</Badge>
                                        <div className="flex gap-2">
                                            {pipelineState === 'reranking' && <Badge variant="purple">Re-scoring...</Badge>}
                                            {(pipelineState === 'generating' || pipelineState === 'complete') && <Badge variant="green">Score: 0.9{i}</Badge>}
                                        </div>
                                    </div>
                                    <div className="h-2 bg-slate-100 rounded-full w-3/4 mb-1"></div>
                                    <div className="h-2 bg-slate-100 rounded-full w-full"></div>
                                    <div className="h-2 bg-slate-100 rounded-full w-5/6"></div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </Card>

            {/* COLUMN 3: GENERATION */}
            <Card className="flex-1 flex flex-col overflow-hidden !p-0 border-t-4 border-t-emerald-500">
                <div className="bg-slate-50/80 border-b border-slate-200 p-4 shrink-0 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-emerald-600" />
                        <h2 className="text-sm font-bold tracking-wide text-slate-900 uppercase">4. Generated Answer</h2>
                    </div>
                </div>

                <div className="flex-1 p-6 bg-white overflow-y-auto">
                    {/* Generation Placeholder */}
                    {pipelineState === 'idle' || pipelineState === 'retrieving' || pipelineState === 'reranking' ? (
                        <div className="h-full flex flex-col items-center justify-center text-center text-slate-400">
                            <div className="w-16 h-16 rounded-full bg-slate-50 flex items-center justify-center mx-auto mb-4">
                                <FileText className="w-8 h-8 text-slate-200" />
                            </div>
                            <p className="text-sm font-medium">Waiting for context...</p>
                        </div>
                    ) : pipelineState === 'generating' ? (
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 text-emerald-600 font-medium mb-4">
                                <div className="w-4 h-4 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
                                Generating response...
                            </div>
                            <div className="space-y-2 animate-pulse">
                                <div className="h-4 bg-slate-100 rounded w-full"></div>
                                <div className="h-4 bg-slate-100 rounded w-5/6"></div>
                                <div className="h-4 bg-slate-100 rounded w-4/6"></div>
                            </div>
                        </div>
                    ) : (
                        <div className="prose prose-sm prose-slate max-w-none">
                            <p className="text-slate-800 leading-relaxed">
                                <strong className="font-semibold text-slate-900 font-sans">Answer: </strong>
                                Based on the retrieved and reranked context, cross-encoders significantly improve retrieval augmented generation by providing a deeper, interaction-based relevance assessment between the query and the documents. Unlike bi-encoders used in initial retrieval, cross-encoders process the query and document simultaneously through the transformer layers, leading to much higher accuracy and better grounding for the final LLM generation phase.
                            </p>

                            <div className="mt-8 pt-4 border-t border-slate-100">
                                <h4 className="text-xs font-semibold uppercase text-slate-500 tracking-wider mb-2">Sources Referenced</h4>
                                <ul className="text-sm text-slate-600 space-y-1">
                                    <li className="flex items-start gap-2">
                                        <span className="text-emerald-500 mt-0.5">•</span>
                                        <span className="truncate">"Understanding Cross-Encoders vs Bi-Encoders in Search"</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-emerald-500 mt-0.5">•</span>
                                        <span className="truncate">"The Impact of Context Quality on LLM Hallucinations"</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </Card>

        </div>
    );
}

