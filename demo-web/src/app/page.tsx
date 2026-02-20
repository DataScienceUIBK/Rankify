"use client";

import { useState } from "react";
import { useChat } from "ai/react";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { ScrollArea } from "../components/ui/scroll-area";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../components/ui/accordion";
import { Badge } from "../components/ui/badge";
import { Database, Search, ListTree, Bot, Send, BrainCircuit, Upload, Sparkles } from "lucide-react";

export default function ChatPage() {
  // Pipeline State
  const [dataSource, setDataSource] = useState("msmarco");
  const [retriever, setRetriever] = useState("bge");
  const [reranker, setReranker] = useState("flashrank");
  const [generator, setGenerator] = useState("openai");

  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat",
    body: {
      configuration: {
        dataSource,
        retriever,
        reranker,
        generator
      }
    }
  });

  return (
    <div className="flex flex-col h-full bg-white relative">

      {/* Top Configuration Area */}
      <div className="shrink-0 p-4 border-b border-slate-100 bg-white/80 backdrop-blur-sm z-10 hidden md:block">
        <div className="max-w-4xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-indigo-500" />
            <span className="text-sm font-semibold text-slate-700">Pipeline Config</span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Data Source */}
            <Select value={dataSource} onValueChange={setDataSource}>
              <SelectTrigger className="w-[140px] h-8 text-xs bg-slate-50 border-slate-200">
                <Database className="w-3.5 h-3.5 mr-2 text-slate-500" />
                <SelectValue placeholder="Data Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="msmarco">MS MARCO</SelectItem>
                <SelectItem value="wikipedia">Wikipedia</SelectItem>
                <SelectItem value="custom">Custom JSONL</SelectItem>
              </SelectContent>
            </Select>

            {/* Retriever */}
            <Select value={retriever} onValueChange={setRetriever}>
              <SelectTrigger className="w-[140px] h-8 text-xs bg-slate-50 border-slate-200">
                <Search className="w-3.5 h-3.5 mr-2 text-slate-500" />
                <SelectValue placeholder="Retriever" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bm25">BM25</SelectItem>
                <SelectItem value="bge">BGE-v1.5</SelectItem>
                <SelectItem value="colbert">ColBERTv2</SelectItem>
              </SelectContent>
            </Select>

            {/* Reranker */}
            <Select value={reranker} onValueChange={setReranker}>
              <SelectTrigger className="w-[140px] h-8 text-xs bg-slate-50 border-slate-200">
                <ListTree className="w-3.5 h-3.5 mr-2 text-slate-500" />
                <SelectValue placeholder="Reranker" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="flashrank">FlashRank</SelectItem>
                <SelectItem value="monot5">MonoT5</SelectItem>
                <SelectItem value="rankzephyr">RankZephyr</SelectItem>
              </SelectContent>
            </Select>

            {/* Generator */}
            <Select value={generator} onValueChange={setGenerator}>
              <SelectTrigger className="w-[140px] h-8 text-xs bg-slate-50 border-slate-200">
                <Bot className="w-3.5 h-3.5 mr-2 text-slate-500" />
                <SelectValue placeholder="Generator" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="openai">gpt-4o</SelectItem>
                <SelectItem value="anthropic">claude-3-5</SelectItem>
                <SelectItem value="vllm">llama-3</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Chat Messages Area */}
      <ScrollArea className="flex-1 p-4 pb-32">
        <div className="max-w-3xl mx-auto flex flex-col gap-6 pt-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center mt-20 space-y-4">
              <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center border border-slate-100 shadow-sm">
                <Sparkles className="w-8 h-8 text-indigo-500" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-semibold text-slate-800">What would you like to evaluate?</h2>
                <p className="text-sm text-slate-500 max-w-sm mx-auto">Enter a query to run the full RAG pipeline with your configured settings.</p>
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <div key={m.id} className={`flex gap-4 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {m.role !== 'user' && (
                  <div className="w-8 h-8 shrink-0 rounded-lg bg-indigo-50 flex items-center justify-center border border-indigo-100">
                    <Bot className="w-5 h-5 text-indigo-600" />
                  </div>
                )}

                <div className={`flex flex-col gap-2 max-w-[85%] ${m.role === 'user' ? 'items-end' : 'items-start'}`}>

                  {/* Only show context expansion if it's the assistant and has context data */}
                  {m.role !== 'user' && (
                    <Accordion type="single" collapsible className="w-full sm:min-w-[400px]">
                      <AccordionItem value="context" className="border border-slate-200 rounded-lg bg-slate-50 overflow-hidden shadow-sm">
                        <AccordionTrigger className="px-4 py-3 hover:no-underline text-xs text-slate-500 font-medium bg-slate-50/50">
                          <div className="flex gap-2 items-center">
                            <Database className="w-3.5 h-3.5" />
                            View Computed Context Pipeline
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="p-4 bg-white border-t border-slate-100 space-y-4">
                          <div className="space-y-2">
                            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide flex justify-between">
                              <span>Retrieved Docs</span>
                              <Badge variant="secondary" className="text-[10px] font-normal tracking-wide bg-slate-100">BM25 / MS MARCO</Badge>
                            </span>
                            <div className="text-xs text-slate-600 bg-slate-50 border border-slate-100 p-3 rounded-md line-clamp-2 italic shadow-inner">
                              "Simulated document retrieval payload... 5 passages extracted."
                            </div>
                          </div>
                          <div className="space-y-2">
                            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide flex justify-between">
                              <span>Reranked Docs</span>
                              <Badge variant="secondary" className="text-[10px] font-normal tracking-wide bg-indigo-50 text-indigo-700">FlashRank</Badge>
                            </span>
                            <div className="text-xs text-slate-600 bg-indigo-50/50 border border-indigo-100 p-3 rounded-md line-clamp-2 italic shadow-sm">
                              "Simulated delta computation. Context reorganized for maximum relevance."
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  )}

                  {/* Message Text bubble */}
                  <div className={`px-5 py-3.5 text-[15px] shadow-sm leading-relaxed ${m.role === 'user'
                    ? 'bg-slate-900 text-white rounded-2xl rounded-br-sm'
                    : 'bg-white border border-slate-200 text-slate-800 rounded-2xl rounded-bl-sm prose prose-sm max-w-none'
                    }`}>
                    {m.content}
                  </div>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex items-center gap-3 text-slate-400 text-sm py-4">
              <div className="w-4 h-4 rounded-full border-2 border-slate-300 border-t-indigo-600 animate-spin" />
              Evaluating pipeline...
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="absolute bottom-0 w-full bg-gradient-to-t from-white via-white to-transparent pt-6 pb-6 px-4">
        <div className="max-w-3xl mx-auto relative">
          <form onSubmit={handleSubmit} className="relative flex items-end gap-2 bg-white border border-slate-300 shadow-sm focus-within:shadow-md focus-within:border-indigo-400 transition-all rounded-2xl p-2 pl-3">
            {dataSource === 'custom' && (
              <button type="button" className="p-2 text-slate-400 hover:text-indigo-500 transition-colors shrink-0 mb-0.5" title="Upload custom .jsonl">
                <Upload className="w-5 h-5" />
              </button>
            )}
            <textarea
              value={input}
              onChange={handleInputChange}
              placeholder="Message Rankify Console..."
              className="flex-1 max-h-32 min-h-[44px] bg-transparent resize-none outline-none text-[15px] pt-3 pb-1 px-1 placeholder:text-slate-500"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  const form = e.currentTarget.form;
                  if (form) form.requestSubmit();
                }
              }}
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              size="icon"
              className="h-10 w-10 shrink-0 rounded-xl bg-slate-900 hover:bg-indigo-600 transition-colors shadow-sm mb-0.5"
            >
              <Send className="w-4 h-4 text-white ml-0.5" />
            </Button>
          </form>
          <div className="text-center mt-2 flex items-center justify-center">
            <span className="text-[11px] text-slate-400 font-medium tracking-wide">Rankify generates evaluated context. Verify the citations.</span>
          </div>
        </div>
      </div>

    </div>
  );
}
