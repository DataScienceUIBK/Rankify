"use client";

import { useState } from "react";
import { Plus, MessageSquare, Trash2, BrainCircuit, ChevronRight } from "lucide-react";
import { Button } from "./ui/button";
import type { ChatSession } from "../hooks/useChatHistory";

interface SidebarProps {
    className?: string;
    sessions: ChatSession[];
    activeId: string | null;
    onCreateSession: () => void;
    onSelectSession: (id: string) => void;
    onDeleteSession: (id: string) => void;
}

function timeAgo(ts: number): string {
    const diff = Date.now() - ts;
    const m = Math.floor(diff / 60000);
    if (m < 1) return "just now";
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

function pipelineBadge(mode: ChatSession["config"]["pipelineMode"]) {
    const map = { retrieve: "R", rerank: "R+R", rag: "RAG" };
    return map[mode];
}

export function Sidebar({ className, sessions, activeId, onCreateSession, onSelectSession, onDeleteSession }: SidebarProps) {
    const [hovered, setHovered] = useState<string | null>(null);

    return (
        <div className={`flex flex-col h-full bg-[#0f1117] ${className}`}>
            {/* Logo */}
            <div className="p-4 border-b border-white/10 flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
                    <BrainCircuit className="w-4 h-4 text-white" />
                </div>
                <span className="font-semibold text-white text-sm tracking-tight">Rankify Console</span>
            </div>

            {/* New Pipeline Button */}
            <div className="p-3">
                <button
                    onClick={onCreateSession}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-white/70 hover:text-white hover:bg-white/10 transition-all border border-white/10 hover:border-white/20"
                >
                    <Plus className="w-4 h-4" />
                    New Pipeline
                </button>
            </div>

            {/* Session List */}
            <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
                {sessions.length === 0 && (
                    <p className="text-xs text-white/30 text-center mt-8 px-4">No sessions yet. Create a new pipeline above.</p>
                )}
                {sessions.map(session => (
                    <div
                        key={session.id}
                        onMouseEnter={() => setHovered(session.id)}
                        onMouseLeave={() => setHovered(null)}
                        className={`group relative flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all text-sm ${session.id === activeId
                                ? "bg-white/10 text-white"
                                : "text-white/60 hover:text-white/90 hover:bg-white/5"
                            }`}
                        onClick={() => onSelectSession(session.id)}
                    >
                        <MessageSquare className="w-4 h-4 shrink-0 opacity-70" />
                        <div className="flex-1 min-w-0">
                            <div className="truncate font-medium text-[13px] leading-tight">{session.title}</div>
                            <div className="flex items-center gap-1.5 mt-0.5">
                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${session.config.pipelineMode === "rag" ? "bg-indigo-900/60 text-indigo-300" :
                                        session.config.pipelineMode === "rerank" ? "bg-purple-900/60 text-purple-300" :
                                            "bg-emerald-900/60 text-emerald-300"
                                    }`}>
                                    {pipelineBadge(session.config.pipelineMode)}
                                </span>
                                <span className="text-[11px] text-white/30">{timeAgo(session.updatedAt)}</span>
                            </div>
                        </div>

                        {/* Hover actions */}
                        {hovered === session.id && (
                            <button
                                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 text-white/30 transition-all shrink-0"
                                onClick={(e) => { e.stopPropagation(); onDeleteSession(session.id); }}
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        )}
                        {session.id === activeId && (
                            <ChevronRight className="w-3.5 h-3.5 shrink-0 text-white/40" />
                        )}
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-white/10">
                <div className="text-xs text-white/25 text-center">
                    Rankify · Retrieval Toolkit
                </div>
            </div>
        </div>
    );
}
