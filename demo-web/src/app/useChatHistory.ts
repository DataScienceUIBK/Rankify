"use client";

import { useState, useEffect, useCallback } from "react";

export interface ChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    pipelineData?: {
        retrievedDocs?: Array<{ id: string; text: string; score: number }>;
        rerankedDocs?: Array<{ id: string; text: string; score: number }>;
        generatorModel?: string;
        retrieverModel?: string;
        rerankerModel?: string;
        dataSource?: string;
    };
}

export interface ChatSession {
    id: string;
    title: string;
    createdAt: number;
    updatedAt: number;
    messages: ChatMessage[];
    config: {
        pipelineMode: "retrieve" | "rerank" | "rag";
        dataSource: string;
        retriever: string;
        rerankerCategory: string;
        rerankerModel: string;
        generator: string;
    };
}

const STORAGE_KEY = "rankify_chat_sessions";
const ACTIVE_KEY = "rankify_active_session";

function generateId() {
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function defaultSession(): ChatSession {
    return {
        id: generateId(),
        title: "New Pipeline",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
        config: {
            pipelineMode: "rag",
            dataSource: "msmarco",
            retriever: "bge",
            rerankerCategory: "flashrank",
            rerankerModel: "ms-marco-MiniLM-L-12-v2",
            generator: "openai",
        },
    };
}

export function useChatHistory() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [activeId, setActiveId] = useState<string | null>(null);
    const [hydrated, setHydrated] = useState(false);

    // Load from localStorage on mount
    useEffect(() => {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            const rawId = localStorage.getItem(ACTIVE_KEY);
            const loaded: ChatSession[] = raw ? JSON.parse(raw) : [];
            if (loaded.length === 0) {
                const s = defaultSession();
                setSessions([s]);
                setActiveId(s.id);
            } else {
                setSessions(loaded);
                setActiveId(rawId && loaded.find(s => s.id === rawId) ? rawId : loaded[0].id);
            }
        } catch {
            const s = defaultSession();
            setSessions([s]);
            setActiveId(s.id);
        }
        setHydrated(true);
    }, []);

    // Persist to localStorage whenever sessions change
    useEffect(() => {
        if (!hydrated) return;
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }, [sessions, hydrated]);

    useEffect(() => {
        if (!hydrated || !activeId) return;
        localStorage.setItem(ACTIVE_KEY, activeId);
    }, [activeId, hydrated]);

    const activeSession = sessions.find(s => s.id === activeId) ?? null;

    const createSession = useCallback(() => {
        const s = defaultSession();
        setSessions(prev => [s, ...prev]);
        setActiveId(s.id);
        return s;
    }, []);

    const selectSession = useCallback((id: string) => {
        setActiveId(id);
    }, []);

    const deleteSession = useCallback((id: string) => {
        setSessions(prev => {
            const next = prev.filter(s => s.id !== id);
            if (next.length === 0) {
                const s = defaultSession();
                setActiveId(s.id);
                return [s];
            }
            return next;
        });
        setActiveId(prev => {
            if (prev !== id) return prev;
            const remaining = sessions.filter(s => s.id !== id);
            return remaining.length > 0 ? remaining[0].id : null;
        });
    }, [sessions]);

    const updateSessionConfig = useCallback((id: string, config: Partial<ChatSession["config"]>) => {
        setSessions(prev => prev.map(s =>
            s.id === id ? { ...s, config: { ...s.config, ...config }, updatedAt: Date.now() } : s
        ));
    }, []);

    const addMessage = useCallback((sessionId: string, message: ChatMessage) => {
        setSessions(prev => prev.map(s => {
            if (s.id !== sessionId) return s;
            const msgs = [...s.messages, message];
            // Auto-title from first user message
            const title = s.messages.length === 0 && message.role === "user"
                ? message.content.slice(0, 50)
                : s.title;
            return { ...s, messages: msgs, title, updatedAt: Date.now() };
        }));
    }, []);

    const updateLastAssistantMessage = useCallback((sessionId: string, content: string, pipelineData?: ChatMessage["pipelineData"]) => {
        setSessions(prev => prev.map(s => {
            if (s.id !== sessionId) return s;
            const msgs = [...s.messages];
            for (let i = msgs.length - 1; i >= 0; i--) {
                if (msgs[i].role === "assistant") {
                    msgs[i] = { ...msgs[i], content, pipelineData: pipelineData ?? msgs[i].pipelineData };
                    break;
                }
            }
            return { ...s, messages: msgs };
        }));
    }, []);

    const clearSession = useCallback((id: string) => {
        setSessions(prev => prev.map(s =>
            s.id === id ? { ...s, messages: [], title: "New Pipeline", updatedAt: Date.now() } : s
        ));
    }, []);

    return {
        sessions,
        activeId,
        activeSession,
        hydrated,
        createSession,
        selectSession,
        deleteSession,
        updateSessionConfig,
        addMessage,
        updateLastAssistantMessage,
        clearSession,
    };
}
