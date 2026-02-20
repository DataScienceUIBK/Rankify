import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Search, ListOrdered, MessageSquare, Bot, Home, Github, Settings } from 'lucide-react';
import { cn } from '../components/ui';

const navItems = [
    { path: '/', label: 'Overview', icon: Home },
    { path: '/retrieval', label: 'Retrieval', icon: Search },
    { path: '/reranking', label: 'Re-Ranking', icon: ListOrdered },
    { path: '/rag', label: 'RAG Pipeline', icon: MessageSquare },
    { path: '/agent', label: 'RankifyAgent', icon: Bot },
];

export function Layout() {
    return (
        <div className="flex min-h-screen text-slate-50 relative overflow-hidden">
            {/* Sidebar background */}
            <div className="w-64 fixed inset-y-0 left-0 z-20 bg-[rgba(15,17,26,0.8)] backdrop-blur-xl border-r border-[#ffffff14] flex flex-col">
                {/* Logo Section */}
                <div className="p-6 pb-2">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.5)]">
                            <Search className="w-6 h-6 text-white" strokeWidth={2.5} />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 m-0">
                                Rankify
                            </h1>
                            <p className="text-xs text-indigo-400 font-medium tracking-wider uppercase m-0">Toolkit</p>
                        </div>
                    </div>
                </div>

                {/* Navigation */}
                <nav className="flex-1 px-4 py-8 space-y-2">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={({ isActive }) =>
                                cn(
                                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 font-medium",
                                    isActive
                                        ? "bg-indigo-500/15 text-indigo-300 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]"
                                        : "text-slate-400 hover:text-white hover:bg-white/5"
                                )
                            }
                        >
                            <item.icon className="w-5 h-5" />
                            {item.label}
                        </NavLink>
                    ))}
                </nav>

                {/* Footer Links */}
                <div className="p-4 border-t border-[#ffffff14]">
                    <a
                        href="https://github.com/DataScienceUIBK/Rankify"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
                    >
                        <Github className="w-4 h-4" />
                        GitHub Repo
                    </a>
                </div>
            </div>

            {/* Main Content Area */}
            <main className="flex-1 ml-64 p-8 relative">
                <div className="max-w-6xl mx-auto h-full">
                    <Outlet />
                </div>
            </main>
        </div>
    );
}
