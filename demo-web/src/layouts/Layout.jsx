import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Home, Layers, Github, BookOpen, Search } from 'lucide-react';
import { cn } from '../components/ui';

const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/console', label: 'Console', icon: Layers },
];

export function Layout() {
    return (
        <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900 font-sans">
            {/* Top Navigation Bar */}
            <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white shadow-sm">
                <div className="flex h-14 items-center justify-between px-6 max-w-7xl mx-auto">
                    {/* Logo Section */}
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center">
                            <Search className="w-4 h-4 text-white" strokeWidth={3} />
                        </div>
                        <span className="text-lg font-bold tracking-tight">Rankify</span>
                        <span className="ml-2 px-2 py-0.5 rounded-full bg-slate-100 text-xs font-medium text-slate-600 border border-slate-200">
                            v1.0
                        </span>
                    </div>

                    {/* Center Navigation Links */}
                    <nav className="hidden md:flex items-center gap-1 bg-slate-100/50 p-1 rounded-lg border border-slate-200/60">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={({ isActive }) =>
                                    cn(
                                        "flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200",
                                        isActive
                                            ? "bg-white text-slate-900 shadow-sm border border-slate-200/50"
                                            : "text-slate-500 hover:text-slate-900 hover:bg-slate-200/50"
                                    )
                                }
                            >
                                <item.icon className="w-4 h-4" />
                                {item.label}
                            </NavLink>
                        ))}
                    </nav>

                    {/* Right Actions */}
                    <div className="flex items-center gap-4">
                        <a
                            href="https://github.com/DataScienceUIBK/rankify"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm font-medium text-slate-500 hover:text-slate-900 flex items-center gap-2 transition-colors"
                        >
                            <Github className="w-4 h-4" />
                            <span className="hidden sm:inline">GitHub</span>
                        </a>
                        <a
                            href="http://rankify.readthedocs.io/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm font-medium text-slate-500 hover:text-slate-900 flex items-center gap-2 transition-colors"
                        >
                            <BookOpen className="w-4 h-4" />
                            <span className="hidden sm:inline">Documentation</span>
                        </a>
                    </div>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="flex-1 w-full bg-slate-50 flex flex-col">
                <div className="flex-1 w-full max-w-[1600px] mx-auto p-4 sm:p-6 lg:p-8">
                    <Outlet />
                </div>
            </main>
        </div>
    );
}
