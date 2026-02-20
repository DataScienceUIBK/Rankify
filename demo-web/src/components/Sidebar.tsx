import { Button } from "./ui/button";
import { Plus, MessageSquare, Settings } from "lucide-react";

export function Sidebar({ className }: { className?: string }) {
    return (
        <div className={`flex flex-col h-full bg-slate-50/50 ${className}`}>
            <div className="p-4 flex flex-col gap-2">
                <Button variant="outline" className="w-full justify-start gap-2 shadow-sm rounded-lg bg-white">
                    <Plus className="w-4 h-4" />
                    New Pipeline
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1">
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">Recent Sessions</div>
                {[1, 2, 3].map(i => (
                    <button key={i} className="w-full text-left px-2 py-2 text-sm text-slate-700 hover:bg-slate-200 rounded-md transition-colors flex items-center gap-2">
                        <MessageSquare className="w-4 h-4 text-slate-400" />
                        <span className="truncate">Evaluating BGE + FlashRank</span>
                    </button>
                ))}
            </div>

            <div className="p-4 border-t border-slate-200">
                <Button variant="ghost" className="w-full justify-start gap-2 text-slate-600 hover:text-slate-900">
                    <Settings className="w-4 h-4" />
                    Settings
                </Button>
            </div>
        </div>
    );
}
