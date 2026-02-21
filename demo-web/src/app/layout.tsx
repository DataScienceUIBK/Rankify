import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Rankify — Open-Source RAG Evaluation Platform',
  description: 'Unified retrieval, reranking and generation toolkit with 42 datasets, 6 retrievers and 140+ reranker models.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning className={`${inter.className} bg-white text-slate-900 antialiased`}>
        {/* Top Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-slate-100 bg-white/90 backdrop-blur-md flex items-center px-6 gap-6">
          <Link href="/" className="flex items-center gap-2 shrink-0">
            <span className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-xs font-bold shadow">R</span>
            <span className="font-bold text-slate-900 tracking-tight">Rankify</span>
          </Link>
          <div className="flex items-center gap-1 ml-4">
            <NavLink href="/" label="Home" />
            <NavLink href="/chat" label="Demo Console" highlight />
            <NavLink href="/arena" label="BEIR Arena" />
            <NavLink href="/agent" label="Rankify Agent" />
            <NavLink href="/performance" label="Benchmarks" />
          </div>
          <div className="ml-auto flex items-center gap-3">
            <a href="https://github.com/DataScienceUIBK/Rankify" target="_blank"
              className="text-xs text-slate-500 hover:text-slate-900 transition-colors">GitHub ↗</a>
            <Link href="/chat"
              className="px-3 py-1.5 bg-indigo-600 text-white text-xs font-semibold rounded-lg hover:bg-indigo-700 transition-colors">
              Launch Console
            </Link>
          </div>
        </nav>
        <main className="pt-14">{children}</main>
      </body>
    </html>
  );
}

function NavLink({ href, label, highlight }: { href: string; label: string; highlight?: boolean }) {
  return (
    <Link href={href}
      className={`px-3 py-1.5 text-sm rounded-md transition-colors font-medium ${highlight ? 'text-indigo-600 hover:bg-indigo-50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
        }`}>
      {label}
    </Link>
  );
}
