import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Sidebar } from '../components/Sidebar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Rankify AI Console',
  description: 'Professional RAG Evaluation Dashboard',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning className={`${inter.className} bg-white text-slate-900 antialiased flex h-screen overflow-hidden`}>
        <Sidebar className="w-[260px] shrink-0 hidden md:flex border-r border-slate-200" />
        <div className="flex flex-col flex-1 h-full min-w-0">
          <header className="h-14 shrink-0 flex items-center px-4 border-b border-slate-200 bg-white shadow-sm z-10">
            <div className="font-semibold text-sm flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-slate-900 text-white flex items-center justify-center text-xs font-bold shadow-sm">R</div>
              <span className="tracking-tight text-slate-700">Rankify Console</span>
            </div>
          </header>
          <main className="flex-1 overflow-hidden relative flex flex-col bg-white">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
