import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Rankify AI Console',
  description: 'Professional RAG Evaluation Dashboard — Retrieval, Reranking, and Generation',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning className={`${inter.className} bg-white text-slate-900 antialiased`}>
        {children}
      </body>
    </html>
  );
}
