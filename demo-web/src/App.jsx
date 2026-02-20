import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './layouts/Layout';
import { Home } from './pages/Home';
import { Retrieval } from './pages/Retrieval';
import { Reranking } from './pages/Reranking';
import { RAG } from './pages/RAG';
import { Agent } from './pages/Agent';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="retrieval" element={<Retrieval />} />
          <Route path="reranking" element={<Reranking />} />
          <Route path="rag" element={<RAG />} />
          <Route path="agent" element={<Agent />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
