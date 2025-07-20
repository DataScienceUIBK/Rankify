import os
import subprocess
import tempfile
import json
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

# DPR evaluation
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
# Lucene search
from pyserini.search.lucene import LuceneSearcher

# Dataset & tools
from rankify.dataset.dataset import Document, Context
from rankify.tools.Tools import WebSearchTool
from rankify.tools.websearch.context.chunker import Chunker

class OnlineRetriever:
    def __init__(
        self,
        n_docs: int = 5,
        api_key: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwags
    ) -> None:
        self.n_docs = n_docs
        self.searcher = WebSearchTool(search_provider_api_key=api_key)
        self.tokenizer = SimpleTokenizer()
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _search_web(self, query: str) -> List[Any]:
        if not self.searcher.is_initialized:
            self.searcher.setup()
        return self.searcher.forward(query, num_result=self.n_docs)


    def retrieve(self, documents: List[Document]) -> List[Document]:
        for idx, doc in enumerate(tqdm(documents, desc="Fetching docs...")):
            question = doc.question.question
            logger.info(f"Fetching contexts for Q{idx}: {question}")

            sources = self._search_web(question)
            processed = []
            for s_idx, source in enumerate(sources):
                text = source.get('fit_markdown', '')
                # with open("log.txt", "a",  encoding="utf-8") as f:
                #     f.writelines(html)
                if text:
                    passages = self.chunker.split_text(text)
                    #print(len(passages), "passages found")
                    for p_idx, p in enumerate(passages):
                        #print(p)
                        
                        #print(f"Processing snippet {s_idx} passage {p_idx}: {p['contents'][:50]}...")
                        processed.append({'id': f'{s_idx}_{p_idx}', 'contents': p})
                else:
                    snippet = source.get('snippet','')
                    if len(snippet.split()) > 20:
                        processed.append({'id': f's{ s_idx }_snip', 'contents': snippet})

            # Indexing
            with tempfile.TemporaryDirectory() as tmpdir:
                json.dump(processed, open(os.path.join(tmpdir,'docs.json'),'w'))
                subprocess.run([
                    'python','-m','pyserini.index.lucene',
                    '-collection','JsonCollection','-generator','DefaultLuceneDocumentGenerator',
                    '-input',tmpdir,'-index',tmpdir,
                    '-storePositions','-storeDocvectors','-storeRaw'
                ], check=True)

                searcher = LuceneSearcher(tmpdir)
                hits = searcher.search(question, k=self.n_docs)
                contexts: List[Context] = []
                for h_idx, hit in enumerate(hits):
                    raw = searcher.doc(hit.docid).raw()
                    data = json.loads(raw)
                    text = data.get('contents','')
                    title = "" #text.split("\n")[0]
                    try:
                        cid = int(hit.docid)
                    except ValueError:
                        cid = h_idx
                    contexts.append(Context(
                        id=cid,
                        title=title,
                        text=text,
                        score=hit.score,
                        has_answer=has_answers(text, doc.answers.answers, self.tokenizer)
                    ))
                doc.contexts = contexts
        return documents
