# bm25_retriever.py
import json
from typing import List
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm
import os
from .base_retriever import BaseRetriever
from .index_manager import IndexManager
from rankify.dataset.dataset import Document, Context

class BM25Retriever(BaseRetriever):
    """
    BM25 retriever implementation using Pyserini's LuceneSearcher.
    
    Implements probabilistic ranking model BM25 for document retrieval.
    """
    
    def __init__(self, index_type: str = "wiki", index_folder: str = None, **kwargs):
        super().__init__(**kwargs)
        self.index_type = index_type
        self.index_folder = index_folder
        self.tokenizer = SimpleTokenizer()
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Get index path and load mappings
        if index_folder:
            self.index_path = index_folder
            self.id_mapping = self.index_manager.load_id_mapping(index_folder)
        else:
            self.index_path = self.index_manager.get_index_path("bm25", index_type)
            self.id_mapping = None
        
        # Initialize searcher
        self.searcher = self._initialize_searcher()
    
    def _initialize_searcher(self) -> LuceneSearcher:
        """Initialize Lucene searcher."""
        if self.index_path.startswith("wikipedia-") or "prebuilt" in self.index_path:
            return LuceneSearcher.from_prebuilt_index(self.index_path)
        else:
            # Adjust path for local indexes
            if self.index_type == "wiki":
                actual_path = os.path.join(self.index_path, "bm25_index")
            else:
                actual_path = os.path.join(self.index_path, f"bm25_index_{self.index_type}")
            return LuceneSearcher(actual_path)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """Retrieve relevant contexts using BM25."""
        queries = [doc.question.question for doc in documents]
        qids = [str(i) for i in range(len(queries))]
        
        print(f"Retrieving {len(documents)} documents with BM25...")
        
        # Perform batch search
        batch_results = self._batch_search(queries, qids)
        
        # Process results
        for i, document in enumerate(tqdm(documents, desc="Processing documents")):
            contexts = []
            hits = batch_results.get(str(i), [])
            
            for hit in hits:
                try:
                    context = self._create_context_from_hit(hit, document)
                    if context:
                        contexts.append(context)
                except Exception as e:
                    print(f"Error processing document ID {hit.docid}: {e}")
            
            document.contexts = contexts
        
        return documents
    
    def _create_context_from_hit(self, hit, document) -> Context:
        """Create Context object from search hit."""
        lucene_doc = self.searcher.doc(hit.docid)
        raw_content = json.loads(lucene_doc.raw())
        
        content = raw_content.get("contents", "")
        has_title = '\n' in content
        title = content.split('\n')[0] if has_title else "No Title"
        text = content.split('\n')[1] if has_title else content
        
        # Handle ID mapping if available
        doc_id = self.id_mapping.get(int(hit.docid)) if self.id_mapping else hit.docid
        
        return Context(
            id=doc_id,
            title=title,
            text=text,
            score=hit.score,
            has_answer=has_answers(text, document.answers.answers, self.tokenizer)
        )
    
    def _batch_search(self, queries: List[str], qids: List[str]) -> dict:
        """Perform batch search using Lucene searcher."""
        batch_results = {}
        
        for start in tqdm(range(0, len(queries), self.batch_size), desc="Batch search"):
            end = min(start + self.batch_size, len(queries))
            batch_queries = queries[start:end]
            batch_qids = qids[start:end]
            
            try:
                batch_hits = self.searcher.batch_search(
                    batch_queries, batch_qids, k=self.n_docs, threads=self.threads
                )
                batch_results.update(batch_hits)
            except Exception as e:
                print(f"Batch search failed for queries {batch_queries}: {e}")
        
        return batch_results