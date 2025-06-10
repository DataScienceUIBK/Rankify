import os
import zipfile
import requests
import json
from typing import List
from pyserini.search.lucene import LuceneSearcher
from rankify.dataset.dataset import Document, Context
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from rankify.utils.pre_defind_models import INDEX_TYPE

class BM25Retriever:
    """
    Implements **BM25**, a **probabilistic ranking model** for retrieving documents from large-scale corpora.

    BM25, based on the **Probabilistic Relevance Framework**, ranks documents based on their relevance to a given query. 
    This retriever utilizes **Pyserini's LuceneSearcher** for efficient batch-based retrieval.

    References:
        - **Robertson & Zaragoza (2009)**: *The Probabilistic Relevance Framework: BM25 and Beyond*.  
          [paper](https://www.nowpublishers.com/article/Details/INR-019)

    Attributes:
        n_docs (int): Number of top documents to retrieve per query.
        batch_size (int): Number of queries processed in a single batch for efficiency.
        threads (int): Number of parallel threads used for retrieval.
        index_type (str): The type of index (`"wiki"` or `"msmarco"`) used for retrieval.
        index_url (str): URL to download the prebuilt BM25 index.
        index_folder (str): Path where the BM25 index is stored.
        index_path (str): Path to the Lucene index used by Pyserini.
        title_map_path (str): Path to a JSON file mapping passage IDs to titles.
        searcher (LuceneSearcher): Pyserini-based search engine for BM25 retrieval.
        pid2title (dict): Dictionary mapping document IDs to their titles.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever
        
        retriever = Retriever(model="bm25", n_docs=5, index_type="wiki")
        
        documents = [Document(question=Question("Who discovered gravity?"))]
        
        retrieved_documents = retriever.retrieve(documents)
        
        print(retrieved_documents[0].contexts[0].text)
        ```
    """
    def __init__(self, model="bm25", n_docs: int = 10, batch_size: int = 36, threads: int = 30 , index_type: str = 'wiki', index_folder: str = '') -> None:
        """
        Initializes the BM25 retriever.

        Args:
            model (str, optional): The retrieval model name (`"bm25"`). Defaults to `"bm25"`.
            n_docs (int, optional): Number of **top documents** to retrieve per query. Defaults to `10`.
            batch_size (int, optional): Number of **queries** processed in a batch. Defaults to `36`.
            threads (int, optional): Number of **parallel threads** for retrieval. Defaults to `30`.
            index_type (str, optional): Type of **index** to use (`"wiki"` or `"msmarco"`). Defaults to `"wiki"`.

        Raises:
            ValueError: If the `index_type` is **not supported**.
        """
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.threads = threads
        self.tokenizer = SimpleTokenizer()

        if 'bm25' not in INDEX_TYPE or index_type not in INDEX_TYPE['bm25']:
            raise ValueError(f"Index type {index_type} is not supported.")
        
        self.index_type = index_type

        # If a custom local index folder is provided, use it; else default to cached folder
        if index_folder:
            self.index_folder = index_folder
        else:
            self.index_url = INDEX_TYPE['bm25'][index_type]['url']
            self.index_folder = os.path.join(os.environ.get("RERANKING_CACHE_DIR", "./cache"), 'index', f"bm25_index_{index_type}")

        if index_type =="wiki":
            self.index_path =  os.path.join(self.index_folder, f"bm25_index")
        else:
            self.index_path =  os.path.join(self.index_folder, f"bm25_index_{index_type}")

        self.title_map_path = os.path.join(self.index_path, "corpus.json")


        # TODO: Check if still supported in future
        # self._ensure_index_downloaded()

        self.searcher = LuceneSearcher(self.index_path)
        #Todo: remove
        #with open(self.title_map_path, "r", encoding="utf-8") as f:
            #self.pid2title = json.load(f)

    def _ensure_index_downloaded(self) -> None:
        """
        Ensures that the BM25 index and associated files are **downloaded and extracted**.

        If the index does not exist locally, it downloads and extracts the index from the 
        specified URL.

        Raises:
            RuntimeError: If the index **download fails**.
        """
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_folder, exist_ok=True)
            zip_path = os.path.join(self.index_folder, f"bm25_index_{self.index_type}.zip")
            print("Downloading BM25 index...")
            response = requests.get(self.index_url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading BM25 Index"):
                    f.write(chunk)
            print("Extracting BM25 index...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.index_folder)
            os.remove(zip_path)
    def retrieve__(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **relevant contexts** for each document in the input list using BM25.

        Args:
            documents (List[Document]): A **list of queries** as `Document` instances.

        Returns:
            List[Document]: Documents **updated** with retrieved `Context` instances.
        """
        print(f"Retrieving {len(documents)} documents one at a time...")

        for document in tqdm(documents, desc="Retrieving documents", unit="doc"):
            query = document.question.question  # Extract the query string
            try:
                # Perform search for the single query
                hits = self.searcher.search(query, self.n_docs)

                contexts = []
                for hit in hits:
                    try:
                        lucene_doc = self.searcher.doc(hit.docid)
                        print(lucene_doc.raw())
                        raw_content = json.loads(lucene_doc.raw())  # Parse the raw JSON
                        text = raw_content.get("contents", "")
                        title = self.pid2title.get(hit.docid, "No Title")

                        context = Context(
                            id=int(hit.docid),
                            title=title,
                            text=text,
                            score=hit.score,
                            has_answer=has_answers(text, document.answers.answers, self.tokenizer)
                        )
                        contexts.append(context)
                    except Exception as e:
                        print(f"Error processing document ID {hit.docid}: {e}")

                # Assign the retrieved contexts to the document
                document.contexts = contexts

            except Exception as e:
                print(f"Error retrieving contexts for query '{query}': {e}")

        return documents

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **relevant contexts** for each document in the input list using BM25.

        Args:
            documents (List[Document]): A **list of queries** as `Document` instances.

        Returns:
            List[Document]: Documents **updated** with retrieved `Context` instances.
        """
        # Extract the question text and unique IDs
        question_texts = [doc.question.question for doc in documents]  # Extract plain string questions
        batch_qids = [str(i) for i in range(len(question_texts))]  # Generate unique IDs for each query

        print(f"Retrieving {len(documents)} documents in batches of {self.batch_size}...")

        # Perform batch search for all questions
        batch_results = self._batch_search(question_texts, batch_qids, self.n_docs, self.batch_size, self.threads)

        for i, qid in enumerate(tqdm(batch_qids, desc="Processing documents", unit="doc")):
            document = documents[i]
            hits = batch_results.get(qid, [])

            contexts = []
            for hit in hits:
                try:
                    lucene_doc = self.searcher.doc(hit.docid)
                    raw_content = json.loads(lucene_doc.raw())

                    content = raw_content.get("contents", "")
                    has_title = '\n' in content
                    title = content.split('\n')[0] if has_title else "No Title"
                    text = content.split('\n')[1] if has_title else content

                    #Todo: Change id type from int to str
                    context = Context(
                        id=int(hit.docid),
                        title=title,
                        text=text,
                        score=hit.score,
                        has_answer=has_answers(text, document.answers.answers, self.tokenizer)
                    )
                    contexts.append(context)
                except Exception as e:
                    print(f"Error processing document ID {hit.docid}: {e}")

            document.contexts = contexts

        return documents

    def _batch_search(self, queries: List[str], qids: List[str], k: int, batch_size: int, threads: int):
        """
        Performs **batch search** using **Pyserini's LuceneSearcher**.

        Args:
            queries (List[str]): **List of query strings**.
            qids (List[str]): **List of corresponding query IDs**.
            k (int): **Number of documents** to retrieve per query.
            batch_size (int): **Batch size** for processing queries.
            threads (int): **Number of parallel threads**.

        Returns:
            dict: **Dictionary** mapping query IDs to retrieved documents.
        """
        batch_results = {}
        for start in tqdm(range(0, len(queries), batch_size), desc="Batch search"):
            end = min(start + batch_size, len(queries))
            batch_queries = queries[start:end]
            batch_qids = qids[start:end]

            try:
                batch_hits = self.searcher.batch_search(batch_queries, batch_qids, k=k, threads=threads)
                batch_results.update(batch_hits)
            except Exception as e:
                print(f"Batch search failed for queries {batch_queries} with error {e}")

        return batch_results
