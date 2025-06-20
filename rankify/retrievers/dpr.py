import os
import json
from pathlib import Path

import requests
import zipfile
from typing import List
from pyserini.search.faiss import FaissSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from rankify.dataset.dataset import Document, Context
from tqdm import tqdm
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

class DenseRetriever:
    """
    Implements **Dense Passage Retrieval (DPR)**, **ANCE**, and **BPR** using **FAISS indexes** 
    for efficient document retrieval. These models leverage **dense vector representations** to find **semantically relevant** 
    passages.

    References:
        - **Karpukhin et al. (2020)**: *Dense Passage Retrieval for Open-Domain Question Answering*.
          [Paper](https://arxiv.org/abs/2004.04906)
        - **Xiong et al. (2020)**: *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE)*.
          [Paper](https://arxiv.org/abs/2007.00808)
        - **Ni et al. (2021)**: *Efficient Passage Retrieval with Hashing for Open-domain Question Answering*.
          [Paper](https://arxiv.org/abs/2106.00882)

    Attributes:
        model (str): The **retriever type** (`"dpr-multi"`, `"dpr-single"`, `"ance-multi"`, `"bpr-single"`).
        index_type (str): The **index type** (`"wiki"` or `"msmarco"`).
        n_docs (int): Number of **documents retrieved per query**.
        batch_size (int): Number of **queries processed in a batch**.
        threads (int): Number of **parallel threads** for retrieval.
        tokenizer (SimpleTokenizer): **Tokenizer** for answer matching.
        searcher (FaissSearcher): **FAISS searcher** for document retrieval.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(method='dpr', model="dpr-multi", index_type="wiki", n_docs=5)

        documents = [Document(question=Question("What is artificial intelligence?"))]

        retrieved_documents = retriever.retrieve(documents)

        print(retrieved_documents[0].contexts[0].text)
        ```
    """

    MSMARCO_CORPUS_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true"

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache" )
    DENSE_INDEX_MAP = {
        "dpr-multi": {
            "wiki": "wikipedia-dpr-100w.dpr-multi",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_multi.zip?download=true"
        },
        "dpr-single": {
            "wiki": "wikipedia-dpr-100w.dpr-single-nq",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_single.zip?download=true"
        },
        "ance-multi": {
            "wiki": "wikipedia-dpr-100w.ance-multi",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_ance.zip?download=true"
        },
        "bpr-single": {
            "wiki": "wikipedia-dpr-100w.bpr-single-nq",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_bpr.zip?download=true"
        }
    }
    QUERY_ENCODER_MAP = {
        "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
        "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
        "ance-multi": "castorini/ance-dpr-question-multi",
        "bpr-single": "castorini/bpr-nq-question-encoder"
    }

    def __init__(self, model: str = "dpr-multi", index_type: str = "wiki", n_docs: int = 10, batch_size: int = 36, threads: int = 30,
                 index_folder: str = ''):
        """
        Initializes the **DenseRetriever** for batch processing.

        Args:
            model (str, optional): The **retriever type** (`"dpr-multi"`, `"dpr-single"`, `"ance-multi"`, `"bpr-single"`). Defaults to `"dpr-multi"`.
            index_type (str, optional): The **index type** (`"wiki"` or `"msmarco"`). Defaults to `"wiki"`.
            n_docs (int, optional): Number of **documents to retrieve per query**. Defaults to `10`.
            batch_size (int, optional): Number of **queries to process in a batch**. Defaults to `36`.
            threads (int, optional): Number of **parallel threads** for retrieval. Defaults to `30`.

        Raises:
            ValueError: If the `model` is **not supported**.
            ValueError: If the `index_type` is **not available** for the selected model.
        """
        if model not in self.DENSE_INDEX_MAP:
            raise ValueError(f"Unsupported retriever model: {model}")

        if not index_folder and index_type not in self.DENSE_INDEX_MAP[model]:
            raise ValueError(f"Unsupported index type '{index_type}' for model '{model}'.")

        self.model = model
        self.index_type = index_type
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.threads = threads
        self.tokenizer = SimpleTokenizer()
        self.index_folder = index_folder
        if  index_folder:
            self.id_to_id = os.path.join(self.index_folder,"id_mapping.json")
            
            with open(self.id_to_id, "r", encoding="utf-8") as f:
                self.idtoid = json.load(f)
                self.idtoid = {v: k for k, v in  self.idtoid.items()}

        # Load the appropriate searcher
        self.searcher = self._initialize_searcher()

        if index_folder and index_type != "wiki":
            self.load_corpus()
            return

        if index_type == "msmarco":
            self._load_msmarco_corpus()

    def _initialize_searcher(self):
        """
        Initializes the FAISS searcher for document retrieval.

        Returns:
            FaissSearcher: A **FAISS searcher** instance.
        """
        if self.index_folder:
            # If a local index folder is provided, use it directly
            print(f"Initializing searcher with local index folder: {self.index_folder}...")
            if self.index_type == "wiki":
                print("Using LuceneSearcher for wiki index...")
                return LuceneSearcher(self.index_folder)
            else:
                print("Using FaissSearcher for custom index...")
                if not os.path.exists(self.index_folder):
                    raise ValueError(f"Index folder '{self.index_folder}' does not exist.")
                return FaissSearcher(self.index_folder, self.QUERY_ENCODER_MAP[self.model])
        
        # Only access DENSE_INDEX_MAP if we're not using a custom index folder
        index_url_or_prebuilt = self.DENSE_INDEX_MAP[self.model][self.index_type]

        if index_url_or_prebuilt.startswith("http"):  # Handle downloadable indexes
            index_folder_name = os.path.basename(index_url_or_prebuilt).split("?")[0].replace(".zip", "")
            local_dir = os.path.join(self.CACHE_DIR, 'index' , index_folder_name ) 

            # Ensure the index is downloaded and extracted only once
            if not os.path.exists(local_dir):
                print(f"Preparing index folder at {local_dir}...")
                self._download_and_extract(index_url_or_prebuilt, local_dir)

            # Use the local directory to initialize FaissSearcher
            print(f"Initializing FaissSearcher with local index at {local_dir}...")
            return FaissSearcher(local_dir, self.QUERY_ENCODER_MAP[self.model])

        else:  # Handle prebuilt indexes from Pyserini
            print(f"Initializing FaissSearcher with prebuilt index: {index_url_or_prebuilt}...")
            return FaissSearcher.from_prebuilt_index(index_url_or_prebuilt, self.QUERY_ENCODER_MAP[self.model])
    def load_corpus(self):
        """
        Loads the **corpus** into memory.
        """
        corpus_file = Path(self.index_folder) / "corpus.jsonl"

        print("Loading corpus...")
        self.corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc.get("docid") or doc.get("id")
                contents = doc.get("contents", "")
                #print(doc, contents)
                #asdasdas
                title = doc.get("title", contents[:100] if contents else "No Title")
                self.corpus[str(doc_id)] = {"contents": contents, "title": title}

    def _load_msmarco_corpus(self):
        """
        Downloads and loads the MSMARCO passage corpus into memory.

        This function checks whether the **MSMARCO passage corpus** exists locally. If not, 
        it downloads the corpus file and stores it in the cache directory. It then loads 
        the corpus into memory as a dictionary, mapping `doc_id` to passage **text** and **title**.

        Notes:
            - The MSMARCO corpus is a **tab-separated file** (TSV) with three columns: `doc_id`, `text`, `title`.
            - This method is only required when using the `"msmarco"` index.

        Raises:
            Exception: If the file format is incorrect or an issue occurs during parsing.

        """
        corpus_file = os.path.join(self.CACHE_DIR , "msmarco-passage-corpus.tsv")
        if not os.path.exists(corpus_file):
            self._download_file(self.MSMARCO_CORPUS_URL, corpus_file)

        print("Loading MSMARCO corpus...")
        self.corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                doc_id, text, title = line.strip().split("\t")
                self.corpus[doc_id] = {"text": text, "title": title}

    def _download_file(self, url, save_path):
        """
        Downloads a file from the specified URL and saves it to the given path.

        This function streams the file in chunks to avoid memory overflow and displays 
        a progress bar using `tqdm`.

        Args:
            url (str): The **URL** of the file to be downloaded.
            save_path (str): The **local file path** where the downloaded file will be saved.

        Raises:
            requests.exceptions.RequestException: If the download fails due to a network issue.
        """
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)

    def _download_and_extract(self, url, destination):
        """
        Downloads and extracts a ZIP file from a given URL.

        This method:
            - Downloads a **ZIP archive** to a temporary location.
            - Extracts its contents to the specified **destination directory**.
            - Removes the ZIP file to save disk space.

        Args:
            url (str): The **URL** of the ZIP file to download.
            destination (str): The **local directory** where the extracted files will be stored.

        Raises:
            zipfile.BadZipFile: If the ZIP file is corrupt or cannot be extracted.
            requests.exceptions.RequestException: If the download fails.
        """
        # Ensure destination directory exists (only once here)
        os.makedirs(destination, exist_ok=True)

        zip_name = os.path.basename(url).split("?")[0]
        zip_path = os.path.join(self.CACHE_DIR, "index", zip_name)  # Temporarily save in CACHE_DIR

        if not os.path.exists(zip_path):
            print(f"Downloading index from {url}...")
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading Index"):
                    f.write(chunk)

        print(f"Extracting {zip_name} into {destination}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)

        os.remove(zip_path)  # Remove the ZIP file to save space
        print("Extraction complete.")




    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **relevant contexts** for a list of **documents**.

        Args:
            documents (List[Document]): A **list of queries** as `Document` instances.

        Returns:
            List[Document]: Documents **updated** with retrieved `Context` instances.
        """
        queries = [doc.question.question for doc in documents]
        batch_results = self._batch_search(queries, [str(i) for i in range(len(queries))])
        
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")):
            contexts = []
            hits = batch_results.get(str(i), [])
            for hit in hits:
                doc_id = hit.docid
                try:
                    if self.index_type == "msmarco":
                        # Handle MSMARCO index
                        doc_id = hit.docid
                        doc_data = self.corpus.get(str( hit.docid), {"text": "Not Found", "title": "Not Found"})
                        text = doc_data["text"]
                        title = doc_data["title"]
                        
                    elif self.index_folder and hasattr(self, 'corpus'):
                        # Handle custom index with loaded corpus
                        doc_id =  self.idtoid.get(int(hit.docid))
                        #print(doc_id)
                        #sadasda
                        
                        doc_data = self.corpus.get(str( hit.docid), {"contents": "Not Found", "title": "Not Found"})
                        text = doc_data["contents"]
                        title = doc_data["title"]
                        
                    elif isinstance(self.searcher, LuceneSearcher):
                        # Handle wiki index with LuceneSearcher
                        lucene_doc = self.searcher.doc(hit.docid)
                        raw_content = json.loads(lucene_doc.raw())
                        content = raw_content.get("contents", "")
                        title = content.split('\n')[0] if '\n' in content else "No Title"
                        text = content.split('\n')[1] if '\n' in content else content
                        
                    else:
                        # Fallback: try to get document info from the hit itself
                        print(f"Warning: Cannot retrieve document content for ID {hit.docid}. Using fallback.")
                        text = f"Document {hit.docid}"
                        title = f"Document {hit.docid}"
                    #print(hit.docid)
                    context = Context(
                        id=doc_id,#hit.docid,
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


    def _batch_search(self, queries, qids):
        """
        Performs batch search using **Pyserini's FAISS retriever**.

        Args:
            queries (List[str]): **List of query strings**.
            qids (List[str]): **List of corresponding query IDs**.

        Returns:
            dict: **Dictionary** mapping query IDs to retrieved documents.
        """
        batch_results = {}
        batch_qids, batch_queries = [], []

        for idx, (qid, query) in enumerate(tqdm(zip(qids, queries), desc="Batch search", total=len(qids))):
            query = self._preprocess_query(query)
            batch_qids.append(qid)
            batch_queries.append(query)

            if (idx + 1) % self.batch_size == 0 or idx == len(qids) - 1:
                try:
                    results = self.searcher.batch_search(batch_queries, batch_qids, self.n_docs, self.threads)
                    batch_results.update(results)
                except Exception as e:
                    #asddddd
                    print(f"Batch search failed for queries: {batch_queries} with error {e}")
                batch_qids.clear()
                batch_queries.clear()

        return batch_results

    def _preprocess_query(self, query):
        """
        Preprocess the query to ensure it does not exceed the maximum token length.

        Parameters:
        query (str): Input query string.

        Returns:
        str: Preprocessed query that is truncated to 512 tokens if necessary.
        """
        # Tokenize the query
        tokenized_query = tokenizer(query, add_special_tokens=True)

        # Check token length
        if len(tokenized_query["input_ids"]) > 511:
            #print(f"Query exceeds maximum token length (512). Truncating...")
            # Truncate to the first 512 tokens
            truncated_query = tokenizer.decode(tokenized_query["input_ids"][:511], skip_special_tokens=True)
            return truncated_query
        return query
