import os
import requests
import numpy as np
import torch
from tqdm import tqdm
from typing import List
from rankify.utils.retrievers.contriever.index import Indexer
from rankify.utils.retrievers.contriever.normalize_text import normalize

from rankify.utils.retrievers.contriever.data import load_passages
from rankify.utils.retrievers.contriever.contriever import load_retriever
from rankify.dataset.dataset import Document, Context
import pickle
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
import glob
from rankify.utils.pre_defind_models import INDEX_TYPE
import tarfile
import zipfile
from urllib.parse import urlparse

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class ContrieverRetriever:
    """
    Implements **Contriever**, an **unsupervised dense retrieval model** that leverages **FAISS indexing**
    for efficient retrieval of relevant documents.

    Contriever allows **dense passage retrieval** using **bi-encoder embeddings**, making it highly scalable for
    zero-shot and supervised document retrieval tasks.

    References:
        - **Izacard, G., & Grave, E. (2021)**: *Unsupervised Dense Information Retrieval with Contriever*.
          [Paper](https://arxiv.org/abs/2112.09118)

    Attributes:
        model_path (str): Path to the pre-trained **Contriever model**.
        n_docs (int): Number of **top-ranked documents** retrieved for each query.
        batch_size (int): Number of **queries processed simultaneously** for efficiency.
        device (str): Computing device (e.g., `"cuda"` or `"cpu"`).
        index_type (str): The type of index used (`"wiki"` or `"msmarco"`).
        index (Indexer): **FAISS index object** used for efficient retrieval.
        passages (dict): Dictionary mapping **passage IDs** to passage content.
        passage_id_map (dict): Dictionary mapping **numeric IDs** to passage metadata.
        tokenizer (Tokenizer): Tokenizer for processing queries.
        model (torch.nn.Module): **Contriever model** used for encoding queries.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(method='contriever', model="facebook/contriever-msmarco", n_docs=5, index_type="wiki")
        documents = [Document(question=Question("What is deep learning?"))]

        retrieved_documents = retriever.retrieve(documents)
        print(retrieved_documents[0].contexts[0].text)
        ```
    """

    
    EMBEDDINGS_DIR = os.path.join(os.environ.get("RERANKING_CACHE_DIR", "./cache"), 'index', "contriever_embeddings")
    CACHE_DIR = os.path.join(os.environ.get("RERANKING_CACHE_DIR", "./cache") )
    
    def __init__(self, 
                 model: str = "facebook/contriever-msmarco",  #
                 n_docs: int = 100, 
                 batch_size: int = 32, 
                 device: str = "cuda", index_type: str = "wiki", index_folder: str = "") -> None:
        """
        Initializes the **ContrieverRetriever**.

        Args:
            model (str, optional): Path or name of the pre-trained **Contriever model** (default: `"facebook/contriever-msmarco"`).
            n_docs (int, optional): Number of **documents to retrieve** per query (default: `100`).
            batch_size (int, optional): Number of **queries processed per batch** for efficient embedding (default: `32`).
            device (str, optional): Device to run the model on (`"cuda"` or `"cpu"`, default: `"cuda"`).
            index_type (str, optional): The type of index to use (`"wiki"` or `"msmarco"`, default: `"wiki"`).

        Raises:
            ValueError: If the specified **index type** is not supported.
        """
        self.model_path = model
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.device = device
        self.index_type = index_type
        if 'contriever' not in INDEX_TYPE or index_type not in INDEX_TYPE['contriever']:
            raise ValueError(f"Index type '{index_type}' is not supported for Contriever.")

        self.index_config = INDEX_TYPE['contriever'][index_type]
        self.passages_url = self.index_config.get("passages_url", None)
        self.embeddings_url = self.index_config.get("url", None)
        self.index_folder = index_folder

        if self.index_folder:
            self.passage_path = os.path.join(self.index_folder, "corpus.jsonl")
        else:
            self._ensure_index_and_passages_downloaded()

        self.index = self._load_index()
        self.passages = load_passages(self.passage_path)
        self.passage_id_map = {int(x["id"]): x for x in self.passages}
        self.model, self.tokenizer, _ = load_retriever(self.model_path)
        self.model = self.model.to(self.device).eval()
    
    def _ensure_index_and_passages_downloaded(self) -> None:
        """
        Ensures that the FAISS index and passages are downloaded and extracted.
        Handles both `.tar` and `.zip` archives.
        """
        index_folder = os.path.join(self.EMBEDDINGS_DIR, self.index_type)
        
        # Handle embeddings download and extraction
        if self.embeddings_url:
            archive_name = self.clean_filename(self.embeddings_url)
            print(archive_name)
            archive_path = os.path.join(index_folder, archive_name)
            print(archive_path)

            if not os.path.exists(index_folder):
                os.makedirs(index_folder, exist_ok=True)
                print(f"Downloading embeddings for '{self.index_type}'...")
                response = requests.get(self.embeddings_url, stream=True)
                with open(archive_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading Embeddings ({self.index_type})"):
                        f.write(chunk)

                # Extract based on file type
                if archive_name.endswith(".tar"):
                    print(f"Extracting TAR archive for '{self.index_type}'...")
                    with tarfile.open(archive_path, "r") as tar:
                        tar.extractall(path=index_folder)
                elif archive_name.endswith(".zip"):
                    print(f"Extracting ZIP archive for '{self.index_type}'...")
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(index_folder)
                else:
                    raise ValueError(f"Unsupported archive format: {archive_name}")

                os.remove(archive_path)  # Clean up archive file

        # Handle passages download
        if self.passages_url:
            # Extract and clean the file name
            passage_file_name = self.clean_filename(self.passages_url)  # Remove any query parameters
            passage_path = os.path.join(self.CACHE_DIR, passage_file_name)  # Store in the shared cache directory

            if not os.path.exists(passage_path):
                print(f"Downloading passages '{passage_file_name}' into main cache...")
                response = requests.get(self.passages_url, stream=True)
                with open(passage_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading Passages ({passage_file_name})"):
                        f.write(chunk)
            else:
                print(f"Passages '{passage_file_name}' already exist in the main cache.")

            self.passage_path = passage_path  # Set the path for use later

    def clean_filename(self,url):
        """
        Extracts the **file name** from a URL, removing query parameters.

        Args:
            url (str): The URL containing the file name.

        Returns:
            str: The **cleaned file name**.
        """
        parsed_url = urlparse(url)
        return os.path.basename(parsed_url.path)  # Only take the path portion, excluding query params

    def _load_index(self) -> Indexer:
        """
        Loads the **FAISS index** using the **Contriever utility**.

        Returns:
            Indexer: The loaded **FAISS index**.
        """
        index = Indexer(vector_sz=768 , n_subquantizers=0,n_bits=8)  # Update vector size if necessary

        if not self.index_folder:
            index_folder = os.path.join(self.EMBEDDINGS_DIR, self.index_type)

            if self.index_type =="wiki":
                index_folder = os.path.join( index_folder, "wikipedia_embeddings")
        else:
            index_folder = self.index_folder
        
        index_path = os.path.join(index_folder, "index.faiss")
        embeddings_files = glob.glob(os.path.join(index_folder, "*.pkl"))
        embeddings_files = sorted(embeddings_files)

        if os.path.exists(index_path):
            index.deserialize_from(index_folder)
        else:
            self._index_encoded_data(index, embeddings_files)
            index.serialize(index_folder)

        return index
    
    
    def _index_encoded_data(self, index, embedding_files, indexing_batch_size=1000000):
        """
        Loads and indexes **encoded passage embeddings** into FAISS.

        Args:
            index (Indexer): The FAISS **index** to store the embeddings.
            embedding_files (List[str]): List of **files containing precomputed embeddings**.
            indexing_batch_size (int, optional): **Batch size for indexing** (default: `1000000`).
        """
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        #print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids
    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embeds queries using the **Contriever model**.

        Args:
            queries (List[str]): List of **query texts**.

        Returns:
            np.ndarray: **Query embeddings**.
        """
        self.model.eval()
        embeddings = []
        batch_queries = []
        with torch.no_grad():
            for i, query in enumerate(queries):
                query = query.lower()
                query = normalize(query)
                batch_queries.append(query)
                if len(batch_queries) == self.batch_size or i == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_queries, return_tensors="pt", max_length=512, padding=True, truncation=True
                    )
                    for k, v in encoded_batch.items():
                        encoded_batch[k] = v.to(self.device)
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_queries = []
        embeddings = torch.cat(embeddings, dim=0)
        #embeddings = np.vstack(embeddings)
        return embeddings.numpy()
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **passages** for each **document**.

        Args:
            documents (List[Document]): A list of **query documents**.

        Returns:
            List[Document]: Documents with retrieved **contexts**.
        """
        queries = [doc.question.question.replace("?","") for doc in documents]
        query_embeddings = self._embed_queries(queries)
        top_ids_and_scores = self.index.search_knn(query_embeddings, self.n_docs, index_batch_size=self.batch_size) #

        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")):
            top_ids, scores = top_ids_and_scores[i]
            contexts = []
            for doc_id, score in zip(top_ids, scores):
                try:
                    #TODO: change when str type is supported
                    doc_id = str(doc_id).replace("doc", "")
                    passage = self.passage_id_map[int(doc_id)]
                    context = Context(
                        id=int(doc_id),
                        title=passage["title"],
                        text=passage.get("contents", passage.get("text", "")),
                        score=score,
                        has_answer=has_answers(passage["contents"], document.answers.answers, SimpleTokenizer(), regex=False)  # Could be updated with a function to check for answers
                    )
                    contexts.append(context)
                except (IndexError, KeyError):
                    # Log or handle the error, and continue with the next passage
                    continue
            document.contexts = contexts
            #break
        return documents
