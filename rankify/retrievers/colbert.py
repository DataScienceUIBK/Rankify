
import os
import requests
import zipfile
from tqdm import tqdm
from typing import List
from rankify.utils.retrievers.colbert.colbert.infra import Run, RunConfig, ColBERTConfig
from rankify.utils.retrievers.colbert.colbert import Searcher
from rankify.dataset.dataset import Document, Context
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from rankify.utils.pre_defind_models import INDEX_TYPE
from urllib.parse import urlparse


class ColBERTRetriever:
    """
    Implements **ColBERT**, a **late interaction** retrieval model that efficiently scores and ranks 
    passages based on **token-wise interactions**.


    ColBERT enables **scalable and efficient document retrieval** by using **compressed representations** 
    and **approximate nearest neighbor search**. This retriever leverages Pyserini's ColBERT implementation.

    References:
        - **Khattab, O. & Zaharia, M. (2020)**: *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*.
          [Paper](https://arxiv.org/abs/2004.12832)

    Attributes:
        model (str): Path or identifier for the **ColBERT model**.
        index_type (str): The type of index (`wiki` or `msmarco`) used for retrieval.
        n_docs (int): Number of **top documents** to retrieve per query.
        tokenizer (SimpleTokenizer): Tokenizer for processing queries and answers.
        index_config (dict): Configuration dictionary containing **index details**.
        passages_url (str): URL from which the **passage dataset** can be downloaded.
        passages_file (str): Path to the **local passage dataset file**.
        searcher (Searcher): **ColBERT search engine** initialized for retrieval.
        passages (dict): Dictionary mapping **passage IDs** to their **text and titles**.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(method='colbert', model="colbert-ir/colbertv2.0", n_docs=5, index_type="wiki")
        documents = [Document(question=Question("Who wrote Hamlet?"))]

        retrieved_documents = retriever.retrieve(documents)
        print(retrieved_documents[0].contexts[0].text)
        ```
    """
    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, model: str ="colbert-ir/colbertv2.0" , index_type: str = "wiki", n_docs: int = 10, batch_size=64,
                 index_folder: str = ""):
        """
        Initializes the **ColBERTRetriever**.

        Args:
            model (str, optional): Path or **identifier for the ColBERT model** 
                (default: `"colbert-ir/colbertv2.0"`).
            index_type (str, optional): **Type of index** (`wiki` or `msmarco`) used for retrieval 
                (default: `"wiki"`).
            n_docs (int, optional): **Number of top documents** to retrieve per query (default: `10`).
            batch_size (int, optional): **Number of queries processed in a single batch** (default: `64`).

        Raises:
            ValueError: If the specified **index type** is not supported.
        """
        self.model = model
        self.index_type = index_type
        self.n_docs = n_docs
        self.tokenizer = SimpleTokenizer()
        self.index_folder = index_folder

        if 'colbert' not in INDEX_TYPE or index_type not in INDEX_TYPE['colbert']:
            raise ValueError(f"Index type '{index_type}' is not supported for ColBERT.")

        self.index_config = INDEX_TYPE['colbert'][index_type]
        self.passages_url = self.index_config.get("passages_url")
        if not self.index_folder:
            self._ensure_index_and_passages_downloaded()
        else:
            self.passages_file = os.path.join(self.index_folder, "passages.tsv")
        self._initialize_searcher()

        self.passages = {}
        with open(self.passages_file, "r", encoding="utf-8") as f:
            # Skip the header line
            next(f)
            for line in f:
                pid, text, title = line.strip().split("\t")
                self.passages[pid] = {"text": text, "title": title}
    def extract_filename_from_url(self, url):
        """
        Extracts the **filename** from a given **URL**, ignoring query parameters.

        Args:
            url (str): The **URL** to extract the filename from.

        Returns:
            str: The **cleaned filename**.
        """
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)  # Extract the path without query parameters
        return filename
    def _ensure_index_and_passages_downloaded(self):
        """
        Ensures that the **ColBERT index** and **passage datasets** are **downloaded and extracted**.

        Raises:
            RuntimeError: If the **index download** or **extraction** fails.
        """
        index_folder = os.path.join(self.CACHE_DIR, "index", "colbert")
        os.makedirs(index_folder, exist_ok=True)

        if not os.path.exists(os.path.join(index_folder, self.index_type)):
            # Handle multi-part or single index
            if isinstance(self.index_config['urls'], list):  # Multi-part index
                for url in self.index_config['urls']:
                    filename = self.extract_filename_from_url(url)  # Correctly extract filename
                    zip_path = os.path.join(index_folder, filename)

                    if not os.path.exists(zip_path):  # Check if the file already exists
                        self._download_file(url, zip_path)

                self._extract_multi_part_zip(index_folder)
            else:
                zip_path = os.path.join(index_folder, "index.zip")
                if not os.path.exists(zip_path):
                    self._download_file(self.index_config['urls'], zip_path)
                self._extract_zip_files(index_folder)
        
        # Download passages
        passages_file = os.path.join(self.CACHE_DIR, self.extract_filename_from_url(self.passages_url))
        print(passages_file)
        if not os.path.exists(passages_file):
            self._download_file(self.passages_url, passages_file)
        self.passages_file = passages_file

        

    

    def _download_file(self, url, save_path):
        """
        Downloads a file from the **specified URL**.

        Args:
            url (str): URL of the **file to download**.
            save_path (str): Local path to **save the downloaded file**.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)

    def _extract_zip_files(self, folder):
        """
        Extracts **ZIP files** from the specified folder.

        Args:
            folder (str): The **folder** containing ZIP files.
        """
        zip_files = [f for f in os.listdir(folder) if f.endswith(".zip")]
        for zip_file in zip_files:
            with zipfile.ZipFile(os.path.join(folder, zip_file), "r") as zip_ref:
                zip_ref.extractall(folder)
    def _extract_multi_part_zip(self, folder):
        """
        Extracts **multi-part ZIP files** by **combining** and **decompressing** them.
        """
        zip_parts = sorted([f for f in os.listdir(folder) if f.startswith("wiki.zip.")], key=lambda x: int(x.split('.')[-1]))
        combined_zip_path = os.path.join(folder, "combined.zip")

        print(f"Combining {len(zip_parts)} parts into {combined_zip_path}...")
        with open(combined_zip_path, "wb") as combined:
            for part in zip_parts:
                part_path = os.path.join(folder, part)
                print(f"Adding {part_path} to {combined_zip_path}...")
                with open(part_path, "rb") as part_file:
                    combined.write(part_file.read())

        print(f"Extracting combined ZIP file: {combined_zip_path}...")
        try:
            with zipfile.ZipFile(combined_zip_path, "r") as zip_ref:
                zip_ref.extractall(folder)
        except zipfile.BadZipFile:
            print("Error: Combined file is not a valid ZIP. Please verify the downloaded parts.")
            raise
        finally:
            os.remove(combined_zip_path)  # Clean up combined file after extraction



    def _initialize_searcher(self):
        """
        Initializes the **ColBERT search engine** for document retrieval.
        """
        #print(self.index_config['passages_url'])
        #self.co
        with Run().context(RunConfig(nranks=1, experiment="colbert")):
            config = ColBERTConfig(
                root=self.index_folder if self.index_folder else os.path.join(self.CACHE_DIR, "index", "colbert", self.index_type),
                index_path=self.index_folder if self.index_folder else self.index_config['passages_url'],
                collection=self.passages_file
            )
            #print(config.collection)
            #config.collection = self.passages_file
            self.searcher = Searcher(index=config.root, config=config)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **relevant contexts** for each **document** in the input list.

        Args:
            documents (List[Document]): A list of **Document** objects containing queries.

        Returns:
            List[Document]: Documents with updated **contexts** after retrieval.
        """
        # Iterate over each document and perform retrieval
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")):
            query = document.question.question
            results = self.searcher.search(query, k=self.n_docs)  # Retrieve results as a tuple

            contexts = []
            document_ids, ranks, scores = results  # Unpack the results tuple

            for pid, rank, score in zip(document_ids, ranks, scores):
                pid = str(pid)  # Ensure the ID is a string
                if pid in self.passages:  # Check if the ID exists in the passages dictionary
                    passage = self.passages[pid]
                    #Todo: also here Change type to str of Context
                    context = Context(
                        id=pid,
                        title=passage["title"],
                        text=passage["text"],
                        score=score,  # Use the score from the results
                        has_answer=has_answers(passage["text"], document.answers.answers, self.tokenizer)
                    )
                    contexts.append(context)

            document.contexts = contexts

        return documents
