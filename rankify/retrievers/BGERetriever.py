import os
import requests
import h5py
import pickle
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rankify.dataset.dataset import Document, Context
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from typing import List
from rankify.utils.pre_defind_models import INDEX_TYPE
import zipfile
from urllib.parse import urlparse

import shutil, gzip

class BGERetriever:
    """
    Implements **BGE Retriever**, a **passage retrieval system** using **precomputed embeddings** and **FAISS indexing**.


    BGE Retriever efficiently retrieves relevant documents by **leveraging dense representations** and **FAISS-based approximate nearest neighbor (ANN) search**.
    It supports **multi-part ZIP extraction** and **passage retrieval** for large-scale information retrieval tasks.

    References:
        - **Xiao et al. (2024)**: *C-Pack: Packed Resources for General Chinese Embeddings*.
          [Paper](https://dl.acm.org/doi/10.1145/3625555.3662062)

    Attributes:
        model_name (str): The embedding model used for generating **dense representations**.
        n_docs (int): Number of **top-ranked documents** retrieved per query.
        batch_size (int): Number of **queries processed per batch** for efficiency.
        device (str): The computing device (`"cuda"` or `"cpu"`).
        index_type (str): The type of **retrieval index** (`"wiki"` or `"msmarco"`).
        index_folder (str): Path where the **FAISS index** and supporting files are stored.
        doc_ids (List[str]): List of **document IDs** mapped to stored embeddings.
        doc_texts (dict): Dictionary mapping **document IDs** to text and title.
        passage_path (str): Path to the downloaded **passage file**.
        index (faiss.IndexFlatIP): The **FAISS index** used for nearest neighbor search.
        model (AutoModel): The transformer model used for **query encoding**.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the embedding model.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever

        retriever = Retriever(method="bge", model="BAAI/bge-large-en-v1.5", n_docs=5, index_type="wiki")
        documents = [Document(question=Question("Who discovered gravity?"))]

        retrieved_documents = retriever.retrieve(documents)
        print(retrieved_documents[0].contexts[0].text)
        ```
    """
    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, model="BAAI/bge-large-en-v1.5", n_docs=10, batch_size=32, device="cuda", index_type="wiki",
                 index_folder=''):
        """
        Initializes the **BGE Retriever**.

        Args:
            model (str, optional): Name of the **embedding model** (default: `"BAAI/bge-large-en-v1.5"`).
            n_docs (int, optional): Number of **top documents** to retrieve per query (default: `10`).
            batch_size (int, optional): Number of **queries processed per batch** (default: `32`).
            device (str, optional): Computing device (`"cuda"` or `"cpu"`, default: `"cuda"`).
            index_type (str, optional): The **type of retrieval index** (`"wiki"` or `"msmarco"`, default: `"wiki"`).

        Raises:
            ValueError: If the specified `index_type` is not supported.
        """
        self.model_name = model
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.device = device
        self.index_type = index_type

        if 'bge' not in INDEX_TYPE or index_type not in INDEX_TYPE['bge']:
            raise ValueError(f"Index type '{index_type}' is not supported for BGERetriever.")

        self.index_config = INDEX_TYPE['bge'][index_type]
        if not index_folder:
            self.index_folder = os.path.join(self.CACHE_DIR, "index", f"bgb_index_{index_type}")
            self.passages_url = self.index_config.get("passages_url")
            self.passage_path = None

            self._ensure_index_and_passages_downloaded()
            self.doc_ids = self._load_document_ids()
            self.doc_texts = self._load_tsv()
            self.build_faiss_index_on_disk()
        else:
            self.index_folder = index_folder
            self.passage_path = os.path.join(self.index_folder, 'passages.tsv')
            self.doc_ids = self._load_document_ids()
            self.doc_texts = self._load_tsv()
            self.build_faiss_index_on_disk()

        # Load model and tokenizer
        self.model_hf = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extract_filename_from_url(self, url):
        parsed_url = urlparse(url)
        return os.path.basename(parsed_url.path)

    def _ensure_index_and_passages_downloaded(self):
        """
        Ensures that the necessary **index files** and **passages** are **downloaded and extracted**.
        """
        os.makedirs(self.index_folder, exist_ok=True)
        required_files = [
            os.path.join(self.index_folder, "bge_doc_ids.pkl"),
            os.path.join(self.index_folder, "bge_embeddings.h5"),
        ]
        if not all(os.path.exists(f) for f in required_files):
            if not os.path.exists(os.path.join(self.index_folder, self.index_type)):
                if isinstance(self.index_config['urls'], list):
                    for url in self.index_config['urls']:
                        filename = self.extract_filename_from_url(url)
                        zip_path = os.path.join(self.index_folder, filename)
                        if not os.path.exists(zip_path):
                            self._download_file(url, zip_path)
                    self._extract_multi_part_zip(self.index_folder)
                else:
                    zip_path = os.path.join(self.index_folder, "index.zip")
                    if not os.path.exists(zip_path):
                        self._download_file(self.index_config['urls'], zip_path)
                    self._extract_zip_files(self.index_folder)

        if self.passages_url:
            passage_file_name = self.extract_filename_from_url(self.passages_url)
            passage_path = os.path.join(self.CACHE_DIR, passage_file_name)
            if not os.path.exists(passage_path):
                self._download_file(self.passages_url, passage_path)
            self.passage_path = passage_path

    def _download_file(self, url, save_path):
        """
        Downloads a **file** from a given **URL** and saves it to a **specified path**.

        Args:
            url (str): The URL of the **file to download**.
            save_path (str): The **path** where the file will be stored.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)

    def _extract_zip_files(self, folder):
        """
        Extracts all `.zip` files in the specified folder and places the extracted files in the same directory.

        This method processes each ZIP archive found in the folder, extracts its contents directly 
        into the `index_folder`, and removes the ZIP file after extraction. 

        Args:
            folder (str): The directory containing the ZIP files to be extracted.

        Raises:
            zipfile.BadZipFile: If any of the ZIP archives are corrupted or not valid.
        """
        zip_files = [f for f in os.listdir(folder) if f.endswith(".zip")]
        target_folder = folder  # Ensure it extracts inside the same folder

        for zip_file in zip_files:
            zip_path = os.path.join(folder, zip_file)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    # Get relative path inside ZIP
                    filename = os.path.basename(member)
                    
                    if not filename:  # Skip directories
                        continue

                    # Extract directly into `bgb_index_msmarco`, ignoring internal structure
                    extracted_path = os.path.join(target_folder, filename)

                    with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)

            print(f"Extracted {zip_file} directly into {target_folder}")
            os.remove(zip_path)


    def _extract_multi_part_zip(self, folder):
        """
        Extracts multi-part `.tar.gz` archives from the specified folder and unpacks them.

        This method first merges multiple `.tar.gz` parts into a single compressed archive.
        It then decompresses the archive and extracts its contents into the given folder. 
        The method ensures that all temporary files (e.g., split parts and the merged archive)  are removed after extraction.

        Args:
            folder (str): The directory containing the multi-part archive files.

        Raises:
            RuntimeError: If the combined archive is not a valid `.tar.gz` file.
        """
        zip_parts = sorted([f for f in os.listdir(folder) if f.startswith("bgb_index.tar.")],
                           key=lambda x: x.split('.')[-1])
        combined_zip_path = os.path.join(folder, "combined.tar.gz")

        with open(combined_zip_path, "wb") as combined:
            for part in zip_parts:
                with open(os.path.join(folder, part), "rb") as part_file:
                    #combined.write(part_file.read())
                    shutil.copyfileobj(part_file, combined)

        try:
            tar_file = combined_zip_path.replace(".gz", "")
            with gzip.open(combined_zip_path, "rb") as f_in, open(tar_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            shutil.unpack_archive(tar_file, folder)
            os.remove(tar_file)  # Remove tar file
        except zipfile.BadZipFile:
            raise RuntimeError("Error: Combined file is not a valid ZIP archive.")
        finally:
            os.remove(combined_zip_path)
            for part in zip_parts:
                os.remove(os.path.join(folder, part))

    def build_faiss_index_on_disk(self):
        """
        Builds or loads a **FAISS index** for efficient **document retrieval**.
        """
        print("Handling FAISS index...")

        index_path = os.path.join(self.index_folder, "faiss_index.bin")
        embeddings_path = os.path.join(self.index_folder,  "bge_embeddings.h5")

        if os.path.exists(index_path):
            print(f"Loading existing FAISS index for '{self.index_type}' from {index_path}...")
            self.index = faiss.read_index(index_path)
        else:
            print(f"Building FAISS index for '{self.index_type}' from embeddings at {embeddings_path}...")
            with h5py.File(embeddings_path, "r") as f:
                embeddings = f["embeddings"][:].astype(np.float32)

            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)

            chunk_size = 50000
            for start in tqdm(range(0, embeddings.shape[0], chunk_size), desc="Adding embeddings to FAISS index"):
                end = min(start + chunk_size, embeddings.shape[0])
                chunk = embeddings[start:end]
                self.index.add(chunk)

            print(f"Saving FAISS index for '{self.index_type}' to {index_path}...")
            faiss.write_index(self.index, index_path)

        print(f"FAISS index loaded with {self.index.ntotal} embeddings.")

    def _load_document_ids(self):
        """
        Loads **document IDs** for the **retrieval index**.

        Returns:
            List[str]: A list of **document IDs**.
        """
        doc_ids_path = os.path.join(self.index_folder,  "bge_doc_ids.pkl")
        print(f"Loading document IDs for '{self.index_type}' from {doc_ids_path}...")
    
        with open(doc_ids_path, "rb") as f:
                doc_ids = []
                while True:
                    try:
                        doc_ids.extend(pickle.load(f))
                    except EOFError:
                        break

        print(f"Loaded {len(doc_ids)} document IDs for '{self.index_type}'.")
        return doc_ids

    def _load_tsv(self):
        doc_texts = {}
        with open(self.passage_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                doc_id, passage, title = line.strip().split("\t")
                doc_texts[doc_id] = {"text": passage, "title": title}
        return doc_texts

    def _encode_queries(self, queries):
        """
        Encodes **queries** into **dense vector representations** using the **embedding model**.

        Args:
            queries (List[str]): List of **query texts**.

        Returns:
            np.ndarray: **Query embeddings**.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), self.batch_size), desc="Encoding queries"):
                batch = queries[i:i + self.batch_size]
                tokenized = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_output = self.model_hf(**tokenized)
                embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieves **contexts** for each **document** based on its **question**.

        Args:
            documents (List[Document]): A list of **Document objects** containing queries.

        Returns:
            List[Document]: The list of **Document objects** updated with retrieved **contexts**.
        """
        queries = [doc.question.question for doc in documents]
        query_embeddings = self._encode_queries(queries)


        # Ensure dimensions match
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Dimensionality mismatch: Query embeddings have dimension {query_embeddings.shape[1]}, "
                f"but FAISS index has dimension {self.index.d}."
            )


        
        # Initialize empty results for batch processing
        all_distances = []
        all_indices = []

        # Batch processing for FAISS search
        for start_idx in tqdm(range(0, len(query_embeddings), self.batch_size), desc="Batch FAISS Search"):
            end_idx = min(start_idx + self.batch_size, len(query_embeddings))
            batch_query_embeddings = query_embeddings[start_idx:end_idx]

            distances, indices = self.index.search(batch_query_embeddings, self.n_docs)
            all_distances.append(distances)
            all_indices.append(indices)

        # Concatenate all batch results
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)

        # Process the results for each document
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")):
            contexts = []
            for dist, idx in zip(all_distances[i], all_indices[i]):
                doc_id = self.doc_ids[idx]
                doc_text = self.doc_texts.get(str(doc_id), {"text": "Text not found", "title": "No Title"})
                context = Context(
                    id=doc_id,
                    title=doc_text["title"],
                    text=doc_text["text"],
                    score=dist,
                    has_answer=has_answers(doc_text["text"], document.answers.answers, SimpleTokenizer())
                )
                contexts.append(context)
            document.contexts = contexts

        return documents
