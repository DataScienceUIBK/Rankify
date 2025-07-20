# index_manager.py - UPDATED VERSION with ANCE support
import os
import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

class IndexManager:
    """
    Manages downloading, caching, and loading of retrieval indexes.
    
    Handles both prebuilt indexes (wiki, msmarco) and custom user indexes.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.environ.get("RERANKING_CACHE_DIR", "./cache")
        self.index_configs = self._load_index_configs()
    
    def _load_index_configs(self) -> Dict:
        """Load index configuration mappings."""
        return {
            "bm25": {
                "wiki": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bm25_wiki.zip",
                    "prebuilt": "wikipedia-dpr-100w"
                },
                "msmarco": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bm25_index_msmarco.zip?download=true",
                    "prebuilt": None
                }
            },
            "dpr-multi": {
                "wiki": {
                    "prebuilt": "wikipedia-dpr-100w.dpr-multi",
                    "encoder": "facebook/dpr-question_encoder-multiset-base"
                },
                "msmarco": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_multi.zip",
                    "encoder": "facebook/dpr-question_encoder-multiset-base"
                }
            },
            "dpr-single": {
                "wiki": {
                    "prebuilt": "wikipedia-dpr-100w.dpr-single-nq",
                    "encoder": "facebook/dpr-question_encoder-single-nq-base"
                },
                "msmarco": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_single.zip",
                    "encoder": "facebook/dpr-question_encoder-single-nq-base"
                }
            },
            # FIXED: ance-multi should use PREBUILT indices, not download URLs
            "ance-multi": {
                "wiki": {
                    "prebuilt": "wikipedia-dpr-100w.ance-multi",
                    "encoder": "castorini/ance-dpr-question-multi"
                },
                "msmarco": {
                    "prebuilt": "msmarco-v1-passage.ance", 
                    "encoder": "castorini/ance-msmarco-passage"
                }
            },
     
            "bpr-single": {
                "wiki": {
                    "prebuilt": "wikipedia-dpr-100w.bpr-single-nq",
                    "encoder": "castorini/bpr-nq-question-encoder"
                },
                "msmarco": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_bpr.zip",
                    "encoder": "castorini/bpr-nq-question-encoder"
                }
            },
            "bge": {
                "wiki": {
                    "urls": [
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.tar.gz.part1?download=true",
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.tar.gz.part2?download=true",
                    ],
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true",
                    "encoder": "BAAI/bge-large-en-v1.5"
                },
                "msmarco": {
                    "urls": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_bgb.zip?download=true",
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true",
                    "encoder": "BAAI/bge-large-en-v1.5"
                }
            },
            "colbert": {
                "wiki": {
                    "urls": [
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.001?download=true",
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.002?download=true",
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.003?download=true",
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.004?download=true",
                        "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.005?download=true"
                    ],
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true",
                    "model": "colbert-ir/colbertv2.0"
                },
                "msmarco": {
                    "urls": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_colbert.zip?download=true",
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true",
                    "model": "colbert-ir/colbertv2.0"
                }
            },
            "contriever": {
                "wiki": {
                    "url": "https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar",
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true",
                    "model": "facebook/contriever-msmarco",
                    "vector_size": 768
                },
                "msmarco": {
                    "url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_contriever.zip?download=true",
                    "passages_url": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true",
                    "model": "facebook/contriever-msmarco",
                    "vector_size": 768
                }
            },
            "online": {
                "web": {
                    "search_provider": "web",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "requires_api_key": True
                }
            },
            "hyde": {
                "wiki": {
                    "base_retriever": "contriever",
                    "base_model": "facebook/contriever-msmarco",
                    "base_index_type": "wiki",
                    "task": "web search",
                    "llm_model": "gpt-3.5-turbo-0125",
                    "num_generated_docs": 1,
                    "max_token_generated_docs": 512,
                    "temperature": 0.7,
                    "requires_api_key": True
                },
                "msmarco": {
                    "base_retriever": "contriever", 
                    "base_model": "facebook/contriever-msmarco",
                    "base_index_type": "msmarco",
                    "task": "web search",
                    "llm_model": "gpt-3.5-turbo-0125",
                    "num_generated_docs": 1,
                    "max_token_generated_docs": 512,
                    "temperature": 0.7,
                    "requires_api_key": True
                }
            }
        }
    
    def get_index_path(self, method: str, index_type: str, custom_path: str = None) -> str:
        """
        Get the path to the index for a given method and index type.
        
        Args:
            method (str): Retrieval method (e.g., 'bm25', 'dpr-multi', 'ance')
            index_type (str): Index type ('wiki', 'msmarco', or 'custom')
            custom_path (str): Path to custom index if index_type is 'custom'
            
        Returns:
            str: Path to the index
        """
        if custom_path:
            return custom_path
        
        if method not in self.index_configs:
            raise ValueError(f"Unsupported method: {method}")
        
        if index_type not in self.index_configs[method]:
            raise ValueError(f"Unsupported index type '{index_type}' for method '{method}'")
        
        config = self.index_configs[method][index_type]
        
        # If it's a prebuilt index, return the identifier
        if "prebuilt" in config and config["prebuilt"]:
            return config["prebuilt"]
        
        # Otherwise, download and return local path
        return self._ensure_index_downloaded(method, index_type)
    
    def _ensure_index_downloaded(self, method: str, index_type: str) -> str:
        """Download and extract index if not already available."""
        config = self.index_configs[method][index_type]
        url = config["url"]
        
        # Create local directory path
        index_name = f"{method}_{index_type}"
        local_dir = os.path.join(self.cache_dir, "index", index_name)
        
        if not os.path.exists(local_dir):
            print(f"Downloading {method} index for {index_type}...")
            self._download_and_extract(url, local_dir)
        
        return local_dir
    
    def _download_and_extract(self, url: str, destination: str):
        """Download and extract a ZIP file."""
        os.makedirs(destination, exist_ok=True)
        
        zip_name = os.path.basename(url).split("?")[0]
        zip_path = os.path.join(self.cache_dir, "temp", zip_name)
        
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        
        if not os.path.exists(zip_path):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), 
                                desc=f"Downloading {zip_name}"):
                    f.write(chunk)
        
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)
        
        # Clean up ZIP file
        os.remove(zip_path)
        print("Extraction complete.")
    
    def load_id_mapping(self, index_path: str) -> Optional[Dict]:
        """Load ID mapping if available."""
        mapping_path = os.path.join(index_path, "id_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return {v: k for k, v in mapping.items()}  # Reverse mapping
        return None
    
    # def load_corpus(self, index_path: str) -> Dict:
    #     """Load corpus data from index path."""
    #     corpus_file = os.path.join(index_path, "corpus.jsonl")
    #     if not os.path.exists(corpus_file):
    #         return {}
        
    #     corpus = {}
    #     with open(corpus_file, "r", encoding="utf-8") as f:
    #         for line in f:
    #             doc = json.loads(line.strip())
    #             doc_id = doc.get("docid") or doc.get("id")
    #             contents = doc.get("contents", "")
    #             title = doc.get("title", contents[:100] if contents else "No Title")
    #             corpus[str(doc_id)] = {"contents": contents, "title": title}
        
    #     return corpus
    def load_corpus(self, index_path: str) -> Dict:
        """Load corpus data from index path (supports multiple formats)."""
        # Try different corpus file formats in order of preference
        corpus_files = [
            "corpus_metadata.json",  # Your custom format
            "corpus.jsonl",          # Standard JSONL format
            "corpus.json"            # Alternative JSON format
        ]
        
        for corpus_filename in corpus_files:
            corpus_file = os.path.join(index_path, corpus_filename)
            if os.path.exists(corpus_file):
                print(f"📚 Loading corpus from {corpus_filename}")
                
                if corpus_filename.endswith('.jsonl'):
                    # JSONL format (one JSON object per line)
                    return self._load_corpus_jsonl(corpus_file)
                else:
                    # JSON format (single JSON object)
                    return self._load_corpus_json(corpus_file)
        
        print(f"❌ No corpus file found in {index_path}")
        return {}

    def _load_corpus_jsonl(self, corpus_file: str) -> Dict:
        """Load corpus from JSONL format (one JSON object per line)."""
        corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc.get("docid") or doc.get("id")
                contents = doc.get("contents", "")
                title = doc.get("title", contents[:100] if contents else "No Title")
                corpus[str(doc_id)] = {"contents": contents, "title": title}
        return corpus

    def _load_corpus_json(self, corpus_file: str) -> Dict:
        """Load corpus from JSON format (single JSON object)."""
        corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # If it's a dict, iterate through it
            for doc_id, doc_data in data.items():
                if isinstance(doc_data, dict):
                    contents = doc_data.get("contents", "")
                    title = doc_data.get("title", contents[:100] if contents else "No Title")
                else:
                    # If doc_data is a string, use it as contents
                    contents = str(doc_data)
                    title = contents[:100] if contents else "No Title"
                corpus[str(doc_id)] = {"contents": contents, "title": title}
        
        elif isinstance(data, list):
            # If it's a list, iterate through documents
            for doc in data:
                doc_id = doc.get("docid") or doc.get("id")
                contents = doc.get("contents", "")
                title = doc.get("title", contents[:100] if contents else "No Title")
                corpus[str(doc_id)] = {"contents": contents, "title": title}
        
        return corpus
    def download_and_extract_index(self, url: str) -> str:
        """Download and extract index from URL, return local path."""
        # This is used by DenseRetriever pattern
        import tempfile
        import zipfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Download
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(tmp_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024)):
                    f.write(chunk)
            
            # Extract to cache directory
            extract_dir = os.path.join(self.cache_dir, "downloaded_index")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            return extract_dir
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)