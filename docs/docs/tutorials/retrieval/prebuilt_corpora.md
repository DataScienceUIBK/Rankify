---
title: "Prebuilt Corpora & Indexes"
---

# 📂 Prebuilt Retrieval Corpora & Indexes

Rankify provides prebuilt indices for immediate use, eliminating the need for index building.

## Available Prebuilt Indices

| Corpus | Size | Methods Available |
|--------|------|-------------------|
| Wikipedia | 21M passages | BM25, DPR, ANCE, ColBERT, BGE, Contriever |
| MS MARCO | 8.8M passages | BM25, DPR, ANCE, ColBERT, BGE, Contriever |

## Using Prebuilt Indices

Simply specify the `index_type` parameter:

```python
from rankify.retrievers.retriever import Retriever

# Wikipedia index
wiki_retriever = Retriever(
    method="bm25",
    n_docs=10,
    index_type="wiki"  # Uses Wikipedia
)

# MS MARCO index
msmarco_retriever = Retriever(
    method="bm25",
    n_docs=10,
    index_type="msmarco"  # Uses MS MARCO
)
```

## Index Download Locations

Indices are automatically downloaded to:
```
~/.cache/rankify/
├── bm25/
│   ├── wiki/
│   └── msmarco/
├── dpr/
├── ance/
├── colbert/
├── bge/
└── contriever/
```

## Pre-Retrieved Datasets

For benchmarking, Rankify provides pre-retrieved datasets on Hugging Face:

- 🔗 [Full Dataset](https://huggingface.co/datasets/abdoelsayed/reranking-datasets) (with passage text)
- 🔗 [Light Dataset](https://huggingface.co/datasets/abdoelsayed/reranking-datasets-light) (IDs only)

### Loading Pre-Retrieved Datasets

```python
from rankify.dataset.dataset import Dataset

# List available datasets
Dataset.available_dataset()

# Download a pre-retrieved dataset
dataset = Dataset(
    retriever="bm25",
    dataset_name="nq-dev",
    n_docs=100
)
documents = dataset.download(force_download=False)

print(f"Loaded {len(documents)} documents")
for doc in documents[:3]:
    print(f"Q: {doc.question.question}")
    print(f"Contexts: {len(doc.contexts)}")
```

### Available Benchmark Datasets

| Dataset | Domain | Size |
|---------|--------|------|
| NQ (Natural Questions) | Wikipedia | 3,610 dev |
| TriviaQA | Wikipedia | 11,313 dev |
| WebQuestions | Freebase | 2,032 test |
| CuratedTREC | Web | 694 test |
| SQuAD | Wikipedia | 10,570 dev |
| HotpotQA | Multi-hop | 7,405 dev |
| MS MARCO | Web | 6,980 dev |

## Corpus Statistics

### Wikipedia (English)

- **Source**: December 2018 dump
- **Passages**: 21,015,324
- **Passage length**: ~100 words
- **Format**: Title + text

### MS MARCO

- **Source**: Microsoft Bing queries
- **Passages**: 8,841,823
- **Passage length**: Variable
- **Format**: Text only

## Caching Behavior

```python
import os

# Set custom cache directory
os.environ["RERANKING_CACHE_DIR"] = "/path/to/cache"

# Indices will download on first use
retriever = Retriever(method="bm25", n_docs=10, index_type="wiki")
```

## Next Steps

- [🔎 Custom Datasets](custom_datasets.md) - Build your own indices
- [📊 Evaluation](../evaluation/retrieval_metrics.md) - Evaluate retrieval performance
