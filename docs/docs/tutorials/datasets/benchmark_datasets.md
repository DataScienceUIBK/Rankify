---
title: "Prebuilt Benchmark Datasets"
---

# 📌 Prebuilt Benchmark Datasets

Rankify provides easy access to standard QA benchmark datasets.

## Available Datasets

| Dataset | Domain | Size |
|---------|--------|------|
| NQ (Natural Questions) | Wikipedia | 3,610 dev |
| TriviaQA | Wikipedia | 11,313 dev |
| WebQuestions | Freebase | 2,032 test |
| SQuAD | Wikipedia | 10,570 dev |
| HotpotQA | Multi-hop | 7,405 dev |
| MS MARCO | Web | 6,980 dev |

## Loading Pre-Retrieved Datasets

```python
from rankify.dataset.dataset import Dataset

# List all available datasets
Dataset.available_dataset()

# Load NQ with BM25 retrieval
dataset = Dataset(
    retriever="bm25",
    dataset_name="nq-dev",
    n_docs=100
)
documents = dataset.download(force_download=False)

print(f"Loaded {len(documents)} questions")
```

## Dataset Sources

- **Full datasets**: [HuggingFace Full](https://huggingface.co/datasets/abdoelsayed/reranking-datasets)
- **Light datasets**: [HuggingFace Light](https://huggingface.co/datasets/abdoelsayed/reranking-datasets-light)

## Different Retrievers

```python
# With different retrievers
bm25_data = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100).download()
dpr_data = Dataset(retriever="dpr", dataset_name="nq-dev", n_docs=100).download()
colbert_data = Dataset(retriever="colbert", dataset_name="nq-dev", n_docs=100).download()
```

## Inspecting Documents

```python
doc = documents[0]

# Question
print(f"Question: {doc.question.question}")

# Gold answers
print(f"Answers: {doc.answers.answers}")

# Retrieved contexts
for i, ctx in enumerate(doc.contexts[:3]):
    print(f"  [{i+1}] Score: {ctx.score:.4f}")
    print(f"      Has answer: {ctx.has_answer}")
    print(f"      Text: {ctx.text[:100]}...")
```
