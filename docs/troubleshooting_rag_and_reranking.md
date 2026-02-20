# Troubleshooting RAG and reranking in Rankify

This page is a practical checklist for users who want to debug retrieval, reranking, and RAG pipelines built with Rankify.

It has two goals:

1. Help you quickly locate the most likely source of failure.
2. Help you prepare a high quality bug report or discussion so that maintainers can reproduce the issue.

You do not need to understand every internal detail of Rankify to use this page. You just need to match your symptoms, and follow the checklists.

---

## 1. How to use this page

1. **Find your failure pattern** in Section 2.  
2. **Walk through the checklist** for that pattern and see if the problem is in configuration, data, or evaluation.  
3. If you still cannot explain the behavior, go to **Section 3** and copy the issue template when opening a GitHub issue or discussion.

---

## 2. Common failure patterns

### 2.1 “Good retrieval, bad reranking”

**Symptom**

- When you inspect the *raw retrieved documents*, they already look relevant.
- After reranking, the top-k results look worse, or the final answers from the RAG model degrade.

**Checklist**

- **K1. Confirm which retriever you are using**

  - What is the retriever class or type? (e.g. BM25, dense, hybrid, ColBERT, etc.)
  - Are you sure the reranker is applied **after** the expected retriever, and not on a different one?

- **K2. Check the reranker configuration**

  - Which reranker class are you using? (e.g. `APIReranker`, `FlashRank`, `SentenceTransformerReranker`, etc.)
  - Is the `model_name` correct and available on your system or API provider?
  - Is the scoring direction correct?  
    For example, higher scores should mean “better”, and Rankify should sort in that direction.

- **K3. Look at scores and text together**

  - Print a small table with: `score`, `document_id`, and `first 1–2 sentences`.
  - Check if higher scores really correspond to more relevant documents.
  - If scores look random or inverted, the issue is probably inside the reranker configuration or model.

- **K4. Check batch size and truncation**

  - Are your documents being truncated too aggressively before reranking?
  - Is the batch size so large that the reranker might be running out of memory or timing out, causing silent failures?

If K1–K4 look correct and the reranker still degrades performance, capture a minimal example for Section 3.

---

### 2.2 “Bad retrieval, reranker cannot rescue it”

**Symptom**

- Even before reranking, the retrieved documents are mostly off-topic.
- The reranker sometimes moves slightly better documents upwards, but the final answers are still poor.

**Checklist**

- **R1. Verify the dataset and split**

  - Which dataset are you using? (HF hub name, local file, or custom dataset)
  - Which **split** is used for indexing (e.g. `train`, `validation`, `test`, or a custom split)?
  - Are you sure you are not indexing an empty or wrong split?

- **R2. Check index type and location**

  - Which index type are you using (e.g. FAISS, BM25, ColBERT index, etc.)?
  - Is the index created with the same embeddings or tokenization as your current retriever?

- **R3. Match query fields and document fields**

  - Are queries and documents using the same language, casing, and normalization?
  - If you use custom fields, are you passing the right field names into the retriever?

- **R4. Try a simpler baseline**

  - Run a very simple retriever (for example BM25) on a small subset.
  - If BM25 already fails badly, the issue is probably in the dataset alignment, not in Rankify itself.

If retrieval is bad, even the best reranker cannot fix it consistently. Focus first on R1–R4.

---

### 2.3 “Configuration or wiring mistakes”

**Symptom**

- Changing settings does not seem to have any effect.
- The pipeline looks correct on paper, but logs suggest that a different component is being used.

**Checklist**

- **C1. Confirm which configuration file or code path is active**

  - Are you editing the same config file that your script or notebook actually loads?
  - If you are using environment variables, are they correctly picked up?

- **C2. Check for stale artifacts**

  - Are you reusing an old index or cache built with a different configuration?
  - Did you change the dataset or embeddings without rebuilding the index?

- **C3. Check logging**

  - Enable or add simple logging to print which retriever, reranker, and generator are actually instantiated.
  - Verify that the classes and model names printed match what you think you are using.

- **C4. Run a tiny, fully explicit example**

  - In a notebook or script, build the retriever, reranker, and generator in a single cell with explicit arguments.
  - Run on 3–5 queries only, so that you can print everything and visually inspect the behavior.

---

### 2.4 “Metric says improvement, answers feel worse”  

**Symptom**

- After a change, evaluation metrics (e.g. MRR, nDCG, RAG metrics) improve.
- When you manually inspect answers, they feel less helpful or less faithful.

**Checklist**

- **E1. Confirm which metric you are optimizing**

  - Is the metric for **retrieval**, **reranking**, or **RAG answers**?
  - Retrieval metrics do not always correlate with human-perceived answer quality.

- **E2. Check the evaluation dataset**

  - Are labels strictly about topical relevance, or do they also encode answer quality or reasoning depth?
  - Are you mixing different task types in one evaluation run?

- **E3. Align your evaluation with your real goal**

  - If your product cares about faithful, grounded answers, consider evaluation that checks *answer correctness* or *hallucination rate*, not only relevance.
  - If you change the reranker or RAG method, verify evaluation both at the retrieval layer and at the answer layer.

If E1–E3 reveal a mismatch, you may need a different evaluation setup rather than a different model.

---

## 3. What to include when opening an issue

When you open a GitHub issue or discussion about RAG or reranking, including the following information will save a lot of time for everyone.

You can copy this list and fill in the blanks.

### 3.1 Dataset and index

- **Dataset source**  
  Example: “HF dataset `my_org/my_dataset`, split `train`” or “Local JSONL file with fields `id`, `text`”.

- **Index type and location**  
  Example: “Dense FAISS index stored at `./indices/my_index.faiss`”.

- **Preprocessing**  
  Example: “Documents lowercased, stopword removal enabled, max length 512 tokens”.

### 3.2 Retrieval configuration

- **Retriever class and key parameters**  
  Example: “Using dense retriever with model `bge-base-en` and `k = 50`”.

- **Query side processing**  
  Example: “Queries are user questions in English, no extra preprocessing”.

- **Any hybrid or multi-stage retrieval**  
  Example: “First BM25 (k = 100), then dense reranking to top 10”.

### 3.3 Reranking configuration

- **Reranker class and model**  
  Example: “Using `SentenceTransformerReranker` with `model_name = 'sentence-transformers/ms-marco-MiniLM-L-6-v3'`”.

- **Top-k before and after reranking**  
  Example: “Retrieve top 50, rerank to top 5”.

- **Observed behavior**  
  Describe what looked wrong (e.g. “many obviously off-topic documents ranked at the top”).

### 3.4 Generation / RAG configuration

- **RAG method**  
  Example: “Using basic RAG method” or “Using FiD RAG method”.

- **Model backend**  
  Example: “OpenAI API, `gpt-4.x`” or “HuggingFace local model `my-org/my-model`”.

- **Prompting style**  
  Example: “Simple ‘Answer the question using the context’ style prompt”.

- **Number of context documents**  
  Example: “Passing top 5 retrieved documents into the generator”.

### 3.5 Evaluation setup

- **Metrics used**  
  Example: “Using RAG metrics from `tutorials/rag/evaluation.md` with `exact_match` and `F1`”.

- **Labeled data**  
  Example: “50 questions with gold answers written by domain experts”.

- **What improved and what got worse**  
  Example: “nDCG@10 improved from 0.42 to 0.48, but manual inspection shows more hallucinations”.

### 3.6 Environment

- **Rankify version**  
  Example: pip version or git commit hash.

- **Python version**

- **OS / hardware**  
  Example: “Ubuntu 22.04, 1 GPU (A100, 40GB)” or “macOS, CPU only”.

### 3.7 Minimal reproducible example

Whenever possible, include a **small code snippet** that:

1. Loads or builds the dataset.
2. Constructs the retriever, reranker, and RAG method you are using.
3. Runs on **one or a few** example queries.
4. Prints both scores and the first part of each document, so that maintainers can see exactly what is happening.

If your data is private, you can still construct a synthetic mini-dataset that reproduces the same failure pattern.

---

If you follow these checklists and include the information from Section 3, it will be much easier for Rankify maintainers and other users to help you debug your RAG and reranking pipelines.
