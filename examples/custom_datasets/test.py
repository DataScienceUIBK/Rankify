import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
from rankify.dataset.dataset import Dataset
from rankify.models.reranking import Reranking
from rankify.metrics.metrics import Metrics

# Define methods and models
methods = {
    'upr': ["t5-small", ], #"t5-base", "t5-large", "t0-3b", "gpt2", "gpt2-medium", "gpt2-large"
    # 'rankgpt': ['Llama-3.2-1B', 'Llama-3.2-3B', 'Qwen2.5-7B', 'llamav3.1-8b', 'Mistral-7B-Instruct-v0.2', "Mistral-7B-Instruct-v0.3"],
    # 'flashrank': ["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2", "ms-marco-MultiBERT-L-12", "rank-T5-flan", "ce-esci-MiniLM-L12-v2"],
    # 'monot5': ["monot5-base-msmarco", "monot5-base-msmarco-10k", "monot5-large-msmarco" , "monot5-large-msmarco-10k", "monot5-base-med-msmarco", "monot5-3b-med-msmarco",  "monot5-3b-msmarco-10k", "mt5-base-en-msmarco", "ptt5-base-pt-msmarco-10k-v2", "ptt5-base-pt-msmarco-100k-v2", "ptt5-base-en-pt-msmarco-100k-v2", "mt5-base-en-pt-msmarco-v2", "mt5-base-mmarco-v2", "mt5-base-en-pt-msmarco-v1", "mt5-base-mmarco-v1", "ptt5-base-pt-msmarco-10k-v1", "ptt5-base-pt-msmarco-100k-v1", "ptt5-base-en-pt-msmarco-10k-v1", "mt5-3B-mmarco-en-pt", "mt5-13b-mmarco-100k", "monoptt5-small", "monoptt5-base", "monoptt5-large", "monoptt5-3b"],
    # 'rankt5': ['rankt5-base', 'rankt5-large', 'rankt5-3b'],
    # 'listt5': ['listt5-base', 'listt5-3b'],
    # 'inranker': ['inranker-small', 'inranker-base', 'inranker-3b'],
    # 'transformer_ranker': ["mxbai-rerank-xsmall", "mxbai-rerank-base", "mxbai-rerank-large", "bge-reranker-base", "bge-reranker-large", "bge-reranker-v2-m3", "bce-reranker-base","jina-reranker-tiny",  "jina-reranker-turbo", "jina-reranker-base-multilingual","gte-multilingual-reranker-base", ],
    # 'transformer_ranker' : ["cross-encoder-mmarco-mMiniLMv2-L12-H384-v1", "nli-deberta-v3-large", "ms-marco-MiniLM-L-12-v2", "ms-marco-MiniLM-L-6-v2","ms-marco-MiniLM-L-4-v2", "ms-marco-MiniLM-L-2-v2", "ms-marco-TinyBERT-L-2-v2","ms-marco-electra-base", "ms-marco-TinyBERT-L-6", "ms-marco-TinyBERT-L-4","ms-marco-TinyBERT-L-2"],
    # 'llm_layerwise_ranker': ["meta-llama/Llama-3.2-1B","bge-multilingual-gemma2", "bge-reranker-v2-gemma", "bge-reranker-v2-minicpm-layerwise", "bge-reranker-v2.5-gemma2-lightweight"],
    # 'first_ranker':["First-Model"],
    # 'lit5dist': ["LiT5-Distill-base", "LiT5-Distill-large", "LiT5-Distill-xl", "LiT5-Distill-base-v2", "LiT5-Distill-large-v2", "LiT5-Distill-xl-v2"],
    # 'lit5score':   ["LiT5-Score-base", "LiT5-Score-large", "LiT5-Score-xl"],  
    # 'vicuna_reranker' : ["rank_vicuna_7b_v1", "rank_vicuna_7b_v1_noda", "rank_vicuna_7b_v1_fp16", "rank_vicuna_7b_v1_noda_fp16" ],      
    # 'zephyr_reranker': ["rank_zephyr_7b_v1_full"],
    # 'blender_reranker' : ["PairRM"],
    # 'splade_reranker' : ["splade-cocondenser"],
    # 'sentence_transformer_reranker':  ["all-MiniLM-L6-v2", "gtr-t5-base", "gtr-t5-large" , "gtr-t5-xl", "gtr-t5-xxl", "sentence-t5-base" , "sentence-t5-xl","sentence-t5-xxl","sentence-t5-large", "distilbert-multilingual-nli-stsb-quora-ranking", "msmarco-bert-co-condensor", "msmarco-roberta-base-v2" ], 
    # 'colbert_ranker': ['Colbert', 'jina-colbert-v1-en', 'mxbai-colbert-large-v1'],
    # 'monobert': ["monobert-large"],
    # 'llm2vec': ["Meta-Llama-31-8B", "Meta-Llama-3-8B", "Mistral-7B"],
    # 'twolar': ['twolar-xl'], 
    # 'echorank' : ['flan-t5-large', 'flan-t5-xl'],
    # 'incontext_reranker': ['llamav3.1-8b', 'Mistral-7B-Instruct-v0.2'], 

}

# Define datasets
datasets = ['FutureQueryEval_bm25.json',] 

# Log file path
log_file = "log.json"
qrel = "./FutureQueryEval_qrels.txt"
# Load existing results if log file exists
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []
else:
    all_results = []

# Convert existing results into a **set** for quick lookups
existing_results = {(r["retriever"], r["dataset"], r["method"], r["model_name"]) for r in all_results}

# Loop through methods and their models
for method, model_names in methods.items():
    print(f"Processing method: {method}, models: {model_names}")

    for model_name in model_names:
        for dataset_name in datasets:
            key = ("bm25", dataset_name, method, model_name)  # Unique identifier

            # **Skip processing if the model-dataset-method is already in log**
            if key in existing_results:
                print(f"✅ Skipping {method}-{model_name} on {dataset_name} (Already processed)")
                continue

            try:
                print(f"🚀 Running {method}-{model_name} on {dataset_name}...")

                # **Download dataset**
                #dataset = Dataset('bm25', dataset_name, 100)
                #data = dataset.download(force_download=False)
                data = Dataset.load_dataset(dataset_name, 100)
                # **Initialize and run reranker**
                model = Reranking(method=method, model_name=model_name, device = "cuda")
                start_time = time.time()
                model.rank(data)
                end_time = time.time()
                execution_time = end_time - start_time

                # **Calculate metrics**
                metrics = Metrics(data)

                map_cuts = [1, 5, 10, 20, 50, 100]

                mrr_cuts = [1, 5, 10, 20, 50, 100]
                before_ranking_metrics = metrics.calculate_trec_metrics(ndcg_cuts=[1, 5, 10, 20, 50, 100], map_cuts= map_cuts, mrr_cuts = mrr_cuts, qrel=qrel, use_reordered=False)
                after_ranking_metrics = metrics.calculate_trec_metrics(ndcg_cuts=[1, 5, 10, 20, 50, 100], map_cuts= map_cuts, mrr_cuts= mrr_cuts , qrel=qrel, use_reordered=True)

                # **Store results**
                result = {
                    "retriever": "bm25",
                    "dataset": dataset_name,
                    "method": method,
                    "model_name": model_name,
                    "number_documents": str(len(data)),
                    "before_ranking_metrics": before_ranking_metrics,
                    "after_ranking_metrics": after_ranking_metrics,
                    "execution_time": execution_time
                }
                all_results.append(result)

                # **Save results to log file after each iteration**
                with open(log_file, "w") as f:
                    json.dump(all_results, f, indent=4)

                # Print results for debugging
                print(f"✔ Completed {method}-{model_name} on {dataset_name}")
                print(f"Before Ranking Metrics: {before_ranking_metrics}")
                print(f"After Ranking Metrics: {after_ranking_metrics}")
                print("#" * 100)
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                error_msg = f"❌ Error processing {method}-{model_name} on {dataset_name}: {e}"
                print(error_msg)
                with open("log.txt", "a") as log_f:
                    log_f.write(error_msg + "\n")


print("✅ All results saved to:", log_file)