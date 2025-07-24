import os
import torch

from rankify.generator.generator import Generator
from rankify.dataset.dataset import Dataset
from rankify.metrics.metrics import Metrics
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summary_path = "benchmark_summary.txt"        # one file for the whole script run
header_written = Path(summary_path).exists()  # keep track of whether we already wrote the header

# Datasets to evaluate
DATASETS = [
    "web_questions-test",
    "nq-test",
    "triviaqa-test",
    "strategyqa-test"
]

# RAG methods to evaluate
RAG_METHODS = [
    "basic-rag",
    "chain-of-thought-rag",
    "fid",
    "in-context-ralm",
    "zero-shot",
    "self-consistency-rag"
]

# Models to evaluate (extend this list as needed, all use same config for now)
MODELS = [
    {
        "model_name": 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        "backend": "huggingface",
        "torch_dtype": torch.float16
    },
    # Add more model configs as dicts here
]

FID_MODEL_NAME = "nq_reader_base"
FID_BACKEND = "fid"

N_DOCS = 5  # Number of docs to retrieve per query

# Number of questions to evaluate per dataset (set to None to use all)
N_QUESTIONS = None  # e.g., 1 for one question, 10 for ten, None for all

# Generation parameters for HuggingFace models (pass as kwargs)
generation_kwargs = dict(
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=32,
    num_return_sequences=1,
)

results = {}

for model_cfg in MODELS:
    print("\n" + "#" * 120)
    print(f"Evaluating model: {model_cfg['model_name']}")
    results[model_cfg['model_name']] = {}
    for dataset_name in DATASETS:
        print("=" * 120)
        print(f"Evaluating dataset: {dataset_name}")
        dataset = Dataset('bm25', dataset_name, N_DOCS)
        documents = dataset.download(force_download=False)
        if N_QUESTIONS is not None:
            documents = documents[:N_QUESTIONS]
        metrics = Metrics(documents)
        results[model_cfg['model_name']][dataset_name] = {}

        for rag_method in RAG_METHODS:
            print("-" * 80)
            print(f"Testing RAG method: {rag_method}")

            # Special handling for FiD
            if rag_method == "fid":
                generator = Generator(
                    method="fid",
                    model_name=FID_MODEL_NAME,
                    backend=FID_BACKEND
                )
                # FiD models typically do not use HuggingFace generation kwargs
                try:
                    generated_answers = generator.generate(documents)
                except Exception as e:
                    print(f"Error with method {rag_method} on dataset {dataset_name}: {e}")
                    generated_answers = [""] * len(documents)
            else:
                generator = Generator(
                    method=rag_method,
                    **model_cfg  # Pass all model parameters as kwargs
                )
                try:
                    generated_answers = generator.generate(
                        documents=documents,
                        **generation_kwargs
                    )
                except Exception as e:
                    print(f"Error with method {rag_method} on dataset {dataset_name}: {e}")
                    generated_answers = [""] * len(documents)

            print(f"Generated answers ({rag_method}): {generated_answers}")
            generation_metrics = metrics.calculate_generation_metrics(generated_answers)
            print(f"Generation metrics ({rag_method}): {generation_metrics}")
            results[model_cfg['model_name']][dataset_name][rag_method] = generation_metrics
            mode = "a"  # always append
            with open(summary_path, mode, encoding="utf-8") as f:
                if not header_written:
                    f.write("Benchmark Results Summary:\n")
                    header_written = True

                f.write(f"\nModel: {model_cfg['model_name']}\n")
                f.write(f"  Dataset: {dataset_name}\n")
                for rag_method, metrics_dict in results[model_cfg['model_name']][dataset_name].items():
                    f.write(f"    {rag_method}: {metrics_dict}\n")

            print(f"Intermediate summary appended to {summary_path}")
            print("=" * 120)
        # Optionally save intermediate results
        dataset.save_dataset(f"{dataset_name}-bm25-eval.json", save_text=True)


# Save summary to file instead of printing
summary_path = "benchmark_summary_final.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Benchmark Results Summary:\n")
    for model_name, model_results in results.items():
        f.write(f"\nModel: {model_name}\n")
        for dataset_name, rag_results in model_results.items():
            f.write(f"  Dataset: {dataset_name}\n")
            for rag_method, metrics_dict in rag_results.items():
                f.write(f"    {rag_method}: {metrics_dict}\n")

print(f"\nSummary written to {summary_path}")

# Print summary
print("\n\nBenchmark Results Summary:")
for model_name, model_results in results.items():
    print(f"\nModel: {model_name}")
    for dataset_name, rag_results in model_results.items():
        print(f"  Dataset: {dataset_name}")
        for rag_method, metrics_dict in rag_results.items():
            print(f"    {rag_method}: {metrics_dict}")