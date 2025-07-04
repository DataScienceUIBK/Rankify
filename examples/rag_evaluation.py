import os

import torch
from vllm import SamplingParams

from rankify.generator.generator import Generator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rankify.dataset.dataset import Dataset, Document, Context, Question, Answer
from rankify.metrics.metrics import Metrics

datasets = ["web_questions-test"] #nq-test , triviaqa-test

for name in datasets:
    print("*" * 100)
    print(name)
    dataset = Dataset('bm25', name, 5)
    documents = dataset.download(force_download=False)

    # Limit to a small subset for fast evaluation
    N = 2  # Change this to the number you want to process
    documents = documents[:N]

    print(len(documents[0].contexts), documents[0].answers)
    print(len(documents[0].answers.answers))

    # Define sampling parameters for vllm
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=32, n=1, stop=["###", "</s>", "\n\n", "\n","."])# stop=["\n"])

    # Inntitialize Generator (e.g., Meta Llama)
    #qwen 2.5  (1,4,7,)
    #llama 3.2 (1,3,) llama 3.1 (8)
    # gemma 3 (4b)
    # 
    generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="vllm", dtype="float16",  max_model_len=2048)

    # Generate answer
    generated_answers = generator.generate(documents=documents, sampling_params=sampling_params)

    metrics = Metrics(documents)

    print(generated_answers)
    generation_metrics = metrics.calculate_generation_metrics(generated_answers)
    print(generation_metrics)
    print("#" * 100)
    dataset.save_dataset("webq-bm25-test-small.json", save_text=True)