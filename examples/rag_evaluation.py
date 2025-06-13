import os

import torch
from vllm import SamplingParams

from rankify.generator.generator import Generator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rankify.dataset.dataset import Dataset, Document, Context, Question, Answer
from rankify.metrics.metrics import Metrics

datasets = ["web_questions-test"]

for name in datasets:
    print("*" * 100)
    print(name)
    dataset = Dataset('bm25', name, 5)
    documents = dataset.download(force_download=False)

    # Limit to a small subset for fast evaluation
    N = 5  # Change this to the number you want to process
    documents = documents[:N]

    print(len(documents[0].contexts), documents[0].answers)
    print(len(documents[0].answers.answers))
    #asdads
    #make the predictions:
    # Initialize Generator (e.g., Meta Llama)
    # generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface", torch_dtype=torch.float16)
    # Define sampling parameters for vllm
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    # Initialize Generator (e.g., Meta Llama)
    generator = Generator(method="basic-rag", model_name='mistralai/Mistral-7B-v0.1', backend="vllm", dtype="float16")

    # Generate answer
    generated_answers = generator.generate(documents=documents, sampling_params=sampling_params)

    # Extract generated answer strings
    #generated_answers = [output[0].outputs[0].text.strip() for output in results]
    


#    results = generator.generate(documents=documents)

    #print(generated_answers)

    metrics = Metrics(documents)

    generation_metrics = metrics.calculate_generation_metrics(generated_answers)
    print(generation_metrics)
    print("#" * 100)
    dataset.save_dataset("webq-bm25-test-small.json", save_text=True)