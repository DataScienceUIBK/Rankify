import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.generator.prompt_generator import PromptGenerator

# Define question and answer
question = Question("What is the capital of France?")
answers=Answer("")
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

custom_prompt = (
    "Please answer the following question using only the provided context.\n"
    "Q: {question}\n"
    "Context:\n{contexts}\n"
    "A:"
)


# Initialize Generator (e.g., Meta Llama)
generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface", torch_dtype=torch.float16)

# Generate answer
generated_answers = generator.generate([doc], custom_prompt=custom_prompt)
print(generated_answers)  # Output: ["Paris"]
