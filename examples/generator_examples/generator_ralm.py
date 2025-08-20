from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator

# Sample question and contexts
question = Question("What is the capital of France?")
answers=Answer('')
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Create a Document
doc = Document(question=question, answers= answers, contexts=contexts)

# Initialize Generator (e.g., Meta Llama, with huggingface backend)
generator = Generator(method="in-context-ralm", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface")

# Generate answer
generated_answers = generator.generate([doc])
print(generated_answers)  # Output: ["Paris"]
