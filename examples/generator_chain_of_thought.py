import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator

# Define question and answer placeholder
question = Question("What is the largest Portuguese-speaking city in the world?")
answers = Answer("")

# Define contexts
contexts = [
    Context(id=1, title="Portuguese Language Distribution", text=(
        "Portuguese is spoken in many countries, including Portugal, Brazil, Angola, Mozambique, and others. "
        "Among these, Brazil has the largest population of Portuguese speakers."), score=0.9),
    
    Context(id=2, title="Population Facts", text=(
        "Brazil has the most populous city in these countries."), score=0.8),

    Context(id=3, title="São Paulo vs Lisbon", text=(
        "São Paulo is the most populous city in Brazil. It is widely recognized as the largest Portuguese-speaking city in the world. "
        "In comparison, Lisbon, the capital of Portugal, has a much smaller population than São Paulo."), score=0.95),
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

# Initialize Generator
generator = Generator(
    method="chain-of-thought-rag",
    model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
    backend="huggingface",
    torch_dtype=torch.float16
)

# Generate answer
generated_answers = generator.generate([doc])
print(generated_answers)  # Expected output: ["São Paulo"]
