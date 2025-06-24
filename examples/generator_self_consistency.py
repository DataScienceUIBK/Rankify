from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
import torch

from rankify.models.sentence_transformer_reranker import SentenceTransformerReranker

# Define a more ambiguous question and answer placeholder
question = Question("Which city is the cultural capital of the United States?")
answers = Answer("")

# Define contexts with evidence for multiple possible answers
contexts = [
    Context(
        id=1,
        title="New York City Culture",
        text=(
            "New York City is often called the cultural capital of the United States. "
            "It is home to Broadway, world-class museums, and a vibrant arts scene."
        ),
        score=0.95
    ),
    Context(
        id=2,
        title="Los Angeles Arts",
        text=(
            "Los Angeles is known for its film and music industries, and is considered by many to be the entertainment capital of the world."
        ),
        score=0.9
    ),
    Context(
        id=3,
        title="Chicago's Cultural Influence",
        text=(
            "Chicago has a rich history in music, theater, and architecture, and is sometimes referred to as the cultural capital of the Midwest."
        ),
        score=0.95
    ),
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)


reranker = SentenceTransformerReranker(method="sentence_transformer_reranker", model_name="all-MiniLM-L6-v2")

generator = Generator(
    method="self-consistency-rag",
    model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
    backend="huggingface",
    torch_dtype=torch.float16,
    num_samples=7,  # Number of samples for self-consistency
    reranker=reranker,  # Optional reranker for better answer selection
)

generated_answers = generator.generate([doc])
print(generated_answers)  # Output may vary: ["New York City"], ["Los Angeles"], ["Chicago"], etc.