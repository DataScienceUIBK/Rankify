import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.n_retreivers.retriever import Retriever

# Define a question that requires reasoning and possibly external search
question = Question("Which city hosted the Summer Olympics after Athens in 2004?")
answers = Answer("")

# Initial contexts do not contain the answer directly
contexts = [
    Context(id=1, title="Athens Olympics", text=(
        "The 2004 Summer Olympics were held in Athens, Greece."), score=0.9),
    Context(id=2, title="Olympic History", text=(
        "The Summer Olympics are held every four years in different cities around the world."), score=0.8),
    Context(id=3, title="Olympic Host Cities", text=(
        "Cities compete to host the Olympic Games, which are awarded by the International Olympic Committee."), score=0.7),
]

# Construct document
docs = [Document(question=question, answers=answers, contexts=contexts)]

retriever = Retriever(method='bm25', n_docs=5, index_type='wiki')
retrieved_documents = retriever.retrieve(docs)
for i, doc in enumerate(retrieved_documents):
     print(f"\nDocument {i+1}:")
     print(doc)

# Initialize retriever (example: BM25, can be any retriever compatible with your pipeline)
#retriever = Retriever(method="bm25", n_docs=3, index_type="wiki")
# Initialize Generator for ReAct RAG
generator = Generator(
    method="react-rag",
    model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
    backend="huggingface",
    torch_dtype=torch.float16,
    retriever=retriever
)

# Generate answer
generated_answers = generator.generate([doc])
print(generated_answers)  # Expected output: ["Beijing"] (the city that hosted after Athens)