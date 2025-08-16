import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.n_retreivers.retriever import Retriever


question = Question("Who won the FIFA World Cup after Germany in 2014?")
answers = Answer("")

contexts = [
    Context(id=1, title="2014 FIFA World Cup", text="Germany won the FIFA World Cup in 2014, held in Brazil.", score=0.9),
    Context(id=2, title="FIFA World Cup History", text="The FIFA World Cup is held every four years, with different countries winning each time.", score=0.8),
    Context(id=3, title="World Cup Winners", text="The winners of the FIFA World Cup are celebrated globally.", score=0.7),
]

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