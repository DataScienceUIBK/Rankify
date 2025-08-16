import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.n_retreivers.retriever import Retriever


test_docs = [
    Document(
        question=Question("Who won the FIFA World Cup in 2018?"),
        answers=Answer(""),
        contexts=[
            Context(id=1, title="2018 FIFA World Cup", text="France won the FIFA World Cup in 2018, held in Russia.", score=0.95),
            Context(id=2, title="World Cup History", text="The FIFA World Cup is held every four years.", score=0.8),
            Context(id=3, title="2014 FIFA World Cup", text="Germany won the FIFA World Cup in 2014.", score=0.7),
        ]
    ),
    Document(
        question=Question("Which city hosted the Summer Olympics after Athens in 2004?"),
        answers=Answer(""),
        contexts=[
            Context(id=4, title="2004 Summer Olympics", text="Athens hosted the Summer Olympics in 2004.", score=0.9),
            Context(id=5, title="2008 Summer Olympics", text="Beijing hosted the Summer Olympics in 2008.", score=0.95),
            Context(id=6, title="Olympic Host Cities", text="The Olympics are held in different cities every four years.", score=0.8),
        ]
    ),
    Document(
        question=Question("Who discovered penicillin?"),
        answers=Answer(""),
        contexts=[
            Context(id=7, title="Penicillin Discovery", text="Alexander Fleming discovered penicillin in 1928.", score=0.98),
            Context(id=8, title="Antibiotics", text="Penicillin is an antibiotic.", score=0.7),
        ]
    ),
    Document(
        question=Question("What is the capital of Japan?"),
        answers=Answer(""),
        contexts=[
            Context(id=9, title="Japan", text="Tokyo is the capital of Japan.", score=0.99),
            Context(id=10, title="Japanese Cities", text="Osaka and Kyoto are major cities in Japan.", score=0.8),
        ]
    ),
    Document(
        question=Question("Who wrote 'Pride and Prejudice'?"),
        answers=Answer(""),
        contexts=[
            Context(id=11, title="Pride and Prejudice", text="Jane Austen wrote 'Pride and Prejudice'.", score=0.97),
            Context(id=12, title="English Literature", text="'Pride and Prejudice' is a classic novel.", score=0.8),
        ]
    ),
]

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
    retriever=retriever,
    stop_at_period=True
)


# Generate answer
generated_answers = generator.generate(test_docs)
print(generated_answers)  # Expected output: ["Beijing"] (the city that hosted after Athens)