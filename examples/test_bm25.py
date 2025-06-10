from rankify.retrievers.bm25 import BM25Retriever
from rankify.dataset.dataset import Document, Question, Answer

def main():
    # Initialize retriever with your local index folder
    retriever = BM25Retriever(
        n_docs=3,
        index_type="wiki",
        index_folder="/media/disk3/Rankify/rankify_indices/"
    )

    query = Document(
        question=Question(question="What is anarchism?"),
        answers=Answer(answers=["Answer"])
    )

    results = retriever.retrieve([query])
    
    if len(results) == 0:
        print("no results")
    else:
        print("we have results\n\n")
        print(query.question)
        print_result(results)


    print("Second Query")
    query = Document(
        question=Question(question="Wo was favourite president of President Obama?"),
        answers=Answer(answers=["Südafrika"])
    )
    results = retriever.retrieve([query])
    if len(results) == 0:
        print("no results")
    else:
        print("we have results\n\n")
        print(query.question)
        print_result(results)

def print_result(results):
    for context in results[0].contexts:
        print(f"Title: {context.title}")
        print(f"Score: {context.score}")
        print(f"Text: {context.text[:300]}...\n")

if __name__ == "__main__":
    main()
