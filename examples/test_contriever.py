from rankify.dataset.dataset import Document, Question, Answer
from rankify.retrievers.contriever import ContrieverRetriever


def main():
    retriever = ContrieverRetriever(
        n_docs=3,
        model="facebook/contriever",
        index_type="wiki",
        index_folder="/media/disk3/Rankify/rankify_indices/contriever/contriever_index_wiki",
        device="cpu"
    )

    query = Document(
        question=Question(question="What is anarchism?"),
        answers=Answer(answers=["Answer"])
    )

    results = retriever.retrieve([query])
    
    if len(results) == 0:
        print("no results")
    else:
        print(query.question)
        print_result(results)

def print_result(results):
    for context in results[0].contexts:
        print(f"Title: {context.title}")
        print(f"Score: {context.score}")
        print(f"Text: {context.text[:300]}...")
        print(f"HasAnswer: {context.has_answer}\n")

if __name__ == "__main__":
    main()
