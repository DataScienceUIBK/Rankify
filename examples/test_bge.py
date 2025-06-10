from rankify.dataset.dataset import Document, Question, Answer
from rankify.retrievers.BGERetriever import BGERetriever

def main():
    retriever = BGERetriever(
        n_docs=3,
        index_type="wiki",
        index_folder="/media/disk3/Rankify/rankify_indices/bge/bge_index_wiki",
        device="cpu"
    )

    query = Document(
        question=Question(question="Who was in charge of the Manhattan Project and what was its goal?"),
        answers=Answer(answers=["The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 … 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves."])
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
