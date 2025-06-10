from rankify.dataset.dataset import Document, Question, Answer
from rankify.retrievers.dpr import DenseRetriever

def wiki_retriever():
    retriever = DenseRetriever(
        n_docs=2,
        index_type="wiki",
        index_folder="/media/disk3/Rankify/rankify_indices/dpr_index_wiki",
    )

    query = Document(
        question=Question(question="What is anarchism?"),
        answers=Answer(answers=["Anarchism is a political philosophy..."]), contexts=[]
    )

    return retriever, query

def msmarco_retriever():
    retriever = DenseRetriever(
        n_docs=5,
        index_type="msmarco",
        index_folder="/media/disk3/Rankify/rankify_indices/dpr_index_msmarco",
    )
    query = Document(
        question=Question(question="Who oversaw the Manhattan Project during World War II?"),
        answers=Answer(answers=["General Leslie R. Groves"]), contexts=[]
    )

    query2 = Document(
        question=Question(question="What dilemma might scientists face when their discoveries are used for purposes beyond their control?"),
        answers=Answer(answers=["The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."]), contexts=[]
    )
    return retriever, query

def main():
    #retriever, query = wiki_retriever()
    retriever, query = msmarco_retriever()

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
        print(f"DocId: {context.id}")
        print(f"hasAnswer: {context.has_answer}\n")

if __name__ == "__main__":
    main()
