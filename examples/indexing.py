# sample indexing code
from rankify.indexing import BGEIndexer, LuceneIndexer, DPRIndexer, ColBERTIndexer

data_path ="./sample.jsonl"
def index_bm25():
    """
    Sample code to index a BM25 index using the BM25Indexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    #rankify-index index ./sample.jsonl --retriever bm25  --output bm25_index

    """
    indexer = LuceneIndexer(corpus_path=data_path, output_dir='./rankify_indices',
                            chunk_size=1024, threads=8, index_type="wiki")

    indexer.build_index()
    indexer.load_index()

    print("BM25 indexing complete.")

def index_dpr_wiki():
        indexer = LuceneIndexer(corpus_path=data_path, output_dir="rankify_indices",
                                chunk_size=1024, threads=8, index_type="wiki", retriever_name="dpr")
        indexer.build_index()
        indexer.load_index()
        print("DPR wiki indexing complete.")

def index_dpr_msmarco():
        indexer = DPRIndexer(
            corpus_path=data_path,output_dir="rankify_indices",
            index_type="msmarco",
            encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
            batch_size=32
        )

        indexer.build_index()
        indexer.load_index()

        print("DPR msmarco indexing complete.")

def index_contriever_wiki():
    """
    Sample code to index a Contriever index using the ContrieverIndexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = DPRIndexer(
        corpus_path=data_path,
        output_dir="rankify_indices",
        index_type="wiki",
        encoder_name="facebook/contriever",
        batch_size=32,
    )

    indexer.build_index()
    indexer.load_index()

    print("Contriever wiki indexing complete.")

def index_contriever_msmarco():
    """
    Sample code to index a Contriever index using the ContrieverIndexer class for MSMARCO.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = DPRIndexer(
        corpus_path=data_path,
        output_dir="rankify_indices",
        index_type="msmarco",
        encoder_name="facebook/contriever",
        batch_size=32
    )

    indexer.build_index()
    indexer.load_index()

    print("Contriever msmarco indexing complete.")

def index_colbert_wiki():
    """
    Sample code to index a ColBERT index using the ColBERTIndexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = ColBERTIndexer(
        corpus_path=data_path,
        output_dir="rankify_indices",
        index_type="wiki",
        batch_size=32
    )

    indexer.build_index()
    indexer.load_index()

    print("ColBERT wiki indexing complete.")

def index_bge_wiki():
    """
    Sample code to index a BGE index using the BGEIndexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = BGEIndexer(
        corpus_path=data_path,
        output_dir="rankify_indices",
        index_type="wiki",
        encoder_name="BAAI/bge-large-en-v1.5",
        batch_size=32
    )

    indexer.build_index()
    indexer.load_index()

    print("BGE wiki indexing complete.")

if __name__ == "__main__":
    #index_bm25()
    #index_dpr_wiki()
    #index_dpr_msmarco()
    #index_contriever_wiki()
    #index_contriever_msmarco()
    #index_colbert_wiki()
    index_bge_wiki()