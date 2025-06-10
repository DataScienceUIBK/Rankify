import argparse

from rankify.indexing.colbert_indexer import ColBERTIndexer
from rankify.indexing.lucene_indexer import LuceneIndexer
from rankify.indexing.dpr_indexer import DPRIndexer
from rankify.indexing.contriever_indexer import ContrieverIndexer
from rankify.indexing.bge_indexer import BGEIndexer
from pathlib import Path

SUPPORTED_RETRIEVERS = ["bm25", "dpr", "contriever", "colbert", "bge"]

def handle_output_directory(output_dir):
    """
    Ensure the output directory exists, creating it if necessary.
    """
    if not output_dir:
        print("Output directory is not specified. Using default 'rankify_indices'.")
        output_dir = "rankify_indices"
    else:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"Output directory {output_dir} does not exist. Creating it.")
            output_path.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_indexer_args(args) -> dict:
    """
    Extracts relevant arguments from the parsed command line arguments
    and prepares them for the indexer constructor.
    This function filters out None values and maps specific argument names
    to their corresponding indexer constructor parameters.

    :param args: Parsed command line arguments
    :return: dict of indexer arguments
    """
    indexer_args = {}
    constructor_names = {"output": "output_dir", "retriever": "retriever_name", "encoder": "encoder_name"}
    not_used_variables = ["command"]
    for k, v in vars(args).items():
        if v is None or k in not_used_variables:
            continue
        if k in constructor_names:
            indexer_args[constructor_names[k]] = v
        else:
            indexer_args[k] = v
    #print(f"Indexer arguments: {indexer_args}")
    return indexer_args

def main():
    print("CLI Started")
    parser = argparse.ArgumentParser(description="Rankify Indexer CLI")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index a corpus")
    index_parser.add_argument("corpus_path", type=str, help="Path to the corpus JSONL file")
    index_parser.add_argument("--retriever", type=str, choices=SUPPORTED_RETRIEVERS, help="Retriever to use")
    index_parser.add_argument("--output", default="rankify_indices", help="Output directory for index")
    index_parser.add_argument("--chunk_size", type=int, help="Lines per chunk")
    index_parser.add_argument("--threads", type=int, default=32, help="Thread count for processing")
    index_parser.add_argument("--index_type", type=str, default="wiki", help="Type of index to create (default: wiki)")

    # Dense-specific option
    index_parser.add_argument("--encoder", type=str, help="Encoder model name for dense indexing")
    index_parser.add_argument("--batch_size", type=int, help="Batch size for encoding")
    index_parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to use for indexing (default: cuda)")

    args = parser.parse_args()

    if args.command == "index":
        handle_output_directory(args.output)
        args.retriever = args.retriever.lower()

        if args.retriever == "bm25":
            indexer = LuceneIndexer(**get_indexer_args(args))
            indexer.build_index()
            indexer.load_index()
            print("BM25 indexing complete.")

        elif args.retriever == "dpr":
            if args.index_type == "wiki":
                indexer = LuceneIndexer(**get_indexer_args(args))
                indexer.build_index()
                indexer.load_index()
                print("DPR wiki indexing complete.")
                return
            else:
                if args.encoder is None:
                    args.encoder = "facebook/dpr-ctx_encoder-single-nq-base"
                    print("No encoder specified. Using default: facebook/dpr-ctx_encoder-single-nq-base")

                indexer = DPRIndexer(**get_indexer_args(args))

                indexer.build_index()
                indexer.load_index()

                print("DPR indexing complete.")

        elif args.retriever == "contriever":
            if args.encoder is None:
                #Todo: check if supported
                args.encoder = "facebook/contriever-msmarco"
                print("No encoder specified. Using default: facebook/contriever-msmarco")

            indexer = ContrieverIndexer(**get_indexer_args(args))
            indexer.build_index()
            indexer.load_index()

            print("Contriever indexing complete.")

        elif args.retriever == "colbert":
            if args.encoder is None:
                args.encoder = "colbert-ir/colbertv2.0"
                print("No encoder specified. Using default: colbert-ir/colbertv2.0")

            indexer = ColBERTIndexer(**get_indexer_args(args))
            indexer.build_index()
            indexer.load_index()

            print("ColBERT indexing complete.")
        elif args.retriever == "bge":
            if args.encoder is None:
                args.encoder = "BAAI/bge-large-en-v1.5"
                print("No encoder specified. Using default: BAAI/bge-large-en-v1.5")

            indexer = BGEIndexer(**get_indexer_args(args))
            indexer.build_index()
            indexer.load_index()

            print("BGE indexing complete.")
        else:
            print(f"Unknown retriever type: {args.retriever}. Supported types are {SUPPORTED_RETRIEVERS}.")
