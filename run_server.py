from rankify.server import RankifyServer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rankify Server")
    parser.add_port = parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_retriever = parser.add_argument("--retriever", type=str, default="bge", help="Retriever model")
    parser.add_reranker = parser.add_argument("--reranker", type=str, default="flashrank", help="Reranker model")
    args = parser.parse_args()

    print(f"Starting Rankify Server on port {args.port} with {args.retriever} and {args.reranker}...")
    server = RankifyServer(retriever=args.retriever, reranker=args.reranker)
    server.start(port=args.port)
