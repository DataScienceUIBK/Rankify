from rankify.server import RankifyServer

if __name__ == "__main__":
    print("Starting Rankify Server...")
    # Initialize the server with requested models
    server = RankifyServer(retriever="bge", reranker="flashrank")
    # Start the server on port 8000
    server.start(port=8000)
