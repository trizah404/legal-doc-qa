import sys
sys.path.append('src')

from ingest import ingest_document
from embed import embed_chunks, embed_query
from retrieve import retrieve
from generate import generate_answer


class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.chunk_embeddings = None
        self.loaded = False

    def load_document(self, file_path: str):
        """Load and embed document once."""
        self.chunks = ingest_document(file_path)
        self.chunk_embeddings = embed_chunks(self.chunks)
        self.loaded = True
        print(f"Document ready. {len(self.chunks)} chunks loaded.")

    def ask(self, query: str) -> str:
        """Ask a question against the loaded document."""
        if not self.loaded:
            return "No document loaded. Call load_document() first."
        query_embedding = embed_query(query)
        results = retrieve(query_embedding, self.chunk_embeddings, self.chunks)
        return generate_answer(query, results)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_document("data/test.pdf")

    while True:
        query = input("\nAsk a question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        answer = pipeline.ask(query)
        print(f"\nAnswer: {answer}")