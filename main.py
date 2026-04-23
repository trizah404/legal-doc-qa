import sys
sys.path.append('src')

from ingest import ingest_document
from embed import embed_chunks, embed_query
from retrieve import retrieve
from generate import generate_answer


def run_pipeline(file_path: str, query: str) -> str:
    """Run the full RAG pipeline on a document and query."""
    
    # Step 1 — Load and chunk the document
    chunks = ingest_document(file_path)
    
    # Step 2 — Embed all chunks
    chunk_embeddings = embed_chunks(chunks)
    
    # Step 3 — Embed the query
    query_embedding = embed_query(query)
    
    # Step 4 — Retrieve most relevant chunks
    results = retrieve(query_embedding, chunk_embeddings, chunks)

    
    # Step 5 — Generate cited answer
    answer = generate_answer(query, results)
    
    return answer


if __name__ == "__main__":
    file_path = input("Enter PDF path: ")
    query = input("Enter your question: ")
    
    print("\nSearching document...\n")
    answer = run_pipeline(file_path, query)
    print("Answer:\n")
    print(answer)