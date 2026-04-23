from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Convert text chunks into embedding vectors."""
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Embedding complete.")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Convert a single query into an embedding vector."""
    return model.encode(query)