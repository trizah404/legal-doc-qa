import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, 
             chunks: list[str], top_k: int = 3) -> list[dict]:
    """Find top_k most relevant chunks for a query."""
    scores = []

    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append({"chunk": chunks[i], "score": float(score), "index": i})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]