import sys
sys.path.append('src')
from ingest import ingest_document
from embed import embed_chunks, embed_query
from retrieve import retrieve
from generate import generate_answer

chunks = ingest_document('data/test.pdf')
chunk_embeddings = embed_chunks(chunks)

query = 'what are your skills?'
query_embedding = embed_query(query)

results = retrieve(query_embedding, chunk_embeddings, chunks)
answer = generate_answer(query, results)

print(answer)