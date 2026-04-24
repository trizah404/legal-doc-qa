# Legal Document Q&A — RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets users upload any legal document and ask questions in plain English, receiving cited answers grounded in the document.

## How it works
PDF Upload → Text Extraction → Chunking → Embedding →
Semantic Search → Groq LLM → Cited Answer
## Stack
- **Sentence Transformers** — semantic embeddings
- **NumPy** — cosine similarity search
- **Groq API (Llama 3)** — answer generation
- **FastAPI** — REST API layer
- **Railway** — deployment

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF document |
| POST | `/ask` | Ask a question against the loaded document |

## Run locally

```bash
git clone https://github.com/trizah404/legal-doc-qa
cd legal-doc-qa
pip install -r requirements.txt
uvicorn api:app --reload
```

