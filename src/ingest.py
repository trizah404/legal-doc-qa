import fitz  # PyMuPDF
import os

def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # ignore empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_document(file_path: str) -> list[str]:
    """Full ingestion pipeline — load PDF and return chunks."""
    print(f"Loading: {file_path}")
    text = load_pdf(file_path)
    print(f"Extracted {len(text)} characters")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks