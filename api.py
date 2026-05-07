import sys
sys.path.append('src')

import os
PORT = int(os.environ.get("PORT", 7860))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from main import RAGPipeline

app = FastAPI(title="Legal Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "Legal Doc QA is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and load it into the pipeline."""
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    pipeline.load_document(file_path)
    
    return {
        "message": f"Document loaded successfully",
        "filename": file.filename,
        "chunks": len(pipeline.chunks)
    }


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Ask a question against the loaded document."""
    if not pipeline.loaded:
        return {"error": "No document loaded. Upload a PDF first."}
    
    answer = pipeline.ask(request.question)
    return {"question": request.question, "answer": answer}