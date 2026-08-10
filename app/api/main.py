import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.ingestion import ingest_document
from app.core.chunking import chunk_documents
from app.core.vectorstore import (
    add_documents_to_store,
    get_chunk_counts_by_source,
)
from agent import run_agent
from tool_router import CHARTS_DIR, TOOLS_BY_NAME

app = FastAPI(title="Document Analysis Engine API")

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    tool_calls: List[dict] = []


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Document Analysis Engine API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document, ingests it, chunks it, and stores the chunks in the vector DB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingestion pipeline
        docs = ingest_document(file_path)
        chunks = chunk_documents(docs)
        add_documents_to_store(chunks)

        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_added": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Runs one turn of the agent: searches the knowledge base (and other
    tools, as needed) and generates a grounded answer. Conversation history
    is kept per session_id.
    """
    try:
        result = run_agent(request.query, session_id=request.session_id)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            tool_calls=result["tool_calls"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """Lists every document in the knowledge base with its chunk count."""
    counts = get_chunk_counts_by_source()
    return {"documents": [{"name": name, "chunks": n} for name, n in counts.items()]}


@app.post("/charts/generate")
def generate_chart():
    """Generates the chunks-per-document chart and returns its filename."""
    message = TOOLS_BY_NAME["generate_chart"].invoke({})
    return {"message": message}


@app.get("/charts/{filename}")
def get_chart(filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(path)
