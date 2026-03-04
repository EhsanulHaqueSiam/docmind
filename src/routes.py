from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    use_pro: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str | None
    chunks_used: int
    fallback: bool


class IngestResponse(BaseModel):
    status: str
    filename: str
    doc_id: str | None = None
    chunks: int = 0
    reason: str | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    timestamp: float


@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    from src.agent import query, query_with_fallback

    try:
        if req.use_pro:
            result = await asyncio.to_thread(query, req.question, True)
        else:
            result = await asyncio.to_thread(query_with_fallback, req.question)

        return QueryResponse(**result)
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    from src.ingest import ingest_file
    from src.storage import get_storage

    data = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    storage = get_storage()
    path = await asyncio.to_thread(storage.save_file, file.filename, data)

    try:
        result = await asyncio.to_thread(ingest_file, path)
        return IngestResponse(**result)
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=list[DocumentInfo])
async def get_documents():
    from src.ingest import list_documents

    docs = await asyncio.to_thread(list_documents)
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    from src.ingest import delete_document

    try:
        await asyncio.to_thread(delete_document, doc_id)
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/ingest")
async def trigger_ingestion():
    from src.ingest import ingest_directory

    docs_dir = Path(settings.docs_directory)
    if not docs_dir.exists():
        raise HTTPException(status_code=404, detail="Docs directory not found")

    results = await asyncio.to_thread(ingest_directory, docs_dir)
    return {"results": results}


@router.get("/health")
async def health_check():
    from src.ingest import get_qdrant

    try:
        client = get_qdrant()
        await asyncio.to_thread(client.get_collections)
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"

    return {
        "status": "healthy",
        "qdrant": qdrant_status,
        "storage_mode": settings.storage_mode,
    }
