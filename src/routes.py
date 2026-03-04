from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.config import settings
from src.storage import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# --- Auth dependency ---


async def verify_api_key(x_api_key: str | None = Header(None)) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])


# --- Request/Response models ---


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    use_pro: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str | None
    chunks_used: int
    fallback: bool


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    rerank: bool = True


class SearchResult(BaseModel):
    text: str
    filename: str
    doc_id: str
    chunk_index: int
    headings: list[str] = []
    score: float
    rerank_score: float | None = None


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


class CollectionStats(BaseModel):
    collection: str
    points_count: int
    indexed_vectors_count: int
    segments_count: int
    status: str


# --- Helpers ---


def _sanitize_filename(filename: str | None) -> str:
    """Sanitize filename: strip path components, reject traversal attempts."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    # Take only the final component (strip directory traversal)
    name = PurePosixPath(filename).name
    # Remove suspicious characters
    name = re.sub(r"[^\w\s\-.]", "_", name)
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


# --- Query endpoints ---


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(req: QueryRequest):
    """Ask a question and get a grounded answer with source citations."""
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


@router.post("/search", response_model=list[SearchResult], tags=["Query"])
async def search_documents(req: SearchRequest):
    """Semantic search without LLM generation. Returns ranked document chunks."""
    from src.retrieve import search, search_with_rerank

    try:
        if req.rerank:
            chunks = await asyncio.to_thread(search_with_rerank, req.query)
        else:
            chunks = await asyncio.to_thread(search, req.query)

        return [SearchResult(**c) for c in chunks]
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- Document endpoints ---


@router.post("/documents/upload", response_model=IngestResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document. Supports PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, HTML, and images."""
    from src.ingest import ingest_file
    from src.storage import get_storage

    filename = _sanitize_filename(file.filename)

    # Validate extension
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    storage = get_storage()
    path = await asyncio.to_thread(storage.save_file, filename, data)

    try:
        result = await asyncio.to_thread(ingest_file, path)
        return IngestResponse(**result)
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def get_documents(
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum documents to return"),
):
    """List all ingested documents with pagination."""
    from src.ingest import list_documents

    try:
        docs = await asyncio.to_thread(list_documents)
        paginated = docs[offset : offset + limit]
        return [DocumentInfo(**d) for d in paginated]
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the index."""
    from src.ingest import delete_document

    try:
        deleted = await asyncio.to_thread(delete_document, doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return {"status": "deleted", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete failed for doc %s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/ingest", tags=["Documents"])
async def trigger_ingestion():
    """Trigger manual re-ingestion of the docs directory."""
    from src.ingest import ingest_directory

    docs_dir = Path(settings.docs_directory)
    if not docs_dir.exists():
        raise HTTPException(status_code=404, detail="Docs directory not found")

    try:
        results = await asyncio.to_thread(ingest_directory, docs_dir)
        return {"results": results}
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- System endpoints ---


@router.get("/health", tags=["System"])
async def health_check():
    """Health check with Qdrant connectivity status."""
    from src.ingest import get_qdrant

    try:
        client = get_qdrant()
        await asyncio.to_thread(client.get_collections)
    except Exception as e:
        logger.error("Health check: Qdrant unreachable: %s", e)
        raise HTTPException(status_code=503, detail="Qdrant unreachable")

    return {
        "status": "healthy",
        "qdrant": "connected",
        "storage_mode": settings.storage_mode,
    }


@router.get("/stats", response_model=CollectionStats, tags=["System"])
async def collection_stats():
    """Get collection statistics (document count, vector count, status)."""
    from src.ingest import get_qdrant

    try:
        client = get_qdrant()
        info = await asyncio.to_thread(
            client.get_collection, settings.qdrant_collection
        )
        return CollectionStats(
            collection=settings.qdrant_collection,
            points_count=info.points_count,
            indexed_vectors_count=info.indexed_vectors_count,
            segments_count=info.segments_count,
            status=info.status.value,
        )
    except Exception as e:
        logger.error("Failed to get collection stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
