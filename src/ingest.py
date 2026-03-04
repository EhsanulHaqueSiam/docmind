from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient, models
from vertexai.language_models import TextEmbeddingModel

from src.config import gcp_retry, settings

logger = logging.getLogger(__name__)

_converter: DocumentConverter | None = None
_chunker: HierarchicalChunker | None = None
_embed_model: TextEmbeddingModel | None = None
_qdrant: QdrantClient | None = None
_init_lock = threading.Lock()


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        with _init_lock:
            if _converter is None:
                _converter = DocumentConverter()
    return _converter


def _get_chunker() -> HierarchicalChunker:
    global _chunker
    if _chunker is None:
        with _init_lock:
            if _chunker is None:
                _chunker = HierarchicalChunker()
    return _chunker


def _get_embed_model() -> TextEmbeddingModel:
    global _embed_model
    if _embed_model is None:
        with _init_lock:
            if _embed_model is None:
                _embed_model = TextEmbeddingModel.from_pretrained(
                    settings.embedding_model
                )
    return _embed_model


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        with _init_lock:
            if _qdrant is None:
                _qdrant = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                )
    return _qdrant


def _match_filter(key: str, value: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )


def ensure_collection() -> None:
    client = get_qdrant()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", settings.qdrant_collection)

    # Create payload indexes for fast filtering
    for field, schema in [
        ("file_hash", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # Index may already exist


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_already_ingested(file_hash: str) -> bool:
    client = get_qdrant()
    results = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_match_filter("file_hash", file_hash),
        limit=1,
    )
    return len(results[0]) > 0


@gcp_retry
def _embed_batch(model: TextEmbeddingModel, batch: list[str]) -> list[list[float]]:
    embeddings = model.get_embeddings(batch)
    return [e.values for e in embeddings]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    batch_size = settings.embedding_batch_size
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(_embed_batch(model, batch))
    return all_embeddings


@gcp_retry
def _upsert_points(client: QdrantClient, points: list[models.PointStruct]) -> None:
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )


def ingest_file(path: Path) -> dict:
    path = Path(path)

    try:
        file_hash = _file_hash(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    if _is_already_ingested(file_hash):
        logger.info("Skipping already ingested file: %s", path.name)
        return {
            "status": "skipped",
            "filename": path.name,
            "reason": "already ingested",
        }

    logger.info("Ingesting file: %s", path.name)

    # Parse document with Docling
    converter = _get_converter()
    result = converter.convert(str(path))
    doc = result.document

    # Chunk the document
    chunker = _get_chunker()
    chunks = list(chunker.chunk(doc))

    if not chunks:
        logger.warning("No chunks extracted from %s", path.name)
        return {"status": "empty", "filename": path.name, "chunks": 0}

    # Extract text from chunks
    chunk_texts = [chunk.text for chunk in chunks]

    # Embed all chunks
    embeddings = embed_texts(chunk_texts)

    # Build Qdrant points
    points = []
    doc_id = str(uuid4())
    timestamp = time.time()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid4())
        payload = {
            "text": chunk.text,
            "doc_id": doc_id,
            "filename": path.name,
            "file_hash": file_hash,
            "chunk_index": i,
            "timestamp": timestamp,
        }
        # Add heading metadata if available
        if hasattr(chunk, "meta") and chunk.meta:
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                payload["headings"] = chunk.meta.headings

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # Upsert into Qdrant (with retry)
    client = get_qdrant()
    _upsert_points(client, points)

    logger.info("Ingested %s: %d chunks", path.name, len(points))
    return {
        "status": "ingested",
        "filename": path.name,
        "doc_id": doc_id,
        "chunks": len(points),
    }


def ingest_directory(directory: Path) -> list[dict]:
    from src.storage import SUPPORTED_EXTENSIONS

    directory = Path(directory)
    results = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                result = ingest_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error("Failed to ingest %s: %s", file_path.name, e)
                results.append(
                    {
                        "status": "error",
                        "filename": file_path.name,
                        "error": str(e),
                    }
                )
    return results


def delete_document(doc_id: str) -> bool:
    client = get_qdrant()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=_match_filter("doc_id", doc_id)),
    )
    return True


def list_documents() -> list[dict]:
    client = get_qdrant()
    # Scroll through all points and aggregate by doc_id
    docs: dict[str, dict] = {}
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            doc_id = point.payload.get("doc_id", "")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": point.payload.get("filename", ""),
                    "chunks": 0,
                    "timestamp": point.payload.get("timestamp", 0),
                }
            docs[doc_id]["chunks"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return sorted(docs.values(), key=lambda d: d["timestamp"], reverse=True)
