from __future__ import annotations

import logging
import threading

from vertexai.generative_models import GenerativeModel

from src.config import gcp_retry, settings
from src.ingest import embed_texts, get_qdrant

logger = logging.getLogger(__name__)

_rerank_model: GenerativeModel | None = None
_rerank_lock = threading.Lock()


def _get_rerank_model() -> GenerativeModel:
    global _rerank_model
    if _rerank_model is None:
        with _rerank_lock:
            if _rerank_model is None:
                _rerank_model = GenerativeModel(settings.gemini_flash_model)
    return _rerank_model


@gcp_retry
def _query_qdrant(query_embedding: list[float]) -> list[dict]:
    client = get_qdrant()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=settings.top_k,
        with_payload=True,
    )
    return [
        {
            "text": point.payload.get("text", ""),
            "filename": point.payload.get("filename", ""),
            "doc_id": point.payload.get("doc_id", ""),
            "chunk_index": point.payload.get("chunk_index", 0),
            "headings": point.payload.get("headings", []),
            "score": point.score,
        }
        for point in results.points
    ]


def search(query: str) -> list[dict]:
    """Search documents using dense vector similarity."""
    query_embedding = embed_texts([query])[0]
    return _query_qdrant(query_embedding)


def search_with_rerank(query: str) -> list[dict]:
    """Search and rerank results using Gemini for relevance scoring."""
    candidates = search(query)

    if not candidates:
        return []

    rerank_top_k = settings.rerank_top_k
    model = _get_rerank_model()
    context_chars = settings.rerank_context_chars
    numbered = "\n".join(
        f"[{i}] {c['text'][:context_chars]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f'Given the query: "{query}"\n\n'
        f"Rank these passages by relevance. Return ONLY a comma-separated list "
        f"of passage numbers (e.g., 2,0,5,1) from most to least relevant. "
        f"Return at most {rerank_top_k} numbers.\n\n{numbered}"
    )

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        logger.warning("Reranking API call failed, returning original order: %s", e)
        return candidates[:rerank_top_k]

    if not response.candidates:
        logger.warning("Reranking response blocked by safety filters")
        return candidates[:rerank_top_k]

    try:
        text = response.text.strip()
    except ValueError as e:
        logger.warning("Could not extract reranking text: %s", e)
        return candidates[:rerank_top_k]

    # Parse the ranking with deduplication
    indices = []
    seen = set()
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(candidates) and idx not in seen:
                indices.append(idx)
                seen.add(idx)

    if indices:
        reranked = [candidates[i] for i in indices[:rerank_top_k]]
        for rank, chunk in enumerate(reranked):
            chunk["rerank_score"] = 1.0 - (rank / len(reranked))
        return reranked

    return candidates[:rerank_top_k]
