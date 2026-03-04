from __future__ import annotations

import logging

from src.config import gcp_retry, get_genai_client, settings
from src.ingest import embed_texts, get_qdrant

logger = logging.getLogger(__name__)


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
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    return _query_qdrant(query_embedding)


@gcp_retry
def _call_rerank(prompt: str) -> str:
    """Call Gemini to rerank passages. Returns raw text response."""
    client = get_genai_client()
    response = client.models.generate_content(
        model=settings.gemini_flash_model,
        contents=prompt,
    )
    if not response.candidates:
        raise ValueError("Reranking response blocked by safety filters")
    return response.text.strip()


def search_with_rerank(query: str) -> list[dict]:
    """Search and rerank results using Gemini for relevance scoring."""
    candidates = search(query)

    if not candidates:
        return []

    rerank_top_k = settings.rerank_top_k
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
        text = _call_rerank(prompt)
    except Exception as e:
        logger.warning("Reranking failed, returning original order: %s", e)
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
