from __future__ import annotations

import logging

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.ingest import embed_texts, get_qdrant

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (ResourceExhausted, ServiceUnavailable, ConnectionError)
    ),
)
def search(query: str, top_k: int | None = None) -> list[dict]:
    """Search documents using dense vector similarity."""
    top_k = top_k or settings.top_k

    # Embed the query
    query_embedding = embed_texts([query])[0]

    client = get_qdrant()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        chunks.append(
            {
                "text": point.payload.get("text", ""),
                "filename": point.payload.get("filename", ""),
                "doc_id": point.payload.get("doc_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "headings": point.payload.get("headings", []),
                "score": point.score,
            }
        )

    return chunks


def search_with_rerank(
    query: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> list[dict]:
    """Search and rerank results using Gemini for relevance scoring."""
    top_k = top_k or settings.top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k

    # Get initial results
    candidates = search(query, top_k=top_k)

    if not candidates:
        return []

    # Rerank using Gemini
    try:
        from vertexai.generative_models import GenerativeModel

        model = GenerativeModel(settings.gemini_flash_model)
        context_chars = settings.rerank_context_chars
        # Build reranking prompt
        numbered = "\n".join(
            f"[{i}] {c['text'][:context_chars]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f'Given the query: "{query}"\n\n'
            f"Rank these passages by relevance. Return ONLY a comma-separated list "
            f"of passage numbers (e.g., 2,0,5,1) from most to least relevant. "
            f"Return at most {rerank_top_k} numbers.\n\n{numbered}"
        )

        response = model.generate_content(prompt)

        # Handle blocked responses
        if not response.candidates:
            logger.warning("Reranking response blocked by safety filters")
            return candidates[:rerank_top_k]

        text = response.text.strip()

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
            # Update scores based on rerank position
            for rank, chunk in enumerate(reranked):
                chunk["rerank_score"] = 1.0 - (rank / len(reranked))
            return reranked

    except (ValueError, Exception) as e:
        logger.warning("Reranking failed, returning original order: %s", e)

    return candidates[:rerank_top_k]
