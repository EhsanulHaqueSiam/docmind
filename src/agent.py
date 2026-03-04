from __future__ import annotations

import logging

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.generative_models import GenerativeModel

from src.config import settings
from src.retrieve import search_with_rerank

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions based on the provided documents.

Rules:
- Ground your answers ONLY in the provided context chunks.
- Cite sources using [filename] notation.
- If the context doesn't contain enough information, say so clearly.
- Be concise and direct.
- For multi-part questions, address each part separately.
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("filename", "unknown")
        headings = " > ".join(chunk.get("headings", []))
        header = f"[Source {i + 1}: {source}"
        if headings:
            header += f" | {headings}"
        header += "]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (ResourceExhausted, ServiceUnavailable, ConnectionError)
    ),
)
def _generate(model: GenerativeModel, prompt: str) -> str:
    response = model.generate_content(prompt)
    if not response.candidates:
        return "The model could not generate a response due to safety filters."
    return response.text


def query(question: str, use_pro: bool = False) -> dict:
    """Answer a question using RAG with Gemini."""
    # Retrieve relevant chunks
    chunks = search_with_rerank(question)

    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": [],
            "model": None,
            "chunks_used": 0,
            "fallback": False,
        }

    # Build context
    context = _build_context(chunks)

    # Select model
    model_name = settings.gemini_pro_model if use_pro else settings.gemini_flash_model
    model = GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question based on the context above. Cite sources using [filename] notation."
    )

    answer = _generate(model, prompt)

    # Extract unique source filenames
    sources = list({c["filename"] for c in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "chunks_used": len(chunks),
        "fallback": False,
    }


def query_with_fallback(question: str) -> dict:
    """Try Flash first, fall back to Pro if the answer seems insufficient."""
    result = query(question, use_pro=False)

    # Simple heuristic: if Flash gives a very short or uncertain answer, try Pro
    answer = result.get("answer", "")
    uncertain_phrases = [
        "i don't have enough",
        "not enough information",
        "cannot determine",
        "unclear from the context",
    ]

    if len(answer) < 50 or any(p in answer.lower() for p in uncertain_phrases):
        logger.info("Flash answer seems insufficient, falling back to Pro")
        pro_result = query(question, use_pro=True)
        pro_result["fallback"] = True
        return pro_result

    result["fallback"] = False
    return result
