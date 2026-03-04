from __future__ import annotations

import logging

from google.genai import types

from src.config import gcp_retry, get_genai_client, settings
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

_SAFETY_BLOCKED = "could not generate a response due to safety filters"

_NO_DOCS_RESULT = {
    "answer": "No relevant documents found. Please upload some documents first.",
    "sources": [],
    "model": None,
    "chunks_used": 0,
    "fallback": False,
}


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


@gcp_retry
def _generate(prompt: str, model_name: str) -> str:
    client = get_genai_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    if not response.candidates:
        logger.warning("Response blocked by safety filters")
        return f"The model {_SAFETY_BLOCKED}."
    return response.text


def _answer_with_chunks(question: str, chunks: list[dict], model_name: str) -> dict:
    """Generate an answer from pre-retrieved chunks."""
    context = _build_context(chunks)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question based on the context above. Cite sources using [filename] notation."
    )

    answer = _generate(prompt, model_name)
    sources = list({c["filename"] for c in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "chunks_used": len(chunks),
        "fallback": False,
    }


def query(question: str, use_pro: bool = False) -> dict:
    """Answer a question using RAG with Gemini."""
    chunks = search_with_rerank(question)

    if not chunks:
        return dict(_NO_DOCS_RESULT)

    model_name = settings.gemini_pro_model if use_pro else settings.gemini_flash_model
    return _answer_with_chunks(question, chunks, model_name)


def query_with_fallback(question: str) -> dict:
    """Try Flash first, fall back to Pro if the answer seems insufficient."""
    chunks = search_with_rerank(question)

    if not chunks:
        return dict(_NO_DOCS_RESULT)

    result = _answer_with_chunks(question, chunks, settings.gemini_flash_model)

    answer = result["answer"]
    uncertain_phrases = [
        "i don't have enough",
        "not enough information",
        "cannot determine",
        "unclear from the context",
        _SAFETY_BLOCKED,
    ]

    if len(answer) < 50 or any(p in answer.lower() for p in uncertain_phrases):
        logger.info("Flash answer seems insufficient, falling back to Pro")
        result = _answer_with_chunks(question, chunks, settings.gemini_pro_model)
        result["fallback"] = True

    return result
