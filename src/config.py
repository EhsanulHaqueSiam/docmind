from __future__ import annotations

import logging
import threading
from typing import Literal

from google.genai.errors import ServerError
from pydantic_settings import BaseSettings
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_retry_logger = logging.getLogger("docmind.retry")

# Shared retry decorator for all API calls (Gemini + Qdrant)
gcp_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ServerError, ConnectionError)),
    before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
)


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # GCP
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"

    # Gemini models
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro"

    # Embedding
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "docmind_documents"

    # Storage
    storage_mode: Literal["local", "gcs"] = "local"
    docs_directory: str = "docs"
    gcs_bucket: str = ""

    # App
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Auth
    api_key: str = ""
    cors_origins: str = "*"

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    rerank_context_chars: int = 500

    # Upload
    max_upload_size_mb: int = 100

    # Ingestion
    embedding_batch_size: int = 250

    # Watcher
    watcher_debounce_seconds: float = 2.0


settings = Settings()

# --- GenAI client singleton ---

_genai_client = None
_genai_lock = threading.Lock()


def get_genai_client():
    from google import genai

    global _genai_client
    if _genai_client is None:
        with _genai_lock:
            if _genai_client is None:
                _genai_client = genai.Client(
                    vertexai=True,
                    project=settings.gcp_project_id,
                    location=settings.gcp_location,
                )
    return _genai_client
