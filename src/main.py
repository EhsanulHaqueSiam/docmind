from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import vertexai
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG application...")

    # Init Vertex AI
    if settings.gcp_project_id:
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

    # Ensure Qdrant collection exists (retry for startup race with qdrant container)
    from src.ingest import ensure_collection

    for attempt in range(5):
        try:
            await asyncio.to_thread(ensure_collection)
            break
        except Exception as e:
            if attempt == 4:
                logger.error("Failed to connect to Qdrant after 5 attempts: %s", e)
                raise
            logger.warning("Qdrant not ready (attempt %d/5): %s", attempt + 1, e)
            await asyncio.sleep(2)

    # Run initial ingestion of docs/ directory
    from src.ingest import ingest_directory

    docs_dir = Path(settings.docs_directory)
    if docs_dir.exists():
        try:
            results = await asyncio.to_thread(ingest_directory, docs_dir)
            ingested = sum(1 for r in results if r["status"] == "ingested")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            logger.info("Initial ingestion: %d ingested, %d skipped", ingested, skipped)
        except Exception as e:
            logger.error("Initial ingestion failed: %s", e)

    # Start file watcher
    watcher = None
    try:
        from src.watcher import DocWatcher

        watcher = DocWatcher()
        watcher.start()
    except Exception as e:
        logger.error("Failed to start file watcher: %s", e)

    yield

    # Shutdown
    if watcher is not None:
        watcher.stop()
    logger.info("RAG application stopped")


app = FastAPI(
    title="RAG API",
    description="RAG system with Vertex AI Gemini, Docling, and Qdrant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


from src.routes import router  # noqa: E402

app.include_router(router)
