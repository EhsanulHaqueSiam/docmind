# RAG API

A production-ready Retrieval-Augmented Generation system built with **Vertex AI Gemini**, **Docling**, and **Qdrant**.

Upload documents, and the system automatically parses, chunks, embeds, and indexes them. Ask questions and get grounded answers with source citations.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, images (PNG/JPG/TIFF/BMP)
- **Semantic search** — Dense vector similarity with Vertex AI embeddings (`text-embedding-005`)
- **Gemini reranking** — LLM-based passage reranking for higher relevance
- **Smart fallback** — Tries Gemini Flash first, falls back to Pro if the answer seems insufficient
- **Auto-ingestion** — File watcher monitors `docs/` and ingests new/modified files automatically
- **Retry resilience** — Exponential backoff on all Vertex AI and Qdrant API calls
- **Dual storage** — Local filesystem or Google Cloud Storage
- **Cloud-ready** — Docker containerized with Cloud Run deployment config

## Prerequisites

- **Python 3.12–3.13** (3.14 not yet supported by Google protobuf)
- **Docker** (for Qdrant, or use Qdrant Cloud)
- **GCP project** with Vertex AI API enabled
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **[mise](https://mise.jdx.dev/)** (optional, for task runner)

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> && cd rag
uv sync
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your GCP project ID:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
```

Make sure you're authenticated with GCP:

```bash
gcloud auth application-default login
```

### 3. Run

**Option A — Local dev (recommended for development):**

```bash
# Start Qdrant + app with hot reload
mise run dev-all

# Or manually:
docker compose up -d qdrant
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B — Full Docker stack:**

```bash
docker compose up -d
```

### 4. Use

Drop files into the `docs/` directory — they'll be auto-ingested.

Or upload via API:

```bash
# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@paper.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'

# List documents
curl http://localhost:8000/documents

# Health check
curl http://localhost:8000/health
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Ask a question (`{"question": "...", "use_pro": false}`) |
| `POST` | `/documents/upload` | Upload and ingest a file (multipart form) |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its chunks |
| `POST` | `/documents/ingest` | Trigger manual re-ingestion of `docs/` |
| `GET` | `/health` | Health check with Qdrant status |

Interactive docs at `http://localhost:8000/docs`.

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | — | Google Cloud project ID (required) |
| `GCP_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_FLASH_MODEL` | `gemini-2.5-flash` | Fast model for queries |
| `GEMINI_PRO_MODEL` | `gemini-2.5-pro` | Fallback model for complex queries |
| `EMBEDDING_MODEL` | `text-embedding-005` | Vertex AI embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_API_KEY` | — | Qdrant API key (for Qdrant Cloud) |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection name |
| `STORAGE_MODE` | `local` | `local` or `gcs` |
| `DOCS_DIRECTORY` | `docs` | Local document directory |
| `GCS_BUCKET` | — | GCS bucket name (when `STORAGE_MODE=gcs`) |
| `TOP_K` | `10` | Number of chunks to retrieve |
| `RERANK_TOP_K` | `5` | Number of chunks after reranking |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum upload file size |
| `LOG_LEVEL` | `info` | Logging level |

## Deploy to Cloud Run

### Prerequisites

- Qdrant Cloud cluster (or self-hosted Qdrant with a stable URL)
- GCS bucket for document storage
- GCP Secret Manager secret `rag-secrets` with:
  - `GCP_PROJECT_ID`, `QDRANT_API_KEY`, `GCS_BUCKET`

### Deploy

```bash
# Build and push image
export PROJECT_ID=your-gcp-project-id
docker build -t gcr.io/$PROJECT_ID/rag-api:latest .
docker push gcr.io/$PROJECT_ID/rag-api:latest

# Update deploy/cloudrun.yaml with your PROJECT_ID and Qdrant URL, then:
sed -i "s/PROJECT_ID/$PROJECT_ID/g" deploy/cloudrun.yaml
gcloud run services replace deploy/cloudrun.yaml --region us-central1
```

## Development

```bash
# Install with dev deps
uv sync

# Run linter
uv run ruff check src/

# Format code
uv run ruff format src/

# Or use mise tasks
mise run lint
mise run format
```

## Architecture

```
POST /query
  → search_with_rerank() → embed query → Qdrant search → Gemini rerank
  → query() → build context → Gemini generate → response with sources

POST /documents/upload
  → save to storage → parse with Docling → chunk → embed → upsert to Qdrant

File watcher (docs/)
  → detect new/modified files → debounce → ingest_file()
```

## License

MIT
