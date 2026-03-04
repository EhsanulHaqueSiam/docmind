# DocMind

Document intelligence API powered by **Gemini**, **Docling**, and **Qdrant**.

Upload documents, and the system automatically parses, chunks, embeds, and indexes them. Ask questions and get grounded answers with source citations, or search for relevant passages directly.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, images (PNG/JPG/TIFF/BMP)
- **Semantic search** — Dense vector similarity with Gemini embeddings (`gemini-embedding-001`)
- **Asymmetric retrieval** — Separate `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types for optimal accuracy
- **Gemini reranking** — LLM-based passage reranking for higher relevance
- **Smart fallback** — Tries Gemini Flash first, falls back to Pro if the answer seems insufficient
- **Auto-ingestion** — File watcher monitors `docs/` and ingests new/modified files automatically
- **Retry resilience** — Exponential backoff on all Gemini and Qdrant API calls
- **Optional API key auth** — Set `API_KEY` to protect all endpoints
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
git clone https://github.com/EhsanulHaqueSiam/docmind.git && cd docmind
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

> **Required GCP APIs**: Enable [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) in your project.

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

Or use the API:

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@paper.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'

# Search without generating an answer
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "rerank": true}'

# List documents
curl http://localhost:8000/api/v1/documents

# Collection stats
curl http://localhost:8000/api/v1/stats

# Health check
curl http://localhost:8000/api/v1/health
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Ask a question (`{"question": "...", "use_pro": false}`) |
| `POST` | `/api/v1/search` | Semantic search (`{"query": "...", "rerank": true}`) |
| `POST` | `/api/v1/documents/upload` | Upload and ingest a file (multipart form) |
| `GET` | `/api/v1/documents` | List all ingested documents (supports `?offset=0&limit=50`) |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete a document and its chunks |
| `POST` | `/api/v1/documents/ingest` | Trigger manual re-ingestion of `docs/` |
| `GET` | `/api/v1/stats` | Collection statistics (point count, vector count) |
| `GET` | `/api/v1/health` | Health check with Qdrant status |

### Accessing Documentation

Once the app is running, open your browser:

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Full documentation page (setup guide, API reference, code examples) |
| `http://localhost:8000/docs` | Interactive Swagger UI — try every endpoint live in the browser |
| `http://localhost:8000/redoc` | Alternative API docs (ReDoc format) |

## Authentication

Set the `API_KEY` environment variable to require an `X-API-Key` header on all requests:

```env
API_KEY=your-secret-key
```

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/v1/health
```

Leave `API_KEY` empty for open access (default).

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | — | Google Cloud project ID (required) |
| `GCP_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_FLASH_MODEL` | `gemini-2.5-flash` | Fast model for queries |
| `GEMINI_PRO_MODEL` | `gemini-2.5-pro` | Fallback model for complex queries |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `EMBEDDING_DIMENSION` | `768` | Embedding output dimensions (768 or 3072) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_API_KEY` | — | Qdrant API key (for Qdrant Cloud) |
| `QDRANT_COLLECTION` | `docmind_documents` | Qdrant collection name |
| `STORAGE_MODE` | `local` | `local` or `gcs` |
| `DOCS_DIRECTORY` | `docs` | Local document directory |
| `GCS_BUCKET` | — | GCS bucket name (when `STORAGE_MODE=gcs`) |
| `API_KEY` | — | API key for authentication (empty = open access) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `TOP_K` | `10` | Number of chunks to retrieve |
| `RERANK_TOP_K` | `5` | Number of chunks after reranking |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum upload file size |
| `LOG_LEVEL` | `info` | Logging level |

## Deploy to Cloud Run

### Prerequisites

- Qdrant Cloud cluster (or self-hosted Qdrant with a stable URL)
- GCS bucket for document storage
- GCP Secret Manager secret `docmind-secrets` with: `GCP_PROJECT_ID`, `QDRANT_API_KEY`, `GCS_BUCKET`

### Deploy

```bash
export PROJECT_ID=your-gcp-project-id

# Build and push image
docker build -t gcr.io/$PROJECT_ID/docmind-api:latest .
docker push gcr.io/$PROJECT_ID/docmind-api:latest

# Update deploy/cloudrun.yaml with your PROJECT_ID and Qdrant URL, then:
sed -i "s/PROJECT_ID/$PROJECT_ID/g" deploy/cloudrun.yaml
gcloud run services replace deploy/cloudrun.yaml --region us-central1
```

## Development

```bash
uv sync                        # Install deps
uv run ruff check src/         # Lint
uv run ruff format src/        # Format

# Or with mise
mise run lint
mise run format
```

## Architecture

```
POST /api/v1/query
  → search_with_rerank() → embed query (RETRIEVAL_QUERY) → Qdrant → Gemini rerank
  → query() → build context → Gemini generate → answer with sources

POST /api/v1/search
  → search_with_rerank() → embed query → Qdrant → Gemini rerank → raw chunks

POST /api/v1/documents/upload
  → validate + sanitize → save to storage → Docling parse → chunk → embed (RETRIEVAL_DOCUMENT) → Qdrant

File watcher (docs/)
  → detect new/modified files → debounce → ingest_file()
```

## License

MIT
