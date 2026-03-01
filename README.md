# calibre-web2rag

`calibre-web2rag` ingests Calibre library content and metadata into Qdrant for RAG workloads.

It reads Calibre metadata from `metadata.db`, ingests DRM-free `PDF`, `EPUB`, and `MOBI` files, extracts text, chunks it, creates embeddings, and upserts points with rich payload metadata (authors, tags, identifiers, file links, and Calibre Web URLs where configured).

## Features

- Reads Calibre metadata directly from SQLite.
- Includes book metadata, identifiers, tags, publisher, series, languages, and comments in payloads.
- Adds ebook links:
  - `source_url` from `CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE` or `CALIBRE_WEB_BASE_URL`.
  - `opds_url` from `CALIBRE_WEB_BASE_URL`.
- Skips files that appear DRM-protected.
- Uses Qdrant as the vector store.
- Supports embeddings via local SentenceTransformers or Ollama (`EMBEDDING_PROVIDER`).
- Runs via Docker/Compose.
- Includes local + Docker-based tests.
- Uses semantic versioning (`MAJOR.MINOR.PATCH`) and Git tags (`vMAJOR.MINOR.PATCH`).

## Configuration

All configuration is done by environment variables outside the container.

1. Copy `.env.example` to `.env`.
2. Fill paths and endpoints.

Required:

- `CALIBRE_METADATA_DB`
- `CALIBRE_LIBRARY_ROOT`

Common optional variables:

- `CALIBRE_WEB_BASE_URL`
- `CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `EMBEDDING_PROVIDER` (`sentence_transformers` or `ollama`)
- `EMBEDDING_MODEL`
- `HF_CACHE_DIR` (for Hugging Face model caching with `sentence_transformers`)
- `HF_CACHE_HOST_DIR` (host path mapped to `HF_CACHE_DIR` in Docker)
- `OLLAMA_URL` (if using `ollama`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `BATCH_SIZE`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
calibre-web2rag
```

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build ingester
```

This starts `qdrant` and then the `ingester` service.

### Hugging Face model cache

If you use `sentence_transformers`, model downloads are cacheable and reusable:

- Local runs: set `HF_CACHE_DIR` (or `HF_HOME` / `TRANSFORMERS_CACHE` / `SENTENCE_TRANSFORMERS_HOME`).
- Docker runs: `docker-compose.yml` maps `HF_CACHE_HOST_DIR` on the host to `HF_CACHE_DIR` in the container.

Example:

```bash
HF_CACHE_HOST_DIR=~/.cache/huggingface
HF_CACHE_DIR=/hf-cache
```

## Run with Ollama embeddings (GPU-friendly)

For higher embedding throughput, run with Ollama and set:

- `EMBEDDING_PROVIDER=ollama`
- `EMBEDDING_MODEL=nomic-embed-text` (or another Ollama embedding model)

Then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build ingester
```

Notes:

- The Ollama override includes a model pull helper service.
- GPU passthrough configuration is primarily for Linux + NVIDIA Docker setups.
- Ensure `VECTOR_SIZE` matches your embedding model dimension.

## Run tests locally

```bash
pip install -e .[dev]
pytest -q
```

## Run tests with test compose file

```bash
cp .env.example .env
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit tests
```

## Semantic versioning and tags

- Version follows semver.
- Tag release commits with `vX.Y.Z`.

Example:

```bash
git commit -m "release: v0.1.0"
scripts/tag-release.sh 0.1.0
git push origin main --tags
```

## Docker image publishing on GitHub

GitHub Actions workflow at `.github/workflows/docker-publish.yml`:

- On push to `main`: builds and publishes `ghcr.io/<owner>/<repo>:latest`.
- On semver tag push (for example `v1.2.3`): also publishes version tags.

The `latest` tag is always updated from `main`.
