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
- Supports embeddings via the OpenAI embeddings API (OpenAI or any OpenAI-compatible server such as Ollama, vLLM, LM Studio) or local SentenceTransformers (`EMBEDDING_PROVIDER`).
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
- `EMBEDDING_PROVIDER` (`openai` (default) or `sentence_transformers`)
- `EMBEDDING_MODEL` (default `nomic-ai/modernbert-embed-base`, ModernBERT from Hugging Face)
- `OPENAI_BASE_URL` (default `http://localhost:8080/v1`, a local OpenAI-compatible server; use `https://api.openai.com/v1` for OpenAI)
- `OPENAI_API_KEY` (optional; sent as `Authorization: Bearer ...` when set)
- `OPENAI_TIMEOUT_SECONDS` (default `120`)
- `EMBEDDING_DIMENSIONS` (optional; reduced output dimensions for `text-embedding-3-*`)
- `VECTOR_SIZE` (default `768`; must match the embedding model's output dimension)
- `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX` (optional; see [Embedding prompts](#embedding-prompts))
- `HF_CACHE_DIR` (for Hugging Face model caching with `sentence_transformers`)
- `HF_CACHE_HOST_DIR` (host path mapped to `HF_CACHE_DIR` in Docker)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `BATCH_SIZE`

## Embedding prompts

Many embedding models expect a task prefix on each input: one for search
queries and another for the documents being searched. Set them with
`EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`. Both are empty by
default, so nothing is added unless you set them.

| Model | Query prefix | Document prefix |
| --- | --- | --- |
| `nomic-ai/modernbert-embed-base` (default) | `search_query: ` | `search_document: ` |
| `nomic-embed-text` | `search_query: ` | `search_document: ` |
| e5 family | `query: ` | `passage: ` |
| `mxbai-embed-large` | `Represent this sentence for searching relevant passages: ` | none |
| Qwen3-Embedding and similar | an instruction line, e.g. `Instruct: <task>\nQuery: ` | none |
| OpenAI `text-embedding-3-*` | none | none |

`.env.example` sets the ModernBERT prefixes. Put the values in double quotes
in `.env` so the trailing space is kept.

- The ingester embeds every stored point (book text, descriptions, book metadata and
  the library summary) with the **document** prefix.
- The prefix counts toward `EMBEDDING_CONTEXT_LENGTH` when long text is truncated.
- Changing `EMBEDDING_DOCUMENT_PREFIX` changes every stored vector. Re-run the
  ingester so the whole library is re-embedded (points are overwritten in place).
- Anything that searches the collection must embed its queries with the
  **query** prefix; see [Querying the collection](#querying-the-collection).

## Querying the collection

Other apps that query Qdrant directly must embed the query with the same
model, through the same server, and with the same query prefix that matches
the document prefix used at ingest time:

```python
import httpx
from qdrant_client import QdrantClient

QUERY_PREFIX = "search_query: "  # must match EMBEDDING_QUERY_PREFIX

resp = httpx.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "nomic-ai/modernbert-embed-base",
        "input": [QUERY_PREFIX + "books about orbital mechanics"],
    },
)
vector = resp.json()["data"][0]["embedding"]

hits = QdrantClient(url="http://localhost:6333").query_points(
    collection_name="calibre_books", query=vector, limit=5
).points
for hit in hits:
    print(hit.score, hit.payload["title"], hit.payload["chunk_type"])
```

From Python inside this project you can use the configured embedder instead,
which adds the prefix for you:

```python
from calibre_web2rag.config import load_settings
from calibre_web2rag.embeddings import build_embedder

vector = build_embedder(load_settings()).encode(
    ["books about orbital mechanics"], kind="query"
)[0]
```

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

Inside the container, `localhost` is the container itself. Point
`OPENAI_BASE_URL` at your embedding server's reachable address (for example
`http://host.docker.internal:8080/v1` or a compose service name).

### Build size and speed

The image keeps runtime libraries installed, including `sentence-transformers`.
To keep builds predictable while avoiding repeated model downloads, use the Hugging Face cache mount (`HF_CACHE_HOST_DIR` -> `HF_CACHE_DIR`).

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

Ollama exposes an OpenAI-compatible API at `/v1`, so it is used through the
`openai` provider. For higher embedding throughput, run a local Ollama with:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build ingester
```

The override sets:

- `EMBEDDING_PROVIDER=openai`
- `OPENAI_BASE_URL=http://ollama:11434/v1`
- `EMBEDDING_MODEL=${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}`
- `VECTOR_SIZE=${OLLAMA_VECTOR_SIZE:-768}`

Set `OLLAMA_EMBEDDING_MODEL` / `OLLAMA_VECTOR_SIZE` to use a different Ollama
model; they override `EMBEDDING_MODEL` / `VECTOR_SIZE` from `.env` for this
compose setup only. `nomic-embed-text` uses the same `search_query: ` /
`search_document: ` prefixes as the ModernBERT defaults in `.env.example`; change
`EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX` if you pick another model.

Notes:

- No `OPENAI_API_KEY` is needed for a local Ollama.
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
