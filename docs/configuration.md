# Configuration

The ingester reads these environment variables at startup. Put them in
`.env` (copy `.env.example`); compose passes it to the container, and a
local run also loads `.env` from the current directory. An invalid value
stops the ingester (exit code 2) with a message naming the variable. An
empty value (`NAME=`) counts as unset.

## Calibre library

| Variable | Default | Meaning |
|----------|---------|---------|
| `CALIBRE_METADATA_DB` | required | Path to Calibre's `metadata.db` |
| `CALIBRE_LIBRARY_ROOT` | required | Library folder holding the book files. Compose mounts it read-only at the same path. |
| `CALIBRE_WEB_BASE_URL` | unset | Calibre Web base URL. Gives each point an `opds_url`, and a `source_url` of `<base>/download/<id>/<fmt>` unless the template below is set. |
| `CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE` | unset | Download link template for `source_url`; `{book_id}` and `{fmt}` (or `{format}`, lowercase) are filled in. |

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. The compose files set it to their own `qdrant` service. |
| `QDRANT_API_KEY` | unset | Qdrant API key, if your Qdrant requires one. |
| `QDRANT_COLLECTION` | `calibre_books` | Collection to create and upsert into. |
| `QDRANT_DISTANCE` | `cosine` | `cosine`, `dot` or `euclid`. Used only when creating the collection. |
| `VECTOR_SIZE` | `768` | Embedding dimension, ≥ 1. Must match the model. Used when creating the collection; an existing collection with a different size is an error. |

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `ollama` | `ollama`, or `sentence_transformers` (also `sentence-transformers`, `st`), which needs the `sentence-transformers` extra installed. |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model name for the provider. |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint. |
| `OLLAMA_API_KEY` | unset | Sent as `Authorization: Bearer …`, for an Ollama behind an authenticating proxy. |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Timeout for each embedding request, ≥ 1. |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens. Texts longer than about 4 characters per token are truncated before embedding. `0` disables truncation. |
| `HF_CACHE_DIR` | unset | Hugging Face model cache for `sentence_transformers`. If unset, `SENTENCE_TRANSFORMERS_HOME`, then `HF_HOME`, are used. |
| `SENTENCE_TRANSFORMERS_HOME` | unset | See `HF_CACHE_DIR`. |
| `HF_HOME` | unset | See `HF_CACHE_DIR`. |

## Chunking and logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHUNK_SIZE` | `1200` | Target chunk length in characters, ≥ 1. Chunks break at sentence boundaries. |
| `CHUNK_OVERLAP` | `120` | Characters of trailing sentences repeated at the start of the next chunk, ≥ 0 and < `CHUNK_SIZE`. |
| `BATCH_SIZE` | `64` | Points per Qdrant upsert, ≥ 1. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

## Compose and tooling (not read by the app)

| Variable | Meaning |
|----------|---------|
| `HF_CACHE_HOST_DIR` | Host folder `docker-compose.yaml` mounts at `HF_CACHE_DIR` (default `./.cache/huggingface`). |
| `APP_VERSION` | Release of the image `docker-compose.prod.yaml` runs (default `latest`). |
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
