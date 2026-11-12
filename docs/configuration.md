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
| `VECTOR_SIZE` | `768` | Embedding dimension, ≥ 1. Must match the model (or `EMBEDDING_DIMENSIONS`). Used when creating the collection; an existing collection with a different size is an error. |

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`: the OpenAI embeddings API, from a local compatible server (text-embeddings-inference, vLLM, Infinity, LM Studio, LiteLLM, Ollama's `/v1`) or api.openai.com. Or `sentence_transformers` (also `sentence-transformers`, `st`), which needs the `sentence-transformers` extra installed. |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model name for the provider. |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` | Embeddings API base URL; requests go to `<base>/embeddings`. OpenAI: `https://api.openai.com/v1`; Ollama: `http://ollama:11434/v1`. In a container, `localhost` is the container itself: use the server's reachable address, e.g. `http://host.docker.internal:8080/v1`. |
| `OPENAI_API_KEY` | unset | Sent as `Authorization: Bearer …`. Not needed for an unauthenticated self-hosted server. |
| `OPENAI_TIMEOUT_SECONDS` | `120` | Timeout for each embedding request, ≥ 1. |
| `EMBEDDING_DIMENSIONS` | unset | Ask for shorter vectors (`text-embedding-3-*` only), ≥ 0; `0` or unset means the model's full size. Must equal `VECTOR_SIZE` when set, or startup fails. |
| `EMBEDDING_QUERY_PREFIX` | empty | Prefix for search queries (see below). Kept exactly, spaces included. The ingester embeds only documents; this is used by `build_embedder(settings).encode(texts, kind="query")`. |
| `EMBEDDING_DOCUMENT_PREFIX` | empty | Prefix added to everything the ingester embeds (see below). Changing it changes every vector: re-run the ingester. |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens. Texts longer than about 4 characters per token are truncated before embedding. `0` disables truncation. |
| `HF_CACHE_DIR` | unset | Hugging Face model cache for `sentence_transformers`. If unset, `SENTENCE_TRANSFORMERS_HOME`, then `HF_HOME`, are used. |
| `SENTENCE_TRANSFORMERS_HOME` | unset | See `HF_CACHE_DIR`. |
| `HF_HOME` | unset | See `HF_CACHE_DIR`. |

### Embedding prefixes

Many embedding models are trained with a task prefix on each input, one for
queries and one for the documents searched. The prefix counts toward
`EMBEDDING_CONTEXT_LENGTH` when long text is truncated. In `.env`, put the
values in double quotes so the trailing space is kept; `\n` inside double
quotes becomes a newline.

| Model | Query prefix | Document prefix |
|-------|--------------|-----------------|
| `nomic-ai/modernbert-embed-base` [768] | `search_query: ` | `search_document: ` |
| `nomic-embed-text` (Ollama) [768] | `search_query: ` | `search_document: ` |
| e5 family (`intfloat/e5-*`) | `query: ` | `passage: ` |
| `mxbai-embed-large` [1024] | `Represent this sentence for searching relevant passages: ` | none |
| Qwen3-Embedding and similar | an instruction line, e.g. `Instruct: <task>\nQuery: ` | none |
| OpenAI `text-embedding-3-*` [1536/3072] | none | none |

Anything that searches the collection must embed its queries with the same
model, server and query prefix; see "Querying the collection" in the README.

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
| `OLLAMA_EMBEDDING_MODEL` | Model `docker-compose.ollama.yaml` pulls and uses (default `nomic-embed-text`). |
| `OLLAMA_VECTOR_SIZE` | Its dimension, used as `VECTOR_SIZE` with that override (default `768`). |
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
