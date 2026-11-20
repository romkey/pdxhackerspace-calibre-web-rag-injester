# Configuration

The ingester reads these environment variables at startup. Put them in
`.env` (copy `.env.example`); compose passes it to the container, and a
local run also loads `.env` from the current directory. An invalid value
stops the ingester (exit code 2) with a message naming the variable. An
empty value (`NAME=`) counts as unset.

## Calibre library

| Variable | Default | Meaning |
|----------|---------|---------|
| `CALIBRE_METADATA_DB` | required | Path to Calibre's `metadata.db`. Each run reads a read-only snapshot of it. Calibre-web's own `app.db` (users, shelves, downloads) is never read. |
| `CALIBRE_LIBRARY_ROOT` | required | Library folder holding the book files. Compose mounts it read-only at the same path. |
| `CALIBRE_WEB_BASE_URL` | unset | Calibre-web base URL. Members' links point at `<base>/book/<id>`; per-format downloads at `<base>/download/<id>/<fmt>`. Also named in the library overview as where to reach the library. |
| `CALIBRE_WEB_BOOK_URL_TEMPLATE` | unset | Book page link template; `{book_id}` is filled in. Overrides `<base>/book/<id>`. |
| `CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE` | unset | Download link template; `{book_id}` and `{fmt}` (or `{format}`, lowercase) are filled in. |

## PostgreSQL

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | required | `postgresql://user:password@host:5432/dbname`. Holds the staging store, entity registry, flat catalog table, LLM cache, enrichment results, quarantine and run status. Every run creates or updates the schema itself. |

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. The compose files set it to their own `qdrant` service. |
| `QDRANT_API_KEY` | unset | Qdrant API key, if your Qdrant requires one. |
| `QDRANT_CATALOG_COLLECTION` | `hs_catalog` | One point per book work (all formats). |
| `QDRANT_MEMBER_COLLECTION` | `hs_member` | Member-tier text chunks; this source writes the library overview document here. |
| `QDRANT_BOOKTEXT_COLLECTION` | `hs_booktext` | Full text of DRM-free books, only when `BOOKTEXT_ENABLED` is true. |
| `QDRANT_DISTANCE` | `cosine` | `cosine`, `dot` or `euclid`. Used only when creating a collection. |
| `VECTOR_SIZE` | `768` | Embedding dimension, ≥ 1. Must match the model (or `EMBEDDING_DIMENSIONS`). Used when creating collections; an existing collection with a different size is an error. |

The three collection names must differ, and none may be `hs_public`: this
source is member tier only. Each collection has a `dense` and a `sparse`
(keyword) named vector and records the embedding model in its metadata; an
existing collection embedded with another model is an error (recreate it
and run every source with `--mode reindex`).

## Embeddings

Embeddings, like generation, go through one OpenAI-compatible gateway;
model names are configuration, not code.

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`: the OpenAI embeddings API, from a local compatible server (text-embeddings-inference, vLLM, Infinity, LM Studio, LiteLLM, Ollama's `/v1`) or api.openai.com. Or `sentence_transformers` (also `sentence-transformers`, `st`), which needs the `sentence-transformers` extra installed. |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model name for the provider. |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` | Embeddings API base URL; requests go to `<base>/embeddings`. OpenAI: `https://api.openai.com/v1`; Ollama: `http://ollama:11434/v1`. In a container, `localhost` is the container itself: use the server's reachable address, e.g. `http://host.docker.internal:8080/v1`. |
| `OPENAI_API_KEY` | unset | Sent as `Authorization: Bearer …`. Not needed for an unauthenticated self-hosted server. |
| `OPENAI_TIMEOUT_SECONDS` | `120` | Timeout for each embedding request, ≥ 1. |
| `EMBEDDING_DIMENSIONS` | unset | Ask for shorter vectors (`text-embedding-3-*` only), ≥ 0; `0` or unset means the model's full size. Must equal `VECTOR_SIZE` when set, or startup fails. |
| `EMBEDDING_QUERY_PREFIX` | empty | Prefix for search queries (see below). Kept exactly, spaces included. The ingester embeds only documents; this is used by `build_embedder(settings).encode(texts, kind="query")`. |
| `EMBEDDING_DOCUMENT_PREFIX` | empty | Prefix added to everything the ingester embeds (see below). Changing it changes every vector: recreate the collections and run `--mode reindex`. |
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

## Generation (ingest-time LLM tasks)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | unset | Chat model on the OpenAI-compatible gateway (`POST <base>/chat/completions`). Maps each book to the topic taxonomy, flags marketing-copy descriptions, writes "what this book is good for" summaries, writes chunk context for the overview, and confirms ambiguous linker matches. Results are cached in Postgres by input hash. Unset: these steps are skipped. |
| `LLM_BASE_URL` | `OPENAI_BASE_URL` | Gateway base URL for generation. |
| `LLM_API_KEY` | `OPENAI_API_KEY` | Sent as `Authorization: Bearer …`. |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout per request, ≥ 1. |
| `TOPIC_TAXONOMY_FILE` | packaged `taxonomy.txt` | Shared topic taxonomy: one slug per line, `#` comments. |

## Enrichment

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENRICHMENT_ENABLED` | `true` | For books missing a description or tags, look them up on Open Library, then Google Books: by ISBN, else by title and author (accepted only when the title matches closely and the first author's surname matches). Stored separately, with their origin; never overwrites library data. Misses are retried after 30 days. `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`). |
| `ENRICHMENT_TIMEOUT_SECONDS` | `20` | Timeout per lookup, ≥ 1. |
| `GOOGLE_BOOKS_API_KEY` | unset | Google Books API key. Without one Google allows only a few requests a day. |

## Library overview and full text

| Variable | Default | Meaning |
|----------|---------|---------|
| `LIBRARY_REQUEST_INFO` | a generic sentence | How members request additions; goes into the library overview document. |
| `BOOKTEXT_ENABLED` | `false` | Also index the full text of DRM-free books into `hs_booktext`, chunked by chapter (EPUB) or page (PDF). That collection is never searched by default. Turning it on stages text for existing books on the next `full` run. |

## Chunking and logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHUNK_SIZE` | `1800` | Target chunk length in characters (about 450 tokens), ≥ 1. Chunks break at sentence boundaries within a section. |
| `CHUNK_OVERLAP` | `150` | Characters of trailing sentences repeated at the start of the next chunk, ≥ 0 and < `CHUNK_SIZE`. |
| `BATCH_SIZE` | `64` | Points per embedding request and Qdrant upsert, ≥ 1. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

## Scheduling

The default command, `serve`, runs until stopped: it scans, sleeps, and
scans again. A scan is incremental (changed and deleted books) unless no
full scan has finished within `FULL_SCAN_INTERVAL_HOURS`, so the first scan
on a fresh database is a full one. `run --mode ...` still does a single scan
and exits.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCAN_INTERVAL_MINUTES` | `60` | Minutes to sleep after one scan ends before the next starts, ≥ 1. |
| `FULL_SCAN_INTERVAL_HOURS` | `24` | Hours between full scans (full read and reconcile), ≥ 1. |

## Compose and tooling (not read by the app)

| Variable | Meaning |
|----------|---------|
| `HF_CACHE_HOST_DIR` | Host folder `docker-compose.yaml` mounts at `HF_CACHE_DIR` (default `./.cache/huggingface`). |
| `APP_VERSION` | Release of the image `docker-compose.prod.yaml` runs (default `latest`). |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | The `db` service's account and database in `docker-compose.prod.yaml`; match them in `DATABASE_URL`. |
| `OLLAMA_EMBEDDING_MODEL` | Model `docker-compose.ollama.yaml` pulls and uses (default `nomic-embed-text`). |
| `OLLAMA_VECTOR_SIZE` | Its dimension, used as `VECTOR_SIZE` with that override (default `768`). |
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
