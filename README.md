# calibre-web2rag

[![CI](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`calibre-web2rag` is the ebook-library source of the PDX Hackerspace RAG
pipeline, feeding the member chatbot (member tier only). It reads Calibre's
`metadata.db` directly, stages every book in PostgreSQL, links each one to
a cross-format *work* in the shared entity registry (so an ebook and an
audiobook of the same book are one record), fills gaps from Open Library
and Google Books, maps books to the shared topic taxonomy with a local
LLM, and indexes them into Qdrant: one hybrid (dense + keyword) point per
work in `hs_catalog`, a library overview document in `hs_member`, and,
optionally, full text in `hs_booktext`. It also writes the flat catalog
table that the realtime catalog tool counts and filters.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/calibre_web2rag/__main__.py` | Command line: `run`, `migrate`, `status`; logging, SIGTERM, exit codes |
| `src/calibre_web2rag/config.py` | Settings from environment variables, validated at startup |
| `src/calibre_web2rag/pipeline.py` | One sync run: staging, registry, indexing, reconcile |
| `src/calibre_web2rag/calibre_db.py` | Read-only snapshot and schema check of `metadata.db`; reads books |
| `src/calibre_web2rag/documents.py` | Point IDs, the shared payload, catalog and overview text |
| `src/calibre_web2rag/db.py`, `migrations/` | PostgreSQL queries; Alembic schema |
| `src/calibre_web2rag/works.py` | ISBN normalization, `work_id`, title similarity |
| `src/calibre_web2rag/toc.py` | Tables of contents from EPUB navigation and PDF outlines |
| `src/calibre_web2rag/extractors.py` | DRM checks and full-text extraction (PDF pages, EPUB chapters, MOBI) |
| `src/calibre_web2rag/sanitize.py` | Deterministic secret scanner and redaction |
| `src/calibre_web2rag/enrichment.py` | Open Library and Google Books lookups |
| `src/calibre_web2rag/llm.py` | Ingest-time LLM tasks through the gateway, cached by input hash |
| `src/calibre_web2rag/linker.py` | Links chunks to registry entities by alias |
| `src/calibre_web2rag/embeddings.py`, `sparse.py` | Dense embedders; keyword (sparse) vectors |
| `src/calibre_web2rag/qdrant_store.py` | Hybrid collections, payload indexes, upserts, reconcile |
| `src/calibre_web2rag/taxonomy.txt` | The shared topic taxonomy |
| `tests/` | pytest against real PostgreSQL and a real `metadata.db`, in-memory Qdrant |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | Short architecture summary and the shared-table contract |

## Technical details

Command line:

```bash
calibre-web2rag run --mode incremental   # hourly: changed (last_modified) and deleted books
calibre-web2rag run --mode full          # nightly: every book, then delete what wasn't seen
calibre-web2rag run --mode reindex       # rebuild every point from staging, no Calibre read
calibre-web2rag status                   # recent runs, quarantine and catalog counts (JSON)
calibre-web2rag migrate                  # only apply schema migrations
calibre-web2rag --version
```

Every command first applies any pending PostgreSQL migrations, so a new or
upgraded install needs no separate migration step.

`calibre-web2rag` alone is `run --mode full`. Exit codes: 0 on success
(including a clean stop on SIGTERM, and a run skipped because another
holds the lock), 1 if the run fails or the safety valve held back
deletions, 2 for invalid configuration.

**What a run does.** It copies `metadata.db` with SQLite's backup API and
checks the tables and columns it reads. Every book is read, with or
without files: title, sort title, subtitle (a `subtitle` custom column),
authors, series and index, publisher, publication date, languages, all
identifiers, the description (HTML cleaned to text), tags, rating, every
custom column, formats with file sizes, cover path and date added. EPUB
navigation or the PDF outline gives the table of contents. Secrets are
redacted and logged to the `quarantine` table. Missing descriptions or
tags are looked up (Open Library, then Google Books) and stored apart from
library data. With `LLM_MODEL` set, an LLM maps the book to 1–5 slugs of
the shared taxonomy and, when the description is missing or marketing
copy, writes a "what this book is good for" summary, marked
machine-generated. LLM results are cached by input hash.

**Works.** `work_id` is `isbn:<ISBN-13>` (ISBN-10s are converted), else
`title:<title-slug>--<first-author-surname>` with edition noise
("2nd edition", "unabridged") stripped. Stored merge decisions are applied
first; otherwise the book is compared with other sources' works by the same
surname: a very close title merges automatically, a close one goes to
`registry_review_queue`. Series are registry entities too.

**Indexes.** `hs_catalog` holds one point per work, rebuilt when any
manifestation changes: its text is title, authors, series, description,
topics and table of contents, and its payload has `work_id`, `formats`
(ebook, audiobook, hardcopy), per-format links and locations, topics,
original tags, year, languages and date added, plus the shared payload
(`source`, `doc_id`, `tier: member`, `authority`, `content_hash`,
`sync_run_id`, entity ID lists…). `hs_member` gets a library overview
document, regenerated each run: size, strongest topics, how to reach the
library and how to request additions. Counts and lists ("how many science
fiction ebooks?") belong to the catalog tool over `catalog_items`, not to
retrieval. Each point has a `dense` vector and a `sparse` keyword vector
(Qdrant applies IDF); collection metadata records the embedding model.

**Sync.** Unchanged books (same `last_modified` and content hash) are not
re-processed, and unchanged text is not re-embedded. Deleted books are
removed from staging, the registry, the catalog table and Qdrant. A full
run deletes this source's points that it didn't stamp; if it saw less than
half the books expected, it deletes nothing and fails instead.

## Running it

Requires Docker. Configuration comes from environment variables, kept in
`.env` (see [docs/configuration.md](docs/configuration.md)):

```bash
cp .env.example .env   # then edit it: the Calibre paths, DATABASE_URL and POSTGRES_PASSWORD
docker compose -f docker-compose.prod.yaml up -d
```

That starts Qdrant (on port 6333) and PostgreSQL and runs one full sync with the latest release of
`ghcr.io/romkey/pdxhackerspace-calibre-web-rag-injester`, with your library
mounted read-only. Set `APP_VERSION=X.Y.Z` to pin a release. Then schedule
the hourly and nightly runs from cron (see the comment at the top of
`docker-compose.prod.yaml`):

```bash
docker compose -f docker-compose.prod.yaml run --rm app calibre-web2rag run --mode incremental
docker compose -f docker-compose.prod.yaml run --rm app calibre-web2rag status
```

By default it embeds with ModernBERT (`nomic-ai/modernbert-embed-base`,
768 dimensions) from a local OpenAI-compatible server at
`http://localhost:8080/v1`, such as Hugging Face text-embeddings-inference,
vLLM or Infinity. Inside the container `localhost` is the container itself,
so set `OPENAI_BASE_URL` to the server's reachable address (for example
`http://host.docker.internal:8080/v1`). For OpenAI, set
`OPENAI_BASE_URL=https://api.openai.com/v1`, `OPENAI_API_KEY`, an
`EMBEDDING_MODEL` such as `text-embedding-3-small`, and empty prefixes.
Make sure `VECTOR_SIZE` matches the model's dimension.

To embed locally with Ollama on an NVIDIA GPU instead (Linux with the
NVIDIA container toolkit), add the override file. It runs Ollama, pulls
`OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`, 768 dimensions; set
`OLLAMA_VECTOR_SIZE` for another model), and points the ingester at
Ollama's OpenAI-compatible API; no API key is needed:

```bash
docker compose -f docker-compose.prod.yaml -f docker-compose.ollama.yaml up -d
```

`nomic-embed-text` uses the same prefixes as ModernBERT (below); change
them if you pick another Ollama model.

### Embedding prefixes

Many embedding models expect a task prefix on each input: one for search
queries, another for the documents being searched. Set them with
`EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`; both are empty
unless set, and `.env.example` sets the ModernBERT ones (`search_query: ` /
`search_document: `). [docs/configuration.md](docs/configuration.md) lists
the prefixes for other models.

- The ingester embeds every point it stores with the **document** prefix.
- Changing `EMBEDDING_DOCUMENT_PREFIX` changes every stored vector: run
  `calibre-web2rag run --mode reindex` to re-embed from staging (points are
  overwritten in place). Changing the model also means recreating the
  collections, which are shared with the pipeline's other sources.
- Anything that searches the collection must embed its queries with the
  **query** prefix.

### Querying the collections

Clients embed the query with the same model, server and query prefix, build
the keyword vector with this package's `sparse_vector`, and fuse the two:

```python
from qdrant_client import QdrantClient, models

from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import build_embedder
from calibre_web2rag.sparse import sparse_vector

question = "Do we have anything on antenna design?"
dense = build_embedder(Settings.from_env()).encode([question], kind="query")[0]

hits = (
    QdrantClient(url="http://localhost:6333")
    .query_points(
        collection_name="hs_catalog",
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=20),
            models.Prefetch(query=sparse_vector(question), using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
    )
    .points
)
for hit in hits:
    print(hit.score, hit.payload["title"], hit.payload["formats"], hit.payload["url"])
```

### With pip

Needs Python 3.15 or newer. Install into a virtual environment, from a
release's wheel:

```bash
python3 -m venv ~/.venvs/calibre-web2rag
~/.venvs/calibre-web2rag/bin/pip install https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/releases/download/vX.Y.Z/calibre_web2rag-X.Y.Z-py3-none-any.whl
~/.venvs/calibre-web2rag/bin/calibre-web2rag migrate                  # only apply schema migrations
calibre-web2rag --version
```

For local sentence-transformers embeddings instead, install the
extra (`pip install "calibre-web2rag[sentence-transformers] @ <wheel URL>"`)
and set `EMBEDDING_PROVIDER=sentence_transformers`. Set the environment
variables yourself, or put them in a `.env` in the directory you run it
from.

## Development

With Docker, nothing else to install:

```bash
cp .env.example .env                                       # then edit it
docker compose up --build                                  # sync from the working tree, with Qdrant and Postgres
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
docker compose up -d db   # tests need PostgreSQL (the dev one, on port 5432)
DATABASE_URL=postgresql://app:app@localhost:5432/app uv run pytest
uv run mypy
uv run calibre-web2rag run --mode full
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
DATABASE_URL=postgresql://app:app@localhost:5432/app pytest
mypy
calibre-web2rag run
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit.

Releases: run the **Release** workflow from the Actions tab, pick the bump,
and merge the release PR it opens; see `CHANGELOG.md`.

## License

[MIT](LICENSE)
