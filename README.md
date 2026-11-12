# calibre-web2rag

[![CI](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`calibre-web2rag` turns a Calibre ebook library into a Qdrant vector
collection for retrieval-augmented generation (RAG). It reads book metadata
straight from Calibre's `metadata.db`, extracts the text of DRM-free PDF,
EPUB and MOBI files, splits it into overlapping sentence-aware chunks,
embeds them through the OpenAI embeddings API (OpenAI itself, or a
compatible server such as Ollama, vLLM or LM Studio) or with local
sentence-transformers, and upserts them with rich
payloads (authors, tags, series, identifiers, chapter titles, and links
back to Calibre Web), so an LLM can answer both "what does this book say
about X?" and "which books do I have by Y?".

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/calibre_web2rag/__main__.py` | Command line: logging, SIGTERM handling, exit codes |
| `src/calibre_web2rag/config.py` | Settings from environment variables, validated at startup |
| `src/calibre_web2rag/calibre_db.py` | Reads books and their files from Calibre's SQLite database |
| `src/calibre_web2rag/extractors.py` | DRM checks and text extraction for PDF, EPUB and MOBI |
| `src/calibre_web2rag/chunking.py` | Sentence-aware chunking with overlap |
| `src/calibre_web2rag/embeddings.py` | OpenAI-API and sentence-transformers embedders |
| `src/calibre_web2rag/ingest.py` | Builds points and payloads and drives the run |
| `src/calibre_web2rag/qdrant_store.py` | Creates the collection and payload indexes, upserts points |
| `src/calibre_web2rag/links.py` | Calibre Web download and OPDS links |
| `tests/` | pytest tests, one file per module |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | Short architecture summary |

## Technical details

The ingester is a batch job: each run reads the whole library, upserts
into Qdrant and exits. Point IDs are deterministic (UUID5 of book, format,
chunk index and text), so re-running updates points in place instead of
duplicating them. Run it again, by hand or from cron, to pick up new books.

For each book with at least one PDF, EPUB or MOBI file it writes:

- a `book_metadata` point: title, authors, publisher, series, tags,
  languages, ISBN, rating, formats and description, for catalog questions;
- `description` points from the Calibre comments;
- `content` points from each file's text. EPUB chapters keep their
  heading as `chapter_title`. Each chunk is prefixed with the book's title,
  authors, tags and series so its embedding carries that context.

It also writes `library_summary` points describing the whole collection
(counts of books, authors, formats, languages, categories and series).

Every payload has `chunk_type` and `chunk_text`, plus the book's metadata,
`source_url` (from `CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE` or
`CALIBRE_WEB_BASE_URL`) and `opds_url`. Keyword payload indexes are created
on `authors`, `tags`, `languages`, `series`, `publisher` and `chunk_type`,
and an integer index on `book_id`, for filtered search.

Encrypted PDFs and EPUBs with an `encryption.xml` manifest are skipped as
DRM-protected; MOBI files are skipped if they can't be unpacked. A file or
book that fails is logged and skipped without stopping the run. Texts
longer than the embedding model's context (`EMBEDDING_CONTEXT_LENGTH`) are
truncated.

Command line:

```bash
calibre-web2rag            # ingest, configured by environment variables
calibre-web2rag --version
```

Exit codes: 0 on success (including a clean stop on SIGTERM, which
finishes the file being read, upserts the points it has and exits), 1 if
ingestion fails, 2 for invalid configuration.

## Running it

Requires Docker. Configuration comes from environment variables, kept in
`.env` (see [docs/configuration.md](docs/configuration.md)):

```bash
cp .env.example .env   # then edit it: at least CALIBRE_METADATA_DB and CALIBRE_LIBRARY_ROOT
docker compose -f docker-compose.prod.yaml up -d
```

That starts Qdrant (on port 6333) and runs the latest release of
`ghcr.io/romkey/pdxhackerspace-calibre-web-rag-injester` once against it,
with your library mounted read-only. Set `APP_VERSION=X.Y.Z` to pin a
release. To ingest again:

```bash
docker compose -f docker-compose.prod.yaml run --rm app
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
- Changing `EMBEDDING_DOCUMENT_PREFIX` changes every stored vector: re-run
  the ingester to re-embed the library (points are overwritten in place).
- Anything that searches the collection must embed its queries with the
  **query** prefix.

### Querying the collection

Apps that query Qdrant directly must embed the query with the same model,
through the same server, with the query prefix:

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

hits = (
    QdrantClient(url="http://localhost:6333")
    .query_points(collection_name="calibre_books", query=vector, limit=5)
    .points
)
for hit in hits:
    print(hit.score, hit.payload["title"], hit.payload["chunk_type"])
```

From Python with this package installed, the configured embedder adds the
prefix for you:

```python
from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import build_embedder

vector = build_embedder(Settings.from_env()).encode(
    ["books about orbital mechanics"], kind="query"
)[0]
```

### With pip

Needs Python 3.15 or newer. Install into a virtual environment, from a
release's wheel:

```bash
python3 -m venv ~/.venvs/calibre-web2rag
~/.venvs/calibre-web2rag/bin/pip install https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester/releases/download/vX.Y.Z/calibre_web2rag-X.Y.Z-py3-none-any.whl
~/.venvs/calibre-web2rag/bin/calibre-web2rag --version
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
docker compose up --build                                  # run from the working tree, with Qdrant
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run calibre-web2rag
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
calibre-web2rag
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit.

Releases: run the **Release** workflow from the Actions tab, pick the bump,
and merge the release PR it opens; see `CHANGELOG.md`.

## License

[MIT](LICENSE)
