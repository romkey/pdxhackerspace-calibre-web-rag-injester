# calibre-web2rag overview

**What:** ingests a Calibre ebook library (metadata and the text of DRM-free
PDF, EPUB and MOBI files) into a Qdrant collection for retrieval-augmented
generation.
**Runs as:** a batch CLI (`calibre-web2rag`): one pass over the library, then
exit. Re-runs are idempotent (deterministic point IDs).
**Talks to:** Calibre's `metadata.db` (SQLite, read-only), the library files,
Qdrant (HTTP), and Ollama (HTTP) or a local sentence-transformers model.

## Flow

1. `calibre_db` reads books and their supported files from `metadata.db`.
2. Per book: one `book_metadata` point, `description` points from the
   Calibre comments, and `content` points from each file's extracted text.
3. One `library_summary` set of points for catalog-wide questions.
4. Points are embedded in batches and upserted; SIGTERM stops after the
   batch in flight.

## Files

- `src/calibre_web2rag/__main__.py`: command line, logging, SIGTERM, exit codes.
- `src/calibre_web2rag/config.py`: `Settings.from_env()`; validates every variable.
- `src/calibre_web2rag/calibre_db.py`: `CalibreRepository`, books from SQLite.
- `src/calibre_web2rag/extractors.py`: DRM check and text extraction per format.
- `src/calibre_web2rag/chunking.py`: sentence-aware `split_text` with overlap.
- `src/calibre_web2rag/embeddings.py`: Ollama and sentence-transformers embedders.
- `src/calibre_web2rag/ingest.py`: builds points and payloads; `ingest()`.
- `src/calibre_web2rag/qdrant_store.py`: creates the collection and indexes; upserts.
- `src/calibre_web2rag/links.py`: Calibre Web download and OPDS URLs.
- `src/calibre_web2rag/models.py`: `BookRecord`, `CalibreFile`, `TextSection`.
- `tests/`: pytest, one file per module; `conftest.py` has settings/book factories.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Qdrant), `.test`, `.lint`, `.prod`,
  and `docker-compose.ollama.yaml` (GPU Ollama override).

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Unreadable or DRM-protected files are logged and skipped; one bad book
  never stops the run.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
