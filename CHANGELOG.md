# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-11-19

### Added

- The ingester is now the ebook-library source of the hackerspace RAG
  pipeline (member tier). New commands: `calibre-web2rag migrate`,
  `calibre-web2rag run --mode incremental|full|reindex` and
  `calibre-web2rag status`.
- PostgreSQL (`DATABASE_URL`, required): a staging store of every book, the
  shared entity registry (works, manifestations, series, merge decisions, a
  review queue, entity aliases), the flat `catalog_items` table for the
  realtime catalog tool, an LLM result cache, enrichment results, a
  quarantine log and per-run status. The schema is applied with Alembic.
- Every book is read, including books without files: sort title, subtitle
  (custom column), series index, date added, cover path, all identifiers
  and custom columns, every format with its size, and the table of contents
  from the EPUB navigation or PDF outline.
- Cross-format works: `work_id` from the ISBN-13 or a normalized title and
  surname; books from other sources (audiobook, hardcopy) are merged
  automatically when the titles match closely, or queued for review.
- Enrichment from Open Library, then Google Books (`ENRICHMENT_ENABLED`,
  `GOOGLE_BOOKS_API_KEY`), for books missing a description or tags.
- Ingest-time LLM tasks through the gateway (`LLM_MODEL`, `LLM_BASE_URL`,
  `LLM_API_KEY`, `LLM_TIMEOUT_SECONDS`): shared topic taxonomy
  (`TOPIC_TAXONOMY_FILE`), marketing-copy detection, machine-generated "good
  for" summaries, chunk context and linker confirmation.
- Hybrid search: each point has a `dense` and a `sparse` keyword vector
  (`calibre_web2rag.sparse.sparse_vector` builds query vectors). Collection
  metadata records the embedding model.
- A secret scanner redacts credentials before staging and logs them to the
  quarantine table.
- A library overview document in `hs_member`, regenerated every run
  (`LIBRARY_REQUEST_INFO`).
- Optional full text in `hs_booktext` (`BOOKTEXT_ENABLED`, off), chunked by
  EPUB chapter or PDF page.
- Sync safety: unchanged books and text are skipped, deletions propagate,
  full runs delete unseen points unless they saw less than half the books
  expected, and an advisory lock stops overlapping runs.

### Changed

- **Breaking:** the Qdrant layout is new. `QDRANT_COLLECTION` is gone;
  points go to `QDRANT_CATALOG_COLLECTION` (`hs_catalog`, one point per
  work), `QDRANT_MEMBER_COLLECTION` (`hs_member`) and
  `QDRANT_BOOKTEXT_COLLECTION` (`hs_booktext`), with the shared payload
  (`source`, `doc_id`, `text`, `context`, `tier`, …) instead of
  `chunk_type`/`chunk_text`. Existing collections are not migrated: start
  them over.
- **Breaking:** full text is no longer indexed by default; set
  `BOOKTEXT_ENABLED=true`.
- **Breaking:** `CHUNK_SIZE` and `CHUNK_OVERLAP` default to 1800 and 150.
- Links point at the Calibre-web book page (`<base>/book/<id>`, or
  `CALIBRE_WEB_BOOK_URL_TEMPLATE`); per-format download links stay.
- Each run reads a snapshot of `metadata.db` and checks its layout first.
- PDF text is extracted page by page, with page numbers.
- **Breaking:** embeddings go through the OpenAI embeddings API
  (`EMBEDDING_PROVIDER=openai`, the new default) instead of Ollama's own
  API; the `ollama` provider and the `OLLAMA_URL`, `OLLAMA_API_KEY` and
  `OLLAMA_TIMEOUT_SECONDS` variables are gone. Use `OPENAI_BASE_URL`,
  `OPENAI_API_KEY` and `OPENAI_TIMEOUT_SECONDS`; for Ollama, set
  `OPENAI_BASE_URL=http://ollama:11434/v1` (the Ollama override file does
  this). The defaults are now ModernBERT (`nomic-ai/modernbert-embed-base`)
  from a local server at `http://localhost:8080/v1`, and `VECTOR_SIZE=768`. Optional `EMBEDDING_DIMENSIONS` asks for shorter
  vectors. Requests are split to stay under the API's input and token
  limits.
- **Breaking:** requires Python 3.15 (was 3.11). The Docker image is based
  on `python:3.15-slim-trixie`.
- **Breaking:** compose files are renamed to `docker-compose.yaml` (dev),
  `docker-compose.test.yaml`, `docker-compose.lint.yaml`,
  `docker-compose.prod.yaml` and `docker-compose.ollama.yaml`, and the
  ingester service is now called `app`.
- **Breaking:** images are published by the Release workflow (a release
  PR from the Actions tab) instead of on every push to `main`; `latest`
  now means the latest release.
- The production image runs as a non-root user with only runtime
  dependencies; a separate `dev` stage has the development tools.
- Invalid configuration now fails at startup with a message naming the
  variable (exit code 2), instead of a traceback.
- Dependencies are managed with uv and locked in `uv.lock`; the build
  backend is `uv_build`. Development dependencies moved from the `dev`
  extra to the `dev` dependency group (`pip install -e . --group dev`).
- All runtime dependencies raised to their current releases (qdrant-client
  1.20, pypdf 7, sentence-transformers 6.3, and others).
- Unreadable files are logged with the reason instead of skipped silently.

### Added

- `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`, for models
  trained with task prefixes. The ingester embeds with the document
  prefix; `encode(texts, kind="query")` uses the query prefix.
- `--version`, and clean shutdown on SIGTERM (`docker stop`): the ingester
  finishes the file being read, upserts what it has and exits.
- `LOG_LEVEL` setting.
- CI (tests with coverage, a pip install of the built wheel, and the
  production image) and Lint (ruff, mypy strict, yamllint, actionlint,
  shellcheck, hadolint, gitleaks and `.env` checks via pre-commit)
  workflows, a Release workflow, and Renovate configuration.
- Tests for every module; coverage is 99% and CI requires at least 95%.
- MIT license, SECURITY.md, docs/configuration.md, docs/overview.md and
  AGENTS.md.

### Removed

- `scripts/tag-release.sh`; use the Release workflow or
  `scripts/bump-version.sh`.

### Fixed

- Ingestion no longer fails with `no such column: b.isbn` on current
  Calibre libraries; the ISBN is read from the book's identifiers.

## [0.2.3]

Earlier releases are recorded only in git history.
