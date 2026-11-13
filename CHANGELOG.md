# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-11-13

### Changed

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
