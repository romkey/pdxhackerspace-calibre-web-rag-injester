# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0 (unreleased; main at 790642a, 2026-10-12).
When re-scanning, apply the entries in its UPGRADING.md newer than that,
then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

## Deliberate differences from the templates

- The app is a batch job, not a service: no port, no `/healthz`, and
  `docker-compose.prod.yaml` runs it with `restart: "no"` and no health
  check (restarting would re-ingest the library in a loop). Qdrant runs
  alongside it in the dev and prod compose files.
- `docker-compose.ollama.yaml` is an extra override file for GPU Ollama
  embeddings, used with either the dev or the prod compose file.
- The repo name (`pdxhackerspace-calibre-web-rag-injester`) differs from
  the distribution and command name (`calibre-web2rag`); the image is
  named after the repo.
- A local `.env` is loaded with python-dotenv for runs outside Docker.
- PostgreSQL is shared with the pipeline's other sources: the shared tables
  are created `IF NOT EXISTS`, and Alembic uses its own version table
  (`calibre_web2rag_alembic_version`). Queries are plain psycopg; Alembic
  migrations hold raw SQL. Migrations live in the package so the installed
  app can run `calibre-web2rag migrate`.
- The design follows the pipeline plans "00 — shared architecture" and
  "02 — ebook library" (see docs/overview.md); tier is member only, and the
  config refuses `hs_public`.
