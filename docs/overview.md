# calibre-web2rag overview

**What:** the ebook-library source of the PDX Hackerspace RAG pipeline
(plans 00 "shared architecture" and 02 "ebook library"). Member tier only.
**Runs as:** a batch CLI. `migrate` applies the schema; `run --mode
incremental` (hourly), `run --mode full` (nightly) or `run --mode reindex`
syncs once and exits; `status` prints recent runs as JSON.
**Talks to:** Calibre's `metadata.db` (a read-only snapshot; never
Calibre-web's `app.db`) and book files, PostgreSQL, Qdrant, an
OpenAI-compatible gateway (embeddings and generation), Open Library and
Google Books.

## Flow

1. Snapshot `metadata.db`, verify its tables, list book IDs with `last_modified`.
2. For each new or changed book: clean the description, read the EPUB/PDF
   table of contents, redact secrets (quarantine log), enrich from Open
   Library/Google Books if description or tags are missing, ask the LLM for
   topics and a "good for" summary, write the staging row.
3. Registry: `work_id` (ISBN-13, else title slug + surname), merge decisions,
   fuzzy match against other sources' works (auto-merge or review queue),
   series, the ebook manifestation; then the flat `catalog_items` row.
4. Index: one `hs_catalog` point per touched work (from all its
   manifestations), the regenerated library overview in `hs_member`, and
   optionally full text in `hs_booktext`. Dense + sparse vectors; unchanged
   text (by content hash) is not re-embedded.
5. Deletions propagate every run; `full` also deletes points not stamped
   with this run's ID. A run seeing < half the expected books deletes
   nothing and exits 1. An advisory lock stops overlapping runs.

## Files

- `__main__.py`: CLI, logging, SIGTERM, exit codes.
- `config.py`: `Settings.from_env()`; validates every variable.
- `pipeline.py`: `Pipeline.run(mode)`: staging, registry, indexing, reconcile.
- `calibre_db.py`: snapshot, schema check, `CalibreRepository`.
- `documents.py`: point IDs, payloads, catalog and overview text.
- `db.py`: psycopg queries; `migrate()`. `migrations/`: Alembic schema.
- `works.py`: ISBN normalization, work IDs, title similarity.
- `toc.py`: EPUB navigation and PDF outline. `extractors.py`: full text.
- `sanitize.py`: secret scanner. `linker.py`: alias matching.
- `llm.py`: gateway chat tasks, cached. `enrichment.py`: book lookups.
- `embeddings.py`: dense embedders. `sparse.py`: keyword vectors.
- `qdrant_store.py`: collections, payload indexes, upserts, reconcile.
- `taxonomy.txt`: the shared topic taxonomy.

## Shared tables (contract with other sources)

`staging_documents`, `registry_works`, `registry_manifestations`,
`registry_series`, `registry_merges`, `registry_review_queue`,
`entity_aliases`, `catalog_items`, `quarantine`, `sync_runs`, `llm_cache`,
`enrichment`. Created `IF NOT EXISTS`; this repo's Alembic version table is
`calibre_web2rag_alembic_version`.

## Conventions

- Configuration from environment variables; see docs/configuration.md.
- One bad book is logged and skipped; it never stops the run.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
