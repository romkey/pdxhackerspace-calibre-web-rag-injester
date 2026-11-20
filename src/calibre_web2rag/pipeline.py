"""One sync run: Calibre -> staging + registry -> sanitize -> index in Qdrant.

Modes:
- ``incremental`` (hourly): books whose last_modified changed, plus deletions.
- ``full`` (nightly): every book; points not seen in this run are deleted,
  unless the run saw less than half the books of the previous full run.
- ``reindex``: rebuild every point from staging and the registry without
  reading Calibre, re-embedding everything (after a model change).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from calibre_web2rag.calibre_db import TEXT_FORMATS, CalibreRepository, snapshot
from calibre_web2rag.chunking import split_text
from calibre_web2rag.config import Settings
from calibre_web2rag.db import Database, RunCounts, StagedDocument
from calibre_web2rag.documents import (
    TIER,
    base_payload,
    book_raw,
    catalog_text,
    html_to_text,
    json_hash,
    overview_sections,
    payload_hash,
    point_id,
    text_hash,
    to_rfc3339,
    year_of,
)
from calibre_web2rag.embeddings import Embedder, build_embedder
from calibre_web2rag.enrichment import Enricher
from calibre_web2rag.extractors import (
    try_extract_epub_sections,
    try_extract_mobi_sections,
    try_extract_pdf_sections,
)
from calibre_web2rag.linker import Linker
from calibre_web2rag.links import build_book_url, build_ebook_url
from calibre_web2rag.llm import LLM, LLMError
from calibre_web2rag.models import BookRecord, TextSection
from calibre_web2rag.qdrant_store import CATALOG_INDEXES, COMMON_INDEXES, QdrantStore
from calibre_web2rag.sanitize import redact
from calibre_web2rag.sparse import SPARSE_ENCODER, sparse_vector
from calibre_web2rag.toc import book_toc, toc_text
from calibre_web2rag.works import (
    author_surname,
    book_isbn13,
    series_id_for,
    similarity,
    work_id_for,
)

logger = logging.getLogger(__name__)

SOURCE = "calibre"
BOOKTEXT_SOURCE = "calibre_booktext"
CATALOG_SOURCE = "catalog"
OVERVIEW_DOC_ID = "calibre:library-overview"
# Fuzzy work matching against other sources' books (same first-author surname).
AUTO_MERGE_SIMILARITY = 0.93
REVIEW_SIMILARITY = 0.80
# Enrichment misses are retried after this long.
ENRICHMENT_RETRY = timedelta(days=30)
# A run that sees less than this share of the previous run's books deletes nothing.
SAFETY_RATIO = 0.5

Mode = Literal["incremental", "full", "reindex"]


@dataclass(frozen=True)
class RunResult:
    status: Literal["ok", "busy", "valve"]
    counts: RunCounts
    message: str = ""


def load_taxonomy(path: str | None) -> list[str]:
    text = (
        Path(path).read_text(encoding="utf-8")
        if path
        else (files("calibre_web2rag") / "taxonomy.txt").read_text(encoding="utf-8")
    )
    slugs = [line.strip() for line in text.splitlines()]
    return [s for s in slugs if s and not s.startswith("#")]


def book_doc_id(book_id: int | str) -> str:
    return f"calibre:book:{book_id}"


def _extract_sections(fmt: str, path: Path) -> list[TextSection]:
    match fmt:
        case "PDF":
            return try_extract_pdf_sections(path)
        case "EPUB":
            return try_extract_epub_sections(path)
        case "MOBI":
            return try_extract_mobi_sections(path)
        case _:
            return []


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        qdrant: QdrantClient,
        llm: LLM | None = None,
        enricher: Enricher | None = None,
        stop: threading.Event | None = None,
        beat: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.enricher = enricher
        self.stop = stop or threading.Event()
        # Touched while work progresses, for the serve loop's health check.
        self.beat = beat or (lambda: None)
        self.taxonomy = load_taxonomy(settings.topic_taxonomy_file)
        metadata = {
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.vector_size,
            "embedding_document_prefix": settings.embedding_document_prefix,
            "embedding_query_prefix": settings.embedding_query_prefix,
            "sparse_encoder": SPARSE_ENCODER,
        }

        def store(name: str, indexes: dict[str, qm.PayloadSchemaType]) -> QdrantStore:
            return QdrantStore(
                client=qdrant,
                collection=name,
                vector_size=settings.vector_size,
                distance=settings.distance,
                metadata=metadata,
                indexes=indexes,
            )

        self.member = store(settings.member_collection, COMMON_INDEXES)
        self.catalog = store(settings.catalog_collection, COMMON_INDEXES | CATALOG_INDEXES)
        self.booktext = (
            store(settings.booktext_collection, COMMON_INDEXES)
            if settings.booktext_enabled
            else None
        )
        self.linker = Linker(
            ((r["entity_type"], r["entity_id"], r["alias"]) for r in db.aliases()),
            self._confirm_link if llm is not None else None,
        )
        self._candidates: dict[str, list[dict[str, Any]]] | None = None

    def _confirm_link(self, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.confirm_link(
                text=text, alias=alias, entity_type=entity_type, entity_id=entity_id
            )
        except LLMError as e:
            logger.warning("Link confirmation failed: %s", e)
            return False

    # -- run ------------------------------------------------------------------

    def run(self, mode: Mode) -> RunResult:
        counts = RunCounts()
        with self.db.source_lock(SOURCE) as locked:
            if not locked:
                return RunResult("busy", counts, "another calibre run holds the lock")
            run_id = f"{SOURCE}-{mode}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            self.db.start_run(run_id=run_id, source=SOURCE, mode=mode)
            try:
                result = self._run(mode, run_id, counts)
            except Exception as e:
                # Record the failure on the status row, then let main() report it.
                counts.errors.append(f"{type(e).__name__}: {e}")
                self.db.conn.rollback()
                self.db.finish_run(run_id=run_id, counts=counts, ok=False, message=str(e))
                raise
            counts.quarantined = self.db.quarantine_count([SOURCE, BOOKTEXT_SOURCE])
            self.db.finish_run(
                run_id=run_id, counts=counts, ok=result.status == "ok", message=result.message
            )
            return result

    def _run(self, mode: Mode, run_id: str, counts: RunCounts) -> RunResult:
        affected: set[str] = set()
        changed_docs: set[str] = set()
        message = ""
        status: Literal["ok", "valve"] = "ok"
        if mode != "reindex":
            with (
                snapshot(self.settings.calibre_metadata_db) as copy,
                CalibreRepository(copy, self.settings.calibre_library_root) as repo,
            ):
                versions = repo.book_versions()
                counts.seen = len(versions)
                staged = self.db.staged_versions(SOURCE)
                candidates = (
                    sorted(versions)
                    if mode == "full"
                    else [
                        i
                        for i in sorted(versions)
                        if str(i) not in staged or staged[str(i)][0] != versions[i]
                    ]
                )
                self._stage_books(repo, candidates, staged, run_id, counts, affected, changed_docs)
            if self.stop.is_set():
                # Index what was staged: the next incremental run sees those
                # books as unchanged and would never index them.
                self._index_or_forget(
                    "incremental", run_id, affected, changed_docs, allow_delete=False
                )
                return RunResult("ok", counts, "stopped early")
            deleted = sorted(set(staged) - {str(i) for i in versions})
            previous = self.db.previous_full_run_seen(SOURCE, run_id)
            baseline = max(previous or 0, len(staged))
            if deleted and counts.seen < baseline * SAFETY_RATIO:
                status = "valve"
                message = (
                    f"Saw {counts.seen} books but {baseline} were expected; not deleting "
                    f"{len(deleted)} books. Check CALIBRE_METADATA_DB."
                )
                logger.error("Safety valve: %s", message)
            elif deleted:
                affected |= self._delete_books(deleted, counts)
        self._index_or_forget(mode, run_id, affected, changed_docs, allow_delete=status == "ok")
        return RunResult(status, counts, message)

    def _index_or_forget(
        self,
        mode: Mode,
        run_id: str,
        affected: set[str],
        changed_docs: set[str],
        *,
        allow_delete: bool,
    ) -> None:
        """Index; if that fails, make the next incremental run restage what changed.

        Staging is committed first, and incremental runs skip books whose
        last_modified matches staging, so a failed index would otherwise leave
        them unindexed until the next full run.
        """
        try:
            self._index(mode, run_id, affected, changed_docs, allow_delete=allow_delete)
        except Exception:
            self.db.conn.rollback()
            self.db.forget_versions(SOURCE, sorted(changed_docs))
            self.db.commit()
            raise

    # -- staging ----------------------------------------------------------------

    def _stage_books(
        self,
        repo: CalibreRepository,
        book_ids: list[int],
        staged: dict[str, tuple[str | None, str]],
        run_id: str,
        counts: RunCounts,
        affected: set[str],
        changed_docs: set[str],
    ) -> None:
        unchanged: list[str] = []
        # Turning booktext on stages text for books that are otherwise unchanged.
        booktext_staged = (
            set(self.db.staged_versions(BOOKTEXT_SOURCE)) if self.booktext is not None else set()
        )
        for book in repo.fetch_books(book_ids):
            if self.stop.is_set():
                logger.info("Stopping early as requested")
                break
            self.beat()
            doc_id = str(book.book_id)
            raw = book_raw(book)
            raw_hash = json_hash(raw)
            if doc_id in staged and staged[doc_id] == (book.updated_at, raw_hash):
                unchanged.append(doc_id)
                if self.booktext is not None and doc_id not in booktext_staged:
                    self._stage_booktext(book, run_id)
                    self.db.commit()
                    changed_docs.add(doc_id)
                continue
            try:
                affected |= self.stage_book(book, raw, raw_hash, run_id)
            except Exception as e:
                # One bad book never stops the run.
                self.db.conn.rollback()
                logger.exception("Failed to stage book id=%s title=%r", book.book_id, book.title)
                counts.errors.append(f"book {book.book_id}: {type(e).__name__}: {e}")
                continue
            self.db.commit()
            changed_docs.add(doc_id)
            if doc_id in staged:
                counts.updated += 1
            else:
                counts.added += 1
        self.db.touch_staging(SOURCE, unchanged, run_id)
        self.db.touch_staging(BOOKTEXT_SOURCE, unchanged, run_id)
        self.db.commit()

    def _sanitize(self, text: str, doc_source: str, doc_id: str, field: str, run_id: str) -> str:
        result = redact(text)
        if result.findings:
            logger.warning("Redacted %s secret(s) in %s %s", len(result.findings), doc_id, field)
            self.db.add_quarantine(
                source=doc_source,
                source_doc_id=doc_id,
                reason="secret-redacted",
                detail={"field": field, "rules": result.rules},
                run_id=run_id,
            )
        return result.text

    # Any: enrichment rows are JSON.
    def _enrich(self, book: BookRecord, isbn13: str | None) -> dict[str, Any]:
        doc_id = str(book.book_id)
        rows = self.db.enrichment_get(SOURCE, doc_id)
        fresh = bool(rows) and all(
            r["matched"] or datetime.now(UTC) - r["fetched_at"] < ENRICHMENT_RETRY for r in rows
        )
        if not fresh and self.enricher is not None:
            for result in self.enricher.lookup(
                isbn13=isbn13, title=book.title, authors=book.authors
            ):
                self.db.enrichment_put(
                    source=SOURCE,
                    source_doc_id=doc_id,
                    origin=result.origin,
                    lookup_key=result.lookup_key,
                    matched=result.matched,
                    confidence=result.confidence,
                    fields=result.fields,
                )
            rows = self.db.enrichment_get(SOURCE, doc_id)
        for row in rows:
            if row["matched"] and (
                row["fields"].get("description") or row["fields"].get("subjects")
            ):
                return {"origin": row["origin"], **row["fields"]}
        return {}

    def stage_book(
        # Any: the staged raw record is JSON.
        self,
        book: BookRecord,
        raw: dict[str, Any],
        raw_hash: str,
        run_id: str,
    ) -> set[str]:
        """Stage one book and attach it to its work; returns the works it touched."""
        doc_id = str(book.book_id)
        description = self._sanitize(
            html_to_text(book.comments), SOURCE, doc_id, "description", run_id
        )
        toc = book_toc(book.files)
        toc_lines = self._sanitize(toc_text(toc), SOURCE, doc_id, "toc", run_id).splitlines()
        isbn13 = book_isbn13(book.identifiers)

        enriched: dict[str, Any] = {}
        if self.settings.enrichment_enabled and (not description or not book.tags):
            enriched = self._enrich(book, isbn13)
        enriched_description = self._sanitize(
            str(enriched.get("description") or ""), SOURCE, doc_id, "enriched_description", run_id
        )
        enriched_subjects = [str(s) for s in enriched.get("subjects") or []]

        topics: list[str] = []
        marketing = False
        good_for = ""
        if self.llm is not None:
            try:
                analysis = self.llm.analyze_book(
                    title=book.title,
                    authors=book.authors,
                    description=description or enriched_description,
                    tags=book.tags + enriched_subjects,
                    toc="\n".join(toc_lines),
                    taxonomy=self.taxonomy,
                )
                topics = analysis.topics
                marketing = analysis.description_is_marketing and bool(description)
                good_for = analysis.good_for
            except LLMError as e:
                logger.warning("Book analysis failed for id=%s: %s", book.book_id, e)

        url = build_book_url(
            book_id=book.book_id,
            base_url=self.settings.calibre_web_base_url,
            template=self.settings.calibre_web_book_url_template,
        )
        formats = [
            {
                "format": f.format,
                "size_bytes": f.size_bytes,
                "available": f.exists,
                "url": build_ebook_url(
                    book_id=book.book_id,
                    fmt=f.format,
                    base_url=self.settings.calibre_web_base_url,
                    template=self.settings.calibre_download_url_template,
                ),
            }
            for f in book.files
        ]
        needs_summary = not description or marketing
        details: dict[str, Any] = {
            "title": book.title,
            "subtitle": book.subtitle,
            "authors": book.authors,
            "series": book.series,
            "series_index": book.series_index,
            "publisher": book.publisher,
            "year": year_of(book.published_at),
            "languages": book.languages,
            "tags": book.tags,
            "topics": topics,
            "identifiers": book.identifiers,
            "rating": book.rating,
            "description": description,
            "description_is_marketing": marketing,
            "enriched_description": enriched_description or None,
            "enriched_subjects": enriched_subjects,
            "enrichment_origin": enriched.get("origin"),
            "good_for": good_for if needs_summary and good_for else None,
            "good_for_generated": bool(needs_summary and good_for),
            "toc": toc_lines,
            "formats": formats,
            "cover_path": book.cover_path,
            "date_added": to_rfc3339(book.added_at),
            "updated_at": to_rfc3339(book.updated_at),
            "custom_columns": raw["custom_columns"],
        }
        flags = {"secret_redacted": "[REDACTED]" in (description + "\n".join(toc_lines))}
        extracted = "\n\n".join(p for p in (description, "\n".join(toc_lines)) if p)
        self.db.upsert_staging(
            StagedDocument(
                source=SOURCE,
                source_doc_id=doc_id,
                title=book.title,
                url=url,
                tier=TIER,
                raw_content=raw,
                extracted_text=extracted,
                content_hash=raw_hash,
                source_updated_at=book.updated_at,
                flags=flags,
            ),
            run_id=run_id,
        )

        work_id = self._resolve_work(book, isbn13)
        series_id = series_id_for(book.series) if book.series else None
        if series_id and book.series:
            self.db.upsert_series(series_id, book.series)
        self.db.upsert_work(
            work_id=work_id,
            title=book.title,
            authors=book.authors,
            isbn13=isbn13,
            series_id=series_id,
            series_index=book.series_index,
        )
        previous = self.db.upsert_manifestation(
            source=SOURCE,
            source_id=doc_id,
            work_id=work_id,
            format="ebook",
            url=url,
            details=details,
        )
        self.db.set_work_topics(work_id)
        touched = {work_id}
        if previous and previous != work_id:
            touched.add(previous)
            self.db.set_work_topics(previous)
            self.db.delete_work_if_empty(previous)

        self.db.upsert_catalog_item(
            {
                "source": SOURCE,
                "source_id": doc_id,
                "work_id": work_id,
                "format": "ebook",
                "title": book.title,
                "sort_title": book.sort_title,
                "subtitle": book.subtitle,
                "authors": book.authors,
                "series": book.series,
                "series_index": book.series_index,
                "publisher": book.publisher,
                "published_year": details["year"],
                "languages": book.languages,
                "tags": book.tags,
                "topics": topics,
                "file_formats": [f.format for f in book.files],
                "rating": book.rating,
                "url": url,
                "location": None,
                "date_added": to_rfc3339(book.added_at),
            }
        )
        if self.booktext is not None:
            self._stage_booktext(book, run_id)
        return touched

    def _stage_booktext(self, book: BookRecord, run_id: str) -> None:
        doc_id = str(book.book_id)
        by_format = {f.format: f for f in book.files if f.exists and f.format in TEXT_FORMATS}
        for fmt in ("EPUB", "PDF", "MOBI"):
            if fmt not in by_format:
                continue
            sections = _extract_sections(fmt, by_format[fmt].file_path)
            if not sections:
                continue
            clean = [
                {
                    "title": s.title,
                    "page": s.page,
                    "text": self._sanitize(s.text, BOOKTEXT_SOURCE, doc_id, f"{fmt} text", run_id),
                }
                for s in sections
            ]
            text = "\n\n".join(str(s["text"]) for s in clean)
            self.db.upsert_staging(
                StagedDocument(
                    source=BOOKTEXT_SOURCE,
                    source_doc_id=doc_id,
                    title=book.title,
                    url=None,
                    tier=TIER,
                    raw_content={"format": fmt, "sections": clean},
                    extracted_text=text,
                    content_hash=text_hash(text),
                    source_updated_at=book.updated_at,
                ),
                run_id=run_id,
            )
            return
        logger.info("No readable DRM-free text for book id=%s", book.book_id)
        # Staged as not indexable, so the next run doesn't try again.
        self.db.upsert_staging(
            StagedDocument(
                source=BOOKTEXT_SOURCE,
                source_doc_id=doc_id,
                title=book.title,
                url=None,
                tier=TIER,
                raw_content={"format": None, "sections": []},
                extracted_text="",
                content_hash=text_hash(""),
                source_updated_at=book.updated_at,
                indexable=False,
            ),
            run_id=run_id,
        )

    # -- works ------------------------------------------------------------------

    def _candidate_index(self) -> dict[str, list[dict[str, Any]]]:
        if self._candidates is None:
            index: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self.db.match_candidates(exclude_source=SOURCE):
                surname = author_surname(row["authors"][0] if row["authors"] else None)
                index[surname].append(row)
            self._candidates = dict(index)
        return self._candidates

    def _resolve_work(self, book: BookRecord, isbn13: str | None) -> str:
        natural = work_id_for(title=book.title, authors=book.authors, isbn13=isbn13)
        canonical = self.db.merge_target(natural)
        if canonical != natural or self.db.get_work(natural) is not None:
            return canonical
        surname = author_surname(book.authors[0] if book.authors else None)
        distinct = self.db.distinct_pairs(natural)
        best: tuple[float, dict[str, Any]] | None = None
        for row in self._candidate_index().get(surname, []) if surname else []:
            if row["work_id"] in distinct:
                continue
            score = similarity(book.title, row["title"])
            if best is None or score > best[0]:
                best = (score, row)
        if best is None:
            return natural
        score, row = best
        if score >= AUTO_MERGE_SIMILARITY:
            logger.info("Merging %s into %s (similarity %.2f)", natural, row["work_id"], score)
            self.db.record_merge(
                from_work_id=natural,
                to_work_id=row["work_id"],
                decided_by="auto",
                similarity=score,
            )
            return str(row["work_id"])
        if score >= REVIEW_SIMILARITY:
            self.db.queue_review(
                kind="work_merge",
                left_id=natural,
                right_id=str(row["work_id"]),
                similarity=score,
                details={"left_title": book.title, "right_title": row["title"]},
            )
        return natural

    def _delete_books(self, doc_ids: list[str], counts: RunCounts) -> set[str]:
        works = set(self.db.delete_manifestations(SOURCE, doc_ids))
        self.db.delete_catalog_items(SOURCE, doc_ids)
        self.db.delete_enrichment(SOURCE, doc_ids)
        counts.deleted += self.db.delete_staging(SOURCE, doc_ids)
        self.db.delete_staging(BOOKTEXT_SOURCE, doc_ids)
        for work_id in works:
            self.db.set_work_topics(work_id)
        self.db.commit()
        if self.booktext is not None:
            self.booktext.delete_docs(SOURCE, [book_doc_id(i) for i in doc_ids])
        logger.info("Deleted %s books no longer in Calibre", len(doc_ids))
        return works

    # -- indexing ---------------------------------------------------------------

    def _embed_points(
        self,
        store: QdrantStore,
        # Any: payloads are JSON.
        items: list[tuple[str, dict[str, Any]]],
        *,
        force: bool,
    ) -> int:
        """Upsert (point id, payload) items, embedding only changed text.

        A point whose text is unchanged but whose other payload fields changed
        (year, languages, url, manifestations) gets its payload rewritten
        without re-embedding.
        """
        if not items:
            return 0
        for _, payload in items:
            payload["payload_hash"] = payload_hash(payload)
        existing = {} if force else store.existing_state(i for i, _ in items)
        todo = [
            (i, p)
            for i, p in items
            if force or (existing.get(i) or (None,))[0] != p["content_hash"]
        ]
        same_text = [
            (i, p)
            for i, p in items
            if not force and i in existing and existing[i][0] == p["content_hash"]
        ]
        stale = [(i, p) for i, p in same_text if existing[i][1] != p["payload_hash"]]
        current = [i for i, p in same_text if existing[i][1] == p["payload_hash"]]
        store.replace_payloads(stale)
        if current:
            store.set_payload(current, {"sync_run_id": items[0][1]["sync_run_id"]})
        size = self.settings.batch_size
        for start in range(0, len(todo), size):
            self.beat()
            batch = todo[start : start + size]
            texts = [
                f"{p['context']}\n\n{p['text']}" if p["context"] else p["text"] for _, p in batch
            ]
            vectors = self.embedder.encode(texts, kind="document")
            store.upsert(
                [
                    qm.PointStruct(
                        id=pid,
                        vector={"dense": vector, "sparse": sparse_vector(text)},
                        payload=payload,
                    )
                    for (pid, payload), text, vector in zip(batch, texts, vectors, strict=True)
                ]
            )
        return len(todo)

    def _index(
        self,
        mode: Mode,
        run_id: str,
        affected: set[str],
        changed_docs: set[str],
        *,
        allow_delete: bool,
    ) -> None:
        force = mode == "reindex"
        if mode == "incremental":
            work_ids = sorted(affected)
        else:
            work_ids = sorted(set(self.db.work_ids_for_source(SOURCE)) | affected)
        self.index_catalog(work_ids, run_id, force=force)
        self.index_overview(run_id, force=force)
        if self.booktext is not None:
            docs = None if mode != "incremental" else sorted(changed_docs)
            self.index_booktext(docs, run_id, force=force)
        if mode in ("full", "reindex") and allow_delete:
            for store in (self.member, self.booktext):
                if store is not None:
                    removed = store.delete_stale(SOURCE, run_id)
                    if removed:
                        logger.info("Deleted %s stale points from %s", removed, store.collection)

    def index_catalog(self, work_ids: list[str], run_id: str, *, force: bool = False) -> None:
        items: list[tuple[str, dict[str, Any]]] = []
        gone: list[str] = []
        for work_id in work_ids:
            pid = point_id(CATALOG_SOURCE, work_id, 0)
            work = self.db.get_work(work_id)
            manifestations = self.db.manifestations(work_id) if work else []
            if work is None or not manifestations:
                gone.append(pid)
                if work is not None:
                    self.db.delete_work_if_empty(work_id)
                continue
            items.append((pid, self._catalog_payload(work, manifestations, run_id)))
        self.db.commit()
        self.catalog.delete_ids(gone)
        embedded = self._embed_points(self.catalog, items, force=force)
        logger.info(
            "Catalog: %s works checked, %s embedded, %s removed", len(items), embedded, len(gone)
        )

    # Any: registry rows are JSON.
    def _catalog_payload(
        self, work: dict[str, Any], manifestations: list[dict[str, Any]], run_id: str
    ) -> dict[str, Any]:
        series = self.db.series_name(work["series_id"]) if work.get("series_id") else None
        text = catalog_text(work, manifestations, series)
        details = [m.get("details") or {} for m in manifestations]
        authors = list(work.get("authors") or [])
        context = (
            f"Hackerspace library catalog record for {work['title']!r}"
            + (f" by {', '.join(authors)}" if authors else "")
            + "."
        )
        links = self.linker.link(text)
        added = sorted(d["date_added"] for d in details if d.get("date_added"))
        updated = sorted(d["updated_at"] for d in details if d.get("updated_at"))
        payload = base_payload(
            source=CATALOG_SOURCE,
            doc_id=str(work["work_id"]),
            chunk_index=0,
            title=str(work["title"]),
            text=text,
            context=context,
            url=next((m["url"] for m in manifestations if m.get("url")), None),
            run_id=run_id,
            created_at=added[0] if added else None,
            updated_at=updated[-1] if updated else None,
            lang=next((d["languages"][0] for d in details if d.get("languages")), None),
            topics=list(work.get("topics") or []),
            work_ids=[str(work["work_id"])],
            links=links.ids,
        )
        payload.update(
            {
                "work_id": work["work_id"],
                "authors": authors,
                "series": series,
                "series_id": work.get("series_id"),
                "series_index": work.get("series_index"),
                "year": next((d["year"] for d in details if d.get("year")), None),
                "languages": sorted({lang for d in details for lang in d.get("languages") or []}),
                "tags": sorted({t for d in details for t in d.get("tags") or []}),
                "formats": sorted({str(m["format"]) for m in manifestations}),
                "narrators": sorted({n for d in details for n in d.get("narrators") or []}),
                "duration_seconds": next(
                    (d["duration_seconds"] for d in details if d.get("duration_seconds")), None
                ),
                "manifestations": [
                    {
                        "format": m["format"],
                        "source": m["source"],
                        "source_id": m["source_id"],
                        "url": m.get("url"),
                        "location": (m.get("details") or {}).get("location"),
                        "files": (m.get("details") or {}).get("formats", []),
                        "cover_path": (m.get("details") or {}).get("cover_path"),
                        "narrators": (m.get("details") or {}).get("narrators", []),
                        "duration_seconds": (m.get("details") or {}).get("duration_seconds"),
                    }
                    for m in manifestations
                ],
                "date_added": added[0] if added else None,
                "description_generated": any(d.get("good_for_generated") for d in details),
            }
        )
        return payload

    def index_overview(self, run_id: str, *, force: bool = False) -> None:
        rows = self.db.catalog_rows(SOURCE)
        sections = overview_sections(
            rows,
            library_url=self.settings.calibre_web_base_url,
            request_info=self.settings.library_request_info,
        )
        document = "\n\n".join(f"{h}\n{b}" for h, b in sections)
        title = sections[0][0]
        now = datetime.now(UTC).isoformat()
        # Chunk 0 is the document-level summary point; then one per section.
        pieces: list[tuple[list[str], str]] = [([title], f"{title}\n{sections[0][1]}")]
        for heading, body in sections[1:]:
            for chunk in split_text(body, self.settings.chunk_size, self.settings.chunk_overlap):
                pieces.append(([title, heading], f"{heading}\n{chunk}"))
        items: list[tuple[str, dict[str, Any]]] = []
        for index, (path, text) in enumerate(pieces):
            context = self._context(
                title, document, text, fallback=f"Part of the {title} overview."
            )
            items.append(
                (
                    point_id(SOURCE, OVERVIEW_DOC_ID, index),
                    base_payload(
                        source=SOURCE,
                        doc_id=OVERVIEW_DOC_ID,
                        chunk_index=index,
                        title=title,
                        text=text,
                        context=context,
                        url=self.settings.calibre_web_base_url,
                        run_id=run_id,
                        created_at=None,
                        updated_at=now,
                        heading_path=path,
                        lang="en",
                        topics=["reference", "hackerspace-and-community"],
                        links=self.linker.link(text).ids,
                    ),
                )
            )
        existing = self.member.existing_hashes(i for i, _ in items)
        if force or any(existing.get(i) != p["content_hash"] for i, p in items):
            self.member.delete_docs(SOURCE, [OVERVIEW_DOC_ID])
            self._embed_points(self.member, items, force=True)
        else:
            self.member.set_payload([i for i, _ in items], {"sync_run_id": run_id})

    def _context(self, title: str, document: str, chunk: str, *, fallback: str) -> str:
        if self.llm is None:
            return fallback
        self.beat()
        try:
            return self.llm.chunk_context(document_title=title, document=document, chunk=chunk)
        except LLMError as e:
            logger.warning("Chunk context failed: %s", e)
            return fallback

    def index_booktext(self, doc_ids: list[str] | None, run_id: str, *, force: bool) -> None:
        if self.booktext is None:
            return
        rows = self.db.staged_documents(BOOKTEXT_SOURCE, doc_ids)
        books = {
            r["source_doc_id"]: r
            for r in self.db.staged_documents(SOURCE, [r["source_doc_id"] for r in rows])
        }
        for row in rows:
            doc_id = book_doc_id(row["source_doc_id"])
            first = point_id(SOURCE, doc_id, 0)
            if (
                not force
                and self.booktext.existing_hashes([first]).get(first) == row["content_hash"]
            ):
                self.booktext.set_payload_for_doc(SOURCE, doc_id, {"sync_run_id": run_id})
                continue
            book = books.get(row["source_doc_id"])
            self.booktext.delete_docs(SOURCE, [doc_id])
            self._embed_points(
                self.booktext, self._booktext_items(row, book, doc_id, run_id), force=True
            )

    def _booktext_items(
        # Any: staging rows are JSON.
        self,
        row: dict[str, Any],
        book: dict[str, Any] | None,
        doc_id: str,
        run_id: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        raw = (book or {}).get("raw_content") or {}
        title = str(row["title"] or "")
        authors = ", ".join(raw.get("authors") or [])
        items: list[tuple[str, dict[str, Any]]] = []
        index = 0
        for section in row["raw_content"].get("sections", []):
            chapter = section.get("title")
            where = f", chapter {chapter!r}" if chapter else ""
            where += f", page {section['page']}" if section.get("page") else ""
            context = f"From {title!r}" + (f" by {authors}" if authors else "") + where + "."
            for chunk in split_text(
                section["text"], self.settings.chunk_size, self.settings.chunk_overlap
            ):
                payload = base_payload(
                    source=SOURCE,
                    doc_id=doc_id,
                    chunk_index=index,
                    title=title,
                    text=chunk,
                    context=context,
                    url=(book or {}).get("url"),
                    run_id=run_id,
                    created_at=to_rfc3339(raw.get("added_at")),
                    updated_at=to_rfc3339(raw.get("updated_at")),
                    heading_path=[title, chapter] if chapter else [title],
                    page=section.get("page"),
                    lang=(raw.get("languages") or [None])[0],
                )
                # The whole book's hash on chunk 0 says whether to re-embed it.
                if index == 0:
                    payload["content_hash"] = row["content_hash"]
                items.append((point_id(SOURCE, doc_id, index), payload))
                index += 1
        return items


def build_pipeline(
    settings: Settings,
    db: Database,
    stop: threading.Event | None,
    beat: Callable[[], None] | None = None,
) -> Pipeline:
    llm = (
        LLM(
            model=settings.llm_model,
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            cache=db,
        )
        if settings.llm_model
        else None
    )
    if llm is None:
        logger.warning("LLM_MODEL is not set: no topics, summaries or chunk context this run")
    enricher = (
        Enricher(
            timeout_seconds=settings.enrichment_timeout_seconds,
            google_api_key=settings.google_books_api_key,
        )
        if settings.enrichment_enabled
        else None
    )
    return Pipeline(
        settings=settings,
        db=db,
        embedder=build_embedder(settings),
        qdrant=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        llm=llm,
        enricher=enricher,
        stop=stop,
        beat=beat,
    )


def run(
    settings: Settings,
    mode: Mode,
    stop: threading.Event | None = None,
    beat: Callable[[], None] | None = None,
) -> RunResult:
    with Database.connect(settings.database_url) as db:
        return build_pipeline(settings, db, stop, beat).run(mode)


def last_full_run(settings: Settings) -> datetime | None:
    """When the last full run finished: the serve loop's cue for the next one."""
    with Database.connect(settings.database_url) as db:
        return db.last_full_run_at(SOURCE)


# Any: status rows are JSON.
def status(settings: Settings) -> dict[str, Any]:
    """Recent runs, quarantine and catalog counts: the source's status page."""
    with Database.connect(settings.database_url) as db:
        return {
            "source": SOURCE,
            "runs": db.recent_runs(SOURCE),
            "quarantined": db.quarantine_count([SOURCE, BOOKTEXT_SOURCE]),
            "catalog_ebooks": db.catalog_count(source=SOURCE, format="ebook"),
        }
