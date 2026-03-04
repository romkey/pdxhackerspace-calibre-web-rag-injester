from __future__ import annotations

import logging
import uuid
from collections import Counter
from collections.abc import Iterator

from bs4 import BeautifulSoup
from qdrant_client.models import PointStruct

from calibre_web2rag.calibre_db import CalibreRepository
from calibre_web2rag.chunking import split_text
from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import Embedder, build_embedder
from calibre_web2rag.extractors import (
    try_extract_epub_sections,
    try_extract_mobi_sections,
    try_extract_pdf_sections,
)
from calibre_web2rag.links import build_ebook_url, build_opds_url
from calibre_web2rag.models import BookRecord, TextSection
from calibre_web2rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _safe_extract_sections(fmt: str, path: str) -> list[TextSection]:
    """Combined DRM-check and extraction in a single pass per format."""
    from pathlib import Path

    p = Path(path)
    if fmt == "PDF":
        return try_extract_pdf_sections(p)
    if fmt == "EPUB":
        return try_extract_epub_sections(p)
    if fmt == "MOBI":
        return try_extract_mobi_sections(p)
    return []


def _point_id(value: str) -> str:
    # Qdrant point IDs must be uint64 or UUID; use deterministic UUID5 for stable re-ingestion.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _strip_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def _contextualize(book: BookRecord, chunk: str, *, chapter: str | None = None) -> str:
    """Prepend metadata to chunk text so the embedding captures book context."""
    parts = [f"Title: {book.title}"]
    if book.authors:
        parts.append(f"Authors: {', '.join(book.authors)}")
    if book.tags:
        parts.append(f"Tags: {', '.join(book.tags)}")
    if book.series:
        parts.append(f"Series: {book.series}")
    if chapter:
        parts.append(f"Chapter: {chapter}")
    header = " | ".join(parts)
    return f"{header}\n\n{chunk}"


# ---------------------------------------------------------------------------
# Book-level metadata text (one point per book for catalog queries)
# ---------------------------------------------------------------------------


def _build_book_metadata_text(book: BookRecord) -> str:
    lines = [book.title]
    if book.authors:
        lines.append(f"by {', '.join(book.authors)}")
    details: list[str] = []
    if book.publisher:
        details.append(f"Publisher: {book.publisher}")
    if book.series:
        details.append(f"Series: {book.series}")
    if book.tags:
        details.append(f"Categories: {', '.join(book.tags)}")
    if book.languages:
        details.append(f"Languages: {', '.join(book.languages)}")
    if book.isbn:
        details.append(f"ISBN: {book.isbn}")
    if book.rating is not None:
        details.append(f"Rating: {book.rating}/10")
    if book.published_at:
        details.append(f"Published: {book.published_at}")
    formats = [f.format for f in book.files]
    if formats:
        details.append(f"Formats: {', '.join(formats)}")
    if details:
        lines.append(". ".join(details) + ".")
    if book.comments:
        desc = _strip_html(book.comments)
        if desc:
            lines.append(desc)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Library-level summary (aggregate stats across all books)
# ---------------------------------------------------------------------------


def _build_library_summary_text(books: list[BookRecord]) -> str:
    author_counts: Counter[str] = Counter()
    all_tags: set[str] = set()
    all_series: set[str] = set()
    all_languages: set[str] = set()
    format_counts: Counter[str] = Counter()

    for book in books:
        for author in book.authors:
            author_counts[author] += 1
        all_tags.update(book.tags)
        if book.series:
            all_series.add(book.series)
        all_languages.update(book.languages)
        for f in book.files:
            format_counts[f.format] += 1

    lines = [
        f"This ebook library contains {len(books)} books "
        f"by {len(author_counts)} authors.",
    ]

    if format_counts:
        fmt_parts = [f"{count} {fmt}" for fmt, count in format_counts.most_common()]
        lines.append(f"Formats: {', '.join(fmt_parts)}.")

    if all_languages:
        lines.append(f"Languages: {', '.join(sorted(all_languages))}.")

    if author_counts:
        author_entries = [
            f"{name} ({count} {'book' if count == 1 else 'books'})"
            for name, count in author_counts.most_common()
        ]
        lines.append(f"Authors: {', '.join(author_entries)}.")

    if all_tags:
        lines.append(f"Categories: {', '.join(sorted(all_tags))}.")

    if all_series:
        lines.append(f"Series: {', '.join(sorted(all_series))}.")

    return "\n".join(lines)


def _build_library_summary_points(
    *,
    books: list[BookRecord],
    embedder: Embedder,
    settings: Settings,
) -> list[PointStruct]:
    summary_text = _build_library_summary_text(books)
    chunks = split_text(
        text=summary_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        return []

    vectors = embedder.encode(chunks)
    points: list[PointStruct] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        payload = {
            "book_id": None,
            "title": None,
            "authors": [],
            "tags": [],
            "publisher": None,
            "series": None,
            "rating": None,
            "languages": [],
            "identifiers": {},
            "isbn": None,
            "uuid": None,
            "published_at": None,
            "updated_at": None,
            "comment": None,
            "library_path": None,
            "file_format": None,
            "file_name": None,
            "file_path": None,
            "source_url": None,
            "opds_url": None,
            "chapter_title": None,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_type": "library_summary",
            "chunk_text": chunk,
        }
        points.append(
            PointStruct(
                id=_point_id(f"library_summary:{idx}:{chunk[:32]}"),
                vector=vector,
                payload=payload,
            )
        )

    logger.info("Indexed %s library summary chunks", len(points))
    return points


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest(settings: Settings) -> int:
    repo = CalibreRepository(settings.calibre_metadata_db, settings.calibre_library_root)
    embedder = build_embedder(settings)
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.vector_size,
        distance=settings.distance,
    )

    inserted = 0
    for batch in _generate_points(repo=repo, embedder=embedder, settings=settings):
        store.upsert(batch)
        inserted += len(batch)
        logger.info("Upserted %s chunks to Qdrant (total=%s)", len(batch), inserted)
    return inserted


def _generate_points(
    *,
    repo: CalibreRepository,
    embedder: Embedder,
    settings: Settings,
) -> Iterator[list[PointStruct]]:
    batch: list[PointStruct] = []
    books = repo.fetch_books()

    for book in books:
        try:
            new_points = _process_book(book=book, embedder=embedder, settings=settings)
            for point in new_points:
                batch.append(point)
                if len(batch) >= settings.batch_size:
                    yield batch
                    batch = []
        except Exception:
            logger.exception(
                "Failed to process book id=%s title=%r, skipping",
                book.book_id,
                book.title,
            )
            continue

    # Library-wide catalog summary
    if books:
        for point in _build_library_summary_points(
            books=books, embedder=embedder, settings=settings
        ):
            batch.append(point)
            if len(batch) >= settings.batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def _process_book(
    *,
    book: BookRecord,
    embedder: Embedder,
    settings: Settings,
) -> list[PointStruct]:
    points: list[PointStruct] = []
    logger.info(
        "Analyzing book id=%s title=%r files=%s",
        book.book_id,
        book.title,
        len(book.files),
    )

    # Per-book metadata point for catalog queries
    metadata_text = _contextualize(book, _build_book_metadata_text(book))
    vector = embedder.encode([metadata_text])[0]
    points.append(
        _build_point(
            book=book,
            file_format=None,
            file_name=None,
            file_path=None,
            chapter_title=None,
            chunk=metadata_text,
            chunk_index=0,
            total_chunks=1,
            chunk_type="book_metadata",
            vector=vector,
            settings=settings,
        )
    )

    # Index book description if available
    if book.comments:
        description = _strip_html(book.comments)
        desc_chunks = split_text(
            text=description,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if desc_chunks:
            contextualized = [_contextualize(book, c) for c in desc_chunks]
            vectors = embedder.encode(contextualized)
            for idx, (ctx_chunk, vec) in enumerate(
                zip(contextualized, vectors, strict=True)
            ):
                points.append(
                    _build_point(
                        book=book,
                        file_format=None,
                        file_name=None,
                        file_path=None,
                        chapter_title=None,
                        chunk=ctx_chunk,
                        chunk_index=idx,
                        total_chunks=len(desc_chunks),
                        chunk_type="description",
                        vector=vec,
                        settings=settings,
                    )
                )
            logger.info(
                "Indexed %s description chunks for book id=%s",
                len(desc_chunks),
                book.book_id,
            )

    # Index content from each file
    for file in book.files:
        file_path = str(file.file_path)
        logger.info(
            "Reading file for book id=%s format=%s path=%s",
            book.book_id,
            file.format,
            file_path,
        )
        sections = _safe_extract_sections(file.format, file_path)
        if not sections:
            logger.info(
                "Skipping unreadable or empty file: %s", file_path
            )
            continue

        all_items: list[tuple[str | None, str]] = []
        for section in sections:
            section_chunks = split_text(
                text=section.text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            for chunk in section_chunks:
                all_items.append((section.title, chunk))

        if not all_items:
            logger.info("No extractable text for file: %s", file_path)
            continue

        logger.info(
            "Extracted %s chunks for book id=%s format=%s",
            len(all_items),
            book.book_id,
            file.format,
        )

        contextualized = [
            _contextualize(book, chunk, chapter=title)
            for title, chunk in all_items
        ]
        vectors = embedder.encode(contextualized)

        for idx, ((chapter_title, _raw), ctx_chunk, vec) in enumerate(
            zip(all_items, contextualized, vectors, strict=True)
        ):
            points.append(
                _build_point(
                    book=book,
                    file_format=file.format,
                    file_name=file.file_name,
                    file_path=file_path,
                    chapter_title=chapter_title,
                    chunk=ctx_chunk,
                    chunk_index=idx,
                    total_chunks=len(all_items),
                    chunk_type="content",
                    vector=vec,
                    settings=settings,
                )
            )

    logger.info(
        "Finished book id=%s title=%r stored_chunks=%s",
        book.book_id,
        book.title,
        len(points),
    )
    return points


def _build_point(
    *,
    book: BookRecord,
    file_format: str | None,
    file_name: str | None,
    file_path: str | None,
    chapter_title: str | None,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    chunk_type: str,
    vector: list[float],
    settings: Settings,
) -> PointStruct:
    source_url = (
        build_ebook_url(
            book_id=book.book_id,
            fmt=file_format or "",
            base_url=settings.calibre_web_base_url,
            template=settings.calibre_download_url_template,
        )
        if file_format
        else None
    )

    payload = {
        "book_id": book.book_id,
        "title": book.title,
        "authors": book.authors,
        "tags": book.tags,
        "publisher": book.publisher,
        "series": book.series,
        "rating": book.rating,
        "languages": book.languages,
        "identifiers": book.identifiers,
        "isbn": book.isbn,
        "uuid": book.uuid,
        "published_at": book.published_at,
        "updated_at": book.updated_at,
        "comment": book.comments,
        "library_path": book.path,
        "file_format": file_format,
        "file_name": file_name,
        "file_path": file_path,
        "source_url": source_url,
        "opds_url": build_opds_url(
            book_id=book.book_id,
            base_url=settings.calibre_web_base_url,
        ),
        "chapter_title": chapter_title,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_type": chunk_type,
        "chunk_text": chunk,
    }

    fmt_part = file_format or chunk_type
    return PointStruct(
        id=_point_id(f"{book.book_id}:{fmt_part}:{chunk_index}:{chunk[:32]}"),
        vector=vector,
        payload=payload,
    )
