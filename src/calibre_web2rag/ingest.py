from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

from qdrant_client.models import PointStruct

from calibre_web2rag.calibre_db import CalibreRepository
from calibre_web2rag.chunking import split_text
from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import Embedder, build_embedder
from calibre_web2rag.extractors import (
    extract_epub_text,
    extract_mobi_text,
    extract_pdf_text,
    is_epub_drm_free,
    is_mobi_drm_free,
    is_pdf_drm_free,
)
from calibre_web2rag.links import build_ebook_url, build_opds_url
from calibre_web2rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _extract_text(fmt: str, path: str) -> str:
    from pathlib import Path

    p = Path(path)
    if fmt == "PDF":
        return extract_pdf_text(p)
    if fmt == "EPUB":
        return extract_epub_text(p)
    if fmt == "MOBI":
        return extract_mobi_text(p)
    return ""


def _drm_free(fmt: str, path: str) -> bool:
    from pathlib import Path

    p = Path(path)
    if fmt == "PDF":
        return is_pdf_drm_free(p)
    if fmt == "EPUB":
        return is_epub_drm_free(p)
    if fmt == "MOBI":
        return is_mobi_drm_free(p)
    return False


def _point_id(value: str) -> str:
    # Qdrant point IDs must be uint64 or UUID; use deterministic UUID5 for stable re-ingestion.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


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
    for book in repo.fetch_books():
        logger.info(
            "Analyzing book id=%s title=%r files=%s",
            book.book_id,
            book.title,
            len(book.files),
        )
        book_chunk_count = 0
        for file in book.files:
            file_path = str(file.file_path)
            logger.info("Reading file for book id=%s format=%s path=%s", book.book_id, file.format, file_path)
            if not _drm_free(file.format, file_path):
                logger.info("Skipping DRM-protected or unreadable file: %s", file_path)
                continue
            text = _extract_text(file.format, file_path)
            chunks = split_text(text=text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
            if not chunks:
                logger.info("No extractable text for file: %s", file_path)
                continue
            logger.info(
                "Extracted %s chunks for book id=%s format=%s",
                len(chunks),
                book.book_id,
                file.format,
            )
            vectors = embedder.encode(chunks)
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                source_url = build_ebook_url(
                    book_id=book.book_id,
                    fmt=file.format,
                    base_url=settings.calibre_web_base_url,
                    template=settings.calibre_download_url_template,
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
                    "file_format": file.format,
                    "file_name": file.file_name,
                    "file_path": file_path,
                    "source_url": source_url,
                    "opds_url": build_opds_url(
                        book_id=book.book_id,
                        base_url=settings.calibre_web_base_url,
                    ),
                    "chunk_index": idx,
                    "chunk_text": chunk,
                }
                point = PointStruct(
                    id=_point_id(f"{book.book_id}:{file.format}:{idx}:{chunk[:32]}"),
                    vector=vector,
                    payload=payload,
                )
                batch.append(point)
                book_chunk_count += 1
                if len(batch) >= settings.batch_size:
                    yield batch
                    batch = []
        logger.info(
            "Finished book id=%s title=%r stored_chunks=%s",
            book.book_id,
            book.title,
            book_chunk_count,
        )
    if batch:
        yield batch
