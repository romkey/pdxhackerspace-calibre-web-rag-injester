from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibre_web2rag.config import Settings
from calibre_web2rag.models import BookRecord, CalibreFile


def _base_settings(tmp_path: Path) -> Settings:
    return Settings(
        calibre_metadata_db="db.sqlite",
        calibre_library_root=str(tmp_path),
        calibre_web_base_url="https://books.example",
        calibre_download_url_template=None,
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="books",
        embedding_provider="ollama",
        embedding_model="dummy",
        hf_cache_dir=None,
        ollama_url="http://localhost:11434",
        ollama_api_key=None,
        ollama_timeout_seconds=30,
        chunk_size=40,
        chunk_overlap=10,
        batch_size=100,
        distance="cosine",
        vector_size=3,
        embedding_context_length=8192,
    )


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Settings for tests; keyword arguments override fields."""

    # Any: overrides may be any Settings field; dataclasses.replace checks names.
    def _make(**overrides: Any) -> Settings:
        return dataclasses.replace(_base_settings(tmp_path), **overrides)

    return _make


@pytest.fixture
def make_book(tmp_path: Path) -> Callable[..., BookRecord]:
    """A book with one (fake) file; keyword arguments override fields."""

    # Any: overrides may be any BookRecord field; dataclasses.replace checks names.
    def _make(book_id: int = 1, fmt: str = "PDF", **overrides: Any) -> BookRecord:
        path = tmp_path / f"book{book_id}.{fmt.lower()}"
        path.write_text("fake", encoding="utf-8")
        book = BookRecord(
            book_id=book_id,
            title=f"Title {book_id}",
            authors=["Author"],
            tags=["tag"],
            comments=None,
            publisher=None,
            series=None,
            rating=None,
            languages=["en"],
            identifiers={},
            isbn=None,
            uuid=f"uuid-{book_id}",
            published_at=None,
            updated_at=None,
            path=f"Author/Title ({book_id})",
            files=[CalibreFile(format=fmt, file_path=path, file_name="Title", size_bytes=4)],
        )
        return dataclasses.replace(book, **overrides)

    return _make
