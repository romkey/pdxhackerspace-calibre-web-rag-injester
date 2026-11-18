from __future__ import annotations

import dataclasses
import hashlib
import os
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from calibre_web2rag.config import Settings
from calibre_web2rag.db import Database, migrate
from calibre_web2rag.embeddings import EmbedKind
from calibre_web2rag.models import BookRecord, CalibreFile

# Tables the tests empty before each database test.
_TABLES = (
    "staging_documents",
    "quarantine",
    "sync_runs",
    "llm_cache",
    "enrichment",
    "registry_manifestations",
    "registry_works",
    "registry_series",
    "registry_merges",
    "registry_review_queue",
    "entity_aliases",
    "catalog_items",
)


def _database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.fail(
            "Tests need PostgreSQL: run them with "
            "`docker compose -f docker-compose.test.yaml run --rm test`, or set DATABASE_URL"
        )
    return url


def _base_settings(tmp_path: Path) -> Settings:
    return Settings(
        calibre_metadata_db=str(tmp_path / "metadata.db"),
        calibre_library_root=str(tmp_path / "library"),
        calibre_web_base_url="https://books.example",
        calibre_download_url_template=None,
        database_url="postgresql://unused",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        embedding_provider="openai",
        embedding_model="dummy",
        hf_cache_dir=None,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=None,
        openai_timeout_seconds=30,
        chunk_size=200,
        chunk_overlap=20,
        batch_size=100,
        distance="cosine",
        vector_size=8,
        embedding_context_length=8192,
        enrichment_enabled=False,
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


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _database_url()
    migrate(url)
    return url


@pytest.fixture
def db(database_url: str) -> Iterator[Database]:
    """A connection to an emptied, migrated database."""
    with Database.connect(database_url) as database:
        database.conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        database.commit()
        yield database
        database.conn.rollback()


# ---------------------------------------------------------------------------
# A real Calibre metadata.db
# ---------------------------------------------------------------------------

_CALIBRE_SCHEMA = """
CREATE TABLE books (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL DEFAULT 'Unknown', sort TEXT,
    timestamp TIMESTAMP, pubdate TIMESTAMP, series_index REAL NOT NULL DEFAULT 1.0,
    author_sort TEXT, isbn TEXT DEFAULT '', lccn TEXT DEFAULT '', path TEXT NOT NULL DEFAULT '',
    flags INTEGER NOT NULL DEFAULT 1, uuid TEXT, has_cover BOOL DEFAULT 0,
    last_modified TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00+00:00'
);
CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, sort TEXT, link TEXT);
CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, sort TEXT);
CREATE TABLE books_publishers_link (id INTEGER PRIMARY KEY, book INTEGER, publisher INTEGER);
CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT NOT NULL, sort TEXT);
CREATE TABLE books_series_link (id INTEGER PRIMARY KEY, book INTEGER, series INTEGER);
CREATE TABLE ratings (id INTEGER PRIMARY KEY, rating INTEGER);
CREATE TABLE books_ratings_link (id INTEGER PRIMARY KEY, book INTEGER, rating INTEGER);
CREATE TABLE languages (id INTEGER PRIMARY KEY, lang_code TEXT NOT NULL);
CREATE TABLE books_languages_link (
    id INTEGER PRIMARY KEY, book INTEGER, lang_code INTEGER, item_order INTEGER DEFAULT 0
);
CREATE TABLE identifiers (id INTEGER PRIMARY KEY, book INTEGER, type TEXT, val TEXT);
CREATE TABLE data (
    id INTEGER PRIMARY KEY, book INTEGER, format TEXT, uncompressed_size INTEGER, name TEXT
);
CREATE TABLE custom_columns (
    id INTEGER PRIMARY KEY, label TEXT, name TEXT, datatype TEXT,
    mark_for_delete BOOL DEFAULT 0, editable BOOL DEFAULT 1, display TEXT DEFAULT '{}',
    is_multiple BOOL DEFAULT 0, normalized BOOL
);
"""


class CalibreLibrary:
    """Builds a metadata.db with Calibre's layout, and the files beside it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.library = root / "library"
        self.library.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "metadata.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_CALIBRE_SCHEMA)
        conn.commit()
        conn.close()

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def _named(self, table: str, name: str, column: str = "name") -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            # Test helper: table and column names are fixed by the callers.
            row = conn.execute(
                f"SELECT id FROM {table} WHERE {column} = ?",  # noqa: S608 - fixed names
                (name,),
            ).fetchone()
            if row:
                return int(row[0])
            cur = conn.execute(
                f"INSERT INTO {table} ({column}) VALUES (?)",  # noqa: S608 - fixed names
                (name,),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def add_book(
        self,
        book_id: int,
        title: str,
        *,
        authors: tuple[str, ...] = ("Ada Author",),
        tags: tuple[str, ...] = (),
        comments: str | None = None,
        isbn: str | None = None,
        series: str | None = None,
        series_index: float = 1.0,
        formats: dict[str, bytes] | None = None,
        last_modified: str = "2024-01-02 03:04:05+00:00",
        pubdate: str = "2019-06-01 00:00:00+00:00",
        has_cover: bool = False,
    ) -> Path:
        path = f"{authors[0] if authors else 'Unknown'}/{title} ({book_id})"
        self.execute(
            "INSERT INTO books (id, title, sort, timestamp, pubdate, series_index, path, uuid, "
            "has_cover, last_modified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                book_id,
                title,
                title,
                "2024-01-01 00:00:00+00:00",
                pubdate,
                series_index,
                path,
                f"uuid-{book_id}",
                int(has_cover),
                last_modified,
            ),
        )
        for author in authors:
            author_id = self._named("authors", author)
            self.execute(
                "INSERT INTO books_authors_link (book, author) VALUES (?, ?)", (book_id, author_id)
            )
        for tag in tags:
            self.execute(
                "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                (book_id, self._named("tags", tag)),
            )
        if comments is not None:
            self.execute("INSERT INTO comments (book, text) VALUES (?, ?)", (book_id, comments))
        if isbn:
            self.execute(
                "INSERT INTO identifiers (book, type, val) VALUES (?, 'isbn', ?)", (book_id, isbn)
            )
        if series:
            self.execute(
                "INSERT INTO books_series_link (book, series) VALUES (?, ?)",
                (book_id, self._named("series", series)),
            )
        self.execute(
            "INSERT INTO books_languages_link (book, lang_code, item_order) VALUES (?, ?, 0)",
            (book_id, self._named("languages", "eng", "lang_code")),
        )
        folder = self.library / path
        folder.mkdir(parents=True, exist_ok=True)
        for fmt, content in (formats or {}).items():
            name = f"{title} - {authors[0] if authors else 'Unknown'}"
            (folder / f"{name}.{fmt.lower()}").write_bytes(content)
            self.execute(
                "INSERT INTO data (book, format, uncompressed_size, name) VALUES (?, ?, ?, ?)",
                (book_id, fmt, len(content), name),
            )
        return folder

    def delete_book(self, book_id: int) -> None:
        self.execute("DELETE FROM books WHERE id = ?", (book_id,))

    def touch(self, book_id: int, last_modified: str, **columns: object) -> None:
        for column, value in columns.items():
            # Test helper: column names come from the test code.
            self.execute(
                f"UPDATE books SET {column} = ? WHERE id = ?",  # noqa: S608 - test names
                (value, book_id),
            )
        self.execute("UPDATE books SET last_modified = ? WHERE id = ?", (last_modified, book_id))


@pytest.fixture
def calibre(tmp_path: Path) -> CalibreLibrary:
    return CalibreLibrary(tmp_path)


class FakeEmbedder:
    """Deterministic 8-dimension vectors from a hash of the text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        self.calls.append(list(chunks))
        vectors = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.encode()).digest()
            vectors.append([b / 255 + 0.01 for b in digest[:8]])
        return vectors

    @property
    def texts(self) -> list[str]:
        return [t for call in self.calls for t in call]
