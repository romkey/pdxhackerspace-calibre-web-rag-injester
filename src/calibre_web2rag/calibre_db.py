"""Reads books from Calibre's metadata.db.

Each run works from a read-only snapshot taken with SQLite's backup API, so
Calibre can keep writing while we read. Only metadata.db is opened: never
Calibre-web's own app.db (shelves, read status, downloads, users).

Schema reference: calibre's ``resources/metadata_sqlite.sql`` and
``src/calibre/library/custom_columns.py`` (https://github.com/kovidgoyal/calibre).
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from calibre_web2rag.models import BookRecord, CalibreFile

logger = logging.getLogger(__name__)

# Formats whose text can be extracted (table of contents, full text).
TEXT_FORMATS = frozenset({"PDF", "EPUB", "MOBI"})

# Tables and columns this module reads. verify_schema() checks them before
# any query runs, so an unexpected Calibre version fails with a clear message.
REQUIRED_SCHEMA: dict[str, frozenset[str]] = {
    "books": frozenset(
        {
            "id",
            "title",
            "sort",
            "timestamp",
            "pubdate",
            "series_index",
            "path",
            "uuid",
            "has_cover",
            "last_modified",
        }
    ),
    "authors": frozenset({"id", "name"}),
    "books_authors_link": frozenset({"id", "book", "author"}),
    "tags": frozenset({"id", "name"}),
    "books_tags_link": frozenset({"book", "tag"}),
    "comments": frozenset({"book", "text"}),
    "publishers": frozenset({"id", "name"}),
    "books_publishers_link": frozenset({"book", "publisher"}),
    "series": frozenset({"id", "name"}),
    "books_series_link": frozenset({"book", "series"}),
    "ratings": frozenset({"id", "rating"}),
    "books_ratings_link": frozenset({"book", "rating"}),
    "languages": frozenset({"id", "lang_code"}),
    "books_languages_link": frozenset({"book", "lang_code", "item_order"}),
    "identifiers": frozenset({"book", "type", "val"}),
    "data": frozenset({"book", "format", "uncompressed_size", "name"}),
    "custom_columns": frozenset(
        {"id", "label", "datatype", "mark_for_delete", "is_multiple", "normalized"}
    ),
}

# Custom column types stored in their own tables. "composite" columns are
# computed by Calibre from templates and have no stored values.
_CUSTOM_TYPES = frozenset(
    {"text", "comments", "series", "enumeration", "int", "float", "bool", "datetime", "rating"}
)


class CalibreSchemaError(RuntimeError):
    """metadata.db is missing a table or column this module needs."""


def verify_schema(conn: sqlite3.Connection) -> None:
    problems: list[str] = []
    for table, columns in REQUIRED_SCHEMA.items():
        found = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if not found:
            problems.append(f"table {table} is missing")
            continue
        missing = sorted(columns - found)
        if missing:
            problems.append(f"{table} lacks {', '.join(missing)}")
    if problems:
        raise CalibreSchemaError(
            "metadata.db does not have the layout this version expects: " + "; ".join(problems)
        )


@contextlib.contextmanager
def snapshot(metadata_db_path: str) -> Iterator[Path]:
    """Copy metadata.db to a temporary file and yield its path."""
    source_path = Path(metadata_db_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Calibre metadata.db not found: {source_path}")
    tempdir = Path(tempfile.mkdtemp(prefix="calibre-web2rag-"))
    copy_path = tempdir / "metadata.db"
    try:
        source = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            target = sqlite3.connect(copy_path)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        yield copy_path
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


class CalibreRepository:
    def __init__(self, metadata_db_path: str | Path, library_root: str) -> None:
        self._db_path = Path(metadata_db_path)
        self._library_root = Path(library_root)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> CalibreRepository:
        conn = sqlite3.connect(f"{self._db_path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            verify_schema(conn)
        except CalibreSchemaError:
            conn.close()
            raise
        self._conn = conn
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("CalibreRepository is used as a context manager")
        return self._conn

    def book_versions(self) -> dict[int, str | None]:
        """Every book ID with its last_modified timestamp."""
        rows = self.conn.execute("SELECT id, last_modified FROM books").fetchall()
        return {int(row["id"]): row["last_modified"] for row in rows}

    def fetch_books(self, book_ids: Iterable[int] | None = None) -> Iterator[BookRecord]:
        """Books in ID order: all of them, or only ``book_ids``."""
        wanted = None if book_ids is None else set(book_ids)
        custom = self._custom_columns()
        rows = self.conn.execute(
            """
            SELECT b.id, b.title, b.sort, b.timestamp, b.pubdate, b.series_index,
                   b.path, b.uuid, b.has_cover, b.last_modified, c.text AS comments
            FROM books b
            LEFT JOIN comments c ON c.book = b.id
            ORDER BY b.id
            """
        )
        for row in rows:
            book_id = int(row["id"])
            if wanted is not None and book_id not in wanted:
                continue
            yield self._book(row, custom)

    def _book(self, row: sqlite3.Row, custom: list[sqlite3.Row]) -> BookRecord:
        book_id = int(row["id"])
        # Calibre keeps ISBNs in the identifiers table; current schemas have
        # no books.isbn column.
        identifiers = self._identifiers(book_id)
        custom_values = self._custom_values(book_id, custom)
        subtitle = custom_values.get("subtitle")
        series = self._scalar(
            "SELECT s.name FROM series s JOIN books_series_link l ON l.series = s.id "
            "WHERE l.book = ? LIMIT 1",
            book_id,
        )
        return BookRecord(
            book_id=book_id,
            title=row["title"],
            sort_title=row["sort"],
            subtitle=subtitle if isinstance(subtitle, str) else None,
            authors=self._names(
                "SELECT a.name FROM authors a JOIN books_authors_link l ON l.author = a.id "
                "WHERE l.book = ? ORDER BY l.id",
                book_id,
            ),
            tags=self._names(
                "SELECT t.name FROM tags t JOIN books_tags_link l ON l.tag = t.id "
                "WHERE l.book = ? ORDER BY t.name",
                book_id,
            ),
            comments=row["comments"],
            publisher=self._scalar(
                "SELECT p.name FROM publishers p JOIN books_publishers_link l "
                "ON l.publisher = p.id WHERE l.book = ? LIMIT 1",
                book_id,
            ),
            series=series,
            series_index=float(row["series_index"]) if series else None,
            rating=self._rating(book_id),
            languages=self._names(
                "SELECT g.lang_code FROM languages g JOIN books_languages_link l "
                "ON l.lang_code = g.id WHERE l.book = ? ORDER BY l.item_order",
                book_id,
            ),
            identifiers=identifiers,
            isbn=identifiers.get("isbn"),
            uuid=row["uuid"],
            published_at=row["pubdate"],
            added_at=row["timestamp"],
            updated_at=row["last_modified"],
            has_cover=bool(row["has_cover"]),
            path=row["path"],
            files=self._files(book_id, row["path"]),
            custom_columns=custom_values,
        )

    def _files(self, book_id: int, rel_path: str) -> list[CalibreFile]:
        files: list[CalibreFile] = []
        rows = self.conn.execute(
            "SELECT format, uncompressed_size, name FROM data WHERE book = ? ORDER BY format",
            (book_id,),
        ).fetchall()
        for row in rows:
            fmt = str(row["format"]).upper()
            file_name = str(row["name"])
            full_path = self._library_root / rel_path / f"{file_name}.{fmt.lower()}"
            files.append(
                CalibreFile(
                    format=fmt,
                    file_path=full_path,
                    file_name=file_name,
                    size_bytes=row["uncompressed_size"],
                    exists=full_path.is_file(),
                )
            )
        return files

    def _custom_columns(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, label, datatype, is_multiple, normalized FROM custom_columns "
            "WHERE mark_for_delete = 0 ORDER BY id"
        ).fetchall()

    def _custom_values(self, book_id: int, columns: list[sqlite3.Row]) -> dict[str, object]:
        values: dict[str, object] = {}
        for column in columns:
            if column["datatype"] not in _CUSTOM_TYPES:
                continue
            # The table names come from Calibre's integer column IDs, never
            # from text, so building them into the SQL is safe.
            number = int(column["id"])
            try:
                if column["normalized"]:
                    rows = self.conn.execute(
                        f"SELECT v.value FROM custom_column_{number} v "  # noqa: S608 - integer id
                        f"JOIN books_custom_column_{number}_link l ON l.value = v.id "
                        "WHERE l.book = ? ORDER BY v.value",
                        (book_id,),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        f"SELECT value FROM custom_column_{number} WHERE book = ?",  # noqa: S608 - integer id
                        (book_id,),
                    ).fetchall()
            except sqlite3.DatabaseError as e:
                logger.warning("Cannot read custom column %r: %s", column["label"], e)
                continue
            found = [row[0] for row in rows if row[0] is not None]
            if not found:
                continue
            if column["is_multiple"]:
                values[str(column["label"])] = [str(v) for v in found]
            elif column["datatype"] == "bool":
                values[str(column["label"])] = bool(found[0])
            else:
                values[str(column["label"])] = found[0]
        return values

    def _names(self, query: str, book_id: int) -> list[str]:
        rows = self.conn.execute(query, (book_id,)).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def _scalar(self, query: str, book_id: int) -> str | None:
        row = self.conn.execute(query, (book_id,)).fetchone()
        return str(row[0]) if row and row[0] else None

    def _identifiers(self, book_id: int) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT type, val FROM identifiers "
            "WHERE book = ? AND type IS NOT NULL AND val IS NOT NULL ORDER BY type",
            (book_id,),
        ).fetchall()
        return {str(row["type"]): str(row["val"]) for row in rows}

    def _rating(self, book_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT r.rating FROM ratings r JOIN books_ratings_link l ON l.rating = r.id "
            "WHERE l.book = ? LIMIT 1",
            (book_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
