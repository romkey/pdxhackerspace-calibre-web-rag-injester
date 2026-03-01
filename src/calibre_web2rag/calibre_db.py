from __future__ import annotations

import sqlite3
from pathlib import Path

from calibre_web2rag.models import BookRecord, CalibreFile

SUPPORTED_FORMATS = {"PDF", "EPUB", "MOBI"}


class CalibreRepository:
    def __init__(self, metadata_db_path: str, library_root: str) -> None:
        self._db_path = metadata_db_path
        self._library_root = Path(library_root)

    def fetch_books(self) -> list[BookRecord]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            books: list[BookRecord] = []
            rows = conn.execute(
                """
                SELECT
                    b.id,
                    b.title,
                    b.path,
                    b.isbn,
                    b.uuid,
                    b.pubdate,
                    b.last_modified,
                    c.text AS comments
                FROM books b
                LEFT JOIN comments c ON c.book = b.id
                ORDER BY b.id
                """
            ).fetchall()
            for row in rows:
                book_id = int(row["id"])
                files = self._get_files(conn, book_id, row["path"])
                if not files:
                    continue
                books.append(
                    BookRecord(
                        book_id=book_id,
                        title=row["title"],
                        authors=self._get_name_list(
                            conn,
                            """
                            SELECT a.name
                            FROM authors a
                            JOIN books_authors_link bal ON bal.author = a.id
                            WHERE bal.book = ?
                            ORDER BY bal.id
                            """,
                            book_id,
                        ),
                        tags=self._get_name_list(
                            conn,
                            """
                            SELECT t.name
                            FROM tags t
                            JOIN books_tags_link btl ON btl.tag = t.id
                            WHERE btl.book = ?
                            ORDER BY t.name
                            """,
                            book_id,
                        ),
                        comments=row["comments"],
                        publisher=self._get_scalar(
                            conn,
                            """
                            SELECT p.name
                            FROM publishers p
                            JOIN books_publishers_link bpl ON bpl.publisher = p.id
                            WHERE bpl.book = ?
                            LIMIT 1
                            """,
                            book_id,
                        ),
                        series=self._get_scalar(
                            conn,
                            """
                            SELECT s.name
                            FROM series s
                            JOIN books_series_link bsl ON bsl.series = s.id
                            WHERE bsl.book = ?
                            LIMIT 1
                            """,
                            book_id,
                        ),
                        rating=self._get_rating(conn, book_id),
                        languages=self._get_name_list(
                            conn,
                            "SELECT lang_code FROM books_languages_link WHERE book = ?",
                            book_id,
                        ),
                        identifiers=self._get_identifiers(conn, book_id),
                        isbn=row["isbn"],
                        uuid=row["uuid"],
                        published_at=row["pubdate"],
                        updated_at=row["last_modified"],
                        path=row["path"],
                        files=files,
                    )
                )
            return books
        finally:
            conn.close()

    def _get_files(self, conn: sqlite3.Connection, book_id: int, rel_path: str) -> list[CalibreFile]:
        files: list[CalibreFile] = []
        rows = conn.execute(
            "SELECT format, uncompressed_size, name FROM data WHERE book = ?", (book_id,)
        ).fetchall()
        for row in rows:
            fmt = str(row["format"]).upper()
            if fmt not in SUPPORTED_FORMATS:
                continue
            file_name = row["name"]
            full_path = self._library_root / rel_path / f"{file_name}.{fmt.lower()}"
            if not full_path.exists():
                continue
            files.append(
                CalibreFile(
                    format=fmt,
                    file_path=full_path,
                    file_name=file_name,
                    size_bytes=row["uncompressed_size"],
                )
            )
        return files

    @staticmethod
    def _get_name_list(conn: sqlite3.Connection, query: str, book_id: int) -> list[str]:
        rows = conn.execute(query, (book_id,)).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    @staticmethod
    def _get_scalar(conn: sqlite3.Connection, query: str, book_id: int) -> str | None:
        row = conn.execute(query, (book_id,)).fetchone()
        return str(row[0]) if row and row[0] else None

    @staticmethod
    def _get_identifiers(conn: sqlite3.Connection, book_id: int) -> dict[str, str]:
        rows = conn.execute(
            "SELECT type, val FROM identifiers WHERE book = ? AND type IS NOT NULL AND val IS NOT NULL",
            (book_id,),
        ).fetchall()
        return {str(row["type"]): str(row["val"]) for row in rows}

    @staticmethod
    def _get_rating(conn: sqlite3.Connection, book_id: int) -> int | None:
        row = conn.execute(
            """
            SELECT r.rating
            FROM ratings r
            JOIN books_ratings_link brl ON brl.rating = r.id
            WHERE brl.book = ?
            LIMIT 1
            """,
            (book_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
