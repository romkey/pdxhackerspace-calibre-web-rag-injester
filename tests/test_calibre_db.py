import sqlite3
from pathlib import Path

import pytest

from calibre_web2rag.calibre_db import CalibreRepository, CalibreSchemaError, snapshot
from tests.conftest import CalibreLibrary


def _add_custom_columns(calibre: CalibreLibrary) -> None:
    conn = sqlite3.connect(calibre.db_path)
    conn.executescript(
        """
        INSERT INTO custom_columns (id, label, name, datatype, is_multiple, normalized)
        VALUES (1, 'subtitle', 'Subtitle', 'text', 0, 1),
               (2, 'genres', 'Genres', 'text', 1, 1),
               (3, 'pages', 'Pages', 'int', 0, 0),
               (4, 'signed', 'Signed', 'bool', 0, 0),
               (5, 'computed', 'Computed', 'composite', 0, 0),
               (6, 'broken', 'Broken', 'int', 0, 0);
        INSERT INTO custom_columns (id, label, name, datatype, mark_for_delete, normalized)
        VALUES (7, 'gone', 'Gone', 'text', 1, 1);
        CREATE TABLE custom_column_1 (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE books_custom_column_1_link (id INTEGER PRIMARY KEY, book INTEGER,
            value INTEGER);
        CREATE TABLE custom_column_2 (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE books_custom_column_2_link (id INTEGER PRIMARY KEY, book INTEGER,
            value INTEGER);
        CREATE TABLE custom_column_3 (id INTEGER PRIMARY KEY, book INTEGER, value INTEGER);
        CREATE TABLE custom_column_4 (id INTEGER PRIMARY KEY, book INTEGER, value BOOL);
        INSERT INTO custom_column_1 VALUES (1, 'A Subtitle');
        INSERT INTO books_custom_column_1_link (book, value) VALUES (1, 1);
        INSERT INTO custom_column_2 VALUES (1, 'Hard SF'), (2, 'Space Opera');
        INSERT INTO books_custom_column_2_link (book, value) VALUES (1, 2), (1, 1);
        INSERT INTO custom_column_3 (book, value) VALUES (1, 412);
        INSERT INTO custom_column_4 (book, value) VALUES (1, 1);
        """
    )
    conn.commit()
    conn.close()


def test_reads_every_field(calibre: CalibreLibrary) -> None:
    calibre.add_book(
        1,
        "Leviathan Wakes",
        authors=("James S. A. Corey", "Second Author"),
        tags=("SF", "Adventure"),
        comments="<p>Space.</p>",
        isbn="9780316129084",
        series="The Expanse",
        series_index=1.0,
        formats={"EPUB": b"epub", "AZW3": b"azw"},
        has_cover=True,
    )
    calibre.execute("INSERT INTO identifiers (book, type, val) VALUES (1, 'goodreads', '8855321')")
    calibre.execute("INSERT INTO publishers (id, name) VALUES (1, 'Orbit')")
    calibre.execute("INSERT INTO books_publishers_link (book, publisher) VALUES (1, 1)")
    calibre.execute("INSERT INTO ratings (id, rating) VALUES (1, 8)")
    calibre.execute("INSERT INTO books_ratings_link (book, rating) VALUES (1, 1)")
    (
        calibre.library
        / "James S. A. Corey/Leviathan Wakes (1)/Leviathan Wakes - James S. A. Corey.azw3"
    ).unlink()
    _add_custom_columns(calibre)

    with CalibreRepository(calibre.db_path, str(calibre.library)) as repo:
        (book,) = list(repo.fetch_books())

    assert book.title == "Leviathan Wakes"
    assert book.sort_title == "Leviathan Wakes"
    assert book.subtitle == "A Subtitle"
    assert book.authors == ["James S. A. Corey", "Second Author"]
    assert book.tags == ["Adventure", "SF"]
    assert book.comments == "<p>Space.</p>"
    assert book.publisher == "Orbit"
    assert book.series == "The Expanse"
    assert book.series_index == 1.0
    assert book.rating == 8
    assert book.languages == ["eng"]
    assert book.identifiers == {"goodreads": "8855321", "isbn": "9780316129084"}
    assert book.isbn == "9780316129084"
    assert book.added_at == "2024-01-01 00:00:00+00:00"
    assert book.cover_path == "James S. A. Corey/Leviathan Wakes (1)/cover.jpg"
    assert [(f.format, f.size_bytes, f.exists) for f in book.files] == [
        ("AZW3", 3, False),
        ("EPUB", 4, True),
    ]
    assert book.custom_columns == {
        "subtitle": "A Subtitle",
        "genres": ["Hard SF", "Space Opera"],
        "pages": 412,
        "signed": True,
    }


def test_books_without_files_are_included(calibre: CalibreLibrary) -> None:
    calibre.add_book(1, "Paper Only")
    calibre.add_book(2, "Other", authors=())
    with CalibreRepository(calibre.db_path, str(calibre.library)) as repo:
        books = {b.book_id: b for b in repo.fetch_books()}
        assert repo.book_versions() == {
            1: "2024-01-02 03:04:05+00:00",
            2: "2024-01-02 03:04:05+00:00",
        }
        assert [b.book_id for b in repo.fetch_books([2])] == [2]
    assert books[1].files == []
    assert books[1].series_index is None
    assert books[1].cover_path is None
    assert books[2].authors == []


def test_schema_is_verified(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(CalibreSchemaError) as info:
        CalibreRepository(path, str(tmp_path)).__enter__()
    message = str(info.value)
    assert "books lacks has_cover" in message
    assert "table authors is missing" in message


def test_repository_needs_a_context(calibre: CalibreLibrary) -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        CalibreRepository(calibre.db_path, "").book_versions()


def test_snapshot_copies_and_cleans_up(calibre: CalibreLibrary, tmp_path: Path) -> None:
    calibre.add_book(1, "Snap")
    with snapshot(str(calibre.db_path)) as copy:
        assert copy != calibre.db_path
        with CalibreRepository(copy, str(calibre.library)) as repo:
            assert list(repo.book_versions()) == [1]
    assert not copy.exists()
    with pytest.raises(FileNotFoundError), snapshot(str(tmp_path / "nope.db")):
        pass
