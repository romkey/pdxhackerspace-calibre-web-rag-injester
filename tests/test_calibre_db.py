import sqlite3
from pathlib import Path

from calibre_web2rag.calibre_db import CalibreRepository


def test_fetch_books_from_calibre_db(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    lib_path = tmp_path / "lib"
    book_dir = lib_path / "Author Name" / "Book Title (1)"
    book_dir.mkdir(parents=True)
    (book_dir / "Book Title - Author Name.pdf").write_text("dummy", encoding="utf-8")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT, isbn TEXT, uuid TEXT, pubdate TEXT, last_modified TEXT);
            CREATE TABLE comments (book INTEGER, text TEXT);
            CREATE TABLE data (book INTEGER, format TEXT, uncompressed_size INTEGER, name TEXT);
            CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_tags_link (book INTEGER, tag INTEGER);
            CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_publishers_link (book INTEGER, publisher INTEGER);
            CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_series_link (book INTEGER, series INTEGER);
            CREATE TABLE ratings (id INTEGER PRIMARY KEY, rating INTEGER);
            CREATE TABLE books_ratings_link (book INTEGER, rating INTEGER);
            CREATE TABLE books_languages_link (book INTEGER, lang_code TEXT);
            CREATE TABLE identifiers (book INTEGER, type TEXT, val TEXT);
            """
        )
        conn.execute(
            "INSERT INTO books(id, title, path, isbn, uuid, pubdate, last_modified) VALUES (1, 'Book Title', 'Author Name/Book Title (1)', 'isbn', 'uuid', '2020', '2021')"
        )
        conn.execute("INSERT INTO comments(book, text) VALUES (1, 'comment')")
        conn.execute(
            "INSERT INTO data(book, format, uncompressed_size, name) VALUES (1, 'PDF', 100, 'Book Title - Author Name')"
        )
        conn.execute("INSERT INTO authors(id, name) VALUES (2, 'Author Name')")
        conn.execute("INSERT INTO books_authors_link(id, book, author) VALUES (1, 1, 2)")
        conn.execute("INSERT INTO tags(id, name) VALUES (3, 'Tag A')")
        conn.execute("INSERT INTO books_tags_link(book, tag) VALUES (1, 3)")
        conn.execute("INSERT INTO publishers(id, name) VALUES (4, 'Publisher')")
        conn.execute("INSERT INTO books_publishers_link(book, publisher) VALUES (1, 4)")
        conn.execute("INSERT INTO series(id, name) VALUES (5, 'Series')")
        conn.execute("INSERT INTO books_series_link(book, series) VALUES (1, 5)")
        conn.execute("INSERT INTO ratings(id, rating) VALUES (6, 8)")
        conn.execute("INSERT INTO books_ratings_link(book, rating) VALUES (1, 6)")
        conn.execute("INSERT INTO books_languages_link(book, lang_code) VALUES (1, 'en')")
        conn.execute("INSERT INTO identifiers(book, type, val) VALUES (1, 'asin', 'B123')")
        conn.commit()
    finally:
        conn.close()

    repo = CalibreRepository(str(db_path), str(lib_path))
    books = repo.fetch_books()
    assert len(books) == 1
    assert books[0].title == "Book Title"
    assert books[0].authors == ["Author Name"]
    assert books[0].tags == ["Tag A"]
    assert books[0].publisher == "Publisher"
    assert books[0].series == "Series"
    assert books[0].languages == ["en"]
    assert books[0].identifiers["asin"] == "B123"
    assert books[0].files[0].format == "PDF"
