import uuid
from pathlib import Path

from calibre_web2rag.config import Settings
from calibre_web2rag.ingest import (
    _build_book_metadata_text,
    _build_library_summary_text,
    _generate_points,
)
from calibre_web2rag.models import BookRecord, CalibreFile, TextSection


class _Repo:
    def __init__(self, books: list[BookRecord]) -> None:
        self._books = books

    def fetch_books(self) -> list[BookRecord]:
        return self._books


class _Embedder:
    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[list[float]]:  # noqa: ARG002
        return [[0.1, 0.2, 0.3] for _ in chunks]


def _make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
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
        ollama_timeout_seconds=30,
        chunk_size=40,
        chunk_overlap=10,
        batch_size=100,
        distance="cosine",
        vector_size=3,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _collect_points(batches):
    return [point for batch in batches for point in batch]


def _by_type(points, chunk_type):
    return [p for p in points if p.payload["chunk_type"] == chunk_type]


def test_generate_points_includes_metadata(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_text("fake", encoding="utf-8")

    book = BookRecord(
        book_id=1,
        title="Title",
        authors=["Author"],
        tags=["tag"],
        comments=None,
        publisher="pub",
        series="series",
        rating=5,
        languages=["en"],
        identifiers={"asin": "123"},
        isbn="x",
        uuid="u",
        published_at="2020",
        updated_at="2021",
        path="Author/Title (1)",
        files=[CalibreFile(format="PDF", file_path=pdf, file_name="Title", size_bytes=100)],
    )

    monkeypatch.setattr(
        "calibre_web2rag.ingest._safe_extract_sections",
        lambda fmt, path: [TextSection(title=None, text="hello world " * 50)],
    )

    settings = _make_settings(tmp_path)

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    content = _by_type(points, "content")
    assert content
    uuid.UUID(str(content[0].id))
    payload = content[0].payload
    assert payload["book_id"] == 1
    assert payload["source_url"] == "https://books.example/download/1/pdf"
    assert payload["opds_url"] == "https://books.example/opds/book/1"
    assert payload["total_chunks"] == len(content)
    assert payload["chapter_title"] is None

    # Book metadata point
    meta = _by_type(points, "book_metadata")
    assert len(meta) == 1
    assert meta[0].payload["book_id"] == 1
    assert "Title" in meta[0].payload["chunk_text"]
    assert "Author" in meta[0].payload["chunk_text"]

    # Library summary
    summary = _by_type(points, "library_summary")
    assert len(summary) >= 1
    assert summary[0].payload["book_id"] is None
    assert "1 books" in summary[0].payload["chunk_text"]


def test_description_chunks_indexed(monkeypatch, tmp_path: Path) -> None:
    book = BookRecord(
        book_id=2,
        title="Described Book",
        authors=["Author"],
        tags=[],
        comments="<p>A fascinating book about <b>science</b>.</p>",
        publisher=None,
        series=None,
        rating=None,
        languages=["en"],
        identifiers={},
        isbn=None,
        uuid="u2",
        published_at=None,
        updated_at=None,
        path="Author/Described Book (2)",
        files=[],
    )

    settings = _make_settings(tmp_path, chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    desc = _by_type(points, "description")
    assert len(desc) == 1
    payload = desc[0].payload
    assert payload["file_format"] is None
    assert payload["source_url"] is None
    assert "science" in payload["chunk_text"]

    # Book metadata also present
    meta = _by_type(points, "book_metadata")
    assert len(meta) == 1


def test_chapter_title_from_sections(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_text("fake", encoding="utf-8")

    book = BookRecord(
        book_id=3,
        title="Chaptered",
        authors=["Author"],
        tags=[],
        comments=None,
        publisher=None,
        series=None,
        rating=None,
        languages=[],
        identifiers={},
        isbn=None,
        uuid="u3",
        published_at=None,
        updated_at=None,
        path="Author/Chaptered (3)",
        files=[CalibreFile(format="PDF", file_path=pdf, file_name="Chaptered", size_bytes=100)],
    )

    monkeypatch.setattr(
        "calibre_web2rag.ingest._safe_extract_sections",
        lambda fmt, path: [
            TextSection(title="Chapter 1", text="First chapter content here."),
            TextSection(title="Chapter 2", text="Second chapter content here."),
        ],
    )

    settings = _make_settings(tmp_path, chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    content = _by_type(points, "content")
    assert len(content) == 2
    assert content[0].payload["chapter_title"] == "Chapter 1"
    assert content[1].payload["chapter_title"] == "Chapter 2"


def test_per_book_error_isolation(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_text("fake", encoding="utf-8")

    good_book = BookRecord(
        book_id=10,
        title="Good",
        authors=[],
        tags=[],
        comments=None,
        publisher=None,
        series=None,
        rating=None,
        languages=[],
        identifiers={},
        isbn=None,
        uuid="u10",
        published_at=None,
        updated_at=None,
        path="Author/Good (10)",
        files=[CalibreFile(format="PDF", file_path=pdf, file_name="Good", size_bytes=100)],
    )
    bad_book = BookRecord(
        book_id=11,
        title="Bad",
        authors=[],
        tags=[],
        comments=None,
        publisher=None,
        series=None,
        rating=None,
        languages=[],
        identifiers={},
        isbn=None,
        uuid="u11",
        published_at=None,
        updated_at=None,
        path="Author/Bad (11)",
        files=[CalibreFile(format="PDF", file_path=pdf, file_name="Bad", size_bytes=100)],
    )

    call_count = {"n": 0}

    def _mock_sections(fmt, path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Simulated extraction failure")
        return [TextSection(title=None, text="good content here for testing")]

    monkeypatch.setattr(
        "calibre_web2rag.ingest._safe_extract_sections", _mock_sections
    )

    settings = _make_settings(tmp_path, chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(
            repo=_Repo([bad_book, good_book]), embedder=_Embedder(), settings=settings
        )
    )
    # Only the good book's per-book points should survive
    book_points = [p for p in points if p.payload.get("book_id") is not None]
    assert len(book_points) >= 1
    assert all(p.payload["book_id"] == 10 for p in book_points)

    # Library summary still generated for both books
    summary = _by_type(points, "library_summary")
    assert len(summary) >= 1
    assert "2 books" in summary[0].payload["chunk_text"]


def test_book_metadata_text_includes_all_fields() -> None:
    book = BookRecord(
        book_id=99,
        title="Deep Learning",
        authors=["Author A", "Author B"],
        tags=["AI", "Machine Learning"],
        comments="<p>A comprehensive <b>guide</b>.</p>",
        publisher="Tech Press",
        series="ML Series",
        rating=8,
        languages=["en"],
        identifiers={"isbn": "123"},
        isbn="978-0000000000",
        uuid="u99",
        published_at="2023-01-01",
        updated_at="2023-06-01",
        path="Authors/Deep Learning (99)",
        files=[
            CalibreFile(format="PDF", file_path=Path("/f.pdf"), file_name="f", size_bytes=1),
            CalibreFile(format="EPUB", file_path=Path("/f.epub"), file_name="f", size_bytes=1),
        ],
    )
    text = _build_book_metadata_text(book)
    assert "Deep Learning" in text
    assert "Author A" in text
    assert "Author B" in text
    assert "Tech Press" in text
    assert "ML Series" in text
    assert "AI" in text
    assert "Machine Learning" in text
    assert "978-0000000000" in text
    assert "8/10" in text
    assert "PDF" in text
    assert "EPUB" in text
    assert "comprehensive guide" in text


def test_library_summary_text_aggregates() -> None:
    books = [
        BookRecord(
            book_id=1,
            title="Book A",
            authors=["Author X", "Author Y"],
            tags=["Fiction", "Fantasy"],
            comments=None,
            publisher=None,
            series="Epic Series",
            rating=None,
            languages=["en"],
            identifiers={},
            isbn=None,
            uuid="u1",
            published_at=None,
            updated_at=None,
            path="p1",
            files=[
                CalibreFile(
                    format="PDF", file_path=Path("/a.pdf"), file_name="a", size_bytes=1
                ),
            ],
        ),
        BookRecord(
            book_id=2,
            title="Book B",
            authors=["Author X"],
            tags=["Non-Fiction"],
            comments=None,
            publisher=None,
            series=None,
            rating=None,
            languages=["en", "fr"],
            identifiers={},
            isbn=None,
            uuid="u2",
            published_at=None,
            updated_at=None,
            path="p2",
            files=[
                CalibreFile(
                    format="EPUB", file_path=Path("/b.epub"), file_name="b", size_bytes=1
                ),
            ],
        ),
    ]
    text = _build_library_summary_text(books)

    assert "2 books" in text
    assert "2 authors" in text
    assert "Author X (2 books)" in text
    assert "Author Y (1 book)" in text
    assert "Fiction" in text
    assert "Fantasy" in text
    assert "Non-Fiction" in text
    assert "Epic Series" in text
    assert "en" in text
    assert "fr" in text
    assert "PDF" in text
    assert "EPUB" in text
