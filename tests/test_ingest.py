from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client.models import PointStruct

from calibre_web2rag import ingest as ingest_mod
from calibre_web2rag.config import Settings
from calibre_web2rag.ingest import (
    _build_book_metadata_text,
    _build_library_summary_text,
    _generate_points,
)
from calibre_web2rag.links import build_ebook_url, build_opds_url
from calibre_web2rag.models import BookRecord, CalibreFile, TextSection

# The make_settings and make_book fixtures (conftest.py).
SettingsFactory = Callable[..., Settings]
BookFactory = Callable[..., BookRecord]


class _Embedder:
    def encode(self, chunks: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in chunks]


class _Repo:
    def __init__(self, books: list[BookRecord]) -> None:
        self._books = books

    def fetch_books(self) -> list[BookRecord]:
        return self._books


# Any: Qdrant payloads are arbitrary JSON.
def _payloads(batches: Iterable[list[PointStruct]]) -> list[dict[str, Any]]:
    payloads = []
    for batch in batches:
        for point in batch:
            assert point.payload is not None
            payloads.append(point.payload)
    return payloads


def _sections(text: str) -> Callable[[str, str], list[TextSection]]:
    """A stand-in for _safe_extract_sections that returns one section."""

    def extract(fmt: str, path: str) -> list[TextSection]:
        return [TextSection(title=None, text=text)]

    return extract


def _collect_points(batches: Iterable[list[PointStruct]]) -> list[PointStruct]:
    return [point for batch in batches for point in batch]


# Any: Qdrant payloads are arbitrary JSON.
def _payload(point: PointStruct) -> dict[str, Any]:
    assert point.payload is not None
    return point.payload


def _by_type(points: list[PointStruct], chunk_type: str) -> list[dict[str, Any]]:
    """Payloads of the points of one chunk_type."""
    return [_payload(p) for p in points if _payload(p)["chunk_type"] == chunk_type]


@pytest.mark.parametrize(
    ("fmt", "target"),
    [
        ("PDF", "try_extract_pdf_sections"),
        ("EPUB", "try_extract_epub_sections"),
        ("MOBI", "try_extract_mobi_sections"),
    ],
)
def test_safe_extract_dispatches_by_format(
    monkeypatch: pytest.MonkeyPatch, fmt: str, target: str
) -> None:
    seen: list[Path] = []

    def _fake(path: Path) -> list[TextSection]:
        seen.append(path)
        return [TextSection(title=None, text=fmt)]

    monkeypatch.setattr(ingest_mod, target, _fake)
    assert ingest_mod._safe_extract_sections(fmt, "/x/book") == [TextSection(title=None, text=fmt)]
    assert seen == [Path("/x/book")]


def test_safe_extract_unknown_format() -> None:
    assert ingest_mod._safe_extract_sections("AZW3", "/x/book") == []


def test_generate_points_batches(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", _sections("Sentence one. " * 30))
    batches = list(
        ingest_mod._generate_points(
            repo=_Repo([make_book(1), make_book(2)]),
            embedder=_Embedder(),
            settings=make_settings(batch_size=3),
        )
    )
    assert len(batches) > 1
    assert all(len(b) == 3 for b in batches[:-1])
    assert 1 <= len(batches[-1]) <= 3
    types = {p["chunk_type"] for p in _payloads(batches)}
    assert {"content", "book_metadata", "library_summary"} <= types


def test_generate_points_skips_failing_book(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    def _extract(fmt: str, path: str) -> list[TextSection]:
        if "book1" in path:
            raise RuntimeError("corrupt")
        return [TextSection(title=None, text="Fine text here. " * 10)]

    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", _extract)
    payloads = _payloads(
        ingest_mod._generate_points(
            repo=_Repo([make_book(1), make_book(2)]),
            embedder=_Embedder(),
            settings=make_settings(),
        )
    )
    content_ids = {p["book_id"] for p in payloads if p["chunk_type"] == "content"}
    assert content_ids == {2}


def test_generate_points_skips_empty_files(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    sections = {
        "PDF": [],
        "EPUB": [TextSection(title=None, text="   ")],
    }
    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", lambda fmt, path: sections[fmt])
    payloads = _payloads(
        ingest_mod._generate_points(
            repo=_Repo([make_book(1, "PDF"), make_book(2, "EPUB")]),
            embedder=_Embedder(),
            settings=make_settings(),
        )
    )
    assert not [p for p in payloads if p["chunk_type"] == "content"]
    assert [p for p in payloads if p["chunk_type"] == "book_metadata"]


def test_generate_points_no_books(make_settings: SettingsFactory) -> None:
    batches = list(
        ingest_mod._generate_points(repo=_Repo([]), embedder=_Embedder(), settings=make_settings())
    )
    assert batches == []


def test_library_summary_empty_text(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    monkeypatch.setattr(ingest_mod, "_build_library_summary_text", lambda books: "")
    points = ingest_mod._build_library_summary_points(
        books=[make_book()], embedder=_Embedder(), settings=make_settings()
    )
    assert points == []


class _Store:
    """Records how ingest() builds and uses the Qdrant store."""

    # Any: the keyword arguments ingest() passes to QdrantStore.
    init: dict[str, Any]
    upserts: list[int]

    def __init__(self, **kwargs: Any) -> None:
        type(self).init = kwargs
        type(self).upserts = []

    def upsert(self, batch: list[PointStruct]) -> None:
        type(self).upserts.append(len(batch))


def _wire_fakes(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, books: list[BookRecord]
) -> None:
    class _RepoCls(_Repo):
        def __init__(self, db: str, root: str) -> None:
            assert db == settings.calibre_metadata_db
            assert root == settings.calibre_library_root
            super().__init__(books)

    monkeypatch.setattr(ingest_mod, "CalibreRepository", _RepoCls)
    monkeypatch.setattr(ingest_mod, "QdrantStore", _Store)
    monkeypatch.setattr(ingest_mod, "build_embedder", lambda s: _Embedder())
    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", _sections("Some content. " * 20))


def test_ingest_wires_components(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    settings = make_settings(batch_size=2)
    _wire_fakes(monkeypatch, settings, [make_book(1)])

    total = ingest_mod.ingest(settings)
    assert total == sum(_Store.upserts)
    assert len(_Store.upserts) > 1
    assert _Store.init == {
        "url": settings.qdrant_url,
        "api_key": settings.qdrant_api_key,
        "collection": settings.qdrant_collection,
        "vector_size": settings.vector_size,
        "distance": settings.distance,
    }


def test_ingest_stopped_before_start_upserts_nothing(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, make_book: BookFactory
) -> None:
    settings = make_settings(batch_size=2)
    _wire_fakes(monkeypatch, settings, [make_book(1), make_book(2)])
    stop = threading.Event()
    stop.set()

    assert ingest_mod.ingest(settings, stop=stop) == 0
    assert _Store.upserts == []


def test_ingest_stop_mid_book_upserts_what_it_has(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    make_book: BookFactory,
    tmp_path: Path,
) -> None:
    settings = make_settings(batch_size=1000)
    first = make_book(1)
    second_file = CalibreFile(
        format="EPUB", file_path=tmp_path / "book1.epub", file_name="Title", size_bytes=4
    )
    first = dataclasses.replace(first, files=[*first.files, second_file])
    _wire_fakes(monkeypatch, settings, [first, make_book(2)])
    stop = threading.Event()
    read: list[str] = []

    def _extract(fmt: str, path: str) -> list[TextSection]:
        # SIGTERM arrives while the first file is being read.
        read.append(fmt)
        stop.set()
        return [TextSection(title=None, text="Some content. " * 20)]

    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", _extract)

    total = ingest_mod.ingest(settings, stop=stop)
    # Only the first file of the first book: no second file, book or summary.
    assert read == ["PDF"]
    assert total > 1
    assert _Store.upserts == [total]


def test_links_return_none_without_config() -> None:
    assert build_ebook_url(book_id=1, fmt="PDF", base_url=None, template=None) is None
    assert build_opds_url(book_id=1, base_url=None) is None
    assert build_opds_url(book_id=1, base_url="") is None


def test_generate_points_includes_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_settings: SettingsFactory
) -> None:
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

    settings = make_settings()

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    content = _by_type(points, "content")
    assert content
    uuid.UUID(str(points[0].id))
    payload = content[0]
    assert payload["book_id"] == 1
    assert payload["source_url"] == "https://books.example/download/1/pdf"
    assert payload["opds_url"] == "https://books.example/opds/book/1"
    assert payload["total_chunks"] == len(content)
    assert payload["chapter_title"] is None

    # Book metadata point
    meta = _by_type(points, "book_metadata")
    assert len(meta) == 1
    assert meta[0]["book_id"] == 1
    assert "Title" in meta[0]["chunk_text"]
    assert "Author" in meta[0]["chunk_text"]

    # Library summary
    summary = _by_type(points, "library_summary")
    assert len(summary) >= 1
    assert summary[0]["book_id"] is None
    assert "1 books" in summary[0]["chunk_text"]


def test_description_chunks_indexed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_settings: SettingsFactory
) -> None:
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

    settings = make_settings(chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    desc = _by_type(points, "description")
    assert len(desc) == 1
    payload = desc[0]
    assert payload["file_format"] is None
    assert payload["source_url"] is None
    assert "science" in payload["chunk_text"]

    # Book metadata also present
    meta = _by_type(points, "book_metadata")
    assert len(meta) == 1


def test_chapter_title_from_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_settings: SettingsFactory
) -> None:
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

    settings = make_settings(chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings)
    )
    content = _by_type(points, "content")
    assert len(content) == 2
    assert content[0]["chapter_title"] == "Chapter 1"
    assert content[1]["chapter_title"] == "Chapter 2"


def test_per_book_error_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_settings: SettingsFactory
) -> None:
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

    def _mock_sections(fmt: str, path: str) -> list[TextSection]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Simulated extraction failure")
        return [TextSection(title=None, text="good content here for testing")]

    monkeypatch.setattr("calibre_web2rag.ingest._safe_extract_sections", _mock_sections)

    settings = make_settings(chunk_size=200, chunk_overlap=20)

    points = _collect_points(
        _generate_points(repo=_Repo([bad_book, good_book]), embedder=_Embedder(), settings=settings)
    )
    # Only the good book's per-book points should survive
    book_points = [_payload(p) for p in points if _payload(p)["book_id"] is not None]
    assert len(book_points) >= 1
    assert all(p["book_id"] == 10 for p in book_points)

    # Library summary still generated for both books
    summary = _by_type(points, "library_summary")
    assert len(summary) >= 1
    assert "2 books" in summary[0]["chunk_text"]


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
                CalibreFile(format="PDF", file_path=Path("/a.pdf"), file_name="a", size_bytes=1),
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
                CalibreFile(format="EPUB", file_path=Path("/b.epub"), file_name="b", size_bytes=1),
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
