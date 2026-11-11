from __future__ import annotations

from pathlib import Path

import pytest

from calibre_web2rag import ingest as ingest_mod
from calibre_web2rag.links import build_ebook_url, build_opds_url
from calibre_web2rag.models import TextSection


class _Embedder:
    def encode(self, chunks: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in chunks]


class _Repo:
    def __init__(self, books) -> None:
        self._books = books

    def fetch_books(self):
        return self._books


def _points(batches):
    return [p for batch in batches for p in batch]


@pytest.mark.parametrize(
    ("fmt", "target"),
    [
        ("PDF", "try_extract_pdf_sections"),
        ("EPUB", "try_extract_epub_sections"),
        ("MOBI", "try_extract_mobi_sections"),
    ],
)
def test_safe_extract_dispatches_by_format(monkeypatch, fmt: str, target: str) -> None:
    seen: list[Path] = []

    def _fake(path: Path):
        seen.append(path)
        return [TextSection(title=None, text=fmt)]

    monkeypatch.setattr(ingest_mod, target, _fake)
    assert ingest_mod._safe_extract_sections(fmt, "/x/book") == [
        TextSection(title=None, text=fmt)
    ]
    assert seen == [Path("/x/book")]


def test_safe_extract_unknown_format() -> None:
    assert ingest_mod._safe_extract_sections("AZW3", "/x/book") == []


def test_generate_points_batches(monkeypatch, make_settings, make_book) -> None:
    monkeypatch.setattr(
        ingest_mod,
        "_safe_extract_sections",
        lambda fmt, path: [TextSection(title="Ch", text="Sentence number one. " * 30)],
    )
    books = [make_book(1), make_book(2)]
    batches = list(
        ingest_mod._generate_points(
            repo=_Repo(books), embedder=_Embedder(), settings=make_settings(batch_size=3)
        )
    )
    assert len(batches) > 1
    assert all(len(b) == 3 for b in batches[:-1])
    assert 1 <= len(batches[-1]) <= 3
    types = {p.payload["chunk_type"] for p in _points(batches)}
    assert {"content", "book_metadata", "library_summary"} <= types


def test_generate_points_skips_failing_book(monkeypatch, make_settings, make_book) -> None:
    def _extract(fmt, path):
        if "book1" in path:
            raise RuntimeError("corrupt")
        return [TextSection(title=None, text="Fine text here. " * 10)]

    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", _extract)
    points = _points(
        ingest_mod._generate_points(
            repo=_Repo([make_book(1), make_book(2)]),
            embedder=_Embedder(),
            settings=make_settings(),
        )
    )
    content_ids = {p.payload["book_id"] for p in points if p.payload["chunk_type"] == "content"}
    assert content_ids == {2}


def test_generate_points_skips_empty_files(monkeypatch, make_settings, make_book) -> None:
    sections = {
        "PDF": [],
        "EPUB": [TextSection(title=None, text="   ")],
    }
    monkeypatch.setattr(ingest_mod, "_safe_extract_sections", lambda fmt, path: sections[fmt])
    points = _points(
        ingest_mod._generate_points(
            repo=_Repo([make_book(1, "PDF"), make_book(2, "EPUB")]),
            embedder=_Embedder(),
            settings=make_settings(),
        )
    )
    assert not [p for p in points if p.payload["chunk_type"] == "content"]
    assert [p for p in points if p.payload["chunk_type"] == "book_metadata"]


def test_generate_points_no_books(make_settings) -> None:
    batches = list(
        ingest_mod._generate_points(
            repo=_Repo([]), embedder=_Embedder(), settings=make_settings()
        )
    )
    assert batches == []


def test_library_summary_empty_text(monkeypatch, make_settings, make_book) -> None:
    monkeypatch.setattr(ingest_mod, "_build_library_summary_text", lambda books: "")
    assert (
        ingest_mod._build_library_summary_points(
            books=[make_book()], embedder=_Embedder(), settings=make_settings()
        )
        == []
    )


def test_ingest_wires_components(monkeypatch, make_settings, make_book) -> None:
    settings = make_settings(batch_size=2)
    store_calls: dict = {}

    class _Store:
        def __init__(self, **kwargs) -> None:
            store_calls["init"] = kwargs
            store_calls["upserts"] = []

        def upsert(self, batch) -> None:
            store_calls["upserts"].append(len(batch))

    class _RepoCls:
        def __init__(self, db: str, root: str) -> None:
            assert db == settings.calibre_metadata_db
            assert root == settings.calibre_library_root

        def fetch_books(self):
            return [make_book(1)]

    monkeypatch.setattr(ingest_mod, "CalibreRepository", _RepoCls)
    monkeypatch.setattr(ingest_mod, "QdrantStore", _Store)
    monkeypatch.setattr(ingest_mod, "build_embedder", lambda s: _Embedder())
    monkeypatch.setattr(
        ingest_mod,
        "_safe_extract_sections",
        lambda fmt, path: [TextSection(title=None, text="Some content. " * 20)],
    )

    total = ingest_mod.ingest(settings)
    assert total == sum(store_calls["upserts"])
    assert total > 0
    assert store_calls["init"] == dict(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.vector_size,
        distance=settings.distance,
    )


def test_links_return_none_without_config() -> None:
    assert build_ebook_url(book_id=1, fmt="PDF", base_url=None, template=None) is None
    assert build_opds_url(book_id=1, base_url=None) is None
    assert build_opds_url(book_id=1, base_url="") is None
