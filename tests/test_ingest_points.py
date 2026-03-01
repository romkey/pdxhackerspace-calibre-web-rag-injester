from pathlib import Path

from calibre_web2rag.config import Settings
from calibre_web2rag.ingest import _generate_points
from calibre_web2rag.models import BookRecord, CalibreFile


class _Repo:
    def __init__(self, books: list[BookRecord]) -> None:
        self._books = books

    def fetch_books(self) -> list[BookRecord]:
        return self._books


class _Embedder:
    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[list[float]]:  # noqa: ARG002
        return [[0.1, 0.2, 0.3] for _ in chunks]


def test_generate_points_includes_metadata(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_text("fake", encoding="utf-8")

    book = BookRecord(
        book_id=1,
        title="Title",
        authors=["Author"],
        tags=["tag"],
        comments="notes",
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

    monkeypatch.setattr("calibre_web2rag.ingest._drm_free", lambda fmt, path: True)
    monkeypatch.setattr("calibre_web2rag.ingest._extract_text", lambda fmt, path: "hello world " * 50)

    settings = Settings(
        calibre_metadata_db="db.sqlite",
        calibre_library_root=str(tmp_path),
        calibre_web_base_url="https://books.example",
        calibre_download_url_template=None,
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="books",
        embedding_model="dummy",
        chunk_size=40,
        chunk_overlap=10,
        batch_size=10,
        distance="cosine",
        vector_size=3,
    )

    batches = list(_generate_points(repo=_Repo([book]), embedder=_Embedder(), settings=settings))
    points = [point for batch in batches for point in batch]
    assert points
    payload = points[0].payload
    assert payload["book_id"] == 1
    assert payload["source_url"] == "https://books.example/download/1/pdf"
    assert payload["opds_url"] == "https://books.example/opds/book/1"
