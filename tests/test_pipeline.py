"""End-to-end runs: a real metadata.db and PostgreSQL, in-memory Qdrant, fake models."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from ebooklib import epub
from qdrant_client import QdrantClient

from calibre_web2rag import pipeline as pipeline_module
from calibre_web2rag.config import Settings
from calibre_web2rag.db import Database
from calibre_web2rag.documents import point_id
from calibre_web2rag.enrichment import Enricher
from calibre_web2rag.llm import LLM
from calibre_web2rag.pipeline import (
    BOOKTEXT_SOURCE,
    CATALOG_SOURCE,
    OVERVIEW_DOC_ID,
    SOURCE,
    Pipeline,
    load_taxonomy,
)
from calibre_web2rag.sparse import sparse_vector
from tests.conftest import CalibreLibrary, FakeEmbedder

SettingsFactory = Callable[..., Settings]


def make_epub(tmp_path: Path, chapters: list[tuple[str, str]]) -> bytes:
    book = epub.EpubBook()
    book.set_identifier("id")
    book.set_title("Book")
    book.set_language("en")
    items = []
    for number, (title, text) in enumerate(chapters, start=1):
        chapter = epub.EpubHtml(title=title, file_name=f"ch{number}.xhtml", lang="en")
        chapter.content = f"<html><body><h1>{title}</h1><p>{text}</p></body></html>"
        book.add_item(chapter)
        items.append(chapter)
    book.toc = [epub.Link(f"ch{n}.xhtml", t, f"ch{n}") for n, (t, _) in enumerate(chapters, 1)]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = items
    path = tmp_path / "made.epub"
    epub.write_epub(str(path), book)
    return path.read_bytes()


def llm_transport(
    topics: list[str] | None = None, marketing: bool = False
) -> tuple[httpx.MockTransport, list[str]]:
    """A gateway that answers book analysis, chunk context and link checks."""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        prompts.append(system)
        if "catalog books" in system:
            content = json.dumps(
                {
                    "topics": topics or ["antennas", "radio-and-rf", "not-a-topic"],
                    "description_is_marketing": marketing,
                    "good_for": "A practical guide for people building radios.",
                }
            )
        elif "place the excerpt" in system:
            content = "This part of the overview describes the library."
        else:
            content = '{"match": true}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler), prompts


class Harness:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        llm: bool = False,
        enricher: Enricher | None = None,
        marketing: bool = False,
    ) -> None:
        self.settings = settings
        self.db = db
        self.qdrant = QdrantClient(location=":memory:")
        self.embedder = FakeEmbedder()
        self.prompts: list[str] = []
        self.llm: LLM | None = None
        if llm:
            transport, self.prompts = llm_transport(marketing=marketing)
            self.llm = LLM(
                model="test-llm",
                base_url="http://gateway/v1",
                api_key="k",
                timeout_seconds=5,
                cache=db,
                transport=transport,
            )
        self.enricher = enricher
        self.stop = threading.Event()
        self.beats = 0

    def pipeline(self) -> Pipeline:
        return Pipeline(
            settings=self.settings,
            db=self.db,
            embedder=self.embedder,
            qdrant=self.qdrant,
            llm=self.llm,
            enricher=self.enricher,
            stop=self.stop,
            beat=self.beat,
        )

    def beat(self) -> None:
        self.beats += 1

    def run(self, mode: pipeline_module.Mode = "full") -> pipeline_module.RunResult:
        return self.pipeline().run(mode)

    # Any: payloads are JSON.
    def payloads(self, collection: str) -> list[dict[str, Any]]:
        records, _ = self.qdrant.scroll(collection, limit=1000, with_payload=True)
        return [r.payload or {} for r in records]

    # Any: payloads are JSON.
    def catalog(self) -> dict[str, dict[str, Any]]:
        return {p["work_id"]: p for p in self.payloads(self.settings.catalog_collection)}


@pytest.fixture
def harness_factory(
    db: Database, calibre: CalibreLibrary, make_settings: SettingsFactory
) -> Callable[..., Harness]:
    # Any: keyword arguments for Harness and Settings.
    def make(**kwargs: Any) -> Harness:
        harness_args = {k: kwargs.pop(k) for k in ("llm", "enricher", "marketing") if k in kwargs}
        settings = make_settings(
            calibre_metadata_db=str(calibre.db_path),
            calibre_library_root=str(calibre.library),
            **kwargs,
        )
        return Harness(settings=settings, db=db, **harness_args)

    return make


def test_full_run_builds_catalog_overview_and_tables(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database, tmp_path: Path
) -> None:
    calibre.add_book(
        1,
        "Practical RF Projects",
        authors=("Rhea Radio",),
        tags=("Electronics",),
        comments="<p>Hands-on projects.</p><p>Second paragraph.</p>",
        isbn="0-306-40615-2",
        series="Maker Shelf",
        series_index=2,
        formats={"EPUB": make_epub(tmp_path, [("Antenna Design Basics", "Dipoles.")])},
        has_cover=True,
    )
    calibre.add_book(2, "Woodshop Safety", authors=("Wendy Wood",), formats={"PDF": b"%PDF"})
    calibre.add_book(3, "No Files At All", authors=("Nina None",))
    harness = harness_factory(llm=True)

    result = harness.run("full")

    assert result.status == "ok"
    assert result.counts.seen == 3
    assert result.counts.added == 3
    assert harness.beats >= 4  # each book and the embedding batches
    # Acceptance: the catalog tool's ebook count equals Calibre's.
    assert db.catalog_count(source=SOURCE, format="ebook") == 3

    catalog = harness.catalog()
    assert set(catalog) == {
        "isbn:9780306406157",
        "title:woodshop-safety--wood",
        "title:no-files-at-all--none",
    }
    rf = catalog["isbn:9780306406157"]
    assert rf["formats"] == ["ebook"]
    assert rf["tier"] == "member"
    assert rf["source"] == CATALOG_SOURCE
    assert rf["series"] == "Maker Shelf"
    assert rf["topics"] == ["antennas", "radio-and-rf"]  # unknown slugs dropped
    assert "Antenna Design Basics" in rf["text"]
    assert "Hands-on projects.\n\nSecond paragraph." in rf["text"]
    assert rf["manifestations"][0]["url"] == "https://books.example/book/1"
    assert rf["manifestations"][0]["cover_path"] == "Rhea Radio/Practical RF Projects (1)/cover.jpg"
    assert rf["manifestations"][0]["files"][0]["url"] == "https://books.example/download/1/epub"
    assert rf["date_added"] == "2024-01-01T00:00:00+00:00"
    assert rf["year"] == 2019

    overview = harness.payloads(harness.settings.member_collection)
    assert {p["doc_id"] for p in overview} == {OVERVIEW_DOC_ID}
    overview.sort(key=lambda p: p["chunk_index"])
    # Chunk 0 is the document-level summary point.
    assert overview[0]["heading_path"] == ["The PDX Hackerspace ebook library"]
    texts = " ".join(p["text"] for p in overview)
    assert "ebook library of 3 books" in texts
    reach = [p["text"] for p in overview if p["heading_path"][-1].startswith("How to reach")]
    assert reach == [
        "How to reach the ebook library\nMembers open the library at https://books.example."
    ]
    assert all(p["context"] for p in overview)

    staged = db.staged_documents(SOURCE)
    assert [r["source_doc_id"] for r in staged] == ["1", "2", "3"]
    runs = db.recent_runs(SOURCE)
    assert runs[0]["ok"]
    assert runs[0]["seen"] == 3


def test_antenna_question_finds_book_by_its_contents(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, tmp_path: Path
) -> None:
    """Acceptance: "anything on antenna design?" finds a book whose title lacks the words."""
    epub_bytes = make_epub(
        tmp_path,
        [("Introduction", "Welcome."), ("Antenna Design for Beginners", "Yagis and dipoles.")],
    )
    calibre.add_book(1, "The Radio Amateur's Workbook", formats={"EPUB": epub_bytes})
    calibre.add_book(2, "Cooking for Crowds", comments="Soups and stews.")
    harness = harness_factory()
    harness.run("full")

    hits = harness.qdrant.query_points(
        harness.settings.catalog_collection,
        query=sparse_vector("Do we have anything on antenna design?"),
        using="sparse",
        limit=1,
    ).points
    assert hits[0].payload is not None
    assert hits[0].payload["title"] == "The Radio Amateur's Workbook"


def test_ebook_and_audiobook_share_one_catalog_point(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    """Acceptance: a book in two formats appears once in hs_catalog with both."""
    db.upsert_work(
        work_id="title:the-hobbit--tolkien",
        title="The Hobbit (Unabridged)",
        authors=["J. R. R. Tolkien"],
        isbn13=None,
        series_id=None,
        series_index=None,
    )
    db.upsert_manifestation(
        source="audiobookshelf",
        source_id="ab-1",
        work_id="title:the-hobbit--tolkien",
        format="audiobook",
        url="https://audio.example/item/ab-1",
        details={"description": "Read by a narrator.", "topics": ["fantasy"]},
    )
    db.commit()
    calibre.add_book(1, "The Hobbit", authors=("J.R.R. Tolkien",), isbn="9780261103344")
    harness = harness_factory()
    harness.run("full")

    catalog = harness.catalog()
    assert list(catalog) == ["title:the-hobbit--tolkien"]
    point = catalog["title:the-hobbit--tolkien"]
    assert point["formats"] == ["audiobook", "ebook"]
    assert {m["source"] for m in point["manifestations"]} == {"audiobookshelf", "calibre"}
    assert point["topics"] == ["fantasy"]
    # The merge decision is stored, so it survives re-indexing.
    assert db.merge_target("isbn:9780261103344") == "title:the-hobbit--tolkien"


def test_uncertain_match_goes_to_review(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    db.upsert_work(
        work_id="title:dune-messiah--herbert",
        title="Dune Messiah",
        authors=["Frank Herbert"],
        isbn13=None,
        series_id=None,
        series_index=None,
    )
    db.upsert_manifestation(
        source="hardcopy",
        source_id="h1",
        work_id="title:dune-messiah--herbert",
        format="hardcopy",
        url=None,
        details={"location": "Shelf B"},
    )
    db.commit()
    calibre.add_book(1, "Dune the Messiah", authors=("Frank Herbert",))
    calibre.add_book(2, "Children of Dune", authors=("Frank Herbert",))
    harness_factory().run("full")

    rows = db.conn.execute("SELECT left_id, right_id FROM registry_review_queue").fetchall()
    assert rows == [
        {"left_id": "title:dune-the-messiah--herbert", "right_id": "title:dune-messiah--herbert"}
    ]


def test_incremental_restages_only_changed_books_and_propagates_deletes(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    for book_id in (1, 2, 3):
        calibre.add_book(book_id, f"Book {book_id}", comments=f"About book {book_id}.")
    harness = harness_factory()
    harness.run("full")
    harness.embedder.calls.clear()

    calibre.touch(2, "2025-05-05 00:00:00+00:00", title="Book Two Revised")
    calibre.delete_book(3)
    result = harness.run("incremental")

    assert (result.counts.added, result.counts.updated, result.counts.deleted) == (0, 1, 1)
    assert set(harness.catalog()) == {"title:book-1--author", "title:book-two-revised--author"}
    assert db.catalog_count(source=SOURCE) == 2
    # Only the changed work and the overview were re-embedded.
    embedded = " ".join(harness.embedder.texts)
    assert "Book Two Revised" in embedded
    assert "About book 1" not in embedded


def test_unchanged_full_run_embeds_nothing_new(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    calibre.add_book(1, "Steady")
    harness = harness_factory()
    harness.run("full")
    harness.embedder.calls.clear()
    harness.run("full")
    assert harness.embedder.calls == []
    run_ids = {p["sync_run_id"] for p in harness.payloads(harness.settings.member_collection)}
    assert len(run_ids) == 1


def test_safety_valve_skips_deletes(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    for book_id in range(1, 6):
        calibre.add_book(book_id, f"Book {book_id}")
    harness = harness_factory()
    harness.run("full")
    for book_id in range(1, 5):
        calibre.delete_book(book_id)

    result = harness.run("full")

    assert result.status == "valve"
    assert "not deleting 4 books" in result.message
    assert db.catalog_count(source=SOURCE) == 5
    assert db.recent_runs(SOURCE)[0]["ok"] is False


def test_reindex_rebuilds_from_staging_without_calibre(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    calibre.add_book(1, "Kept")
    harness = harness_factory()
    harness.run("full")
    calibre.db_path.unlink()  # reindex must not read Calibre
    harness.embedder.calls.clear()

    result = harness.run("reindex")

    assert result.status == "ok"
    assert any("Kept" in t for t in harness.embedder.texts)


def test_secrets_are_redacted_and_quarantined(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    calibre.add_book(1, "Leaky", comments="Wifi password = hunter22secret for the shop.")
    harness = harness_factory()
    result = harness.run("full")

    assert result.counts.quarantined == 1
    text = harness.catalog()["title:leaky--author"]["text"]
    assert "hunter22secret" not in text
    assert "[REDACTED]" in text
    staged = db.staged_documents(SOURCE)[0]
    assert staged["flags"] == {"secret_redacted": True}


def test_marketing_description_gets_generated_summary(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    calibre.add_book(1, "Hyped", comments="The most amazing book of the decade!")
    harness = harness_factory(llm=True, marketing=True)
    harness.run("full")
    point = harness.catalog()["title:hyped--author"]
    assert point["description_generated"] is True
    assert "machine-generated summary" in point["text"]


def test_llm_results_are_cached(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    calibre.add_book(1, "Cached", comments="Text.")
    harness = harness_factory(llm=True)
    harness.run("full")
    first = len(harness.prompts)
    calibre.touch(1, "2025-01-01 00:00:00+00:00")
    harness.run("incremental")
    assert len(harness.prompts) == first


def test_llm_failure_does_not_stop_the_run(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    calibre.add_book(1, "Offline", comments="Text.")
    harness = harness_factory()
    harness.llm = LLM(
        model="m",
        base_url="http://gateway/v1",
        api_key=None,
        timeout_seconds=5,
        cache=db,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    db.conn.execute(
        "INSERT INTO entity_aliases VALUES ('asset', 'a1', 'Offline'), ('asset', 'a2', 'Offline')"
    )
    db.commit()
    assert harness.run("full").status == "ok"
    point = harness.catalog()["title:offline--author"]
    assert point["topics"] == []
    assert point["asset_ids"] == []  # ambiguous alias, unconfirmed


def test_enrichment_fills_missing_description(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/isbn/9780306406157.json":
            return httpx.Response(200, json={"title": "Bare", "works": [{"key": "/works/OL1W"}]})
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(
                200, json={"description": {"value": "All about bare things."}, "subjects": ["X"]}
            )
        return httpx.Response(404)

    enricher = Enricher(
        timeout_seconds=5, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    calibre.add_book(1, "Bare", isbn="9780306406157")
    harness = harness_factory(enricher=enricher, enrichment_enabled=True)
    harness.run("full")

    point = harness.catalog()["isbn:9780306406157"]
    assert "All about bare things." in point["text"]
    assert "Subjects: X" in point["text"]
    rows = db.enrichment_get(SOURCE, "1")
    assert [r["origin"] for r in rows] == ["openlibrary"]
    # Library data is never overwritten.
    assert db.staged_documents(SOURCE)[0]["raw_content"]["comments"] is None


def test_booktext_is_off_by_default_and_indexed_when_enabled(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, tmp_path: Path
) -> None:
    epub_bytes = make_epub(tmp_path, [("Chapter One", "The quick brown fox. " * 20)])
    calibre.add_book(1, "Foxes", formats={"EPUB": epub_bytes})
    off = harness_factory()
    off.run("full")
    assert not off.qdrant.collection_exists(off.settings.booktext_collection)

    on = harness_factory(booktext_enabled=True)
    on.run("full")
    points = on.payloads(on.settings.booktext_collection)
    assert points
    assert {p["doc_id"] for p in points} == {"calibre:book:1"}
    assert points[0]["heading_path"] == ["Foxes", "Chapter One"]
    on.embedder.calls.clear()
    on.run("full")
    assert on.embedder.calls == []  # unchanged books are not re-embedded

    calibre.delete_book(1)
    calibre.add_book(2, "Other")
    calibre.add_book(3, "Another")
    on.run("full")
    assert on.payloads(on.settings.booktext_collection) == []
    assert on.db.staged_documents(BOOKTEXT_SOURCE) == []


def test_tier_and_collections(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    """Every point is member tier, and nothing goes to hs_public."""
    calibre.add_book(1, "Members Only")
    harness = harness_factory()
    harness.run("full")
    assert not harness.qdrant.collection_exists("hs_public")
    for collection in (harness.settings.member_collection, harness.settings.catalog_collection):
        assert {p["tier"] for p in harness.payloads(collection)} == {"member"}
    info = harness.qdrant.get_collection(harness.settings.catalog_collection)
    assert info.config.metadata is not None
    assert info.config.metadata["embedding_model"] == "dummy"


def test_no_per_user_data_is_staged(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    calibre.add_book(1, "Plain")
    harness_factory().run("full")
    raw = db.staged_documents(SOURCE)[0]["raw_content"]
    forbidden = {"shelves", "read_status", "downloads", "user", "users", "bookmarks"}
    assert not forbidden & set(raw)


def test_busy_lock_skips_the_run(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, database_url: str
) -> None:
    calibre.add_book(1, "Locked")
    harness = harness_factory()
    with Database.connect(database_url) as other, other.source_lock(SOURCE) as locked:
        assert locked
        assert harness.run("full").status == "busy"


def test_stop_ends_the_run_early(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary, db: Database
) -> None:
    calibre.add_book(1, "One")
    harness = harness_factory()
    harness.stop.set()
    result = harness.run("full")
    assert result.message == "stopped early"
    assert db.staged_documents(SOURCE) == []


def test_stopped_run_indexes_what_it_staged(
    harness_factory: Callable[..., Harness],
    calibre: CalibreLibrary,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibre.add_book(1, "First")
    calibre.add_book(2, "Second")
    harness = harness_factory(booktext_enabled=True)
    original = Pipeline.stage_book

    # Any: stage_book's arguments.
    def stop_after(self: Pipeline, book: Any, *args: Any) -> set[str]:
        touched = original(self, book, *args)
        self.stop.set()  # SIGTERM arrives while the first book is staged
        return touched

    monkeypatch.setattr(Pipeline, "stage_book", stop_after)
    result = harness.run("full")

    assert result.message == "stopped early"
    assert [r["source_doc_id"] for r in db.staged_documents(SOURCE)] == ["1"]
    assert set(harness.catalog()) == {"title:first--author"}
    assert harness.payloads(harness.settings.member_collection)


def test_a_bad_book_is_skipped(
    harness_factory: Callable[..., Harness],
    calibre: CalibreLibrary,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibre.add_book(1, "Good")
    calibre.add_book(2, "Bad")
    original = Pipeline.stage_book

    # Any: stage_book's arguments.
    def flaky(self: Pipeline, book: Any, *args: Any) -> set[str]:
        if book.title == "Bad":
            raise ValueError("broken record")
        return original(self, book, *args)

    monkeypatch.setattr(Pipeline, "stage_book", flaky)
    result = harness_factory().run("full")
    assert result.counts.errors == ["book 2: ValueError: broken record"]
    assert [r["source_doc_id"] for r in db.staged_documents(SOURCE)] == ["1"]


def test_failure_is_recorded_and_raised(
    harness_factory: Callable[..., Harness], db: Database
) -> None:
    harness = harness_factory()  # no metadata.db written beyond the schema
    Path(harness.settings.calibre_metadata_db).unlink()
    with pytest.raises(FileNotFoundError):
        harness.run("full")
    run = db.recent_runs(SOURCE)[0]
    assert run["ok"] is False
    assert "metadata.db not found" in run["message"]


def test_status_and_run_helpers(
    calibre: CalibreLibrary,
    make_settings: SettingsFactory,
    database_url: str,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibre.add_book(1, "Status")
    settings = make_settings(
        calibre_metadata_db=str(calibre.db_path),
        calibre_library_root=str(calibre.library),
        database_url=database_url,
        llm_model=None,
        enrichment_enabled=True,
    )
    monkeypatch.setattr(pipeline_module, "build_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(
        pipeline_module, "QdrantClient", lambda url, api_key: QdrantClient(location=":memory:")
    )
    result = pipeline_module.run(settings, "full")
    assert result.status == "ok"
    report = pipeline_module.status(settings)
    assert report["catalog_ebooks"] == 1
    assert report["runs"][0]["mode"] == "full"


def test_build_pipeline_wires_llm_and_enricher(
    make_settings: SettingsFactory, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "build_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(
        pipeline_module, "QdrantClient", lambda url, api_key: QdrantClient(location=":memory:")
    )
    pipe = pipeline_module.build_pipeline(
        make_settings(llm_model="gen", enrichment_enabled=True), db, None
    )
    assert pipe.llm is not None
    assert pipe.llm.model == "gen"
    assert pipe.enricher is not None
    bare = pipeline_module.build_pipeline(make_settings(), db, None)
    assert bare.llm is None
    assert bare.enricher is None
    assert bare._confirm_link("t", "a", "asset", "x") is False


def test_load_taxonomy(tmp_path: Path) -> None:
    default = load_taxonomy(None)
    assert "antennas" in default
    assert "science-fiction" in default
    custom = tmp_path / "topics.txt"
    custom.write_text("# comment\nalpha\n\nbeta\n", encoding="utf-8")
    assert load_taxonomy(str(custom)) == ["alpha", "beta"]


def test_point_ids_are_stable() -> None:
    assert point_id(SOURCE, "doc", 0) == point_id(SOURCE, "doc", 0)
    assert point_id(SOURCE, "doc", 0) != point_id(SOURCE, "doc", 1)


def test_payload_only_change_is_written_without_reembedding(
    harness_factory: Callable[..., Harness], calibre: CalibreLibrary
) -> None:
    calibre.add_book(1, "Woodshop Safety", authors=("Wendy Wood",), formats={"PDF": b"%PDF"})
    harness = harness_factory()
    harness.run("full")
    (before,) = harness.catalog().values()
    assert before["year"] == 2019
    harness.embedder.calls.clear()
    # The year isn't in the catalog text: the payload changes, the vector doesn't.
    calibre.touch(1, "2025-01-01 00:00:00+00:00", pubdate="2021-03-01 00:00:00+00:00")
    harness.run("incremental")
    (after,) = harness.catalog().values()
    assert after["year"] == 2021
    assert after["text"] == before["text"]
    assert not [t for t in harness.embedder.texts if "Woodshop" in t]


def test_failed_index_is_retried_by_the_next_incremental_run(
    harness_factory: Callable[..., Harness],
    calibre: CalibreLibrary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibre.add_book(1, "Woodshop Safety", authors=("Wendy Wood",), formats={"PDF": b"%PDF"})
    harness = harness_factory()
    pipeline = harness.pipeline()

    # Any: whatever index_catalog is called with.
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(pipeline, "index_catalog", broken)
    with pytest.raises(RuntimeError, match="qdrant down"):
        pipeline.run("incremental")
    assert harness.catalog() == {}
    # Staging kept the book but forgot its version: the next run restages and indexes it.
    result = harness.run("incremental")
    assert result.counts.updated == 1
    assert len(harness.catalog()) == 1


def test_close_releases_http_clients(harness_factory: Callable[..., Harness]) -> None:
    # serve builds a pipeline per scan, so each one must close what it opened.
    harness = harness_factory(llm=True, enricher=Enricher(timeout_seconds=5))
    p = harness.pipeline()
    p.close()
    assert harness.llm is not None
    assert harness.llm._client.is_closed
    assert harness.enricher is not None
    assert harness.enricher._client.is_closed
