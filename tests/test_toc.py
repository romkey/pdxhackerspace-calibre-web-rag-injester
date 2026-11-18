from pathlib import Path

import pytest
from ebooklib import epub
from pypdf import PdfWriter

from calibre_web2rag import toc
from calibre_web2rag.models import CalibreFile, TocEntry


def _epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("id")
    book.set_title("T")
    book.set_language("en")
    chapters = []
    for n in (1, 2, 3):
        chapter = epub.EpubHtml(title=f"C{n}", file_name=f"c{n}.xhtml", lang="en")
        chapter.content = f"<html><body><p>{n}</p></body></html>"
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = [
        epub.Link("c1.xhtml", "Introduction", "c1"),
        (epub.Section("Part One"), [epub.Link("c2.xhtml", "Antenna Design", "c2")]),
        chapters[2],
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = chapters
    epub.write_epub(str(path), book)
    return path


def _pdf(path: Path, *, outline: bool = True, encrypted: bool = False) -> Path:
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    if outline:
        parent = writer.add_outline_item("Chapter 1", 0)
        writer.add_outline_item("Section 1.1", 1, parent=parent)
        writer.add_outline_item("Chapter 2", 2)
    if encrypted:
        writer.encrypt(user_password="pw", algorithm="RC4-128")  # noqa: S106 -- test PDF
    with path.open("wb") as fp:
        writer.write(fp)
    return path


def test_epub_toc(tmp_path: Path) -> None:
    entries = toc.epub_toc(_epub(tmp_path / "b.epub"))
    assert entries == [
        TocEntry("Introduction", 0),
        TocEntry("Part One", 0),
        TocEntry("Antenna Design", 1),
        TocEntry("C3", 0),
    ]
    assert toc.toc_text(entries) == "Introduction\nPart One\n  Antenna Design\nC3"


def test_pdf_toc(tmp_path: Path) -> None:
    assert toc.pdf_toc(_pdf(tmp_path / "b.pdf")) == [
        TocEntry("Chapter 1", 0, 1),
        TocEntry("Section 1.1", 1, 2),
        TocEntry("Chapter 2", 0, 3),
    ]
    assert toc.pdf_toc(_pdf(tmp_path / "e.pdf", encrypted=True)) == []


def test_unreadable_files(tmp_path: Path) -> None:
    junk = tmp_path / "junk"
    junk.write_text("not a book", encoding="utf-8")
    assert toc.epub_toc(junk) == []
    assert toc.pdf_toc(junk) == []


def test_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toc, "MAX_ENTRIES", 1)
    assert len(toc.epub_toc(_epub(tmp_path / "b.epub"))) == 1
    assert len(toc.pdf_toc(_pdf(tmp_path / "b.pdf"))) == 1
    monkeypatch.setattr(toc, "MAX_ENTRIES", 200)
    monkeypatch.setattr(toc, "MAX_DEPTH", 1)
    assert [e.title for e in toc.pdf_toc(_pdf(tmp_path / "c.pdf"))] == ["Chapter 1", "Chapter 2"]
    assert "Antenna Design" not in [e.title for e in toc.epub_toc(_epub(tmp_path / "d.epub"))]


def test_book_toc_prefers_epub_and_falls_back(tmp_path: Path) -> None:
    pdf = CalibreFile("PDF", _pdf(tmp_path / "b.pdf"), "b", 1)
    no_outline = CalibreFile("PDF", _pdf(tmp_path / "n.pdf", outline=False), "n", 1)
    book_epub = CalibreFile("EPUB", _epub(tmp_path / "b.epub"), "b", 1)
    missing = CalibreFile("EPUB", tmp_path / "gone.epub", "gone", 1, exists=False)
    assert toc.book_toc([pdf, book_epub])[0].title == "Introduction"
    assert toc.book_toc([missing, pdf])[0].title == "Chapter 1"
    assert toc.book_toc([no_outline]) == []
    assert toc.book_toc([CalibreFile("MOBI", tmp_path / "x.mobi", "x", 1)]) == []
