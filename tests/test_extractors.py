"""Extractor tests, using real EPUB and PDF files and a stubbed MOBI unpacker."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import mobi
import pytest
from ebooklib import epub
from pypdf import PdfWriter

from calibre_web2rag import extractors
from calibre_web2rag.extractors import (
    extract_epub_sections,
    extract_epub_text,
    extract_mobi_sections,
    extract_mobi_text,
    extract_pdf_sections,
    extract_pdf_text,
    is_epub_drm_free,
    is_mobi_drm_free,
    is_pdf_drm_free,
    try_extract_epub_sections,
    try_extract_mobi_sections,
    try_extract_pdf_sections,
)
from calibre_web2rag.models import TextSection

# The fake_mobi fixture: call it with HTML; returns the temp dirs it creates.
MobiInstaller = Callable[[str], list[Path]]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def epub_path(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("id-1")
    book.set_title("Test Book")
    book.set_language("en")

    ch1 = epub.EpubHtml(title="One", file_name="ch1.xhtml", lang="en")
    ch1.content = "<html><body><h1>Chapter One</h1><p>First chapter text.</p></body></html>"
    ch2 = epub.EpubHtml(title="Two", file_name="ch2.xhtml", lang="en")
    ch2.content = "<html><body><h3>Chapter Two</h3><p>Second chapter text.</p></body></html>"
    ch3 = epub.EpubHtml(title="Three", file_name="ch3.xhtml", lang="en")
    ch3.content = "<html><body><p>No heading here.</p></body></html>"
    for ch in (ch1, ch2, ch3):
        book.add_item(ch)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [ch1, ch2, ch3]

    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def drm_epub_path(epub_path: Path, tmp_path: Path) -> Path:
    path = tmp_path / "drm.epub"
    with ZipFile(epub_path) as src, ZipFile(path, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("META-INF/encryption.xml", "<encryption/>")
    return path


@pytest.fixture
def blank_pdf_path(tmp_path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as fp:
        writer.write(fp)
    return path


@pytest.fixture
def encrypted_pdf_path(tmp_path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="secret", algorithm="RC4-128")  # noqa: S106 -- test PDF, not a credential
    path = tmp_path / "encrypted.pdf"
    with path.open("wb") as fp:
        writer.write(fp)
    return path


class _FakePdfReader:
    """PdfReader stand-in that yields pages with fixed text."""

    def __init__(self, path: str, *, texts: list[str | None], encrypted: bool = False) -> None:
        self.is_encrypted = encrypted
        self.pages = [SimpleNamespace(extract_text=lambda t=t: t) for t in texts]


def _patch_pdf_reader(
    monkeypatch: pytest.MonkeyPatch, texts: list[str | None], encrypted: bool = False
) -> None:
    monkeypatch.setattr(
        extractors,
        "PdfReader",
        lambda path: _FakePdfReader(path, texts=texts, encrypted=encrypted),
    )


@pytest.fixture
def fake_mobi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MobiInstaller:
    """Replace mobi.extract with one that writes the given HTML to a temp dir."""

    created: list[Path] = []

    def _install(html: str) -> list[Path]:
        def _extract(path: str) -> tuple[str, str]:
            tempdir = tmp_path / f"mobi-{len(created)}"
            tempdir.mkdir()
            (tempdir / "book.html").write_text(html, encoding="utf-8")
            created.append(tempdir)
            return str(tempdir), "book.html"

        monkeypatch.setattr(mobi, "extract", _extract)
        return created

    return _install


# ---------------------------------------------------------------------------
# DRM checks
# ---------------------------------------------------------------------------


def test_pdf_drm_checks(blank_pdf_path: Path, encrypted_pdf_path: Path, tmp_path: Path) -> None:
    assert is_pdf_drm_free(blank_pdf_path)
    assert not is_pdf_drm_free(encrypted_pdf_path)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"garbage")
    assert not is_pdf_drm_free(bad)


def test_epub_empty_encryption_manifest_is_drm_free(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    with ZipFile(path, "w") as archive:
        archive.writestr("META-INF/encryption.xml", "   ")
    assert is_epub_drm_free(path)


def test_epub_drm_check_on_non_zip(tmp_path: Path) -> None:
    path = tmp_path / "bad.epub"
    path.write_bytes(b"not a zip")
    assert not is_epub_drm_free(path)


def test_mobi_drm_check_edge_cases(tmp_path: Path) -> None:
    short = tmp_path / "short.mobi"
    short.write_bytes(b"\x00" * 4)
    assert not is_mobi_drm_free(short)
    assert not is_mobi_drm_free(tmp_path / "missing.mobi")


def test_epub_drm_free_when_encryption_manifest_missing(tmp_path: Path) -> None:
    book = tmp_path / "book.epub"
    with ZipFile(book, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("content.xhtml", "<html><body>hello</body></html>")
    assert is_epub_drm_free(book)


def test_epub_drm_detection(tmp_path: Path) -> None:
    book = tmp_path / "book.epub"
    with ZipFile(book, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("META-INF/encryption.xml", "<enc/>")
    assert not is_epub_drm_free(book)


def test_mobi_drm_detection(tmp_path: Path) -> None:
    plain = tmp_path / "plain.mobi"
    encrypted = tmp_path / "encrypted.mobi"
    plain.write_bytes(b"\x00" * 12 + b"\x00\x00" + b"\x00" * 4)
    encrypted.write_bytes(b"\x00" * 12 + b"\x00\x01" + b"\x00" * 4)
    assert is_mobi_drm_free(plain)
    assert not is_mobi_drm_free(encrypted)


def test_try_extract_mobi_returns_empty_on_bad_file(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "garbage.mobi"
    bad.write_bytes(b"not a real mobi file")
    assert try_extract_mobi_sections(bad) == []


def test_try_extract_pdf_returns_empty_on_bad_file(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"not a real pdf file")
    assert try_extract_pdf_sections(bad) == []


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


def test_extract_epub_sections_titles(epub_path: Path) -> None:
    sections = extract_epub_sections(epub_path)
    by_title = {s.title: s.text for s in sections}
    assert "First chapter text." in by_title["Chapter One"]
    assert "Second chapter text." in by_title["Chapter Two"]
    assert any(s.title is None and "No heading here." in s.text for s in sections)


def test_extract_epub_text(epub_path: Path) -> None:
    text = extract_epub_text(epub_path)
    assert "First chapter text." in text
    assert "Second chapter text." in text


def test_try_extract_epub_sections_clean(epub_path: Path) -> None:
    assert try_extract_epub_sections(epub_path) == extract_epub_sections(epub_path)


def test_try_extract_epub_sections_drm(drm_epub_path: Path) -> None:
    assert try_extract_epub_sections(drm_epub_path) == []


def test_try_extract_epub_sections_not_a_zip(tmp_path: Path) -> None:
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"not a zip")
    assert try_extract_epub_sections(bad) == []


def test_try_extract_epub_sections_zip_but_not_epub(tmp_path: Path) -> None:
    path = tmp_path / "plain.epub"
    with ZipFile(path, "w") as archive:
        archive.writestr("hello.txt", "hi")
    assert try_extract_epub_sections(path) == []


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_extract_pdf_text_and_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdf_reader(monkeypatch, ["page one", None, "page three"])
    assert extract_pdf_text(Path("x.pdf")) == "page one\n\npage three"
    assert extract_pdf_sections(Path("x.pdf")) == [
        TextSection(title=None, text="page one\n\npage three")
    ]


def test_extract_pdf_sections_empty(blank_pdf_path: Path) -> None:
    assert extract_pdf_sections(blank_pdf_path) == []


def test_try_extract_pdf_sections_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdf_reader(monkeypatch, ["hello", "world"])
    assert try_extract_pdf_sections(Path("x.pdf")) == [TextSection(title=None, text="hello\nworld")]


def test_try_extract_pdf_sections_encrypted(encrypted_pdf_path: Path) -> None:
    assert try_extract_pdf_sections(encrypted_pdf_path) == []


def test_try_extract_pdf_sections_blank(blank_pdf_path: Path) -> None:
    assert try_extract_pdf_sections(blank_pdf_path) == []


# ---------------------------------------------------------------------------
# MOBI
# ---------------------------------------------------------------------------


def test_extract_mobi_text_and_sections_clean_up(fake_mobi: MobiInstaller) -> None:
    created = fake_mobi("<html><body><p>Mobi body text</p></body></html>")
    assert extract_mobi_text(Path("x.mobi")) == "Mobi body text"
    assert extract_mobi_sections(Path("x.mobi")) == [TextSection(title=None, text="Mobi body text")]
    assert len(created) == 2
    assert all(not d.exists() for d in created)


def test_extract_mobi_sections_empty(fake_mobi: MobiInstaller) -> None:
    fake_mobi("<html><body></body></html>")
    assert extract_mobi_sections(Path("x.mobi")) == []


def test_try_extract_mobi_sections(fake_mobi: MobiInstaller) -> None:
    created = fake_mobi("<html><body><p>Hello mobi</p></body></html>")
    assert try_extract_mobi_sections(Path("x.mobi")) == [TextSection(title=None, text="Hello mobi")]
    assert not created[0].exists()


def test_try_extract_mobi_sections_empty(fake_mobi: MobiInstaller) -> None:
    created = fake_mobi("<html><body>   </body></html>")
    assert try_extract_mobi_sections(Path("x.mobi")) == []
    assert not created[0].exists()
