"""Tables of contents from EPUB navigation and PDF outlines.

Chapter titles make technical books findable by topic, so each book's TOC
goes into its catalog text. MOBI files have no reliable TOC to read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ebooklib import epub
from pypdf import PdfReader

from calibre_web2rag.models import CalibreFile, TocEntry

logger = logging.getLogger(__name__)

# Enough to describe a book's topics without bloating its catalog text.
MAX_ENTRIES = 200
MAX_DEPTH = 3

# Prefer EPUB: its navigation is structured; PDF outlines are often absent.
_FORMAT_ORDER = ("EPUB", "PDF")


# Any: ebooklib's toc is a nested list of Link, Section, EpubHtml and
# (Section, children) tuples with no type information.
def _walk_epub(items: Any, level: int, out: list[TocEntry]) -> None:
    for item in items:
        if len(out) >= MAX_ENTRIES:
            return
        if isinstance(item, tuple) and len(item) == 2:
            section, children = item
            title = getattr(section, "title", None)
            if title and level < MAX_DEPTH:
                out.append(TocEntry(title=str(title).strip(), level=level))
            if level + 1 < MAX_DEPTH:
                _walk_epub(children, level + 1, out)
            continue
        title = getattr(item, "title", None)
        if title and str(title).strip():
            out.append(TocEntry(title=str(title).strip(), level=level))


def epub_toc(path: Path) -> list[TocEntry]:
    try:
        book = epub.read_epub(str(path))
    except Exception as e:  # ebooklib raises many types on damaged files
        logger.warning("Cannot read EPUB table of contents from %s: %s", path, e)
        return []
    entries: list[TocEntry] = []
    _walk_epub(book.toc, 0, entries)
    return entries


# Any: pypdf's outline is a nested list of Destination objects and lists.
def _walk_pdf(reader: PdfReader, items: Any, level: int, out: list[TocEntry]) -> None:
    for item in items:
        if len(out) >= MAX_ENTRIES:
            return
        if isinstance(item, list):
            if level + 1 < MAX_DEPTH:
                _walk_pdf(reader, item, level + 1, out)
            continue
        title = str(getattr(item, "title", "") or "").strip()
        if not title:
            continue
        number: int | None
        try:
            number = reader.get_destination_page_number(item)
        except Exception:  # broken destinations are common; keep the title
            number = None
        page = number + 1 if number is not None and number >= 0 else None
        out.append(TocEntry(title=title, level=level, page=page))


def pdf_toc(path: Path) -> list[TocEntry]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return []
        outline = reader.outline
    except Exception as e:  # pypdf raises many types on damaged files
        logger.warning("Cannot read PDF outline from %s: %s", path, e)
        return []
    entries: list[TocEntry] = []
    _walk_pdf(reader, outline, 0, entries)
    return entries


def book_toc(files: list[CalibreFile]) -> list[TocEntry]:
    """The table of contents from the book's first EPUB or PDF that has one."""
    by_format = {f.format: f for f in files if f.exists}
    for fmt in _FORMAT_ORDER:
        file = by_format.get(fmt)
        if file is None:
            continue
        entries = epub_toc(file.file_path) if fmt == "EPUB" else pdf_toc(file.file_path)
        if entries:
            return entries
    return []


def toc_text(entries: list[TocEntry]) -> str:
    return "\n".join(("  " * e.level) + e.title for e in entries)
