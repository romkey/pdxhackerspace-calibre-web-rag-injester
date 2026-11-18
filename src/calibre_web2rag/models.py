"""Plain records passed between the stages of a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TextSection:
    title: str | None
    text: str
    # 1-based page number, for sections that come from a PDF page.
    page: int | None = None


@dataclass(frozen=True)
class TocEntry:
    title: str
    level: int = 0
    page: int | None = None


@dataclass(frozen=True)
class CalibreFile:
    format: str
    file_path: Path
    file_name: str
    size_bytes: int | None
    # False when metadata.db lists the format but the file isn't on disk.
    exists: bool = True


@dataclass(frozen=True)
class BookRecord:
    """One book as Calibre's metadata.db describes it."""

    book_id: int
    title: str
    authors: list[str]
    tags: list[str]
    comments: str | None
    publisher: str | None
    series: str | None
    rating: int | None
    languages: list[str]
    identifiers: dict[str, str]
    isbn: str | None
    uuid: str | None
    published_at: str | None
    updated_at: str | None
    path: str
    files: list[CalibreFile] = field(default_factory=list)
    sort_title: str | None = None
    subtitle: str | None = None
    series_index: float | None = None
    added_at: str | None = None
    has_cover: bool = False
    # Custom columns by label; values are str, int, float, bool or lists of str.
    custom_columns: dict[str, object] = field(default_factory=dict)

    @property
    def cover_path(self) -> str | None:
        """Cover image path relative to the library root (Calibre's layout)."""
        return f"{self.path}/cover.jpg" if self.has_cover else None
