from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CalibreFile:
    format: str
    file_path: Path
    file_name: str
    size_bytes: int | None


@dataclass(frozen=True)
class BookRecord:
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
