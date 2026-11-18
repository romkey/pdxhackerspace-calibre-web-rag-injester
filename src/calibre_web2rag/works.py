"""Work identity: ISBN normalization, work_id and fuzzy title matching.

A work is a book in any format. Its ``work_id`` is ``isbn:<ISBN-13>`` when
the book has a valid ISBN, else ``title:<normalized-title>--<surname>``.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Edition noise stripped before titles are compared or slugged.
_EDITION_NOISE = re.compile(
    r"""
    \b(?:
        \d+(?:st|nd|rd|th)\s+(?:revised\s+|expanded\s+|updated\s+)?ed(?:ition|\.)?
      | (?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)
        \s+(?:revised\s+|expanded\s+|updated\s+)?edition
      | (?:revised|expanded|updated|anniversary|special|illustrated|deluxe|
           collector'?s|international|kindle|large\s+print)\s+edition
      | unabridged | abridged
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BRACKETED = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _isbn10_valid(digits: str) -> bool:
    total = sum((10 - i) * (10 if ch == "X" else int(ch)) for i, ch in enumerate(digits))
    return total % 11 == 0


def _isbn13_check_digit(first12: str) -> str:
    total = sum(int(ch) * (1 if i % 2 == 0 else 3) for i, ch in enumerate(first12))
    return str((10 - total % 10) % 10)


def normalize_isbn(value: str | None) -> str | None:
    """Return the ISBN-13 for an ISBN-10 or ISBN-13 string, or None if invalid."""
    if not value:
        return None
    cleaned = re.sub(r"[^0-9Xx]", "", value).upper()
    if len(cleaned) == 10 and re.fullmatch(r"\d{9}[\dX]", cleaned) and _isbn10_valid(cleaned):
        first12 = "978" + cleaned[:9]
        return first12 + _isbn13_check_digit(first12)
    if (
        len(cleaned) == 13
        and cleaned.isdigit()
        and cleaned.startswith(("978", "979"))
        and _isbn13_check_digit(cleaned[:12]) == cleaned[12]
    ):
        return cleaned
    return None


def book_isbn13(identifiers: dict[str, str]) -> str | None:
    """The book's ISBN-13 from Calibre identifiers (``isbn``, then ``isbn13``)."""
    for key in ("isbn", "isbn13", "isbn10"):
        isbn = normalize_isbn(identifiers.get(key))
        if isbn:
            return isbn
    return None


def normalize_title(title: str) -> str:
    """Lowercase ASCII words without edition noise, brackets or articles."""
    text = _ascii_fold(title).lower()
    # "Title: Subtitle" matches "Title" in other sources more often than not.
    text = text.split(":", 1)[0]
    text = _BRACKETED.sub(" ", text)
    text = _EDITION_NOISE.sub(" ", text)
    text = _NON_WORD.sub(" ", text).strip()
    return _LEADING_ARTICLE.sub("", text)


def author_surname(author: str | None) -> str:
    """Surname from "First Last" or Calibre's "Last, First"; lowercase ASCII."""
    if not author:
        return ""
    name = _ascii_fold(author).strip()
    if "," in name:
        name = name.split(",", 1)[0]
    else:
        parts = name.split()
        name = parts[-1] if parts else ""
    return _NON_WORD.sub("", name.lower())


def work_id_for(*, title: str, authors: list[str], isbn13: str | None) -> str:
    if isbn13:
        return f"isbn:{isbn13}"
    slug = normalize_title(title).replace(" ", "-") or "untitled"
    surname = author_surname(authors[0] if authors else None) or "unknown"
    return f"title:{slug}--{surname}"


def series_id_for(name: str) -> str:
    return "series:" + (_NON_WORD.sub("-", _ascii_fold(name).lower()).strip("-") or "unnamed")


def similarity(title_a: str, title_b: str) -> float:
    """How alike two titles are after normalization, 0.0 to 1.0."""
    a, b = normalize_title(title_a), normalize_title(title_b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
