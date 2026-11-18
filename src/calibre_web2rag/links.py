"""Links members can open: the Calibre-web book page and per-format downloads."""

from __future__ import annotations


def build_book_url(*, book_id: int, base_url: str | None, template: str | None) -> str | None:
    """The Calibre-web page for a book (``<base>/book/<id>`` unless templated)."""
    if template:
        return template.format(book_id=book_id)
    if base_url:
        return f"{base_url.rstrip('/')}/book/{book_id}"
    return None


def build_ebook_url(
    *,
    book_id: int,
    fmt: str,
    base_url: str | None,
    template: str | None,
) -> str | None:
    """A download link for one format (``<base>/download/<id>/<fmt>`` unless templated)."""
    if template:
        return template.format(book_id=book_id, format=fmt.lower(), fmt=fmt.lower())
    if base_url:
        root = base_url.rstrip("/")
        return f"{root}/download/{book_id}/{fmt.lower()}"
    return None


def build_opds_url(*, book_id: int, base_url: str | None) -> str | None:
    if not base_url:
        return None
    root = base_url.rstrip("/")
    return f"{root}/opds/book/{book_id}"
