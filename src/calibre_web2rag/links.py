from __future__ import annotations


def build_ebook_url(
    *,
    book_id: int,
    fmt: str,
    base_url: str | None,
    template: str | None,
) -> str | None:
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
