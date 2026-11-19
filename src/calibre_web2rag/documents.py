"""Pure functions that turn books and registry rows into text and payloads.

``point_id``, ``base_payload``, ``format_duration`` and ``catalog_text`` are
shared with the pipeline's audiobook source (audiobookshelf-rag-ingester),
which writes the same ``hs_catalog`` points from the same registry rows;
keep them identical there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from calibre_web2rag.models import BookRecord

TIER = "member"
# Shared architecture: 5 policy/sign, 4 wiki, 3 asset/geowiki/event,
# 2 manual/repo/esphome, 1 slack fact. Library records sit with manuals.
AUTHORITY = 2
_BLANK_LINES = re.compile(r"\n\s*\n+")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLOCKS = ["p", "div", "li", "blockquote", "tr", "h1", "h2", "h3", "h4", "h5", "h6"]


def point_id(source: str, doc_id: str, chunk_index: int) -> str:
    """uuid5(source:doc_id:chunk_index): stable, so a re-sync overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_id}:{chunk_index}"))


def text_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# Payload fields that change on every run without the point changing.
_VOLATILE = frozenset({"indexed_at", "sync_run_id", "payload_hash"})


# Any: payload values are JSON.
def payload_hash(payload: dict[str, Any]) -> str:
    """Hash of everything in a payload but the per-run fields."""
    return json_hash({k: v for k, v in payload.items() if k not in _VOLATILE})


def html_to_text(html: str | None) -> str:
    """Calibre's comments field (HTML) as plain text with paragraph breaks."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(_BLOCKS):
        block.insert_after("\n\n")
    text = soup.get_text("")
    paragraphs = [_SPACES.sub(" ", p).strip() for p in _BLANK_LINES.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def to_rfc3339(value: str | None) -> str | None:
    """Calibre's "2024-01-02 03:04:05.678+00:00" as an RFC 3339 UTC timestamp."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    # Calibre stores "0101-01-01" for an unknown date.
    if parsed.year < 1000:
        return None
    return parsed.astimezone(UTC).isoformat()


def year_of(value: str | None) -> int | None:
    stamp = to_rfc3339(value)
    return int(stamp[:4]) if stamp else None


def now_rfc3339() -> str:
    return datetime.now(UTC).isoformat()


# Any: a JSON-ready dict of the record.
def book_raw(book: BookRecord) -> dict[str, Any]:
    """The book as stored in staging: every field, paths as strings."""

    # Any: dataclasses.asdict values.
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    raw: dict[str, Any] = convert(dataclasses.asdict(book))
    raw["cover_path"] = book.cover_path
    return raw


# Any: JSON-ready values.
def json_hash(value: Any) -> str:
    return text_hash(json.dumps(value, sort_keys=True, default=str))


def format_duration(seconds: float | None) -> str:
    """'15 hr 30 min' or '42 min'; '' for no length."""
    if not seconds or seconds <= 0:
        return ""
    hours, minutes = divmod(round(seconds / 60), 60)
    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    return f"{max(minutes, 1)} min"


def _join(label: str, values: Iterable[str]) -> str | None:
    items = [v for v in values if v]
    return f"{label}: {', '.join(items)}" if items else None


# Any: registry rows and manifestation details are JSON.
def catalog_text(
    work: dict[str, Any], manifestations: list[dict[str, Any]], series: str | None
) -> str:
    """What a work's catalog point embeds: title, authors, series, formats, narrators,
    length, topics, description, contents and audiobook chapters."""
    details = [m.get("details") or {} for m in manifestations]
    # Prefer the library's own description; use the machine summary when the
    # description is missing or is marketing copy.
    description = ""
    for d in details:
        if d.get("description") and not d.get("description_is_marketing"):
            description = str(d["description"])
            break
    good_for = next((str(d["good_for"]) for d in details if d.get("good_for")), "")
    if not description:
        description = next(
            (str(d["enriched_description"]) for d in details if d.get("enriched_description")),
            "",
        )
    toc: list[str] = next((list(d["toc"]) for d in details if d.get("toc")), [])
    chapters: list[str] = next((list(d["chapters"]) for d in details if d.get("chapters")), [])
    subtitle = next((str(d["subtitle"]) for d in details if d.get("subtitle")), "")
    tags = sorted({t for d in details for t in d.get("tags") or []})
    subjects = sorted({s for d in details for s in d.get("enriched_subjects") or []})
    narrators = sorted({n for d in details for n in d.get("narrators") or []})
    length = next((d["duration_seconds"] for d in details if d.get("duration_seconds")), None)
    formats = sorted({str(m["format"]) for m in manifestations})

    title = str(work["title"]) + (
        f": {subtitle}" if subtitle and subtitle not in work["title"] else ""
    )
    lines = [title]
    if work.get("authors"):
        lines.append(f"by {', '.join(work['authors'])}")
    if series:
        index = work.get("series_index")
        number = f" #{index:g}" if isinstance(index, int | float) else ""
        lines.append(f"Series: {series}{number}")
    for line in (
        _join("Formats in the library", formats),
        _join("Narrated by", narrators),
        f"Audiobook length: {format_duration(length)}" if length else None,
        _join("Topics", work.get("topics") or []),
        _join("Tags", tags),
        _join("Subjects", subjects),
    ):
        if line:
            lines.append(line)
    if description:
        lines.extend(["", description])
    if good_for and good_for != description:
        lines.extend(["", f"What this book is good for (machine-generated summary): {good_for}"])
    if toc:
        lines.extend(["", "Contents:", *(str(t) for t in toc)])
    if chapters:
        lines.extend(["", "Audiobook chapters:", *(str(c) for c in chapters)])
    return "\n".join(lines)


# Any: catalog rows from Postgres.
def overview_sections(
    rows: list[dict[str, Any]],
    *,
    library_url: str | None,
    request_info: str | None,
) -> list[tuple[str, str]]:
    """(heading, body) sections of the regenerated library overview document."""
    ebooks = [r for r in rows if r.get("format") == "ebook"]
    authors = {a for r in ebooks for a in r.get("authors") or []}
    topics: Counter[str] = Counter(t for r in ebooks for t in r.get("topics") or [])
    tags: Counter[str] = Counter(t for r in ebooks for t in r.get("tags") or [])
    formats: Counter[str] = Counter(f for r in ebooks for f in r.get("file_formats") or [])
    languages: Counter[str] = Counter(lang for r in ebooks for lang in r.get("languages") or [])

    size = len(ebooks)
    approx = f"about {round(size, -1):,}" if size >= 100 else f"{size}"
    intro = (
        f"PDX Hackerspace members have access to an ebook library of {approx} books by "
        f"{len(authors):,} authors. It is a Calibre library, served to members through "
        "Calibre-web, where they can browse, search, read and download the books."
    )
    sections = [("The PDX Hackerspace ebook library", intro)]
    contents: list[str] = []
    if topics:
        contents.append(
            "Its strongest topics are "
            + ", ".join(f"{t} ({n})" for t, n in topics.most_common(12))
            + "."
        )
    if tags:
        contents.append(
            "Common tags: " + ", ".join(f"{t} ({n})" for t, n in tags.most_common(15)) + "."
        )
    if formats:
        contents.append(
            "Formats: " + ", ".join(f"{f} ({n})" for f, n in formats.most_common()) + "."
        )
    if languages:
        contents.append(
            "Languages: " + ", ".join(f"{lang} ({n})" for lang, n in languages.most_common(8)) + "."
        )
    contents.append(
        "For exact counts, lists and recent additions, ask the catalog; it is answered "
        "from the live catalog, not from this summary."
    )
    sections.append(("What is in the ebook library", " ".join(contents)))
    reach = (
        f"Members open the library at {library_url}."
        if library_url
        else "Members reach the library through its Calibre-web site; ask a library "
        "maintainer for the address."
    )
    sections.append(("How to reach the ebook library", reach))
    sections.append(
        (
            "How to request additions to the ebook library",
            request_info
            or "To suggest a book for the library, ask the library maintainers in the "
            "hackerspace's usual channels.",
        )
    )
    return sections


# Any: payload values are JSON.
def base_payload(
    *,
    source: str,
    doc_id: str,
    chunk_index: int,
    title: str,
    text: str,
    context: str,
    url: str | None,
    run_id: str,
    created_at: str | None,
    updated_at: str | None,
    heading_path: list[str] | None = None,
    page: int | None = None,
    lang: str | None = None,
    topics: list[str] | None = None,
    work_ids: list[str] | None = None,
    links: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The payload every text collection shares (see the shared architecture plan)."""
    links = links or {}
    return {
        "source": source,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "title": title,
        "heading_path": heading_path or [],
        "text": text,
        "context": context,
        "url": url,
        "tier": TIER,
        "authority": AUTHORITY,
        "created_at": created_at,
        "updated_at": updated_at,
        "indexed_at": now_rfc3339(),
        "stale_after": None,
        "superseded_by": None,
        "content_hash": text_hash(context, text),
        "sync_run_id": run_id,
        "asset_ids": links.get("asset_ids", []),
        "location_ids": links.get("location_ids", []),
        "work_ids": sorted(set(work_ids or []) | set(links.get("work_ids", []))),
        "event_ids": links.get("event_ids", []),
        "training_ids": links.get("training_ids", []),
        "topics": topics or [],
        "section": None,
        "page": page,
        "lang": lang,
    }
