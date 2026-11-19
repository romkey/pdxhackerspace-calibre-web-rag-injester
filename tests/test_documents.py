from calibre_web2rag.documents import (
    catalog_text,
    format_duration,
    html_to_text,
    overview_sections,
    to_rfc3339,
    year_of,
)


def test_html_to_text_keeps_paragraphs() -> None:
    assert html_to_text("<p>One  two</p><p>Three<br>four</p>") == "One two\n\nThree\nfour"
    assert html_to_text(None) == ""


def test_timestamps() -> None:
    assert to_rfc3339("2024-01-02 03:04:05.678+00:00") == "2024-01-02T03:04:05.678000+00:00"
    assert to_rfc3339("2024-01-02") == "2024-01-02T00:00:00+00:00"
    assert to_rfc3339("0101-01-01 00:00:00+00:00") is None
    assert to_rfc3339("garbage") is None
    assert to_rfc3339(None) is None
    assert year_of("1999-05-01 00:00:00+00:00") == 1999
    assert year_of(None) is None


def test_catalog_text_prefers_library_description() -> None:
    work = {"title": "Radio", "authors": ["A"], "topics": ["antennas"], "series_index": 2.0}
    manifestations = [
        {
            "format": "ebook",
            "details": {
                "description": "Real description.",
                "good_for": "Summary.",
                "toc": ["Ch 1"],
                "subtitle": "A Guide",
                "tags": ["t"],
                "enriched_subjects": ["s"],
            },
        },
        {"format": "hardcopy", "details": {}},
    ]
    text = catalog_text(work, manifestations, "Series")
    assert text.startswith("Radio: A Guide\nby A\nSeries: Series #2")
    assert "Formats in the library: ebook, hardcopy" in text
    assert "Real description." in text
    assert "machine-generated summary): Summary." in text
    assert text.endswith("Contents:\nCh 1")


def test_catalog_text_falls_back_to_enriched_description() -> None:
    work = {"title": "T", "authors": []}
    manifestations = [{"format": "ebook", "details": {"enriched_description": "From OL."}}]
    assert "From OL." in catalog_text(work, manifestations, None)


def test_overview_sections() -> None:
    rows = [
        {
            "format": "ebook",
            "authors": ["A", "B"],
            "topics": ["antennas"],
            "tags": ["RF"],
            "file_formats": ["EPUB"],
            "languages": ["eng"],
        }
        for _ in range(150)
    ] + [{"format": "audiobook"}]
    sections = dict(overview_sections(rows, library_url=None, request_info="Ask Pat."))
    intro = sections["The PDX Hackerspace ebook library"]
    assert "about 150 books by 2 authors" in intro
    assert "antennas (150)" in sections["What is in the ebook library"]
    assert "ask a library maintainer" in sections["How to reach the ebook library"]
    assert sections["How to request additions to the ebook library"] == "Ask Pat."
    empty = dict(overview_sections([], library_url="https://b", request_info=None))
    assert "library of 0 books" in empty["The PDX Hackerspace ebook library"]
    assert empty["How to reach the ebook library"] == "Members open the library at https://b."


def test_catalog_text_renders_audiobook_manifestations() -> None:
    # The audiobook source writes these details; both sources render the
    # same hs_catalog point, so the text must not depend on who wrote it.
    work = {"title": "Dune", "authors": ["Frank Herbert"]}
    manifestations = [
        {
            "format": "audiobook",
            "details": {
                "narrators": ["Scott Brick", "Orlagh Cassidy"],
                "duration_seconds": 75600,
                "chapters": ["Prologue", "Arrakis"],
            },
        },
        {"format": "ebook", "details": {"description": "Spice.", "toc": ["Book One"]}},
    ]
    text = catalog_text(work, manifestations, None)
    assert "Formats in the library: audiobook, ebook" in text
    assert "Narrated by: Orlagh Cassidy, Scott Brick" in text
    assert "Audiobook length: 21 hr" in text
    assert text.endswith("Contents:\nBook One\n\nAudiobook chapters:\nPrologue\nArrakis")


def test_format_duration() -> None:
    assert format_duration(None) == ""
    assert format_duration(30) == "1 min"
    assert format_duration(3600) == "1 hr"
    assert format_duration(3600 * 15.5) == "15 hr 30 min"
