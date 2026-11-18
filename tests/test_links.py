from calibre_web2rag.links import build_book_url, build_ebook_url, build_opds_url


def test_build_url_from_template() -> None:
    url = build_ebook_url(
        book_id=12,
        fmt="PDF",
        base_url=None,
        template="https://books.example/download/{book_id}/{fmt}",
    )
    assert url == "https://books.example/download/12/pdf"


def test_build_url_from_base_url() -> None:
    url = build_ebook_url(
        book_id=12,
        fmt="EPUB",
        base_url="https://books.example/",
        template=None,
    )
    assert url == "https://books.example/download/12/epub"


def test_build_opds_url() -> None:
    assert (
        build_opds_url(book_id=9, base_url="https://books.example")
        == "https://books.example/opds/book/9"
    )


def test_build_book_url() -> None:
    assert build_book_url(book_id=3, base_url="https://b.example/", template=None) == (
        "https://b.example/book/3"
    )
    assert build_book_url(book_id=3, base_url=None, template="https://x/{book_id}") == "https://x/3"
    assert build_book_url(book_id=3, base_url=None, template=None) is None
    assert build_opds_url(book_id=3, base_url=None) is None
    assert build_ebook_url(book_id=3, fmt="PDF", base_url=None, template=None) is None
