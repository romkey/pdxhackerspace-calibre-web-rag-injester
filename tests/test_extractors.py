from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from calibre_web2rag.extractors import is_epub_drm_free, is_mobi_drm_free


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
