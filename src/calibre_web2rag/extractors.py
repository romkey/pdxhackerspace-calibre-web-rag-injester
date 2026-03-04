from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import mobi
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader

from calibre_web2rag.models import TextSection


def is_pdf_drm_free(path: Path) -> bool:
    try:
        reader = PdfReader(str(path))
        return not reader.is_encrypted
    except Exception:
        return False


def is_epub_drm_free(path: Path) -> bool:
    try:
        with ZipFile(path, "r") as archive:
            names = {name.lower() for name in archive.namelist()}
            # Most DRM-protected EPUB files include this manifest entry.
            if "meta-inf/encryption.xml" in names:
                data = archive.read("META-INF/encryption.xml")
                return len(data.strip()) == 0
        return True
    except Exception:
        return False


def is_mobi_drm_free(path: Path) -> bool:
    try:
        with path.open("rb") as fp:
            header = fp.read(16)
            if len(header) < 14:
                return False
            encryption_type = int.from_bytes(header[12:14], byteorder="big", signed=False)
            return encryption_type == 0
    except Exception:
        return False


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_epub_text(path: Path) -> str:
    book = epub.read_epub(str(path))
    blocks: list[str] = []
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def extract_mobi_text(path: Path) -> str:
    tempdir, extracted = mobi.extract(str(path))
    try:
        extracted_path = Path(tempdir) / extracted
        data = extracted_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(data, "html.parser")
        return soup.get_text(" ", strip=True)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def extract_pdf_sections(path: Path) -> list[TextSection]:
    text = extract_pdf_text(path)
    if not text.strip():
        return []
    return [TextSection(title=None, text=text)]


def extract_epub_sections(path: Path) -> list[TextSection]:
    book = epub.read_epub(str(path))
    sections: list[TextSection] = []
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        title = None
        for tag in ("h1", "h2", "h3"):
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                break
        text = soup.get_text(" ", strip=True)
        if text:
            sections.append(TextSection(title=title, text=text))
    return sections


def extract_mobi_sections(path: Path) -> list[TextSection]:
    tempdir, extracted = mobi.extract(str(path))
    try:
        extracted_path = Path(tempdir) / extracted
        data = extracted_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(data, "html.parser")
        text = soup.get_text(" ", strip=True)
        if not text:
            return []
        return [TextSection(title=None, text=text)]
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Combined DRM-check + extraction (single-pass per format)
# ---------------------------------------------------------------------------


def try_extract_pdf_sections(path: Path) -> list[TextSection]:
    """Check DRM and extract in one pass, creating PdfReader only once."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return []
        text = "\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
        if not text.strip():
            return []
        return [TextSection(title=None, text=text)]
    except Exception:
        return []


def try_extract_epub_sections(path: Path) -> list[TextSection]:
    """Check DRM via ZipFile, then extract sections if clean."""
    try:
        with ZipFile(path, "r") as archive:
            names = {name.lower() for name in archive.namelist()}
            if "meta-inf/encryption.xml" in names:
                data = archive.read("META-INF/encryption.xml")
                if len(data.strip()) > 0:
                    return []
    except Exception:
        return []

    try:
        book = epub.read_epub(str(path))
        sections: list[TextSection] = []
        for item in book.get_items():
            if item.get_type() != ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            title = None
            for tag in ("h1", "h2", "h3"):
                heading = soup.find(tag)
                if heading:
                    title = heading.get_text(strip=True)
                    break
            text = soup.get_text(" ", strip=True)
            if text:
                sections.append(TextSection(title=title, text=text))
        return sections
    except Exception:
        return []


def try_extract_mobi_sections(path: Path) -> list[TextSection]:
    """Attempt extraction directly instead of relying on the fragile
    16-byte header DRM check, which is unreliable for KFX-format files."""
    try:
        tempdir, extracted = mobi.extract(str(path))
        try:
            extracted_path = Path(tempdir) / extracted
            data = extracted_path.read_text(
                encoding="utf-8", errors="ignore"
            )
            soup = BeautifulSoup(data, "html.parser")
            text = soup.get_text(" ", strip=True)
            if not text:
                return []
            return [TextSection(title=None, text=text)]
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)
    except Exception:
        return []
