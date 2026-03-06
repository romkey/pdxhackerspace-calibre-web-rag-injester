from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    calibre_metadata_db: str
    calibre_library_root: str
    calibre_web_base_url: str | None
    calibre_download_url_template: str | None
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_provider: str
    embedding_model: str
    hf_cache_dir: str | None
    ollama_url: str
    ollama_timeout_seconds: int
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    distance: str
    vector_size: int
    embedding_context_length: int


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        calibre_metadata_db=_get_required("CALIBRE_METADATA_DB"),
        calibre_library_root=_get_required("CALIBRE_LIBRARY_ROOT"),
        calibre_web_base_url=os.getenv("CALIBRE_WEB_BASE_URL"),
        calibre_download_url_template=os.getenv("CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE"),
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "calibre_books"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        hf_cache_dir=(
            os.getenv("HF_CACHE_DIR")
            or os.getenv("SENTENCE_TRANSFORMERS_HOME")
            or os.getenv("HF_HOME")
        ),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        batch_size=int(os.getenv("BATCH_SIZE", "64")),
        distance=os.getenv("QDRANT_DISTANCE", "cosine").lower(),
        vector_size=int(os.getenv("VECTOR_SIZE", "768")),
        embedding_context_length=int(
            os.getenv("EMBEDDING_CONTEXT_LENGTH", "8192")
        ),
    )
