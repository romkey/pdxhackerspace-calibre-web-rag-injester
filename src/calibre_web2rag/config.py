"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

EMBEDDING_PROVIDERS = frozenset({"openai", "sentence_transformers", "sentence-transformers", "st"})
DISTANCES = frozenset({"cosine", "dot", "euclid"})


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


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
    openai_base_url: str
    openai_api_key: str | None
    openai_timeout_seconds: int
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    distance: str
    vector_size: int
    embedding_context_length: int
    embedding_dimensions: int | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def required(name: str) -> str:
            value = env.get(name)
            if not value:
                raise ConfigError(f"Missing required environment variable: {name}")
            return value

        def optional(name: str) -> str | None:
            # An empty value (NAME= in .env) means unset.
            return env.get(name) or None

        def integer(name: str, default: int, minimum: int) -> int:
            text = env.get(name) or str(default)
            try:
                value = int(text)
            except ValueError:
                raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
            if value < minimum:
                raise ConfigError(f"{name} must be at least {minimum}, not {value}")
            return value

        def choice(name: str, default: str, allowed: frozenset[str]) -> str:
            value = (env.get(name) or default).lower()
            if value not in allowed:
                options = ", ".join(sorted(allowed))
                raise ConfigError(f"{name} must be one of {options}, not {value!r}")
            return value

        chunk_size = integer("CHUNK_SIZE", 1200, 1)
        chunk_overlap = integer("CHUNK_OVERLAP", 120, 0)
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
            )

        log_level = (env.get("LOG_LEVEL") or "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        return cls(
            calibre_metadata_db=required("CALIBRE_METADATA_DB"),
            calibre_library_root=required("CALIBRE_LIBRARY_ROOT"),
            calibre_web_base_url=optional("CALIBRE_WEB_BASE_URL"),
            calibre_download_url_template=optional("CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE"),
            qdrant_url=env.get("QDRANT_URL") or "http://qdrant:6333",
            qdrant_api_key=optional("QDRANT_API_KEY"),
            qdrant_collection=env.get("QDRANT_COLLECTION") or "calibre_books",
            embedding_provider=choice("EMBEDDING_PROVIDER", "openai", EMBEDDING_PROVIDERS),
            embedding_model=env.get("EMBEDDING_MODEL") or "nomic-ai/modernbert-embed-base",
            hf_cache_dir=(
                optional("HF_CACHE_DIR")
                or optional("SENTENCE_TRANSFORMERS_HOME")
                or optional("HF_HOME")
            ),
            openai_base_url=env.get("OPENAI_BASE_URL") or "http://localhost:8080/v1",
            openai_api_key=optional("OPENAI_API_KEY"),
            openai_timeout_seconds=integer("OPENAI_TIMEOUT_SECONDS", 120, 1),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=integer("BATCH_SIZE", 64, 1),
            distance=choice("QDRANT_DISTANCE", "cosine", DISTANCES),
            vector_size=integer("VECTOR_SIZE", 768, 1),
            # 0 disables truncation.
            embedding_context_length=integer("EMBEDDING_CONTEXT_LENGTH", 8192, 0),
            # 0 or unset: the model's full size.
            embedding_dimensions=integer("EMBEDDING_DIMENSIONS", 0, 0) or None,
            # Kept exactly as given: the trailing space usually matters.
            embedding_query_prefix=env.get("EMBEDDING_QUERY_PREFIX", ""),
            embedding_document_prefix=env.get("EMBEDDING_DOCUMENT_PREFIX", ""),
            log_level=log_level,
        )
