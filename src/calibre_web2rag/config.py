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
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


@dataclass(frozen=True)
class Settings:
    calibre_metadata_db: str
    calibre_library_root: str
    calibre_web_base_url: str | None
    calibre_download_url_template: str | None
    database_url: str
    qdrant_url: str
    qdrant_api_key: str | None
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
    calibre_web_book_url_template: str | None = None
    member_collection: str = "hs_member"
    catalog_collection: str = "hs_catalog"
    booktext_collection: str = "hs_booktext"
    booktext_enabled: bool = False
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: int = 120
    enrichment_enabled: bool = True
    enrichment_timeout_seconds: int = 20
    google_books_api_key: str | None = None
    topic_taxonomy_file: str | None = None
    library_request_info: str | None = None
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

        def boolean(name: str, default: bool) -> bool:
            text = (env.get(name) or "").strip().lower()
            if not text:
                return default
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ConfigError(f"{name} must be true or false, not {text!r}")

        def choice(name: str, default: str, allowed: frozenset[str]) -> str:
            value = (env.get(name) or default).lower()
            if value not in allowed:
                options = ", ".join(sorted(allowed))
                raise ConfigError(f"{name} must be one of {options}, not {value!r}")
            return value

        # About 450 tokens at ~4 characters per token (plan: 300-500 tokens).
        chunk_size = integer("CHUNK_SIZE", 1800, 1)
        chunk_overlap = integer("CHUNK_OVERLAP", 150, 0)
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
            )

        vector_size = integer("VECTOR_SIZE", 768, 1)
        # 0 or unset: the model's full size.
        embedding_dimensions = integer("EMBEDDING_DIMENSIONS", 0, 0) or None
        if embedding_dimensions is not None and embedding_dimensions != vector_size:
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS ({embedding_dimensions}) must equal "
                f"VECTOR_SIZE ({vector_size})"
            )

        log_level = (env.get("LOG_LEVEL") or "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        collections = [
            env.get(name) or default
            for name, default in (
                ("QDRANT_MEMBER_COLLECTION", "hs_member"),
                ("QDRANT_CATALOG_COLLECTION", "hs_catalog"),
                ("QDRANT_BOOKTEXT_COLLECTION", "hs_booktext"),
            )
        ]
        if len(set(collections)) != len(collections):
            raise ConfigError(
                "QDRANT_MEMBER_COLLECTION, QDRANT_CATALOG_COLLECTION and "
                "QDRANT_BOOKTEXT_COLLECTION must be three different collections"
            )
        if "hs_public" in collections:
            raise ConfigError("This ingester is member tier only; it never writes to hs_public")

        return cls(
            calibre_metadata_db=required("CALIBRE_METADATA_DB"),
            calibre_library_root=required("CALIBRE_LIBRARY_ROOT"),
            calibre_web_base_url=optional("CALIBRE_WEB_BASE_URL"),
            calibre_download_url_template=optional("CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE"),
            calibre_web_book_url_template=optional("CALIBRE_WEB_BOOK_URL_TEMPLATE"),
            database_url=required("DATABASE_URL"),
            qdrant_url=env.get("QDRANT_URL") or "http://qdrant:6333",
            qdrant_api_key=optional("QDRANT_API_KEY"),
            member_collection=env.get("QDRANT_MEMBER_COLLECTION") or "hs_member",
            catalog_collection=env.get("QDRANT_CATALOG_COLLECTION") or "hs_catalog",
            booktext_collection=env.get("QDRANT_BOOKTEXT_COLLECTION") or "hs_booktext",
            booktext_enabled=boolean("BOOKTEXT_ENABLED", False),
            llm_model=optional("LLM_MODEL"),
            llm_base_url=optional("LLM_BASE_URL"),
            llm_api_key=optional("LLM_API_KEY"),
            llm_timeout_seconds=integer("LLM_TIMEOUT_SECONDS", 120, 1),
            enrichment_enabled=boolean("ENRICHMENT_ENABLED", True),
            enrichment_timeout_seconds=integer("ENRICHMENT_TIMEOUT_SECONDS", 20, 1),
            google_books_api_key=optional("GOOGLE_BOOKS_API_KEY"),
            topic_taxonomy_file=optional("TOPIC_TAXONOMY_FILE"),
            library_request_info=optional("LIBRARY_REQUEST_INFO"),
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
            vector_size=vector_size,
            # 0 disables truncation.
            embedding_context_length=integer("EMBEDDING_CONTEXT_LENGTH", 8192, 0),
            embedding_dimensions=embedding_dimensions,
            # Kept exactly as given: the trailing space usually matters.
            embedding_query_prefix=env.get("EMBEDDING_QUERY_PREFIX", ""),
            embedding_document_prefix=env.get("EMBEDDING_DOCUMENT_PREFIX", ""),
            log_level=log_level,
        )

    @property
    def gateway_base_url(self) -> str:
        """Where generation requests go: LLM_BASE_URL, else the embeddings server."""
        return self.llm_base_url or self.openai_base_url

    @property
    def gateway_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key
