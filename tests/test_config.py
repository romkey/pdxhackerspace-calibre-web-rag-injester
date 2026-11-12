from __future__ import annotations

import pytest

from calibre_web2rag.config import ConfigError, Settings

REQUIRED = {"CALIBRE_METADATA_DB": "/lib/metadata.db", "CALIBRE_LIBRARY_ROOT": "/lib"}


def test_defaults() -> None:
    settings = Settings.from_env(REQUIRED)
    assert settings.calibre_metadata_db == "/lib/metadata.db"
    assert settings.calibre_library_root == "/lib"
    assert settings.calibre_web_base_url is None
    assert settings.calibre_download_url_template is None
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_api_key is None
    assert settings.qdrant_collection == "calibre_books"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "nomic-ai/modernbert-embed-base"
    assert settings.hf_cache_dir is None
    assert settings.openai_base_url == "http://localhost:8080/v1"
    assert settings.openai_api_key is None
    assert settings.openai_timeout_seconds == 120
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 120
    assert settings.batch_size == 64
    assert settings.distance == "cosine"
    assert settings.vector_size == 768
    assert settings.embedding_dimensions is None
    assert settings.embedding_query_prefix == ""
    assert settings.embedding_document_prefix == ""
    assert settings.embedding_context_length == 8192
    assert settings.log_level == "INFO"


def test_overrides() -> None:
    settings = Settings.from_env(
        {
            **REQUIRED,
            "CALIBRE_WEB_BASE_URL": "https://books.example",
            "EMBEDDING_PROVIDER": "Sentence_Transformers",
            "QDRANT_DISTANCE": "DOT",
            "HF_HOME": "/hf-home",
            "CHUNK_SIZE": "500",
            "CHUNK_OVERLAP": "0",
            "VECTOR_SIZE": "384",
            "EMBEDDING_CONTEXT_LENGTH": "0",
            "OPENAI_API_KEY": "dummy-key",
            "EMBEDDING_DIMENSIONS": "384",  # matches VECTOR_SIZE
            "EMBEDDING_QUERY_PREFIX": "query: ",
            "EMBEDDING_DOCUMENT_PREFIX": "passage: ",
            "LOG_LEVEL": "debug",
        }
    )
    assert settings.calibre_web_base_url == "https://books.example"
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.distance == "dot"
    assert settings.hf_cache_dir == "/hf-home"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 0
    assert settings.vector_size == 384
    assert settings.embedding_context_length == 0
    assert settings.openai_api_key == "dummy-key"
    assert settings.embedding_dimensions == 384
    assert settings.embedding_query_prefix == "query: "
    assert settings.embedding_document_prefix == "passage: "
    assert settings.log_level == "DEBUG"


def test_empty_values_mean_unset() -> None:
    settings = Settings.from_env({**REQUIRED, "QDRANT_API_KEY": "", "CHUNK_SIZE": ""})
    assert settings.qdrant_api_key is None
    assert settings.chunk_size == 1200


def test_hf_cache_dir_precedence() -> None:
    env = {
        **REQUIRED,
        "HF_HOME": "/hf-home",
        "SENTENCE_TRANSFORMERS_HOME": "/st-home",
        "HF_CACHE_DIR": "/hf-cache",
    }
    assert Settings.from_env(env).hf_cache_dir == "/hf-cache"
    del env["HF_CACHE_DIR"]
    assert Settings.from_env(env).hf_cache_dir == "/st-home"


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_required_variables(missing: str) -> None:
    env = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CHUNK_SIZE", "big", "CHUNK_SIZE must be a whole number"),
        ("CHUNK_SIZE", "0", "CHUNK_SIZE must be at least 1"),
        ("BATCH_SIZE", "-1", "BATCH_SIZE must be at least 1"),
        ("VECTOR_SIZE", "1.5", "VECTOR_SIZE must be a whole number"),
        ("OPENAI_TIMEOUT_SECONDS", "0", "OPENAI_TIMEOUT_SECONDS must be at least 1"),
        ("EMBEDDING_DIMENSIONS", "-1", "EMBEDDING_DIMENSIONS must be at least 0"),
        (
            "EMBEDDING_DIMENSIONS",
            "512",
            r"EMBEDDING_DIMENSIONS \(512\) must equal VECTOR_SIZE \(768\)",
        ),
        ("EMBEDDING_PROVIDER", "ollama", "EMBEDDING_PROVIDER must be one of"),
        ("CHUNK_OVERLAP", "1200", "CHUNK_OVERLAP .* must be smaller than CHUNK_SIZE"),
        ("QDRANT_DISTANCE", "manhattan", "QDRANT_DISTANCE must be one of"),
        ("LOG_LEVEL", "LOUD", "LOG_LEVEL 'LOUD' is not a logging level"),
    ],
)
def test_invalid_values_name_the_variable(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({**REQUIRED, name: value})
