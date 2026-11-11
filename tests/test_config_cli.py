from __future__ import annotations

import pytest

from calibre_web2rag import cli, config

_ENV_VARS = [
    "CALIBRE_METADATA_DB",
    "CALIBRE_LIBRARY_ROOT",
    "CALIBRE_WEB_BASE_URL",
    "CALIBRE_WEB_DOWNLOAD_URL_TEMPLATE",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "HF_CACHE_DIR",
    "SENTENCE_TRANSFORMERS_HOME",
    "HF_HOME",
    "OLLAMA_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_TIMEOUT_SECONDS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "BATCH_SIZE",
    "QDRANT_DISTANCE",
    "VECTOR_SIZE",
    "EMBEDDING_CONTEXT_LENGTH",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Never read a developer's real .env file during tests.
    monkeypatch.setattr(config, "load_dotenv", lambda: None)


def test_load_settings_requires_metadata_db(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_LIBRARY_ROOT", "/lib")
    with pytest.raises(ValueError, match="CALIBRE_METADATA_DB"):
        config.load_settings()


def test_load_settings_requires_library_root(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_METADATA_DB", "/lib/metadata.db")
    with pytest.raises(ValueError, match="CALIBRE_LIBRARY_ROOT"):
        config.load_settings()


def test_load_settings_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_METADATA_DB", "/lib/metadata.db")
    monkeypatch.setenv("CALIBRE_LIBRARY_ROOT", "/lib")
    settings = config.load_settings()
    assert settings.calibre_metadata_db == "/lib/metadata.db"
    assert settings.calibre_library_root == "/lib"
    assert settings.calibre_web_base_url is None
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_collection == "calibre_books"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.hf_cache_dir is None
    assert settings.ollama_timeout_seconds == 120
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 120
    assert settings.batch_size == 64
    assert settings.distance == "cosine"
    assert settings.vector_size == 768
    assert settings.embedding_context_length == 8192


def test_load_settings_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_METADATA_DB", "/lib/metadata.db")
    monkeypatch.setenv("CALIBRE_LIBRARY_ROOT", "/lib")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "Sentence_Transformers")
    monkeypatch.setenv("QDRANT_DISTANCE", "DOT")
    monkeypatch.setenv("HF_HOME", "/hf-home")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("VECTOR_SIZE", "384")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    settings = config.load_settings()
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.distance == "dot"
    assert settings.hf_cache_dir == "/hf-home"
    assert settings.chunk_size == 500
    assert settings.vector_size == 384
    assert settings.ollama_api_key == "secret"


def test_hf_cache_dir_precedence(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_METADATA_DB", "/lib/metadata.db")
    monkeypatch.setenv("CALIBRE_LIBRARY_ROOT", "/lib")
    monkeypatch.setenv("HF_HOME", "/hf-home")
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/st-home")
    monkeypatch.setenv("HF_CACHE_DIR", "/hf-cache")
    assert config.load_settings().hf_cache_dir == "/hf-cache"


def test_cli_main_success(monkeypatch, caplog) -> None:
    sentinel = object()
    monkeypatch.setattr(cli, "load_settings", lambda: sentinel)

    def _ingest(settings):
        assert settings is sentinel
        return 42

    monkeypatch.setattr(cli, "ingest", _ingest)
    with caplog.at_level("INFO"):
        cli.main()
    assert "Upserted 42 chunks" in caplog.text


def test_cli_main_failure_exits_nonzero(monkeypatch) -> None:
    def _boom():
        raise ValueError("Missing required environment variable: X")

    monkeypatch.setattr(cli, "load_settings", _boom)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
