from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    _truncate,
    build_embedder,
)


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}


class _FakeClient:
    def __init__(self, base_url: str, timeout: int) -> None:  # noqa: ARG002
        pass

    def post(self, path: str, json: dict) -> _FakeResponse:  # noqa: A002
        assert path == "/api/embed"
        assert json["model"] == "nomic-embed-text"
        return _FakeResponse()


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder

    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[list[float]]:  # noqa: ARG002
        return [[0.5, 0.6] for _ in chunks]


def test_ollama_embedder_calls_embed_api(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    embedder = OllamaEmbedder(
        model_name="nomic-embed-text",
        ollama_url="http://ollama:11434",
        timeout_seconds=30,
    )
    vectors = embedder.encode(["a", "b"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]


def test_build_embedder_uses_ollama_provider(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    settings = Settings(
        calibre_metadata_db="db.sqlite",
        calibre_library_root="/tmp",
        calibre_web_base_url=None,
        calibre_download_url_template=None,
        qdrant_url="http://qdrant:6333",
        qdrant_api_key=None,
        qdrant_collection="books",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        hf_cache_dir=None,
        ollama_url="http://ollama:11434",
        ollama_timeout_seconds=30,
        chunk_size=1000,
        chunk_overlap=100,
        batch_size=10,
        distance="cosine",
        vector_size=2,
        embedding_context_length=8192,
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.max_tokens == 8192


def test_build_embedder_uses_st_cache_directory(monkeypatch) -> None:
    monkeypatch.setattr(
        "calibre_web2rag.embeddings._load_sentence_transformer",
        lambda: _FakeSentenceTransformer,
    )
    settings = Settings(
        calibre_metadata_db="db.sqlite",
        calibre_library_root="/tmp",
        calibre_web_base_url=None,
        calibre_download_url_template=None,
        qdrant_url="http://qdrant:6333",
        qdrant_api_key=None,
        qdrant_collection="books",
        embedding_provider="sentence_transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        hf_cache_dir="/tmp/hf-cache",
        ollama_url="http://ollama:11434",
        ollama_timeout_seconds=30,
        chunk_size=1000,
        chunk_overlap=100,
        batch_size=10,
        distance="cosine",
        vector_size=2,
        embedding_context_length=8192,
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder._model.cache_folder == "/tmp/hf-cache"


def test_truncate_shortens_long_texts() -> None:
    short = "hello"
    long_text = "x" * 100
    result = _truncate([short, long_text], max_tokens=10)
    assert result[0] == short
    assert len(result[1]) == 10 * 4  # _CHARS_PER_TOKEN = 4


def test_truncate_noop_when_disabled() -> None:
    long_text = "x" * 100
    result = _truncate([long_text], max_tokens=0)
    assert result[0] == long_text
