import pytest

from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import (
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    _truncate,
    build_embedder,
)


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        # Deliberately out of order to check sorting by index.
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
        }


class _FakeClient:
    headers: dict | None = None
    base_url: str | None = None
    last_json: dict | None = None

    def __init__(
        self,
        base_url: str,
        headers: dict | None,
        timeout: int,  # noqa: ARG002
    ) -> None:
        self.__class__.headers = headers
        self.__class__.base_url = base_url

    def post(self, path: str, json: dict) -> _FakeResponse:  # noqa: A002
        assert path == "/embeddings"
        assert json["model"] == "text-embedding-3-small"
        self.__class__.last_json = json
        return _FakeResponse()


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder

    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[list[float]]:  # noqa: ARG002
        return [[0.5, 0.6] for _ in chunks]


def test_openai_embedder_calls_embeddings_api(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url="https://api.openai.com/v1/",
        api_key=None,
        timeout_seconds=30,
    )
    vectors = embedder.encode(["a", "b"])
    assert _FakeClient.base_url == "https://api.openai.com/v1"
    assert _FakeClient.headers is None
    assert _FakeClient.last_json["input"] == ["a", "b"]
    assert "dimensions" not in _FakeClient.last_json
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_openai_embedder_sends_dimensions(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key=None,
        timeout_seconds=30,
        dimensions=256,
    )
    embedder.encode(["a", "b"])
    assert _FakeClient.last_json["dimensions"] == 256


def test_openai_embedder_rejects_bad_response(monkeypatch) -> None:
    class _BadResponse(_FakeResponse):
        def json(self) -> dict:
            return {"error": "nope"}

    class _BadClient(_FakeClient):
        def post(self, path: str, json: dict) -> _FakeResponse:  # noqa: A002, ARG002
            return _BadResponse()

    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _BadClient)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key=None,
        timeout_seconds=30,
    )
    with pytest.raises(ValueError):
        embedder.encode(["a", "b"])


def test_openai_embedder_empty_input_skips_request(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key=None,
        timeout_seconds=30,
    )
    assert embedder.encode([]) == []


def test_build_embedder_uses_openai_provider(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    settings = Settings(
        calibre_metadata_db="db.sqlite",
        calibre_library_root="/tmp",
        calibre_web_base_url=None,
        calibre_download_url_template=None,
        qdrant_url="http://qdrant:6333",
        qdrant_api_key=None,
        qdrant_collection="books",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        hf_cache_dir=None,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=None,
        openai_timeout_seconds=30,
        chunk_size=1000,
        chunk_overlap=100,
        batch_size=10,
        distance="cosine",
        vector_size=2,
        embedding_context_length=8192,
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAIEmbedder)
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
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=None,
        openai_timeout_seconds=30,
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


def test_openai_embedder_uses_api_key_header(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.embeddings.httpx.Client", _FakeClient)
    OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="secret",
        timeout_seconds=30,
    )
    assert _FakeClient.headers == {"Authorization": "Bearer secret"}


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
