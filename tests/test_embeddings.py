from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
import pytest

from calibre_web2rag import embeddings
from calibre_web2rag.config import Settings
from calibre_web2rag.embeddings import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    _load_sentence_transformer,
    _truncate,
    build_embedder,
)

SettingsFactory = Callable[..., Settings]  # the make_settings fixture


class _FakeResponse:
    # Any: a JSON body.
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeClient:
    """Stands in for httpx.Client; records headers and returns `body`."""

    headers: Mapping[str, str] | None = None
    # Any: a JSON body.
    body: dict[str, Any] = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}  # noqa: RUF012 -- read-only default

    def __init__(self, base_url: str, headers: Mapping[str, str] | None, timeout: int) -> None:
        type(self).headers = headers

    # Any: a JSON request body.
    def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
        assert path == "/api/embed"
        assert json["model"] == "nomic-embed-text"
        return _FakeResponse(self.body)


class _Vector(list[float]):
    """Like a numpy row: has tolist()."""

    def tolist(self) -> list[float]:
        return list(self)


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.seen: list[str] = []

    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[Sequence[float]]:
        self.seen = chunks
        # Mix numpy-like rows (with tolist) and plain tuples.
        return [_Vector([1.0, 2.0]) if i % 2 == 0 else (3.0, 4.0) for i in range(len(chunks))]


def _ollama(api_key: str | None = None) -> OllamaEmbedder:
    return OllamaEmbedder(
        model_name="nomic-embed-text",
        ollama_url="http://ollama:11434",
        api_key=api_key,
        timeout_seconds=30,
    )


def test_ollama_embedder_calls_embed_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    vectors = _ollama().encode(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_embedder_uses_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _ollama(api_key="secret")
    assert _FakeClient.headers == {"Authorization": "Bearer secret"}


def test_ollama_missing_embeddings_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorClient(_FakeClient):
        body = {"error": "model not found"}  # noqa: RUF012 -- read-only

    monkeypatch.setattr(httpx, "Client", _ErrorClient)
    with pytest.raises(ValueError, match="missing 'embeddings'"):
        _ollama().encode(["a"])


def test_build_embedder_uses_ollama_provider(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    embedder = build_embedder(make_settings(embedding_provider="ollama"))
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.max_tokens == 8192


def test_build_embedder_uses_st_cache_directory(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda: _FakeSentenceTransformer)
    settings = make_settings(embedding_provider="sentence_transformers", hf_cache_dir="/hf-cache")
    embedder = build_embedder(settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder._model.cache_folder == "/hf-cache"


def test_build_embedder_unsupported_provider(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER 'openai'"):
        build_embedder(make_settings(embedding_provider="openai"))


def test_sentence_transformer_encode_and_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda: _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder(model_name="m", max_tokens=2)
    assert embedder.encode(["x" * 100, "short"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert embedder._model.seen == ["x" * 8, "short"]


def test_load_sentence_transformer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    # Any: __import__'s signature.
    def _import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match="pip install calibre-web2rag"):
        _load_sentence_transformer()


def test_load_sentence_transformer_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("sentence_transformers")
    setattr(fake, "SentenceTransformer", _FakeSentenceTransformer)  # noqa: B010 -- module has no such attribute for mypy
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    assert _load_sentence_transformer() is _FakeSentenceTransformer


def test_truncate_shortens_long_texts() -> None:
    result = _truncate(["hello", "x" * 100], max_tokens=10)
    assert result[0] == "hello"
    assert len(result[1]) == 10 * 4  # _CHARS_PER_TOKEN = 4


def test_truncate_noop_when_disabled() -> None:
    long_text = "x" * 100
    assert _truncate([long_text], max_tokens=0) == [long_text]
