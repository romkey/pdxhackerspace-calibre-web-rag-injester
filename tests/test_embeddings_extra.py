from __future__ import annotations

import builtins

import pytest

from calibre_web2rag import embeddings
from calibre_web2rag.embeddings import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    _load_sentence_transformer,
    build_embedder,
)


class _Vector(list):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeModel:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.seen: list[str] = []

    def encode(self, chunks, show_progress_bar=False):
        self.seen = chunks
        # Mix numpy-like (has tolist) and plain tuples.
        return [_Vector([1.0, 2.0]) if i % 2 == 0 else (3.0, 4.0) for i in range(len(chunks))]


def test_sentence_transformer_encode_and_truncate(monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda: _FakeModel)
    embedder = SentenceTransformerEmbedder(model_name="m", max_tokens=2)
    vectors = embedder.encode(["x" * 100, "short"])
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert embedder._model.seen == ["x" * 8, "short"]


def test_load_sentence_transformer_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match="pip install calibre-web2rag"):
        _load_sentence_transformer()


def test_load_sentence_transformer_present(monkeypatch) -> None:
    import sys
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    assert _load_sentence_transformer() is _FakeModel


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_ollama_missing_embeddings_raises(monkeypatch) -> None:
    class _Client:
        def __init__(self, **kwargs) -> None:
            pass

        def post(self, path, json):
            return _Response({"error": "model not found"})

    monkeypatch.setattr(embeddings.httpx, "Client", _Client)
    embedder = OllamaEmbedder(
        model_name="m", ollama_url="http://o", api_key=None, timeout_seconds=1
    )
    with pytest.raises(ValueError, match="missing 'embeddings'"):
        embedder.encode(["a"])


def test_build_embedder_unsupported_provider(make_settings) -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER 'openai'"):
        build_embedder(make_settings(embedding_provider="openai"))
