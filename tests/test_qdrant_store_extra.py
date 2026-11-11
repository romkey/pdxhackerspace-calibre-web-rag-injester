from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, PointStruct

from calibre_web2rag.qdrant_store import QdrantStore, _distance


def _collection(vectors):
    return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))


class _Client:
    exists = True
    vectors: object = SimpleNamespace(size=3)
    fail_indexes = False

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.created: dict | None = None
        self.upserted: list | None = None

    def collection_exists(self, name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs) -> None:
        self.created = kwargs

    def get_collection(self, name: str):
        return _collection(self.vectors)

    def create_payload_index(self, **kwargs) -> None:
        if self.fail_indexes:
            raise RuntimeError("index already exists")

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserted = points


def _store(monkeypatch, client_cls=_Client, **kwargs) -> QdrantStore:
    monkeypatch.setattr("calibre_web2rag.qdrant_store.QdrantClient", client_cls)
    params = dict(
        url="http://qdrant:6333",
        api_key="key",
        collection="books",
        vector_size=3,
        distance="cosine",
    )
    params.update(kwargs)
    return QdrantStore(**params)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cosine", Distance.COSINE),
        ("DOT", Distance.DOT),
        ("Euclid", Distance.EUCLID),
    ],
)
def test_distance_mapping(name: str, expected: Distance) -> None:
    assert _distance(name) == expected


def test_distance_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported Qdrant distance metric"):
        _distance("manhattan")


def test_creates_missing_collection(monkeypatch) -> None:
    class _New(_Client):
        exists = False

    store = _store(monkeypatch, _New, distance="dot")
    assert store.client.api_key == "key"
    created = store.client.created
    assert created["collection_name"] == "books"
    assert created["vectors_config"].size == 3
    assert created["vectors_config"].distance == Distance.DOT


def test_existing_collection_matching_size(monkeypatch) -> None:
    store = _store(monkeypatch)
    assert store.client.created is None


def test_named_vectors_size_mismatch(monkeypatch) -> None:
    class _Named(_Client):
        vectors = {"default": SimpleNamespace(size=768)}

    with pytest.raises(ValueError, match="vector size 768"):
        _store(monkeypatch, _Named)


@pytest.mark.parametrize("vectors", [{}, {"x": object()}, None])
def test_unknown_vector_shape_is_tolerated(monkeypatch, vectors) -> None:
    class _Odd(_Client):
        pass

    _Odd.vectors = vectors
    store = _store(monkeypatch, _Odd)
    assert store._existing_vector_size() is None


def test_payload_index_errors_are_ignored(monkeypatch) -> None:
    class _Failing(_Client):
        fail_indexes = True

    _store(monkeypatch, _Failing)


def test_upsert_materializes_iterable(monkeypatch) -> None:
    store = _store(monkeypatch)
    points = (PointStruct(id=i, vector=[0.0, 0.0, 0.0], payload={}) for i in range(2))
    store.upsert(points)
    assert isinstance(store.client.upserted, list)
    assert len(store.client.upserted) == 2
