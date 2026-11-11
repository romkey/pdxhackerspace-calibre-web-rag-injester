from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from calibre_web2rag.qdrant_store import QdrantStore, _distance


class _Client:
    """Stands in for QdrantClient; subclasses change what it reports."""

    exists: ClassVar[bool] = True
    # object: whatever shape the collection's vectors config takes.
    vectors: ClassVar[object] = SimpleNamespace(size=3)
    fail_indexes: ClassVar[bool] = False

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        # Any: the keyword arguments QdrantStore passes to create_collection.
        self.created: dict[str, Any] | None = None
        self.indexes: list[str] = []
        self.upserted: list[PointStruct] | None = None

    def collection_exists(self, name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs

    def get_collection(self, name: str) -> SimpleNamespace:
        params = SimpleNamespace(vectors=self.vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def create_payload_index(self, *, collection_name: str, field_name: str, **_: object) -> None:
        if self.fail_indexes:
            raise RuntimeError("index already exists")
        self.indexes.append(field_name)

    def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        self.upserted = points


def _store(
    monkeypatch: pytest.MonkeyPatch,
    client_cls: type[_Client] = _Client,
    *,
    vector_size: int = 3,
    distance: str = "cosine",
) -> tuple[QdrantStore, _Client]:
    monkeypatch.setattr("calibre_web2rag.qdrant_store.QdrantClient", client_cls)
    store = QdrantStore(
        url="http://qdrant:6333",
        api_key="key",
        collection="books",
        vector_size=vector_size,
        distance=distance,
    )
    assert isinstance(store.client, _Client)
    return store, store.client


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


def test_creates_missing_collection_and_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _New(_Client):
        exists = False

    _, client = _store(monkeypatch, _New, distance="dot")
    assert client.api_key == "key"
    assert client.created is not None
    assert client.created["collection_name"] == "books"
    config = client.created["vectors_config"]
    assert isinstance(config, VectorParams)
    assert config.size == 3
    assert config.distance == Distance.DOT
    assert {"authors", "tags", "book_id", "chunk_type"} <= set(client.indexes)


def test_existing_collection_matching_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _, client = _store(monkeypatch)
    assert client.created is None


def test_vector_size_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="already exists with vector size 3"):
        _store(monkeypatch, vector_size=384)


def test_named_vectors_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Named(_Client):
        vectors: ClassVar[object] = {"default": SimpleNamespace(size=768)}

    with pytest.raises(ValueError, match="vector size 768"):
        _store(monkeypatch, _Named)


@pytest.mark.parametrize("vectors", [{}, {"x": object()}, None])
def test_unknown_vector_shape_is_tolerated(
    monkeypatch: pytest.MonkeyPatch, vectors: object
) -> None:
    class _Odd(_Client):
        pass

    _Odd.vectors = vectors
    store, _ = _store(monkeypatch, _Odd)
    assert store._existing_vector_size() is None


def test_payload_index_errors_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Failing(_Client):
        fail_indexes = True

    _store(monkeypatch, _Failing)


def test_upsert_materializes_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = _store(monkeypatch)
    points = (PointStruct(id=i, vector=[0.0, 0.0, 0.0], payload={}) for i in range(2))
    store.upsert(points)
    assert isinstance(client.upserted, list)
    assert len(client.upserted) == 2
