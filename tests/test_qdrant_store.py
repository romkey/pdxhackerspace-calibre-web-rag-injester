from types import SimpleNamespace

import pytest

from calibre_web2rag.qdrant_store import QdrantStore


class _FakeQdrantClient:
    def __init__(self, url: str, api_key: str | None = None) -> None:  # noqa: ARG002
        self.created = False

    def collection_exists(self, name: str) -> bool:  # noqa: ARG002
        return True

    def create_collection(self, **kwargs) -> None:  # pragma: no cover
        self.created = True

    def get_collection(self, name: str):  # noqa: ARG002
        vectors = SimpleNamespace(size=768)
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)

    def upsert(self, collection_name: str, points: list) -> None:  # noqa: ARG002
        return None

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema=None  # noqa: ARG002
    ) -> None:
        return None


def test_vector_size_mismatch_raises(monkeypatch) -> None:
    monkeypatch.setattr("calibre_web2rag.qdrant_store.QdrantClient", _FakeQdrantClient)
    with pytest.raises(ValueError, match="already exists with vector size 768"):
        QdrantStore(
            url="http://qdrant:6333",
            api_key=None,
            collection="books",
            vector_size=384,
            distance="cosine",
        )


def test_payload_indexes_created(monkeypatch) -> None:
    created_indexes: list[str] = []

    class _TrackingClient(_FakeQdrantClient):
        def collection_exists(self, name: str) -> bool:  # noqa: ARG002
            return False

        def get_collection(self, name: str):  # noqa: ARG002
            vectors = SimpleNamespace(size=768)
            params = SimpleNamespace(vectors=vectors)
            config = SimpleNamespace(params=params)
            return SimpleNamespace(config=config)

        def create_payload_index(
            self, collection_name: str, field_name: str, field_schema=None  # noqa: ARG002
        ) -> None:
            created_indexes.append(field_name)

    monkeypatch.setattr("calibre_web2rag.qdrant_store.QdrantClient", _TrackingClient)
    QdrantStore(
        url="http://qdrant:6333",
        api_key=None,
        collection="books",
        vector_size=768,
        distance="cosine",
    )
    assert "authors" in created_indexes
    assert "tags" in created_indexes
    assert "book_id" in created_indexes
    assert "chunk_type" in created_indexes
