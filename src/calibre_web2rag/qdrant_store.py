from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def _distance(name: str) -> Distance:
    match name.lower():
        case "cosine":
            return Distance.COSINE
        case "dot":
            return Distance.DOT
        case "euclid":
            return Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported Qdrant distance metric: {name}")


class QdrantStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection: str,
        vector_size: int,
        distance: str,
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection(vector_size=vector_size, distance=distance)

    def _ensure_collection(self, *, vector_size: int, distance: str) -> None:
        exists = self.client.collection_exists(self.collection)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=_distance(distance)),
            )

    def upsert(self, points: Iterable[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=list(points))
