from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

logger = logging.getLogger(__name__)


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


_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "authors": PayloadSchemaType.KEYWORD,
    "tags": PayloadSchemaType.KEYWORD,
    "languages": PayloadSchemaType.KEYWORD,
    "series": PayloadSchemaType.KEYWORD,
    "publisher": PayloadSchemaType.KEYWORD,
    "book_id": PayloadSchemaType.INTEGER,
    "chunk_type": PayloadSchemaType.KEYWORD,
}


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
        else:
            existing_size = self._existing_vector_size()
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' already exists with vector size "
                    f"{existing_size}, but VECTOR_SIZE is {vector_size}. "
                    "VECTOR_SIZE is only used at collection creation time. "
                    "Use a new QDRANT_COLLECTION name or recreate the collection."
                )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        for field_name, schema_type in _PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                logger.debug("Payload index for '%s' may already exist, skipping", field_name)

    def upsert(self, points: Iterable[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=list(points))

    def _existing_vector_size(self) -> int | None:
        info = self.client.get_collection(self.collection)
        vectors: Any = info.config.params.vectors
        # Single-vector collections return VectorParams directly.
        if hasattr(vectors, "size"):
            return int(vectors.size)
        # Named-vector collections return dict[str, VectorParams].
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            if hasattr(first, "size"):
                return int(first.size)
        return None
