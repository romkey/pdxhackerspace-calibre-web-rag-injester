from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
from sentence_transformers import SentenceTransformer

from calibre_web2rag.config import Settings


class Embedder(Protocol):
    def encode(self, chunks: list[str]) -> list[list[float]]:
        ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

    def encode(self, chunks: list[str]) -> list[list[float]]:
        vectors = self._model.encode(chunks, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


@dataclass
class OllamaEmbedder:
    model_name: str
    ollama_url: str
    timeout_seconds: int

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.ollama_url.rstrip("/"), timeout=self.timeout_seconds)

    def encode(self, chunks: list[str]) -> list[list[float]]:
        response = self._client.post("/api/embed", json={"model": self.model_name, "input": chunks})
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Ollama /api/embed response missing 'embeddings'")
        return embeddings


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.hf_cache_dir,
        )
    if provider == "ollama":
        return OllamaEmbedder(
            model_name=settings.embedding_model,
            ollama_url=settings.ollama_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
        "Supported: sentence_transformers, ollama"
    )
