from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from calibre_web2rag.config import Settings

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4


class Embedder(Protocol):
    def encode(self, chunks: list[str]) -> list[list[float]]:
        ...


def _truncate(texts: list[str], max_tokens: int) -> list[str]:
    """Truncate texts that would exceed the model's context length."""
    if max_tokens <= 0:
        return texts
    max_chars = max_tokens * _CHARS_PER_TOKEN
    result: list[str] = []
    for text in texts:
        if len(text) > max_chars:
            logger.warning(
                "Truncating text from %d to %d chars "
                "(~%d token limit)",
                len(text),
                max_chars,
                max_tokens,
            )
            result.append(text[:max_chars])
        else:
            result.append(text)
    return result


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install calibre-web2rag[sentence-transformers]"
        ) from None
    return SentenceTransformer


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    cache_dir: str | None = None
    max_tokens: int = 0

    def __post_init__(self) -> None:
        cls = _load_sentence_transformer()
        self._model = cls(self.model_name, cache_folder=self.cache_dir)

    def encode(self, chunks: list[str]) -> list[list[float]]:
        chunks = _truncate(chunks, self.max_tokens)
        vectors = self._model.encode(chunks, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


@dataclass
class OllamaEmbedder:
    model_name: str
    ollama_url: str
    timeout_seconds: int
    max_tokens: int = 0

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.ollama_url.rstrip("/"), timeout=self.timeout_seconds
        )

    def encode(self, chunks: list[str]) -> list[list[float]]:
        chunks = _truncate(chunks, self.max_tokens)
        response = self._client.post(
            "/api/embed",
            json={"model": self.model_name, "input": chunks},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError(
                "Ollama /api/embed response missing 'embeddings'"
            )
        return embeddings


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    ctx = settings.embedding_context_length
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.hf_cache_dir,
            max_tokens=ctx,
        )
    if provider == "ollama":
        return OllamaEmbedder(
            model_name=settings.embedding_model,
            ollama_url=settings.ollama_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            max_tokens=ctx,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
        "Supported: sentence_transformers, ollama"
    )
