from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from calibre_web2rag.config import Settings

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4

# OpenAI rejects /embeddings requests with more than 2048 inputs or more than
# 300k tokens in total. Stay well under both.
_OPENAI_MAX_INPUTS_PER_REQUEST = 2048
_OPENAI_MAX_TOKENS_PER_REQUEST = 250_000


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
class OpenAIEmbedder:
    """Embedder for the OpenAI embeddings API (``POST {base_url}/embeddings``).

    Works with api.openai.com and any OpenAI-compatible server (Ollama's
    ``/v1`` endpoint, vLLM, LM Studio, LiteLLM, ...). The API key is optional
    so that unauthenticated self-hosted servers can be used.
    """

    model_name: str
    base_url: str
    api_key: str | None
    timeout_seconds: int
    max_tokens: int = 0
    dimensions: int | None = None

    def __post_init__(self) -> None:
        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else None
        )
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def encode(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        chunks = _truncate(chunks, self.max_tokens)
        vectors: list[list[float]] = []
        for batch in _request_batches(chunks):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, chunks: list[str]) -> list[list[float]]:
        payload: dict = {
            "model": self.model_name,
            "input": chunks,
            "encoding_format": "float",
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        response = self._client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(chunks):
            raise ValueError(
                "OpenAI /embeddings response missing 'data' or wrong length"
            )
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


def _request_batches(
    texts: list[str],
    max_inputs: int | None = None,
    max_tokens: int | None = None,
) -> list[list[str]]:
    """Split texts into request-sized batches under the API's input/token caps."""
    max_inputs = max_inputs or _OPENAI_MAX_INPUTS_PER_REQUEST
    max_tokens = max_tokens or _OPENAI_MAX_TOKENS_PER_REQUEST
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = len(text) // _CHARS_PER_TOKEN + 1
        if current and (
            len(current) >= max_inputs or current_tokens + tokens > max_tokens
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    ctx = settings.embedding_context_length
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.hf_cache_dir,
            max_tokens=ctx,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_seconds=settings.openai_timeout_seconds,
            max_tokens=ctx,
            dimensions=settings.embedding_dimensions,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
        "Supported: openai, sentence_transformers"
    )
