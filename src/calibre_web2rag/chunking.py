from __future__ import annotations


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(normalized[start:end])
        start += step
    return chunks
