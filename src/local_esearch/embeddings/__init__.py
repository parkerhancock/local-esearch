"""Embedding backends for semantic search."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend


def get_backend(name: str | None) -> EmbeddingBackend | None:
    """Get embedding backend by name.

    Args:
        name: Backend name - "voyage", "gemini", "openai", or None

    Returns:
        EmbeddingBackend instance or None if name is None

    Raises:
        ValueError: If backend name is unknown
        ImportError: If required dependency is not installed
    """
    if name is None:
        return None

    name = name.lower()

    if name == "voyage":
        from local_esearch.embeddings.voyage import VoyageBackend

        return VoyageBackend()

    if name == "gemini":
        from local_esearch.embeddings.gemini import GeminiBackend

        return GeminiBackend()

    if name == "openai":
        from local_esearch.embeddings.openai import OpenAIBackend

        return OpenAIBackend()

    raise ValueError(f"Unknown embedding backend: {name}")


__all__ = ["get_backend"]
