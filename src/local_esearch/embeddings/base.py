"""Base protocol for embedding backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding backends.

    Implementations must provide:
    - dimensions: Vector dimensionality
    - model_name: Name of the embedding model
    - embed(): Embed a single text
    - embed_batch(): Embed multiple texts efficiently
    """

    dimensions: int
    model_name: str

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...
