"""Voyage AI embedding backend."""

from __future__ import annotations

import os


class VoyageBackend:
    """Voyage AI embedding backend.

    Uses voyage-3-lite model (1024 dimensions) for efficient semantic search.
    Requires VOYAGE_API_KEY environment variable.
    """

    backend_name: str = "voyage"
    dimensions: int = 1024
    model_name: str = "voyage-3-lite"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize Voyage backend.

        Args:
            model: Model name (default: voyage-3-lite)
            api_key: API key (default: from VOYAGE_API_KEY env var)
        """
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "voyageai package required for Voyage backend. Install with: pip install voyageai"
            ) from e

        if model:
            self.model_name = model
            # Update dimensions based on model
            if "3-lite" in model:
                self.dimensions = 1024
            elif "3" in model:
                self.dimensions = 1024
            elif "2" in model:
                self.dimensions = 1024

        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY environment variable required")

        self._client = voyageai.Client(api_key=key)

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        result = self._client.embed([text], model=self.model_name, input_type="document")
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        result = self._client.embed(texts, model=self.model_name, input_type="document")
        return result.embeddings
