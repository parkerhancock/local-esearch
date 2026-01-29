"""OpenAI embedding backend."""

from __future__ import annotations

import os


class OpenAIBackend:
    """OpenAI embedding backend.

    Uses text-embedding-3-small model (1536 dimensions).
    Requires OPENAI_API_KEY environment variable.
    """

    dimensions: int = 1536
    model_name: str = "text-embedding-3-small"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize OpenAI backend.

        Args:
            model: Model name (default: text-embedding-3-small)
            api_key: API key (default: from OPENAI_API_KEY env var)
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Install with: pip install openai"
            ) from e

        if model:
            self.model_name = model
            # Update dimensions based on model
            if "3-small" in model:
                self.dimensions = 1536
            elif "3-large" in model:
                self.dimensions = 3072
            elif "ada" in model:
                self.dimensions = 1536

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self._client = OpenAI(api_key=key)

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
