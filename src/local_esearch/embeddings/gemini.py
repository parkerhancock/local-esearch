"""Google Gemini embedding backend."""

from __future__ import annotations

import os


class GeminiBackend:
    """Google Gemini embedding backend.

    Uses text-embedding-004 model (768 dimensions).
    Requires GOOGLE_API_KEY environment variable.
    """

    backend_name: str = "gemini"
    dimensions: int = 768
    model_name: str = "text-embedding-004"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize Gemini backend.

        Args:
            model: Model name (default: text-embedding-004)
            api_key: API key (default: from GOOGLE_API_KEY env var)
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required for Gemini backend. "
                "Install with: pip install google-generativeai"
            ) from e

        if model:
            self.model_name = model

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable required")

        genai.configure(api_key=key)
        self._genai = genai

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        result = self._genai.embed_content(
            model=f"models/{self.model_name}",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        embeddings = []
        # Gemini doesn't have native batch API, so we call individually
        # Could potentially use async for better performance
        for text in texts:
            result = self._genai.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings
