"""Tests for embedding backends and factory."""

from unittest.mock import MagicMock, patch

import pytest
from local_esearch.embeddings import get_backend
from local_esearch.embeddings.base import EmbeddingBackend


class TestGetBackend:
    """Test get_backend factory function."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert get_backend(None) is None

    def test_unknown_backend_raises(self):
        """Unknown backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_backend("unknown_backend")

    def test_case_insensitive(self):
        """Backend names are case-insensitive."""
        # We can't actually test this without API keys, so we mock
        with patch("local_esearch.embeddings.voyage.VoyageBackend") as mock_cls:
            mock_cls.return_value = MagicMock(spec=EmbeddingBackend)
            get_backend("VOYAGE")
            assert mock_cls.called

    def test_voyage_backend(self):
        """Test voyage backend instantiation."""
        with patch("local_esearch.embeddings.voyage.VoyageBackend") as mock_cls:
            mock_instance = MagicMock(spec=EmbeddingBackend)
            mock_cls.return_value = mock_instance
            result = get_backend("voyage")
            assert result is mock_instance

    def test_gemini_backend(self):
        """Test gemini backend instantiation."""
        with patch("local_esearch.embeddings.gemini.GeminiBackend") as mock_cls:
            mock_instance = MagicMock(spec=EmbeddingBackend)
            mock_cls.return_value = mock_instance
            result = get_backend("gemini")
            assert result is mock_instance

    def test_openai_backend(self):
        """Test openai backend instantiation."""
        with patch("local_esearch.embeddings.openai.OpenAIBackend") as mock_cls:
            mock_instance = MagicMock(spec=EmbeddingBackend)
            mock_cls.return_value = mock_instance
            result = get_backend("openai")
            assert result is mock_instance


class TestEmbeddingBackendProtocol:
    """Test EmbeddingBackend protocol compliance."""

    def test_mock_backend_satisfies_protocol(self):
        """Mock backend satisfies the protocol."""

        class MockBackend:
            backend_name = "mock"
            dimensions = 8
            model_name = "mock-model"

            def embed(self, text: str) -> list[float]:
                return [0.1] * self.dimensions

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(t) for t in texts]

        backend = MockBackend()
        assert backend.backend_name == "mock"
        assert backend.dimensions == 8
        assert len(backend.embed("test")) == 8
        assert len(backend.embed_batch(["a", "b"])) == 2
