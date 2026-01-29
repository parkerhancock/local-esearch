"""Tests for text chunking functionality."""

from local_esearch.chunking import Chunk, chunk_text, create_chunker


class TestChunkText:
    """Test chunk_text function."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns single chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0

    def test_basic_chunking(self):
        """Text is split into overlapping chunks."""
        # Create text with 100 words
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        # Chunk with 30 words, 10 word overlap
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should be around 30 words (except maybe last)
        for chunk in chunks[:-1]:
            word_count = len(chunk.text.split())
            assert 25 <= word_count <= 35

    def test_chunk_overlap(self):
        """Chunks should overlap."""
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        # Check that chunks overlap
        if len(chunks) >= 2:
            chunk1_words = set(chunks[0].text.split())
            chunk2_words = set(chunks[1].text.split())
            overlap = chunk1_words & chunk2_words
            assert len(overlap) > 0

    def test_chunk_indices(self):
        """Chunk indices should be sequential."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_positions(self):
        """Chunk character positions should be valid."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        for chunk in chunks:
            # Position should be within text bounds
            assert 0 <= chunk.start_char < len(text)
            assert chunk.start_char < chunk.end_char <= len(text)

            # Extracted text should match
            assert text[chunk.start_char : chunk.end_char].strip() == chunk.text


class TestCreateChunker:
    """Test create_chunker factory."""

    def test_creates_callable(self):
        """create_chunker returns a callable."""
        chunker = create_chunker()
        assert callable(chunker)

    def test_chunker_returns_chunks(self):
        """Chunker returns list of Chunk objects."""
        chunker = create_chunker(chunk_size=20, chunk_overlap=5)
        text = "Word " * 50

        chunks = chunker(text)

        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_custom_parameters(self):
        """Chunker uses custom parameters."""
        chunker = create_chunker(chunk_size=10, chunk_overlap=2)
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)

        chunks = chunker(text)

        # With chunk_size=10 and 30 words, should have multiple chunks
        assert len(chunks) > 1


class TestESDefaults:
    """Test that defaults match Elasticsearch semantic_text behavior."""

    def test_default_chunk_size(self):
        """Default chunk size should be 250 words (ES default)."""
        from local_esearch.chunking import DEFAULT_CHUNK_SIZE

        assert DEFAULT_CHUNK_SIZE == 250

    def test_default_chunk_overlap(self):
        """Default chunk overlap should be 100 words (ES default)."""
        from local_esearch.chunking import DEFAULT_CHUNK_OVERLAP

        assert DEFAULT_CHUNK_OVERLAP == 100

    def test_long_document_chunking(self):
        """Long document is chunked with ES-like defaults."""
        # Create ~500 word document
        paragraphs = [" ".join([f"paragraph{p}word{w}" for w in range(50)]) for p in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text)  # Use defaults

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Each chunk should be around 250 words
        for chunk in chunks[:-1]:
            word_count = len(chunk.text.split())
            # Allow some variance for word boundaries
            assert 200 <= word_count <= 300
