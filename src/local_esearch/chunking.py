"""Text chunking utilities for embedding search.

Mirrors Elasticsearch's semantic_text chunking behavior:
- Default 250 words per chunk
- Default 100 word overlap
- Chunks stored as nested structure with inner_hits support
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ES semantic_text defaults: 250 words per chunk, 100 word overlap
DEFAULT_CHUNK_SIZE = 250  # words
DEFAULT_CHUNK_OVERLAP = 100  # words


@dataclass
class Chunk:
    """A chunk of text with position information."""

    text: str
    index: int
    start_char: int
    end_char: int


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic.

    Uses ~4 characters per token as rough approximation.
    """
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Mirrors Elasticsearch semantic_text behavior: splits on word boundaries
    with configurable size and overlap.

    Args:
        text: Text to chunk
        chunk_size: Words per chunk (default: 250, matching ES)
        chunk_overlap: Overlap words between chunks (default: 100, matching ES)
        min_chunk_size: Minimum words for final chunk (default: 20% of chunk_size)

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    # Default min_chunk_size to 20% of chunk_size
    if min_chunk_size is None:
        min_chunk_size = max(1, chunk_size // 5)

    # Short text - return as single chunk
    if len(words) <= chunk_size:
        return [
            Chunk(
                text=text.strip(),
                index=0,
                start_char=0,
                end_char=len(text),
            )
        ]

    # Build word position index for character offsets
    word_positions = []  # (start_char, end_char) for each word
    pos = 0
    for word in words:
        start = text.find(word, pos)
        end = start + len(word)
        word_positions.append((start, end))
        pos = end

    chunks = []
    chunk_idx = 0
    word_idx = 0
    step = chunk_size - chunk_overlap  # Words to advance each chunk

    while word_idx < len(words):
        # Determine chunk boundaries in words
        start_word = word_idx
        end_word = min(word_idx + chunk_size, len(words))

        # Get character positions
        start_char = word_positions[start_word][0]
        end_char = word_positions[end_word - 1][1]

        chunk_text_str = text[start_char:end_char].strip()

        # Add chunk if it meets minimum size or is the last one
        word_count = end_word - start_word
        if word_count >= min_chunk_size or word_idx + step >= len(words):
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    index=chunk_idx,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            chunk_idx += 1

        # Advance by step (chunk_size - overlap)
        word_idx += step

        # If we'd create a tiny final chunk, just stop
        if word_idx < len(words) and len(words) - word_idx < min_chunk_size:
            # Extend the last chunk to include remaining words
            if chunks:
                last_chunk = chunks[-1]
                end_char = word_positions[-1][1]
                chunks[-1] = Chunk(
                    text=text[last_chunk.start_char : end_char].strip(),
                    index=last_chunk.index,
                    start_char=last_chunk.start_char,
                    end_char=end_char,
                )
            break

    return chunks


def create_chunker(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int | None = None,
) -> Callable[[str], list[Chunk]]:
    """Create a chunking function with specified parameters.

    Args:
        chunk_size: Words per chunk (default: 250)
        chunk_overlap: Overlap words between chunks (default: 100)
        min_chunk_size: Minimum words for final chunk (default: 20% of chunk_size)

    Returns:
        Function that takes text and returns list of Chunk objects
    """

    def chunker(text: str) -> list[Chunk]:
        return chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    return chunker
