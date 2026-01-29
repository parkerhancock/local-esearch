"""Hybrid search with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScoredDoc:
    """Document with score for fusion."""

    index: str
    doc_id: str
    source: dict[str, Any]
    keyword_rank: int | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    vector_score: float | None = None
    fused_score: float = 0.0


def reciprocal_rank_fusion(
    keyword_results: list[tuple[str, str, dict[str, Any], float]],  # (index, id, source, score)
    vector_results: list[tuple[str, str, dict[str, Any], float]],  # (index, id, source, score)
    k: int = 60,
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[ScoredDoc]:
    """Fuse keyword and vector search results using RRF.

    RRF formula: score = sum(weight / (k + rank))
    where rank starts at 1 for the best result.

    Args:
        keyword_results: Results from keyword/FTS5 search (index, id, source, score)
        vector_results: Results from vector search (index, id, source, score)
        k: RRF constant (default 60, standard value)
        keyword_weight: Weight for keyword results
        vector_weight: Weight for vector results

    Returns:
        Fused and sorted list of ScoredDoc
    """
    # Build document map
    docs: dict[str, ScoredDoc] = {}

    # Process keyword results
    for rank, (index, doc_id, source, score) in enumerate(keyword_results, start=1):
        key = f"{index}:{doc_id}"
        if key not in docs:
            docs[key] = ScoredDoc(index=index, doc_id=doc_id, source=source)
        docs[key].keyword_rank = rank
        docs[key].keyword_score = score
        docs[key].fused_score += keyword_weight / (k + rank)

    # Process vector results
    for rank, (index, doc_id, source, score) in enumerate(vector_results, start=1):
        key = f"{index}:{doc_id}"
        if key not in docs:
            docs[key] = ScoredDoc(index=index, doc_id=doc_id, source=source)
        docs[key].vector_rank = rank
        docs[key].vector_score = score
        docs[key].fused_score += vector_weight / (k + rank)

    # Sort by fused score descending
    return sorted(docs.values(), key=lambda d: d.fused_score, reverse=True)


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]
