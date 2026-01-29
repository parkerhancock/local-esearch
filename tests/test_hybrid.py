"""Tests for hybrid search functionality."""

import pytest
from local_esearch.hybrid import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Test RRF algorithm."""

    def test_basic_fusion(self):
        """Test basic RRF with two result sets."""
        keyword_results = [
            ("idx", "doc1", {"title": "Doc 1"}, 10.0),
            ("idx", "doc2", {"title": "Doc 2"}, 8.0),
            ("idx", "doc3", {"title": "Doc 3"}, 5.0),
        ]
        vector_results = [
            ("idx", "doc2", {"title": "Doc 2"}, 0.95),
            ("idx", "doc3", {"title": "Doc 3"}, 0.90),
            ("idx", "doc4", {"title": "Doc 4"}, 0.85),
        ]

        fused = reciprocal_rank_fusion(keyword_results, vector_results)

        # All unique docs should be in result
        assert len(fused) == 4

        # Doc2 appears in both, should have highest score
        doc_scores = {d.doc_id: d.fused_score for d in fused}
        assert doc_scores["doc2"] > doc_scores["doc1"]

    def test_empty_results(self):
        """Test RRF with empty result sets."""
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_single_source(self):
        """Test RRF with only one source having results."""
        keyword_results = [
            ("idx", "doc1", {"title": "Doc 1"}, 10.0),
        ]

        fused = reciprocal_rank_fusion(keyword_results, [])

        assert len(fused) == 1
        assert fused[0].doc_id == "doc1"

    def test_weighted_fusion(self):
        """Test RRF with custom weights."""
        keyword_results = [
            ("idx", "doc1", {"title": "Doc 1"}, 10.0),
        ]
        vector_results = [
            ("idx", "doc2", {"title": "Doc 2"}, 0.95),
        ]

        # Heavy keyword weight
        fused_keyword = reciprocal_rank_fusion(
            keyword_results, vector_results, keyword_weight=2.0, vector_weight=1.0
        )

        # Heavy vector weight
        fused_vector = reciprocal_rank_fusion(
            keyword_results, vector_results, keyword_weight=1.0, vector_weight=2.0
        )

        # First result should differ based on weights
        assert fused_keyword[0].doc_id == "doc1"
        assert fused_vector[0].doc_id == "doc2"

    def test_score_calculation(self):
        """Test that RRF score is calculated correctly."""
        keyword_results = [
            ("idx", "doc1", {"title": "Doc 1"}, 10.0),  # rank 1
        ]
        vector_results = [
            ("idx", "doc1", {"title": "Doc 1"}, 0.95),  # rank 1
        ]

        fused = reciprocal_rank_fusion(keyword_results, vector_results, k=60)

        # Score should be 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2 / 61
        assert abs(fused[0].fused_score - expected_score) < 0.0001

    def test_preserves_source_data(self):
        """Test that source documents are preserved."""
        source = {"title": "Test Doc", "body": "Content here"}
        keyword_results = [
            ("test_index", "doc123", source, 10.0),
        ]

        fused = reciprocal_rank_fusion(keyword_results, [])

        assert fused[0].index == "test_index"
        assert fused[0].doc_id == "doc123"
        assert fused[0].source == source
        assert fused[0].keyword_rank == 1
        assert fused[0].keyword_score == 10.0
        assert fused[0].vector_rank is None


class TestSearchModes:
    """Test different search mode behaviors."""

    def test_keyword_mode_default(self, es):
        """Test that keyword mode is default."""
        es.index(index="test", id="1", document={"title": "Hello World"})

        # Default mode should be keyword
        response = es.search(index="test", body={"query": {"match": {"title": "hello"}}})

        assert response["hits"]["total"]["value"] == 1

    def test_semantic_mode_without_backend(self, es):
        """Test semantic mode falls back gracefully without embedding backend."""
        es.index(index="test", id="1", document={"title": "Hello World"})

        # Should work but return keyword results (no semantic backend)
        response = es.search(
            index="test",
            body={"query": {"match": {"title": "hello"}}},
            mode="semantic",
        )

        # Without embedding backend, falls back to empty results for semantic
        # which means we only get keyword results
        assert "hits" in response

    def test_hybrid_mode_without_backend(self, es):
        """Test hybrid mode falls back to keyword without embedding backend."""
        es.index(index="test", id="1", document={"title": "Hello World"})

        response = es.search(
            index="test",
            body={"query": {"match": {"title": "hello"}}},
            mode="hybrid",
        )

        # Should still work, falling back to keyword-only
        assert response["hits"]["total"]["value"] == 1


@pytest.fixture
def es():
    """Create an in-memory Elasticsearch client."""
    from local_esearch import Elasticsearch

    client = Elasticsearch(path=":memory:")
    yield client
    client.close()
