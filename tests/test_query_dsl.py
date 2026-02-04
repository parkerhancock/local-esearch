"""Tests for Query DSL compiler."""

import pytest
from local_esearch.query_dsl import QueryCompiler


@pytest.fixture
def compiler():
    return QueryCompiler()


class TestBasicQueries:
    """Test basic query types."""

    def test_match_all(self, compiler):
        """Test match_all returns empty where clause."""
        result = compiler.compile({"match_all": {}})
        assert result.where_clause == ""
        assert result.params == []

    def test_match(self, compiler):
        """Test match query compiles to FTS."""
        result = compiler.compile({"match": {"title": "hello world"}})

        # FTS is handled by document_fts_search_sql, not where_clause
        assert result.where_clause == ""
        assert result.uses_fts is True
        assert result.fts_query == "hello world"

    def test_match_with_query_object(self, compiler):
        """Test match query with nested query object."""
        result = compiler.compile({"match": {"title": {"query": "hello world"}}})

        assert result.uses_fts is True
        assert result.fts_query == "hello world"

    def test_match_phrase(self, compiler):
        """Test match_phrase compiles to FTS5 phrase query."""
        result = compiler.compile({"match_phrase": {"title": "hello world"}})

        assert result.uses_fts is True
        assert '"hello world"' in result.fts_query

    def test_term(self, compiler):
        """Test term query compiles to JSON extract."""
        result = compiler.compile({"term": {"status": "active"}})

        assert "json_extract" in result.where_clause
        assert "$.status" in result.params
        assert "active" in result.params

    def test_terms(self, compiler):
        """Test terms query compiles to IN clause."""
        result = compiler.compile({"terms": {"status": ["active", "pending"]}})

        assert "IN" in result.where_clause
        assert "$.status" in result.params
        assert "active" in result.params
        assert "pending" in result.params

    def test_range_gte_lte(self, compiler):
        """Test range query with gte and lte."""
        result = compiler.compile({"range": {"price": {"gte": 10, "lte": 100}}})

        assert ">=" in result.where_clause
        assert "<=" in result.where_clause
        assert 10 in result.params
        assert 100 in result.params

    def test_range_gt_lt(self, compiler):
        """Test range query with gt and lt."""
        result = compiler.compile({"range": {"price": {"gt": 10, "lt": 100}}})

        assert "> ?" in result.where_clause
        assert "< ?" in result.where_clause

    def test_exists(self, compiler):
        """Test exists query."""
        result = compiler.compile({"exists": {"field": "tags"}})

        assert "IS NOT NULL" in result.where_clause
        assert "$.tags" in result.params

    def test_ids(self, compiler):
        """Test ids query."""
        result = compiler.compile({"ids": {"values": ["1", "2", "3"]}})

        assert "_id IN" in result.where_clause
        assert "1" in result.params
        assert "2" in result.params
        assert "3" in result.params

    def test_prefix(self, compiler):
        """Test prefix query."""
        result = compiler.compile({"prefix": {"title": "hel"}})

        assert "LIKE" in result.where_clause
        assert "hel%" in result.params

    def test_wildcard(self, compiler):
        """Test wildcard query."""
        result = compiler.compile({"wildcard": {"title": "hel*o"}})

        assert "LIKE" in result.where_clause
        assert "hel%o" in result.params


class TestBoolQuery:
    """Test bool query combinations."""

    def test_bool_must(self, compiler):
        """Test bool with must clause."""
        result = compiler.compile(
            {
                "bool": {
                    "must": [
                        {"term": {"status": "active"}},
                        {"term": {"type": "article"}},
                    ]
                }
            }
        )

        assert "AND" in result.where_clause
        assert "$.status" in result.params
        assert "$.type" in result.params

    def test_bool_must_not(self, compiler):
        """Test bool with must_not clause."""
        result = compiler.compile(
            {
                "bool": {
                    "must_not": [
                        {"term": {"status": "deleted"}},
                    ]
                }
            }
        )

        assert "NOT" in result.where_clause
        assert "$.status" in result.params

    def test_bool_should(self, compiler):
        """Test bool with should clause."""
        result = compiler.compile(
            {
                "bool": {
                    "should": [
                        {"term": {"status": "active"}},
                        {"term": {"status": "pending"}},
                    ]
                }
            }
        )

        assert "OR" in result.where_clause

    def test_bool_filter(self, compiler):
        """Test bool with filter clause (same as must for our purposes)."""
        result = compiler.compile(
            {
                "bool": {
                    "filter": [
                        {"term": {"status": "active"}},
                    ]
                }
            }
        )

        assert "$.status" in result.params

    def test_bool_combined(self, compiler):
        """Test bool with multiple clause types.

        When must is present, should becomes optional (minimum_should_match defaults to 0).
        """
        result = compiler.compile(
            {
                "bool": {
                    "must": [{"term": {"type": "article"}}],
                    "must_not": [{"term": {"status": "deleted"}}],
                    "should": [{"match": {"title": "important"}}],
                }
            }
        )

        assert "AND" in result.where_clause
        assert "NOT" in result.where_clause
        # When must is present, should is optional (minimum_should_match=0 by default)
        # So OR may not appear in WHERE clause
        assert result.uses_fts is True  # Should still affect scoring

    def test_nested_bool(self, compiler):
        """Test nested bool queries."""
        result = compiler.compile(
            {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"term": {"category": "tech"}},
                                    {"term": {"category": "science"}},
                                ]
                            }
                        }
                    ],
                    "filter": [{"term": {"status": "published"}}],
                }
            }
        )

        assert "OR" in result.where_clause
        assert "AND" in result.where_clause


class TestFieldPaths:
    """Test field path handling."""

    def test_simple_field(self, compiler):
        """Test simple field name."""
        result = compiler.compile({"term": {"status": "active"}})
        assert "$.status" in result.params

    def test_nested_field(self, compiler):
        """Test nested field path."""
        result = compiler.compile({"term": {"author.name": "John"}})
        assert "$.author.name" in result.params


class TestFTS5Sanitization:
    """Test FTS5 query sanitization."""

    def test_removes_special_chars(self, compiler):
        """Test that special FTS5 chars are removed."""
        result = compiler.compile({"match": {"title": 'hello "world" (test)'}})

        # Should not have special chars in FTS query
        assert '"' not in result.fts_query
        assert "(" not in result.fts_query
        assert ")" not in result.fts_query

    def test_removes_operators(self, compiler):
        """Test that FTS5 operators are removed."""
        result = compiler.compile({"match": {"title": "hello AND world OR test NOT this"}})

        # Operators should be removed
        assert " AND " not in result.fts_query
        assert " OR " not in result.fts_query
        assert " NOT " not in result.fts_query
