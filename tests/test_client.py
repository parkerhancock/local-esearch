"""Tests for the main Elasticsearch client."""

import pytest
from local_esearch import ConflictError, Elasticsearch, NotFoundError


class TestDocumentOperations:
    """Test document CRUD operations."""

    def test_index_and_get(self, es):
        """Test indexing and retrieving a document."""
        result = es.index(index="test", id="1", document={"title": "Hello", "body": "World"})

        assert result["_index"] == "test"
        assert result["_id"] == "1"
        assert result["result"] == "created"
        assert result["_version"] == 1

        doc = es.get(index="test", id="1")
        assert doc["found"] is True
        assert doc["_source"]["title"] == "Hello"
        assert doc["_source"]["body"] == "World"

    def test_index_auto_id(self, es):
        """Test indexing with auto-generated ID."""
        result = es.index(index="test", document={"title": "Auto ID"})

        assert result["_index"] == "test"
        assert result["_id"] is not None
        assert len(result["_id"]) == 36  # UUID format

    def test_index_update_version(self, es):
        """Test that reindexing updates version."""
        es.index(index="test", id="1", document={"title": "V1"})
        result = es.index(index="test", id="1", document={"title": "V2"})

        assert result["result"] == "updated"
        assert result["_version"] == 2

    def test_index_create_op_type(self, es):
        """Test create op_type fails if document exists."""
        es.index(index="test", id="1", document={"title": "First"})

        with pytest.raises(ConflictError):
            es.index(index="test", id="1", document={"title": "Second"}, op_type="create")

    def test_get_not_found(self, es):
        """Test getting non-existent document raises NotFoundError."""
        with pytest.raises(NotFoundError):
            es.get(index="test", id="nonexistent")

    def test_exists(self, es):
        """Test document existence check."""
        assert es.exists(index="test", id="1") is False

        es.index(index="test", id="1", document={"title": "Exists"})
        assert es.exists(index="test", id="1") is True

    def test_delete(self, es):
        """Test document deletion."""
        es.index(index="test", id="1", document={"title": "Delete me"})
        assert es.exists(index="test", id="1") is True

        result = es.delete(index="test", id="1")
        assert result["result"] == "deleted"
        assert es.exists(index="test", id="1") is False

    def test_delete_not_found(self, es):
        """Test deleting non-existent document raises NotFoundError."""
        with pytest.raises(NotFoundError):
            es.delete(index="test", id="nonexistent")

    def test_update(self, es):
        """Test partial document update."""
        es.index(index="test", id="1", document={"title": "Original", "body": "Content"})

        result = es.update(index="test", id="1", body={"doc": {"title": "Updated"}})
        assert result["result"] == "updated"
        assert result["_version"] == 2

        doc = es.get(index="test", id="1")
        assert doc["_source"]["title"] == "Updated"
        assert doc["_source"]["body"] == "Content"  # Unchanged

    def test_update_not_found(self, es):
        """Test updating non-existent document raises NotFoundError."""
        with pytest.raises(NotFoundError):
            es.update(index="test", id="nonexistent", body={"doc": {"title": "X"}})

    def test_mget(self, es):
        """Test multi-get operation."""
        es.index(index="test", id="1", document={"title": "Doc 1"})
        es.index(index="test", id="2", document={"title": "Doc 2"})

        result = es.mget(
            docs=[
                {"_index": "test", "_id": "1"},
                {"_index": "test", "_id": "2"},
                {"_index": "test", "_id": "nonexistent"},
            ]
        )

        assert len(result["docs"]) == 3
        assert result["docs"][0]["found"] is True
        assert result["docs"][1]["found"] is True
        assert result["docs"][2]["found"] is False


class TestSearch:
    """Test search operations."""

    def test_search_match_all(self, es_with_docs):
        """Test match_all query."""
        response = es_with_docs.search(index="test", body={"query": {"match_all": {}}})

        assert response["hits"]["total"]["value"] == 5
        assert len(response["hits"]["hits"]) == 5

    def test_search_match(self, es_with_docs):
        """Test match query."""
        response = es_with_docs.search(index="test", body={"query": {"match": {"body": "python"}}})

        assert response["hits"]["total"]["value"] >= 1
        # Should find docs mentioning python
        titles = [h["_source"]["title"] for h in response["hits"]["hits"]]
        assert any("Python" in t for t in titles)

    def test_search_term(self, es_with_docs):
        """Test term query (exact match)."""
        response = es_with_docs.search(index="test", body={"query": {"term": {"category": "data"}}})

        assert response["hits"]["total"]["value"] == 2
        for hit in response["hits"]["hits"]:
            assert hit["_source"]["category"] == "data"

    def test_search_terms(self, es_with_docs):
        """Test terms query (match any)."""
        response = es_with_docs.search(
            index="test", body={"query": {"terms": {"category": ["data", "web"]}}}
        )

        assert response["hits"]["total"]["value"] == 3
        for hit in response["hits"]["hits"]:
            assert hit["_source"]["category"] in ["data", "web"]

    def test_search_bool_must(self, es_with_docs):
        """Test bool query with must clauses."""
        response = es_with_docs.search(
            index="test",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"category": "programming"}},
                        ]
                    }
                }
            },
        )

        assert response["hits"]["total"]["value"] == 2
        for hit in response["hits"]["hits"]:
            assert hit["_source"]["category"] == "programming"

    def test_search_bool_must_not(self, es_with_docs):
        """Test bool query with must_not clauses."""
        response = es_with_docs.search(
            index="test",
            body={
                "query": {
                    "bool": {
                        "must_not": [
                            {"term": {"category": "programming"}},
                        ]
                    }
                }
            },
        )

        assert response["hits"]["total"]["value"] == 3
        for hit in response["hits"]["hits"]:
            assert hit["_source"]["category"] != "programming"

    def test_search_range(self, es):
        """Test range query."""
        es.index(index="test", id="1", document={"title": "A", "price": 10})
        es.index(index="test", id="2", document={"title": "B", "price": 20})
        es.index(index="test", id="3", document={"title": "C", "price": 30})

        response = es.search(
            index="test", body={"query": {"range": {"price": {"gte": 15, "lte": 25}}}}
        )

        assert response["hits"]["total"]["value"] == 1
        assert response["hits"]["hits"][0]["_source"]["price"] == 20

    def test_search_exists(self, es):
        """Test exists query."""
        es.index(index="test", id="1", document={"title": "Has tag", "tag": "important"})
        es.index(index="test", id="2", document={"title": "No tag"})

        response = es.search(index="test", body={"query": {"exists": {"field": "tag"}}})

        assert response["hits"]["total"]["value"] == 1
        assert response["hits"]["hits"][0]["_source"]["title"] == "Has tag"

    def test_search_q_param(self, es_with_docs):
        """Test simple query string via q parameter."""
        response = es_with_docs.search(index="test", q="python")

        assert response["hits"]["total"]["value"] >= 1

    def test_search_pagination(self, es_with_docs):
        """Test search pagination with from and size."""
        # Get all with size 2
        page1 = es_with_docs.search(
            index="test", body={"query": {"match_all": {}}}, size=2, from_=0
        )
        page2 = es_with_docs.search(
            index="test", body={"query": {"match_all": {}}}, size=2, from_=2
        )

        assert len(page1["hits"]["hits"]) == 2
        assert len(page2["hits"]["hits"]) == 2

        # Pages should have different documents
        ids1 = {h["_id"] for h in page1["hits"]["hits"]}
        ids2 = {h["_id"] for h in page2["hits"]["hits"]}
        assert ids1.isdisjoint(ids2)

    def test_search_source_filtering(self, es_with_docs):
        """Test _source field filtering."""
        response = es_with_docs.search(
            index="test",
            body={"query": {"match_all": {}}},
            _source=["title"],
            size=1,
        )

        hit = response["hits"]["hits"][0]
        assert "title" in hit["_source"]
        assert "body" not in hit["_source"]

    def test_count(self, es_with_docs):
        """Test count query."""
        result = es_with_docs.count(index="test")
        assert result["count"] == 5

        result = es_with_docs.count(index="test", body={"query": {"term": {"category": "data"}}})
        assert result["count"] == 2


class TestIndices:
    """Test index management operations."""

    def test_create_and_exists(self, es):
        """Test index creation and existence check."""
        assert es.indices.exists("myindex") is False

        result = es.indices.create("myindex")
        assert result["acknowledged"] is True
        assert es.indices.exists("myindex") is True

    def test_create_with_mappings(self, es):
        """Test creating index with mappings."""
        es.indices.create(
            "myindex",
            mappings={
                "properties": {
                    "title": {"type": "text"},
                    "status": {"type": "keyword"},
                }
            },
        )

        info = es.indices.get("myindex")
        assert "title" in info["myindex"]["mappings"]["properties"]

    def test_delete_index(self, es):
        """Test index deletion."""
        es.indices.create("myindex")
        es.index(index="myindex", id="1", document={"title": "Test"})

        result = es.indices.delete("myindex")
        assert result["acknowledged"] is True
        assert es.indices.exists("myindex") is False

    def test_stats(self, es_with_docs):
        """Test index statistics."""
        stats = es_with_docs.indices.stats("test")

        assert stats["indices"]["test"]["primaries"]["docs"]["count"] == 5

    def test_put_mapping(self, es):
        """Test updating index mappings."""
        es.indices.create("myindex")
        es.indices.put_mapping("myindex", properties={"new_field": {"type": "keyword"}})

        mappings = es.indices.get_mapping("myindex")
        assert "new_field" in mappings["myindex"]["mappings"]["properties"]


class TestContextManager:
    """Test context manager behavior."""

    def test_context_manager(self):
        """Test using client as context manager."""
        with Elasticsearch(path=":memory:") as es:
            es.index(index="test", id="1", document={"title": "Test"})
            doc = es.get(index="test", id="1")
            assert doc["found"] is True
