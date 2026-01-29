"""Tests for helper functions (bulk, scan, etc.)."""

from local_esearch import helpers


class TestBulk:
    """Test bulk operations."""

    def test_bulk_index(self, es):
        """Test bulk indexing."""
        actions = [
            {"_index": "test", "_id": "1", "_source": {"title": "Doc 1"}},
            {"_index": "test", "_id": "2", "_source": {"title": "Doc 2"}},
            {"_index": "test", "_id": "3", "_source": {"title": "Doc 3"}},
        ]

        success, errors = helpers.bulk(es, actions, stats_only=True)

        assert success == 3
        assert errors == 0

        # Verify documents exist
        assert es.exists(index="test", id="1")
        assert es.exists(index="test", id="2")
        assert es.exists(index="test", id="3")

    def test_bulk_mixed_operations(self, es):
        """Test bulk with mixed operations."""
        # First index some docs
        es.index(index="test", id="to-update", document={"title": "Original"})
        es.index(index="test", id="to-delete", document={"title": "Delete me"})

        actions = [
            {
                "_op_type": "index",
                "_index": "test",
                "_id": "new",
                "_source": {"title": "New"},
            },
            {
                "_op_type": "update",
                "_index": "test",
                "_id": "to-update",
                "_source": {"doc": {"title": "Updated"}},
            },
            {"_op_type": "delete", "_index": "test", "_id": "to-delete"},
        ]

        success, errors = helpers.bulk(es, actions, stats_only=True)

        assert success == 3
        assert errors == 0

        # Verify results
        assert es.exists(index="test", id="new")
        assert es.get(index="test", id="to-update")["_source"]["title"] == "Updated"
        assert not es.exists(index="test", id="to-delete")

    def test_bulk_create_conflict(self, es):
        """Test bulk create with existing document."""
        es.index(index="test", id="1", document={"title": "Existing"})

        actions = [
            {"_op_type": "create", "_index": "test", "_id": "1", "_source": {"title": "New"}},
        ]

        success, errors = helpers.bulk(es, actions, stats_only=True, raise_on_error=False)

        assert success == 0
        assert errors == 1


class TestScan:
    """Test scan iterator."""

    def test_scan_all(self, es):
        """Test scanning all documents."""
        # Index some documents
        for i in range(10):
            es.index(index="test", id=str(i), document={"num": i})

        docs = list(helpers.scan(es, index="test"))

        assert len(docs) == 10

    def test_scan_with_query(self, es):
        """Test scanning with a query filter."""
        for i in range(10):
            es.index(index="test", id=str(i), document={"num": i, "even": i % 2 == 0})

        docs = list(helpers.scan(es, index="test", query={"term": {"even": True}}))

        assert len(docs) == 5
        for doc in docs:
            assert doc["_source"]["even"] is True

    def test_scan_pagination(self, es):
        """Test that scan handles pagination correctly."""
        for i in range(25):
            es.index(index="test", id=str(i), document={"num": i})

        # Use small page size to test pagination
        docs = list(helpers.scan(es, index="test", size=10))

        assert len(docs) == 25

    def test_scan_source_filtering(self, es):
        """Test source filtering in scan."""
        for i in range(5):
            es.index(index="test", id=str(i), document={"title": f"Doc {i}", "body": "Content"})

        docs = list(helpers.scan(es, index="test", _source=["title"]))

        for doc in docs:
            assert "title" in doc["_source"]
            assert "body" not in doc["_source"]


class TestStreamingBulk:
    """Test streaming bulk operations."""

    def test_streaming_bulk(self, es):
        """Test streaming bulk indexing."""
        actions = [{"_index": "test", "_id": str(i), "_source": {"num": i}} for i in range(5)]

        results = list(helpers.streaming_bulk(es, actions))

        assert len(results) == 5
        assert all(success for success, _ in results)

    def test_streaming_bulk_with_errors(self, es):
        """Test streaming bulk with some failures."""
        es.index(index="test", id="existing", document={"title": "Existing"})

        actions = [
            {
                "_op_type": "create",
                "_index": "test",
                "_id": "new",
                "_source": {"title": "New"},
            },
            {
                "_op_type": "create",
                "_index": "test",
                "_id": "existing",
                "_source": {"title": "Conflict"},
            },
        ]

        results = list(helpers.streaming_bulk(es, actions, raise_on_exception=False))

        assert len(results) == 2
        assert results[0][0] is True  # First succeeded
        assert results[1][0] is False  # Second failed (conflict)
