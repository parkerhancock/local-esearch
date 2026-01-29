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

    def test_streaming_bulk_update(self, es):
        """Test streaming bulk with update operation."""
        es.index(index="test", id="1", document={"title": "Original"})

        actions = [
            {
                "_op_type": "update",
                "_index": "test",
                "_id": "1",
                "_source": {"title": "Updated"},  # Direct source, not {"doc": ...}
            },
        ]

        results = list(helpers.streaming_bulk(es, actions))
        assert len(results) == 1
        assert results[0][0] is True

        # Verify update
        doc = es.get(index="test", id="1")
        assert doc["_source"]["title"] == "Updated"

    def test_streaming_bulk_delete(self, es):
        """Test streaming bulk with delete operation."""
        es.index(index="test", id="1", document={"title": "To delete"})

        actions = [{"_op_type": "delete", "_index": "test", "_id": "1"}]

        results = list(helpers.streaming_bulk(es, actions))
        assert len(results) == 1
        assert results[0][0] is True
        assert not es.exists(index="test", id="1")

    def test_streaming_bulk_unknown_op_type(self, es):
        """Test streaming bulk with unknown operation type."""
        actions = [
            {"_op_type": "unknown", "_index": "test", "_id": "1", "_source": {"title": "Test"}}
        ]

        results = list(helpers.streaming_bulk(es, actions, raise_on_exception=False))
        assert len(results) == 1
        assert results[0][0] is False  # Failed
        assert "Unknown op_type" in results[0][1]["unknown"]["error"]["reason"]


class TestBulkEdgeCases:
    """Test bulk edge cases and error handling."""

    def test_bulk_missing_index(self, es):
        """Test bulk with missing index field."""
        actions = [
            {"_id": "1", "_source": {"title": "No index"}},
        ]

        success, errors = helpers.bulk(es, actions, stats_only=False, raise_on_error=False)

        assert success == 0
        assert len(errors) == 1
        assert "index is missing" in str(errors[0])

    def test_bulk_unknown_op_type(self, es):
        """Test bulk with unknown operation type."""
        actions = [
            {
                "_op_type": "invalid",
                "_index": "test",
                "_id": "1",
                "_source": {"title": "Test"},
            }
        ]

        success, errors = helpers.bulk(es, actions, stats_only=False, raise_on_error=False)

        assert success == 0
        assert len(errors) == 1
        assert "Unknown op_type" in str(errors[0])

    def test_bulk_with_refresh(self, es):
        """Test bulk with refresh=True."""
        actions = [
            {"_index": "test", "_id": "1", "_source": {"title": "Doc 1"}},
        ]

        success, errors = helpers.bulk(es, actions, refresh=True, stats_only=True)

        assert success == 1
        assert errors == 0

    def test_bulk_stats_only_false(self, es):
        """Test bulk with stats_only=False returns error list."""
        actions = [
            {"_index": "test", "_id": "1", "_source": {"title": "Doc 1"}},
        ]

        success, errors = helpers.bulk(es, actions, stats_only=False)

        assert success == 1
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_bulk_update_with_doc_wrapper(self, es):
        """Test bulk update with {"doc": ...} wrapper."""
        es.index(index="test", id="1", document={"title": "Original", "count": 0})

        actions = [
            {
                "_op_type": "update",
                "_index": "test",
                "_id": "1",
                "_source": {"doc": {"title": "Updated"}},  # With doc wrapper
            },
        ]

        success, errors = helpers.bulk(es, actions, stats_only=True)
        assert success == 1

        doc = es.get(index="test", id="1")
        assert doc["_source"]["title"] == "Updated"


class TestParallelBulk:
    """Test parallel bulk operations."""

    def test_parallel_bulk(self, es):
        """Test parallel bulk (sequential in SQLite)."""
        actions = [{"_index": "test", "_id": str(i), "_source": {"num": i}} for i in range(5)]

        results = list(helpers.parallel_bulk(es, actions))

        assert len(results) == 5
        assert all(success for success, _ in results)

    def test_parallel_bulk_with_errors(self, es):
        """Test parallel bulk handles errors."""
        es.index(index="test", id="existing", document={"title": "Existing"})

        actions = [
            {
                "_op_type": "create",
                "_index": "test",
                "_id": "existing",
                "_source": {"title": "Dup"},
            },
        ]

        # parallel_bulk passes raise_on_error to streaming_bulk as raise_on_exception
        results = list(helpers.parallel_bulk(es, actions, raise_on_error=False))

        assert len(results) == 1
        assert results[0][0] is False
