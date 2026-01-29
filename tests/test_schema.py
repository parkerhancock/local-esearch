"""Tests for schema management."""

import sqlite3

import pytest
from local_esearch.schema import (
    SCHEMA_VERSION,
    ensure_vector_table,
    extract_text_fields,
    init_database,
    vector_schema,
)


class TestExtractTextFields:
    """Test text extraction from documents."""

    def test_simple_strings(self):
        """Extract simple string fields."""
        source = {"title": "Hello", "body": "World"}
        result = extract_text_fields(source)
        assert "Hello" in result
        assert "World" in result

    def test_nested_objects(self):
        """Extract text from nested objects."""
        source = {
            "title": "Doc",
            "metadata": {"author": "Alice", "tags": ["python", "search"]},
        }
        result = extract_text_fields(source)
        assert "Doc" in result
        assert "Alice" in result
        assert "python" in result
        assert "search" in result

    def test_lists_of_strings(self):
        """Extract text from lists."""
        source = {"tags": ["one", "two", "three"]}
        result = extract_text_fields(source)
        assert "one" in result
        assert "two" in result
        assert "three" in result

    def test_mixed_types_ignored(self):
        """Non-string values are ignored."""
        source = {"title": "Test", "count": 42, "active": True, "data": None}
        result = extract_text_fields(source)
        assert result == "Test"

    def test_with_mappings_filters_fields(self):
        """Mappings filter to only text fields."""
        source = {"title": "Hello", "body": "World", "category": "tech"}
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "body": {"type": "text"},
                "category": {"type": "keyword"},  # Not text type
            }
        }
        result = extract_text_fields(source, mappings)
        assert "Hello" in result
        assert "World" in result
        assert "tech" not in result

    def test_with_mappings_missing_field(self):
        """Mappings handle missing fields gracefully."""
        source = {"title": "Hello"}
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "body": {"type": "text"},  # Not in source
            }
        }
        result = extract_text_fields(source, mappings)
        assert result == "Hello"

    def test_empty_document(self):
        """Empty document returns empty string."""
        result = extract_text_fields({})
        assert result == ""

    def test_deeply_nested(self):
        """Handle deeply nested structures."""
        source = {
            "level1": {
                "level2": {
                    "level3": {"text": "deep value"},
                },
            },
        }
        result = extract_text_fields(source)
        assert "deep value" in result


class TestVectorSchema:
    """Test vector schema generation."""

    def test_generates_correct_dimensions(self):
        """Schema has correct dimension placeholder."""
        schema = vector_schema(1024)
        assert "float[1024]" in schema

    def test_different_dimensions(self):
        """Schema works with different dimensions."""
        assert "float[768]" in vector_schema(768)
        assert "float[1536]" in vector_schema(1536)


class TestInitDatabase:
    """Test database initialization."""

    def test_creates_core_tables(self):
        """Init creates all core tables."""
        conn = sqlite3.connect(":memory:")
        init_database(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        assert "_documents" in tables
        assert "_documents_fts" in tables
        assert "_indices" in tables
        assert "_schema_version" in tables
        assert "_table_indexes" in tables

    def test_records_schema_version(self):
        """Init records schema version."""
        conn = sqlite3.connect(":memory:")
        init_database(conn)

        cursor = conn.execute("SELECT version FROM _schema_version")
        version = cursor.fetchone()[0]
        assert version == SCHEMA_VERSION

    def test_idempotent(self):
        """Multiple init calls are safe."""
        conn = sqlite3.connect(":memory:")
        init_database(conn)
        init_database(conn)  # Should not error

        cursor = conn.execute("SELECT COUNT(*) FROM _schema_version")
        count = cursor.fetchone()[0]
        assert count == 1

    def test_with_embedding_backend_no_vec(self):
        """Init with embedding backend but no sqlite-vec available."""

        class MockBackend:
            dimensions = 8

        conn = sqlite3.connect(":memory:")
        # sqlite-vec not loaded, so vec_version() will fail
        init_database(conn, embedding_backend=MockBackend())

        # Should still work, just no vector table
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "_documents_vec" not in tables


class TestEnsureVectorTable:
    """Test ensure_vector_table function."""

    def test_returns_false_without_vec(self):
        """Returns False when sqlite-vec not available."""
        conn = sqlite3.connect(":memory:")
        result = ensure_vector_table(conn, 8)
        assert result is False

    def test_with_sqlite_vec(self):
        """Returns True and creates table when sqlite-vec available."""
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        result = ensure_vector_table(conn, 8)
        assert result is True

        # Table should exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_documents_vec'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent_with_vec(self):
        """Multiple calls are safe."""
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        assert ensure_vector_table(conn, 8) is True
        assert ensure_vector_table(conn, 8) is True  # Second call
