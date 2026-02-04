"""Tests for bolt-on table index functionality."""

import sqlite3

import pytest
from local_esearch import Elasticsearch, TableIndex
from local_esearch.backends.sqlite import SQLiteBackend


class MockEmbeddingBackend:
    """Mock embedding backend for testing vector operations."""

    backend_name = "mock"
    dimensions = 8
    model_name = "mock-test-model"

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text hash."""
        h = hash(text) % 1000
        return [(h + i) / 1000.0 for i in range(self.dimensions)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def db_with_table():
    """Create an in-memory database backend with an existing table."""
    backend = SQLiteBackend(":memory:")
    backend.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT
        )
    """)
    # Insert sample data
    docs = [
        (1, "Python Programming", "Learn Python basics and advanced topics", "tech"),
        (2, "JavaScript Guide", "Modern JavaScript development patterns", "tech"),
        (3, "Data Science", "Python for data analysis and machine learning", "data"),
        (4, "Machine Learning", "Introduction to ML algorithms and neural networks", "data"),
        (5, "Web Development", "Building web apps with JavaScript and React", "web"),
    ]
    backend.executemany(
        "INSERT INTO documents (id, title, content, category) VALUES (?, ?, ?, ?)",
        docs,
    )
    backend.commit()
    yield backend
    backend.close()


@pytest.fixture
def es_with_table(tmp_path):
    """Create ES client with an existing user table."""
    db_path = tmp_path / "test.db"

    # First, create the user's table directly
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT
        )
    """)
    docs = [
        (1, "Python Programming", "Learn Python basics and advanced topics", "tech"),
        (2, "JavaScript Guide", "Modern JavaScript development patterns", "tech"),
        (3, "Data Science", "Python for data analysis and machine learning", "data"),
        (4, "Machine Learning", "Introduction to ML algorithms", "data"),
        (5, "Web Development", "Building web apps with JavaScript and React", "web"),
    ]
    conn.executemany(
        "INSERT INTO documents (id, title, content, category) VALUES (?, ?, ?, ?)",
        docs,
    )
    conn.commit()
    conn.close()

    # Now connect with ES client (will add our schema tables)
    es = Elasticsearch(path=str(db_path))

    # Register the existing table
    es.register_table(
        index="docs",
        table="documents",
        id_column="id",
        text_columns=["title", "content"],
    )

    yield es
    es.close()


class TestTableIndex:
    """Test TableIndex class directly."""

    def test_setup_creates_fts_table(self, db_with_table):
        """Test that setup creates FTS5 table."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()

        # Check FTS table exists
        rows = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert len(rows) > 0

    def test_setup_creates_triggers(self, db_with_table):
        """Test that setup creates sync triggers."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()

        # Check triggers exist
        rows = db_with_table.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        triggers = [row["name"] for row in rows]
        assert "documents_fts_ai" in triggers
        assert "documents_fts_ad" in triggers
        assert "documents_fts_au" in triggers

    def test_reindex_populates_fts(self, db_with_table):
        """Test that reindex populates FTS5 table."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        stats = index.reindex()

        assert stats["fts_indexed"] == 5

    def test_keyword_search(self, db_with_table):
        """Test keyword search returns matching rows."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        results = index.search("python", mode="keyword")

        assert len(results) >= 1
        # Should find Python Programming and Data Science docs
        result_ids = [r["id"] for r in results]
        assert 1 in result_ids or 3 in result_ids

    def test_search_with_limit(self, db_with_table):
        """Test search respects limit parameter."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        results = index.search("development", mode="keyword", limit=1)
        assert len(results) <= 1

    def test_fts_sync_on_insert(self, db_with_table):
        """Test FTS5 updates automatically on insert."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        # Insert new document
        db_with_table.execute(
            "INSERT INTO documents (id, title, content, category) VALUES (?, ?, ?, ?)",
            (6, "Rust Programming", "Systems programming with Rust language", "tech"),
        )
        db_with_table.commit()

        # Should find it immediately without reindex
        results = index.search("rust", mode="keyword")
        assert len(results) == 1
        assert results[0]["id"] == 6

    def test_fts_sync_on_update(self, db_with_table):
        """Test FTS5 updates automatically on update."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        # Update existing document
        db_with_table.execute(
            "UPDATE documents SET content = ? WHERE id = ?",
            ("Learn Golang basics", 1),
        )
        db_with_table.commit()

        # Should find by new content
        results = index.search("golang", mode="keyword")
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_fts_sync_on_delete(self, db_with_table):
        """Test FTS5 updates automatically on delete."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        # Verify document is searchable
        results = index.search("python programming", mode="keyword")
        assert any(r["id"] == 1 for r in results)

        # Delete document
        db_with_table.execute("DELETE FROM documents WHERE id = ?", (1,))
        db_with_table.commit()

        # Should no longer find it
        results = index.search("python programming", mode="keyword")
        assert not any(r["id"] == 1 for r in results)

    def test_chunks_and_vec_cleanup_on_delete(self, db_with_table):
        """Test chunks and vector entries are deleted when parent row is deleted."""
        # Check if sqlite-vec is available in the backend
        if not db_with_table.vector_available():
            pytest.skip("sqlite-vec not available")

        backend = MockEmbeddingBackend()
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
            embedding_backend=backend,
            chunk_size=10,  # Small chunks for test
            chunk_overlap=2,
        )
        index.setup()
        index.reindex()

        # Verify chunks exist for row 1
        rows = db_with_table.execute(
            "SELECT COUNT(*) as cnt FROM documents_chunks WHERE row_id = ?", (1,)
        )
        chunks_before = rows[0]["cnt"]
        assert chunks_before > 0, "Expected chunks for row 1"

        # Verify vector entries exist for row 1 (format: "row_id:chunk_idx")
        rows = db_with_table.execute(
            "SELECT COUNT(*) as cnt FROM documents_vec WHERE chunk_id LIKE '1:%'"
        )
        vec_before = rows[0]["cnt"]
        assert vec_before > 0, "Expected vector entries for row 1"

        # Delete the parent row
        db_with_table.execute("DELETE FROM documents WHERE id = ?", (1,))
        db_with_table.commit()

        # Verify chunks are gone
        rows = db_with_table.execute(
            "SELECT COUNT(*) as cnt FROM documents_chunks WHERE row_id = ?", (1,)
        )
        chunks_after = rows[0]["cnt"]
        assert chunks_after == 0, "Chunks should be deleted"

        # Verify vector entries are gone
        rows = db_with_table.execute(
            "SELECT COUNT(*) as cnt FROM documents_vec WHERE chunk_id LIKE '1:%'"
        )
        vec_after = rows[0]["cnt"]
        assert vec_after == 0, "Vector entries should be deleted"

        # Verify other rows' data is still intact
        rows = db_with_table.execute(
            "SELECT COUNT(*) as cnt FROM documents_chunks WHERE row_id = ?", (2,)
        )
        other_chunks = rows[0]["cnt"]
        assert other_chunks > 0, "Other rows should still have chunks"

    def test_stats(self, db_with_table):
        """Test stats returns correct counts."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()

        stats = index.stats()
        assert stats["table"] == "documents"
        assert stats["total_rows"] == 5
        assert stats["fts_indexed"] == 5

    def test_drop_removes_everything(self, db_with_table):
        """Test drop removes FTS table and triggers."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()
        index.drop()

        # FTS table should be gone
        rows = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert len(rows) == 0

        # Triggers should be gone
        rows = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'documents_fts%'"
        )
        assert len(rows) == 0


class TestElasticsearchWithTable:
    """Test ES client with registered tables."""

    def test_register_table(self, tmp_path):
        """Test registering an existing table."""
        db_path = tmp_path / "test.db"

        # Create existing table first
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT
            )
        """)
        conn.execute("INSERT INTO documents VALUES (1, 'Test', 'Content')")
        conn.commit()
        conn.close()

        # Connect with ES client
        es = Elasticsearch(path=str(db_path))

        table_index = es.register_table(
            index="docs",
            table="documents",
            text_columns=["title", "content"],
        )

        assert table_index is not None
        assert "docs" in es._table_indexes
        es.close()

    def test_search_registered_table(self, es_with_table):
        """Test searching a registered table via ES API."""
        es_with_table.indices.reindex("docs")

        response = es_with_table.search(index="docs", q="python")

        assert response["hits"]["total"]["value"] >= 1
        hits = response["hits"]["hits"]
        assert all(h["_index"] == "docs" for h in hits)

    def test_search_with_body(self, es_with_table):
        """Test searching with query body."""
        es_with_table.indices.reindex("docs")

        response = es_with_table.search(
            index="docs",
            body={"query": {"match": {"content": "machine learning"}}},
        )

        assert response["hits"]["total"]["value"] >= 1

    def test_reindex_via_indices(self, es_with_table):
        """Test reindex through indices client."""
        result = es_with_table.indices.reindex("docs")

        assert result["acknowledged"] is True
        assert result["index"] == "docs"
        assert result["stats"]["fts_indexed"] == 5

    def test_get_table_index(self, es_with_table):
        """Test getting table index instance."""
        table_index = es_with_table.get_table_index("docs")
        assert table_index is not None

        # Non-existent returns None
        assert es_with_table.get_table_index("nonexistent") is None


class TestEmbeddingText:
    """Test different embedding_text configurations."""

    def test_embedding_text_as_column_name(self, db_with_table):
        """Test using a column name for embedding text."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
            embedding_text="content",  # Just use content column
        )

        row = {"id": 1, "title": "Test", "content": "Hello world", "category": "test"}
        text = index._get_embedding_text(row)
        assert text == "Hello world"

    def test_embedding_text_as_callable(self, db_with_table):
        """Test using a callable for embedding text."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
            embedding_text=lambda row: f"{row['title']}: {row['content']}",
        )

        row = {"id": 1, "title": "Test", "content": "Hello world", "category": "test"}
        text = index._get_embedding_text(row)
        assert text == "Test: Hello world"

    def test_embedding_text_default_concatenation(self, db_with_table):
        """Test default concatenation of text columns."""
        index = TableIndex(
            db_backend=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
            embedding_text=None,  # Default
        )

        row = {"id": 1, "title": "Test", "content": "Hello world", "category": "test"}
        text = index._get_embedding_text(row)
        assert text == "Test Hello world"


class TestPersistence:
    """Test that table registrations persist across reconnection."""

    def test_registration_persists(self, tmp_path):
        """Test that register_table survives reconnection."""
        db_path = tmp_path / "test.db"

        # First connection: create table and register
        conn1 = sqlite3.connect(str(db_path))
        conn1.execute("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                body TEXT
            )
        """)
        conn1.execute("INSERT INTO articles VALUES (1, 'Test', 'Content')")
        conn1.commit()
        conn1.close()

        # Register with ES client
        es1 = Elasticsearch(path=str(db_path))
        es1.register_table(
            index="articles",
            table="articles",
            text_columns=["title", "body"],
        )
        es1.indices.reindex("articles")
        es1.close()

        # Second connection: should auto-load registration
        es2 = Elasticsearch(path=str(db_path))

        # Should already be registered
        assert "articles" in es2._table_indexes

        # Should be searchable without re-registering
        response = es2.search(index="articles", q="test")
        assert response["hits"]["total"]["value"] >= 1

        es2.close()

    def test_unregister_table(self, tmp_path):
        """Test unregistering a table."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, text TEXT)")
        conn.commit()
        conn.close()

        es = Elasticsearch(path=str(db_path))
        es.register_table(index="docs", table="docs", text_columns=["text"])

        assert "docs" in es._table_indexes

        # Unregister
        result = es.unregister_table("docs")
        assert result is True
        assert "docs" not in es._table_indexes

        # Unregister again returns False
        result = es.unregister_table("docs")
        assert result is False

        es.close()

    def test_unregister_with_drop(self, tmp_path):
        """Test unregistering with drop_indexes=True removes FTS tables."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, text TEXT)")
        conn.commit()
        conn.close()

        es = Elasticsearch(path=str(db_path))
        es.register_table(index="docs", table="docs", text_columns=["text"])

        # Verify FTS table exists
        rows = es._backend.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_fts'"
        )
        assert len(rows) > 0

        # Unregister with drop
        es.unregister_table("docs", drop_indexes=True)

        # FTS table should be gone
        rows = es._backend.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_fts'"
        )
        assert len(rows) == 0

        es.close()

    def test_multiple_registrations_persist(self, tmp_path):
        """Test multiple table registrations all persist."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT)")
        conn.commit()
        conn.close()

        # Register multiple tables
        es1 = Elasticsearch(path=str(db_path))
        es1.register_table(index="users", table="users", text_columns=["name"])
        es1.register_table(index="posts", table="posts", text_columns=["title"])
        es1.close()

        # Reconnect and verify both loaded
        es2 = Elasticsearch(path=str(db_path))
        assert "users" in es2._table_indexes
        assert "posts" in es2._table_indexes
        es2.close()


class TestMultiTableSearch:
    """Test searching across multiple registered tables."""

    @pytest.fixture
    def es_multi_table(self, tmp_path):
        """Create ES client with multiple registered tables."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        # Create emails table
        conn.execute("""
            CREATE TABLE emails (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                body TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO emails (id, subject, body) VALUES (?, ?, ?)",
            [
                (1, "Meeting about Python project", "Let's discuss the Python implementation"),
                (2, "JavaScript review", "Please review the JS code changes"),
                (3, "Quarterly report", "Attached is the quarterly financial report"),
            ],
        )
        # Create sessions table
        conn.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                project TEXT,
                summary TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO sessions (id, project, summary) VALUES (?, ?, ?)",
            [
                (1, "api-server", "Fixed Python async bug in the API"),
                (2, "web-app", "Implemented JavaScript animations"),
                (3, "data-pipeline", "Python data processing optimization"),
            ],
        )
        conn.commit()
        conn.close()

        es = Elasticsearch(path=str(db_path))
        es.register_table(
            index="emails",
            table="emails",
            text_columns=["subject", "body"],
        )
        es.register_table(
            index="sessions",
            table="sessions",
            text_columns=["project", "summary"],
        )
        es.indices.reindex("emails")
        es.indices.reindex("sessions")

        yield es
        es.close()

    def test_search_multiple_registered_tables(self, es_multi_table):
        """Test searching across multiple registered tables."""
        response = es_multi_table.search(
            index=["emails", "sessions"],
            q="python",
        )

        # Should find results from both tables
        hits = response["hits"]["hits"]
        assert len(hits) >= 2

        indices_found = {h["_index"] for h in hits}
        assert "emails" in indices_found
        assert "sessions" in indices_found

    def test_search_single_registered_table(self, es_multi_table):
        """Test searching a single registered table still works."""
        response = es_multi_table.search(index="emails", q="python")

        hits = response["hits"]["hits"]
        assert len(hits) >= 1
        assert all(h["_index"] == "emails" for h in hits)

    def test_search_all_indices_includes_registered_tables(self, es_multi_table):
        """Test that searching with no index includes registered tables."""
        # Also add a regular index document
        es_multi_table.index(
            index="docs",
            id="1",
            document={"title": "Python guide", "content": "Learn Python basics"},
        )

        response = es_multi_table.search(q="python")

        hits = response["hits"]["hits"]
        indices_found = {h["_index"] for h in hits}

        # Should find results from registered tables and regular index
        assert "emails" in indices_found or "sessions" in indices_found
        assert "docs" in indices_found

    def test_search_mixed_indices(self, es_multi_table):
        """Test searching mix of registered tables and regular indices."""
        # Add a regular index document
        es_multi_table.index(
            index="docs",
            id="1",
            document={"title": "Python tutorial"},
        )

        response = es_multi_table.search(
            index=["emails", "docs"],
            q="python",
        )

        hits = response["hits"]["hits"]
        indices_found = {h["_index"] for h in hits}

        assert "emails" in indices_found
        assert "docs" in indices_found

    def test_multi_table_total_count(self, es_multi_table):
        """Test that total count includes results from all tables."""
        response = es_multi_table.search(
            index=["emails", "sessions"],
            q="python",
        )

        # Total should reflect documents from both tables
        total = response["hits"]["total"]["value"]
        assert total >= 2  # At least 1 from emails + 2 from sessions
