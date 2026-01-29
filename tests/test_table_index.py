"""Tests for bolt-on table index functionality."""

import sqlite3

import pytest
from local_esearch import Elasticsearch, TableIndex


@pytest.fixture
def db_with_table():
    """Create an in-memory database with an existing table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
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
    conn.executemany(
        "INSERT INTO documents (id, title, content, category) VALUES (?, ?, ?, ?)",
        docs,
    )
    conn.commit()
    yield conn
    conn.close()


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
            conn=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()

        # Check FTS table exists
        cursor = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert cursor.fetchone() is not None

    def test_setup_creates_triggers(self, db_with_table):
        """Test that setup creates sync triggers."""
        index = TableIndex(
            conn=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()

        # Check triggers exist
        cursor = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
        triggers = [row[0] for row in cursor.fetchall()]
        assert "documents_fts_ai" in triggers
        assert "documents_fts_ad" in triggers
        assert "documents_fts_au" in triggers

    def test_reindex_populates_fts(self, db_with_table):
        """Test that reindex populates FTS5 table."""
        index = TableIndex(
            conn=db_with_table,
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
            conn=db_with_table,
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
            conn=db_with_table,
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
            conn=db_with_table,
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
            conn=db_with_table,
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
            conn=db_with_table,
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

    def test_stats(self, db_with_table):
        """Test stats returns correct counts."""
        index = TableIndex(
            conn=db_with_table,
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
            conn=db_with_table,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
        )
        index.setup()
        index.reindex()
        index.drop()

        # FTS table should be gone
        cursor = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert cursor.fetchone() is None

        # Triggers should be gone
        cursor = db_with_table.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'documents_fts%'"
        )
        assert cursor.fetchall() == []


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
            conn=db_with_table,
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
            conn=db_with_table,
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
            conn=db_with_table,
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
        cursor = es._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_fts'"
        )
        assert cursor.fetchone() is not None

        # Unregister with drop
        es.unregister_table("docs", drop_indexes=True)

        # FTS table should be gone
        cursor = es._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_fts'"
        )
        assert cursor.fetchone() is None

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
