"""Tests for PostgreSQL backend.

These tests require a running PostgreSQL server with pgvector extension.
Skip if PostgreSQL is not available.
"""

import os

import pytest

# Check if PostgreSQL is available
try:
    import psycopg

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Get DSN from environment or use default local connection
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://parkerhancock@localhost/local_esearch_test"
)


def can_connect_postgres() -> bool:
    """Check if we can connect to PostgreSQL."""
    if not POSTGRES_AVAILABLE:
        return False
    try:
        import psycopg

        conn = psycopg.connect(POSTGRES_DSN, autocommit=True)
        conn.close()
        return True
    except Exception:
        return False


# Skip all tests in this module if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not can_connect_postgres(),
    reason="PostgreSQL not available or cannot connect",
)


@pytest.fixture
def pg_es():
    """Create Elasticsearch instance with PostgreSQL backend."""
    from local_esearch import Elasticsearch

    es = Elasticsearch(path=POSTGRES_DSN)

    # Clean up any existing test indices
    for idx in ["test", "articles", "docs"]:
        if es.indices.exists(idx):
            es.indices.delete(idx)

    yield es

    # Cleanup
    for idx in ["test", "articles", "docs"]:
        try:
            if es.indices.exists(idx):
                es.indices.delete(idx)
        except Exception:
            pass
    es.close()


@pytest.fixture
def pg_table(pg_es):
    """Create a test table for addon mode testing."""
    # Create a user table
    pg_es._backend.execute("DROP TABLE IF EXISTS test_articles CASCADE")
    pg_es._backend.execute(
        """
        CREATE TABLE test_articles (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT
        )
    """
    )
    pg_es._backend.execute(
        """
        INSERT INTO test_articles (title, content) VALUES
        ('Python Guide', 'Learn Python programming basics'),
        ('ML Tutorial', 'Neural networks and deep learning'),
        ('Web Dev', 'Building modern web applications')
    """
    )
    pg_es._backend.commit()

    yield pg_es

    # Cleanup
    try:
        pg_es.unregister_table("test_articles", drop_indexes=True)
    except Exception:
        pass
    pg_es._backend.execute("DROP TABLE IF EXISTS test_articles CASCADE")
    pg_es._backend.commit()


class TestPostgresBackendBasics:
    """Test basic PostgreSQL backend functionality."""

    def test_backend_type(self, pg_es):
        """Test that we're using PostgreSQL backend."""
        from local_esearch.backends.postgres import PostgresBackend

        assert isinstance(pg_es._backend, PostgresBackend)

    def test_vector_available(self, pg_es):
        """Test that pgvector is available."""
        assert pg_es._backend.vector_available() is True


class TestPostgresIndexOperations:
    """Test index operations with PostgreSQL."""

    def test_create_index(self, pg_es):
        """Test index creation."""
        result = pg_es.indices.create("test")
        assert result["acknowledged"] is True
        assert pg_es.indices.exists("test") is True

    def test_delete_index(self, pg_es):
        """Test index deletion."""
        pg_es.indices.create("test")
        result = pg_es.indices.delete("test")
        assert result["acknowledged"] is True
        assert pg_es.indices.exists("test") is False


class TestPostgresDocumentCRUD:
    """Test document CRUD operations with PostgreSQL."""

    def test_index_and_get(self, pg_es):
        """Test indexing and retrieving a document."""
        pg_es.index(
            index="test", id="1", document={"title": "Hello", "body": "World"}
        )

        doc = pg_es.get(index="test", id="1")
        assert doc["found"] is True
        assert doc["_source"]["title"] == "Hello"

    def test_update(self, pg_es):
        """Test document update."""
        pg_es.index(index="test", id="1", document={"title": "Original"})
        pg_es.update(index="test", id="1", doc={"title": "Updated"})

        doc = pg_es.get(index="test", id="1")
        assert doc["_source"]["title"] == "Updated"

    def test_delete(self, pg_es):
        """Test document deletion."""
        pg_es.index(index="test", id="1", document={"title": "ToDelete"})
        pg_es.delete(index="test", id="1")

        assert pg_es.exists(index="test", id="1") is False


class TestPostgresSearch:
    """Test search functionality with PostgreSQL."""

    def test_keyword_search(self, pg_es):
        """Test keyword search using tsvector."""
        pg_es.index(
            index="test",
            id="1",
            document={"title": "Python Programming", "body": "Learn Python basics"},
        )
        pg_es.index(
            index="test",
            id="2",
            document={"title": "JavaScript", "body": "Modern JS"},
        )

        response = pg_es.search(index="test", q="python")

        assert response["hits"]["total"]["value"] == 1
        assert response["hits"]["hits"][0]["_id"] == "1"

    def test_match_query(self, pg_es):
        """Test match query DSL."""
        pg_es.index(index="test", id="1", document={"title": "Hello World"})
        pg_es.index(index="test", id="2", document={"title": "Goodbye World"})

        response = pg_es.search(
            index="test", body={"query": {"match": {"title": "hello"}}}
        )

        assert response["hits"]["total"]["value"] == 1
        assert response["hits"]["hits"][0]["_id"] == "1"

    def test_bool_query_with_term(self, pg_es):
        """Test bool query with term (non-FTS) must_not."""
        pg_es.index(
            index="test", id="1", document={"title": "Python", "category": "web"}
        )
        pg_es.index(
            index="test", id="2", document={"title": "Python", "category": "ml"}
        )
        pg_es.index(
            index="test", id="3", document={"title": "JavaScript", "category": "web"}
        )

        # FTS must_not doesn't work correctly (known limitation)
        # Use term query for must_not instead
        response = pg_es.search(
            index="test",
            body={
                "query": {
                    "bool": {
                        "must": [{"match": {"title": "python"}}],
                        "must_not": [{"term": {"category": "web"}}],
                    }
                }
            },
        )

        assert response["hits"]["total"]["value"] == 1
        assert response["hits"]["hits"][0]["_id"] == "2"

    def test_count(self, pg_es):
        """Test document counting."""
        pg_es.index(index="test", id="1", document={"title": "Doc 1"})
        pg_es.index(index="test", id="2", document={"title": "Doc 2"})
        pg_es.index(index="test", id="3", document={"title": "Doc 3"})

        result = pg_es.count(index="test")
        assert result["count"] == 3


class TestPostgresAddonMode:
    """Test addon mode (register_table) with PostgreSQL."""

    def test_register_table(self, pg_table):
        """Test registering an existing table."""
        pg_table.register_table(
            index="test_articles",
            table="test_articles",
            id_column="id",
            text_columns=["title", "content"],
        )

        assert "test_articles" in pg_table._table_indexes

    def test_reindex(self, pg_table):
        """Test reindexing a registered table."""
        pg_table.register_table(
            index="test_articles",
            table="test_articles",
            id_column="id",
            text_columns=["title", "content"],
        )

        result = pg_table.indices.reindex("test_articles")

        assert result["stats"]["fts_indexed"] == 3

    def test_search_registered_table(self, pg_table):
        """Test searching a registered table."""
        pg_table.register_table(
            index="test_articles",
            table="test_articles",
            id_column="id",
            text_columns=["title", "content"],
        )
        pg_table.indices.reindex("test_articles")

        response = pg_table.search(index="test_articles", q="python")

        assert response["hits"]["total"]["value"] == 1

    def test_search_neural(self, pg_table):
        """Test searching for different term."""
        pg_table.register_table(
            index="test_articles",
            table="test_articles",
            id_column="id",
            text_columns=["title", "content"],
        )
        pg_table.indices.reindex("test_articles")

        response = pg_table.search(index="test_articles", q="neural")

        assert response["hits"]["total"]["value"] == 1

    def test_unregister_table(self, pg_table):
        """Test unregistering a table."""
        pg_table.register_table(
            index="test_articles",
            table="test_articles",
            id_column="id",
            text_columns=["title", "content"],
        )
        pg_table.indices.reindex("test_articles")

        result = pg_table.unregister_table("test_articles", drop_indexes=True)

        assert result is True
        assert "test_articles" not in pg_table._table_indexes


class TestPostgresMultiIndex:
    """Test multi-index operations."""

    def test_search_multiple_indices(self, pg_es):
        """Test searching across multiple indices."""
        pg_es.index(
            index="docs", id="1", document={"title": "Python Docs", "body": "Python"}
        )
        pg_es.index(
            index="articles",
            id="1",
            document={"title": "Python Article", "body": "Python"},
        )

        response = pg_es.search(index=["docs", "articles"], q="python")

        assert response["hits"]["total"]["value"] == 2
