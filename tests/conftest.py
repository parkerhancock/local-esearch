"""Pytest fixtures for local_esearch tests."""

import pytest
from local_esearch import Elasticsearch


class MockEmbeddingBackend:
    """Mock embedding backend for testing."""

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
def mock_backend():
    """Create a mock embedding backend."""
    return MockEmbeddingBackend()


@pytest.fixture
def es():
    """Create an in-memory Elasticsearch client."""
    client = Elasticsearch(path=":memory:")
    yield client
    client.close()


@pytest.fixture
def es_with_embeddings(tmp_path, mock_backend):
    """Create ES client with mock embedding backend and sqlite-vec."""
    import sqlite3

    db_path = tmp_path / "test_vec.db"
    conn = sqlite3.connect(str(db_path))

    # Try to load sqlite-vec
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.close()
    except Exception:
        conn.close()
        pytest.skip("sqlite-vec not available")

    client = Elasticsearch(path=str(db_path), embedding_backend=mock_backend)
    yield client
    client.close()


@pytest.fixture
def es_with_docs(es):
    """Create client with sample documents."""
    docs = [
        {
            "id": "1",
            "title": "Python Programming",
            "body": "Learn Python basics",
            "category": "programming",
        },
        {
            "id": "2",
            "title": "JavaScript Guide",
            "body": "Modern JavaScript development",
            "category": "programming",
        },
        {
            "id": "3",
            "title": "Data Science",
            "body": "Python for data analysis",
            "category": "data",
        },
        {
            "id": "4",
            "title": "Machine Learning",
            "body": "Introduction to ML algorithms",
            "category": "data",
        },
        {
            "id": "5",
            "title": "Web Development",
            "body": "Building web applications with JavaScript",
            "category": "web",
        },
    ]

    for doc in docs:
        es.index(index="test", id=doc["id"], document=doc)

    es.indices.refresh("test")
    return es
