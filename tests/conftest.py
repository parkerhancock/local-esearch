"""Pytest fixtures for local_esearch tests."""

import pytest
from local_esearch import Elasticsearch


@pytest.fixture
def es():
    """Create an in-memory Elasticsearch client."""
    client = Elasticsearch(path=":memory:")
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
