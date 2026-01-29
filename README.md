# local_esearch

Elasticsearch-compatible API backed by SQLite + FTS5 + sqlite-vec.

## Installation

```bash
pip install local-esearch
```

## Usage

```python
from local_esearch import Elasticsearch

# Create client (in-memory or file-based)
es = Elasticsearch(path="./search.db")

# Index documents
es.index(index="docs", id="1", document={"title": "Hello", "body": "World"})

# Search
response = es.search(
    index="docs",
    body={"query": {"match": {"body": "world"}}}
)

# Get document
doc = es.get(index="docs", id="1")

# Bulk operations
from local_esearch import helpers

actions = [
    {"_index": "docs", "_id": "1", "_source": {"title": "Doc 1"}},
    {"_index": "docs", "_id": "2", "_source": {"title": "Doc 2"}},
]
success, failed = helpers.bulk(es, actions)
```

## Hybrid Search

Enable semantic search with embedding backends:

```python
es = Elasticsearch(
    path="./search.db",
    embedding_backend="voyage",  # or "gemini", "openai"
)

# Search with hybrid mode (keyword + semantic)
response = es.search(
    index="docs",
    body={"query": {"match": {"body": "search terms"}}},
    mode="hybrid",
)
```

## Query DSL Support

- `match` - Full-text search
- `match_phrase` - Phrase search
- `term` / `terms` - Exact matching
- `range` - Numeric/date ranges
- `bool` - Combine queries (must, should, must_not, filter)
- `exists` - Field existence
- `prefix` / `wildcard` - Pattern matching
