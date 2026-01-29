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

## Bolt-On Search for Existing Tables

Add search to existing SQLite tables without changing your schema:

```python
from local_esearch import Elasticsearch

# Connect to existing database
es = Elasticsearch(path="./myapp.db")

# Register an existing table for search
es.register_table(
    index="articles",           # ES index name
    table="articles",           # Your existing table
    id_column="id",             # Primary key column
    text_columns=["title", "content", "summary"],
    embedding_backend="voyage", # Optional: for semantic search
)

# Build the search index (one-time or periodic)
es.indices.reindex("articles")

# Search with familiar ES API
response = es.search(
    index="articles",
    q="machine learning",
    mode="hybrid",  # keyword + semantic
)

# Results contain row IDs from your table
for hit in response["hits"]["hits"]:
    row_id = hit["_id"]  # Use to join back to your table
```

**How it works:**
- Creates FTS5 virtual table pointing at your table (no data duplication)
- Auto-syncs via triggers on INSERT/UPDATE/DELETE
- Vector embeddings stored in separate table, rebuilt on `reindex()`

## Query DSL Support

- `match` - Full-text search
- `match_phrase` - Phrase search
- `term` / `terms` - Exact matching
- `range` - Numeric/date ranges
- `bool` - Combine queries (must, should, must_not, filter)
- `exists` - Field existence
- `prefix` / `wildcard` - Pattern matching
