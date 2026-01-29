# local_esearch

<p align="center">
  <strong>Elasticsearch-compatible API backed by SQLite + FTS5 + sqlite-vec</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#query-dsl">Query DSL</a>
</p>

---

Drop-in replacement for `elasticsearch-py` that runs entirely on SQLite. No server required. Full-text search via FTS5, optional semantic search via sqlite-vec.

## Quick Start

```bash
pip install local-esearch
```

```python
from local_esearch import Elasticsearch

es = Elasticsearch(path="./search.db")

# Index a document
es.index(index="docs", id="1", document={"title": "Hello", "body": "World"})

# Search
response = es.search(index="docs", q="world")
print(response["hits"]["hits"][0]["_source"])
# {"title": "Hello", "body": "World"}
```

## Features

| Feature | Description |
|---------|-------------|
| **ES-Compatible API** | Same method signatures as `elasticsearch-py` |
| **FTS5 Full-Text Search** | Porter stemming, BM25 ranking |
| **Hybrid Search** | Combine keyword + semantic search with RRF |
| **Chunking + inner_hits** | Long docs split into passages, matching chunks returned |
| **Bolt-On for Existing Tables** | Add search to any SQLite table without schema changes |
| **Zero Config** | No server, no setup, just `pip install` |
| **Embedding Backends** | Voyage AI, Google Gemini, OpenAI |

## Usage

### Document Operations

```python
from local_esearch import Elasticsearch

es = Elasticsearch(path="./search.db")

# Index
es.index(index="docs", id="1", document={"title": "Python Guide", "body": "Learn Python"})

# Get
doc = es.get(index="docs", id="1")

# Update
es.update(index="docs", id="1", doc={"body": "Learn Python basics"})

# Delete
es.delete(index="docs", id="1")

# Check existence
exists = es.exists(index="docs", id="1")
```

### Search

```python
# Simple query string
response = es.search(index="docs", q="python")

# Query DSL
response = es.search(
    index="docs",
    body={
        "query": {
            "bool": {
                "must": {"match": {"body": "python"}},
                "filter": {"term": {"category": "tutorial"}}
            }
        }
    },
    size=10,
    from_=0,
)

for hit in response["hits"]["hits"]:
    print(hit["_source"]["title"], hit["_score"])
```

### Bulk Operations

```python
from local_esearch import Elasticsearch, helpers

es = Elasticsearch(path="./search.db")

# Bulk index
actions = [
    {"_index": "docs", "_id": "1", "_source": {"title": "Doc 1"}},
    {"_index": "docs", "_id": "2", "_source": {"title": "Doc 2"}},
    {"_index": "docs", "_id": "3", "_source": {"title": "Doc 3"}},
]
success, failed = helpers.bulk(es, actions)

# Scan all documents
for doc in helpers.scan(es, index="docs"):
    print(doc["_id"])
```

### Hybrid Search

Combine keyword search with semantic similarity using embedding backends:

```python
es = Elasticsearch(
    path="./search.db",
    embedding_backend="voyage",  # or "gemini", "openai"
)

# Index documents (embeddings generated automatically)
es.index(index="docs", id="1", document={"title": "Machine Learning", "body": "Neural networks and deep learning"})

# Hybrid search: FTS5 + vector similarity fused with RRF
response = es.search(
    index="docs",
    q="AI algorithms",
    mode="hybrid",  # "keyword", "semantic", or "hybrid"
)
```

**Embedding backends:**
- `voyage` - Voyage AI (1024 dimensions, requires `VOYAGE_API_KEY`)
- `gemini` - Google Gemini (768 dimensions, requires `GOOGLE_API_KEY`)
- `openai` - OpenAI (1536 dimensions, requires `OPENAI_API_KEY`)

### Bolt-On Search for Existing Tables

Add full-text and semantic search to existing SQLite tables without modifying your schema:

```python
from local_esearch import Elasticsearch

# Connect to your existing database
es = Elasticsearch(path="./myapp.db")

# Register an existing table
es.register_table(
    index="articles",                # ES index name
    table="articles",                # Your SQLite table
    id_column="id",                  # Primary key
    text_columns=["title", "body"],  # Columns to index
    embedding_backend="voyage",      # Optional: for semantic search
    chunk_size=250,                  # Words per chunk (ES default)
    chunk_overlap=100,               # Overlap between chunks (ES default)
)

# Build the index
es.indices.reindex("articles")

# Search with ES API
response = es.search(index="articles", q="machine learning", mode="hybrid")

# Results contain row IDs and matching chunks (inner_hits)
for hit in response["hits"]["hits"]:
    row_id = hit["_id"]  # Join back to your table

    # inner_hits shows which chunks matched (for semantic/hybrid)
    if "inner_hits" in hit:
        for chunk in hit["inner_hits"]["chunks"]:
            print(f"  Matched: {chunk['text'][:50]}... (score: {chunk['_score']})")
```

**How it works:**
- Creates FTS5 content table pointing at your table (no data duplication)
- Auto-syncs FTS5 via triggers on INSERT/UPDATE/DELETE
- Long text is chunked (250 words, 100 overlap - matching ES `semantic_text`)
- Each chunk is embedded separately for precise semantic matching
- Search returns documents with `inner_hits` showing matching passages
- Registrations persist across reconnections

**Chunking (mirrors Elasticsearch `semantic_text`):**
- Documents are split into 250-word chunks with 100-word overlap
- Semantic search finds the best matching chunks
- Results include `inner_hits` with the matching passages
- This lets you pinpoint which section of a long document matched

**Reconnecting later:**
```python
# Registrations are restored automatically
es = Elasticsearch(path="./myapp.db")
# "articles" index is ready to search
response = es.search(index="articles", q="python")
```

### Index Management

```python
# Create index with mappings
es.indices.create(
    index="docs",
    mappings={
        "properties": {
            "title": {"type": "text"},
            "category": {"type": "keyword"},
        }
    }
)

# Check if index exists
es.indices.exists(index="docs")

# Delete index
es.indices.delete(index="docs")

# Get index stats
stats = es.indices.stats(index="docs")
```

## Query DSL

Supported query types:

| Query | Example | Description |
|-------|---------|-------------|
| `match` | `{"match": {"body": "search terms"}}` | Full-text search with analysis |
| `match_phrase` | `{"match_phrase": {"body": "exact phrase"}}` | Phrase matching |
| `term` | `{"term": {"status": "published"}}` | Exact value match |
| `terms` | `{"terms": {"tag": ["python", "rust"]}}` | Match any of values |
| `range` | `{"range": {"price": {"gte": 10, "lte": 100}}}` | Numeric/date ranges |
| `exists` | `{"exists": {"field": "author"}}` | Field has value |
| `prefix` | `{"prefix": {"title": "intro"}}` | Prefix matching |
| `wildcard` | `{"wildcard": {"title": "pyth*"}}` | Wildcard patterns |
| `ids` | `{"ids": {"values": ["1", "2"]}}` | Match document IDs |
| `match_all` | `{"match_all": {}}` | Match all documents |

### Bool Queries

Combine queries with boolean logic:

```python
response = es.search(
    index="docs",
    body={
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "python"}}
                ],
                "should": [
                    {"term": {"featured": True}}
                ],
                "must_not": [
                    {"term": {"status": "draft"}}
                ],
                "filter": [
                    {"range": {"date": {"gte": "2024-01-01"}}}
                ]
            }
        }
    }
)
```

- `must` - All conditions must match (AND, affects score)
- `should` - Any condition can match (OR, boosts score)
- `must_not` - No condition can match (NOT)
- `filter` - Must match but doesn't affect score

## API Reference

### Elasticsearch Client

```python
Elasticsearch(
    path=":memory:",           # Database path or ":memory:"
    embedding_backend=None,    # "voyage", "gemini", "openai", or backend instance
)
```

**Document methods:**
- `index(index, document, id=None, refresh=False)` - Index a document
- `get(index, id)` - Get document by ID
- `exists(index, id)` - Check if document exists
- `update(index, id, doc=None, body=None)` - Update document
- `delete(index, id)` - Delete document
- `mget(docs=None, body=None)` - Get multiple documents

**Search methods:**
- `search(index, body=None, q=None, size=10, from_=0, mode="keyword")` - Search documents
- `count(index, body=None, q=None)` - Count matching documents

**Table registration:**
- `register_table(index, table, text_columns, id_column="id")` - Register existing table
- `unregister_table(index, drop_indexes=False)` - Unregister table
- `get_table_index(index)` - Get TableIndex instance

### Helpers

```python
from local_esearch import helpers

# Bulk operations
success, errors = helpers.bulk(client, actions, chunk_size=500)

# Iterate all documents
for doc in helpers.scan(client, index="docs", query={"match_all": {}}):
    print(doc)

# Streaming bulk with results
for ok, result in helpers.streaming_bulk(client, actions):
    if not ok:
        print(f"Error: {result}")
```

### IndicesClient

```python
es.indices.create(index, mappings=None, settings=None)
es.indices.delete(index)
es.indices.exists(index)
es.indices.refresh(index=None)
es.indices.stats(index=None)
es.indices.reindex(index)  # For registered tables
```

## Limitations

This library prioritizes simplicity and local operation over full ES compatibility:

- No distributed features (shards, replicas, clusters)
- No custom analyzers (FTS5 porter stemmer only)
- No aggregations (planned for future)
- No nested object queries
- No scroll API (scan uses pagination)
- Script updates not supported

## License

MIT
