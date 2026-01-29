# local_esearch

Elasticsearch-compatible API backed by SQLite + FTS5 + sqlite-vec. Drop-in replacement for `elasticsearch-py` that runs locally.

## Installation

```bash
pip install local-esearch
```

Optional for vector search:
```bash
pip install sqlite-vec voyageai  # or google-generativeai, openai
```

## Quick Reference

```python
from local_esearch import Elasticsearch, helpers

# Initialize
es = Elasticsearch(path="./search.db")  # or ":memory:"
es = Elasticsearch(path="./search.db", embedding_backend="voyage")  # with vectors

# Document CRUD
es.index(index="docs", id="1", document={"title": "Hello", "body": "World"})
doc = es.get(index="docs", id="1")
es.update(index="docs", id="1", doc={"title": "Updated"})
es.delete(index="docs", id="1")
exists = es.exists(index="docs", id="1")

# Search
response = es.search(index="docs", q="hello")
response = es.search(index="docs", body={"query": {"match": {"title": "hello"}}})
response = es.search(index="docs", q="hello", mode="hybrid")  # keyword + semantic

# Bulk
actions = [{"_index": "docs", "_id": "1", "_source": {"title": "Doc 1"}}]
success, errors = helpers.bulk(es, actions)

# Scan all
for doc in helpers.scan(es, index="docs"):
    print(doc["_source"])

# Index management
es.indices.create(index="docs", mappings={"properties": {"title": {"type": "text"}}})
es.indices.delete(index="docs")
es.indices.exists(index="docs")
es.indices.refresh(index="docs")

# Bolt-on search for existing tables
es.register_table(
    index="articles",
    table="articles",  # existing SQLite table
    id_column="id",
    text_columns=["title", "body"],
    embedding_backend="voyage",
    chunk_size=250,    # words per chunk
    chunk_overlap=100, # overlap words
)
es.indices.reindex("articles")

# Close
es.close()
```

## Type Signatures

```python
class Elasticsearch:
    def __init__(
        self,
        path: str | Path = ":memory:",
        embedding_backend: str | EmbeddingBackend | None = None,  # "voyage", "gemini", "openai"
    ): ...

    def index(
        self,
        index: str,
        document: dict[str, Any],
        id: str | None = None,
        refresh: bool | Literal["wait_for"] = False,
        op_type: Literal["index", "create"] | None = None,
    ) -> dict[str, Any]: ...

    def get(self, index: str, id: str) -> dict[str, Any]: ...
    def exists(self, index: str, id: str) -> bool: ...
    def delete(self, index: str, id: str) -> dict[str, Any]: ...
    def update(self, index: str, id: str, doc: dict | None = None, body: dict | None = None) -> dict[str, Any]: ...

    def search(
        self,
        index: str | list[str] | None = None,
        body: dict[str, Any] | None = None,
        q: str | None = None,
        size: int = 10,
        from_: int = 0,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
    ) -> dict[str, Any]: ...

    def count(self, index: str | None = None, body: dict | None = None, q: str | None = None) -> dict[str, Any]: ...

    def register_table(
        self,
        index: str,
        table: str,
        id_column: str = "id",
        text_columns: list[str] | None = None,
        embedding_text: Callable[[dict], str] | str | None = None,
        embedding_backend: str | EmbeddingBackend | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 100,
    ) -> TableIndex: ...

    def unregister_table(self, index: str, drop_indexes: bool = False) -> bool: ...
```

## Response Formats

### Search Response
```python
{
    "took": 5,  # milliseconds
    "timed_out": False,
    "hits": {
        "total": {"value": 42, "relation": "eq"},
        "max_score": 1.5,
        "hits": [
            {
                "_index": "docs",
                "_id": "1",
                "_score": 1.5,
                "_source": {"title": "Hello", "body": "World"},
                # For semantic/hybrid mode with registered tables:
                "inner_hits": {
                    "chunks": [
                        {"chunk_idx": 2, "text": "matching passage...", "_score": 0.92},
                        {"chunk_idx": 5, "text": "another match...", "_score": 0.85},
                    ]
                }
            }
        ]
    }
}
```

### Index Response
```python
{"_index": "docs", "_id": "1", "_version": 1, "result": "created"}
```

### Get Response
```python
{"_index": "docs", "_id": "1", "_version": 1, "found": True, "_source": {...}}
```

## Query DSL

### Match (full-text search)
```python
{"query": {"match": {"body": "search terms"}}}
{"query": {"match": {"body": {"query": "search terms", "operator": "and"}}}}
```

### Match Phrase
```python
{"query": {"match_phrase": {"body": "exact phrase"}}}
```

### Term (exact match)
```python
{"query": {"term": {"status": "published"}}}
{"query": {"terms": {"tag": ["python", "rust"]}}}
```

### Range
```python
{"query": {"range": {"price": {"gte": 10, "lte": 100}}}}
{"query": {"range": {"date": {"gte": "2024-01-01", "lt": "2025-01-01"}}}}
```

### Bool (combine queries)
```python
{
    "query": {
        "bool": {
            "must": [{"match": {"title": "python"}}],      # AND, affects score
            "should": [{"term": {"featured": True}}],      # OR, boosts score
            "must_not": [{"term": {"status": "draft"}}],   # NOT
            "filter": [{"range": {"date": {"gte": "2024-01-01"}}}]  # AND, no score
        }
    }
}
```

### Other Queries
```python
{"query": {"exists": {"field": "author"}}}
{"query": {"prefix": {"title": "intro"}}}
{"query": {"wildcard": {"title": "pyth*"}}}
{"query": {"ids": {"values": ["1", "2", "3"]}}}
{"query": {"match_all": {}}}
```

## Bulk Actions Format

```python
# Index
{"_index": "docs", "_id": "1", "_source": {"title": "Doc"}}

# Create (fails if exists)
{"_index": "docs", "_id": "1", "_op_type": "create", "_source": {"title": "Doc"}}

# Update
{"_index": "docs", "_id": "1", "_op_type": "update", "_source": {"doc": {"title": "New"}}}

# Delete
{"_index": "docs", "_id": "1", "_op_type": "delete"}
```

## Search Modes

| Mode | Description | Requires |
|------|-------------|----------|
| `keyword` | FTS5 full-text search with BM25 scoring | Nothing |
| `semantic` | Vector similarity search | `embedding_backend` + `sqlite-vec` |
| `hybrid` | Keyword + semantic fused with RRF | `embedding_backend` + `sqlite-vec` |

## Chunking (for semantic search)

Long documents are split into overlapping chunks for better semantic precision:

- **Default**: 250 words per chunk, 100 word overlap (matches ES `semantic_text`)
- **Storage**: Chunks stored with metadata (text, position)
- **Search**: Returns documents ranked by best chunk score
- **inner_hits**: Contains matching passages for each document

```python
# Configure chunking
es.register_table(
    index="articles",
    table="articles",
    text_columns=["body"],
    chunk_size=250,    # words per chunk
    chunk_overlap=100, # words of overlap
)

# Search returns inner_hits with matching chunks
response = es.search(index="articles", q="machine learning", mode="hybrid")
for hit in response["hits"]["hits"]:
    print(f"Document {hit['_id']}")
    for chunk in hit.get("inner_hits", {}).get("chunks", []):
        print(f"  Chunk {chunk['chunk_idx']}: {chunk['text'][:50]}...")
```

## SQLite Schema

```sql
-- Documents table
CREATE TABLE _documents (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    _index TEXT NOT NULL,
    _id TEXT NOT NULL,
    _source TEXT NOT NULL,  -- JSON
    _text TEXT,             -- extracted text for FTS
    _version INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL,
    UNIQUE (_index, _id)
);

-- FTS5 full-text index
CREATE VIRTUAL TABLE _documents_fts USING fts5(
    _text,
    content='_documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Vector embeddings (if sqlite-vec available)
CREATE VIRTUAL TABLE _documents_vec USING vec0(
    doc_key TEXT PRIMARY KEY,  -- "{index}:{id}"
    embedding float[{dimensions}]
);

-- Registered table indexes
CREATE TABLE _table_indexes (
    index_name TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    id_column TEXT NOT NULL,
    text_columns_json TEXT NOT NULL,
    embedding_text TEXT,
    embedding_backend TEXT,
    created_at REAL
);

-- For registered tables: chunks metadata
CREATE TABLE {table}_chunks (
    chunk_id TEXT PRIMARY KEY,  -- "{row_id}:{chunk_idx}"
    row_id INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER
);
```

## Embedding Backends

| Backend | Dimensions | Model | Env Var |
|---------|------------|-------|---------|
| `voyage` | 1024 | voyage-3-lite | `VOYAGE_API_KEY` |
| `gemini` | 768 | text-embedding-004 | `GOOGLE_API_KEY` |
| `openai` | 1536 | text-embedding-3-small | `OPENAI_API_KEY` |

## Exceptions

```python
from local_esearch import NotFoundError, ConflictError, RequestError

try:
    doc = es.get(index="docs", id="missing")
except NotFoundError as e:
    print(f"Not found: {e}")

try:
    es.index(index="docs", id="1", document={...}, op_type="create")
except ConflictError as e:
    print(f"Already exists: {e}")
```

## Limitations vs Real Elasticsearch

- No distributed features (shards, replicas, clusters)
- No custom analyzers (FTS5 porter stemmer only)
- No aggregations
- No nested object queries
- No scroll API (scan uses pagination)
- No script updates
