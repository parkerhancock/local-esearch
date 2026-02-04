"""Main Elasticsearch client implementation."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from local_esearch.backends import DatabaseBackend, create_backend
from local_esearch.embeddings import get_backend
from local_esearch.embeddings.base import EmbeddingBackend
from local_esearch.exceptions import ConflictError, NotFoundError, RequestError
from local_esearch.hybrid import reciprocal_rank_fusion
from local_esearch.indices import IndicesClient
from local_esearch.query_dsl import QueryCompiler
from local_esearch.response import (
    format_delete_response,
    format_get_response,
    format_hit,
    format_index_response,
    format_search_response,
    format_update_response,
)
from local_esearch.table_index import TableIndex


def _extract_text_fields(source: dict, mappings: dict | None = None) -> str:
    """Extract text from document for FTS indexing.

    Concatenates all string fields (recursively) into a single text blob.
    If mappings specify text fields, only those are extracted.
    """
    if mappings and "properties" in mappings:
        text_fields = []
        for field, config in mappings["properties"].items():
            if config.get("type") == "text" and field in source:
                value = source[field]
                if isinstance(value, str):
                    text_fields.append(value)
        return " ".join(text_fields)

    texts: list[str] = []

    def extract(obj: Any) -> None:
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)

    extract(source)
    return " ".join(texts)


class Elasticsearch:
    """Elasticsearch-compatible client backed by SQLite/PostgreSQL + FTS + vectors.

    Provides a drop-in replacement for elasticsearch-py in local/embedded use cases.

    Example:
        # SQLite (default)
        es = Elasticsearch(path="./search.db")

        # PostgreSQL
        es = Elasticsearch(path="postgresql://user:pass@localhost/dbname")

        es.index(index="docs", id="1", document={"title": "Hello", "body": "World"})
        response = es.search(index="docs", body={"query": {"match": {"body": "world"}}})
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedding_backend: str | EmbeddingBackend | None = None,
    ):
        """Initialize Elasticsearch client.

        Args:
            path: Database path. Can be:
                - SQLite file path (e.g., "./search.db")
                - ":memory:" for in-memory SQLite
                - PostgreSQL URI (e.g., "postgresql://user:pass@localhost/dbname")
            embedding_backend: Embedding backend name ("voyage", "gemini", "openai")
                              or EmbeddingBackend instance, or None for keyword-only
        """
        self._path = str(path)

        # Initialize embedding backend
        if isinstance(embedding_backend, str):
            self._embedding_backend: EmbeddingBackend | None = get_backend(
                embedding_backend
            )
        else:
            self._embedding_backend = embedding_backend

        # Create database backend
        self._backend: DatabaseBackend = create_backend(path, self._embedding_backend)

        # Initialize schema
        vector_dims = (
            self._embedding_backend.dimensions if self._embedding_backend else None
        )
        self._backend.init_schema(vector_dims)

        # Ensure vector table exists if we have an embedding backend
        if self._embedding_backend and self._backend.vector_available():
            self._backend.create_documents_vec_table(self._embedding_backend.dimensions)

        # Sub-clients
        self.indices = IndicesClient(self)

        # Query compiler
        self._query_compiler = QueryCompiler(self._backend)

        # Registry for table indexes (bolt-on search over existing tables)
        self._table_indexes: dict[str, TableIndex] = {}

        # Auto-load any previously registered table indexes
        self._load_table_indexes()

    def close(self) -> None:
        """Close the database connection."""
        self._backend.close()

    def __enter__(self) -> Elasticsearch:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Table Registration (bolt-on search for existing tables)
    # -------------------------------------------------------------------------

    def register_table(
        self,
        index: str,
        table: str,
        *,
        id_column: str = "id",
        text_columns: list[str] | None = None,
        embedding_text: Callable[[dict], str] | str | None = None,
        embedding_backend: str | EmbeddingBackend | None = None,
        setup: bool = True,
    ) -> TableIndex:
        """Register an existing table for search.

        Creates FTS and vector indexes over an existing table,
        enabling search without duplicating data.

        Args:
            index: ES index name to use for this table
            table: Actual table name
            id_column: Primary key column name
            text_columns: Columns to include in full-text search
            embedding_text: How to generate text for embeddings:
                - str: column name to embed
                - Callable: function(row_dict) -> str
                - None: concatenate all text_columns
            embedding_backend: Backend name or instance (overrides client default)
            setup: Auto-create FTS table and triggers

        Returns:
            TableIndex instance for further configuration

        Example:
            es.register_table(
                index="emails",
                table="emails",
                text_columns=["subject", "body", "sender"],
                embedding_backend="voyage",
            )
            es.indices.reindex("emails")
            results = es.search(index="emails", q="contract", mode="hybrid")
        """
        # Resolve embedding backend
        if isinstance(embedding_backend, str):
            backend = get_backend(embedding_backend)
        elif embedding_backend is not None:
            backend = embedding_backend
        else:
            backend = self._embedding_backend

        # Create table index
        table_index = TableIndex(
            db_backend=self._backend,
            table=table,
            id_column=id_column,
            text_columns=text_columns or [],
            embedding_text=embedding_text,
            embedding_backend=backend,
        )

        if setup:
            table_index.setup()

        self._table_indexes[index] = table_index

        # Persist registration for reconnection
        backend_name = None
        if isinstance(embedding_backend, str):
            backend_name = embedding_backend
        elif backend and hasattr(backend, "backend_name"):
            backend_name = backend.backend_name

        self._save_table_index(
            index=index,
            table=table,
            id_column=id_column,
            text_columns=text_columns or [],
            embedding_text=embedding_text if isinstance(embedding_text, str) else None,
            embedding_backend=backend_name,
        )

        return table_index

    def get_table_index(self, index: str) -> TableIndex | None:
        """Get the TableIndex for a registered table."""
        return self._table_indexes.get(index)

    def _load_table_indexes(self) -> None:
        """Load previously registered table indexes from metadata."""
        try:
            rows = self._backend.execute("""
                SELECT index_name, table_name, id_column, text_columns_json,
                       embedding_text, embedding_backend
                FROM _table_indexes
            """)
        except Exception:
            # Table doesn't exist yet (old schema)
            return

        for row in rows:
            index_name = row["index_name"]
            table_name = row["table_name"]
            id_column = row["id_column"]
            text_cols_json = row["text_columns_json"]
            emb_text = row["embedding_text"]
            emb_backend = row["embedding_backend"]

            # Parse text columns (PostgreSQL JSONB returns list directly, SQLite returns JSON string)
            text_columns = text_cols_json if isinstance(text_cols_json, list) else (json.loads(text_cols_json) if text_cols_json else [])

            # Resolve embedding backend
            if emb_backend:
                backend = get_backend(emb_backend)
            else:
                backend = self._embedding_backend

            # Recreate TableIndex (FTS tables and triggers already exist)
            table_index = TableIndex(
                db_backend=self._backend,
                table=table_name,
                id_column=id_column,
                text_columns=text_columns,
                embedding_text=emb_text,
                embedding_backend=backend,
            )
            # Don't call setup() - tables already exist
            self._table_indexes[index_name] = table_index

    def _save_table_index(
        self,
        index: str,
        table: str,
        id_column: str,
        text_columns: list[str],
        embedding_text: str | None,
        embedding_backend: str | None,
    ) -> None:
        """Persist table index registration to metadata."""
        self._backend.upsert_table_registration(
            index=index,
            table=table,
            id_column=id_column,
            text_columns_json=json.dumps(text_columns),
            embedding_text=embedding_text if isinstance(embedding_text, str) else None,
            embedding_backend=embedding_backend,
            created_at=self._backend.timestamp_now(),
        )
        self._backend.commit()

    def unregister_table(self, index: str, *, drop_indexes: bool = False) -> bool:
        """Unregister a table index.

        Args:
            index: ES index name to unregister
            drop_indexes: Also drop the FTS and vector tables

        Returns:
            True if index was registered, False otherwise
        """
        table_index = self._table_indexes.pop(index, None)
        if not table_index:
            return False

        # Remove from metadata
        self._backend.execute(
            "DELETE FROM _table_indexes WHERE index_name = ?", (index,)
        )
        self._backend.commit()

        # Optionally drop the FTS/vector tables
        if drop_indexes:
            table_index.drop()

        return True

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    def index(
        self,
        index: str,
        document: dict[str, Any],
        id: str | None = None,
        *,
        body: dict[str, Any] | None = None,
        refresh: bool | Literal["wait_for"] = False,
        op_type: Literal["index", "create"] | None = None,
    ) -> dict[str, Any]:
        """Index a document.

        Args:
            index: Index name
            document: Document to index
            id: Document ID (auto-generated if not provided)
            body: Alternative way to pass document (for compatibility)
            refresh: Refresh index after operation
            op_type: "create" fails if doc exists, "index" upserts

        Returns:
            Index operation response
        """
        doc = body or document
        doc_id = id or str(uuid.uuid4())

        # Ensure index exists (auto-create like ES)
        self._ensure_index(index)

        # Get mappings for text extraction
        mappings = self._get_index_mappings(index)

        # Extract text for FTS
        text = _extract_text_fields(doc, mappings)

        now = self._backend.timestamp_now()

        # Check if document exists
        rows = self._backend.execute(
            "SELECT _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, doc_id),
        )
        existing = rows[0] if rows else None

        if existing:
            if op_type == "create":
                raise ConflictError(
                    f"Document '{doc_id}' already exists in index '{index}'"
                )

            # Update existing document
            new_version = existing["_version"] + 1
            self._backend.execute(
                """
                UPDATE _documents
                SET _source = ?, _text = ?, _version = ?, updated_at = ?
                WHERE _index = ? AND _id = ?
                """,
                (json.dumps(doc), text, new_version, now, index, doc_id),
            )
            result = "updated"
        else:
            # Insert new document
            new_version = 1
            self._backend.execute(
                """
                INSERT INTO _documents
                    (_index, _id, _source, _text, _version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (index, doc_id, json.dumps(doc), text, new_version, now, now),
            )
            result = "created"

        # Update vector embedding if backend available
        if (
            self._embedding_backend
            and self._backend.vector_available()
            and text
        ):
            self._update_embedding(index, doc_id, text)

        self._backend.commit()

        return format_index_response(index, doc_id, new_version, result)

    def get(
        self,
        index: str,
        id: str,
        *,
        _source: bool | list[str] = True,
    ) -> dict[str, Any]:
        """Get a document by ID.

        Args:
            index: Index name
            id: Document ID
            _source: Include source (True/False or list of fields)

        Returns:
            Document response

        Raises:
            NotFoundError: If document not found
        """
        rows = self._backend.execute(
            "SELECT _source, _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )

        if not rows:
            raise NotFoundError(
                f"Document '{id}' not found in index '{index}'",
                body={"_index": index, "_id": id, "found": False},
            )

        row = rows[0]
        # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
        source = row["_source"] if isinstance(row["_source"], dict) else json.loads(row["_source"])
        version = row["_version"]

        # Filter source fields if specified
        if isinstance(_source, list):
            source = {k: v for k, v in source.items() if k in _source}
        elif not _source:
            source = {}

        return format_get_response(index, id, source, version, found=True)

    def exists(self, index: str, id: str) -> bool:
        """Check if a document exists.

        Args:
            index: Index name
            id: Document ID

        Returns:
            True if document exists
        """
        rows = self._backend.execute(
            "SELECT 1 FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        return len(rows) > 0

    def delete(
        self,
        index: str,
        id: str,
        *,
        refresh: bool | Literal["wait_for"] = False,
    ) -> dict[str, Any]:
        """Delete a document.

        Args:
            index: Index name
            id: Document ID
            refresh: Refresh index after operation

        Returns:
            Delete operation response

        Raises:
            NotFoundError: If document not found
        """
        # Get version before delete
        rows = self._backend.execute(
            "SELECT _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        if not rows:
            raise NotFoundError(f"Document '{id}' not found in index '{index}'")

        version = rows[0]["_version"]

        # Delete document
        self._backend.execute(
            "DELETE FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )

        # Delete vector embedding if exists
        if self._backend.vector_available():
            self._backend.vector_delete("_documents_vec", f"{index}:{id}")

        self._backend.commit()

        return format_delete_response(index, id, version)

    def update(
        self,
        index: str,
        id: str,
        *,
        body: dict[str, Any] | None = None,
        doc: dict[str, Any] | None = None,
        script: dict[str, Any] | None = None,
        refresh: bool | Literal["wait_for"] = False,
    ) -> dict[str, Any]:
        """Update a document.

        Args:
            index: Index name
            id: Document ID
            body: Update body containing "doc" or "script"
            doc: Partial document to merge (alternative to body)
            script: Script update (not fully supported)
            refresh: Refresh index after operation

        Returns:
            Update operation response
        """
        if body:
            doc = body.get("doc", doc)
            script = body.get("script", script)

        if script:
            raise RequestError("Script updates not supported")

        if not doc:
            raise RequestError("Missing 'doc' in update body")

        # Get existing document
        rows = self._backend.execute(
            "SELECT _source, _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        if not rows:
            raise NotFoundError(f"Document '{id}' not found in index '{index}'")

        row = rows[0]
        # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
        existing_source = row["_source"] if isinstance(row["_source"], dict) else json.loads(row["_source"])
        version = row["_version"]

        # Merge documents
        existing_source.update(doc)

        # Get mappings for text extraction
        mappings = self._get_index_mappings(index)
        text = _extract_text_fields(existing_source, mappings)

        # Update
        new_version = version + 1
        now = self._backend.timestamp_now()
        self._backend.execute(
            """
            UPDATE _documents
            SET _source = ?, _text = ?, _version = ?, updated_at = ?
            WHERE _index = ? AND _id = ?
            """,
            (json.dumps(existing_source), text, new_version, now, index, id),
        )

        # Update vector embedding
        if (
            self._embedding_backend
            and self._backend.vector_available()
            and text
        ):
            self._update_embedding(index, id, text)

        self._backend.commit()

        return format_update_response(index, id, new_version)

    def mget(
        self,
        *,
        body: dict[str, Any] | None = None,
        docs: list[dict[str, Any]] | None = None,
        index: str | None = None,
    ) -> dict[str, Any]:
        """Get multiple documents.

        Args:
            body: Body containing "docs" list
            docs: List of {_index, _id} dicts
            index: Default index if not specified in docs

        Returns:
            Multi-get response with docs array
        """
        if body:
            docs = body.get("docs", docs)

        if not docs:
            return {"docs": []}

        results = []
        for doc_spec in docs:
            doc_index = doc_spec.get("_index", index)
            doc_id = doc_spec.get("_id")

            if not doc_index or not doc_id:
                results.append({"found": False, "error": "Missing _index or _id"})
                continue

            try:
                result = self.get(index=doc_index, id=doc_id)
                results.append(result)
            except NotFoundError:
                results.append(
                    {
                        "_index": doc_index,
                        "_id": doc_id,
                        "found": False,
                    }
                )

        return {"docs": results}

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(
        self,
        *,
        index: str | list[str] | None = None,
        body: dict[str, Any] | None = None,
        q: str | None = None,
        size: int = 10,
        from_: int = 0,
        sort: list[str] | str | None = None,
        _source: bool | list[str] = True,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
    ) -> dict[str, Any]:
        """Search for documents.

        Args:
            index: Index name(s) to search. Can mix registered tables and regular indices.
            body: Search body with "query" key
            q: Simple query string (alternative to body)
            size: Maximum results to return
            from_: Offset for pagination
            sort: Sort specification
            _source: Source filtering
            mode: Search mode - "keyword" (FTS), "semantic" (vector), "hybrid" (RRF fusion)

        Returns:
            Search response with hits
        """
        start_time = time.time()

        # Normalize index to list
        if isinstance(index, str):
            indices = [index]
        elif index:
            indices = list(index)
        else:
            indices = None  # Search all

        # Categorize indices into registered tables vs regular indices
        registered_indices = []
        regular_indices = []
        if indices:
            for idx in indices:
                if idx in self._table_indexes:
                    registered_indices.append(idx)
                else:
                    regular_indices.append(idx)
        else:
            # Search all: include all registered tables + regular indices
            registered_indices = list(self._table_indexes.keys())
            regular_indices = None  # Will search all regular indices

        # Build query from body or q parameter
        query: dict[str, Any] = {}
        if body:
            query = body.get("query", {})
            size = body.get("size", size)
            from_ = body.get("from", from_)
            sort = body.get("sort", sort)
            _source = body.get("_source", _source)

        if q:
            # Simple query string - search all text fields
            query = {"match": {"_text": q}}

        # Extract query text for semantic/table searches
        query_text = self._extract_query_text(query, q)

        # Determine search mode
        use_semantic = (
            mode in ("semantic", "hybrid")
            and self._embedding_backend
            and self._backend.vector_available()
        )
        use_keyword = mode in ("keyword", "hybrid")

        keyword_results: list[tuple[str, Any, dict, float]] = []
        vector_results: list[tuple[str, Any, dict, float]] = []
        inner_hits_map: dict[tuple[str, Any], list[dict]] = {}

        # Search registered tables
        if registered_indices and query_text:
            for idx in registered_indices:
                table_index = self._table_indexes[idx]
                kw, vec, inner_hits = table_index.search_raw(
                    query_text,
                    mode=mode,
                    limit=size + from_,
                    index_name=idx,
                )
                keyword_results.extend(kw)
                vector_results.extend(vec)
                for doc_id, chunks in inner_hits.items():
                    inner_hits_map[(idx, doc_id)] = chunks

        # Search regular indices
        has_regular = regular_indices is None or len(regular_indices) > 0
        if has_regular:
            if use_keyword:
                kw = self._keyword_search(query, regular_indices, size=size + from_)
                keyword_results.extend(kw)

            if use_semantic and query_text:
                vec = self._vector_search(query_text, regular_indices, size=size + from_)
                vector_results.extend(vec)

        # Get total count
        total = 0
        if has_regular:
            total += self._count_matches(query, regular_indices)
        if registered_indices:
            seen_docs: set[tuple[str, Any]] = set()
            for idx, doc_id, _, _ in keyword_results:
                if idx in self._table_indexes:
                    seen_docs.add((idx, doc_id))
            for idx, doc_id, _, _ in vector_results:
                if idx in self._table_indexes:
                    seen_docs.add((idx, doc_id))
            total += len(seen_docs)

        # Combine results with RRF fusion
        if mode == "hybrid" and keyword_results and vector_results:
            fused = reciprocal_rank_fusion(keyword_results, vector_results)
            hits = []
            for doc in fused[from_ : from_ + size]:
                source = doc.source.copy() if doc.source else {}
                if doc.index in self._table_indexes:
                    source["_table_row_id"] = doc.doc_id
                hit = format_hit(doc.index, str(doc.doc_id), source, score=doc.fused_score)
                if (doc.index, doc.doc_id) in inner_hits_map:
                    hit["inner_hits"] = {"chunks": inner_hits_map[(doc.index, doc.doc_id)]}
                hits.append(hit)
            max_score = fused[0].fused_score if fused else None
        elif vector_results and mode == "semantic":
            hits = []
            for idx, doc_id, source, score in vector_results[from_ : from_ + size]:
                source = source.copy() if source else {}
                if idx in self._table_indexes:
                    source["_table_row_id"] = doc_id
                hit = format_hit(idx, str(doc_id), source, score=score)
                if (idx, doc_id) in inner_hits_map:
                    hit["inner_hits"] = {"chunks": inner_hits_map[(idx, doc_id)]}
                hits.append(hit)
            max_score = vector_results[0][3] if vector_results else None
        else:
            # Keyword only
            hits = []
            for idx, doc_id, source, score in keyword_results[from_ : from_ + size]:
                source = source.copy() if source else {}
                if idx in self._table_indexes:
                    source["_table_row_id"] = doc_id
                hit = format_hit(idx, str(doc_id), source, score=score)
                hits.append(hit)
            max_score = keyword_results[0][3] if keyword_results else None

        # Apply source filtering
        if isinstance(_source, list):
            for hit in hits:
                hit["_source"] = {k: v for k, v in hit["_source"].items() if k in _source}
        elif not _source:
            for hit in hits:
                hit["_source"] = {}

        took_ms = int((time.time() - start_time) * 1000)
        return format_search_response(hits, total, took_ms, max_score)

    def count(
        self,
        *,
        index: str | list[str] | None = None,
        body: dict[str, Any] | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        """Count documents matching a query.

        Args:
            index: Index name(s)
            body: Query body
            q: Simple query string

        Returns:
            Count response
        """
        # Normalize index
        if isinstance(index, str):
            indices = [index]
        elif index:
            indices = list(index)
        else:
            indices = None

        query: dict[str, Any] = {}
        if body:
            query = body.get("query", {})

        if q:
            query = {"match": {"_text": q}}

        # Compile and count
        compiled = self._query_compiler.compile(query)

        where_parts = []
        params: list[Any] = []

        if indices:
            placeholders = self._backend.placeholders(len(indices))
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        sql = f"SELECT COUNT(*) as cnt FROM _documents WHERE {where_clause}"

        rows = self._backend.execute(sql, params)
        count = rows[0]["cnt"] if rows else 0

        return {
            "count": count,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        }

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _ensure_index(self, index: str) -> None:
        """Ensure index exists, creating if needed."""
        rows = self._backend.execute(
            "SELECT 1 FROM _indices WHERE name = ?", (index,)
        )
        if not rows:
            self._backend.execute(
                "INSERT INTO _indices (name, created_at) VALUES (?, ?)",
                (index, self._backend.timestamp_now()),
            )
            self._backend.commit()

    def _get_index_mappings(self, index: str) -> dict | None:
        """Get mappings for an index."""
        rows = self._backend.execute(
            "SELECT mappings_json FROM _indices WHERE name = ?", (index,)
        )
        if rows and rows[0]["mappings_json"]:
            data = rows[0]["mappings_json"]
            # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
            return data if isinstance(data, dict) else json.loads(data)
        return None

    def _count_matches(
        self,
        query: dict[str, Any],
        indices: list[str] | None,
    ) -> int:
        """Count documents matching a query."""
        compiled = self._query_compiler.compile(query)

        where_parts = []
        params: list[Any] = []

        if indices:
            placeholders = self._backend.placeholders(len(indices))
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        # Get SQL from backend (handles FTS for SQLite vs PostgreSQL)
        sql, num_fts_params = self._backend.document_fts_count_sql(
            where_clause, compiled.uses_fts
        )

        if compiled.uses_fts:
            # Add FTS params at the end
            fts_params = [compiled.fts_query] * num_fts_params
            params = params + fts_params

        rows = self._backend.execute(sql, params)
        return rows[0]["cnt"] if rows else 0

    def _keyword_search(
        self,
        query: dict[str, Any],
        indices: list[str] | None,
        size: int,
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        """Execute keyword search using FTS."""
        compiled = self._query_compiler.compile(query)

        where_parts = []
        params: list[Any] = []

        if indices:
            placeholders = self._backend.placeholders(len(indices))
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        # Get SQL from backend (handles SQLite vs PostgreSQL differences)
        sql, num_fts_params = self._backend.document_fts_search_sql(
            where_clause, compiled.uses_fts, compiled.fts_query
        )

        if compiled.uses_fts:
            # FTS params go at the beginning (1 for SQLite, 2 for PostgreSQL)
            fts_params = [compiled.fts_query] * num_fts_params
            params = fts_params + params + [size]
        else:
            params.append(size)

        rows = self._backend.execute(sql, params)
        results = []
        for row in rows:
            # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
            source = row["_source"] if isinstance(row["_source"], dict) else json.loads(row["_source"])
            score = -row["score"] if row["score"] else 0.0  # BM25 returns negative
            results.append((row["_index"], row["_id"], source, score))

        return results

    def _vector_search(
        self,
        query_text: str,
        indices: list[str] | None,
        size: int,
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        """Execute vector similarity search."""
        if not self._embedding_backend or not self._backend.vector_available():
            return []

        # Get query embedding
        query_embedding = self._embedding_backend.embed(query_text)
        embedding_json = json.dumps(query_embedding)

        # Vector search with optional index filter
        if indices:
            # Filter by index prefix
            index_filters = " OR ".join("doc_key LIKE ?" for _ in indices)
            index_params: list[Any] = [f"{idx}:%" for idx in indices]

            sql = f"""
                SELECT doc_key, distance
                FROM _documents_vec
                WHERE ({index_filters})
                ORDER BY embedding <-> ?
                LIMIT ?
            """
            params = index_params + [embedding_json, size]
        else:
            sql = """
                SELECT doc_key, distance
                FROM _documents_vec
                ORDER BY embedding <-> ?
                LIMIT ?
            """
            params = [embedding_json, size]

        try:
            rows = self._backend.execute(sql, params)
        except Exception:
            return []

        results = []
        for row in rows:
            doc_key = row["doc_key"]
            distance = row["distance"]
            # Parse index:id from doc_key
            parts = doc_key.split(":", 1)
            if len(parts) != 2:
                continue
            idx, doc_id = parts

            # Get full document
            doc_rows = self._backend.execute(
                "SELECT _source FROM _documents WHERE _index = ? AND _id = ?",
                (idx, doc_id),
            )
            if doc_rows:
                # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
                src = doc_rows[0]["_source"]
                source = src if isinstance(src, dict) else json.loads(src)
                score = 1.0 - float(distance)
                results.append((idx, doc_id, source, score))

        return results

    def _update_embedding(self, index: str, doc_id: str, text: str) -> None:
        """Update vector embedding for a document."""
        if not self._embedding_backend or not self._backend.vector_available():
            return

        try:
            embedding = self._embedding_backend.embed(text)
            doc_key = f"{index}:{doc_id}"
            self._backend.vector_upsert("_documents_vec", doc_key, embedding)
        except Exception:
            pass  # Don't fail document indexing if embedding fails

    def _extract_query_text(self, query: dict[str, Any], q: str | None) -> str | None:
        """Extract text from query for semantic search."""
        if q:
            return q

        if not query:
            return None

        # Extract from match query
        if "match" in query:
            match_body = query["match"]
            field = next(iter(match_body.keys()), None)
            if field:
                value = match_body[field]
                if isinstance(value, dict):
                    return value.get("query")
                return str(value)

        # Extract from match_phrase
        if "match_phrase" in query:
            match_body = query["match_phrase"]
            field = next(iter(match_body.keys()), None)
            if field:
                value = match_body[field]
                if isinstance(value, dict):
                    return value.get("query")
                return str(value)

        # Extract from bool query
        if "bool" in query:
            bool_body = query["bool"]
            for clause_type in ["must", "should"]:
                clauses = bool_body.get(clause_type, [])
                if isinstance(clauses, dict):
                    clauses = [clauses]
                for clause in clauses:
                    text = self._extract_query_text(clause, None)
                    if text:
                        return text

        return None
