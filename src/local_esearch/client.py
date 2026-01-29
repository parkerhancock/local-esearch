"""Main Elasticsearch client implementation."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

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
from local_esearch.schema import ensure_vector_table, extract_text_fields, init_database
from local_esearch.table_index import TableIndex


class Elasticsearch:
    """Elasticsearch-compatible client backed by SQLite + FTS5 + sqlite-vec.

    Provides a drop-in replacement for elasticsearch-py in local/embedded use cases.

    Example:
        es = Elasticsearch(path="./search.db")
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
            path: SQLite database path or ":memory:" for in-memory
            embedding_backend: Embedding backend name ("voyage", "gemini", "openai")
                              or EmbeddingBackend instance, or None for keyword-only
        """
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable foreign keys and WAL mode for better performance
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")

        # Initialize embedding backend
        if isinstance(embedding_backend, str):
            self._embedding_backend = get_backend(embedding_backend)
        else:
            self._embedding_backend = embedding_backend

        # Try to load sqlite-vec if available
        self._has_vec = self._load_sqlite_vec()

        # Initialize schema
        init_database(self._conn, self._embedding_backend)

        # Ensure vector table exists if we have an embedding backend
        if self._embedding_backend and self._has_vec:
            ensure_vector_table(self._conn, self._embedding_backend.dimensions)

        # Sub-clients
        self.indices = IndicesClient(self)

        # Query compiler
        self._query_compiler = QueryCompiler()

        # Registry for table indexes (bolt-on search over existing tables)
        self._table_indexes: dict[str, TableIndex] = {}

        # Auto-load any previously registered table indexes
        self._load_table_indexes()

    def _load_sqlite_vec(self) -> bool:
        """Try to load sqlite-vec extension."""
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except (ImportError, Exception):
            return False

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Elasticsearch:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

        Creates FTS5 and vector indexes over an existing SQLite table,
        enabling search without duplicating data.

        Args:
            index: ES index name to use for this table
            table: Actual SQLite table name
            id_column: Primary key column name
            text_columns: Columns to include in full-text search
            embedding_text: How to generate text for embeddings:
                - str: column name to embed
                - Callable: function(row_dict) -> str
                - None: concatenate all text_columns
            embedding_backend: Backend name or instance (overrides client default)
            setup: Auto-create FTS5 table and triggers

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
            conn=self._conn,
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
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                SELECT index_name, table_name, id_column, text_columns_json,
                       embedding_text, embedding_backend
                FROM _table_indexes
            """)
        except sqlite3.OperationalError:
            # Table doesn't exist yet (old schema)
            return

        for row in cursor.fetchall():
            index_name, table_name, id_column, text_cols_json, emb_text, emb_backend = row

            # Parse text columns
            text_columns = json.loads(text_cols_json) if text_cols_json else []

            # Resolve embedding backend
            if emb_backend:
                backend = get_backend(emb_backend)
            else:
                backend = self._embedding_backend

            # Recreate TableIndex (FTS5 tables and triggers already exist)
            table_index = TableIndex(
                conn=self._conn,
                table=table_name,
                id_column=id_column,
                text_columns=text_columns,
                embedding_text=emb_text,  # Note: callable not supported for persistence
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
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO _table_indexes
                (index_name, table_name, id_column, text_columns_json,
                 embedding_text, embedding_backend, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                index,
                table,
                id_column,
                json.dumps(text_columns),
                embedding_text if isinstance(embedding_text, str) else None,
                embedding_backend,
                time.time(),
            ),
        )
        self._conn.commit()

    def unregister_table(self, index: str, *, drop_indexes: bool = False) -> bool:
        """Unregister a table index.

        Args:
            index: ES index name to unregister
            drop_indexes: Also drop the FTS5 and vector tables

        Returns:
            True if index was registered, False otherwise
        """
        table_index = self._table_indexes.pop(index, None)
        if not table_index:
            return False

        # Remove from metadata
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM _table_indexes WHERE index_name = ?", (index,))
        self._conn.commit()

        # Optionally drop the FTS5/vector tables
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
        text = extract_text_fields(doc, mappings)

        cursor = self._conn.cursor()
        now = time.time()

        # Check if document exists
        cursor.execute(
            "SELECT _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, doc_id),
        )
        existing = cursor.fetchone()

        if existing:
            if op_type == "create":
                raise ConflictError(f"Document '{doc_id}' already exists in index '{index}'")

            # Update existing document
            new_version = existing[0] + 1
            cursor.execute(
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
            cursor.execute(
                """
                INSERT INTO _documents
                    (_index, _id, _source, _text, _version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (index, doc_id, json.dumps(doc), text, new_version, now, now),
            )
            result = "created"

        # Update vector embedding if backend available
        if self._embedding_backend and self._has_vec and text:
            self._update_embedding(index, doc_id, text)

        if refresh:
            self._conn.commit()
        else:
            self._conn.commit()

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
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT _source, _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        row = cursor.fetchone()

        if not row:
            raise NotFoundError(
                f"Document '{id}' not found in index '{index}'",
                body={"_index": index, "_id": id, "found": False},
            )

        source = json.loads(row[0])
        version = row[1]

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
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        return cursor.fetchone() is not None

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
        cursor = self._conn.cursor()

        # Get version before delete
        cursor.execute(
            "SELECT _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"Document '{id}' not found in index '{index}'")

        version = row[0]

        # Delete document
        cursor.execute(
            "DELETE FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )

        # Delete vector embedding if exists
        if self._has_vec:
            try:
                cursor.execute(
                    "DELETE FROM _documents_vec WHERE doc_key = ?",
                    (f"{index}:{id}",),
                )
            except sqlite3.OperationalError:
                pass

        self._conn.commit()

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
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT _source, _version FROM _documents WHERE _index = ? AND _id = ?",
            (index, id),
        )
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"Document '{id}' not found in index '{index}'")

        existing_source = json.loads(row[0])
        version = row[1]

        # Merge documents
        existing_source.update(doc)

        # Get mappings for text extraction
        mappings = self._get_index_mappings(index)
        text = extract_text_fields(existing_source, mappings)

        # Update
        new_version = version + 1
        now = time.time()
        cursor.execute(
            """
            UPDATE _documents
            SET _source = ?, _text = ?, _version = ?, updated_at = ?
            WHERE _index = ? AND _id = ?
            """,
            (json.dumps(existing_source), text, new_version, now, index, id),
        )

        # Update vector embedding
        if self._embedding_backend and self._has_vec and text:
            self._update_embedding(index, id, text)

        self._conn.commit()

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
            index: Index name(s) to search
            body: Search body with "query" key
            q: Simple query string (alternative to body)
            size: Maximum results to return
            from_: Offset for pagination
            sort: Sort specification
            _source: Source filtering
            mode: Search mode - "keyword" (FTS5), "semantic" (vector), "hybrid" (RRF fusion)

        Returns:
            Search response with hits
        """
        start_time = time.time()

        # Check if this is a registered table index
        if isinstance(index, str) and index in self._table_indexes:
            return self._search_table_index(
                index=index,
                body=body,
                q=q,
                size=size,
                from_=from_,
                mode=mode,
                start_time=start_time,
            )

        # Normalize index to list
        if isinstance(index, str):
            indices = [index]
        elif index:
            indices = list(index)
        else:
            indices = None  # Search all

        # Build query from body or q parameter
        query = {}
        if body:
            query = body.get("query", {})
            size = body.get("size", size)
            from_ = body.get("from", from_)
            sort = body.get("sort", sort)
            _source = body.get("_source", _source)

        if q:
            # Simple query string - search all text fields
            query = {"match": {"_text": q}}

        # Determine search mode
        use_semantic = mode in ("semantic", "hybrid") and self._embedding_backend and self._has_vec
        use_keyword = mode in ("keyword", "hybrid")

        keyword_results = []
        vector_results = []

        if use_keyword:
            keyword_results = self._keyword_search(query, indices, size=size + from_)

        if use_semantic:
            # Extract query text for embedding
            query_text = self._extract_query_text(query, q)
            if query_text:
                vector_results = self._vector_search(query_text, indices, size=size + from_)

        # Get total count for the query (independent of size/pagination)
        total = self._count_matches(query, indices)

        # Combine results
        if mode == "hybrid" and keyword_results and vector_results:
            fused = reciprocal_rank_fusion(keyword_results, vector_results)
            # Convert to hit format
            hits = []
            for doc in fused[from_ : from_ + size]:
                hit = format_hit(doc.index, doc.doc_id, doc.source, score=doc.fused_score)
                hits.append(hit)
            max_score = fused[0].fused_score if fused else None
        elif vector_results and mode == "semantic":
            hits = []
            for idx, doc_id, source, score in vector_results[from_ : from_ + size]:
                hit = format_hit(idx, doc_id, source, score=score)
                hits.append(hit)
            max_score = vector_results[0][3] if vector_results else None
        else:
            # Keyword only
            hits = []
            for idx, doc_id, source, score in keyword_results[from_ : from_ + size]:
                hit = format_hit(idx, doc_id, source, score=score)
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

        query = {}
        if body:
            query = body.get("query", {})

        if q:
            query = {"match": {"_text": q}}

        # Compile and count
        compiled = self._query_compiler.compile(query)

        cursor = self._conn.cursor()
        where_parts = []
        params = []

        if indices:
            placeholders = ",".join("?" for _ in indices)
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        sql = f"SELECT COUNT(*) FROM _documents WHERE {where_clause}"

        cursor.execute(sql, params)
        count = cursor.fetchone()[0]

        return {
            "count": count,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        }

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _search_table_index(
        self,
        index: str,
        body: dict[str, Any] | None,
        q: str | None,
        size: int,
        from_: int,
        mode: Literal["keyword", "semantic", "hybrid"],
        start_time: float,
    ) -> dict[str, Any]:
        """Search a registered table index."""
        table_index = self._table_indexes[index]

        # Extract query text
        query_text = q
        if not query_text and body:
            query = body.get("query", {})
            query_text = self._extract_query_text(query, None)

        if not query_text:
            # No query - return empty results
            took_ms = int((time.time() - start_time) * 1000)
            return format_search_response([], 0, took_ms, None)

        # Search the table index
        results = table_index.search(
            query=query_text,
            mode=mode,
            limit=size,
            offset=from_,
        )

        # Format as ES response
        hits = []
        for result in results:
            hit = format_hit(
                index=index,
                doc_id=str(result["id"]),
                source={"_table_row_id": result["id"]},
                score=result["score"],
            )
            hits.append(hit)

        total = len(results)  # Note: This is the returned count, not true total
        max_score = results[0]["score"] if results else None
        took_ms = int((time.time() - start_time) * 1000)

        return format_search_response(hits, total, took_ms, max_score)

    def _ensure_index(self, index: str) -> None:
        """Ensure index exists, creating if needed."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM _indices WHERE name = ?", (index,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO _indices (name, created_at) VALUES (?, ?)",
                (index, time.time()),
            )
            self._conn.commit()

    def _get_index_mappings(self, index: str) -> dict | None:
        """Get mappings for an index."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT mappings_json FROM _indices WHERE name = ?", (index,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def _count_matches(
        self,
        query: dict[str, Any],
        indices: list[str] | None,
    ) -> int:
        """Count documents matching a query."""
        compiled = self._query_compiler.compile(query)

        cursor = self._conn.cursor()
        where_parts = []
        params = []

        if indices:
            placeholders = ",".join("?" for _ in indices)
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        sql = f"SELECT COUNT(*) FROM _documents WHERE {where_clause}"

        cursor.execute(sql, params)
        return cursor.fetchone()[0]

    def _keyword_search(
        self,
        query: dict[str, Any],
        indices: list[str] | None,
        size: int,
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        """Execute keyword search using FTS5."""
        compiled = self._query_compiler.compile(query)

        cursor = self._conn.cursor()
        where_parts = []
        params = []

        if indices:
            placeholders = ",".join("?" for _ in indices)
            where_parts.append(f"_index IN ({placeholders})")
            params.extend(indices)

        if compiled.where_clause:
            where_parts.append(compiled.where_clause)
            params.extend(compiled.params)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        # Use BM25 scoring if FTS query
        if compiled.uses_fts:
            sql = f"""
                SELECT d._index, d._id, d._source,
                       (SELECT bm25(_documents_fts) FROM _documents_fts
                        WHERE _documents_fts.rowid = d.rowid AND _documents_fts MATCH ?) as score
                FROM _documents d
                WHERE {where_clause}
                ORDER BY score
                LIMIT ?
            """
            # Add FTS query for scoring
            params = [compiled.fts_query] + params + [size]
        else:
            sql = f"""
                SELECT _index, _id, _source, 1.0 as score
                FROM _documents
                WHERE {where_clause}
                LIMIT ?
            """
            params.append(size)

        cursor.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            source = json.loads(row[2])
            score = -row[3] if row[3] else 0.0  # BM25 returns negative scores
            results.append((row[0], row[1], source, score))

        return results

    def _vector_search(
        self,
        query_text: str,
        indices: list[str] | None,
        size: int,
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        """Execute vector similarity search."""
        if not self._embedding_backend or not self._has_vec:
            return []

        # Get query embedding
        query_embedding = self._embedding_backend.embed(query_text)
        embedding_json = json.dumps(query_embedding)

        cursor = self._conn.cursor()

        # Vector search with optional index filter
        if indices:
            # Filter by index prefix
            index_filters = " OR ".join("doc_key LIKE ?" for _ in indices)
            index_params = [f"{idx}:%" for idx in indices]

            sql = f"""
                SELECT v.doc_key, v.distance
                FROM _documents_vec v
                WHERE ({index_filters})
                ORDER BY v.embedding <-> ?
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
            cursor.execute(sql, params)
        except sqlite3.OperationalError:
            return []

        results = []
        for row in cursor.fetchall():
            doc_key, distance = row
            # Parse index:id from doc_key
            parts = doc_key.split(":", 1)
            if len(parts) != 2:
                continue
            idx, doc_id = parts

            # Get full document
            cursor.execute(
                "SELECT _source FROM _documents WHERE _index = ? AND _id = ?",
                (idx, doc_id),
            )
            doc_row = cursor.fetchone()
            if doc_row:
                source = json.loads(doc_row[0])
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - float(distance)
                results.append((idx, doc_id, source, score))

        return results

    def _update_embedding(self, index: str, doc_id: str, text: str) -> None:
        """Update vector embedding for a document."""
        if not self._embedding_backend or not self._has_vec:
            return

        try:
            embedding = self._embedding_backend.embed(text)
            embedding_json = json.dumps(embedding)
            doc_key = f"{index}:{doc_id}"

            cursor = self._conn.cursor()
            # Upsert embedding
            cursor.execute(
                """
                INSERT OR REPLACE INTO _documents_vec (doc_key, embedding)
                VALUES (?, ?)
                """,
                (doc_key, embedding_json),
            )
        except Exception:
            # Don't fail document indexing if embedding fails
            pass

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
