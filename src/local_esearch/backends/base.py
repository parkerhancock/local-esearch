"""Base protocol and types for database backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class FTSResult:
    """Result from a full-text search operation."""

    rowid: int
    score: float


@dataclass
class VectorResult:
    """Result from a vector similarity search."""

    key: str
    distance: float


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol defining the database backend interface.

    Implementations handle database-specific SQL dialects, FTS engines,
    and vector search capabilities.
    """

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute SQL and return rows as dicts.

        Args:
            sql: SQL statement with placeholders
            params: Parameter values

        Returns:
            List of row dicts with column names as keys
        """
        ...

    def execute_returning_lastrowid(
        self, sql: str, params: Sequence[Any] = ()
    ) -> int | None:
        """Execute SQL and return the last inserted row ID.

        Args:
            sql: SQL statement
            params: Parameter values

        Returns:
            Last inserted row ID, or None if not applicable
        """
        ...

    def executemany(self, sql: str, params_list: Sequence[Sequence[Any]]) -> None:
        """Execute SQL for multiple parameter sets.

        Args:
            sql: SQL statement with placeholders
            params_list: List of parameter tuples
        """
        ...

    def executescript(self, script: str) -> None:
        """Execute multiple SQL statements.

        Args:
            script: SQL script with multiple statements
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    def init_schema(self, vector_dims: int | None = None) -> None:
        """Initialize core database schema.

        Creates _schema_version, _indices, _documents, _table_indexes tables
        and FTS index.

        Args:
            vector_dims: Vector dimensions if embedding backend is configured
        """
        ...

    def create_documents_vec_table(self, dims: int) -> None:
        """Create the _documents_vec table for document embeddings.

        Args:
            dims: Embedding dimensions
        """
        ...

    def create_table_fts(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create FTS index for a registered table.

        Args:
            fts_table: Name for the FTS table/index
            source_table: Source table to index
            text_columns: Columns to include in FTS
            id_column: Primary key column
        """
        ...

    def create_table_fts_triggers(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create triggers to sync FTS with source table.

        Args:
            fts_table: FTS table name
            source_table: Source table name
            text_columns: Indexed columns
            id_column: Primary key column
        """
        ...

    def create_table_vec(self, vec_table: str, dims: int) -> None:
        """Create vector table for registered table chunks.

        Args:
            vec_table: Vector table name
            dims: Embedding dimensions
        """
        ...

    def create_chunks_table(self, chunks_table: str) -> None:
        """Create chunks metadata table.

        Args:
            chunks_table: Chunks table name
        """
        ...

    def create_chunks_delete_triggers(
        self,
        source_table: str,
        chunks_table: str,
        vec_table: str,
        id_column: str,
    ) -> None:
        """Create triggers to clean up chunks when source rows deleted.

        Args:
            source_table: Source table name
            chunks_table: Chunks table name
            vec_table: Vector table name
            id_column: Primary key column
        """
        ...

    def rebuild_fts(self, fts_table: str, source_table: str) -> None:
        """Rebuild FTS index from source table.

        Args:
            fts_table: FTS table name
            source_table: Source table name
        """
        ...

    def drop_table_indexes(
        self,
        fts_table: str,
        vec_table: str,
        chunks_table: str,
    ) -> None:
        """Drop FTS table, triggers, vector table, and chunks table.

        Args:
            fts_table: FTS table name
            vec_table: Vector table name
            chunks_table: Chunks table name
        """
        ...

    # -------------------------------------------------------------------------
    # Full-Text Search
    # -------------------------------------------------------------------------

    def fts_match_clause(self, fts_table: str) -> str:
        """Return SQL clause for FTS matching in documents table.

        Args:
            fts_table: FTS table name (ignored for _documents_fts)

        Returns:
            SQL fragment like "rowid IN (SELECT rowid FROM _documents_fts WHERE ...)"
        """
        ...

    def fts_score_sql(self, fts_table: str) -> str:
        """Return SQL expression for FTS relevance score.

        Args:
            fts_table: FTS table name

        Returns:
            SQL expression that returns a score (lower is better for SQLite BM25)
        """
        ...

    def document_fts_search_sql(
        self,
        where_clause: str,
        uses_fts: bool,
    ) -> tuple[str, list[str]]:
        """Return SQL for searching _documents table with FTS.

        Args:
            where_clause: Additional WHERE conditions
            uses_fts: Whether FTS query is involved

        Returns:
            Tuple of (sql_template, param_names) where param_names indicates
            which params are needed (e.g., ['fts_query', 'indices', 'size'])
        """
        ...

    def table_fts_search_sql(
        self,
        fts_table: str,
        source_table: str,
        id_column: str,
    ) -> str:
        """Return SQL for searching a registered table's FTS index.

        Args:
            fts_table: FTS table name
            source_table: Source table name
            id_column: Primary key column

        Returns:
            SQL with ? placeholder for query and limit
        """
        ...

    # -------------------------------------------------------------------------
    # Vector Search
    # -------------------------------------------------------------------------

    def vector_available(self) -> bool:
        """Check if vector search is available.

        Returns:
            True if vector extension is loaded
        """
        ...

    def vector_upsert(self, table: str, key: str, embedding: list[float]) -> None:
        """Insert or update a vector embedding.

        Args:
            table: Vector table name
            key: Document key
            embedding: Embedding vector
        """
        ...

    def vector_delete(self, table: str, key: str) -> None:
        """Delete a vector embedding.

        Args:
            table: Vector table name
            key: Document key
        """
        ...

    def vector_delete_pattern(self, table: str, pattern: str) -> None:
        """Delete vector embeddings matching a pattern.

        Args:
            table: Vector table name
            pattern: LIKE pattern to match keys
        """
        ...

    def vector_search_sql(
        self,
        table: str,
        has_filter: bool = False,
    ) -> str:
        """Return SQL for vector similarity search.

        Args:
            table: Vector table name
            has_filter: Whether to include filter clause

        Returns:
            SQL with placeholders for embedding, limit, and optional filter
        """
        ...

    def vector_search_chunks_sql(
        self,
        vec_table: str,
        chunks_table: str,
    ) -> str:
        """Return SQL for vector search with chunk metadata join.

        Args:
            vec_table: Vector table name
            chunks_table: Chunks table name

        Returns:
            SQL with placeholders for embedding and k value
        """
        ...

    # -------------------------------------------------------------------------
    # Query Compilation Helpers
    # -------------------------------------------------------------------------

    def json_extract(self, field: str) -> str:
        """Return SQL fragment for extracting JSON field from _source.

        Args:
            field: JSON path like "$.title"

        Returns:
            SQL fragment like "json_extract(_source, '$.title')"
        """
        ...

    def json_path(self, field: str) -> str:
        """Convert field name to database-specific JSON path.

        Args:
            field: ES field name like "title" or "author.name"

        Returns:
            "$.title" for SQLite, "title" for PostgreSQL
        """
        ...

    def placeholder(self, n: int = 1) -> str:
        """Return placeholder(s) for parameterized queries.

        Args:
            n: Number of placeholders

        Returns:
            "?" for SQLite, "%s" for PostgreSQL (n times, comma-separated)
        """
        ...

    def placeholders(self, n: int) -> str:
        """Return n placeholders comma-separated.

        Args:
            n: Number of placeholders

        Returns:
            "?, ?, ?" for SQLite, "%s, %s, %s" for PostgreSQL
        """
        ...

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def table_exists(self, table: str) -> bool:
        """Check if a table exists.

        Args:
            table: Table name

        Returns:
            True if table exists
        """
        ...

    def clear_table(self, table: str) -> None:
        """Delete all rows from a table.

        Args:
            table: Table name
        """
        ...

    def get_table_columns(self, table: str) -> list[str]:
        """Get column names for a table.

        Args:
            table: Table name

        Returns:
            List of column names
        """
        ...

    def timestamp_now(self) -> float | str:
        """Return current timestamp in database-appropriate format.

        Returns:
            time.time() for SQLite, datetime for PostgreSQL
        """
        ...

    def document_fts_count_sql(
        self,
        where_clause: str,
        uses_fts: bool,
    ) -> tuple[str, int]:
        """Return SQL for counting _documents with FTS.

        Args:
            where_clause: WHERE conditions (already includes index filter)
            uses_fts: Whether FTS is being used

        Returns:
            Tuple of (sql, num_fts_params) where num_fts_params is number
            of additional params needed for the FTS query
        """
        ...

    def upsert_table_registration(
        self,
        index: str,
        table: str,
        id_column: str,
        text_columns_json: str,
        embedding_text: str | None,
        embedding_backend: str | None,
        created_at: float | str,
    ) -> None:
        """Insert or update a table registration.

        Args:
            index: Index name
            table: Table name
            id_column: Primary key column
            text_columns_json: JSON-encoded text columns list
            embedding_text: Embedding text column/template
            embedding_backend: Embedding backend name
            created_at: Creation timestamp
        """
        ...

    def upsert_chunk(
        self,
        chunks_table: str,
        chunk_id: str,
        row_id: int,
        chunk_idx: int,
        chunk_text: str,
        start_char: int | None,
        end_char: int | None,
    ) -> None:
        """Insert or update a chunk record.

        Args:
            chunks_table: Chunks table name
            chunk_id: Chunk ID (row_id:chunk_idx)
            row_id: Source row ID
            chunk_idx: Chunk index within row
            chunk_text: Chunk text content
            start_char: Start character position
            end_char: End character position
        """
        ...
