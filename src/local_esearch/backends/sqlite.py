"""SQLite database backend implementation."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend


SCHEMA_VERSION = 2

CORE_SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL
);

-- Index metadata
CREATE TABLE IF NOT EXISTS _indices (
    name TEXT PRIMARY KEY,
    mappings_json TEXT,
    settings_json TEXT,
    created_at REAL
);

-- All documents across all indices
CREATE TABLE IF NOT EXISTS _documents (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    _index TEXT NOT NULL,
    _id TEXT NOT NULL,
    _source TEXT NOT NULL,
    _text TEXT,
    _version INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL,
    UNIQUE (_index, _id)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_documents_index_id ON _documents(_index, _id);
CREATE INDEX IF NOT EXISTS idx_documents_index ON _documents(_index);

-- Registered table indexes (for bolt-on search)
CREATE TABLE IF NOT EXISTS _table_indexes (
    index_name TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    id_column TEXT NOT NULL,
    text_columns_json TEXT NOT NULL,
    embedding_text TEXT,
    embedding_backend TEXT,
    created_at REAL
);
"""

FTS5_SCHEMA = """
-- FTS5 full-text search with porter stemmer
CREATE VIRTUAL TABLE IF NOT EXISTS _documents_fts USING fts5(
    _text,
    content='_documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with documents table
CREATE TRIGGER IF NOT EXISTS _documents_ai AFTER INSERT ON _documents BEGIN
    INSERT INTO _documents_fts(rowid, _text) VALUES (new.rowid, new._text);
END;

CREATE TRIGGER IF NOT EXISTS _documents_ad AFTER DELETE ON _documents BEGIN
    INSERT INTO _documents_fts(_documents_fts, rowid, _text) VALUES('delete', old.rowid, old._text);
END;

CREATE TRIGGER IF NOT EXISTS _documents_au AFTER UPDATE ON _documents BEGIN
    INSERT INTO _documents_fts(_documents_fts, rowid, _text) VALUES('delete', old.rowid, old._text);
    INSERT INTO _documents_fts(rowid, _text) VALUES (new.rowid, new._text);
END;
"""


def _vector_schema(dimensions: int) -> str:
    """Generate vector table schema for given dimensions."""
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS _documents_vec USING vec0(
    doc_key TEXT PRIMARY KEY,
    embedding float[{dimensions}]
);
"""


class SQLiteBackend:
    """SQLite database backend with FTS5 and sqlite-vec support."""

    def __init__(
        self,
        path: str,
        embedding_backend: "EmbeddingBackend | None" = None,
    ):
        """Initialize SQLite backend.

        Args:
            path: Database path or ":memory:"
            embedding_backend: Optional embedding backend for vector search
        """
        self._path = path
        self._embedding_backend = embedding_backend
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable foreign keys and WAL mode for better performance
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")

        self._has_vec = self._load_vec()

    def _load_vec(self) -> bool:
        """Load sqlite-vec extension if available."""
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except (ImportError, sqlite3.OperationalError, Exception):
            return False

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute SQL and return rows as dicts."""
        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def execute_returning_lastrowid(
        self, sql: str, params: Sequence[Any] = ()
    ) -> int | None:
        """Execute SQL and return the last inserted row ID."""
        cursor = self._conn.execute(sql, params)
        return cursor.lastrowid

    def executemany(self, sql: str, params_list: Sequence[Sequence[Any]]) -> None:
        """Execute SQL for multiple parameter sets."""
        self._conn.executemany(sql, params_list)

    def executescript(self, script: str) -> None:
        """Execute multiple SQL statements."""
        self._conn.executescript(script)

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    def init_schema(self, vector_dims: int | None = None) -> None:
        """Initialize core database schema."""
        cursor = self._conn.cursor()

        # Check current schema version
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_version'"
        )
        if cursor.fetchone():
            cursor.execute("SELECT MAX(version) FROM _schema_version")
            row = cursor.fetchone()
            current_version = row[0] if row and row[0] else 0
        else:
            current_version = 0

        if current_version >= SCHEMA_VERSION:
            return

        # Apply core schema
        self._conn.executescript(CORE_SCHEMA)

        # Apply FTS5 schema
        self._conn.executescript(FTS5_SCHEMA)

        # Apply vector schema if dimensions provided
        if vector_dims is not None and self._has_vec:
            try:
                self._conn.executescript(_vector_schema(vector_dims))
            except sqlite3.OperationalError:
                pass

        # Record schema version
        cursor.execute(
            "INSERT OR REPLACE INTO _schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, time.time()),
        )

        self._conn.commit()

    def create_documents_vec_table(self, dims: int) -> None:
        """Create the _documents_vec table for document embeddings."""
        if not self._has_vec:
            return
        try:
            self._conn.executescript(_vector_schema(dims))
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def create_table_fts(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create FTS5 virtual table for a registered table."""
        columns = ", ".join(text_columns)
        sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(
                {columns},
                content='{source_table}',
                content_rowid='{id_column}',
                tokenize='porter unicode61'
            )
        """
        self._conn.execute(sql)
        self._conn.commit()

    def create_table_fts_triggers(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create triggers to sync FTS with source table."""
        col_list = ", ".join(text_columns)
        new_vals = ", ".join(f"new.{c}" for c in text_columns)
        old_vals = ", ".join(f"old.{c}" for c in text_columns)

        # Insert trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts_table}_ai AFTER INSERT ON {source_table} BEGIN
                INSERT INTO {fts_table}(rowid, {col_list})
                VALUES (new.{id_column}, {new_vals});
            END
        """)

        # Delete trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts_table}_ad AFTER DELETE ON {source_table} BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, {col_list})
                VALUES ('delete', old.{id_column}, {old_vals});
            END
        """)

        # Update trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts_table}_au AFTER UPDATE ON {source_table} BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, {col_list})
                VALUES ('delete', old.{id_column}, {old_vals});
                INSERT INTO {fts_table}(rowid, {col_list})
                VALUES (new.{id_column}, {new_vals});
            END
        """)

        self._conn.commit()

    def create_table_vec(self, vec_table: str, dims: int) -> None:
        """Create vector table for registered table chunks."""
        if not self._has_vec:
            return
        sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table} USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dims}]
            )
        """
        self._conn.execute(sql)
        self._conn.commit()

    def create_chunks_table(self, chunks_table: str) -> None:
        """Create chunks metadata table."""
        sql = f"""
            CREATE TABLE IF NOT EXISTS {chunks_table} (
                chunk_id TEXT PRIMARY KEY,
                row_id INTEGER NOT NULL,
                chunk_idx INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER
            )
        """
        self._conn.execute(sql)
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{chunks_table}_row ON {chunks_table}(row_id)"
        )
        self._conn.commit()

    def create_chunks_delete_triggers(
        self,
        source_table: str,
        chunks_table: str,
        vec_table: str,
        id_column: str,
    ) -> None:
        """Create triggers to clean up chunks when source rows deleted."""
        # Delete from chunks table
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {chunks_table}_ad AFTER DELETE ON {source_table} BEGIN
                DELETE FROM {chunks_table} WHERE row_id = old.{id_column};
            END
        """)

        # Delete from vector table
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {vec_table}_ad AFTER DELETE ON {source_table} BEGIN
                DELETE FROM {vec_table} WHERE chunk_id LIKE (old.{id_column} || ':%');
            END
        """)

        self._conn.commit()

    def rebuild_fts(self, fts_table: str, source_table: str) -> None:
        """Rebuild FTS index from source table."""
        self._conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")
        self._conn.commit()

    def drop_table_indexes(
        self,
        fts_table: str,
        vec_table: str,
        chunks_table: str,
    ) -> None:
        """Drop FTS table, triggers, vector table, and chunks table."""
        # Drop FTS triggers
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts_table}_ai")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts_table}_ad")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts_table}_au")

        # Drop chunks/vec cleanup triggers
        self._conn.execute(f"DROP TRIGGER IF EXISTS {chunks_table}_ad")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {vec_table}_ad")

        # Drop tables
        self._conn.execute(f"DROP TABLE IF EXISTS {fts_table}")
        self._conn.execute(f"DROP TABLE IF EXISTS {chunks_table}")
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {vec_table}")
        except sqlite3.OperationalError:
            pass

        self._conn.commit()

    # -------------------------------------------------------------------------
    # Full-Text Search
    # -------------------------------------------------------------------------

    def fts_match_clause(self, fts_table: str = "_documents_fts") -> str:
        """Return SQL clause for FTS matching."""
        return f"rowid IN (SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH ?)"

    def fts_score_sql(self, fts_table: str = "_documents_fts") -> str:
        """Return SQL expression for FTS relevance score."""
        return f"bm25({fts_table})"

    def table_fts_search_sql(
        self,
        fts_table: str,
        source_table: str,
        id_column: str,
    ) -> str:
        """Return SQL for searching a registered table's FTS index."""
        return f"""
            SELECT t.{id_column}, bm25({fts_table}) as score
            FROM {fts_table}
            JOIN {source_table} t ON {fts_table}.rowid = t.{id_column}
            WHERE {fts_table} MATCH ?
            ORDER BY score
            LIMIT ?
        """

    def document_fts_search_sql(
        self,
        where_clause: str,
        uses_fts: bool,
        fts_query: str | None,
    ) -> tuple[str, int]:
        """Return SQL for searching _documents with FTS.

        Args:
            where_clause: WHERE conditions (already includes index filter)
            uses_fts: Whether FTS is being used
            fts_query: The FTS query string (if uses_fts)

        Returns:
            Tuple of (sql, num_fts_params) where num_fts_params is number
            of additional params needed for the FTS query (1 for SQLite)
        """
        if uses_fts:
            sql = f"""
                SELECT d._index, d._id, d._source,
                       (SELECT bm25(_documents_fts) FROM _documents_fts
                        WHERE _documents_fts.rowid = d.rowid AND _documents_fts MATCH ?) as score
                FROM _documents d
                WHERE {where_clause}
                ORDER BY score
                LIMIT ?
            """
            return sql, 1
        else:
            sql = f"""
                SELECT _index, _id, _source, 1.0 as score
                FROM _documents
                WHERE {where_clause}
                LIMIT ?
            """
            return sql, 0

    # -------------------------------------------------------------------------
    # Vector Search
    # -------------------------------------------------------------------------

    def vector_available(self) -> bool:
        """Check if vector search is available."""
        return self._has_vec

    def vector_upsert(self, table: str, key: str, embedding: list[float]) -> None:
        """Insert or update a vector embedding."""
        embedding_json = json.dumps(embedding)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (doc_key, embedding) VALUES (?, ?)",
            (key, embedding_json),
        )

    def vector_upsert_chunk(
        self, table: str, chunk_id: str, embedding: list[float]
    ) -> None:
        """Insert or update a chunk vector embedding."""
        embedding_json = json.dumps(embedding)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, embedding_json),
        )

    def vector_delete(self, table: str, key: str) -> None:
        """Delete a vector embedding."""
        try:
            self._conn.execute(f"DELETE FROM {table} WHERE doc_key = ?", (key,))
        except sqlite3.OperationalError:
            pass

    def vector_delete_pattern(self, table: str, pattern: str) -> None:
        """Delete vector embeddings matching a pattern."""
        try:
            self._conn.execute(f"DELETE FROM {table} WHERE doc_key LIKE ?", (pattern,))
        except sqlite3.OperationalError:
            pass

    def vector_search_sql(
        self,
        table: str = "_documents_vec",
        has_filter: bool = False,
    ) -> str:
        """Return SQL for vector similarity search."""
        if has_filter:
            return f"""
                SELECT doc_key, distance
                FROM {table}
                WHERE ({{filter_clause}})
                ORDER BY embedding <-> ?
                LIMIT ?
            """
        return f"""
            SELECT doc_key, distance
            FROM {table}
            ORDER BY embedding <-> ?
            LIMIT ?
        """

    def vector_search_chunks_sql(
        self,
        vec_table: str,
        chunks_table: str,
    ) -> str:
        """Return SQL for vector search with chunk metadata join."""
        return f"""
            SELECT vec_results.chunk_id, vec_results.distance, c.row_id, c.chunk_idx, c.chunk_text
            FROM (
                SELECT chunk_id, distance
                FROM {vec_table}
                WHERE embedding MATCH ? AND k = ?
            ) vec_results
            JOIN {chunks_table} c ON vec_results.chunk_id = c.chunk_id
            ORDER BY vec_results.distance
        """

    # -------------------------------------------------------------------------
    # Query Compilation Helpers
    # -------------------------------------------------------------------------

    def json_extract(self, field: str) -> str:
        """Return SQL fragment for extracting JSON field from _source."""
        return f"json_extract(_source, ?)"

    def json_extract_literal(self, path: str) -> str:
        """Return SQL fragment with literal path for extracting JSON field."""
        return f"json_extract(_source, '{path}')"

    def json_path(self, field: str) -> str:
        """Convert field name to SQLite JSON path ($.field)."""
        if field.startswith("$."):
            return field
        return f"$.{field}"

    def placeholder(self, n: int = 1) -> str:
        """Return placeholder(s) for parameterized queries."""
        return ", ".join("?" for _ in range(n))

    def placeholders(self, n: int) -> str:
        """Return n placeholders comma-separated."""
        return ", ".join("?" for _ in range(n))

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cursor.fetchone() is not None

    def clear_table(self, table: str) -> None:
        """Delete all rows from a table."""
        self._conn.execute(f"DELETE FROM {table}")

    def get_table_columns(self, table: str) -> list[str]:
        """Get column names for a table."""
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]

    def timestamp_now(self) -> float:
        """Return current timestamp as Unix time (float)."""
        return time.time()

    def document_fts_count_sql(
        self,
        where_clause: str,
        uses_fts: bool,
    ) -> tuple[str, int]:
        """Return SQL for counting _documents with FTS."""
        if uses_fts:
            sql = f"""
                SELECT COUNT(*) as cnt FROM _documents d
                WHERE {where_clause}
                  AND d.rowid IN (SELECT rowid FROM _documents_fts WHERE _documents_fts MATCH ?)
            """
            return sql, 1
        else:
            sql = f"SELECT COUNT(*) as cnt FROM _documents WHERE {where_clause}"
            return sql, 0

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
        """Insert or update a table registration."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO _table_indexes
                (index_name, table_name, id_column, text_columns_json,
                 embedding_text, embedding_backend, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (index, table, id_column, text_columns_json,
             embedding_text, embedding_backend, created_at),
        )

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
        """Insert or update a chunk record."""
        self._conn.execute(
            f"""INSERT OR REPLACE INTO {chunks_table}
                (chunk_id, row_id, chunk_idx, chunk_text, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_id, row_id, chunk_idx, chunk_text, start_char, end_char),
        )

    # -------------------------------------------------------------------------
    # Direct connection access (for compatibility during migration)
    # -------------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the underlying SQLite connection."""
        return self._conn
