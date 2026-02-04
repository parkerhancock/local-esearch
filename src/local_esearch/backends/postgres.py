"""PostgreSQL database backend implementation."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend


SCHEMA_VERSION = 2


class PostgresBackend:
    """PostgreSQL database backend with tsvector FTS and pgvector support."""

    def __init__(
        self,
        dsn: str,
        embedding_backend: "EmbeddingBackend | None" = None,
    ):
        """Initialize PostgreSQL backend.

        Args:
            dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/db")
            embedding_backend: Optional embedding backend for vector search
        """
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "psycopg is required for PostgreSQL support. "
                "Install with: pip install 'local-esearch[postgres]'"
            ) from e

        self._dsn = dsn
        self._embedding_backend = embedding_backend
        self._conn = psycopg.connect(dsn, autocommit=False)
        self._has_pgvector = self._check_pgvector()

    def _check_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if cur.fetchone():
                    return True
                # Try to create it
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self._conn.commit()
                return True
        except Exception:
            self._conn.rollback()
            return False

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute SQL and return rows as dicts."""
        # Convert SQLite-style ? placeholders to PostgreSQL %s
        pg_sql = self._convert_placeholders(sql)

        with self._conn.cursor() as cur:
            cur.execute(pg_sql, params)
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def execute_returning_lastrowid(
        self, sql: str, params: Sequence[Any] = ()
    ) -> int | None:
        """Execute SQL and return the last inserted row ID."""
        pg_sql = self._convert_placeholders(sql)

        # Add RETURNING clause for INSERT if not present
        if "INSERT" in pg_sql.upper() and "RETURNING" not in pg_sql.upper():
            pg_sql = pg_sql.rstrip(";") + " RETURNING id"

        with self._conn.cursor() as cur:
            cur.execute(pg_sql, params)
            row = cur.fetchone()
            return row[0] if row else None

    def executemany(self, sql: str, params_list: Sequence[Sequence[Any]]) -> None:
        """Execute SQL for multiple parameter sets."""
        pg_sql = self._convert_placeholders(sql)
        with self._conn.cursor() as cur:
            cur.executemany(pg_sql, params_list)

    def executescript(self, script: str) -> None:
        """Execute multiple SQL statements."""
        # PostgreSQL can execute multiple statements directly
        with self._conn.cursor() as cur:
            cur.execute(script)

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _convert_placeholders(self, sql: str) -> str:
        """Convert SQLite ? placeholders to PostgreSQL %s."""
        # Simple conversion - replace ? with %s
        # This is a basic implementation; complex cases may need more work
        return sql.replace("?", "%s")

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    def init_schema(self, vector_dims: int | None = None) -> None:
        """Initialize core database schema."""
        with self._conn.cursor() as cur:
            # Check current schema version
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public' AND tablename = '_schema_version'
                )
            """)
            exists = cur.fetchone()[0]

            if exists:
                cur.execute("SELECT MAX(version) FROM _schema_version")
                row = cur.fetchone()
                current_version = row[0] if row and row[0] else 0
            else:
                current_version = 0

            if current_version >= SCHEMA_VERSION:
                return

            # Create core tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS _indices (
                    name TEXT PRIMARY KEY,
                    mappings_json JSONB,
                    settings_json JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS _documents (
                    id BIGSERIAL PRIMARY KEY,
                    _index TEXT NOT NULL,
                    _id TEXT NOT NULL,
                    _source JSONB NOT NULL,
                    _text TEXT,
                    _tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(_text, ''))) STORED,
                    _version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (_index, _id)
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tsv ON _documents USING GIN(_tsv)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_index_id ON _documents(_index, _id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_index ON _documents(_index)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS _table_indexes (
                    index_name TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    id_column TEXT NOT NULL,
                    text_columns_json JSONB NOT NULL,
                    embedding_text TEXT,
                    embedding_backend TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Apply vector schema if dimensions provided and pgvector available
            if vector_dims is not None and self._has_pgvector:
                self._create_vector_table(cur, "_documents_vec", vector_dims)

            # Record schema version
            cur.execute("""
                INSERT INTO _schema_version (version, applied_at)
                VALUES (%s, CURRENT_TIMESTAMP)
                ON CONFLICT (version) DO NOTHING
            """, (SCHEMA_VERSION,))

        self._conn.commit()

    def _create_vector_table(self, cur: Any, table: str, dims: int) -> None:
        """Create a vector table."""
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doc_key TEXT PRIMARY KEY,
                embedding vector({dims})
            )
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_embedding
            ON {table} USING hnsw (embedding vector_cosine_ops)
        """)

    def create_documents_vec_table(self, dims: int) -> None:
        """Create the _documents_vec table for document embeddings."""
        if not self._has_pgvector:
            return
        with self._conn.cursor() as cur:
            self._create_vector_table(cur, "_documents_vec", dims)
        self._conn.commit()

    def create_table_fts(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create FTS support for a registered table.

        For PostgreSQL, we add a tsvector column and GIN index to the table.
        """
        # Check if tsvector column already exists
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = '_tsv'
            """, (source_table,))

            if not cur.fetchone():
                # Add tsvector column
                text_concat = " || ' ' || ".join(f"COALESCE({col}, '')" for col in text_columns)
                cur.execute(f"""
                    ALTER TABLE {source_table}
                    ADD COLUMN _tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', {text_concat})) STORED
                """)

                # Create GIN index
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{source_table}_tsv
                    ON {source_table} USING GIN(_tsv)
                """)

        self._conn.commit()

    def create_table_fts_triggers(
        self,
        fts_table: str,
        source_table: str,
        text_columns: list[str],
        id_column: str,
    ) -> None:
        """Create triggers for FTS sync.

        For PostgreSQL with generated tsvector column, no triggers needed.
        """
        # PostgreSQL GENERATED ALWAYS AS handles this automatically
        pass

    def create_table_vec(self, vec_table: str, dims: int) -> None:
        """Create vector table for registered table chunks."""
        if not self._has_pgvector:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {vec_table} (
                    chunk_id TEXT PRIMARY KEY,
                    embedding vector({dims})
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{vec_table}_embedding
                ON {vec_table} USING hnsw (embedding vector_cosine_ops)
            """)
        self._conn.commit()

    def create_chunks_table(self, chunks_table: str) -> None:
        """Create chunks metadata table."""
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {chunks_table} (
                    chunk_id TEXT PRIMARY KEY,
                    row_id BIGINT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{chunks_table}_row
                ON {chunks_table}(row_id)
            """)
        self._conn.commit()

    def create_chunks_delete_triggers(
        self,
        source_table: str,
        chunks_table: str,
        vec_table: str,
        id_column: str,
    ) -> None:
        """Create triggers to clean up chunks when source rows deleted."""
        with self._conn.cursor() as cur:
            # Create function and trigger for chunks cleanup
            cur.execute(f"""
                CREATE OR REPLACE FUNCTION {chunks_table}_cleanup()
                RETURNS TRIGGER AS $$
                BEGIN
                    DELETE FROM {chunks_table} WHERE row_id = OLD.{id_column};
                    DELETE FROM {vec_table} WHERE chunk_id LIKE (OLD.{id_column}::text || ':%');
                    RETURN OLD;
                END;
                $$ LANGUAGE plpgsql
            """)

            cur.execute(f"""
                DROP TRIGGER IF EXISTS {chunks_table}_cleanup_trigger ON {source_table}
            """)

            cur.execute(f"""
                CREATE TRIGGER {chunks_table}_cleanup_trigger
                AFTER DELETE ON {source_table}
                FOR EACH ROW EXECUTE FUNCTION {chunks_table}_cleanup()
            """)

        self._conn.commit()

    def rebuild_fts(self, fts_table: str, source_table: str) -> None:
        """Rebuild FTS index from source table.

        For PostgreSQL with generated tsvector, this is a no-op as the
        column is automatically maintained.
        """
        # Force index refresh by running REINDEX
        with self._conn.cursor() as cur:
            cur.execute(f"REINDEX INDEX idx_{source_table}_tsv")
        self._conn.commit()

    def drop_table_indexes(
        self,
        fts_table: str,
        vec_table: str,
        chunks_table: str,
    ) -> None:
        """Drop FTS triggers, vector table, and chunks table."""
        with self._conn.cursor() as cur:
            # Drop chunks cleanup trigger and function
            cur.execute(f"DROP TRIGGER IF EXISTS {chunks_table}_cleanup_trigger ON {chunks_table}")
            cur.execute(f"DROP FUNCTION IF EXISTS {chunks_table}_cleanup()")

            # Drop tables
            cur.execute(f"DROP TABLE IF EXISTS {chunks_table}")
            cur.execute(f"DROP TABLE IF EXISTS {vec_table}")

        self._conn.commit()

    # -------------------------------------------------------------------------
    # Full-Text Search
    # -------------------------------------------------------------------------

    def fts_match_clause(self, fts_table: str = "_documents_fts") -> str:
        """Return SQL clause for FTS matching."""
        return "_tsv @@ plainto_tsquery('english', %s)"

    def fts_score_sql(self, fts_table: str = "_documents_fts") -> str:
        """Return SQL expression for FTS relevance score."""
        return "ts_rank_cd(_tsv, plainto_tsquery('english', %s))"

    def table_fts_search_sql(
        self,
        fts_table: str,
        source_table: str,
        id_column: str,
    ) -> str:
        """Return SQL for searching a registered table's FTS index."""
        # Use CTE to store the query once, keeping same param count as SQLite (query, limit)
        return f"""
            WITH q AS (SELECT plainto_tsquery('english', %s) as query)
            SELECT {id_column}, ts_rank_cd(_tsv, q.query) as score
            FROM {source_table}, q
            WHERE _tsv @@ q.query
            ORDER BY score DESC
            LIMIT %s
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
            of additional params needed for the FTS query (1 for PostgreSQL using CTE)
        """
        if uses_fts:
            # Use CTE to store the query once, avoiding param ordering issues
            sql = f"""
                WITH q AS (SELECT plainto_tsquery('english', %s) as query)
                SELECT _index, _id, _source,
                       ts_rank_cd(_tsv, q.query) as score
                FROM _documents, q
                WHERE {where_clause} AND _tsv @@ q.query
                ORDER BY score DESC
                LIMIT %s
            """
            return sql, 1
        else:
            sql = f"""
                SELECT _index, _id, _source, 1.0 as score
                FROM _documents
                WHERE {where_clause}
                LIMIT %s
            """
            return sql, 0

    # -------------------------------------------------------------------------
    # Vector Search
    # -------------------------------------------------------------------------

    def vector_available(self) -> bool:
        """Check if vector search is available."""
        return self._has_pgvector

    def vector_upsert(self, table: str, key: str, embedding: list[float]) -> None:
        """Insert or update a vector embedding."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        with self._conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table} (doc_key, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (doc_key) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (key, embedding_str))

    def vector_upsert_chunk(
        self, table: str, chunk_id: str, embedding: list[float]
    ) -> None:
        """Insert or update a chunk vector embedding."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        with self._conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table} (chunk_id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (chunk_id, embedding_str))

    def vector_delete(self, table: str, key: str) -> None:
        """Delete a vector embedding."""
        if not self.table_exists(table):
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE doc_key = %s", (key,))

    def vector_delete_pattern(self, table: str, pattern: str) -> None:
        """Delete vector embeddings matching a pattern."""
        if not self.table_exists(table):
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE doc_key LIKE %s", (pattern,))

    def vector_search_sql(
        self,
        table: str = "_documents_vec",
        has_filter: bool = False,
    ) -> str:
        """Return SQL for vector similarity search."""
        if has_filter:
            return f"""
                SELECT doc_key, embedding <=> %s::vector as distance
                FROM {table}
                WHERE ({{filter_clause}})
                ORDER BY distance
                LIMIT %s
            """
        return f"""
            SELECT doc_key, embedding <=> %s::vector as distance
            FROM {table}
            ORDER BY distance
            LIMIT %s
        """

    def vector_search_chunks_sql(
        self,
        vec_table: str,
        chunks_table: str,
    ) -> str:
        """Return SQL for vector search with chunk metadata join."""
        return f"""
            SELECT v.chunk_id, v.embedding <=> %s::vector as distance,
                   c.row_id, c.chunk_idx, c.chunk_text
            FROM {vec_table} v
            JOIN {chunks_table} c ON v.chunk_id = c.chunk_id
            ORDER BY distance
            LIMIT %s
        """

    # -------------------------------------------------------------------------
    # Query Compilation Helpers
    # -------------------------------------------------------------------------

    def json_extract(self, field: str) -> str:
        """Return SQL fragment for extracting JSON field from _source."""
        return "_source->>%s"

    def json_extract_literal(self, path: str) -> str:
        """Return SQL fragment with literal path for extracting JSON field."""
        # Remove $. prefix if present and use PostgreSQL JSONB syntax
        if path.startswith("$."):
            path = path[2:]
        return f"_source->>'{path}'"

    def json_path(self, field: str) -> str:
        """Convert field name to PostgreSQL JSON key (just the field name)."""
        # PostgreSQL ->> operator expects just the key name, not $.prefix
        if field.startswith("$."):
            return field[2:]
        return field

    def placeholder(self, n: int = 1) -> str:
        """Return placeholder(s) for parameterized queries."""
        return ", ".join("%s" for _ in range(n))

    def placeholders(self, n: int) -> str:
        """Return n placeholders comma-separated."""
        return ", ".join("%s" for _ in range(n))

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public' AND tablename = %s
                )
            """, (table,))
            return cur.fetchone()[0]

    def clear_table(self, table: str) -> None:
        """Delete all rows from a table."""
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table}")

    def get_table_columns(self, table: str) -> list[str]:
        """Get column names for a table."""
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table,))
            return [row[0] for row in cur.fetchall()]

    def timestamp_now(self) -> str:
        """Return current timestamp as ISO format string for PostgreSQL."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def document_fts_count_sql(
        self,
        where_clause: str,
        uses_fts: bool,
    ) -> tuple[str, int]:
        """Return SQL for counting _documents with FTS."""
        if uses_fts:
            sql = f"""
                SELECT COUNT(*) as cnt FROM _documents
                WHERE {where_clause} AND _tsv @@ plainto_tsquery('english', %s)
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
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO _table_indexes
                    (index_name, table_name, id_column, text_columns_json,
                     embedding_text, embedding_backend, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (index_name) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    id_column = EXCLUDED.id_column,
                    text_columns_json = EXCLUDED.text_columns_json,
                    embedding_text = EXCLUDED.embedding_text,
                    embedding_backend = EXCLUDED.embedding_backend,
                    created_at = EXCLUDED.created_at
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
        with self._conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {chunks_table}
                    (chunk_id, row_id, chunk_idx, chunk_text, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        row_id = EXCLUDED.row_id,
                        chunk_idx = EXCLUDED.chunk_idx,
                        chunk_text = EXCLUDED.chunk_text,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char""",
                (chunk_id, row_id, chunk_idx, chunk_text, start_char, end_char),
            )
