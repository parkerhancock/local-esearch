"""Table index for bolt-on search over existing SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from local_esearch.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend


@dataclass
class TableConfig:
    """Configuration for a registered table index."""

    table: str
    id_column: str
    text_columns: list[str]
    embedding_text: Callable[[dict], str] | str | None = None
    embedding_backend: EmbeddingBackend | None = None
    fts_table: str = field(default="")
    vec_table: str = field(default="")

    def __post_init__(self):
        self.fts_table = f"{self.table}_fts"
        self.vec_table = f"{self.table}_vec"


class TableIndex:
    """Search index over an existing SQLite table.

    Creates FTS5 and vector indexes that reference the original table,
    enabling full-text and semantic search without duplicating data.

    Example:
        index = TableIndex(
            conn=conn,
            table="documents",
            id_column="id",
            text_columns=["title", "content"],
            embedding_backend=VoyageBackend(),
        )
        index.setup()
        index.reindex()

        results = index.search("machine learning", mode="hybrid")
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table: str,
        id_column: str = "id",
        text_columns: list[str] | None = None,
        embedding_text: Callable[[dict], str] | str | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ):
        """Initialize table index.

        Args:
            conn: SQLite connection
            table: Name of existing table to index
            id_column: Primary key column name
            text_columns: Columns to include in FTS5 index
            embedding_text: How to generate text for embeddings:
                - str: column name to embed
                - Callable: function(row_dict) -> str
                - None: concatenate all text_columns
            embedding_backend: Embedding backend for vector search
        """
        self._conn = conn
        self.config = TableConfig(
            table=table,
            id_column=id_column,
            text_columns=text_columns or [],
            embedding_text=embedding_text,
            embedding_backend=embedding_backend,
        )
        self._has_vec = self._check_vec_available()

    def _check_vec_available(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            self._conn.execute("SELECT vec_version()")
            return True
        except sqlite3.OperationalError:
            return False

    def setup(self) -> None:
        """Create FTS5 table, triggers, and vector table.

        Safe to call multiple times - uses IF NOT EXISTS.
        """
        self._create_fts_table()
        self._create_fts_triggers()
        if self.config.embedding_backend and self._has_vec:
            self._create_vec_table()

    def _create_fts_table(self) -> None:
        """Create FTS5 virtual table pointing at the source table."""
        columns = ", ".join(self.config.text_columns)
        sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.config.fts_table} USING fts5(
                {columns},
                content='{self.config.table}',
                content_rowid='{self.config.id_column}',
                tokenize='porter unicode61'
            )
        """
        self._conn.execute(sql)
        self._conn.commit()

    def _create_fts_triggers(self) -> None:
        """Create triggers to keep FTS5 in sync with source table."""
        table = self.config.table
        fts = self.config.fts_table
        id_col = self.config.id_column
        cols = self.config.text_columns

        col_list = ", ".join(cols)
        new_vals = ", ".join(f"new.{c}" for c in cols)
        old_vals = ", ".join(f"old.{c}" for c in cols)

        # Insert trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts}_ai AFTER INSERT ON {table} BEGIN
                INSERT INTO {fts}(rowid, {col_list})
                VALUES (new.{id_col}, {new_vals});
            END
        """)

        # Delete trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts}_ad AFTER DELETE ON {table} BEGIN
                INSERT INTO {fts}({fts}, rowid, {col_list})
                VALUES ('delete', old.{id_col}, {old_vals});
            END
        """)

        # Update trigger
        self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {fts}_au AFTER UPDATE ON {table} BEGIN
                INSERT INTO {fts}({fts}, rowid, {col_list})
                VALUES ('delete', old.{id_col}, {old_vals});
                INSERT INTO {fts}(rowid, {col_list})
                VALUES (new.{id_col}, {new_vals});
            END
        """)

        self._conn.commit()

    def _create_vec_table(self) -> None:
        """Create vector table for embeddings."""
        if not self.config.embedding_backend:
            return

        dims = self.config.embedding_backend.dimensions
        sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.config.vec_table} USING vec0(
                row_id INTEGER PRIMARY KEY,
                embedding float[{dims}]
            )
        """
        self._conn.execute(sql)
        self._conn.commit()

    def reindex(
        self,
        only_missing: bool = False,
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Rebuild FTS5 and vector indexes.

        Args:
            only_missing: Only index rows missing from vector table
            batch_size: Rows per batch for embedding API calls
            progress_callback: Called with (processed, total) counts

        Returns:
            Dict with counts: {"fts_indexed": N, "vectors_indexed": N}
        """
        stats = {"fts_indexed": 0, "vectors_indexed": 0}

        # Rebuild FTS5 index
        if not only_missing:
            stats["fts_indexed"] = self._rebuild_fts()

        # Build vector index
        if self.config.embedding_backend and self._has_vec:
            stats["vectors_indexed"] = self._rebuild_vectors(
                only_missing=only_missing,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )

        return stats

    def _rebuild_fts(self) -> int:
        """Rebuild FTS5 index from source table."""
        fts = self.config.fts_table
        table = self.config.table

        # For content tables, use the 'rebuild' command to sync from source
        self._conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
        self._conn.commit()

        # Get count
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def _rebuild_vectors(
        self,
        only_missing: bool,
        batch_size: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> int:
        """Rebuild vector embeddings."""
        backend = self.config.embedding_backend
        if not backend:
            return 0

        table = self.config.table
        vec_table = self.config.vec_table
        id_col = self.config.id_column

        # Get rows to process
        if only_missing:
            sql = f"""
                SELECT t.* FROM {table} t
                LEFT JOIN {vec_table} v ON t.{id_col} = v.row_id
                WHERE v.row_id IS NULL
            """
        else:
            # Clear existing vectors
            self._conn.execute(f"DELETE FROM {vec_table}")
            sql = f"SELECT * FROM {table}"

        cursor = self._conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]

        # Process in batches
        indexed = 0
        batch = []
        batch_ids = []

        # Get total for progress
        if progress_callback:
            if only_missing:
                total_cursor = self._conn.execute(f"""
                    SELECT COUNT(*) FROM {table} t
                    LEFT JOIN {vec_table} v ON t.{id_col} = v.row_id
                    WHERE v.row_id IS NULL
                """)
            else:
                total_cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            total = total_cursor.fetchone()[0]
        else:
            total = 0

        for row in cursor:
            row_dict = dict(zip(columns, row, strict=True))
            row_id = row_dict[id_col]
            text = self._get_embedding_text(row_dict)

            batch.append(text)
            batch_ids.append(row_id)

            if len(batch) >= batch_size:
                self._embed_batch(batch, batch_ids)
                indexed += len(batch)
                if progress_callback:
                    progress_callback(indexed, total)
                batch = []
                batch_ids = []

        # Final batch
        if batch:
            self._embed_batch(batch, batch_ids)
            indexed += len(batch)
            if progress_callback:
                progress_callback(indexed, total)

        self._conn.commit()
        return indexed

    def _get_embedding_text(self, row: dict) -> str:
        """Extract text to embed from a row."""
        if self.config.embedding_text is None:
            # Concatenate all text columns
            parts = [str(row.get(c, "")) for c in self.config.text_columns]
            return " ".join(parts)
        elif callable(self.config.embedding_text):
            return self.config.embedding_text(row)
        else:
            # It's a column name
            return str(row.get(self.config.embedding_text, ""))

    def _embed_batch(self, texts: list[str], row_ids: list[int]) -> None:
        """Embed a batch of texts and store in vector table."""
        backend = self.config.embedding_backend
        if not backend:
            return

        embeddings = backend.embed_batch(texts)
        vec_table = self.config.vec_table

        for row_id, embedding in zip(row_ids, embeddings, strict=True):
            self._conn.execute(
                f"INSERT OR REPLACE INTO {vec_table} (row_id, embedding) VALUES (?, ?)",
                (row_id, json.dumps(embedding)),
            )

    def search(
        self,
        query: str,
        *,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
        limit: int = 10,
        offset: int = 0,
        backfill_vectors: bool = True,
    ) -> list[dict[str, Any]]:
        """Search the index.

        Args:
            query: Search query text
            mode: Search mode - keyword (FTS5), semantic (vector), or hybrid (RRF)
            limit: Maximum results
            offset: Pagination offset
            backfill_vectors: Auto-embed rows missing vectors (for hybrid/semantic)

        Returns:
            List of dicts with 'id' and 'score' keys
        """
        # Backfill missing vectors if needed
        if backfill_vectors and mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._has_vec:
                self._backfill_missing_vectors(limit=100)

        keyword_results = []
        vector_results = []

        if mode in ("keyword", "hybrid"):
            keyword_results = self._keyword_search(query, limit=limit + offset)

        if mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._has_vec:
                vector_results = self._vector_search(query, limit=limit + offset)

        # Combine results
        if mode == "hybrid" and keyword_results and vector_results:
            fused = reciprocal_rank_fusion(keyword_results, vector_results)
            results = [
                {"id": doc.doc_id, "score": doc.fused_score}
                for doc in fused[offset : offset + limit]
            ]
        elif mode == "semantic" and vector_results:
            results = [
                {"id": doc_id, "score": score}
                for _, doc_id, _, score in vector_results[offset : offset + limit]
            ]
        else:
            results = [
                {"id": doc_id, "score": score}
                for _, doc_id, _, score in keyword_results[offset : offset + limit]
            ]

        return results

    def _keyword_search(
        self, query: str, limit: int
    ) -> list[tuple[str, Any, dict, float]]:
        """Execute FTS5 keyword search."""
        fts = self.config.fts_table
        table = self.config.table
        id_col = self.config.id_column

        # Sanitize query for FTS5
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        sql = f"""
            SELECT t.{id_col}, bm25({fts}) as score
            FROM {fts}
            JOIN {table} t ON {fts}.rowid = t.{id_col}
            WHERE {fts} MATCH ?
            ORDER BY score
            LIMIT ?
        """

        cursor = self._conn.execute(sql, (safe_query, limit))
        results = []
        for row in cursor:
            row_id, score = row
            # Return tuple format expected by RRF: (index, id, source, score)
            results.append((table, row_id, {}, -score))  # Negate BM25 score

        return results

    def _vector_search(
        self, query: str, limit: int
    ) -> list[tuple[str, Any, dict, float]]:
        """Execute vector similarity search."""
        backend = self.config.embedding_backend
        if not backend:
            return []

        query_embedding = backend.embed(query)
        embedding_json = json.dumps(query_embedding)

        vec_table = self.config.vec_table
        table = self.config.table

        sql = f"""
            SELECT row_id, distance
            FROM {vec_table}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """

        try:
            cursor = self._conn.execute(sql, (embedding_json, limit))
        except sqlite3.OperationalError:
            return []

        results = []
        for row in cursor:
            row_id, distance = row
            score = 1.0 - float(distance)  # Convert distance to similarity
            results.append((table, row_id, {}, score))

        return results

    def _backfill_missing_vectors(self, limit: int = 100) -> int:
        """Embed rows that are missing from the vector table."""
        return self._rebuild_vectors(
            only_missing=True,
            batch_size=limit,
            progress_callback=None,
        )

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query string for FTS5."""
        # Remove FTS5 special characters
        for char in ['"', "'", "(", ")", "*", ":", "^", "-", "+", "OR", "AND", "NOT", "NEAR"]:
            query = query.replace(char, " ")
        return " ".join(query.split())

    def drop(self) -> None:
        """Remove the FTS5 table, triggers, and vector table."""
        fts = self.config.fts_table
        vec = self.config.vec_table

        # Drop triggers
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_ai")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_ad")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_au")

        # Drop tables
        self._conn.execute(f"DROP TABLE IF EXISTS {fts}")
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {vec}")
        except sqlite3.OperationalError:
            pass

        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        table = self.config.table
        fts = self.config.fts_table
        vec = self.config.vec_table

        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]

        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {fts}")
        fts_rows = cursor.fetchone()[0]

        vec_rows = 0
        if self._has_vec:
            try:
                cursor = self._conn.execute(f"SELECT COUNT(*) FROM {vec}")
                vec_rows = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                pass

        return {
            "table": table,
            "total_rows": total_rows,
            "fts_indexed": fts_rows,
            "vectors_indexed": vec_rows,
            "vectors_missing": total_rows - vec_rows if self._has_vec else None,
        }
