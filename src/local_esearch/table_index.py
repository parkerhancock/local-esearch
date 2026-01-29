"""Table index for bolt-on search over existing SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from local_esearch.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text
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
    chunk_size: int = DEFAULT_CHUNK_SIZE  # 250 words (ES default)
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP  # 100 words (ES default)
    fts_table: str = field(default="")
    vec_table: str = field(default="")
    chunks_table: str = field(default="")

    def __post_init__(self):
        self.fts_table = f"{self.table}_fts"
        self.vec_table = f"{self.table}_vec"
        self.chunks_table = f"{self.table}_chunks"


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
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
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
            chunk_size: Words per chunk for embeddings (default: 250, matching ES)
            chunk_overlap: Overlap words between chunks (default: 100, matching ES)
        """
        self._conn = conn
        self.config = TableConfig(
            table=table,
            id_column=id_column,
            text_columns=text_columns or [],
            embedding_text=embedding_text,
            embedding_backend=embedding_backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
            self._create_chunks_table()

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
        """Create vector table for chunk embeddings."""
        if not self.config.embedding_backend:
            return

        dims = self.config.embedding_backend.dimensions
        # Use chunk_id as primary key: "rowid:chunk_idx"
        sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.config.vec_table} USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dims}]
            )
        """
        self._conn.execute(sql)
        self._conn.commit()

    def _create_chunks_table(self) -> None:
        """Create metadata table for chunks (stores text for inner_hits)."""
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.chunks_table} (
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
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_row "
            f"ON {self.config.chunks_table}(row_id)"
        )
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
        """Rebuild vector embeddings with chunking.

        Chunks each document's text and embeds each chunk separately.
        Stores chunk metadata for inner_hits retrieval.
        """
        backend = self.config.embedding_backend
        if not backend:
            return 0

        table = self.config.table
        vec_table = self.config.vec_table
        chunks_table = self.config.chunks_table
        id_col = self.config.id_column

        # Get rows to process
        if only_missing:
            # Find rows that have no chunks in the vec table
            sql = f"""
                SELECT t.* FROM {table} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {chunks_table} c WHERE c.row_id = t.{id_col}
                )
            """
        else:
            # Clear existing vectors and chunks
            self._conn.execute(f"DELETE FROM {vec_table}")
            self._conn.execute(f"DELETE FROM {chunks_table}")
            sql = f"SELECT * FROM {table}"

        cursor = self._conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]

        # Get total for progress
        if progress_callback:
            if only_missing:
                total_cursor = self._conn.execute(f"""
                    SELECT COUNT(*) FROM {table} t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {chunks_table} c WHERE c.row_id = t.{id_col}
                    )
                """)
            else:
                total_cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            total = total_cursor.fetchone()[0]
        else:
            total = 0

        # Collect chunks for batch embedding
        chunk_texts = []
        chunk_metadata = []  # (chunk_id, row_id, chunk_idx, text, start, end)
        rows_processed = 0

        for row in cursor:
            row_dict = dict(zip(columns, row, strict=True))
            row_id = row_dict[id_col]
            text = self._get_embedding_text(row_dict)

            # Chunk the text
            chunks = chunk_text(
                text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            for chunk in chunks:
                chunk_id = f"{row_id}:{chunk.index}"
                chunk_texts.append(chunk.text)
                chunk_metadata.append(
                    (chunk_id, row_id, chunk.index, chunk.text, chunk.start_char, chunk.end_char)
                )

            # Embed in batches
            if len(chunk_texts) >= batch_size:
                self._embed_chunks(chunk_texts, chunk_metadata)
                chunk_texts = []
                chunk_metadata = []

            rows_processed += 1
            if progress_callback:
                progress_callback(rows_processed, total)

        # Final batch
        if chunk_texts:
            self._embed_chunks(chunk_texts, chunk_metadata)

        self._conn.commit()
        return rows_processed

    def _embed_chunks(
        self, texts: list[str], metadata: list[tuple[str, int, int, str, int, int]]
    ) -> None:
        """Embed a batch of chunks and store in vec + chunks tables."""
        backend = self.config.embedding_backend
        if not backend or not texts:
            return

        embeddings = backend.embed_batch(texts)
        vec_table = self.config.vec_table
        chunks_table = self.config.chunks_table

        for embedding, meta in zip(embeddings, metadata, strict=True):
            chunk_id, row_id, chunk_idx, chunk_text_str, start_char, end_char = meta

            # Store embedding
            self._conn.execute(
                f"INSERT OR REPLACE INTO {vec_table} (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(embedding)),
            )

            # Store chunk metadata
            self._conn.execute(
                f"""INSERT OR REPLACE INTO {chunks_table}
                    (chunk_id, row_id, chunk_idx, chunk_text, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (chunk_id, row_id, chunk_idx, chunk_text_str, start_char, end_char),
            )

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

    def search(
        self,
        query: str,
        *,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
        limit: int = 10,
        offset: int = 0,
        backfill_vectors: bool = True,
        inner_hits_size: int = 3,
    ) -> list[dict[str, Any]]:
        """Search the index.

        Args:
            query: Search query text
            mode: Search mode - keyword (FTS5), semantic (vector), or hybrid (RRF)
            limit: Maximum results
            offset: Pagination offset
            backfill_vectors: Auto-embed rows missing vectors (for hybrid/semantic)
            inner_hits_size: Max chunks to return per document in inner_hits

        Returns:
            List of dicts with 'id', 'score', and optional 'inner_hits' keys.
            inner_hits contains matching chunks for semantic/hybrid modes.
        """
        # Backfill missing vectors if needed
        if backfill_vectors and mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._has_vec:
                self._backfill_missing_vectors(limit=100)

        keyword_results = []
        vector_results = []
        inner_hits_map: dict[Any, list[dict]] = {}

        if mode in ("keyword", "hybrid"):
            keyword_results = self._keyword_search(query, limit=limit + offset)

        if mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._has_vec:
                vector_results, inner_hits_map = self._vector_search(
                    query, limit=limit + offset, inner_hits_size=inner_hits_size
                )

        # Combine results
        if mode == "hybrid" and keyword_results and vector_results:
            fused = reciprocal_rank_fusion(keyword_results, vector_results)
            results = []
            for doc in fused[offset : offset + limit]:
                result = {"id": doc.doc_id, "score": doc.fused_score}
                if doc.doc_id in inner_hits_map:
                    result["inner_hits"] = {"chunks": inner_hits_map[doc.doc_id]}
                results.append(result)
        elif mode == "semantic" and vector_results:
            results = []
            for _, doc_id, _, score in vector_results[offset : offset + limit]:
                result = {"id": doc_id, "score": score}
                if doc_id in inner_hits_map:
                    result["inner_hits"] = {"chunks": inner_hits_map[doc_id]}
                results.append(result)
        else:
            # Keyword-only - no inner_hits
            results = [
                {"id": doc_id, "score": score}
                for _, doc_id, _, score in keyword_results[offset : offset + limit]
            ]

        return results

    def _keyword_search(self, query: str, limit: int) -> list[tuple[str, Any, dict, float]]:
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
        self, query: str, limit: int, inner_hits_size: int = 3
    ) -> tuple[list[tuple[str, Any, dict, float]], dict[Any, list[dict]]]:
        """Execute vector similarity search with chunking.

        Returns documents with their best chunk scores, plus inner_hits
        containing the matching chunks for each document.

        Args:
            query: Search query
            limit: Max documents to return
            inner_hits_size: Max chunks per document in inner_hits

        Returns:
            Tuple of:
            - List of (table, row_id, source, score) for RRF fusion
            - Dict mapping row_id -> list of chunk hits for inner_hits
        """
        backend = self.config.embedding_backend
        if not backend:
            return [], {}

        query_embedding = backend.embed(query)
        embedding_json = json.dumps(query_embedding)

        vec_table = self.config.vec_table
        chunks_table = self.config.chunks_table
        table = self.config.table

        # Query chunks and join with metadata
        # Get more chunks than needed to ensure we have enough unique documents
        sql = f"""
            SELECT v.chunk_id, v.distance, c.row_id, c.chunk_idx, c.chunk_text
            FROM {vec_table} v
            JOIN {chunks_table} c ON v.chunk_id = c.chunk_id
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
        """

        try:
            cursor = self._conn.execute(sql, (embedding_json, limit * inner_hits_size * 2))
        except sqlite3.OperationalError:
            return [], {}

        # Group chunks by row_id
        doc_chunks: dict[Any, list[dict]] = defaultdict(list)
        doc_best_score: dict[Any, float] = {}

        for row in cursor:
            chunk_id, distance, row_id, chunk_idx, chunk_text_str = row
            score = 1.0 - float(distance)

            # Track best score per document
            if row_id not in doc_best_score or score > doc_best_score[row_id]:
                doc_best_score[row_id] = score

            # Add to inner_hits (up to inner_hits_size per doc)
            if len(doc_chunks[row_id]) < inner_hits_size:
                doc_chunks[row_id].append(
                    {
                        "chunk_idx": chunk_idx,
                        "text": chunk_text_str,
                        "_score": score,
                    }
                )

        # Build document results sorted by best chunk score
        results = []
        for row_id, score in sorted(doc_best_score.items(), key=lambda x: -x[1])[:limit]:
            results.append((table, row_id, {}, score))

        return results, dict(doc_chunks)

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
        """Remove the FTS5 table, triggers, vector table, and chunks table."""
        fts = self.config.fts_table
        vec = self.config.vec_table
        chunks = self.config.chunks_table

        # Drop triggers
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_ai")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_ad")
        self._conn.execute(f"DROP TRIGGER IF EXISTS {fts}_au")

        # Drop tables
        self._conn.execute(f"DROP TABLE IF EXISTS {fts}")
        self._conn.execute(f"DROP TABLE IF EXISTS {chunks}")
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {vec}")
        except sqlite3.OperationalError:
            pass

        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        table = self.config.table
        fts = self.config.fts_table
        chunks = self.config.chunks_table

        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]

        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {fts}")
        fts_rows = cursor.fetchone()[0]

        chunks_count = 0
        rows_with_chunks = 0
        if self._has_vec:
            try:
                cursor = self._conn.execute(f"SELECT COUNT(*) FROM {chunks}")
                chunks_count = cursor.fetchone()[0]
                cursor = self._conn.execute(f"SELECT COUNT(DISTINCT row_id) FROM {chunks}")
                rows_with_chunks = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                pass

        return {
            "table": table,
            "total_rows": total_rows,
            "fts_indexed": fts_rows,
            "chunks_indexed": chunks_count,
            "rows_with_vectors": rows_with_chunks,
            "rows_missing_vectors": total_rows - rows_with_chunks if self._has_vec else None,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
