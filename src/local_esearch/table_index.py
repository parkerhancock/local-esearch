"""Table index for bolt-on search over existing tables."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from local_esearch.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text
from local_esearch.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from local_esearch.backends import DatabaseBackend
    from local_esearch.embeddings.base import EmbeddingBackend


@dataclass
class TableConfig:
    """Configuration for a registered table index."""

    table: str
    id_column: str
    text_columns: list[str]
    embedding_text: Callable[[dict], str] | str | None = None
    embedding_backend: "EmbeddingBackend | None" = None
    chunk_size: int = DEFAULT_CHUNK_SIZE  # 250 words (ES default)
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP  # 100 words (ES default)
    fts_table: str = field(default="")
    vec_table: str = field(default="")
    chunks_table: str = field(default="")

    def __post_init__(self) -> None:
        self.fts_table = f"{self.table}_fts"
        self.vec_table = f"{self.table}_vec"
        self.chunks_table = f"{self.table}_chunks"


class TableIndex:
    """Search index over an existing table.

    Creates FTS and vector indexes that reference the original table,
    enabling full-text and semantic search without duplicating data.

    Example:
        index = TableIndex(
            db_backend=backend,
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
        db_backend: "DatabaseBackend",
        table: str,
        id_column: str = "id",
        text_columns: list[str] | None = None,
        embedding_text: Callable[[dict], str] | str | None = None,
        embedding_backend: "EmbeddingBackend | None" = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """Initialize table index.

        Args:
            db_backend: Database backend
            table: Name of existing table to index
            id_column: Primary key column name
            text_columns: Columns to include in FTS index
            embedding_text: How to generate text for embeddings:
                - str: column name to embed
                - Callable: function(row_dict) -> str
                - None: concatenate all text_columns
            embedding_backend: Embedding backend for vector search
            chunk_size: Words per chunk for embeddings (default: 250, matching ES)
            chunk_overlap: Overlap words between chunks (default: 100, matching ES)
        """
        self._backend = db_backend
        self.config = TableConfig(
            table=table,
            id_column=id_column,
            text_columns=text_columns or [],
            embedding_text=embedding_text,
            embedding_backend=embedding_backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def setup(self) -> None:
        """Create FTS table, triggers, and vector table.

        Safe to call multiple times - uses IF NOT EXISTS.
        """
        self._backend.create_table_fts(
            self.config.fts_table,
            self.config.table,
            self.config.text_columns,
            self.config.id_column,
        )
        self._backend.create_table_fts_triggers(
            self.config.fts_table,
            self.config.table,
            self.config.text_columns,
            self.config.id_column,
        )
        if self.config.embedding_backend and self._backend.vector_available():
            dims = self.config.embedding_backend.dimensions
            self._backend.create_table_vec(self.config.vec_table, dims)
            self._backend.create_chunks_table(self.config.chunks_table)
            self._backend.create_chunks_delete_triggers(
                self.config.table,
                self.config.chunks_table,
                self.config.vec_table,
                self.config.id_column,
            )

    def reindex(
        self,
        only_missing: bool = False,
        batch_size: int = 100,
        max_rows: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Rebuild FTS and vector indexes.

        Args:
            only_missing: Only index rows missing from vector table
            batch_size: Rows per batch for embedding API calls
            max_rows: Maximum number of rows to index (None = all)
            progress_callback: Called with (processed, total) counts

        Returns:
            Dict with counts: {"fts_indexed": N, "vectors_indexed": N}
        """
        stats = {"fts_indexed": 0, "vectors_indexed": 0}

        # Rebuild FTS index
        if not only_missing:
            stats["fts_indexed"] = self._rebuild_fts()

        # Build vector index
        if self.config.embedding_backend and self._backend.vector_available():
            stats["vectors_indexed"] = self._rebuild_vectors(
                only_missing=only_missing,
                batch_size=batch_size,
                max_rows=max_rows,
                progress_callback=progress_callback,
            )

        return stats

    def _rebuild_fts(self) -> int:
        """Rebuild FTS index from source table."""
        self._backend.rebuild_fts(self.config.fts_table, self.config.table)

        # Get count
        rows = self._backend.execute(f"SELECT COUNT(*) as cnt FROM {self.config.table}")
        return rows[0]["cnt"] if rows else 0

    def _rebuild_vectors(
        self,
        only_missing: bool,
        batch_size: int,
        max_rows: int | None,
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
            # Find rows that have no chunks in the chunks table
            sql = f"""
                SELECT t.* FROM {table} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {chunks_table} c WHERE c.row_id = t.{id_col}
                )
            """
        else:
            # Clear existing vectors and chunks
            self._backend.clear_table(vec_table)
            self._backend.clear_table(chunks_table)
            sql = f"SELECT * FROM {table}"

        # Add LIMIT if max_rows specified
        if max_rows is not None:
            sql += f" LIMIT {max_rows}"

        all_rows = self._backend.execute(sql)
        total = len(all_rows)

        # Collect chunks for batch embedding
        chunk_texts: list[str] = []
        chunk_metadata: list[tuple[str, int, int, str, int, int]] = []
        rows_processed = 0

        for row in all_rows:
            row_id = row[id_col]
            text = self._get_embedding_text(row)

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

        self._backend.commit()
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
            self._backend.vector_upsert_chunk(vec_table, chunk_id, embedding)

            # Store chunk metadata
            self._backend.upsert_chunk(
                chunks_table=chunks_table,
                chunk_id=chunk_id,
                row_id=row_id,
                chunk_idx=chunk_idx,
                chunk_text=chunk_text_str,
                start_char=start_char,
                end_char=end_char,
            )

    def _get_embedding_text(self, row: dict[str, Any]) -> str:
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

    def search_raw(
        self,
        query: str,
        *,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
        limit: int = 10,
        backfill_vectors: bool = True,
        inner_hits_size: int = 3,
        index_name: str | None = None,
    ) -> tuple[
        list[tuple[str, Any, dict, float]],
        list[tuple[str, Any, dict, float]],
        dict[Any, list[dict]],
    ]:
        """Search and return raw results for cross-table fusion.

        Args:
            query: Search query text
            mode: Search mode - keyword (FTS), semantic (vector), or hybrid (both)
            limit: Maximum results
            backfill_vectors: Auto-embed rows missing vectors (for hybrid/semantic)
            inner_hits_size: Max chunks to return per document in inner_hits
            index_name: Index name to use in results (defaults to table name)

        Returns:
            Tuple of (keyword_results, vector_results, inner_hits_map).
            Results are tuples of (index, doc_id, source, score).
        """
        idx = index_name or self.config.table

        # Backfill missing vectors if needed
        if backfill_vectors and mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._backend.vector_available():
                self._backfill_missing_vectors(limit=100)

        keyword_results: list[tuple[str, Any, dict, float]] = []
        vector_results: list[tuple[str, Any, dict, float]] = []
        inner_hits_map: dict[Any, list[dict]] = {}

        if mode in ("keyword", "hybrid"):
            raw_kw = self._keyword_search(query, limit=limit)
            # Replace table name with index name
            keyword_results = [(idx, doc_id, src, score) for _, doc_id, src, score in raw_kw]

        if mode in ("semantic", "hybrid"):
            if self.config.embedding_backend and self._backend.vector_available():
                raw_vec, inner_hits_map = self._vector_search(
                    query, limit=limit, inner_hits_size=inner_hits_size
                )
                # Replace table name with index name
                vector_results = [(idx, doc_id, src, score) for _, doc_id, src, score in raw_vec]

        return keyword_results, vector_results, inner_hits_map

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
            mode: Search mode - keyword (FTS), semantic (vector), or hybrid (RRF)
            limit: Maximum results
            offset: Pagination offset
            backfill_vectors: Auto-embed rows missing vectors (for hybrid/semantic)
            inner_hits_size: Max chunks to return per document in inner_hits

        Returns:
            List of dicts with 'id', 'score', and optional 'inner_hits' keys.
            inner_hits contains matching chunks for semantic/hybrid modes.
        """
        keyword_results, vector_results, inner_hits_map = self.search_raw(
            query,
            mode=mode,
            limit=limit + offset,
            backfill_vectors=backfill_vectors,
            inner_hits_size=inner_hits_size,
        )

        # Combine results
        if mode == "hybrid" and keyword_results and vector_results:
            fused = reciprocal_rank_fusion(keyword_results, vector_results)
            results = []
            for doc in fused[offset : offset + limit]:
                result: dict[str, Any] = {"id": doc.doc_id, "score": doc.fused_score}
                if doc.doc_id in inner_hits_map:
                    result["inner_hits"] = {"chunks": inner_hits_map[doc.doc_id]}
                results.append(result)
        elif mode == "semantic" and vector_results:
            results = []
            for _, doc_id, _, score in vector_results[offset : offset + limit]:
                result: dict[str, Any] = {"id": doc_id, "score": score}
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
        """Execute FTS keyword search."""
        fts = self.config.fts_table
        table = self.config.table
        id_col = self.config.id_column

        # Sanitize query for FTS
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        sql = self._backend.table_fts_search_sql(fts, table, id_col)

        rows = self._backend.execute(sql, (safe_query, limit))
        results = []
        for row in rows:
            row_id = row[id_col]
            score = row["score"]
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
        k_value = limit * inner_hits_size * 2
        sql = self._backend.vector_search_chunks_sql(vec_table, chunks_table)

        try:
            rows = self._backend.execute(sql, (embedding_json, k_value))
        except Exception:
            return [], {}

        # Group chunks by row_id
        doc_chunks: dict[Any, list[dict]] = defaultdict(list)
        doc_best_score: dict[Any, float] = {}

        for row in rows:
            chunk_id = row["chunk_id"]
            distance = row["distance"]
            row_id = row["row_id"]
            chunk_idx = row["chunk_idx"]
            chunk_text_str = row["chunk_text"]
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
            max_rows=limit,
            progress_callback=None,
        )

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query string for FTS."""
        # Remove FTS special characters
        for char in ['"', "'", "(", ")", "*", ":", "^", "-", "+", "OR", "AND", "NOT", "NEAR"]:
            query = query.replace(char, " ")
        return " ".join(query.split())

    def drop(self) -> None:
        """Remove the FTS table, triggers, vector table, and chunks table."""
        self._backend.drop_table_indexes(
            self.config.fts_table,
            self.config.vec_table,
            self.config.chunks_table,
        )

    def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        table = self.config.table
        fts = self.config.fts_table
        chunks = self.config.chunks_table

        rows = self._backend.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        total_rows = rows[0]["cnt"] if rows else 0

        rows = self._backend.execute(f"SELECT COUNT(*) as cnt FROM {fts}")
        fts_rows = rows[0]["cnt"] if rows else 0

        chunks_count = 0
        rows_with_chunks = 0
        if self._backend.vector_available():
            try:
                rows = self._backend.execute(f"SELECT COUNT(*) as cnt FROM {chunks}")
                chunks_count = rows[0]["cnt"] if rows else 0
                rows = self._backend.execute(
                    f"SELECT COUNT(DISTINCT row_id) as cnt FROM {chunks}"
                )
                rows_with_chunks = rows[0]["cnt"] if rows else 0
            except Exception:
                pass

        return {
            "table": table,
            "total_rows": total_rows,
            "fts_indexed": fts_rows,
            "chunks_indexed": chunks_count,
            "rows_with_vectors": rows_with_chunks,
            "rows_missing_vectors": total_rows - rows_with_chunks if self._backend.vector_available() else None,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
