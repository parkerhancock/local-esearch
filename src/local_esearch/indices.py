"""Elasticsearch IndicesClient implementation."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from local_esearch.exceptions import NotFoundError, RequestError

if TYPE_CHECKING:
    from local_esearch.client import Elasticsearch


class IndicesClient:
    """Elasticsearch-compatible indices management API."""

    def __init__(self, client: "Elasticsearch"):
        self._client = client

    @property
    def _backend(self):
        return self._client._backend

    def create(
        self,
        index: str,
        *,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new index.

        Args:
            index: Index name
            mappings: Field mappings
            settings: Index settings
            body: Alternative way to pass mappings/settings

        Returns:
            Acknowledgment response
        """
        # Handle body parameter for compatibility
        if body:
            mappings = mappings or body.get("mappings")
            settings = settings or body.get("settings")

        # Check if index already exists
        rows = self._backend.execute("SELECT 1 FROM _indices WHERE name = ?", (index,))
        if rows:
            raise RequestError(f"Index '{index}' already exists")

        self._backend.execute(
            """
            INSERT INTO _indices (name, mappings_json, settings_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                index,
                json.dumps(mappings) if mappings else None,
                json.dumps(settings) if settings else None,
                self._backend.timestamp_now(),
            ),
        )
        self._backend.commit()

        return {
            "acknowledged": True,
            "shards_acknowledged": True,
            "index": index,
        }

    def delete(self, index: str) -> dict[str, Any]:
        """Delete an index and all its documents.

        Args:
            index: Index name (supports wildcards with *)

        Returns:
            Acknowledgment response
        """
        # Handle wildcard pattern
        if "*" in index:
            pattern = index.replace("*", "%")
            rows = self._backend.execute(
                "SELECT name FROM _indices WHERE name LIKE ?", (pattern,)
            )
            indices_to_delete = [row["name"] for row in rows]
        else:
            rows = self._backend.execute(
                "SELECT 1 FROM _indices WHERE name = ?", (index,)
            )
            if not rows:
                raise NotFoundError(f"Index '{index}' not found")
            indices_to_delete = [index]

        for idx in indices_to_delete:
            # Delete documents first
            self._backend.execute("DELETE FROM _documents WHERE _index = ?", (idx,))
            # Delete vector embeddings if table exists
            if self._backend.vector_available():
                self._backend.vector_delete_pattern("_documents_vec", f"{idx}:%")
            # Delete index metadata
            self._backend.execute("DELETE FROM _indices WHERE name = ?", (idx,))

        self._backend.commit()

        return {"acknowledged": True}

    def exists(self, index: str) -> bool:
        """Check if an index exists.

        Args:
            index: Index name

        Returns:
            True if index exists
        """
        rows = self._backend.execute(
            "SELECT 1 FROM _indices WHERE name = ?", (index,)
        )
        return len(rows) > 0

    def get(self, index: str) -> dict[str, Any]:
        """Get index metadata.

        Args:
            index: Index name

        Returns:
            Index settings and mappings
        """
        rows = self._backend.execute(
            "SELECT mappings_json, settings_json, created_at FROM _indices WHERE name = ?",
            (index,),
        )
        if not rows:
            raise NotFoundError(f"Index '{index}' not found")

        row = rows[0]
        mappings_json = row["mappings_json"]
        settings_json = row["settings_json"]
        # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
        mappings = mappings_json if isinstance(mappings_json, dict) else (json.loads(mappings_json) if mappings_json else {})
        settings = settings_json if isinstance(settings_json, dict) else (json.loads(settings_json) if settings_json else {})
        return {
            index: {
                "mappings": mappings,
                "settings": settings,
            }
        }

    def refresh(self, index: str | None = None) -> dict[str, Any]:
        """Refresh index to make recent changes searchable.

        For SQLite/FTS5, this is essentially a no-op since changes are
        immediately visible. We keep it for API compatibility.

        Args:
            index: Index name (optional, refreshes all if not provided)

        Returns:
            Refresh response
        """
        # FTS5 triggers handle sync automatically
        # Just commit any pending transactions
        self._backend.commit()

        return {
            "_shards": {
                "total": 1,
                "successful": 1,
                "failed": 0,
            }
        }

    def stats(self, index: str | None = None) -> dict[str, Any]:
        """Get index statistics.

        Args:
            index: Index name (optional, returns all if not provided)

        Returns:
            Index statistics
        """
        if index:
            indices = [index]
        else:
            rows = self._backend.execute("SELECT name FROM _indices")
            indices = [row["name"] for row in rows]

        all_stats = {}
        total_docs = 0

        for idx in indices:
            rows = self._backend.execute(
                "SELECT COUNT(*) as cnt FROM _documents WHERE _index = ?",
                (idx,),
            )
            doc_count = rows[0]["cnt"] if rows else 0
            total_docs += doc_count

            all_stats[idx] = {
                "primaries": {
                    "docs": {"count": doc_count, "deleted": 0},
                    "store": {"size_in_bytes": 0},
                },
                "total": {
                    "docs": {"count": doc_count, "deleted": 0},
                    "store": {"size_in_bytes": 0},
                },
            }

        return {
            "_shards": {"total": 1, "successful": 1, "failed": 0},
            "_all": {
                "primaries": {"docs": {"count": total_docs}},
                "total": {"docs": {"count": total_docs}},
            },
            "indices": all_stats,
        }

    def put_mapping(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        *,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update index mappings.

        Args:
            index: Index name
            body: Mapping body (for compatibility)
            properties: Field properties

        Returns:
            Acknowledgment response
        """
        # Get existing mappings
        rows = self._backend.execute(
            "SELECT mappings_json FROM _indices WHERE name = ?", (index,)
        )
        if not rows:
            raise NotFoundError(f"Index '{index}' not found")

        mappings_data = rows[0]["mappings_json"]
        # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
        existing = mappings_data if isinstance(mappings_data, dict) else (json.loads(mappings_data) if mappings_data else {})

        # Merge new mappings
        if body:
            if "properties" in body:
                existing.setdefault("properties", {}).update(body["properties"])
            else:
                existing.update(body)
        if properties:
            existing.setdefault("properties", {}).update(properties)

        self._backend.execute(
            "UPDATE _indices SET mappings_json = ? WHERE name = ?",
            (json.dumps(existing), index),
        )
        self._backend.commit()

        return {"acknowledged": True}

    def get_mapping(self, index: str) -> dict[str, Any]:
        """Get index mappings.

        Args:
            index: Index name

        Returns:
            Index mappings
        """
        rows = self._backend.execute(
            "SELECT mappings_json FROM _indices WHERE name = ?", (index,)
        )
        if not rows:
            raise NotFoundError(f"Index '{index}' not found")

        mappings_data = rows[0]["mappings_json"]
        # PostgreSQL JSONB returns dict directly, SQLite returns JSON string
        mappings = mappings_data if isinstance(mappings_data, dict) else (json.loads(mappings_data) if mappings_data else {})
        return {
            index: {
                "mappings": mappings,
            }
        }

    def reindex(
        self,
        index: str,
        *,
        only_missing: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Rebuild search indexes for a registered table.

        Only works for indexes created via `es.register_table()`.

        Args:
            index: Index name (must be a registered table)
            only_missing: Only index rows missing from vector table
            batch_size: Rows per batch for embedding API calls

        Returns:
            Stats dict with indexed counts
        """
        table_index = self._client._table_indexes.get(index)
        if not table_index:
            raise RequestError(
                f"Index '{index}' is not a registered table. Use es.register_table() first."
            )

        stats = table_index.reindex(
            only_missing=only_missing,
            batch_size=batch_size,
        )

        return {
            "acknowledged": True,
            "index": index,
            "stats": stats,
        }
