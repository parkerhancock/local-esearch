"""Elasticsearch IndicesClient implementation."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from local_esearch.exceptions import NotFoundError, RequestError

if TYPE_CHECKING:
    from local_esearch.client import Elasticsearch


class IndicesClient:
    """Elasticsearch-compatible indices management API."""

    def __init__(self, client: Elasticsearch):
        self._client = client

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._client._conn

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

        cursor = self._conn.cursor()

        # Check if index already exists
        cursor.execute("SELECT 1 FROM _indices WHERE name = ?", (index,))
        if cursor.fetchone():
            raise RequestError(f"Index '{index}' already exists")

        cursor.execute(
            """
            INSERT INTO _indices (name, mappings_json, settings_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                index,
                json.dumps(mappings) if mappings else None,
                json.dumps(settings) if settings else None,
                time.time(),
            ),
        )
        self._conn.commit()

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
        cursor = self._conn.cursor()

        # Handle wildcard pattern
        if "*" in index:
            pattern = index.replace("*", "%")
            cursor.execute("SELECT name FROM _indices WHERE name LIKE ?", (pattern,))
            indices_to_delete = [row[0] for row in cursor.fetchall()]
        else:
            cursor.execute("SELECT 1 FROM _indices WHERE name = ?", (index,))
            if not cursor.fetchone():
                raise NotFoundError(f"Index '{index}' not found")
            indices_to_delete = [index]

        for idx in indices_to_delete:
            # Delete documents first
            cursor.execute("DELETE FROM _documents WHERE _index = ?", (idx,))
            # Delete vector embeddings if table exists
            try:
                cursor.execute(
                    "DELETE FROM _documents_vec WHERE doc_key LIKE ?",
                    (f"{idx}:%",),
                )
            except sqlite3.OperationalError:
                pass  # Vector table doesn't exist
            # Delete index metadata
            cursor.execute("DELETE FROM _indices WHERE name = ?", (idx,))

        self._conn.commit()

        return {"acknowledged": True}

    def exists(self, index: str) -> bool:
        """Check if an index exists.

        Args:
            index: Index name

        Returns:
            True if index exists
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM _indices WHERE name = ?", (index,))
        return cursor.fetchone() is not None

    def get(self, index: str) -> dict[str, Any]:
        """Get index metadata.

        Args:
            index: Index name

        Returns:
            Index settings and mappings
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT mappings_json, settings_json, created_at FROM _indices WHERE name = ?",
            (index,),
        )
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"Index '{index}' not found")

        mappings_json, settings_json, created_at = row
        return {
            index: {
                "mappings": json.loads(mappings_json) if mappings_json else {},
                "settings": json.loads(settings_json) if settings_json else {},
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
        self._conn.commit()

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
        cursor = self._conn.cursor()

        if index:
            indices = [index]
        else:
            cursor.execute("SELECT name FROM _indices")
            indices = [row[0] for row in cursor.fetchall()]

        all_stats = {}
        total_docs = 0

        for idx in indices:
            cursor.execute(
                "SELECT COUNT(*) FROM _documents WHERE _index = ?",
                (idx,),
            )
            doc_count = cursor.fetchone()[0]
            total_docs += doc_count

            all_stats[idx] = {
                "primaries": {
                    "docs": {"count": doc_count, "deleted": 0},
                    "store": {"size_in_bytes": 0},  # Could calculate actual size
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
        cursor = self._conn.cursor()

        # Get existing mappings
        cursor.execute("SELECT mappings_json FROM _indices WHERE name = ?", (index,))
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"Index '{index}' not found")

        existing = json.loads(row[0]) if row[0] else {}

        # Merge new mappings
        if body:
            if "properties" in body:
                existing.setdefault("properties", {}).update(body["properties"])
            else:
                existing.update(body)
        if properties:
            existing.setdefault("properties", {}).update(properties)

        cursor.execute(
            "UPDATE _indices SET mappings_json = ? WHERE name = ?",
            (json.dumps(existing), index),
        )
        self._conn.commit()

        return {"acknowledged": True}

    def get_mapping(self, index: str) -> dict[str, Any]:
        """Get index mappings.

        Args:
            index: Index name

        Returns:
            Index mappings
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT mappings_json FROM _indices WHERE name = ?", (index,))
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"Index '{index}' not found")

        return {
            index: {
                "mappings": json.loads(row[0]) if row[0] else {},
            }
        }
