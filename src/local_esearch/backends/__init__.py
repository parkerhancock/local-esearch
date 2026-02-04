"""Database backend factory and detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from local_esearch.backends.base import DatabaseBackend, FTSResult, VectorResult

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend

__all__ = ["DatabaseBackend", "FTSResult", "VectorResult", "create_backend"]


def is_postgres_uri(path: str) -> bool:
    """Check if path is a PostgreSQL connection string."""
    return path.startswith(("postgresql://", "postgres://"))


def create_backend(
    path: str | Path,
    embedding_backend: "EmbeddingBackend | None" = None,
) -> DatabaseBackend:
    """Create appropriate database backend based on path.

    Args:
        path: Database path. SQLite file path, ":memory:", or postgresql:// URI
        embedding_backend: Optional embedding backend for vector dimensions

    Returns:
        Configured DatabaseBackend instance

    Examples:
        # SQLite (default)
        backend = create_backend("./search.db")
        backend = create_backend(":memory:")

        # PostgreSQL
        backend = create_backend("postgresql://user:pass@localhost/dbname")
    """
    path_str = str(path)

    if is_postgres_uri(path_str):
        from local_esearch.backends.postgres import PostgresBackend

        return PostgresBackend(path_str, embedding_backend)
    else:
        from local_esearch.backends.sqlite import SQLiteBackend

        return SQLiteBackend(path_str, embedding_backend)
