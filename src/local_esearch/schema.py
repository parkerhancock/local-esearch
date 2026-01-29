"""SQLite schema management for local_esearch."""

from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_esearch.embeddings.base import EmbeddingBackend


SCHEMA_VERSION = 2

# Core schema - always created
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

# FTS5 schema - separate for clarity
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


def vector_schema(dimensions: int) -> str:
    """Generate vector table schema for given dimensions."""
    return f"""
-- Vector embeddings table (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS _documents_vec USING vec0(
    doc_key TEXT PRIMARY KEY,
    embedding float[{dimensions}]
);
"""


def init_database(
    conn: sqlite3.Connection,
    embedding_backend: EmbeddingBackend | None = None,
) -> None:
    """Initialize database schema.

    Args:
        conn: SQLite connection
        embedding_backend: Optional embedding backend for vector search
    """
    cursor = conn.cursor()

    # Check current schema version
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_version'")
    if cursor.fetchone():
        cursor.execute("SELECT MAX(version) FROM _schema_version")
        row = cursor.fetchone()
        current_version = row[0] if row and row[0] else 0
    else:
        current_version = 0

    if current_version >= SCHEMA_VERSION:
        return

    # Apply core schema
    conn.executescript(CORE_SCHEMA)

    # Apply FTS5 schema
    conn.executescript(FTS5_SCHEMA)

    # Apply vector schema if embedding backend provided
    if embedding_backend is not None:
        try:
            # Check if sqlite-vec is available
            conn.execute("SELECT vec_version()")
            conn.executescript(vector_schema(embedding_backend.dimensions))
        except sqlite3.OperationalError:
            # sqlite-vec not available, skip vector table
            pass

    # Record schema version
    cursor.execute(
        "INSERT OR REPLACE INTO _schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, time.time()),
    )

    conn.commit()


def ensure_vector_table(
    conn: sqlite3.Connection,
    dimensions: int,
) -> bool:
    """Ensure vector table exists with correct dimensions.

    Returns True if vector table is available, False otherwise.
    """
    cursor = conn.cursor()

    # Check if vec0 is available
    try:
        conn.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        return False

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_documents_vec'")
    if cursor.fetchone():
        return True

    # Create vector table
    try:
        conn.executescript(vector_schema(dimensions))
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False


def extract_text_fields(source: dict, mappings: dict | None = None) -> str:
    """Extract text from document for FTS indexing.

    Concatenates all string fields (recursively) into a single text blob.
    If mappings specify text fields, only those are extracted.
    """
    if mappings and "properties" in mappings:
        # Use mapping to find text fields
        text_fields = []
        for field, config in mappings["properties"].items():
            if config.get("type") == "text" and field in source:
                value = source[field]
                if isinstance(value, str):
                    text_fields.append(value)
        return " ".join(text_fields)

    # Default: extract all string values recursively
    texts = []

    def extract(obj):
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)

    extract(source)
    return " ".join(texts)
