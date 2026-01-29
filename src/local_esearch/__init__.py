"""
local_esearch - Elasticsearch-compatible API backed by SQLite + FTS5 + sqlite-vec

Provides a drop-in replacement for elasticsearch-py in local/embedded use cases.
"""

from local_esearch import helpers
from local_esearch.client import Elasticsearch
from local_esearch.exceptions import (
    ConflictError,
    ElasticsearchException,
    NotFoundError,
    RequestError,
)

__version__ = "0.1.0"
__all__ = [
    "Elasticsearch",
    "ElasticsearchException",
    "NotFoundError",
    "ConflictError",
    "RequestError",
    "helpers",
]
