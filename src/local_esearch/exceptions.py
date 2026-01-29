"""Elasticsearch-compatible exceptions."""

from __future__ import annotations

from typing import Any


class ElasticsearchException(Exception):
    """Base exception for Elasticsearch errors."""

    def __init__(
        self,
        message: str = "",
        meta: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.meta = meta or {}
        self.body = body or {}


class NotFoundError(ElasticsearchException):
    """Document or index not found."""

    status_code = 404


class ConflictError(ElasticsearchException):
    """Version conflict or document already exists."""

    status_code = 409


class RequestError(ElasticsearchException):
    """Invalid request (bad query, missing params, etc.)."""

    status_code = 400
