"""Elasticsearch-compatible response formatting."""

from __future__ import annotations

from typing import Any


def format_hit(
    index: str,
    doc_id: str,
    source: dict[str, Any],
    score: float | None = None,
    version: int | None = None,
) -> dict[str, Any]:
    """Format a single search hit in ES format."""
    hit = {
        "_index": index,
        "_id": doc_id,
        "_source": source,
    }
    if score is not None:
        hit["_score"] = score
    if version is not None:
        hit["_version"] = version
    return hit


def format_search_response(
    hits: list[dict[str, Any]],
    total: int,
    took_ms: int,
    max_score: float | None = None,
) -> dict[str, Any]:
    """Format search response in ES format."""
    return {
        "took": took_ms,
        "timed_out": False,
        "_shards": {
            "total": 1,
            "successful": 1,
            "skipped": 0,
            "failed": 0,
        },
        "hits": {
            "total": {
                "value": total,
                "relation": "eq",
            },
            "max_score": max_score,
            "hits": hits,
        },
    }


def format_get_response(
    index: str,
    doc_id: str,
    source: dict[str, Any],
    version: int,
    found: bool = True,
) -> dict[str, Any]:
    """Format document get response in ES format."""
    response = {
        "_index": index,
        "_id": doc_id,
        "found": found,
    }
    if found:
        response["_version"] = version
        response["_source"] = source
    return response


def format_index_response(
    index: str,
    doc_id: str,
    version: int,
    result: str = "created",
    seq_no: int = 0,
    primary_term: int = 1,
) -> dict[str, Any]:
    """Format index operation response in ES format."""
    return {
        "_index": index,
        "_id": doc_id,
        "_version": version,
        "result": result,
        "_shards": {
            "total": 1,
            "successful": 1,
            "failed": 0,
        },
        "_seq_no": seq_no,
        "_primary_term": primary_term,
    }


def format_delete_response(
    index: str,
    doc_id: str,
    version: int,
    result: str = "deleted",
) -> dict[str, Any]:
    """Format delete operation response in ES format."""
    return {
        "_index": index,
        "_id": doc_id,
        "_version": version,
        "result": result,
        "_shards": {
            "total": 1,
            "successful": 1,
            "failed": 0,
        },
    }


def format_update_response(
    index: str,
    doc_id: str,
    version: int,
    result: str = "updated",
) -> dict[str, Any]:
    """Format update operation response in ES format."""
    return {
        "_index": index,
        "_id": doc_id,
        "_version": version,
        "result": result,
        "_shards": {
            "total": 1,
            "successful": 1,
            "failed": 0,
        },
    }


def format_bulk_item(
    operation: str,
    index: str,
    doc_id: str,
    version: int | None = None,
    result: str | None = None,
    status: int = 200,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format a single bulk operation result."""
    item = {
        "_index": index,
        "_id": doc_id,
        "status": status,
    }
    if version is not None:
        item["_version"] = version
    if result is not None:
        item["result"] = result
    if error is not None:
        item["error"] = error
    return {operation: item}


def format_bulk_response(
    items: list[dict[str, Any]],
    took_ms: int,
    errors: bool = False,
) -> dict[str, Any]:
    """Format bulk operation response in ES format."""
    return {
        "took": took_ms,
        "errors": errors,
        "items": items,
    }
