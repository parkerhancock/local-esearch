"""Elasticsearch-compatible helper functions (bulk, scan, etc.)."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Any

from local_esearch.response import format_bulk_item

if TYPE_CHECKING:
    from local_esearch.client import Elasticsearch


def bulk(
    client: Elasticsearch,
    actions: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 500,
    refresh: bool = False,
    raise_on_error: bool = True,
    stats_only: bool = False,
) -> tuple[int, int | list[dict[str, Any]]]:
    """Perform bulk indexing operations.

    Args:
        client: Elasticsearch client instance
        actions: Iterable of action dicts. Each dict should have:
            - _op_type: "index", "create", "update", "delete" (default: "index")
            - _index: Index name
            - _id: Document ID (optional for index/create)
            - _source: Document body (for index/create/update)
        chunk_size: Number of actions per batch
        refresh: Refresh indices after bulk
        raise_on_error: Raise exception if any action fails
        stats_only: Return (success_count, error_count) instead of error details

    Returns:
        Tuple of (success_count, error_count) if stats_only=True
        Tuple of (success_count, error_list) otherwise
    """
    success_count = 0
    errors = []

    for action in actions:
        op_type = action.get("_op_type", "index")
        index = action.get("_index")
        doc_id = action.get("_id")
        source = action.get("_source")

        if not index:
            errors.append(format_bulk_item(
                op_type, "", doc_id or "",
                status=400,
                error={"type": "action_request_validation_exception", "reason": "index is missing"}
            ))
            continue

        try:
            if op_type == "index":
                client.index(index=index, id=doc_id, document=source)
                success_count += 1

            elif op_type == "create":
                client.index(index=index, id=doc_id, document=source, op_type="create")
                success_count += 1

            elif op_type == "update":
                # Handle both {"doc": {...}} and direct {...} formats
                if isinstance(source, dict) and "doc" in source:
                    body = source
                else:
                    body = {"doc": source}
                client.update(index=index, id=doc_id, body=body)
                success_count += 1

            elif op_type == "delete":
                client.delete(index=index, id=doc_id)
                success_count += 1

            else:
                errors.append(format_bulk_item(
                    op_type, index, doc_id or "",
                    status=400,
                    error={
                        "type": "illegal_argument_exception",
                        "reason": f"Unknown op_type: {op_type}",
                    },
                ))

        except Exception as e:
            error_item = format_bulk_item(
                op_type, index, doc_id or "",
                status=getattr(e, "status_code", 500),
                error={"type": type(e).__name__, "reason": str(e)}
            )
            errors.append(error_item)

            if raise_on_error:
                raise

    # Refresh if requested
    if refresh:
        client.indices.refresh()

    if stats_only:
        return success_count, len(errors)
    return success_count, errors


def scan(
    client: Elasticsearch,
    *,
    index: str | list[str] | None = None,
    query: dict[str, Any] | None = None,
    scroll: str = "5m",
    size: int = 1000,
    preserve_order: bool = False,
    _source: bool | list[str] = True,
) -> Generator[dict[str, Any], None, None]:
    """Iterate over all documents matching a query.

    This is a simplified implementation that uses pagination instead of
    the scroll API (which doesn't exist in our SQLite backend).

    Args:
        client: Elasticsearch client instance
        index: Index name(s) to search
        query: Query DSL (default: match_all)
        scroll: Scroll timeout (ignored, kept for compatibility)
        size: Number of documents per batch
        preserve_order: Maintain document order (default: False)
        _source: Source filtering

    Yields:
        Document dicts with _index, _id, _source
    """
    if query is None:
        query = {"match_all": {}}

    offset = 0
    while True:
        response = client.search(
            index=index,
            body={"query": query},
            size=size,
            from_=offset,
            _source=_source,
        )

        hits = response["hits"]["hits"]
        if not hits:
            break

        yield from hits

        offset += len(hits)

        # Check if we've gotten all documents
        total = response["hits"]["total"]["value"]
        if offset >= total:
            break


def streaming_bulk(
    client: Elasticsearch,
    actions: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 500,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
) -> Generator[tuple[bool, dict[str, Any]], None, None]:
    """Streaming version of bulk that yields results as they complete.

    Args:
        client: Elasticsearch client instance
        actions: Iterable of action dicts
        chunk_size: Batch size (not used in this simple implementation)
        raise_on_error: Raise on ES errors
        raise_on_exception: Raise on Python exceptions

    Yields:
        Tuples of (success: bool, result_dict)
    """
    for action in actions:
        op_type = action.get("_op_type", "index")
        index = action.get("_index")
        doc_id = action.get("_id")
        source = action.get("_source")

        try:
            if op_type == "index":
                result = client.index(index=index, id=doc_id, document=source)
            elif op_type == "create":
                result = client.index(index=index, id=doc_id, document=source, op_type="create")
            elif op_type == "update":
                # Handle both {"doc": {...}} and direct {...} formats
                if isinstance(source, dict) and "doc" in source:
                    body = source
                else:
                    body = {"doc": source}
                result = client.update(index=index, id=doc_id, body=body)
            elif op_type == "delete":
                result = client.delete(index=index, id=doc_id)
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            yield True, {op_type: result}

        except Exception as e:
            error_info = {
                op_type: {
                    "_index": index,
                    "_id": doc_id,
                    "error": {"type": type(e).__name__, "reason": str(e)},
                    "status": getattr(e, "status_code", 500),
                }
            }
            yield False, error_info

            if raise_on_exception:
                raise


def parallel_bulk(
    client: Elasticsearch,
    actions: Iterable[dict[str, Any]],
    *,
    thread_count: int = 4,
    chunk_size: int = 500,
    raise_on_error: bool = True,
) -> Generator[tuple[bool, dict[str, Any]], None, None]:
    """Parallel bulk indexing (simplified - actually sequential for SQLite).

    SQLite doesn't benefit from parallel writes due to locking, so this
    implementation just calls streaming_bulk for compatibility.

    Args:
        client: Elasticsearch client instance
        actions: Iterable of action dicts
        thread_count: Ignored (SQLite is single-writer)
        chunk_size: Batch size
        raise_on_error: Raise on errors

    Yields:
        Tuples of (success: bool, result_dict)
    """
    yield from streaming_bulk(
        client,
        actions,
        chunk_size=chunk_size,
        raise_on_error=raise_on_error,
    )
