"""Elasticsearch Query DSL to SQL compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompiledQuery:
    """Result of compiling an ES query to SQL."""

    where_clause: str = ""
    params: list[Any] = field(default_factory=list)
    fts_query: str | None = None  # For FTS5 MATCH
    uses_fts: bool = False
    order_by: str | None = None


class QueryCompiler:
    """Compiles Elasticsearch Query DSL to SQL.

    Supports: match, match_phrase, match_all, term, terms, range, bool, exists
    """

    def compile(self, query: dict[str, Any], index: str | None = None) -> CompiledQuery:
        """Compile an ES query dict to SQL components."""
        if not query:
            return CompiledQuery()

        return self._compile_query(query)

    def _compile_query(self, query: dict[str, Any]) -> CompiledQuery:
        """Compile a single query clause."""
        if not query:
            return CompiledQuery()

        # Get the query type (should be exactly one key)
        query_type = next(iter(query.keys()), None)
        if query_type is None:
            return CompiledQuery()

        query_body = query[query_type]

        if query_type == "match_all":
            return CompiledQuery()

        if query_type == "match":
            return self._compile_match(query_body)

        if query_type == "match_phrase":
            return self._compile_match_phrase(query_body)

        if query_type == "term":
            return self._compile_term(query_body)

        if query_type == "terms":
            return self._compile_terms(query_body)

        if query_type == "range":
            return self._compile_range(query_body)

        if query_type == "bool":
            return self._compile_bool(query_body)

        if query_type == "exists":
            return self._compile_exists(query_body)

        if query_type == "ids":
            return self._compile_ids(query_body)

        if query_type == "prefix":
            return self._compile_prefix(query_body)

        if query_type == "wildcard":
            return self._compile_wildcard(query_body)

        # Unknown query type - return empty (match all)
        return CompiledQuery()

    def _compile_match(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile match query to FTS5."""
        # body is {field: value} or {field: {query: value, ...}}
        field_name = next(iter(body.keys()))
        field_value = body[field_name]

        if isinstance(field_value, dict):
            query_text = field_value.get("query", "")
        else:
            query_text = str(field_value)

        if not query_text:
            return CompiledQuery()

        # Convert to FTS5 query - simple word matching
        # FTS5 uses implicit OR between terms by default
        fts_terms = self._sanitize_fts_query(query_text)

        return CompiledQuery(
            where_clause="rowid IN (SELECT rowid FROM _documents_fts WHERE _documents_fts MATCH ?)",
            params=[fts_terms],
            fts_query=fts_terms,
            uses_fts=True,
        )

    def _compile_match_phrase(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile match_phrase query to FTS5 phrase search."""
        field_name = next(iter(body.keys()))
        field_value = body[field_name]

        if isinstance(field_value, dict):
            query_text = field_value.get("query", "")
        else:
            query_text = str(field_value)

        if not query_text:
            return CompiledQuery()

        # FTS5 phrase query - wrap in double quotes
        fts_query = f'"{self._sanitize_fts_query(query_text)}"'

        return CompiledQuery(
            where_clause="rowid IN (SELECT rowid FROM _documents_fts WHERE _documents_fts MATCH ?)",
            params=[fts_query],
            fts_query=fts_query,
            uses_fts=True,
        )

    def _compile_term(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile term query (exact match on keyword field)."""
        field_name = next(iter(body.keys()))
        field_value = body[field_name]

        if isinstance(field_value, dict):
            value = field_value.get("value")
        else:
            value = field_value

        # Use JSON extract for field access
        json_path = self._field_to_json_path(field_name)
        return CompiledQuery(
            where_clause="json_extract(_source, ?) = ?",
            params=[json_path, value],
        )

    def _compile_terms(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile terms query (match any of multiple values)."""
        field_name = next(iter(body.keys()))
        values = body[field_name]

        if not values:
            return CompiledQuery(where_clause="1=0")  # Match nothing

        json_path = self._field_to_json_path(field_name)
        placeholders = ",".join("?" for _ in values)
        return CompiledQuery(
            where_clause=f"json_extract(_source, ?) IN ({placeholders})",
            params=[json_path] + list(values),
        )

    def _compile_range(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile range query."""
        field_name = next(iter(body.keys()))
        conditions = body[field_name]

        json_path = self._field_to_json_path(field_name)
        clauses = []
        params = []

        for op, value in conditions.items():
            if op == "gt":
                clauses.append("json_extract(_source, ?) > ?")
                params.extend([json_path, value])
            elif op == "gte":
                clauses.append("json_extract(_source, ?) >= ?")
                params.extend([json_path, value])
            elif op == "lt":
                clauses.append("json_extract(_source, ?) < ?")
                params.extend([json_path, value])
            elif op == "lte":
                clauses.append("json_extract(_source, ?) <= ?")
                params.extend([json_path, value])

        if not clauses:
            return CompiledQuery()

        return CompiledQuery(
            where_clause=" AND ".join(clauses),
            params=params,
        )

    def _compile_bool(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile bool query combining must, should, must_not, filter."""
        must = body.get("must", [])
        should = body.get("should", [])
        must_not = body.get("must_not", [])
        filter_clauses = body.get("filter", [])
        minimum_should_match = body.get("minimum_should_match", 1 if should and not must else 0)

        # Normalize to lists
        if isinstance(must, dict):
            must = [must]
        if isinstance(should, dict):
            should = [should]
        if isinstance(must_not, dict):
            must_not = [must_not]
        if isinstance(filter_clauses, dict):
            filter_clauses = [filter_clauses]

        all_clauses = []
        all_params = []
        uses_fts = False
        fts_queries = []

        # Must clauses (AND)
        for clause in must:
            compiled = self._compile_query(clause)
            if compiled.where_clause:
                all_clauses.append(f"({compiled.where_clause})")
                all_params.extend(compiled.params)
            if compiled.uses_fts:
                uses_fts = True
                if compiled.fts_query:
                    fts_queries.append(compiled.fts_query)

        # Filter clauses (AND, no scoring)
        for clause in filter_clauses:
            compiled = self._compile_query(clause)
            if compiled.where_clause:
                all_clauses.append(f"({compiled.where_clause})")
                all_params.extend(compiled.params)
            if compiled.uses_fts:
                uses_fts = True
                if compiled.fts_query:
                    fts_queries.append(compiled.fts_query)

        # Must not clauses (AND NOT)
        for clause in must_not:
            compiled = self._compile_query(clause)
            if compiled.where_clause:
                all_clauses.append(f"NOT ({compiled.where_clause})")
                all_params.extend(compiled.params)

        # Should clauses (OR, at least minimum_should_match)
        if should:
            should_parts = []
            for clause in should:
                compiled = self._compile_query(clause)
                if compiled.where_clause:
                    should_parts.append(f"({compiled.where_clause})")
                    all_params.extend(compiled.params)
                if compiled.uses_fts:
                    uses_fts = True
                    if compiled.fts_query:
                        fts_queries.append(compiled.fts_query)

            if should_parts and minimum_should_match > 0:
                # For simplicity, if minimum_should_match >= 1, any should match works
                all_clauses.append(f"({' OR '.join(should_parts)})")

        where = " AND ".join(all_clauses) if all_clauses else ""
        fts_query = " ".join(fts_queries) if fts_queries else None

        return CompiledQuery(
            where_clause=where,
            params=all_params,
            uses_fts=uses_fts,
            fts_query=fts_query,
        )

    def _compile_exists(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile exists query - check if field has value."""
        field_name = body.get("field", "")
        if not field_name:
            return CompiledQuery()

        json_path = self._field_to_json_path(field_name)
        return CompiledQuery(
            where_clause="json_extract(_source, ?) IS NOT NULL",
            params=[json_path],
        )

    def _compile_ids(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile ids query - match specific document IDs."""
        values = body.get("values", [])
        if not values:
            return CompiledQuery(where_clause="1=0")

        placeholders = ",".join("?" for _ in values)
        return CompiledQuery(
            where_clause=f"_id IN ({placeholders})",
            params=list(values),
        )

    def _compile_prefix(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile prefix query."""
        field_name = next(iter(body.keys()))
        field_value = body[field_name]

        if isinstance(field_value, dict):
            value = field_value.get("value", "")
        else:
            value = str(field_value)

        json_path = self._field_to_json_path(field_name)
        return CompiledQuery(
            where_clause="json_extract(_source, ?) LIKE ?",
            params=[json_path, f"{value}%"],
        )

    def _compile_wildcard(self, body: dict[str, Any]) -> CompiledQuery:
        """Compile wildcard query (* and ? wildcards)."""
        field_name = next(iter(body.keys()))
        field_value = body[field_name]

        if isinstance(field_value, dict):
            value = field_value.get("value", "")
        else:
            value = str(field_value)

        # Convert ES wildcards to SQL LIKE
        # * -> %  (match any characters)
        # ? -> _  (match single character)
        sql_pattern = value.replace("*", "%").replace("?", "_")

        json_path = self._field_to_json_path(field_name)
        return CompiledQuery(
            where_clause="json_extract(_source, ?) LIKE ?",
            params=[json_path, sql_pattern],
        )

    def _field_to_json_path(self, field: str) -> str:
        """Convert ES field name to JSON path.

        Examples:
            'title' -> '$.title'
            'author.name' -> '$.author.name'
        """
        return f"$.{field}"

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query string for FTS5.

        Removes special FTS5 operators that could cause syntax errors.
        """
        # Remove FTS5 special characters
        for char in ['"', "'", "(", ")", "*", ":", "^", "-", "+", "OR", "AND", "NOT", "NEAR"]:
            query = query.replace(char, " ")
        # Collapse whitespace
        return " ".join(query.split())
