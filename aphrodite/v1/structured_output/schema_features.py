# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect JSON Schema keywords the structured output backends cannot handle.

Structured outputs use regexes or CFGs and are always greedy, which prevents some
schema features (uniqueItems, minContains, maxContains, etc) from being
practical. This file contains tools to search schemas for these features and
gracefully reject the unenforceable ones.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# Value is a single subschema ("items" also accepts the draft-04 tuple form).
_SUBSCHEMA_KEYS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)

# Value is a list of subschemas.
_SUBSCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})

# Value maps user-chosen names to subschemas; the names are not keywords.
_SUBSCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)

# Enforcing these means comparing an element against every element before it,
# which needs unbounded memory across elements and so is not expressible in any
# backend's formalism. Rejected for all backends, not just the ones that notice.
UNENFORCEABLE_JSON_SCHEMA_KEYS = frozenset(
    {
        "contains",
        "maxContains",
        "minContains",
        "uniqueItems",
    }
)


def iter_schema_nodes(schema: Any) -> Iterator[dict[str, Any]]:
    """Yield every node of ``schema`` sitting in JSON Schema keyword position."""
    if not isinstance(schema, dict):
        return
    yield schema

    for key, value in schema.items():
        if key in _SUBSCHEMA_KEYS:
            if isinstance(value, list):
                for item in value:
                    yield from iter_schema_nodes(item)
            else:
                yield from iter_schema_nodes(value)
        elif key in _SUBSCHEMA_LIST_KEYS:
            if isinstance(value, list):
                for item in value:
                    yield from iter_schema_nodes(item)
        elif key in _SUBSCHEMA_MAP_KEYS:
            if isinstance(value, dict):
                for item in value.values():
                    yield from iter_schema_nodes(item)


def find_schema_keys(schema: Any, keys: frozenset[str]) -> list[str]:
    """Return the members of ``keys`` present anywhere in ``schema``, sorted."""
    found: set[str] = set()
    for node in iter_schema_nodes(schema):
        found.update(keys.intersection(node))
    return sorted(found)


def get_unenforceable_json_schema_keys(schema: Any) -> list[str]:
    """Return the unenforceable keywords used by ``schema``, sorted."""
    return find_schema_keys(schema, UNENFORCEABLE_JSON_SCHEMA_KEYS)
