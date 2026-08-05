# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen JSON schemas before they reach a structured output backend.

Rejected here: schemas that are malformed (`get_schema_validation_error`), and
schemas using keywords no regex or CFG can enforce
(`get_unenforceable_json_schema_keys`).

Not checked here: whether a schema can be *satisfied*. `{"enum": []}` and a
contradictory `allOf` are well-formed but match nothing; deciding that in
general is intractable, and a partial check implies a guarantee it cannot
keep. Those end at runtime with `finish_reason="constraint"` instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import jsonschema
import jsonschema.exceptions
import jsonschema.validators

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

# These can't be enforced with regex or CFG. Maybe someday.
_UNENFORCEABLE_JSON_SCHEMA_KEYS = frozenset(
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


def get_unenforceable_json_schema_keys(schema: Any) -> list[str]:
    """Return the unenforceable keywords used by ``schema``, sorted.

    ``schema`` may be a dict or JSON text. Unparseable text yields nothing;
    malformed JSON is not ours to report.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            return []
    found: set[str] = set()
    for node in iter_schema_nodes(schema):
        found.update(_UNENFORCEABLE_JSON_SCHEMA_KEYS.intersection(node))
    return sorted(found)


def unenforceable_keys_message(keys: list[str]) -> str:
    """The rejection text, shared so the API and engine layers cannot drift."""
    return (
        f"JSON schema keyword(s) {keys} cannot be enforced by structured output. "
        "Remove them from the schema and validate the generated output instead."
    )


def get_schema_validation_error(schema: Any) -> str | None:
    """Return why ``schema`` is not a valid JSON Schema, or None if it is.

    The dialect comes from the schema's own ``$schema``, falling back to the
    newest jsonschema knows.

    ``schema`` may be a dict or JSON text. Unparseable text yields None;
    malformed JSON is reported by whoever parses it for real.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            return None
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.exceptions.SchemaError as err:
        location = "/".join(str(part) for part in err.absolute_path)
        return f"{err.message} at /{location}" if location else err.message
    except Exception:
        # Unrecognised `$schema`, unresolvable metaschema, etc. Not evidence
        # the caller's schema is bad, so let the backend decide.
        return None
    return None


def schema_validation_message(reason: str) -> str:
    """The rejection text, shared so the API and engine layers cannot drift."""
    return f"JSON schema is not valid: {reason}"


def iter_structural_tag_schemas(structural_tag: Any) -> Iterator[dict[str, Any]]:
    """Yield the JSON schemas embedded in a structural tag payload.

    Covers both shapes: the legacy ``structures`` list whose entries carry a
    ``schema``, and the newer nested ``format`` spelling them as
    ``{"type": "json_schema", "json_schema": ...}``. Tool calls arrive this
    way, so without this the screening above would skip every tool call.
    """
    if isinstance(structural_tag, str):
        try:
            structural_tag = json.loads(structural_tag)
        except ValueError:
            return
    if not isinstance(structural_tag, dict):
        return

    structures = structural_tag.get("structures")
    if isinstance(structures, list):
        for structure in structures:
            if isinstance(structure, dict) and isinstance(structure.get("schema"), dict):
                yield structure["schema"]

    yield from _iter_nested_json_schemas(structural_tag.get("format"))


def _iter_nested_json_schemas(node: Any) -> Iterator[dict[str, Any]]:
    """Walk the newer structural tag format for its ``json_schema`` nodes."""
    if isinstance(node, list):
        for item in node:
            yield from _iter_nested_json_schemas(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "json_schema" and isinstance(node.get("json_schema"), dict):
        # Its contents are schema, not more tag format; do not recurse in.
        yield node["json_schema"]
        return
    for value in node.values():
        yield from _iter_nested_json_schemas(value)
