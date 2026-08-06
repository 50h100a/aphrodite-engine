# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen JSON schemas before they reach a structured output backend, and say
which backends can enforce them.

`get_structured_outputs_schema_error` is the screen, shared by the API
entrypoints and the engine so the two layers cannot drift, and covering both
routes into the grammar -- `response_format` and the structural tag a tool call
is wrapped in -- so the two routes cannot drift either. It rejects schemas that
are malformed (`get_schema_validation_error`), schemas using keywords no
backend enforces (`get_unenforceable_json_schema_keys`), and schemas whose
keywords are individually enforceable but have no backend in common
(`get_json_schema_backend_conflict`), since a request decodes with one backend.

`get_json_schema_backends` answers the routing question for everything that
survives. It exists because a backend cannot be asked: xgrammar compiles a
schema containing `allOf`, lowers it to "any JSON value", warns only on stderr
from C++, and then decodes constraining nothing -- indistinguishable, from the
caller's side, from a model that happened to comply.

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

JSON_SCHEMA_BACKENDS = frozenset({"xgrammar", "guidance", "outlines", "lm-format-enforcer"})

# Which backends actually *enforce* each constraining keyword.
#
# Measured, not read off documentation: each backend was given a schema and an
# instance violating only the keyword under test, and counts as enforcing it
# only if it refuses that instance. Compiling without error is not enforcement.
# xgrammar accepts `allOf` and then lowers the whole schema to "any JSON value",
# warning on stderr from C++ and constraining nothing at all; llguidance is the
# only one that reports what it did not implement ("Unimplemented keys").
#
# Keywords absent from this table are enforced by every backend: type,
# properties, required, additionalProperties, enum, const, anyOf, oneOf, $ref,
# $defs, pattern, minLength/maxLength, minItems/maxItems, maxProperties.
# `format` is deliberately absent: it is an annotation, not an assertion.
_KEYWORD_BACKENDS: dict[str, frozenset[str]] = {
    # Nothing enforces these. A CFG cannot count, compare across the document,
    # or remember what it already emitted.
    "contains": frozenset(),
    "dependentRequired": frozenset(),
    "dependentSchemas": frozenset(),
    "else": frozenset(),
    "if": frozenset(),
    "maxContains": frozenset(),
    "minContains": frozenset(),
    "then": frozenset(),
    "uniqueItems": frozenset(),
    # Enforced by some. `auto` routes to one of these; naming a backend that is
    # not listed is refused rather than silently unenforced.
    "allOf": frozenset({"guidance"}),
    "exclusiveMaximum": frozenset({"xgrammar", "guidance"}),
    "exclusiveMinimum": frozenset({"xgrammar", "guidance"}),
    "maximum": frozenset({"xgrammar", "guidance"}),
    "minimum": frozenset({"xgrammar", "guidance"}),
    "minProperties": frozenset({"xgrammar", "guidance"}),
    "multipleOf": frozenset({"guidance"}),
    "not": frozenset({"outlines", "lm-format-enforcer"}),
    "patternProperties": frozenset({"guidance"}),
    "prefixItems": frozenset({"xgrammar", "guidance", "outlines"}),
    "propertyNames": frozenset({"xgrammar"}),
    "unevaluatedItems": frozenset({"xgrammar", "outlines"}),
    "unevaluatedProperties": frozenset({"xgrammar", "outlines", "lm-format-enforcer"}),
}

# Table keywords that constrain nothing when set to their permissive value, so
# they are not held against the schema.
_VACUOUS_WHEN = {
    "unevaluatedItems": True,
    "unevaluatedProperties": True,
    "uniqueItems": False,
}


def _constrains(node: dict[str, Any], key: str) -> bool:
    """Whether ``node[key]`` actually restricts anything.

    A keyword that is present but inert should not cost the caller a backend or
    a rejection: `uniqueItems: false` says nothing, and `if` without a `then` or
    an `else` to apply is a no-op the caller most likely did not intend as one.
    """
    value = node[key]
    if key in _VACUOUS_WHEN and value is _VACUOUS_WHEN[key]:
        return False
    if isinstance(value, (list, dict)) and not value:
        # `allOf: []`, `dependentRequired: {}` -- nothing to apply.
        return False
    if key == "if":
        return "then" in node or "else" in node
    if key in ("then", "else"):
        return "if" in node
    if key in ("minContains", "maxContains"):
        return "contains" in node
    return True


def _iter_constraining_keywords(schema: Any) -> Iterator[str]:
    """Yield each keyword in ``schema`` that is in the table and constrains."""
    for node in iter_schema_nodes(schema):
        for key in _KEYWORD_BACKENDS.keys() & node.keys():
            if _constrains(node, key):
                yield key


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


def _as_schema(schema: Any) -> Any:
    """Parse ``schema`` if it is JSON text. Unparseable text is left alone;
    malformed JSON is reported by whoever parses it for real."""
    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except ValueError:
            return None
    return schema


def get_unenforceable_json_schema_keys(schema: Any) -> list[str]:
    """Return the keywords in ``schema`` that no backend enforces, sorted.

    ``schema`` may be a dict or JSON text. Unparseable text yields nothing.
    """
    return sorted({key for key in _iter_constraining_keywords(_as_schema(schema)) if not _KEYWORD_BACKENDS[key]})


def get_json_schema_backends(schema: Any) -> frozenset[str]:
    """Return the backends that enforce every constraining keyword in ``schema``.

    Empty means the schema cannot be enforced by anything we have;
    ``get_unenforceable_json_schema_keys`` says which keyword is to blame.
    """
    backends = JSON_SCHEMA_BACKENDS
    for key in _iter_constraining_keywords(_as_schema(schema)):
        backends &= _KEYWORD_BACKENDS[key]
        if not backends:
            break
    return backends


def get_json_schema_backend_conflict(schema: Any) -> list[str]:
    """Return the keywords that pull ``schema`` towards different backends.

    Every one of them is enforceable on its own, but no single backend enforces
    all of them, and a request decodes with one backend. Empty when the schema
    has a home (or when some keyword has no home at all, which
    ``get_unenforceable_json_schema_keys`` reports instead).
    """
    keywords = {key: _KEYWORD_BACKENDS[key] for key in _iter_constraining_keywords(_as_schema(schema))}
    if not keywords or not all(keywords.values()):
        return []
    remaining = JSON_SCHEMA_BACKENDS
    for backends in keywords.values():
        remaining &= backends
    if remaining:
        return []
    return sorted(keywords)


# Which backends can compile a structural tag, by the form the tag takes.
#
# The tag is how a tool call reaches the grammar, and it is not a JSON schema,
# so schema keywords alone do not say where a tool request can go. xgrammar
# understands both spellings; guidance only the older `structures`/`triggers`
# one; outlines and lm-format-enforcer neither. The two that cannot do not say
# so when asked to validate one -- they accept it in silence and then raise from
# the engine's compile thread, which reaches the caller as a 500 -- so routing
# has to know this before it dispatches.
_STRUCTURES_TAG_BACKENDS = frozenset({"xgrammar", "guidance"})
_NESTED_TAG_BACKENDS = frozenset({"xgrammar"})


def get_structural_tag_backends(structural_tag: Any) -> frozenset[str]:
    """Backends that can compile ``structural_tag``, or all of them if there is none."""
    if structural_tag is None:
        return JSON_SCHEMA_BACKENDS
    if isinstance(structural_tag, str):
        try:
            structural_tag = json.loads(structural_tag)
        except ValueError:
            # Malformed; whoever parses it for real is the one to say so.
            return JSON_SCHEMA_BACKENDS
    if isinstance(structural_tag, dict) and "structures" in structural_tag:
        return _STRUCTURES_TAG_BACKENDS
    return _NESTED_TAG_BACKENDS


def get_backends_for_request(structured_outputs: Any) -> frozenset[str]:
    """Backends that can enforce everything a request asks for: schemas and tag.

    Routing reads this rather than ``get_json_schema_backends_for_request``,
    which answers only half the question for anything arriving by the tool route.
    """
    return get_json_schema_backends_for_request(structured_outputs) & get_structural_tag_backends(
        getattr(structured_outputs, "structural_tag", None)
    )


def get_structural_tag_backend_conflict(structured_outputs: Any) -> list[str]:
    """Return the tool-schema keywords that no structural-tag backend enforces.

    Each is enforceable somewhere and the tag is compilable somewhere, but never
    by the same backend, and a request decodes with one backend. Empty when the
    request has a home, or when it has no structural tag to place.
    """
    tag = getattr(structured_outputs, "structural_tag", None)
    if tag is None:
        return []
    tag_backends = get_structural_tag_backends(tag)
    if get_json_schema_backends_for_request(structured_outputs) & tag_backends:
        return []
    blame = {
        key
        for schema in iter_request_json_schemas(structured_outputs)
        for key in _iter_constraining_keywords(_as_schema(schema))
        if not (_KEYWORD_BACKENDS[key] & tag_backends)
    }
    return sorted(blame)


def unenforceable_keys_message(keys: list[str]) -> str:
    """The rejection text, shared so the API and engine layers cannot drift."""
    return (
        f"JSON schema keyword(s) {keys} cannot be enforced by structured output. "
        "Remove them from the schema and validate the generated output instead."
    )


def backend_conflict_message(keys: list[str]) -> str:
    """The rejection text for a schema no single backend can enforce whole."""
    wanted = ", ".join(f"{key} needs {sorted(_KEYWORD_BACKENDS[key])}" for key in keys)
    return (
        f"JSON schema keyword(s) {keys} cannot be enforced together: no one structured "
        f"output backend enforces all of them ({wanted}), and a request decodes with "
        "one backend. Drop one of them and validate the generated output instead."
    )


def structural_tag_conflict_message(keys: list[str], tag_backends: frozenset[str]) -> str:
    """The rejection text for a tool schema that cannot ride its own tag."""
    wanted = ", ".join(f"{key} needs {sorted(_KEYWORD_BACKENDS[key])}" for key in keys)
    return (
        f"JSON schema keyword(s) {keys} cannot be enforced in a tool call: the structural "
        "tag a tool call is wrapped in can only be compiled by "
        f"{sorted(tag_backends)}, which do not enforce them ({wanted}), and a request "
        "decodes with one backend. Drop them from the tool's parameters and validate "
        "the arguments instead."
    )


def get_schema_validation_error(schema: Any) -> str | None:
    """Return why ``schema`` is not a valid JSON Schema, or None if it is.

    The dialect comes from the schema's own ``$schema``, falling back to the
    newest jsonschema knows.

    ``schema`` may be a dict or JSON text. Unparseable text yields None;
    malformed JSON is reported by whoever parses it for real.
    """
    schema = _as_schema(schema)
    if schema is None:
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


def iter_request_json_schemas(structured_outputs: Any) -> Iterator[Any]:
    """Yield every JSON schema a request carries, whichever route it arrived by.

    ``json`` is the ``response_format`` route. The structural tag is the tool
    route: a tool's parameters are wrapped in a tag before they reach the
    grammar, so a screen that reads only ``json`` passes every tool call
    through unchecked.
    """
    if structured_outputs is None:
        return
    if (schema := getattr(structured_outputs, "json", None)) is not None:
        yield schema
    yield from iter_structural_tag_schemas(getattr(structured_outputs, "structural_tag", None))


def get_json_schema_backends_for_request(structured_outputs: Any) -> frozenset[str]:
    """Backends that enforce every keyword across all of a request's schemas."""
    backends = JSON_SCHEMA_BACKENDS
    for schema in iter_request_json_schemas(structured_outputs):
        backends &= get_json_schema_backends(schema)
    return backends


def get_structured_outputs_schema_error(structured_outputs: Any) -> str | None:
    """Return why this request's schemas cannot be enforced, or None if they can.

    The single screen behind both the API entrypoints and the engine, covering
    both routes into the grammar, so that neither the two layers nor the two
    routes can drift apart.
    """
    schemas = list(iter_request_json_schemas(structured_outputs))
    # Well-formedness first: the keywords of a malformed schema mean nothing.
    for schema in schemas:
        if (invalid := get_schema_validation_error(schema)) is not None:
            return schema_validation_message(invalid)
    for schema in schemas:
        if unenforceable := get_unenforceable_json_schema_keys(schema):
            return unenforceable_keys_message(unenforceable)
    for schema in schemas:
        if conflict := get_json_schema_backend_conflict(schema):
            return backend_conflict_message(conflict)
    # Last, because it is the same question asked across the tag rather than
    # within one schema: keywords with a home, and a tag with a home, that are
    # not the same home.
    if conflict := get_structural_tag_backend_conflict(structured_outputs):
        return structural_tag_conflict_message(
            conflict,
            get_structural_tag_backends(getattr(structured_outputs, "structural_tag", None)),
        )
    return None
