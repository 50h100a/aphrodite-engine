# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen JSON schemas before they reach a structured output backend, and say
which backends can enforce them.

`get_structured_outputs_schema_error` is the screen, shared by the API
entrypoints and the engine so the two layers cannot drift, and covering both
routes into the grammar -- `response_format` and the structural tag a tool call
is wrapped in -- so the two routes cannot drift either. It rejects schemas that
are malformed (`get_schema_validation_error`), schemas using keywords nothing
enforces -- neither a backend nor the `postconditions` layer above them
(`get_unenforceable_json_schema_keys`) -- and schemas whose
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

import copy
import json
from collections.abc import Iterator
from typing import Any

import jsonschema
import jsonschema.exceptions
import jsonschema.validators

import aphrodite.envs as envs
from aphrodite.logger import init_logger
from aphrodite.v1.structured_output import postconditions

logger = init_logger(__name__)

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
    # No backend enforces these. A CFG cannot count, compare across the
    # document, or remember what it already emitted.
    #
    # `contains`/`minContains`/`uniqueItems` are still enforceable, by
    # `postconditions` on top of whichever backend decodes, and only where they
    # sit -- so ask `_LayerVerdict.backends` about a particular node rather than
    # reading them off this table.
    "contains": frozenset(),
    "dependentRequired": frozenset(),
    "dependentSchemas": frozenset(),
    "else": frozenset(),
    "if": frozenset(),
    "maxContains": frozenset(),
    "minContains": frozenset(),
    # Outlines and lm-format-enforcer used to be recorded here as enforcing
    # `not`. Neither does: both drop it and close the object instead, which
    # refuses a violation of a `not` over `required` without ever reading the
    # keyword, and refuses the permitted extra keys along with it.
    "not": frozenset(),
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
    "patternProperties": frozenset({"guidance"}),
    "prefixItems": frozenset({"xgrammar", "guidance", "outlines"}),
    "propertyNames": frozenset({"xgrammar"}),
    "unevaluatedItems": frozenset({"xgrammar", "outlines"}),
    "unevaluatedProperties": frozenset({"xgrammar", "outlines", "lm-format-enforcer"}),
}

# Which backends can carry a keyword the postcondition layer enforces. The
# layer works with any backend, but the schema still has to *compile*, which is
# measured the same way as the table above: xgrammar, outlines and
# lm-format-enforcer read these keywords, ignore them, and enforce the rest, so
# the layer supplies only what they dropped. llguidance refuses the whole schema
# ("Unimplemented keys"), so routing has to divert before it dispatches or the
# compile throws on the grammar thread and reaches the caller as a 500.
_LAYER_ENFORCED_BACKENDS = JSON_SCHEMA_BACKENDS - {"guidance"}

# Table keywords that constrain nothing when set to their permissive value, so
# they are not held against the schema.
_VACUOUS_WHEN = {
    "unevaluatedItems": True,
    "unevaluatedProperties": True,
}

# The keywords no *backend* enforces -- read off the table rather than repeated,
# so a keyword that later gains a backend leaves this set on its own. Not the
# last word on whether a schema is enforceable: `contains` and `minContains` are
# in here and the postcondition layer still enforces them.
_UNENFORCEABLE_KEYWORDS = frozenset(key for key, backends in _KEYWORD_BACKENDS.items() if not backends)


def _constrains(node: dict[str, Any], key: str) -> bool:
    """Whether ``node[key]`` actually restricts anything.

    A keyword that is present but inert should not cost the caller a backend or
    a rejection: `unevaluatedItems: true` says nothing, and `if` without a
    `then` or an `else` to apply is a no-op the caller most likely did not
    intend as one.
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
    if key in ("contains", "minContains", "maxContains"):
        # All three stand or fall together, on the question the decode-time
        # layer asks, so a keyword it would never fire on is not one the caller
        # is refused for.
        return postconditions.contains_obligation(node) is not None
    if key == "uniqueItems":
        return postconditions.unique_obligation(node)
    return True


def iter_schema_nodes(schema: Any) -> Iterator[dict[str, Any]]:
    """Yield every node of ``schema`` sitting in JSON Schema keyword position.

    Subschemas under an inert keyword are skipped -- an `if` with no `then` or
    `else` applies to nothing, so a keyword inside it can never constrain the
    output and must not be held against the caller.
    """
    if not isinstance(schema, dict):
        return
    yield schema

    for key, value in schema.items():
        if key in _SUBSCHEMA_KEYS:
            if key in _KEYWORD_BACKENDS and not _constrains(schema, key):
                continue
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


class _LayerVerdict:
    """Which backends enforce a keyword *at one node* of one schema.

    Everything but the postcondition keywords comes straight off the table. For
    those, enforceability depends on where the keyword sits, so the answer comes
    from the same `analyze` the runtime uses and the two cannot disagree about
    what was promised.
    """

    __slots__ = ("_analysis",)

    def __init__(self, schema: Any, available: bool = True):
        if not available or not isinstance(schema, dict):
            # The structural-tag route, which the layer does not serve: the
            # scanner would have to find the schema body inside the tag's own
            # trigger/begin/end output to know where the array starts.
            self._analysis = postconditions.SchemaAnalysis()
        else:
            self._analysis = postconditions.analyze(schema)

    @property
    def reasons(self) -> list[str]:
        return self._analysis.problems

    def backends(self, node: dict[str, Any], key: str) -> frozenset[str]:
        if self._analysis.enforces(node, key):
            return _LAYER_ENFORCED_BACKENDS
        return _KEYWORD_BACKENDS[key]


def _iter_constraining_nodes(schema: Any) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield each (node, keyword) in ``schema`` that is in the table and constrains."""
    for node in iter_schema_nodes(schema):
        for key in _KEYWORD_BACKENDS.keys() & node.keys():
            if _constrains(node, key):
                yield node, key


def _as_schema(schema: Any) -> Any:
    """Parse ``schema`` if it is JSON text. Unparseable text is left alone;
    malformed JSON is reported by whoever parses it for real."""
    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except ValueError:
            return None
    return schema


def get_unenforceable_json_schema_keys(schema: Any, *, postconditions_available: bool = True) -> list[str]:
    """Return the keywords in ``schema`` that nothing enforces, sorted.

    ``schema`` may be a dict or JSON text. Unparseable text yields nothing.

    ``postconditions_available`` is False for routes the decode-time layer
    cannot serve, which is how a keyword can be enforceable in a response format
    and refused in a tool call.
    """
    parsed = _as_schema(schema)
    verdict = _LayerVerdict(parsed, postconditions_available)
    return sorted({key for node, key in _iter_constraining_nodes(parsed) if not verdict.backends(node, key)})


def get_unenforceable_reasons(schema: Any) -> list[str]:
    """Why the decode-time layer had to refuse ``schema``, if it did.

    The keyword name alone does not distinguish a schema the caller should
    rewrite from one they should give up on; the layer knows which.
    """
    return _LayerVerdict(_as_schema(schema)).reasons


def get_json_schema_backends(schema: Any, *, postconditions_available: bool = True) -> frozenset[str]:
    """Return the backends that enforce every constraining keyword in ``schema``.

    Empty means the schema cannot be enforced by anything we have;
    ``get_unenforceable_json_schema_keys`` says which keyword is to blame.
    """
    parsed = _as_schema(schema)
    verdict = _LayerVerdict(parsed, postconditions_available)
    backends = JSON_SCHEMA_BACKENDS
    for node, key in _iter_constraining_nodes(parsed):
        backends &= verdict.backends(node, key)
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
    parsed = _as_schema(schema)
    verdict = _LayerVerdict(parsed)
    keywords = {key: verdict.backends(node, key) for node, key in _iter_constraining_nodes(parsed)}
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
    blame: set[str] = set()
    for schema in iter_request_json_schemas(structured_outputs):
        parsed = _as_schema(schema)
        verdict = _LayerVerdict(parsed, available=False)
        blame |= {
            key for node, key in _iter_constraining_nodes(parsed) if not (verdict.backends(node, key) & tag_backends)
        }
    return sorted(blame)


def unenforceable_keys_message(keys: list[str], reasons: list[str] | None = None) -> str:
    """The rejection text, shared so the API and engine layers cannot drift.

    ``reasons`` says what the keyword name cannot: `contains` is enforceable in
    general, so "this one is not" leaves the caller nowhere to go.
    """
    detail = f" ({reasons[0]})" if reasons else ""
    return (
        f"JSON schema keyword(s) {keys} cannot be enforced by structured output{detail}. "
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
    """Backends that enforce every keyword across all of a request's schemas.

    The two routes are asked separately: the decode-time layer serves
    `response_format` and not the tag a tool call rides in.
    """
    backends = JSON_SCHEMA_BACKENDS
    if (schema := getattr(structured_outputs, "json", None)) is not None:
        backends &= get_json_schema_backends(schema)
    for schema in iter_structural_tag_schemas(getattr(structured_outputs, "structural_tag", None)):
        backends &= get_json_schema_backends(schema, postconditions_available=False)
    return backends


def _strip_unenforceable_in_place(schema: dict[str, Any]) -> set[str]:
    """Remove every unenforceable keyword from ``schema``, returning what went.

    Exactly the keywords ``get_unenforceable_json_schema_keys`` would have
    reported: a keyword that is present but inert is not reported, is not the
    reason for any rejection, and so is left where it is.
    """
    removed: set[str] = set()
    verdict = _LayerVerdict(schema)
    # Materialised before the first deletion: the walk is about to read each
    # node's keys, and mutating a node it has not descended into yet would pull
    # the ground out from under it.
    for node in list(iter_schema_nodes(schema)):
        # Decided for the whole node before anything is removed. `if`, `then`
        # and `else` each constrain only in the company of the others, so
        # deleting one first would make the rest look inert and spare them.
        doomed = {
            key
            for key in _KEYWORD_BACKENDS.keys() & node.keys()
            if _constrains(node, key) and not verdict.backends(node, key)
        }
        for key in doomed:
            del node[key]
        removed |= doomed
    return removed


def _without_unenforceable(schema: Any) -> tuple[Any, set[str]]:
    """``schema`` minus its unenforceable keywords, and which those were.

    The original is never touched -- a tool's parameters dict may be shared with
    whatever else the caller does with the tool -- and it is handed straight back
    when there was nothing to remove. JSON text in, JSON text out.
    """
    parsed = _as_schema(schema)
    if not isinstance(parsed, dict):
        return schema, set()
    stripped = copy.deepcopy(parsed)
    removed = _strip_unenforceable_in_place(stripped)
    if not removed:
        return schema, set()
    return (json.dumps(stripped) if isinstance(schema, str) else stripped), removed


def _without_unenforceable_tag(structural_tag: Any) -> tuple[Any, set[str]]:
    """The same, for the schemas riding inside a structural tag.

    A tool's parameters arrive by this route and no other, so a tag left
    unwalked is a tool call still rejected for a keyword the flag said to drop.
    """
    parsed = structural_tag
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            # Malformed; whoever parses it for real is the one to say so.
            return structural_tag, set()
    if not isinstance(parsed, dict):
        return structural_tag, set()

    stripped = copy.deepcopy(parsed)
    removed: set[str] = set()
    # The schemas come back by reference, so stripping them lands on the copy.
    for schema in list(iter_structural_tag_schemas(stripped)):
        removed |= _strip_unenforceable_in_place(schema)
    if not removed:
        return structural_tag, set()
    return (json.dumps(stripped) if isinstance(structural_tag, str) else stripped), removed


def relax_unenforceable_keywords(structured_outputs: Any) -> list[str]:
    """Drop the keywords nothing enforces from a request's schemas, in place.

    Best-effort mode: what would have been a 400 becomes a schema the decoder
    can actually compile, with everything still enforceable left standing and
    the dropped keywords the caller's to check afterwards.

    The request's own fields are rewritten rather than a cleaned copy returned,
    because the schema that reaches the grammar has to be the one that was
    screened. Routing asks which backends can enforce this request the moment
    the screen returns; a keyword left behind for that question to find would
    answer "none of them" and fail the request anyway, one layer further down.

    Returns the keywords removed, sorted.
    """
    removed: set[str] = set()

    schema = getattr(structured_outputs, "json", None)
    if schema is not None:
        structured_outputs.json, gone = _without_unenforceable(schema)
        removed |= gone

    tag = getattr(structured_outputs, "structural_tag", None)
    if tag is not None:
        structured_outputs.structural_tag, gone = _without_unenforceable_tag(tag)
        removed |= gone

    return sorted(removed)


def get_structured_outputs_schema_error(structured_outputs: Any) -> str | None:
    """Return why this request's schemas cannot be enforced, or None if they can.

    The single screen behind both the API entrypoints and the engine, covering
    both routes into the grammar, so that neither the two layers nor the two
    routes can drift apart.

    In best-effort mode this also *edits* ``structured_outputs``, dropping the
    keywords it would otherwise have rejected. It belongs here rather than at
    either call site for the same reason the screen does: two layers doing it
    separately is two layers that can come to disagree about what was dropped.
    """
    schemas = list(iter_request_json_schemas(structured_outputs))
    # Well-formedness first: the keywords of a malformed schema mean nothing.
    for schema in schemas:
        if (invalid := get_schema_validation_error(schema)) is not None:
            return schema_validation_message(invalid)

    if envs.APHRODITE_STRUCTURED_OUTPUT_BEST_EFFORT:
        # Only the keywords with no backend at all. The two conflict checks
        # below still reject, because there the keywords *are* enforceable and
        # choosing which of them to break is the caller's call, not ours.
        if dropped := relax_unenforceable_keywords(structured_outputs):
            logger.warning_once(
                "APHRODITE_STRUCTURED_OUTPUT_BEST_EFFORT is set: dropped JSON schema "
                "keyword(s) %s from a request, which no structured output backend can "
                "enforce while decoding. The rest of the schema is still enforced; "
                "these are the caller's to validate on the generated output.",
                str(dropped),
            )
            # Not the same objects any more.
            schemas = list(iter_request_json_schemas(structured_outputs))

    # The same schema can be enforceable as a response format and refused as a
    # tool, so each is asked by the route it arrived on.
    tag = getattr(structured_outputs, "structural_tag", None)
    tag_schemas = {id(schema) for schema in iter_structural_tag_schemas(tag)}
    for schema in schemas:
        available = id(schema) not in tag_schemas
        if unenforceable := get_unenforceable_json_schema_keys(schema, postconditions_available=available):
            return unenforceable_keys_message(unenforceable, get_unenforceable_reasons(schema) if available else None)
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
