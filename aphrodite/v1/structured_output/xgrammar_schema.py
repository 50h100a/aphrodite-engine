# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject schema constructs XGrammar 0.2.8 silently lowers lossily.

Successful grammar construction is not a support check: the native parser
prioritizes $ref/const/enum/combinators over their siblings, and changes an
unproven oneOf into anyOf. Keep the installed dependency unchanged and refuse
such requests before decoding. This is a capability check, not output repair.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import Unresolvable

from aphrodite.v1.structured_output.schema_features import (
    _DEFINITION_KEYS,
    _MISSING,
    _SUBSCHEMA_KEYS,
    _SUBSCHEMA_LIST_KEYS,
    _SUBSCHEMA_MAP_KEYS,
    _resolve_local_pointer,
)

_ASSERTIONS = frozenset(jsonschema.Draft202012Validator.VALIDATORS) - {"format"}
_PRIORITY = ("$ref", "const", "enum", "anyOf", "oneOf", "allOf")
_LEGACY_REF = (
    jsonschema.Draft3Validator,
    jsonschema.Draft4Validator,
    jsonschema.Draft6Validator,
    jsonschema.Draft7Validator,
)
_SIMPLE_PREDICATES = frozenset(
    {
        "type",
        "const",
        "enum",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
    }
)
# JSONSchemaConverter::JSONFormatToRegexPattern in the pinned stock 0.2.8.
# Other format names remain annotations and do not supersede string bounds.
_REGEX_FORMATS = frozenset(
    {
        "email",
        "date",
        "time",
        "date-time",
        "duration",
        "ipv4",
        "ipv6",
        "hostname",
        "uuid",
        "uri",
        "uri-reference",
        "uri-template",
        "json-pointer",
        "relative-json-pointer",
    }
)


def get_xgrammar_schema_error(schema: Any) -> str | None:
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            return None  # The backend reports malformed JSON.
    if not isinstance(schema, dict):
        return None
    cls = jsonschema.validators.validator_for(schema)
    # Explicit registry prevents this support check fetching external refs.
    validator = cls(schema, registry=Registry())

    def accepts(node: Any, value: Any) -> bool:
        return validator.evolve(schema=node).is_valid(value)

    def simple(node: Any) -> bool:
        # Never run a caller's regex, or follow a remote reference, to prove
        # redundancy/disjointness on the API thread.
        return isinstance(node, bool) or (
            isinstance(node, dict) and not ((_ASSERTIONS & node.keys()) - _SIMPLE_PREDICATES)
        )

    def resolve(node: Any, seen=frozenset()) -> Any:
        if not isinstance(node, dict) or "$ref" not in node or id(node) in seen:
            return node
        ref = node["$ref"]
        if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
            return node
        target = _resolve_local_pointer(schema, ref)
        if target is _MISSING:
            return node
        # Only used for proofs; sibling assertions are checked separately.
        return resolve(target, seen | {id(node)})

    def finite(node: Any) -> list | None:
        node = resolve(node)
        if not isinstance(node, dict):
            return None
        if "const" in node:
            return [node["const"]]
        values = node.get("enum")
        return values if isinstance(values, list) else None

    def types(node: Any) -> set[str] | None:
        node = resolve(node)
        if not isinstance(node, dict):
            return None
        value = node.get("type")
        if isinstance(value, str):
            return {value}
        return set(value) if isinstance(value, list) else None

    def implies_type(node: Any, wanted: Any, seen=frozenset()) -> bool:
        node = resolve(node)
        if not isinstance(node, dict) or id(node) in seen:
            return False
        allowed = {wanted} if isinstance(wanted, str) else set(wanted)
        if (values := finite(node)) is not None:
            return all(accepts({"type": wanted}, value) for value in values)
        if (declared := types(node)) is not None:
            return all(t in allowed or (t == "integer" and "number" in allowed) for t in declared)
        for key in ("anyOf", "oneOf"):
            if isinstance(node.get(key), list):
                return all(implies_type(arm, wanted, seen | {id(node)}) for arm in node[key])
        return False

    def disjoint(left: Any, right: Any, seen=frozenset()) -> bool:
        left, right = resolve(left), resolve(right)
        pair = (id(left), id(right))
        if pair in seen:
            return False
        if left is False or right is False:
            return True
        for first, second in ((left, right), (right, left)):
            if (values := finite(first)) is not None:
                declared = types(second)
                if declared is not None and all(not accepts({"type": list(declared)}, value) for value in values):
                    return True
                if simple(second):
                    return all(not accepts(second, value) for value in values)
        lt, rt = types(left), types(right)
        if lt is not None and rt is not None:
            # A number arm overlaps an integer arm. Booleans are distinct.
            lt = lt | ({"integer"} if "number" in lt else set())
            rt = rt | ({"integer"} if "number" in rt else set())
            if lt.isdisjoint(rt):
                return True
        if lt != {"object"} or rt != {"object"}:
            return False
        lp, rp = left.get("properties", {}), right.get("properties", {})
        lr, rr = set(left.get("required", [])), set(right.get("required", []))
        for key in lr & rr & lp.keys() & rp.keys():
            if disjoint(lp[key], rp[key], seen | {pair}):
                return True
        # Closed objects can be disjoint even without a shared discriminator.
        for first, required in ((left, rr), (right, lr)):
            if (
                first.get("additionalProperties") is False
                and not first.get("patternProperties")
                and required - first.get("properties", {}).keys()
            ):
                return True
        return False

    pending = [schema]
    visited = set()
    try:
        while pending:
            node = pending.pop()
            if not isinstance(node, dict) or id(node) in visited:
                continue
            visited.add(id(node))
            for key in ("$dynamicRef", "$recursiveRef"):
                if key in node:
                    return f"{key} is not supported by XGrammar"
            dominant = next((key for key in _PRIORITY if key in node), None)
            if dominant is not None:
                ignored = (_ASSERTIONS & node.keys()) - {dominant}
                if dominant == "$ref" and cls in _LEGACY_REF:
                    ignored = set()  # Those dialects ignore $ref siblings too.
                elif dominant in ("const", "enum"):
                    # Permit genuinely redundant siblings, e.g. enum + type.
                    residual = {key: node[key] for key in ignored}
                    if simple(residual) and all(accepts(residual, value) for value in finite(node) or []):
                        ignored = set()
                elif "type" in ignored:
                    arms = node.get(dominant) if dominant in ("anyOf", "oneOf") else [resolve(node)]
                    if isinstance(arms, list) and all(implies_type(arm, node["type"]) for arm in arms):
                        ignored -= {"type"}
                if ignored:
                    return f"{dominant} with sibling constraint(s) {sorted(ignored)} is not supported by XGrammar"
            if isinstance(node.get("oneOf"), list) and not all(
                disjoint(a, b) for a, b in combinations(node["oneOf"], 2)
            ):
                return "oneOf branches overlap or cannot be proved disjoint; XGrammar would enforce anyOf instead"
            declared = types(node)
            if (declared is None or "string" in declared) and ("pattern" in node or "format" in node):
                bounded = node.get("minLength", 0) != 0 or "maxLength" in node
                if bounded and "pattern" in node:
                    return "pattern with minLength/maxLength is not jointly enforced by XGrammar"
                # Recognized formats take priority over pattern and lengths.
                if node.get("format") in _REGEX_FORMATS and (bounded or "pattern" in node):
                    return "format with pattern/minLength/maxLength is not jointly enforced by XGrammar"

            if "$ref" in node:
                ref = node["$ref"]
                if isinstance(ref, str) and (ref == "#" or ref.startswith("#/")):
                    target = _resolve_local_pointer(schema, ref)
                    if target is not _MISSING:
                        pending.append(target)
            for key, value in node.items():
                if key in _SUBSCHEMA_KEYS:
                    pending.extend(value if isinstance(value, list) else [value])
                elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
                    pending.extend(value)
                elif key in _SUBSCHEMA_MAP_KEYS - _DEFINITION_KEYS and isinstance(value, dict):
                    pending.extend(value.values())
    except (Unresolvable, ValueError, TypeError, RecursionError):
        return "schema references or alternatives cannot be validated for XGrammar"
    return None
