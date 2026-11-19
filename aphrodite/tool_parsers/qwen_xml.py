# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen XML parameters composed with stock XGrammar's public format API.

Only the parameter list is ours. XGrammar still supplies the call, choice and
reasoning envelopes, and compiles every value schema. Raw string scalars use
its XML scalar mode; objects/arrays use JSON mode. Literal template newlines
surround each value, so whitespace in the value remains data.
"""

from __future__ import annotations

import itertools
import json
import re
from functools import cached_property
from types import SimpleNamespace

import jsonschema
from pydantic import BaseModel
from referencing import Registry
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TriggeredTagsFormat,
)

from aphrodite.exceptions import AphroditeValidationError
from aphrodite.tool_parsers.utils import resolve_local_pointer
from aphrodite.v1.structured_output.schema_features import (
    _SUBSCHEMA_KEYS,
    _SUBSCHEMA_LIST_KEYS,
    _SUBSCHEMA_MAP_KEYS,
    get_structured_outputs_schema_error,
    iter_schema_nodes,
)
from aphrodite.v1.structured_output.xgrammar_schema import get_xgrammar_schema_error

_TYPES = ("string", "number", "boolean", "null", "object", "array")
_ANNOTATIONS = {
    "$schema",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
}
_OBJECT_KEYS = {"type", "properties", "required", "additionalProperties", "minProperties", "maxProperties"}
_LEGACY_REF = (
    jsonschema.Draft3Validator,
    jsonschema.Draft4Validator,
    jsonschema.Draft6Validator,
    jsonschema.Draft7Validator,
)


def _error(reason: str):
    raise AphroditeValidationError(f"Qwen XML tool schema is not supported: {reason}")


def _choice(elements):
    if not elements:
        _error("schema has no representable values")
    return elements[0] if len(elements) == 1 else OrFormat(elements=elements)


def _sequence(elements):
    return SequenceFormat(elements=elements) if elements else ConstStringFormat(value="")


def _other_key_pattern(keys: list[str]) -> str:
    """XML identifier language minus the declared keys, without lookahead.

    A trie complement prevents an additional parameter from overwriting a
    declared parameter. Repeated *additional* keys remain harmless when no
    cardinality constraint depends on them.
    """
    trie: dict = {}
    for key in keys:
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
            node = trie
            for char in key:
                node = node.setdefault(char, {})
            node[None] = {}

    def walk(node, first=False):
        chars = [c for c in node if c is not None]
        # A negated class would admit punctuation/Unicode outside the identifier
        # language. Subtract the trie edges from its explicit ASCII alphabet.
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_" + ("" if first else "0123456789")
        remaining = "".join(c for c in alphabet if c not in chars)
        parts = ["[" + remaining + "][A-Za-z_0-9]*"] if remaining else []
        if not first and None not in node:
            parts.append("")
        for char in chars:
            parts.append(re.escape(char) + walk(node[char]))
        return "(?:" + "|".join(parts) + ")"

    return walk(trie, True)


class QwenXMLSchema:
    """Keep the root/reference context for both grammar construction and decoding."""

    def __init__(self, schema: dict | bool):
        self.schema = schema
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        self.validator = cls(schema, registry=Registry())
        self.legacy_ref = cls in _LEGACY_REF
        self.branches = self._branches(schema)

    def _resolve(self, node, seen=frozenset()):
        if not isinstance(node, dict) or "$ref" not in node:
            return node
        ref = node["$ref"]
        if "$id" in node or "id" in node:
            _error("scoped references in XML values require a separate schema resource")
        if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")) or ref in seen:
            _error("nonlocal or recursive reference in an XML parameter list/value")
        target = resolve_local_pointer(self.schema, ref)
        if not isinstance(target, (dict, bool)):
            _error(f"unresolved reference {ref!r}")
        # The capability screen rejects nonredundant assertion siblings. Keep
        # annotations out of semantic normalization; never overwrite assertions.
        siblings = set(node) - {"$ref", "type"} - _ANNOTATIONS
        if siblings and not self.legacy_ref:
            _error("reference siblings require conjunction")
        return self._resolve(target, seen | {ref})

    def _branches(self, node, depth=0):
        if depth > 64:
            _error("too many nested XML alternatives")
        node = self._resolve(node)
        if not isinstance(node, dict):
            return [node]
        for key in ("anyOf", "oneOf"):
            if key in node:
                if set(node) - _ANNOTATIONS - {key, "type"}:
                    _error("alternative siblings require conjunction")
                branches = [leaf for arm in node[key] for leaf in self._branches(arm, depth + 1)]
                if len(branches) > 256:
                    _error("too many XML schema alternatives")
                return branches
        return [node]

    def _valid(self, schema, value):
        return self.validator.evolve(schema=schema).is_valid(value)

    def _context(self, node):
        """Rebase document-local pointers when a property becomes a format root."""
        node = dict(node)
        if isinstance(self.schema, dict) and "$schema" in self.schema:
            node.setdefault("$schema", self.schema["$schema"])
        if not any(
            isinstance((ref := child.get("$ref")), str) and (ref == "#" or ref.startswith("#/"))
            for child in iter_schema_nodes(node)
        ):
            return node

        def walk(value):
            if isinstance(value, list):
                return [walk(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = dict(value)
            for key, item in value.items():
                if key in {"$id", "id", "$anchor", "$dynamicAnchor"}:
                    _error("scoped references cannot be rebased as XML parameter fragments")
                if key == "$ref" and isinstance(item, str) and (item == "#" or item.startswith("#/")):
                    result[key] = "#/$defs/__qwen_document" + item[1:]
                elif key in _SUBSCHEMA_MAP_KEYS and isinstance(item, dict):
                    result[key] = {name: walk(child) for name, child in item.items()}
                elif key in _SUBSCHEMA_KEYS or key in _SUBSCHEMA_LIST_KEYS:
                    result[key] = walk(item)
            return result

        result = walk(node)
        result.setdefault("$defs", {})["__qwen_document"] = walk(self.schema)
        return result

    def value_format(self, node):
        formats = []
        for branch in self._branches(node):
            if branch is False:
                continue
            branch = {} if branch is True else branch
            if not (set(branch) - _ANNOTATIONS):
                # A schema-free XML value can always be a raw string, including
                # JSON-looking text. Do not invent a parameter constraint for
                # strict=false merely to express the native envelope.
                formats.append(AnyTextFormat(excludes=["</parameter>"]))
                continue
            # Finite mixed enums must retain their actual native type. A root
            # object in XML scalar mode would otherwise become a parameter list.
            if "const" in branch or "enum" in branch:
                values = [branch["const"]] if "const" in branch else branch["enum"]
                for value in values:
                    if self._valid(branch, value):
                        formats.append(
                            JSONSchemaFormat(
                                json_schema=self._context({"const": value}),
                                style="qwen_xml" if isinstance(value, str) else "json",
                            )
                        )
                continue
            declared = branch.get("type", list(_TYPES))
            declared = [declared] if isinstance(declared, str) else declared
            for string in (True, False):
                types = [t for t in declared if (t == "string") == string]
                if not types:
                    continue
                leaf = dict(branch, type=types[0] if len(types) == 1 else types)
                formats.append(
                    JSONSchemaFormat(json_schema=self._context(leaf), style="qwen_xml" if string else "json")
                )
        return _choice(formats)

    def parameter_format(self):
        return _choice([self._object_format(branch) for branch in self.branches if branch is not False])

    def _object_format(self, branch):
        branch = self._resolve(branch)
        branch = {} if branch is True else branch
        if "const" in branch or "enum" in branch:
            values = [branch["const"]] if "const" in branch else branch["enum"]
            return _choice(
                [
                    self._object_format(
                        {
                            "type": "object",
                            "properties": {k: {"const": v} for k, v in value.items()},
                            "required": list(value),
                            "additionalProperties": False,
                        }
                    )
                    for value in values
                    if isinstance(value, dict) and self._valid(branch, value)
                ]
            )
        if unsupported := set(branch) - _OBJECT_KEYS - _ANNOTATIONS:
            _error(f"root constraint(s) {sorted(unsupported)} cannot be composed faithfully")
        declared = branch.get("type", "object")
        if declared != "object" and declared != ["object"]:
            _error("tool parameters must be objects")
        properties = branch.get("properties", {})
        required = set(branch.get("required", []))
        additional = branch.get("additionalProperties", True)
        # A required but undeclared name is an additional property with a fixed
        # name. Materialize it so presence and uniqueness are enforced.
        properties = dict(properties)
        for name in sorted(required - properties.keys()):
            if additional is False:
                _error("required name is forbidden by additionalProperties=false")
            properties[name] = additional
        if len(properties) > 128:
            _error("more than 128 named XML parameters")
        tags = {}
        for name, prop in properties.items():
            if not name or any(c in name for c in "<>\r\n"):
                _error("parameter name contains XML delimiters")
            if prop is False:
                if name in required:
                    _error("required parameter has a false schema")
                continue
            tags[name] = _sequence(
                [
                    TagFormat(begin=f"<parameter={name}>\n", content=self.value_format(prop), end="\n</parameter>"),
                    ConstStringFormat(value="\n"),
                ]
            )
        minimum, maximum = branch.get("minProperties", 0), branch.get("maxProperties")
        if maximum is not None and maximum < max(minimum, len(required)):
            _error("impossible object cardinality")
        if additional is not False and (minimum > len(required) or maximum is not None):
            # Arbitrary repeated names cannot enforce distinct-key cardinality.
            _error("cardinality constraints with open XML properties")
        if minimum > len(required) or maximum is not None:
            optional = [name for name in tags if name not in required]
            variants = []
            upper = len(tags) if maximum is None else maximum
            for n in range(max(0, minimum - len(required)), min(len(optional), upper - len(required)) + 1):
                for subset in itertools.combinations(optional, n):
                    selected = required | set(subset)
                    variants.append(_sequence([tag for key, tag in tags.items() if key in selected]))
                    if len(variants) > 1024:
                        _error("object cardinality requires too many XML alternatives")
            body = _choice(variants)
        else:
            body = _sequence([tag if name in required else OptionalFormat(content=tag) for name, tag in tags.items()])
        if additional is not False:
            dynamic = _sequence(
                [
                    ConstStringFormat(value="<parameter="),
                    RegexFormat(pattern=_other_key_pattern(list(properties))),
                    ConstStringFormat(value=">\n"),
                    self.value_format(additional),
                    ConstStringFormat(value="\n</parameter>\n"),
                ]
            )
            body = _sequence([body, StarFormat(content=dynamic)])
        return body

    def decode(self, values: dict):
        """Choose native types using complete branches, preserving valid strings.

        Nested values are already JSON: never recursively guess their strings.
        Branch selection runs against the whole argument object, so later
        discriminators can decide the type of an earlier ambiguous XML slot.
        """
        fallback = values
        for branch in self.branches:
            if branch is False:
                continue
            branch = {} if branch is True else branch
            props = branch.get("properties", {})
            candidate = {}
            for name, raw in values.items():
                prop = props.get(name, branch.get("additionalProperties", True))
                if not isinstance(raw, str) or self._valid(prop, raw):
                    candidate[name] = raw
                    continue
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = raw
                candidate[name] = parsed if self._valid(prop, parsed) else raw
            # Finite whole-object enums need their own property schemas.
            if "const" in branch or "enum" in branch:
                finite = [branch["const"]] if "const" in branch else branch["enum"]
                for value in finite:
                    if (
                        isinstance(value, dict)
                        and set(value) == set(values)
                        and all(
                            values[k]
                            == (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                            or (not isinstance(v, str) and _json_equal(values[k], v))
                            for k, v in value.items()
                        )
                        and self._valid(self.schema, value)
                    ):
                        return value
            fallback = candidate
            if self._valid(self.schema, candidate):
                return candidate
        return fallback

    @cached_property
    def needs_complete_object(self):
        return len(self.branches) != 1 or any(
            isinstance(branch, dict) and ("const" in branch or "enum" in branch) for branch in self.branches
        )

    @cached_property
    def string_keys(self):
        if len(self.branches) != 1:
            return set()
        branch = self.branches[0]
        if not isinstance(branch, dict):
            return set()
        return {name for name, prop in branch.get("properties", {}).items() if self._string_only(prop)}

    def _string_only(self, node):
        for branch in self._branches(node):
            if not isinstance(branch, dict):
                return False
            values = [branch["const"]] if "const" in branch else branch.get("enum")
            if values is not None:
                if not values or not all(isinstance(value, str) for value in values):
                    return False
            elif branch.get("type") not in ("string", ["string"]):
                return False
        return True


def _json_equal(raw, value):
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, sort_keys=True) == json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return False


def replace_qwen_parameter_formats(tag):
    """Retain stock call/choice/reasoning envelopes, replacing parameter lists."""

    def visit(node):
        if isinstance(node, TriggeredTagsFormat) and "<tool_call>\n<function=" in node.triggers:
            # Commit to the native call on its opener, rather than allowing a
            # malformed call prefix or unmatched closers as ordinary prose.
            node.triggers = ["<tool_call>"]
            node.excludes = list(
                dict.fromkeys(
                    [
                        *(node.excludes or []),
                        "</tool_call>",
                        "<function=",
                        "</function>",
                        "<parameter=",
                        "</parameter>",
                    ]
                )
            )
        if isinstance(node, JSONSchemaFormat) and node.style == "qwen_xml":
            schema = node.json_schema
            screen = SimpleNamespace(structural_tag={"format": {"type": "json_schema", "json_schema": schema}})
            if reason := get_structured_outputs_schema_error(screen):
                _error(reason)
            if reason := get_xgrammar_schema_error(schema):
                _error(reason)
            return QwenXMLSchema(schema).parameter_format()
        if isinstance(node, BaseModel):
            if (
                isinstance(node, TagFormat)
                and isinstance(node.content, JSONSchemaFormat)
                and node.content.style == "qwen_xml"
            ):
                # The composed body terminates parameters with a newline;
                # the stock envelope puts the same separator in its end.
                node.end = node.end.removeprefix("\n")
            for name in type(node).model_fields:
                value = getattr(node, name)
                if isinstance(value, BaseModel):
                    setattr(node, name, visit(value))
                elif isinstance(value, list):
                    setattr(node, name, [visit(item) for item in value])
        return node

    visit(tag)
