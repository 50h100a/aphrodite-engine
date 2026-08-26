# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enforce, while decoding, the JSON Schema keywords no grammar can express.

A context-free grammar cannot count how many items matched a subschema, nor
compare an item against the ones before it, so no backend enforces
`contains`/`minContains` or `uniqueItems`. `StructuredOutputGrammar` is a
per-request stateful object, so enforcement lands here instead: a decorator
around whatever grammar the backend compiled, scanning the bytes the model has
emitted and ANDing extra vetoes into the bitmask.

The two keywords are held from opposite ends. `contains` is a floor, so it is
kept by refusing to close the array, and the decision can wait until the array
ends: each completed item is handed to `jsonschema`, and the matched subschema
may be arbitrarily structural. `uniqueItems` is a prohibition on an item that
has not been emitted yet, and by the time the repeat is complete the model is
already committed to it -- so it is kept instead by keeping every item on a
path to a value the array has not used, which needs the possible values in
advance. That is what confines it to a finite domain of scalars.

Two invariants tie this to the screen in `schema_features`:

- **Reachability.** The screen refuses any schema whose obligations sit
  somewhere this scanner cannot navigate to, using `analyze` below -- the same
  walk, so the two cannot disagree about what is enforceable.
- **No walls.** Every veto leaves at least one legal continuation, given the
  static rules the screen applies: `minContains` alongside `maxItems` is
  refused because masking `]` in a bounded array can wall, and a `uniqueItems`
  array is refused unless its domain is large enough to reach `minItems` and
  whatever `contains` asks for beside it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jsonschema.validators
import regex as re

from aphrodite.logger import init_logger
from aphrodite.utils.import_utils import LazyLoader
from aphrodite.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from aphrodite.tokenizers import TokenizerLike
else:
    np = LazyLoader("np", globals(), "numpy")
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)

LAYER_ENFORCED_KEYWORDS = frozenset({"contains", "minContains", "uniqueItems"})

# Ceiling on the DNF expansion of one schema position. `anyOf` sites multiply.
_MAX_ALTERNATIVES = 64

# Ceiling on the item domain a `uniqueItems` array may draw from. The cost of a
# larger one is the trie, not the count, so this is a bound on how much of a
# schema is worth reading rather than on how much the veto can afford.
_MAX_DOMAIN = 128


class _Domain:
    """The values an array's items may take, as a trie over their spellings.

    A repeat has to be refused before the item that would be one is finished,
    so what the veto reads is not the values but their prefixes: it keeps the
    item on a path towards one the array has not used, and the byte that leaves
    every such path is the byte that goes.
    """

    __slots__ = ("values", "index", "children", "terminal", "reachable")

    def __init__(self, values: tuple[bytes, ...]):
        self.values = values
        self.index = {spelling: position for position, spelling in enumerate(values)}
        self.children: list[dict[int, int]] = [{}]
        self.terminal: list[int | None] = [None]
        reachable: list[set[int]] = [set()]
        for position, spelling in enumerate(values):
            node = 0
            reachable[0].add(position)
            for byte in spelling:
                child = self.children[node].get(byte)
                if child is None:
                    child = len(self.children)
                    self.children[node][byte] = child
                    self.children.append({})
                    self.terminal.append(None)
                    reachable.append(set())
                node = child
                reachable[node].add(position)
            self.terminal[node] = position
        # Which values are still spellable from each node, so a subtree the
        # array has used up can be recognised without walking it.
        self.reachable: list[frozenset[int]] = [frozenset(found) for found in reachable]

    def locate(self, prefix: bytes) -> int | None:
        """The node `prefix` reaches, or None if it left the domain."""
        node = 0
        for byte in prefix:
            child = self.children[node].get(byte)
            if child is None:
                return None
            node = child
        return node


# ---------------------------------------------------------------------------
# Static analysis: what this layer can enforce, and where it cannot reach.
# ---------------------------------------------------------------------------


@dataclass
class SchemaAnalysis:
    """What `analyze` found: the obligations, and every reason one was refused.

    A refused obligation is simply absent, which is what the screen reads.
    `problems` says why, and decides nothing.
    """

    problems: list[str] = field(default_factory=list)
    # Nodes carrying a live `contains` obligation, by identity.
    obligations: set[int] = field(default_factory=set)
    # Nodes carrying an enforceable `uniqueItems`, with the values their items
    # may take. The two are kept apart because one node can carry both keywords
    # and have only one of them served.
    domains: dict[int, _Domain] = field(default_factory=dict)
    # Nodes from which an obligation is reachable. Navigation stops where this
    # does, so an ordinary subtree costs nothing to walk past.
    relevant: set[int] = field(default_factory=set)

    def enforces(self, node: dict[str, Any], key: str) -> bool:
        """Whether this layer keeps `key` where it sits in `node`."""
        if key == "uniqueItems":
            return id(node) in self.domains
        return key in ("contains", "minContains") and id(node) in self.obligations

    def __bool__(self) -> bool:
        return bool(self.obligations or self.domains)


class _Unresolvable(Exception):
    """A `$ref` this layer cannot follow, so it cannot see what is under it."""


class _TooManyAlternatives(Exception):
    """The DNF expansion of a schema position exceeded `_MAX_ALTERNATIVES`."""


# Keys whose subschemas describe a *child* of the current document position, and
# so the only ones the scanner walks into. The reachability check is exactly
# "was the obligation found through one of these?".
_NAV_SINGLE = ("items", "additionalItems", "additionalProperties")
_NAV_LIST = ("prefixItems", "allOf", "anyOf", "oneOf")
_NAV_MAP = ("properties", "patternProperties")

# Live subschemas the scanner cannot navigate to. An obligation under one of
# these is reported rather than quietly missed.
_UNREACHABLE_KEYS = ("unevaluatedItems", "unevaluatedProperties")


def _resolve_pointer(root: Any, ref: str) -> Any:
    """Follow a local JSON pointer, or raise. External and `$anchor` references
    are unfollowable, and this layer cannot enforce what it cannot see into."""
    if not isinstance(ref, str) or not ref.startswith("#"):
        raise _Unresolvable(ref)
    fragment = ref[1:]
    if fragment == "":
        return root
    if not fragment.startswith("/"):
        raise _Unresolvable(ref)
    node = root
    for raw in fragment[1:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if part not in node:
                raise _Unresolvable(ref)
            node = node[part]
        elif isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise _Unresolvable(ref) from None
        else:
            raise _Unresolvable(ref)
    return node


# An alternative is a conjunction of schema nodes that all apply at one document
# position; a cursor set is a list of alternatives, i.e. disjunctive normal form.
# Built lazily as the document is walked rather than hoisted at the root, so
# recursion through `$ref` only expands as deep as the document goes.
_Alternative = tuple[dict[str, Any], ...]


def _product(left: list[_Alternative], right: list[_Alternative]) -> list[_Alternative]:
    """Conjoin two cursor sets. An empty side means "matches nothing", and
    annihilates: `items: false` kills the alternative it is under."""
    if not left or not right:
        return []
    out = [a + b for a in left for b in right]
    if len(out) > _MAX_ALTERNATIVES:
        raise _TooManyAlternatives(len(out))
    return out


def _expand(node: Any, root: Any, path: tuple[int, ...] = ()) -> list[_Alternative]:
    """The cursor set for one schema position, in DNF.

    `allOf` and `$ref` are conjunction: several nodes applying at once, not
    ambiguity. `anyOf`/`oneOf` become separate alternatives, and the veto
    downstream fires only when all of them agree -- a document still live in
    some branch can still come out valid under it.

    `$ref` siblings are conjoined rather than ignored. Draft-7 ignores them and
    2019-09 onwards applies them; conjoining only over-constrains a draft-7
    schema, whereas ignoring would under-enforce a modern one.
    """
    if node is False:
        return []
    if not isinstance(node, dict):
        # `true`, `{}`, or something that is not a schema: constrains nothing.
        return [()]

    alternatives: list[_Alternative] = [(node,)]

    for sub in node.get("allOf") or ():
        alternatives = _product(alternatives, _expand(sub, root, path))

    if "$ref" in node:
        if id(node) in path:
            # A `$ref` cycle that never passes through the document. Nothing
            # can satisfy it; refusing beats spinning.
            raise _Unresolvable(node["$ref"])
        target = _resolve_pointer(root, node["$ref"])
        alternatives = _product(alternatives, _expand(target, root, path + (id(node),)))

    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if not isinstance(branches, list) or not branches:
            continue
        # `oneOf` is treated as `anyOf`: exclusivity is the backend's business,
        # and a wider cursor set only makes this layer more permissive.
        union: list[_Alternative] = []
        for branch in branches:
            union.extend(_expand(branch, root, path))
        alternatives = _product(alternatives, union)

    if len(alternatives) > _MAX_ALTERNATIVES:
        raise _TooManyAlternatives(len(alternatives))
    return alternatives


def _item_schemas(node: dict[str, Any], index: int) -> Iterator[Any]:
    """The subschemas describing item `index` of an array matching `node`."""
    prefix = node.get("prefixItems")
    items = node.get("items")
    if isinstance(prefix, list):
        if index < len(prefix):
            yield prefix[index]
            return
        if items is not None:
            yield items
        return
    if isinstance(items, list):
        # Draft-7 tuple form: `items` is the prefix, `additionalItems` the tail.
        if index < len(items):
            yield items[index]
            return
        if (extra := node.get("additionalItems")) is not None:
            yield extra
        return
    if items is not None:
        yield items


def _property_schemas(node: dict[str, Any], key: str) -> Iterator[Any]:
    """The subschemas describing property `key` of an object matching `node`."""
    matched = False
    props = node.get("properties")
    if isinstance(props, dict) and key in props:
        yield props[key]
        matched = True
    patterns = node.get("patternProperties")
    if isinstance(patterns, dict):
        for pattern, sub in patterns.items():
            try:
                if re.search(pattern, key):
                    yield sub
                    matched = True
            except re.error:
                continue
    if not matched and (extra := node.get("additionalProperties")) is not None:
        yield extra


def contains_obligation(node: dict[str, Any]) -> int | None:
    """How many items must match `node["contains"]`, or None if it asks nothing.

    `minContains: 0` asks nothing: "at least zero items match" holds of every
    array including the empty one, so it makes the keyword vacuous rather than
    loosening it. Nor does a node typed away from arrays.

    The screen reads this too, so a keyword this layer would never fire on is
    not one the caller gets refused for.
    """
    if "contains" not in node:
        return None
    minimum = node.get("minContains", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        return None
    declared = node.get("type")
    if isinstance(declared, str) and declared != "array":
        return None
    if isinstance(declared, list) and "array" not in declared:
        return None
    return minimum


def unique_obligation(node: dict[str, Any]) -> bool:
    """Whether `node`'s `uniqueItems` asks for anything.

    `uniqueItems: false` is the keyword at its permissive setting, and a node
    typed away from arrays never has items to compare. Read by the screen too,
    so a keyword this layer would never fire on is not one the caller is
    refused for.
    """
    if node.get("uniqueItems") is not True:
        return False
    declared = node.get("type")
    if isinstance(declared, str):
        return declared == "array"
    if isinstance(declared, list):
        return "array" in declared
    return True


def _canonical(value: Any) -> bytes | None:
    """The one spelling of `value` this layer will accept, or None for a value
    it will not take on.

    Items are compared as bytes, so a value the backends do not all spell the
    same way would have to be normalised as it is read. A container has key
    order and whitespace to normalise; a non-ASCII string is `\\uXXXX` to one
    backend and raw UTF-8 to another. Refusing beats vetoing the spelling the
    backend actually chose, which would wall the array.
    """
    if not (value is None or isinstance(value, (bool, int, float, str))):
        return None
    plain = json.dumps(value)
    if plain != json.dumps(value, ensure_ascii=False):
        return None
    return plain.encode()


def _spellings(values: Any) -> set[bytes] | None:
    """`values` as canonical bytes, or None if any of them has no single spelling."""
    if not isinstance(values, list):
        return None
    out: set[bytes] = set()
    for value in values:
        spelling = _canonical(value)
        if spelling is None:
            return None
        out.add(spelling)
    return out


def _value_domain(node: Any, root: Any, path: tuple[int, ...] = ()) -> set[bytes] | None:
    """Every value `node` can take, or None if it does not bound them.

    A superset is enough here -- the caller narrows it back down by validating
    each candidate against the whole subschema -- so only the keywords that
    *pin* a value are read: `const`, `enum`, and a `type` of nothing but
    booleans and nulls. `allOf` and `$ref` intersect, `anyOf`/`oneOf` unions,
    and a branch that bounds nothing leaves the whole position unbounded.
    """
    if node is False:
        return set()
    if not isinstance(node, dict):
        return None

    bounds: list[set[bytes]] = []

    listed: set[bytes] | None = None
    if "const" in node:
        if (listed := _spellings([node["const"]])) is None:
            return None
    if "enum" in node:
        if (enumerated := _spellings(node["enum"])) is None:
            return None
        listed = enumerated if listed is None else listed & enumerated
    if listed is not None:
        bounds.append(listed)

    declared = node.get("type")
    kinds = [declared] if isinstance(declared, str) else declared if isinstance(declared, list) else None
    if kinds is not None and set(kinds) <= {"boolean", "null"}:
        finite: set[bytes] = set()
        if "boolean" in kinds:
            finite |= {b"true", b"false"}
        if "null" in kinds:
            finite.add(b"null")
        bounds.append(finite)

    for sub in node.get("allOf") or ():
        if (nested := _value_domain(sub, root, path)) is not None:
            bounds.append(nested)

    if "$ref" in node:
        if id(node) in path:
            raise _Unresolvable(node["$ref"])
        target = _resolve_pointer(root, node["$ref"])
        if (nested := _value_domain(target, root, path + (id(node),))) is not None:
            bounds.append(nested)

    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if not isinstance(branches, list) or not branches:
            continue
        union: set[bytes] | None = set()
        for branch in branches:
            nested = _value_domain(branch, root, path)
            if nested is None:
                union = None
                break
            union |= nested
        if union is not None:
            bounds.append(union)

    if not bounds:
        return None
    narrowed = set(bounds[0])
    for extra in bounds[1:]:
        narrowed &= extra
    return narrowed


def _walk(root: Any, navigable_only: bool) -> Iterator[dict[str, Any]]:
    """Every node that can apply to some instance, `$ref` followed.

    With `navigable_only`, restricted to the positions the scanner can walk
    into. The difference between the two walks is the reachability check: a node
    found by one and not the other carries an obligation nothing would enforce.

    Both are narrower than `schema_features.iter_schema_nodes`, which also
    descends into `if`/`not` branches and unreferenced `$defs`. Nothing is
    checked against those; referenced `$defs` are reached through the `$ref`.
    """
    singles = _NAV_SINGLE if navigable_only else (*_NAV_SINGLE, *_UNREACHABLE_KEYS)
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        yield node

        if "$ref" in node:
            try:
                stack.append(_resolve_pointer(root, node["$ref"]))
            except _Unresolvable:
                pass
        for key in singles:
            value = node.get(key)
            if isinstance(value, list):
                # `items` in its draft-7 tuple spelling.
                stack.extend(value)
            elif value is not None:
                stack.append(value)
        for key in _NAV_LIST:
            value = node.get(key)
            if isinstance(value, list):
                stack.extend(value)
        for key in _NAV_MAP:
            value = node.get(key)
            if isinstance(value, dict):
                stack.extend(value.values())


# Keys that give some positions of an array a schema of their own, so the items
# no longer share one domain.
_POSITIONAL_KEYS = frozenset({"prefixItems", "additionalItems", "unevaluatedItems"})


def _domain_for(node: dict[str, Any], root: Any, validator: Any, analysis: SchemaAnalysis) -> _Domain | None:
    """The values `node`'s items may take, or None with the reason recorded.

    `_value_domain` only reads the keywords that pin a value, so what comes back
    is a superset; it is narrowed to the exact domain by validating each
    candidate against `items` whole, which is what makes the counts in
    `_refuse_walls` trustworthy.
    """
    items = node.get("items")
    try:
        candidates = _value_domain(items, root)
    except _Unresolvable:
        candidates = None
    if candidates is None or len(candidates) > _MAX_DOMAIN:
        analysis.problems.append(
            "uniqueItems is enforced by keeping each item on a path to a value the "
            "array has not used yet, which needs those values in advance: give "
            f"`items` an `enum` or a `const` of at most {_MAX_DOMAIN} ASCII scalars, "
            "or drop the keyword and check the generated output instead"
        )
        return None

    matcher = validator.evolve(schema=items)
    return _Domain(tuple(sorted(value for value in candidates if matcher.is_valid(json.loads(value)))))


def analyze(schema: Any) -> SchemaAnalysis:
    """What this layer can enforce in `schema`, and every reason it cannot.

    Read by both the screen and the runtime, so the two cannot disagree about
    which obligations are covered.
    """
    analysis = SchemaAnalysis()
    if not isinstance(schema, dict):
        return analysis

    counted: list[dict[str, Any]] = []
    distinct: list[dict[str, Any]] = []
    for node in _walk(schema, navigable_only=False):
        if contains_obligation(node) is not None:
            counted.append(node)
        if unique_obligation(node):
            distinct.append(node)
    if not counted and not distinct:
        return analysis

    live = {id(node): node for node in _walk(schema, navigable_only=True)}

    # Static refusals, in schema order so the message names the first problem a
    # caller would look for.
    for node in counted:
        if "maxContains" in node:
            analysis.problems.append(
                "maxContains cannot be enforced while decoding: refusing an item "
                "requires recognising one that does not match `contains`, which is "
                "only decidable once the item is already emitted"
            )
            continue

        if "maxItems" in node:
            analysis.problems.append(
                "minContains cannot be enforced alongside maxItems: holding the "
                "array open until enough items match can run past the item limit, "
                "leaving the request with no legal token to emit"
            )
            continue

        if id(node) not in live:
            analysis.problems.append(
                "this `contains` sits where the decoder cannot follow it "
                f"(reachable only through {' or '.join(_UNREACHABLE_KEYS)})"
            )
            continue

        analysis.obligations.add(id(node))

    validator: Any = None
    for node in distinct:
        if isinstance(node.get("items"), list) or not _POSITIONAL_KEYS.isdisjoint(node):
            analysis.problems.append(
                "uniqueItems cannot be enforced on an array whose positions have "
                "their own schemas: there is no one set of values for the items to "
                "be distinct within"
            )
            continue

        if validator is None:
            validator = jsonschema.validators.validator_for(schema)(schema)
        domain = _domain_for(node, schema, validator, analysis)
        if domain is None:
            continue

        if id(node) not in live:
            analysis.problems.append(
                "this `uniqueItems` sits where the decoder cannot follow it "
                f"(reachable only through {' or '.join(_UNREACHABLE_KEYS)})"
            )
            continue

        analysis.domains[id(node)] = domain

    if not analysis:
        return analysis

    # The cursor set has to be buildable everywhere on the way down, or the
    # runtime would navigate past an obligation without seeing it. What it
    # builds is kept: which keywords land on one array together is the question
    # the wall check below asks, and this is the walk that answers it.
    conjunctions: list[_Alternative] = []
    try:
        conjunctions += _expand(schema, schema)
        for node in live.values():
            for key in _NAV_SINGLE:
                if key in node:
                    conjunctions += _expand(node[key], schema)
            for key in _NAV_LIST:
                value = node.get(key)
                if isinstance(value, list):
                    for sub in value:
                        conjunctions += _expand(sub, schema)
            for key in _NAV_MAP:
                value = node.get(key)
                if isinstance(value, dict):
                    for sub in value.values():
                        conjunctions += _expand(sub, schema)
    except _Unresolvable as err:
        return _refuse_everything(
            analysis,
            f"the schema contains a reference this decoder cannot follow ({err.args[0]!r}), "
            "so it cannot tell where the `contains` applies",
        )
    except _TooManyAlternatives as err:
        return _refuse_everything(
            analysis,
            f"the schema branches into more than {_MAX_ALTERNATIVES} alternatives "
            f"({err.args[0]}) at one position, too many to track while decoding",
        )

    if analysis.domains:
        _refuse_walls(analysis, conjunctions, validator)
    _mark_relevant(schema, analysis)
    return analysis


def _refuse_walls(analysis: SchemaAnalysis, conjunctions: list[_Alternative], validator: Any) -> None:
    """Drop any `uniqueItems` whose array could be left with nothing legal to say.

    This is where the two vetoes meet. One holds an array open until enough
    items match `contains`; the other closes it when the domain runs out of
    values. If the values run out first there is no token either of them
    allows, so the pair is refused rather than decoded into a wall -- and so is
    a `minItems` no domain is long enough to reach.

    Asked per conjunction rather than per node because the keywords need not sit
    on the same one: an `allOf` can put `uniqueItems` in one branch and
    `contains` in another, and they still land on the same array.
    """
    doomed: set[int] = set()
    for conjunction in conjunctions:
        domains = [analysis.domains[id(node)] for node in conjunction if id(node) in analysis.domains]
        if not domains:
            continue
        values = set(domains[0].values)
        for extra in domains[1:]:
            values &= set(extra.values)

        floors = [
            node["minItems"]
            for node in conjunction
            if isinstance(node.get("minItems"), int) and not isinstance(node["minItems"], bool)
        ]
        floor = max(floors, default=0)
        problem: str | None = None
        if floor > len(values):
            problem = (
                f"minItems asks for {floor} items, all distinct under uniqueItems, but "
                f"`items` allows only {len(values)} values, so the array runs out before "
                "it is long enough"
            )
        for node in conjunction:
            if problem is not None:
                break
            minimum = contains_obligation(node)
            if minimum is None or id(node) not in analysis.obligations:
                continue
            matcher = validator.evolve(schema=node["contains"])
            matching = sum(1 for value in values if matcher.is_valid(json.loads(value)))
            if minimum > matching:
                problem = (
                    f"minContains asks for {minimum} matching items, all distinct under "
                    f"uniqueItems, but only {matching} of the values `items` allows match "
                    "`contains` at all"
                )
        if problem is not None:
            analysis.problems.append(problem)
            doomed |= {id(node) for node in conjunction}

    for node_id in doomed:
        analysis.domains.pop(node_id, None)


def _refuse_everything(analysis: SchemaAnalysis, reason: str) -> SchemaAnalysis:
    """Give up on the whole schema, not just the obligation that tripped over it.

    Best-effort mode strips exactly what was refused, so a schema-wide reason
    must take every obligation with it or the strip leaves the screen still
    failing on the next pass.
    """
    analysis.problems.append(reason)
    analysis.obligations.clear()
    analysis.domains.clear()
    return analysis


def _mark_relevant(root: Any, analysis: SchemaAnalysis) -> None:
    """Record every node an obligation is reachable from. Navigation stops where
    this set does, so unrelated subtrees cost nothing per token."""
    parents: dict[int, list[int]] = {}
    nodes: dict[int, Any] = {}

    def link(parent: Any, child: Any) -> None:
        if isinstance(child, dict):
            parents.setdefault(id(child), []).append(id(parent))

    for node in _walk(root, navigable_only=True):
        nodes[id(node)] = node
        if "$ref" in node:
            try:
                link(node, _resolve_pointer(root, node["$ref"]))
            except _Unresolvable:
                pass
        for key in _NAV_SINGLE:
            if key in node:
                link(node, node[key])
        for key in _NAV_LIST:
            value = node.get(key)
            if isinstance(value, list):
                for sub in value:
                    link(node, sub)
            elif isinstance(value, dict):
                link(node, value)
        for key in _NAV_MAP:
            value = node.get(key)
            if isinstance(value, dict):
                for sub in value.values():
                    link(node, sub)

    pending = list(analysis.obligations | analysis.domains.keys())
    while pending:
        current = pending.pop()
        if current in analysis.relevant:
            continue
        analysis.relevant.add(current)
        pending.extend(parents.get(current, ()))


# ---------------------------------------------------------------------------
# The byte-level lexer. One transition function, shared by the document scanner
# and by the lookahead that decides which tokens to veto -- a second, subtly
# different lexer is how the veto and the state it protects drift apart.
# ---------------------------------------------------------------------------

_S_VALUE = 0  # expecting a value to start, or the bracket that closes an empty one
_S_ATOM = 1  # inside a number, `true`, `false` or `null`
_S_STR = 2  # inside a string value
_S_STR_ESC = 3
_S_AFTER = 4  # a value just finished: expecting a comma or a close
_S_KEY = 5  # expecting an object key, or the brace that closes an empty object
_S_KEY_STR = 6
_S_KEY_ESC = 7
_S_COLON = 8

_E_VALUE_BEGIN = 0  # a value starts at this byte
_E_VALUE_END_EXCL = 1  # the value in progress ended just *before* this byte
_E_VALUE_END_INCL = 2  # the value in progress ended *with* this byte
_E_PUSH_ARRAY = 3
_E_PUSH_OBJECT = 4
_E_POP = 5  # a container closed with this byte
_E_COMMA = 6
_E_KEY_BEGIN = 7
_E_KEY_END = 8

_WS = frozenset(b" \t\n\r")
# Bytes that keep a number or a literal going. Everything else ends it, and the
# byte that ends it belongs to whatever comes next.
_ATOM = frozenset(b"0123456789+-.eE") | frozenset(b"abcdefghijklmnopqrstuvwxyz")

_QUOTE = 0x22
_COMMA = 0x2C
_COLON = 0x3A
_LBRACKET = 0x5B
_RBRACKET = 0x5D
_BACKSLASH = 0x5C
_LBRACE = 0x7B
_RBRACE = 0x7D

_NO_EVENTS: tuple[int, ...] = ()

# What a closing token finishes on its way out: for each item, whether it
# continues one already under way in the document, and the bytes it contributes.
_CloseKey = tuple[tuple[bool, bytes], ...]


def _lex_step(state: int, byte: int, in_object: bool) -> tuple[int, tuple[int, ...]]:
    """Advance one byte of JSON. Returns the new state and what happened.

    `in_object` is the kind of the innermost open container, which the byte
    alone does not settle: a comma is followed by a key inside an object and a
    value inside an array.

    Forgiving by design -- the backend grammar has already decided what is
    well-formed. This locates array item boundaries, it does not re-validate.
    """
    if state == _S_STR:
        if byte == _BACKSLASH:
            return _S_STR_ESC, _NO_EVENTS
        if byte == _QUOTE:
            return _S_AFTER, (_E_VALUE_END_INCL,)
        return _S_STR, _NO_EVENTS
    if state == _S_STR_ESC:
        return _S_STR, _NO_EVENTS
    if state == _S_KEY_STR:
        if byte == _BACKSLASH:
            return _S_KEY_ESC, _NO_EVENTS
        if byte == _QUOTE:
            return _S_COLON, (_E_KEY_END,)
        return _S_KEY_STR, _NO_EVENTS
    if state == _S_KEY_ESC:
        return _S_KEY_STR, _NO_EVENTS

    if state == _S_ATOM:
        if byte in _ATOM:
            return _S_ATOM, _NO_EVENTS
        # The atom ended before this byte; re-read the byte as if we had just
        # finished a value.
        following = _lex_step(_S_AFTER, byte, in_object)
        return following[0], (_E_VALUE_END_EXCL, *following[1])

    if state == _S_AFTER:
        if byte in _WS:
            return _S_AFTER, _NO_EVENTS
        if byte == _COMMA:
            return (_S_KEY if in_object else _S_VALUE), (_E_COMMA,)
        if byte in (_RBRACKET, _RBRACE):
            # The container that just closed is itself a value of its parent.
            return _S_AFTER, (_E_POP, _E_VALUE_END_INCL)
        return _S_AFTER, _NO_EVENTS

    if state == _S_VALUE:
        if byte in _WS:
            return _S_VALUE, _NO_EVENTS
        if byte in (_RBRACKET, _RBRACE):
            # An empty container: no value began, so there is none to end.
            return _S_AFTER, (_E_POP, _E_VALUE_END_INCL)
        if byte == _LBRACKET:
            return _S_VALUE, (_E_VALUE_BEGIN, _E_PUSH_ARRAY)
        if byte == _LBRACE:
            return _S_KEY, (_E_VALUE_BEGIN, _E_PUSH_OBJECT)
        if byte == _QUOTE:
            return _S_STR, (_E_VALUE_BEGIN,)
        return _S_ATOM, (_E_VALUE_BEGIN,)

    if state == _S_KEY:
        if byte in _WS:
            return _S_KEY, _NO_EVENTS
        if byte == _RBRACE:
            return _S_AFTER, (_E_POP, _E_VALUE_END_INCL)
        if byte == _QUOTE:
            return _S_KEY_STR, (_E_KEY_BEGIN,)
        return _S_KEY, _NO_EVENTS

    # _S_COLON
    if byte == _COLON:
        return _S_VALUE, _NO_EVENTS
    return _S_COLON, _NO_EVENTS


def _probe(state: int, kinds: tuple[bool, ...], data: bytes) -> tuple[tuple[int | None, int], ...] | None:
    """Would `data` close the array that `kinds[0]` describes?

    `kinds` holds the container kind for each container from that array inwards,
    so `len(kinds) - 1` is how many are open inside it.

    Returns None if `data` would not close it. Otherwise the spans of that
    array's own items finishing inside `data` before it closes; a start of None
    means the item was already under way, so its opening bytes are in the
    document rather than the token.

    Pure in `(state, kinds, data)`, so the answer is computed once per
    vocabulary and reused for the life of the process.
    """
    open_kinds = list(kinds)
    start: int | None = None
    items: list[tuple[int | None, int]] = []
    for index, byte in enumerate(data):
        state, events = _lex_step(state, byte, open_kinds[-1])
        for event in events:
            if event == _E_POP:
                if len(open_kinds) == 1:
                    return tuple(items)
                open_kinds.pop()
            elif event == _E_PUSH_ARRAY:
                open_kinds.append(False)
            elif event == _E_PUSH_OBJECT:
                open_kinds.append(True)
            elif len(open_kinds) != 1:
                continue
            elif event == _E_VALUE_BEGIN:
                start = index
            elif event == _E_VALUE_END_EXCL:
                items.append((start, index))
                start = None
            elif event == _E_VALUE_END_INCL:
                items.append((start, index + 1))
                start = None
    return None


# Where a token leaves an array whose items come from a `_Domain`. Coarser than
# the lexer above, and it can be: the items are scalars, so nothing nests inside
# one, and the trie already knows which bytes are inside a string and which have
# ended it.
_P_BETWEEN = 0  # a comma has committed to another item, or the array just opened
_P_INSIDE = 1  # part of an item has been spelled
_P_AFTER = 2  # an item finished: expecting a comma or the close
_P_CLOSED = 3  # the array ended; the rest of the token is not this frame's business

# Where a token leaves the array, and which values it finished on the way.
_Landing = tuple[int, int, tuple[int, ...]]


def _walk_unique(domain: _Domain, position: int, node: int, data: bytes) -> _Landing | None:
    """Where `data` leaves an array sitting at `(position, node)`.

    None if it steps off the domain. That is a veto rather than a shrug: every
    legal item spells one of a known set of values, so a byte that leaves them
    all is one the backend's own `enum` would have refused too, and treating it
    as unreachable is what keeps the trie an exact account of where the array
    can still go.

    Independent of which values the array has already used, so a position is
    walked once per vocabulary and the used set only filters what comes back.
    """
    finished: list[int] = []
    for byte in data:
        if position == _P_CLOSED:
            break
        if position == _P_INSIDE:
            child = domain.children[node].get(byte)
            if child is not None:
                # Checked before the separators, so a comma or a bracket inside
                # a string value stays inside it.
                node = child
                continue
            value = domain.terminal[node]
            if value is None:
                return None
            finished.append(value)
            position, node = _P_AFTER, 0
        if byte in _WS:
            continue
        if byte == _RBRACKET:
            position = _P_CLOSED
        elif position == _P_AFTER:
            if byte != _COMMA:
                return None
            position = _P_BETWEEN
        else:
            child = domain.children[0].get(byte)
            if child is None:
                return None
            position, node = _P_INSIDE, child
    return position, node, tuple(finished)


def _permits(domain: _Domain, landing: _Landing, used: set[int]) -> bool:
    """Whether `landing` keeps every item distinct and leaves somewhere to go."""
    position, node, finished = landing
    seen = used
    for value in finished:
        if value in seen:
            return False
        seen = seen | {value}
    if position == _P_INSIDE:
        # Mid-item, so the array is committed to finishing this one.
        return bool(domain.reachable[node] - seen)
    if position == _P_BETWEEN:
        # A comma committed to another item; without a value left for it the
        # next step would have nothing legal at all.
        return len(seen) < len(domain.values)
    return True


# ---------------------------------------------------------------------------
# Per-vocabulary lookahead tables.
# ---------------------------------------------------------------------------


class _VocabProfile:
    """Everything about a tokenizer this layer needs, derived once.

    Both lookaheads are affordable because both start from a handful of bytes
    rather than from the vocabulary: only a token spelling a `]` can close an
    array, and only a token starting on the domain's trie can continue an item.
    A few hundred candidates out of a hundred thousand, and what each one does
    depends on the position alone, so it is answered once and reused.
    """

    def __init__(self, token_bytes: list[bytes]):
        self.token_bytes = token_bytes
        closers: list[int] = []
        nothing: list[int] = []
        buckets: list[list[int]] = [[] for _ in range(256)]
        singles: set[int] = set()
        for token_id, text in enumerate(token_bytes):
            if not text:
                nothing.append(token_id)
                continue
            buckets[text[0]].append(token_id)
            if len(text) == 1:
                singles.add(text[0])
            if _RBRACKET in text:
                closers.append(token_id)
        self.closers = tuple(closers)
        # Tokens by the byte they start with, which is how a `uniqueItems` array
        # narrows a vocabulary of a hundred thousand down to the handful that
        # could continue the item in hand.
        self.by_first_byte = [tuple(bucket) for bucket in buckets]
        # Tokens contributing no document text -- special tokens, and ids the
        # tokenizer does not use. They move no array, so they are never vetoed.
        self.spell_nothing = frozenset(nothing)
        # The bytes the model can emit one at a time, which is what makes a veto
        # a detour rather than a dead end.
        self.singles = frozenset(singles)
        self._tables: dict[tuple[int, tuple[bool, ...]], tuple[tuple[_CloseKey, frozenset[int]], ...]] = {}
        self._landings: dict[tuple[bytes, ...], dict[tuple[int, int], tuple[tuple[_Landing, frozenset[int]], ...]]] = {}
        self._nothing_words: np.ndarray | None = None

    def nothing_words(self, num_words: int) -> "np.ndarray":
        """A bitmask of the tokens that spell nothing, the floor every
        `uniqueItems` mask is built up from."""
        if self._nothing_words is None or len(self._nothing_words) < num_words:
            words = np.zeros(max(num_words, (len(self.token_bytes) + 31) // 32), dtype=np.uint32)
            for token in self.spell_nothing:
                words[token >> 5] |= np.uint32(1 << (token & 31))
            self._nothing_words = words
        return self._nothing_words[:num_words]

    def landing_groups(
        self, domain: _Domain, position: int, node: int
    ) -> tuple[tuple[_Landing, frozenset[int]], ...]:
        """The tokens that could continue an array sitting at `(position, node)`,
        grouped by where they leave it.

        Only tokens starting with a byte the array can legally take next are
        walked; everything else is off the domain by its first byte, and the
        caller masks it in one go rather than naming it token by token.
        """
        table = self._landings.setdefault(domain.values, {})
        groups = table.get((position, node))
        if groups is None:
            collected: dict[_Landing, set[int]] = {}
            for token_id in self._candidates(domain, position, node):
                landing = _walk_unique(domain, position, node, self.token_bytes[token_id])
                if landing is not None:
                    collected.setdefault(landing, set()).add(token_id)
            groups = tuple((landing, frozenset(ids)) for landing, ids in collected.items())
            table[(position, node)] = groups
        return groups

    def _candidates(self, domain: _Domain, position: int, node: int) -> Iterator[int]:
        first: set[int] = set(_WS)
        if position == _P_INSIDE:
            first |= domain.children[node].keys()
            if domain.terminal[node] is not None:
                first |= {_COMMA, _RBRACKET}
        elif position == _P_BETWEEN:
            first |= domain.children[0].keys() | {_RBRACKET}
        else:
            first |= {_COMMA, _RBRACKET}
        for byte in first:
            yield from self.by_first_byte[byte]

    def closing_groups(self, state: int, kinds: tuple[bool, ...]) -> tuple[tuple[_CloseKey, frozenset[int]], ...]:
        """The candidate tokens that close that array, grouped by what they do.

        Whether a token closes is fixed by the lexer state and settled once
        here; what has to be decided per step is whether the items it finishes
        on the way satisfy the count. Hundreds of tokens finish nothing, or the
        same bytes, so they share that answer -- deciding it per group rather
        than per token is worth two orders of magnitude here.
        """
        groups = self._tables.get((state, kinds))
        if groups is None:
            collected: dict[_CloseKey, set[int]] = {}
            for token_id in self.closers:
                data = self.token_bytes[token_id]
                spans = _probe(state, kinds, data)
                if spans is None:
                    continue
                # `start is None`: the token contributes a suffix to be joined
                # onto the document. Otherwise it holds the whole item.
                key = tuple((start is None, data[:end] if start is None else data[start:end]) for start, end in spans)
                collected.setdefault(key, set()).add(token_id)
            groups = tuple((key, frozenset(ids)) for key, ids in collected.items())
            self._tables[(state, kinds)] = groups
        return groups


def _token_bytes(tokenizer: TokenizerLike, vocab_size: int) -> list[bytes]:
    """Token id to the bytes it spells, empty for the ids that spell nothing.

    Shares the outlines backend's vocabulary reduction, so tokenizer quirks are
    worked around in one place. Special tokens come back empty: they contribute
    no document text and so can neither open nor close an array.
    """
    from aphrodite.v1.structured_output.utils import _reduced_vocabulary

    table: list[bytes] = [b""] * vocab_size
    for text, ids in _reduced_vocabulary(tokenizer).items():
        for token_id in ids:
            if token_id < vocab_size:
                table[token_id] = text
    return table


def _vocab_profile(tokenizer: TokenizerLike, vocab_size: int) -> _VocabProfile:
    profile = getattr(tokenizer, "_postcondition_profile", None)
    if profile is None or len(profile.token_bytes) != vocab_size:
        profile = _VocabProfile(_token_bytes(tokenizer, vocab_size))
        tokenizer._postcondition_profile = profile  # type: ignore[attr-defined]
    return profile


# ---------------------------------------------------------------------------
# The document scanner.
# ---------------------------------------------------------------------------

_TRIVIAL: list[_Alternative] = [()]


def _dedupe(alternatives: list[_Alternative]) -> list[_Alternative]:
    seen: set[tuple[int, ...]] = set()
    out: list[_Alternative] = []
    for conjunction in alternatives:
        key = tuple(sorted(id(node) for node in conjunction))
        if key not in seen:
            seen.add(key)
            out.append(conjunction)
    return out


def _admits(conjunction: _Alternative, kind: str) -> bool:
    """Whether this conjunction can describe a value of type `kind`.

    Only `type` is consulted, which is enough to drop an `anyOf` sibling that
    was never about arrays. Without it, `anyOf: [{type: string}, {type: array,
    contains: ...}]` never vetoes: the string branch carries no obligation, so
    it permits every close.
    """
    for node in conjunction:
        declared = node.get("type")
        if isinstance(declared, str):
            if declared != kind:
                return False
        elif isinstance(declared, list) and kind not in declared:
            return False
    return True


class _Frame:
    """One open container in the document, and what the schema says about it."""

    __slots__ = (
        "is_array",
        "alternatives",
        "index",
        "key",
        "key_start",
        "item_start",
        "rules",
        "counts",
        "groups",
        "domains",
        "used",
        "unique_groups",
        "_matched",
    )

    def __init__(self, is_array: bool, alternatives: list[_Alternative], validator: Any, domains: dict[int, _Domain]):
        self.is_array = is_array
        # Alternatives that cannot describe this kind of container are dropped
        # here rather than at the veto: descending from one yields a cursor that
        # constrains nothing, so an `anyOf` branch saying "integer" would
        # reappear a level down saying nothing at all, permitting every close.
        kind = "array" if is_array else "object"
        self.alternatives = [c for c in alternatives if _admits(c, kind)] if alternatives is not _TRIVIAL else _TRIVIAL
        self.index = 0
        self.key: str | None = None
        self.key_start = 0
        self.item_start: int | None = None
        # One entry per distinct obligation reaching this array, and one group
        # of indices into it per alternative. Closing is permitted as soon as
        # *some* alternative is satisfied.
        self.rules: list[tuple[Any, int]] = []
        self.groups: list[tuple[int, ...]] = []
        self.counts: list[int] = []
        # The same arrangement for `uniqueItems`, and read the other way round:
        # a token is refused only when *every* alternative refuses it.
        self.domains: list[_Domain] = []
        self.unique_groups: list[tuple[int, ...]] = []
        self.used: list[set[int]] = []
        # Which rules an item's bytes satisfy. A pure function of those bytes,
        # so it survives rollback; the lookahead asks about the same handful of
        # candidate endings on every token of the array.
        self._matched: dict[bytes, tuple[int, ...]] = {}
        if is_array:
            self._build_rules(validator, domains)

    def _build_rules(self, validator: Any, domains: dict[int, _Domain]) -> None:
        counted: dict[int, int] = {}
        distinct: dict[int, int] = {}
        for conjunction in self.alternatives:
            group: list[int] = []
            unique_group: list[int] = []
            for node in conjunction:
                minimum = contains_obligation(node)
                if minimum is not None:
                    position = counted.get(id(node))
                    if position is None:
                        position = counted[id(node)] = len(self.rules)
                        self.rules.append((validator.evolve(schema=node["contains"]), minimum))
                    group.append(position)
                domain = domains.get(id(node))
                if domain is not None:
                    position = distinct.get(id(node))
                    if position is None:
                        position = distinct[id(node)] = len(self.domains)
                        self.domains.append(domain)
                    unique_group.append(position)
            self.groups.append(tuple(group))
            self.unique_groups.append(tuple(unique_group))
        self.counts = [0] * len(self.rules)
        self.used = [set() for _ in self.domains]

    def matched(self, raw: bytes) -> tuple[int, ...]:
        """Which of this array's obligations the item `raw` satisfies."""
        hit = self._matched.get(raw)
        if hit is None:
            try:
                value = json.loads(raw)
            except ValueError:
                # Not a complete value. Not counting an item we cannot read is
                # the safe direction; well-formedness is the backend's job.
                hit = ()
            else:
                hit = tuple(position for position, (matcher, _) in enumerate(self.rules) if matcher.is_valid(value))
            if len(self._matched) > 256:
                self._matched.clear()
            self._matched[raw] = hit
        return hit

    def record(self, raw: bytes) -> None:
        """Count a finished item against every obligation on this array."""
        if self.rules:
            for position in self.matched(raw):
                self.counts[position] += 1
        for position, domain in enumerate(self.domains):
            # The veto keeps items on the trie, so the bytes are canonical
            # already; only the whitespace around them is not part of the value.
            value = domain.index.get(raw.strip())
            if value is not None:
                self.used[position].add(value)

    def permits_close(self, extra: dict[int, int] | None = None) -> bool:
        if not self.groups:
            return True
        for group in self.groups:
            if all(self.counts[i] + (extra or {}).get(i, 0) >= self.rules[i][1] for i in group):
                return True
        return False

    def snapshot(self) -> tuple[Any, ...]:
        return (
            self.index,
            self.key,
            self.key_start,
            self.item_start,
            tuple(self.counts),
            tuple(frozenset(seen) for seen in self.used),
        )

    def restore(self, state: tuple[Any, ...]) -> None:
        self.index, self.key, self.key_start, self.item_start, counts, used = state
        self.counts = list(counts)
        self.used = [set(seen) for seen in used]


@dataclass(frozen=True)
class _Vetoes:
    """What one step takes away: groups the token may not come from, and groups
    it must come from. `key` identifies the pair by content, for the mask cache."""

    key: tuple[Any, ...]
    refused: tuple[frozenset[int], ...]
    confined: tuple[frozenset[int], ...]


class _Document:
    """The emitted bytes so far, the schema cursors that follow them, and the
    obligations those cursors carry."""

    def __init__(self, schema: Any, analysis: SchemaAnalysis, validator: Any):
        self.schema = schema
        self.relevant = analysis.relevant
        self.domains = analysis.domains
        self.validator = validator
        self.buf = bytearray()
        self.state = _S_VALUE
        self.stack: list[_Frame] = []
        self.pending = self._prune(_expand(schema, schema))

    # -- cursor movement ---------------------------------------------------

    def _prune(self, alternatives: list[_Alternative]) -> list[_Alternative]:
        """Drop everything the obligations do not depend on.

        A node is kept if an obligation is reachable from it, or if it carries a
        `type` -- the latter is what lets `_admits` rule out an irrelevant
        `anyOf` branch instead of letting it permit every close. When nothing
        relevant survives, navigation stops.
        """
        kept: list[_Alternative] = []
        interesting = False
        for conjunction in alternatives:
            keep: list[dict[str, Any]] = []
            for node in conjunction:
                if id(node) in self.relevant:
                    interesting = True
                    keep.append(node)
                elif "type" in node:
                    keep.append(node)
            kept.append(tuple(keep))
        if not interesting:
            return _TRIVIAL
        return _dedupe(kept)

    def _descend(self, alternatives: list[_Alternative], subschemas: Any) -> list[_Alternative]:
        if alternatives is _TRIVIAL:
            return _TRIVIAL
        out: list[_Alternative] = []
        try:
            for conjunction in alternatives:
                nested: list[_Alternative] = [()]
                for node in conjunction:
                    for sub in subschemas(node):
                        nested = _product(nested, _expand(sub, self.schema))
                out.extend(nested)
        except (_Unresolvable, _TooManyAlternatives):
            # Screened out at arrival; if one reaches here anyway, stop
            # constraining rather than guess.
            return _TRIVIAL
        return self._prune(out)

    # -- feeding -----------------------------------------------------------

    def feed(self, data: bytes) -> None:
        for byte in data:
            index = len(self.buf)
            self.buf.append(byte)
            in_object = bool(self.stack) and not self.stack[-1].is_array
            self.state, events = _lex_step(self.state, byte, in_object)
            for event in events:
                self._apply(event, index)

    def _apply(self, event: int, index: int) -> None:
        frame = self.stack[-1] if self.stack else None
        if event == _E_VALUE_BEGIN:
            if frame is not None and frame.is_array:
                frame.item_start = index
        elif event == _E_VALUE_END_EXCL:
            self._end_item(frame, index)
        elif event == _E_VALUE_END_INCL:
            self._end_item(frame, index + 1)
        elif event == _E_PUSH_ARRAY:
            pushed = _Frame(True, self.pending, self.validator, self.domains)
            self.stack.append(pushed)
            self.pending = self._descend(pushed.alternatives, lambda node: _item_schemas(node, 0))
        elif event == _E_PUSH_OBJECT:
            self.stack.append(_Frame(False, self.pending, self.validator, self.domains))
        elif event == _E_POP:
            if self.stack:
                self.stack.pop()
        elif event == _E_COMMA:
            if frame is not None and frame.is_array:
                frame.index += 1
                self.pending = self._descend(frame.alternatives, lambda node: _item_schemas(node, frame.index))
        elif event == _E_KEY_BEGIN:
            if frame is not None:
                frame.key_start = index + 1
        elif event == _E_KEY_END:
            if frame is not None:
                frame.key = self._read_key(frame.key_start, index)
                self.pending = self._descend(frame.alternatives, lambda node: _property_schemas(node, frame.key))

    def _read_key(self, start: int, end: int) -> str:
        raw = bytes(self.buf[start:end])
        try:
            return json.loads(b'"' + raw + b'"')
        except ValueError:
            return raw.decode("utf-8", "replace")

    def _end_item(self, frame: _Frame | None, end: int) -> None:
        if frame is not None and frame.is_array and frame.item_start is not None:
            frame.record(bytes(self.buf[frame.item_start : end]))
            frame.item_start = None

    # -- the veto ----------------------------------------------------------

    def vetoed_tokens(self, profile: _VocabProfile) -> _Vetoes | None:
        """What this step takes away, or None if it takes nothing.

        Two shapes, because the two keywords rule out opposite amounts of the
        vocabulary. `minContains` forbids a few hundred tokens spelling a `]`
        and leaves the rest alone; `uniqueItems` allows only the tokens still on
        the domain's trie and rules out everything else, which is far too many
        to name. Both are left as groups: callers test membership or build a
        mask, and merging would rebuild the sets on every token.
        """
        refused = self._closes_too_early(profile)
        confined, keys = self._confined_to_domain(profile)
        if not refused and not confined:
            return None
        return _Vetoes(
            key=(tuple(sorted(id(group) for group in refused)), keys),
            refused=refused,
            confined=confined,
        )

    def _closes_too_early(self, profile: _VocabProfile) -> tuple[frozenset[int], ...]:
        """Tokens that would close an array still short of its `minContains`."""
        blocked = [
            (tuple(not f.is_array for f in self.stack[position:]), frame)
            for position, frame in enumerate(self.stack)
            if frame.is_array and frame.rules and not frame.permits_close()
        ]
        vetoed: list[frozenset[int]] = []
        for kinds, frame in blocked:
            for key, tokens in profile.closing_groups(self.state, kinds):
                items = self._materialise(frame, key)
                if not frame.permits_close(self._increments(frame, items)):
                    vetoed.append(tokens)
        return tuple(vetoed)

    def _confined_to_domain(
        self, profile: _VocabProfile
    ) -> tuple[tuple[frozenset[int], ...], tuple[Any, ...]]:
        """The tokens that keep the innermost array's items distinct.

        Only the innermost frame, and only when it is the array under the
        keyword: an enforceable domain holds scalars, so nothing is open inside
        one of its items and there is no deeper array to be in the middle of.

        The second half of the return is a content key for the mask cache. The
        sets are rebuilt each step, so their identity says nothing.
        """
        frame = self.stack[-1] if self.stack else None
        if frame is None or not frame.domains:
            return (), ()

        allowed: set[int] = set()
        keys: list[Any] = []
        for group in frame.unique_groups:
            if not group:
                # An alternative with nothing to say about uniqueness. The
                # document can still come out valid under it, so nothing goes.
                return (), ()
            within: set[int] | None = None
            for position in group:
                permitted = self._on_domain(frame, position, profile, keys)
                if permitted is None:
                    return (), ()
                within = permitted if within is None else within & permitted
            allowed |= within or set()
        return (frozenset(allowed),), tuple(keys)

    def _on_domain(
        self, frame: _Frame, position: int, profile: _VocabProfile, keys: list[Any]
    ) -> set[int] | None:
        """The tokens that leave this array still able to finish, under one
        domain. None if the document has left that domain, where saying nothing
        beats guessing."""
        domain = frame.domains[position]
        used = frame.used[position]
        if frame.item_start is None:
            where, node = (_P_BETWEEN if self.state == _S_VALUE else _P_AFTER), 0
        else:
            where = _P_INSIDE
            located = domain.locate(bytes(self.buf[frame.item_start :]))
            if located is None:
                return None
            node = located
        keys.append((id(domain), where, node, frozenset(used)))
        return {
            token
            for landing, tokens in profile.landing_groups(domain, where, node)
            if _permits(domain, landing, used)
            for token in tokens
        }

    def _materialise(self, frame: _Frame, key: _CloseKey) -> list[bytes]:
        items: list[bytes] = []
        for continues, fragment in key:
            if not continues:
                items.append(fragment)
            elif frame.item_start is not None:
                items.append(bytes(self.buf[frame.item_start :]) + fragment)
        return items

    def _increments(self, frame: _Frame, items: list[bytes]) -> dict[int, int]:
        extra: dict[int, int] = {}
        for raw in items:
            for position in frame.matched(raw):
                extra[position] = extra.get(position, 0) + 1
        return extra

    # -- undo --------------------------------------------------------------

    def snapshot(self) -> tuple[Any, ...]:
        return (len(self.buf), self.state, self.pending, tuple((f, f.snapshot()) for f in self.stack))

    def restore(self, state: tuple[Any, ...]) -> None:
        length, self.state, self.pending, frames = state
        del self.buf[length:]
        self.stack = [frame for frame, _ in frames]
        for frame, saved in frames:
            frame.restore(saved)


# ---------------------------------------------------------------------------
# The decorator grammar.
# ---------------------------------------------------------------------------


class PostconditionGrammar(StructuredOutputGrammar):
    """A backend grammar plus the vetoes that backend cannot express.

    Everything is delegated to the wrapped grammar first, so the extra
    constraint can only remove tokens the backend had already allowed.
    """

    def __init__(
        self,
        inner: StructuredOutputGrammar,
        schema: Any,
        analysis: SchemaAnalysis,
        profile: _VocabProfile,
        vocab_size: int,
        max_rollback: int,
    ):
        self.inner = inner
        self.schema = schema
        self.analysis = analysis
        self.profile = profile
        self.vocab_size = vocab_size
        self.num_words = (vocab_size + 31) // 32
        self._validator = jsonschema.validators.validator_for(schema)(schema)
        self._document = _Document(schema, analysis, self._validator)
        # One entry per token advanced through, newest last, capped at
        # `max_rollback`. Rollback is exact rather than replayed so its cost
        # scales with the speculative window rather than the output.
        self._undo: list[tuple[Any, ...]] = []
        # Document length after each token, kept for every token rather than
        # just the retained window, so a rewind past the log can still rebuild
        # the exact document the model emitted instead of an approximation.
        self._lengths: list[int] = []
        self._max_rollback = max_rollback
        self._masks: dict[tuple[int, ...], torch.Tensor] = {}

    # -- StructuredOutputGrammar ------------------------------------------

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        for token in tokens:
            vetoed = self._document.vetoed_tokens(self.profile)
            if _is_vetoed(vetoed, token, self.profile):
                logger.debug(
                    "Request %s: token %d would break a postcondition on an open array.",
                    request_id,
                    token,
                )
                return False
            if not self.inner.accept_tokens(request_id, [token]):
                return False
            self._advance(token)
            if self.inner.is_terminated():
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted = self.inner.validate_tokens(tokens)
        if not accepted:
            return accepted
        probed = 0
        try:
            for position, token in enumerate(accepted):
                vetoed = self._document.vetoed_tokens(self.profile)
                if _is_vetoed(vetoed, token, self.profile):
                    return accepted[:position]
                self._advance(token)
                probed += 1
        finally:
            # `validate_tokens` must not advance the FSM.
            self._rewind(probed)
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.inner.rollback(num_tokens)
        self._rewind(num_tokens)

    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        self.inner.fill_bitmask(bitmask, batch_index)
        vetoed = self._document.vetoed_tokens(self.profile)
        if vetoed is None:
            return
        mask = self._mask_for(vetoed)
        row = bitmask[batch_index]
        # The bitmask is shared and sized for the widest backend, as in
        # `XgrammarGrammar.fill_bitmask`.
        width = min(row.shape[-1], self.num_words)
        row[:width] &= mask[:width]

    def is_terminated(self) -> bool:
        return self.inner.is_terminated()

    def reset(self):
        self.inner.reset()
        self._document = _Document(self.schema, self.analysis, self._validator)
        self._undo.clear()
        self._lengths.clear()

    # -- internals ---------------------------------------------------------

    def _advance(self, token: int) -> None:
        self._undo.append(self._document.snapshot())
        if len(self._undo) > self._max_rollback:
            del self._undo[0]
        text = self.profile.token_bytes[token] if token < self.vocab_size else b""
        if text:
            self._document.feed(text)
        self._lengths.append(len(self._document.buf))

    def _rewind(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        if num_tokens > len(self._undo):
            logger.warning(
                "Postcondition rollback of %d tokens exceeds the %d retained; rescanning.",
                num_tokens,
                len(self._undo),
            )
            self._rescan(num_tokens)
            return
        state = self._undo[-num_tokens]
        del self._undo[-num_tokens:]
        self._document.restore(state)
        del self._lengths[-num_tokens:]

    def _rescan(self, num_tokens: int) -> None:
        """Rebuild the scanner from the document text, minus the last `num_tokens`.

        Exact, and costs the whole document. The snapshot log exists to keep
        this path from being taken.
        """
        keep = len(self._lengths) - num_tokens
        text = bytes(self._document.buf[: self._lengths[keep - 1]]) if keep > 0 else b""
        del self._lengths[keep if keep > 0 else 0 :]
        self._undo.clear()
        self._document = _Document(self.schema, self.analysis, self._validator)
        if text:
            self._document.feed(text)

    def _mask_for(self, vetoes: _Vetoes) -> "torch.Tensor":
        mask = self._masks.get(vetoes.key)
        if mask is None:
            words = np.full(self.num_words, 0xFFFFFFFF, dtype=np.uint32)
            for group in vetoes.refused:
                for token in group:
                    if token < self.vocab_size:
                        words[token >> 5] &= np.uint32(0xFFFFFFFF ^ (1 << (token & 31)))
            for group in vetoes.confined:
                # Built up from the tokens that spell nothing, which move no
                # array and so are never the ones confined.
                permitted = self.profile.nothing_words(self.num_words).copy()
                for token in group:
                    if token < self.vocab_size:
                        permitted[token >> 5] |= np.uint32(1 << (token & 31))
                words &= permitted
            mask = torch.from_numpy(words.view(np.int32))
            if len(self._masks) > 16:
                self._masks.clear()
            self._masks[vetoes.key] = mask
        return mask


def _is_vetoed(vetoes: _Vetoes | None, token: int, profile: _VocabProfile) -> bool:
    if vetoes is None:
        return False
    if any(token in group for group in vetoes.refused):
        return True
    return token not in profile.spell_nothing and any(token not in group for group in vetoes.confined)


def maybe_wrap(
    grammar: StructuredOutputGrammar,
    request_type: StructuredOutputOptions,
    grammar_spec: Any,
    tokenizer: TokenizerLike,
    vocab_size: int,
    num_speculative_tokens: int = 0,
) -> StructuredOutputGrammar:
    """Wrap `grammar` if its schema carries an obligation this layer can keep,
    and return it untouched otherwise."""
    if request_type != StructuredOutputOptions.JSON:
        # Structural tags are refused at arrival for these keywords instead:
        # finding the schema body inside the tag's own trigger/begin/end output
        # would need a second scanner.
        return grammar

    schema = grammar_spec
    if isinstance(schema, (str, bytes)):
        try:
            schema = json.loads(schema)
        except ValueError:
            return grammar
    if not isinstance(schema, dict):
        return grammar

    analysis = analyze(schema)
    if not analysis:
        return grammar
    if analysis.problems:
        # The screen should have refused this. Enforcing half of it is worse
        # than the rejection the caller was owed.
        logger.warning(
            "Schema reached the decoder with an unenforceable postcondition (%s). "
            "It will not be enforced; this is a screening bug, please file an issue.",
            analysis.problems[0],
        )
        return grammar

    profile = _vocab_profile(tokenizer, vocab_size)
    # Every veto here assumes the model can spell its way around one, byte by
    # byte if it has to: the detour past a refused `]` is a `,`, and the detour
    # past an item that would repeat is the next byte of one that would not.
    needed = {_RBRACKET, _COMMA}.union(
        byte for domain in analysis.domains.values() for value in domain.values for byte in value
    )
    if not needed <= profile.singles:
        logger.warning_once(
            "This tokenizer cannot spell every byte these keywords may have to "
            "steer around one at a time, so a veto could refuse a token the model "
            "has no other way to say. Leaving contains/minContains/uniqueItems "
            "unenforced."
        )
        return grammar

    return PostconditionGrammar(
        inner=grammar,
        schema=schema,
        analysis=analysis,
        profile=profile,
        vocab_size=vocab_size,
        max_rollback=max(num_speculative_tokens + 8, 16),
    )
