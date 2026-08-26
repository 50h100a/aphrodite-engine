# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enforce, while decoding, the JSON Schema keywords no grammar can express.

A context-free grammar cannot count how many items matched a subschema, so no
backend enforces `contains`/`minContains`. `StructuredOutputGrammar` is a
per-request stateful object, so enforcement lands here instead: a decorator
around whatever grammar the backend compiled, scanning the bytes the model has
emitted and ANDing extra vetoes into the bitmask.

`contains` is decided by handing each completed array item to `jsonschema`, so
the matched subschema may be arbitrarily structural; only array item boundaries
are tracked incrementally. Vetoes are empty except at an array's closing bracket.

Two invariants tie this to the screen in `schema_features`:

- **Reachability.** The screen refuses any schema whose obligations sit
  somewhere this scanner cannot navigate to, using `analyze` below -- the same
  walk, so the two cannot disagree about what is enforceable.
- **No walls.** Every veto leaves at least one legal continuation, given the
  static rules the screen applies (notably: `minContains` alongside `maxItems`
  is refused, because masking `]` in a bounded array can wall).
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

LAYER_ENFORCED_KEYWORDS = frozenset({"contains", "minContains"})

# Ceiling on the DNF expansion of one schema position. `anyOf` sites multiply.
_MAX_ALTERNATIVES = 64


# ---------------------------------------------------------------------------
# Static analysis: what this layer can enforce, and where it cannot reach.
# ---------------------------------------------------------------------------


@dataclass
class SchemaAnalysis:
    """What `analyze` found: the obligations, and every reason one was refused.

    A refused obligation is simply absent from `obligations`, which is what the
    screen reads. `problems` says why, and decides nothing.
    """

    problems: list[str] = field(default_factory=list)
    # Nodes carrying a live `contains` obligation, by identity.
    obligations: set[int] = field(default_factory=set)
    # Nodes from which an obligation is reachable. Navigation stops where this
    # does, so an ordinary subtree costs nothing to walk past.
    relevant: set[int] = field(default_factory=set)


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


def analyze(schema: Any) -> SchemaAnalysis:
    """What this layer can enforce in `schema`, and every reason it cannot.

    Read by both the screen and the runtime, so the two cannot disagree about
    which obligations are covered.
    """
    analysis = SchemaAnalysis()
    if not isinstance(schema, dict):
        return analysis

    applicable = [node for node in _walk(schema, navigable_only=False) if contains_obligation(node) is not None]
    if not applicable:
        return analysis

    live = {id(node): node for node in _walk(schema, navigable_only=True)}

    # Static refusals, in schema order so the message names the first problem a
    # caller would look for.
    for node in applicable:
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

    if not analysis.obligations:
        return analysis

    # The cursor set has to be buildable everywhere on the way down, or the
    # runtime would navigate past an obligation without seeing it.
    try:
        _expand(schema, schema)
        for node in live.values():
            for key in _NAV_SINGLE:
                if key in node:
                    _expand(node[key], schema)
            for key in _NAV_LIST:
                value = node.get(key)
                if isinstance(value, list):
                    for sub in value:
                        _expand(sub, schema)
            for key in _NAV_MAP:
                value = node.get(key)
                if isinstance(value, dict):
                    for sub in value.values():
                        _expand(sub, schema)
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

    _mark_relevant(schema, analysis)
    return analysis


def _refuse_everything(analysis: SchemaAnalysis, reason: str) -> SchemaAnalysis:
    """Give up on the whole schema, not just the obligation that tripped over it.

    Best-effort mode strips exactly what was refused, so a schema-wide reason
    must take every obligation with it or the strip leaves the screen still
    failing on the next pass.
    """
    analysis.problems.append(reason)
    analysis.obligations.clear()
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

    pending = list(analysis.obligations)
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


# ---------------------------------------------------------------------------
# Per-vocabulary lookahead tables.
# ---------------------------------------------------------------------------


class _VocabProfile:
    """Everything about a tokenizer this layer needs, derived once.

    Only tokens spelling a `]` can close an array, which keeps the lookahead
    affordable: a few hundred candidates out of a vocabulary of a hundred
    thousand, and the answer for each depends only on the lexer state and the
    nesting offset.
    """

    def __init__(self, token_bytes: list[bytes]):
        self.token_bytes = token_bytes
        self.closers: tuple[int, ...] = tuple(
            token_id for token_id, text in enumerate(self.token_bytes) if _RBRACKET in text
        )
        # A veto is only escapable if the model has another way to spell the
        # same bytes. Without these, an over-broad veto becomes a dead end.
        self.can_spell_singly = all(
            any(text == bytes([byte]) for text in self.token_bytes) for byte in (_RBRACKET, _COMMA)
        )
        self._tables: dict[tuple[int, tuple[bool, ...]], tuple[tuple[_CloseKey, frozenset[int]], ...]] = {}

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
        "_matched",
    )

    def __init__(self, is_array: bool, alternatives: list[_Alternative], validator: Any):
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
        # Which rules an item's bytes satisfy. A pure function of those bytes,
        # so it survives rollback; the lookahead asks about the same handful of
        # candidate endings on every token of the array.
        self._matched: dict[bytes, tuple[int, ...]] = {}
        if is_array:
            self._build_rules(validator)

    def _build_rules(self, validator: Any) -> None:
        positions: dict[int, int] = {}
        for conjunction in self.alternatives:
            group: list[int] = []
            for node in conjunction:
                minimum = contains_obligation(node)
                if minimum is None:
                    continue
                position = positions.get(id(node))
                if position is None:
                    position = positions[id(node)] = len(self.rules)
                    self.rules.append((validator.evolve(schema=node["contains"]), minimum))
                group.append(position)
            self.groups.append(tuple(group))
        self.counts = [0] * len(self.rules)

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
        if not self.rules:
            return
        for position in self.matched(raw):
            self.counts[position] += 1

    def permits_close(self, extra: dict[int, int] | None = None) -> bool:
        if not self.groups:
            return True
        for group in self.groups:
            if all(self.counts[i] + (extra or {}).get(i, 0) >= self.rules[i][1] for i in group):
                return True
        return False

    def snapshot(self) -> tuple[Any, ...]:
        return (self.index, self.key, self.key_start, self.item_start, tuple(self.counts))

    def restore(self, state: tuple[Any, ...]) -> None:
        self.index, self.key, self.key_start, self.item_start, counts = state
        self.counts = list(counts)


class _Document:
    """The emitted bytes so far, the schema cursors that follow them, and the
    obligations those cursors carry."""

    def __init__(self, schema: Any, analysis: SchemaAnalysis, validator: Any):
        self.schema = schema
        self.relevant = analysis.relevant
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
            pushed = _Frame(True, self.pending, self.validator)
            self.stack.append(pushed)
            self.pending = self._descend(pushed.alternatives, lambda node: _item_schemas(node, 0))
        elif event == _E_PUSH_OBJECT:
            self.stack.append(_Frame(False, self.pending, self.validator))
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

    def vetoed_tokens(self, profile: _VocabProfile) -> list[frozenset[int]] | None:
        """Token groups that would close an array still short of its `minContains`.

        Left as groups rather than merged: callers only test membership or build
        a mask keyed on group identity, and merging would rebuild a
        several-hundred-element set on every token.
        """
        blocked = [
            (tuple(not f.is_array for f in self.stack[position:]), frame)
            for position, frame in enumerate(self.stack)
            if frame.is_array and frame.rules and not frame.permits_close()
        ]
        if not blocked:
            return None

        vetoed: list[frozenset[int]] = []
        for kinds, frame in blocked:
            for key, tokens in profile.closing_groups(self.state, kinds):
                items = self._materialise(frame, key)
                if not frame.permits_close(self._increments(frame, items)):
                    vetoed.append(tokens)
        return vetoed or None

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
            if _is_vetoed(vetoed, token):
                logger.debug(
                    "Request %s: token %d would close an array short of its minContains.",
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
                if _is_vetoed(vetoed, token):
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

    def _mask_for(self, vetoed: list[frozenset[int]]) -> "torch.Tensor":
        # Keyed on group identity, which the vocabulary profile keeps alive for
        # the life of the process. Hashing the token ids would mean re-reading
        # several hundred of them per token.
        key = tuple(sorted(id(group) for group in vetoed))
        mask = self._masks.get(key)
        if mask is None:
            words = np.full(self.num_words, 0xFFFFFFFF, dtype=np.uint32)
            for group in vetoed:
                for token in group:
                    if token < self.vocab_size:
                        words[token >> 5] &= np.uint32(0xFFFFFFFF ^ (1 << (token & 31)))
            mask = torch.from_numpy(words.view(np.int32))
            if len(self._masks) > 16:
                self._masks.clear()
            self._masks[key] = mask
        return mask


def _is_vetoed(vetoed: list[frozenset[int]] | None, token: int) -> bool:
    return vetoed is not None and any(token in group for group in vetoed)


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
    if not analysis.obligations:
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
    if not profile.can_spell_singly:
        logger.warning_once(
            "This tokenizer has no single-byte tokens for `]` or `,`, so the "
            "contains/minContains layer could refuse a token the model has no "
            "other way to spell. Leaving those keywords unenforced."
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
