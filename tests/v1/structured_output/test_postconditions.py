# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The keywords enforced after the grammar, and the promises that layer makes.

`test_backend_capability` measures backends, which this layer is invisible to,
so it needs its own harness: one token per byte, an inner grammar that permits
everything, and a driver that reads the bitmask at every position.

Two separate things are held here:

- **Soundness.** A violating document never gets through. What the vetoes are
  for.
- **No walls.** A veto never leaves the model with nothing legal to say. What
  the static rejections in `schema_features` are for, and the easier of the two
  to break: a mask that empties ends the request with
  `finish_reason="constraint"`, visible but still a death over a satisfiable
  schema.
"""

import json

import pytest
import torch

from aphrodite.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from aphrodite.v1.structured_output.postconditions import (
    PostconditionGrammar,
    _VocabProfile,
    analyze,
    maybe_wrap,
)

pytestmark = pytest.mark.cpu_test


# One token per byte, plus an id that spells nothing (an EOS stand-in), so the
# mask reads directly as "which characters may come next".
VOCAB_SIZE = 257
PROFILE = _VocabProfile([bytes([i]) for i in range(256)] + [b""])
EOS = 256


class _PermitEverything(StructuredOutputGrammar):
    """Stands in for the backend grammar. Constrains nothing, so anything the
    driver sees masked was masked by the layer under test and nothing else."""

    def __init__(self):
        self.accepted: list[int] = []

    def accept_tokens(self, request_id, tokens):
        self.accepted.extend(tokens)
        return True

    def validate_tokens(self, tokens):
        return list(tokens)

    def rollback(self, num_tokens):
        if num_tokens:
            del self.accepted[-num_tokens:]

    def fill_bitmask(self, bitmask, batch_index):
        bitmask[batch_index].fill_(-1)

    def is_terminated(self):
        return False

    def reset(self):
        self.accepted.clear()


def build(schema, inner=None):
    analysis = analyze(schema)
    assert not analysis.problems, f"schema was refused: {analysis.problems[0].reason}"
    assert analysis.obligations, "schema carries nothing for this layer to enforce"
    return PostconditionGrammar(
        inner=inner or _PermitEverything(),
        schema=schema,
        analysis=analysis,
        profile=PROFILE,
        vocab_size=VOCAB_SIZE,
        max_rollback=16,
    )


def allowed(grammar):
    """The characters the layer would permit as the very next byte."""
    bitmask = torch.full((1, (VOCAB_SIZE + 31) // 32), -1, dtype=torch.int32)
    grammar.fill_bitmask(bitmask, 0)
    row = bitmask[0]
    return {chr(t) for t in range(256) if (int(row[t >> 5]) >> (t & 31)) & 1}


def emit(grammar, text):
    """Feed `text` byte by byte, returning what was allowed before each byte."""
    history = []
    for byte in text.encode():
        history.append(allowed(grammar))
        assert grammar.accept_tokens("r", [byte]), f"layer refused a legal byte {chr(byte)!r} of {text!r}"
    history.append(allowed(grammar))
    return history


def closable(schema, text):
    """Whether a fresh decode of `text` under `schema` could close the array now."""
    return "]" in emit(build(schema), text)[-1]


# ---------------------------------------------------------------------------
# contains / minContains
# ---------------------------------------------------------------------------

CONTAINS_SEVEN = {"type": "array", "items": {"type": "integer"}, "contains": {"const": 7}}
TWO_SEVENS = {**CONTAINS_SEVEN, "minContains": 2}


def test_the_array_cannot_close_until_an_item_matches():
    assert not closable(CONTAINS_SEVEN, "[")
    assert not closable(CONTAINS_SEVEN, "[1")
    assert not closable(CONTAINS_SEVEN, "[1,2")


def test_it_closes_as_soon_as_one_matches():
    assert closable(CONTAINS_SEVEN, "[1,7")


def test_the_pending_item_counts_before_it_is_finished():
    """`[7` can close: the bracket ends the number *and* the array, so by the
    time the array is closed the item exists and matches. Refusing here would
    force a spurious extra element for no reason the schema asks for."""
    assert closable(CONTAINS_SEVEN, "[7")
    assert not closable(CONTAINS_SEVEN, "[8")


def test_min_contains_counts_matches_not_items():
    assert not closable(TWO_SEVENS, "[7")
    assert not closable(TWO_SEVENS, "[7,1,2,3")
    assert closable(TWO_SEVENS, "[7,1,2,3,7")


def test_a_satisfied_array_is_left_entirely_alone():
    """Nothing is masked once the obligation is met -- the layer stops having
    an opinion rather than continuing to police the array."""
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[7,1")
    assert allowed(grammar) == {chr(i) for i in range(256)}


def test_contains_may_be_structural():
    """The matched subschema is handed to jsonschema whole, so it is not
    restricted to the scalars a grammar could have compared against."""
    schema = {
        "type": "array",
        "items": {"type": "object"},
        "contains": {"type": "object", "properties": {"role": {"const": "admin"}}, "required": ["role"]},
    }
    assert not closable(schema, '[{"role":"guest"}')
    assert closable(schema, '[{"role":"guest"},{"role":"admin"}')


def test_equality_is_by_value_not_by_bytes():
    """`1.0` and `1` are the same JSON value, and whitespace is not part of it."""
    assert closable({"type": "array", "contains": {"const": 1}}, "[ 1.0 ")


# ---------------------------------------------------------------------------
# Where the obligation sits
# ---------------------------------------------------------------------------


def test_an_obligation_nested_under_a_property():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "contains": {"const": "x"}}},
        "required": ["tags"],
    }
    assert not closable(schema, '{"tags":["a"')
    assert closable(schema, '{"tags":["a","x"')


def test_only_the_array_the_obligation_names_is_policed():
    """A sibling array with no `contains` must close freely, including from
    inside the constrained one -- the veto is per array, not per bracket."""
    schema = {
        "type": "object",
        "properties": {
            "free": {"type": "array"},
            "held": {"type": "array", "contains": {"const": 7}},
        },
    }
    assert closable(schema, '{"free":[1,2')
    assert not closable(schema, '{"free":[1,2],"held":[1')


def test_an_inner_array_still_closes_while_the_outer_one_is_held():
    """The outer array is short of its `contains`, so its own `]` is masked --
    but the inner array's `]` is a different bracket at a different depth."""
    schema = {"type": "array", "items": {"type": "array"}, "contains": {"const": []}}
    grammar = build(schema)
    history = emit(grammar, "[[1,2")
    assert "]" in history[-1], "the inner array was walled by the outer array's obligation"


def test_a_token_that_would_close_both_at_once_is_refused():
    """`]]` closes the inner array and then the outer one. The outer is still
    short, so the whole token goes -- a veto has to look at what a token spells,
    not at the bracket it happens to start with."""
    schema = {"type": "array", "items": {"type": "array"}, "contains": {"const": [7]}}
    grammar = build(schema)
    emit(grammar, "[[1")
    profile = _VocabProfile([b"]", b"]]", b"1"])
    grammar.profile = profile
    grammar.vocab_size = 3
    grammar.num_words = 1
    grammar._masks.clear()
    bitmask = torch.full((1, 1), -1, dtype=torch.int32)
    grammar.fill_bitmask(bitmask, 0)
    assert int(bitmask[0][0]) & 0b001, "the inner array's own close was masked"
    assert not int(bitmask[0][0]) & 0b010, "a token closing both arrays was allowed"


def test_a_ref_is_followed_to_the_obligation():
    schema = {
        "$defs": {"Bag": {"type": "array", "contains": {"const": 7}}},
        "type": "object",
        "properties": {"bag": {"$ref": "#/$defs/Bag"}},
    }
    assert not closable(schema, '{"bag":[1')
    assert closable(schema, '{"bag":[7')


def test_a_recursive_schema_expands_only_as_deep_as_the_document():
    schema = {
        "$defs": {
            "Node": {
                "type": "array",
                "contains": {"const": 0},
                "items": {"anyOf": [{"type": "integer"}, {"$ref": "#/$defs/Node"}]},
            }
        },
        "$ref": "#/$defs/Node",
    }
    assert not closable(schema, "[1,[1,[1")
    assert closable(schema, "[1,[1,[0")


# ---------------------------------------------------------------------------
# Ambiguity: anyOf, oneOf, allOf
# ---------------------------------------------------------------------------


def test_a_branch_that_asks_for_nothing_permits_the_close():
    """Sound, and deliberately permissive: while the document is still live in
    a branch with no obligation, it can still come out valid under that branch,
    so vetoing would refuse a legal document."""
    schema = {"anyOf": [{"type": "array", "contains": {"const": 7}}, {"type": "array", "maxItems": 3}]}
    assert closable(schema, "[1")


def test_a_branch_of_the_wrong_type_does_not_excuse_the_obligation():
    """The permissive rule above must not degenerate into never vetoing. A
    `string` branch cannot describe an array, so it is dropped and the array
    branch is left holding the obligation on its own."""
    schema = {"anyOf": [{"type": "string"}, {"type": "array", "contains": {"const": 7}}]}
    assert not closable(schema, "[1")
    assert closable(schema, "[7")


def test_all_of_is_conjunction_so_both_obligations_hold():
    schema = {
        "allOf": [
            {"type": "array", "contains": {"const": 7}},
            {"type": "array", "contains": {"const": 8}},
        ]
    }
    assert not closable(schema, "[7")
    assert not closable(schema, "[8")
    assert closable(schema, "[7,8")


def test_two_branches_that_both_demand_something_still_veto():
    schema = {
        "anyOf": [
            {"type": "array", "contains": {"const": 7}},
            {"type": "array", "contains": {"const": 8}},
        ]
    }
    assert not closable(schema, "[1")
    assert closable(schema, "[8")


# ---------------------------------------------------------------------------
# Speculative decoding: the layer is rewound and re-probed constantly.
# ---------------------------------------------------------------------------


def test_rollback_restores_the_match_count_exactly():
    grammar = build(TWO_SEVENS)
    emit(grammar, "[7,7")
    assert "]" in allowed(grammar)
    grammar.rollback(2)  # undo the "7" and the "," before it
    assert "]" not in allowed(grammar)
    emit(grammar, ",7")
    assert "]" in allowed(grammar)


def test_rollback_unwinds_a_closed_array():
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[7]")
    grammar.rollback(1)
    assert "]" in allowed(grammar), "reopening the array lost the item that satisfied it"
    grammar.rollback(1)
    assert "]" not in allowed(grammar), "the 7 was still counted after being rolled back"


def test_validate_tokens_does_not_advance_the_scanner():
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[1")
    before = allowed(grammar)
    assert grammar.validate_tokens(list(b",7")) == list(b",7")
    assert allowed(grammar) == before


def test_validate_tokens_stops_at_the_first_veto():
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[1")
    assert grammar.validate_tokens(list(b"2]")) == list(b"2")


def test_a_draft_token_that_would_violate_is_refused():
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[1")
    assert grammar.accept_tokens("r", [ord("]")]) is False
    # And the refusal left nothing behind: the array is still open.
    assert grammar.accept_tokens("r", list(b",7]")) is True


def test_reset_forgets_the_document():
    grammar = build(CONTAINS_SEVEN)
    emit(grammar, "[7")
    grammar.reset()
    assert "]" not in emit(grammar, "[1")[-1]


# ---------------------------------------------------------------------------
# Static refusals: what this layer will not take on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema,expected",
    [
        (
            {"type": "array", "contains": {"const": 7}, "maxContains": 2},
            "maxContains cannot be enforced",
        ),
        (
            {"type": "array", "contains": {"const": 7}, "minContains": 2, "maxItems": 3},
            "cannot be enforced alongside maxItems",
        ),
        (
            {
                "type": "object",
                "unevaluatedProperties": {"type": "array", "contains": {"const": 7}},
            },
            "cannot follow it",
        ),
        (
            {"type": "array", "contains": {"const": 7}, "items": {"$ref": "https://example.com/x"}},
            "cannot follow",
        ),
    ],
    ids=["maxContains", "maxItems", "unevaluated", "remote-ref"],
)
def test_schemas_this_layer_refuses(schema, expected):
    problems = analyze(schema).problems
    assert problems, "expected a refusal"
    assert expected in problems[0]


def test_min_contains_zero_is_inert():
    """`minContains: 0` makes `contains` vacuous by the spec: every array,
    including the empty one, has at least zero matching items."""
    analysis = analyze({"type": "array", "contains": {"const": 7}, "minContains": 0})
    assert not analysis.obligations
    assert not analysis.problems


def test_a_dead_branch_is_not_held_against_the_schema():
    """`if` without `then`/`else` applies to nothing, and a `$defs` entry no
    `$ref` reaches is never checked against anything. Refusing a schema over a
    constraint that can never fire would be a rejection the caller cannot act
    on."""
    for schema in (
        {"type": "object", "if": {"type": "array", "contains": {"const": 7}, "maxContains": 1}},
        {"$defs": {"Unused": {"type": "array", "maxContains": 1, "contains": {"const": 7}}}, "type": "object"},
    ):
        analysis = analyze(schema)
        assert not analysis.problems, schema
        assert not analysis.obligations, schema


def test_an_ordinary_schema_is_not_wrapped():
    """The layer costs nothing when there is nothing for it to do."""
    inner = _PermitEverything()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert maybe_wrap(inner, StructuredOutputOptions.JSON, json.dumps(schema), None, VOCAB_SIZE) is inner


def test_a_structural_tag_is_never_wrapped():
    """A tag is refused at arrival for these keywords instead: finding the
    schema body inside the tag's own trigger/begin/end output would take a
    second scanner."""
    inner = _PermitEverything()
    spec = json.dumps({"structures": [{"begin": "<f>", "schema": CONTAINS_SEVEN, "end": "</f>"}], "triggers": ["<f>"]})
    assert maybe_wrap(inner, StructuredOutputOptions.STRUCTURAL_TAG, spec, None, VOCAB_SIZE) is inner


# ---------------------------------------------------------------------------
# Wiring: the layer only helps if requests actually reach it.
# ---------------------------------------------------------------------------


class _FakeHFTokenizer:
    """The narrow slice of a HF tokenizer that the vocabulary reduction reads."""

    def __init__(self, tokens):
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self.eos_token_id = len(tokens) - 1
        self.all_special_tokens = [tokens[-1]]

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def test_token_bytes_are_read_off_the_tokenizer():
    from aphrodite.v1.structured_output.postconditions import _token_bytes

    tokenizer = _FakeHFTokenizer(["[", "]", ",", "7", "77", "<eos>"])
    assert _token_bytes(tokenizer, 6) == [b"[", b"]", b",", b"7", b"77", b""]


def test_a_schema_with_contains_is_wrapped_end_to_end():
    tokenizer = _FakeHFTokenizer(["[", "]", ",", "7", "1", "<eos>"])
    inner = _PermitEverything()
    grammar = maybe_wrap(
        inner,
        StructuredOutputOptions.JSON,
        json.dumps(CONTAINS_SEVEN),
        tokenizer=tokenizer,
        vocab_size=6,
    )
    assert isinstance(grammar, PostconditionGrammar)

    ids = {text: index for index, text in enumerate([b"[", b"]", b",", b"7", b"1", b""])}
    assert grammar.accept_tokens("r", [ids[b"["], ids[b"1"]]) is True
    assert grammar.accept_tokens("r", [ids[b"]"]]) is False
    assert grammar.accept_tokens("r", [ids[b","], ids[b"7"], ids[b"]"]]) is True


def test_a_vocabulary_that_cannot_spell_around_a_veto_is_left_unenforced():
    """Every veto here assumes the model can say the same bytes another way. A
    tokenizer with no bare `]` breaks that, and an over-broad veto stops being
    a detour and becomes a dead end -- so the layer declines rather than wall
    requests it cannot promise a way out of."""
    tokenizer = _FakeHFTokenizer(["[", "7]", ",", "7", "<eos>"])
    inner = _PermitEverything()
    assert maybe_wrap(inner, StructuredOutputOptions.JSON, json.dumps(CONTAINS_SEVEN), tokenizer, 5) is inner


def test_an_unenforceable_schema_that_slips_past_the_screen_is_not_half_enforced():
    """The screen owes this schema a rejection. If one arrives anyway, decoding
    it with `contains` enforced and `maxContains` silently dropped would be a
    worse answer than the 400 the caller was owed."""
    schema = {"type": "array", "contains": {"const": 7}, "maxContains": 2}
    inner = _PermitEverything()
    assert maybe_wrap(inner, StructuredOutputOptions.JSON, json.dumps(schema), None, VOCAB_SIZE) is inner


def test_a_rollback_past_the_retained_log_is_exact_rather_than_lost():
    """Nothing should rewind past its own speculative window, so this is the
    path that is not supposed to be taken. It still has to be right: resetting
    the scanner would leave it describing an empty document while the model has
    already emitted half an array, and every veto after that would be wrong."""
    grammar = build(TWO_SEVENS)
    grammar._max_rollback = 2
    emit(grammar, "[7,7,1,2,3")

    grammar.rollback(6)  # back to "[7,7"
    assert grammar._document.buf == bytearray(b"[7,7")
    assert "]" in allowed(grammar), "a satisfied array was walled after the rescan"

    grammar.rollback(2)  # back to "[7"
    assert grammar._document.buf == bytearray(b"[7")
    assert "]" not in allowed(grammar), "the second 7 was still counted after being rolled back"
