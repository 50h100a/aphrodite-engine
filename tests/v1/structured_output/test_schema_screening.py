# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One screen, both routes into the grammar.

A schema reaches the decoder two ways: as `response_format`, which lands in
`structured_outputs.json`, and as a tool, whose parameters are wrapped in a
structural tag first. They used to be screened unequally -- the unenforceable
keyword check read only `json` while the validity check read both -- so the
same `uniqueItems` schema was a 400 as a `response_format` and a silent 200 as
a tool, generating unconstrained arguments with duplicate elements.

`get_structured_outputs_schema_error` is now the single screen, shared by the
API entrypoint and the engine so the two layers cannot drift either.
"""

import json

import pytest

from aphrodite.v1.structured_output.postconditions import LAYER_ENFORCED_KEYWORDS
from aphrodite.v1.structured_output.schema_features import (
    _KEYWORD_BACKENDS,
    _UNENFORCEABLE_KEYWORDS,
    get_json_schema_backends_for_request,
    get_structured_outputs_schema_error,
    get_unenforceable_json_schema_keys,
    iter_request_json_schemas,
)

pytestmark = pytest.mark.cpu_test

UNENFORCEABLE = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
MALFORMED = {"type": "strng"}
ORDINARY = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


class FakeStructuredOutputs:
    """Stands in for StructuredOutputsParams: the screen reads two attributes."""

    def __init__(self, json=None, structural_tag=None):
        self.json = json
        self.structural_tag = structural_tag


def _legacy_tag(schema):
    """The `structures` spelling: what a tool call is wrapped in."""
    return json.dumps(
        {
            "structures": [{"begin": "<tool>", "schema": schema, "end": "</tool>"}],
            "triggers": ["<tool>"],
        }
    )


def _nested_tag(schema):
    """The newer `format` spelling of the same thing."""
    return json.dumps(
        {
            "format": {
                "type": "any_text",
                "elements": [{"type": "json_schema", "json_schema": schema}],
            }
        }
    )


@pytest.mark.parametrize("wrap", [_legacy_tag, _nested_tag], ids=["structures", "format"])
def test_unenforceable_keyword_is_caught_through_a_tool(wrap):
    """The gap: this was a 200 with duplicate elements in the arguments."""
    error = get_structured_outputs_schema_error(FakeStructuredOutputs(structural_tag=wrap(UNENFORCEABLE)))

    assert error is not None
    assert "uniqueItems" in error


def test_unenforceable_keyword_is_caught_through_response_format():
    error = get_structured_outputs_schema_error(FakeStructuredOutputs(json=UNENFORCEABLE))

    assert error is not None
    assert "uniqueItems" in error


def test_both_routes_give_the_same_verdict():
    """The point of the unification: one schema, one answer, whichever way in."""
    as_reply = get_structured_outputs_schema_error(FakeStructuredOutputs(json=UNENFORCEABLE))
    as_tool = get_structured_outputs_schema_error(FakeStructuredOutputs(structural_tag=_legacy_tag(UNENFORCEABLE)))

    assert as_reply == as_tool


def test_malformed_schema_is_caught_through_a_tool():
    error = get_structured_outputs_schema_error(FakeStructuredOutputs(structural_tag=_legacy_tag(MALFORMED)))

    assert error is not None
    assert "not valid" in error


def test_validity_is_reported_before_enforceability():
    """The keywords of a malformed schema mean nothing, so that is the more
    useful complaint of the two."""
    broken_and_unenforceable = {"type": "strng", "uniqueItems": True}

    error = get_structured_outputs_schema_error(FakeStructuredOutputs(json=broken_and_unenforceable))

    assert error is not None
    assert "not valid" in error


def test_a_bad_tool_is_caught_even_when_the_reply_schema_is_fine():
    error = get_structured_outputs_schema_error(
        FakeStructuredOutputs(json=ORDINARY, structural_tag=_legacy_tag(UNENFORCEABLE))
    )

    assert error is not None
    assert "uniqueItems" in error


@pytest.mark.parametrize(
    "structured_outputs",
    [
        None,
        FakeStructuredOutputs(),
        FakeStructuredOutputs(json=ORDINARY),
        FakeStructuredOutputs(structural_tag=_legacy_tag(ORDINARY)),
        FakeStructuredOutputs(json=ORDINARY, structural_tag=_legacy_tag(ORDINARY)),
        # Not a schema route at all, and not ours to complain about.
        FakeStructuredOutputs(structural_tag="not json at all"),
    ],
)
def test_enforceable_requests_pass(structured_outputs):
    assert get_structured_outputs_schema_error(structured_outputs) is None


def test_every_schema_a_request_carries_is_offered_to_the_screen():
    """If a route stops being enumerated here, the screen silently stops
    covering it -- which is exactly how the tool route was missed."""
    schemas = list(
        iter_request_json_schemas(
            FakeStructuredOutputs(json=ORDINARY, structural_tag=_legacy_tag(UNENFORCEABLE)),
        )
    )

    assert schemas == [ORDINARY, UNENFORCEABLE]


# --------------------------------------------------------------------------
# Best effort: APHRODITE_STRUCTURED_OUTPUT_BEST_EFFORT=1
#
# The default answer to "nothing can enforce this" is a 400, on the grounds that
# accepting the schema would promise a constraint no one applies. For a caller
# who would rather have the request than the promise, the flag turns the
# rejection into a removal: the keyword comes out of the schema, everything else
# in it is still enforced, and the caller validates the rest afterwards.
# --------------------------------------------------------------------------


@pytest.fixture
def best_effort(monkeypatch):
    monkeypatch.setenv("APHRODITE_STRUCTURED_OUTPUT_BEST_EFFORT", "1")


@pytest.mark.parametrize("wrap", [_legacy_tag, _nested_tag], ids=["structures", "format"])
def test_best_effort_accepts_a_tool_and_drops_the_keyword(best_effort, wrap):
    structured_outputs = FakeStructuredOutputs(structural_tag=wrap(UNENFORCEABLE))

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert "uniqueItems" not in structured_outputs.structural_tag


def test_best_effort_accepts_a_reply_schema_and_drops_the_keyword(best_effort):
    structured_outputs = FakeStructuredOutputs(json=dict(UNENFORCEABLE))

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert structured_outputs.json == {"type": "array", "items": {"type": "string"}}


def test_best_effort_keeps_the_rest_of_the_schema(best_effort):
    """Best effort, not no effort: what is left is still enforced."""
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
        "required": ["tags"],
        "additionalProperties": False,
    }
    structured_outputs = FakeStructuredOutputs(json=schema)

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert structured_outputs.json == {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
        "additionalProperties": False,
    }


def test_best_effort_leaves_the_request_routable(best_effort):
    """The point of editing the request rather than just staying quiet: routing
    asks this question the moment the screen returns, and a keyword left behind
    for it to find would answer 'nobody' and fail the request one layer down."""
    structured_outputs = FakeStructuredOutputs(json=dict(UNENFORCEABLE))

    get_structured_outputs_schema_error(structured_outputs)

    assert get_json_schema_backends_for_request(structured_outputs)


def test_best_effort_does_not_touch_the_caller_s_schema(best_effort):
    """A tool's parameters dict belongs to whoever built the tool."""
    schema = dict(UNENFORCEABLE)

    get_structured_outputs_schema_error(FakeStructuredOutputs(json=schema))

    assert schema == UNENFORCEABLE


def test_best_effort_returns_json_text_as_json_text(best_effort):
    structured_outputs = FakeStructuredOutputs(json=json.dumps(UNENFORCEABLE))

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert json.loads(structured_outputs.json) == {"type": "array", "items": {"type": "string"}}


def test_best_effort_still_rejects_a_malformed_schema(best_effort):
    """Not a gap in what can be enforced -- a schema that does not mean
    anything. The flag says how much to enforce, not whether to read it."""
    error = get_structured_outputs_schema_error(FakeStructuredOutputs(json=MALFORMED))

    assert error is not None
    assert "not valid" in error


def test_best_effort_still_rejects_a_backend_conflict(best_effort):
    """Both keywords are enforceable; they just have no backend in common.
    Deciding which one to break would be answering a question the caller never
    asked, so this stays a 400."""
    schema = {
        "type": "object",
        "propertyNames": {"pattern": "^a"},
        "properties": {"x": {"type": "number", "multipleOf": 2}},
    }

    error = get_structured_outputs_schema_error(FakeStructuredOutputs(json=schema))

    assert error is not None
    assert "cannot be enforced together" in error


def test_best_effort_leaves_an_inert_keyword_alone(best_effort):
    """`uniqueItems: false` is not why anything was ever rejected, so there is
    nothing here to relax."""
    schema = {"type": "array", "uniqueItems": False}
    structured_outputs = FakeStructuredOutputs(json=schema)

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert structured_outputs.json == schema


def test_best_effort_drops_a_conditional_group_whole(best_effort):
    """`if` constrains only in the company of `then`/`else`, and `contains` is
    unenforceable only in the company of `maxContains`. Dropped one at a time,
    whichever went first would make the rest look inert and spare them, leaving
    a schema that still carries a keyword nothing can enforce."""
    schema = {
        "type": "array",
        "items": {"type": "string"},
        "if": {"minItems": 1},
        "then": {"maxItems": 3},
        "else": {"maxItems": 9},
        "contains": {"const": "x"},
        "maxContains": 2,
    }
    structured_outputs = FakeStructuredOutputs(json=schema)

    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert structured_outputs.json == {"type": "array", "items": {"type": "string"}}


# Derived, not listed, so a keyword that gains an enforcer drops out on its own.
SCREENED_OUT = sorted(_UNENFORCEABLE_KEYWORDS - LAYER_ENFORCED_KEYWORDS)

# Groups whose members are only meaningful together. `if`/`then`/`else` is
# symmetric. The `contains` group is directional: `minContains`/`maxContains`
# mean nothing alone, while `contains` is well-defined by itself, which is why
# this group can be split where the other cannot.
SYMMETRIC_GROUPS = [{"if", "then", "else"}]
DEPENDENTS = {"minContains": "contains", "maxContains": "contains"}


def test_a_companion_group_is_never_split():
    """What makes dropping a group safe: nothing is left behind holding a
    reference to something that left.

    Symmetric groups are all-in or all-out. A dependent keyword may not outlive
    what it modifies, but the reverse is fine -- which is what lets `contains`
    be enforced while `maxContains` is refused.
    """
    for group in SYMMETRIC_GROUPS:
        present = group & _UNENFORCEABLE_KEYWORDS
        assert present in (set(), group), f"{sorted(group)} is split across the enforceability line"

    for dependent, needed in DEPENDENTS.items():
        if needed in _UNENFORCEABLE_KEYWORDS:
            assert dependent in _UNENFORCEABLE_KEYWORDS, (
                f"{needed} is unenforceable but {dependent} is not, which would leave {dependent} "
                "modifying a keyword that has been stripped out from under it"
            )

    assert _UNENFORCEABLE_KEYWORDS == {key for key, backends in _KEYWORD_BACKENDS.items() if not backends}


@pytest.mark.parametrize("keyword", SCREENED_OUT)
def test_best_effort_covers_every_screened_out_keyword(best_effort, keyword):
    """Whatever the screen would refuse, the flag has to be able to remove.
    A keyword the strip missed would be a 400 the flag promised to prevent,
    surfacing later as an unroutable request rather than a clear message."""
    schema = {
        "type": "array",
        "items": {"type": "string"},
        "if": {"minItems": 1},
        "then": {"maxItems": 3},
        "else": {"maxItems": 9},
        "contains": {"const": "x"},
        "minContains": 1,
        "maxContains": 2,
        "not": {"minItems": 2},
        "uniqueItems": True,
        "dependentRequired": {"a": ["b"]},
        "dependentSchemas": {"a": {"required": ["b"]}},
    }
    only_this_one = {key: value for key, value in schema.items() if key not in _UNENFORCEABLE_KEYWORDS}
    # Whatever the keyword under test needs to mean anything comes with it.
    for group in (*SYMMETRIC_GROUPS, {keyword, DEPENDENTS.get(keyword, keyword)}):
        if keyword in group:
            only_this_one.update({key: schema[key] for key in group})
    only_this_one[keyword] = schema[keyword]

    assert get_unenforceable_json_schema_keys(only_this_one), f"{keyword} is not screened for"

    structured_outputs = FakeStructuredOutputs(json=only_this_one)
    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert get_unenforceable_json_schema_keys(structured_outputs.json) == []


@pytest.mark.parametrize("keyword", sorted(LAYER_ENFORCED_KEYWORDS))
def test_best_effort_leaves_the_enforced_keywords_alone(best_effort, keyword):
    """Best-effort drops what cannot be enforced and nothing else; stripping an
    enforced keyword would quietly widen the caller's schema."""
    if keyword == "uniqueItems":
        schema = {"type": "array", "items": {"enum": ["x", "y"]}, "uniqueItems": True}
    else:
        schema = {"type": "array", "items": {"type": "string"}, "contains": {"const": "x"}}
        if keyword == "minContains":
            schema["minContains"] = 2

    structured_outputs = FakeStructuredOutputs(json=dict(schema))
    assert get_structured_outputs_schema_error(structured_outputs) is None
    assert structured_outputs.json == schema


def test_best_effort_drops_a_keyword_that_is_enforceable_elsewhere(best_effort):
    """`uniqueItems` is enforced where the items come from a known set and
    refused where they do not, so the strip has to be as position-dependent as
    the screen. A keyword left behind would fail routing one layer down, which
    is the 400 the flag promised to prevent."""
    schema = {
        "type": "object",
        "properties": {
            "picked": {"type": "array", "items": {"enum": ["x", "y"]}, "uniqueItems": True},
            "notes": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        },
    }
    structured_outputs = FakeStructuredOutputs(json=schema)

    assert get_structured_outputs_schema_error(structured_outputs) is None
    kept = structured_outputs.json["properties"]
    assert kept["picked"]["uniqueItems"] is True
    assert "uniqueItems" not in kept["notes"]


def test_a_serveable_uniqueitems_schema_is_refused_as_a_tool():
    """The asymmetry the layer cannot help: a tool's parameters ride a
    structural tag, and the scanner would have to find the array inside the
    tag's own output to know where the items start."""
    schema = {"type": "array", "items": {"enum": ["x", "y"]}, "uniqueItems": True}
    assert get_unenforceable_json_schema_keys(schema) == []
    assert get_unenforceable_json_schema_keys(schema, postconditions_available=False) == ["uniqueItems"]
