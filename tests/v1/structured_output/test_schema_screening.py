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

from aphrodite.v1.structured_output.schema_features import (
    get_structured_outputs_schema_error,
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
