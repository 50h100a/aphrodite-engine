# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A successful stock XGrammar compile must not weaken the accepted schema."""

import json

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from aphrodite.exceptions import AphroditeValidationError
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
from aphrodite.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
from aphrodite.v1.structured_output.xgrammar_schema import get_xgrammar_schema_error

pytestmark = pytest.mark.cpu_test

BAD = [
    {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 10}]},
    {"type": "integer", "enum": [1, 2], "minimum": 2},
    {"const": "x", "minLength": 3},
    {"$ref": "#/$defs/v", "minimum": 5, "$defs": {"v": {"type": "integer", "minimum": 0}}},
    {"anyOf": [{"type": "integer"}], "minimum": 5},
    {"type": "string", "pattern": "^[a-z]+$", "minLength": 3},
    {"type": "string", "format": "uuid", "maxLength": 3},
    {"type": "string", "format": "uuid", "pattern": "^x$"},
    {"oneOf": [{"type": "number"}, {"type": "integer"}]},
    {"oneOf": [{"const": 1}, {"enum": [1.0, 2]}]},
    {"oneOf": [True, {"type": "string"}]},
    {"$dynamicRef": "#node"},
]


def params(schema, route):
    if route == "json":
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    tag = (
        {"structures": [{"begin": "<tool>", "schema": schema, "end": "</tool>"}], "triggers": ["<tool>"]}
        if route == "legacy"
        else {
            "format": {
                "type": "tag",
                "begin": "<tool>",
                "end": "</tool>",
                "content": {"type": "json_schema", "json_schema": schema},
            }
        }
    )
    return SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=json.dumps(tag)))


@pytest.mark.parametrize("schema", BAD)
@pytest.mark.parametrize("route", ["json", "legacy", "modern"])
def test_silent_weakening_is_rejected_on_every_schema_route(schema, route):
    assert get_xgrammar_schema_error(schema)
    with pytest.raises(AphroditeValidationError, match="XGrammar cannot enforce"):
        validate_xgrammar_grammar(params(schema, route))


@pytest.mark.parametrize(
    ("schema", "sample"),
    [
        ({"type": "string", "enum": [" stable "]}, " stable "),
        ({"type": "integer", "pattern": "^x$", "minLength": 3}, 2),
        ({"type": "integer", "enum": [2, 3], "minimum": 2}, 2),
        ({"const": "abc", "type": "string", "minLength": 2}, "abc"),
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, 2),
        ({"oneOf": [{"const": True}, {"const": 1}]}, True),
        (
            {
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"a": {"const": "A"}},
                        "required": ["a"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"b": {"const": 2}},
                        "required": ["b"],
                        "additionalProperties": False,
                    },
                ],
            },
            {"b": 2},
        ),
        (
            {
                "oneOf": [
                    {"type": "object", "properties": {"kind": {"const": "a"}}, "required": ["kind"]},
                    {"type": "object", "properties": {"kind": {"const": "b"}}, "required": ["kind"]},
                ]
            },
            {"kind": "a"},
        ),
        ({"$ref": "#/$defs/v", "description": "reference", "$defs": {"v": {"type": "integer"}}}, 2),
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$ref": "#/definitions/v",
                "minimum": 5,
                "definitions": {"v": {"type": "integer"}},
            },
            2,
        ),
        ({"type": "integer", "$defs": {"unused": {"enum": [1, 2], "minimum": 2}}}, 1),
    ],
)
def test_supported_and_provably_redundant_constraints_remain_usable(schema, sample):
    assert get_xgrammar_schema_error(schema) is None
    validate_xgrammar_grammar(params(schema, "json"))
    assert _is_grammar_accept_string(Grammar.from_json_schema(schema), json.dumps(sample))


@pytest.mark.parametrize("constraint", [{"minLength": 3, "maxLength": 5}, {"pattern": "^[a-z]+$"}])
def test_unknown_format_annotation_does_not_mask_a_tool_string_constraint(constraint):
    schema = {"type": "string", "format": "opaque", **constraint}
    assert get_xgrammar_schema_error(schema) is None
    validate_xgrammar_grammar(params(schema, "modern"))
    assert _is_grammar_accept_string(Grammar.from_json_schema(schema), '"abc"')


def test_referenced_intermediate_assertions_are_not_skipped():
    schema = {"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/b", "minimum": 5}, "b": {"type": "integer"}}}
    assert "sibling" in get_xgrammar_schema_error(schema)


def test_recursive_reference_terminates_without_losing_reachable_checks():
    schema = {"type": "object", "properties": {"next": {"$ref": "#"}, "v": {"type": "integer"}}}
    assert get_xgrammar_schema_error(schema) is None
    schema["properties"]["v"] = {"enum": [1, 2], "minimum": 2}
    assert "sibling" in get_xgrammar_schema_error(schema)


def test_sibling_proof_never_executes_a_user_regex(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute caller regex")

    monkeypatch.setitem(__import__("jsonschema").Draft202012Validator.VALIDATORS, "pattern", forbidden)
    assert get_xgrammar_schema_error({"enum": ["a"], "pattern": "(a+)+b"})
