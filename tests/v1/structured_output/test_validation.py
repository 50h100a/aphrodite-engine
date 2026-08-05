# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

import json

import pytest

from aphrodite.config import StructuredOutputsConfig
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams

pytestmark = pytest.mark.cpu_test

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "customer": {"type": "string"},
    },
    "required": ["invoice_id", "customer"],
    "additionalProperties": False,
}


class _StubModelConfig:
    def __init__(self, is_diffusion: bool):
        self.is_diffusion = is_diffusion


def test_structured_outputs_rejected_for_diffusion_models():
    """Diffusion LLMs denoise the canvas in parallel, which is incompatible
    with the token-by-token grammar FSM. The request must fail with a clear
    validation error instead of an FSM rejection mid-generation (#45436)."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
    with pytest.raises(ValueError, match="not yet supported for diffusion"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=True),
            StructuredOutputsConfig(),
            tokenizer=None,
        )


def test_plain_request_allowed_for_diffusion_models():
    """Requests without structured outputs are unaffected by the guard."""
    params = SamplingParams()
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=True),
        StructuredOutputsConfig(),
        tokenizer=None,
    )


@pytest.mark.parametrize(
    "structured_outputs, match",
    [
        (StructuredOutputsParams(json_object=False), "json_object must be True"),
        (StructuredOutputsParams(json=""), "json cannot be an empty string"),
    ],
)
def test_degenerate_structured_outputs_rejected(structured_outputs, match):
    """json_object=False and an empty json schema pass the `is not None`
    exclusivity check but resolve to no structured-output key, so they must be
    rejected at request validation (-> 400) instead of reaching and crashing
    the engine."""
    params = SamplingParams(structured_outputs=structured_outputs)
    with pytest.raises(ValueError, match=match):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": []},
        {"type": "object", "properties": {"mode": {"type": "strng"}}},
        {"type": "string", "minLength": "five"},
    ],
)
def test_malformed_schema_rejected(schema):
    """Backends disagree about malformed schemas -- some raise, some drop the
    keyword and leave the constraint silently unenforced -- so a schema that is
    not valid JSON Schema is rejected up front, whichever backend `auto` would
    have picked."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    with pytest.raises(ValueError, match="JSON schema is not valid"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )


def test_malformed_tool_schema_rejected():
    """Tool calls carry their schemas inside a structural tag, so screening
    that only looked at `json` would miss every tool-constrained request."""
    tag = {
        "structures": [{"begin": "<f>", "schema": {"anyOf": []}, "end": "</f>"}],
        "triggers": ["<f>"],
    }
    params = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=json.dumps(tag)))
    with pytest.raises(ValueError, match="JSON schema is not valid"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )


@pytest.mark.parametrize(
    "subschema",
    [
        {"enum": []},
        {"type": "string", "minLength": 5, "maxLength": 3},
        {"allOf": [{"type": "string"}, {"type": "integer"}]},
    ],
)
def test_unsatisfiable_optional_property_accepted(subschema):
    """Nothing satisfies these, but they sit on an optional property, so the
    schema as a whole still describes a non-empty language ("objects without
    `mode`") and generation is well defined. Deciding satisfiability in general
    is intractable, so they are admitted; a request that does walk into the
    dead branch ends with finish_reason="constraint" at runtime."""
    schema = {"type": "object", "properties": {"mode": subschema, "ok": {"type": "integer"}}}
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="guidance"),
        tokenizer=object(),
    )


@pytest.mark.parametrize("schema", [{"enum": []}, {"type": "string", "minLength": 5, "maxLength": 3}])
def test_unsatisfiable_root_still_rejected_by_backend(schema):
    """When the *whole* schema admits nothing, no token can ever be emitted, so
    there is no partial response to preserve and a 400 is the right answer.
    llguidance already detects this; the point of the test is that our
    validation does not get in the way of it."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    with pytest.raises(ValueError, match="[Uu]nsatisfiable"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(backend="guidance"),
            tokenizer=object(),
        )
