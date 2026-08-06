# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A reply schema and a tool schema cannot both drive the decoder.

Sending `response_format` together with tools used to be resolved silently in
the tool's favour: the structural tag replaced `structured_outputs` and
`response_format` was set to None, with nothing said. The reply then came back
completely unconstrained -- for a `{"type": "string", "enum": [...]}` schema,
two tokens of prose that was not JSON and not in the enum -- and a caller
checking only that the reply parsed had no way to tell that its schema had
been discarded rather than honoured.

Something does have to give, since one decoder cannot run two grammars. The
answer is to say so.
"""

from unittest.mock import Mock

import pytest

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.engine.protocol import ResponseFormat
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.sampling_params import StructuredOutputsParams
from aphrodite.tool_parsers.abstract_tool_parser import reject_reply_schema_conflict

pytestmark = pytest.mark.cpu_test

ENUM_SCHEMA = {
    "name": "decision",
    "schema": {"type": "string", "enum": ["approve", "reject", "escalate"]},
    "strict": True,
}


def _request(**kwargs):
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kwargs)


def test_json_schema_reply_format_conflicts():
    request = _request(response_format={"type": "json_schema", "json_schema": ENUM_SCHEMA})

    with pytest.raises(AphroditeValidationError) as excinfo:
        reject_reply_schema_conflict(request)

    assert "response_format" in str(excinfo.value)
    assert excinfo.value.parameter == "response_format"


def test_json_object_reply_format_conflicts():
    request = _request(response_format={"type": "json_object"})

    with pytest.raises(AphroditeValidationError):
        reject_reply_schema_conflict(request)


def test_structured_outputs_conflicts():
    """The same constraint asked for directly rather than via response_format."""
    request = _request(structured_outputs=StructuredOutputsParams(json={"type": "object"}))

    with pytest.raises(AphroditeValidationError) as excinfo:
        reject_reply_schema_conflict(request)

    assert excinfo.value.parameter == "structured_outputs"


@pytest.mark.parametrize(
    "response_format",
    [
        None,
        {"type": "text"},  # the API default: constrains nothing
    ],
)
def test_unconstraining_reply_formats_do_not_conflict(response_format):
    """`response_format: {"type": "text"}` is what a client sends when it wants
    nothing in particular. Rejecting it would break every tool call that spells
    the default out."""
    reject_reply_schema_conflict(_request(response_format=response_format))


def test_responses_api_text_format_conflicts():
    request = Mock()
    request.text = Mock()
    request.text.format = {"type": "json_schema"}
    request.structured_outputs = None
    # Route on the real type rather than on hasattr, so patch the isinstance
    # check the function makes.
    from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest

    request.__class__ = ResponsesRequest

    with pytest.raises(AphroditeValidationError) as excinfo:
        reject_reply_schema_conflict(request)

    assert excinfo.value.parameter == "text.format"


def test_error_is_a_value_error_so_it_becomes_a_400():
    """`create_error_response` maps ValueError and its subclasses to 400; a
    different base would surface as a 500."""
    assert issubclass(AphroditeValidationError, ValueError)


def test_the_conflict_is_reported_not_resolved():
    """The regression guard: whatever else changes, the reply schema must not
    come back as None with the request still going through."""
    request = _request(response_format=ResponseFormat(type="json_schema", json_schema=ENUM_SCHEMA))

    with pytest.raises(AphroditeValidationError):
        reject_reply_schema_conflict(request)

    assert request.response_format is not None
