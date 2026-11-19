# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.parser import ParserManager
from aphrodite.v1.structured_output.schema_features import (
    get_backends_for_request,
    get_structured_outputs_schema_error,
)
from tests.parser.engine.conftest import make_mock_tokenizer


def adjust(request):
    cls = ParserManager.get_parser(
        tool_parser_name="openai",
        reasoning_parser_name="openai_gptoss",
        is_harmony=True,
        enable_auto_tools=True,
        model_name="openai/gpt-oss-120b",
    )
    return cls(make_mock_tokenizer({}), request.tools).adjust_request(request)


@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("strict", [True, False])
def test_none_constrains_headers_independently_of_tool_strictness(route, strict):
    function = {"name": "weather", "parameters": {"type": "object"}, "strict": strict}
    request = (
        ChatCompletionRequest(
            model="offline", messages=[], tools=[{"type": "function", "function": function}], tool_choice="none"
        )
        if route == "chat"
        else ResponsesRequest(
            model="offline", input="weather?", tools=[{"type": "function", **function}], tool_choice="none"
        )
    )
    params = adjust(request).extract_structured_outputs()
    assert get_structured_outputs_schema_error(params) is None
    grammar = Grammar.from_structural_tag(params.structural_tag)
    final = "<|channel|>final<|message|>The weather is unknown."
    assert _is_grammar_accept_string(grammar, final)
    assert _is_grammar_accept_string(
        grammar,
        "<|channel|>analysis<|message|>Think first.<|end|><|start|>assistant" + final,
    )
    for recipient in ("functions.weather", "browser", "python"):
        for header in (f" to={recipient}<|channel|>commentary", f"<|channel|>analysis to={recipient}"):
            assert not _is_grammar_accept_string(grammar, header + "<|message|>{}")


@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("choice", ["none", None])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "contains": {"const": 7}},
        {"type": "array", "contains": {"const": 7}, "minContains": 2},
        {"type": "array", "items": {"enum": ["a", "b"]}, "uniqueItems": True},
    ],
)
def test_final_json_preserves_postcondition_capabilities(route, choice, schema):
    fmt = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}}
    selection = {"tool_choice": choice} if choice is not None else {}
    request = (
        ChatCompletionRequest(model="offline", messages=[], response_format=fmt, **selection)
        if route == "chat"
        else ResponsesRequest(
            model="offline",
            input="test",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": schema,
                    "strict": True,
                }
            },
            **selection,
        )
    )
    params = adjust(request).extract_structured_outputs()
    assert get_structured_outputs_schema_error(params) is None
    assert "xgrammar" in get_backends_for_request(params)
    assert json.loads(params.structural_tag)["format"]["type"] == "sequence"
