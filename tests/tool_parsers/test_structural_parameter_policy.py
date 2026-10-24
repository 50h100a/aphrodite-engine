# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
from jsonschema import Draft202012Validator
from openai.types.responses import FunctionTool, ToolChoiceFunction
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from aphrodite import envs
from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag
from aphrodite.tool_parsers.tool_strict_level import ToolStrictLevel
from aphrodite.tool_parsers.utils import get_json_schema_from_tools


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("strict", [None, False, True])
@pytest.mark.parametrize("level", list(ToolStrictLevel))
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_native_tool_format_survives_parameter_opt_out(monkeypatch, enabled, strict, level, responses, choice):
    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", enabled)
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = (
        FunctionTool(type="function", name="weather", parameters=parameters, strict=strict)
        if responses
        else ChatCompletionToolsParam(
            type="function", function={"name": "weather", "parameters": parameters, "strict": strict}
        )
    )
    tag = get_model_structural_tag("hermes", [tool], choice, False, strict_level=level)
    assert tag is not None
    grammar = Grammar.from_structural_tag(tag)

    def accepts(name, arguments):
        call = json.dumps({"name": name, "arguments": arguments})
        return _is_grammar_accept_string(grammar, f"<tool_call>\n{call}\n</tool_call>")

    assert accepts("weather", {"city": "Paris"})
    assert not accepts("undeclared", {"city": "Paris"})
    enforced = strict if strict is not None else enabled or level == ToolStrictLevel.PARAMETER
    assert accepts("weather", {"city": 42}) == (not enforced)
    assert accepts("weather", {}) == (not enforced)
    assert (tool.strict if responses else tool.function.strict) is strict


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("strict", [None, False, True])
@pytest.mark.parametrize("responses", [False, True])
def test_generic_named_tool_uses_same_parameter_default(monkeypatch, enabled, strict, responses):
    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", enabled)
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    if responses:
        tool = FunctionTool(type="function", name="weather", parameters=parameters, strict=strict)
        choice = ToolChoiceFunction(type="function", name="weather")
    else:
        tool = ChatCompletionToolsParam(
            type="function", function={"name": "weather", "parameters": parameters, "strict": strict}
        )
        choice = ChatCompletionNamedToolChoiceParam(function=ChatCompletionNamedFunction(name="weather"))
    schema = get_json_schema_from_tools(choice, [tool])
    validator = Draft202012Validator(schema)
    enforced = strict if strict is not None else enabled
    assert validator.is_valid({"city": 42}) == (not enforced)
    assert validator.is_valid({"city": "Paris"})
    assert parameters["properties"]["city"]["type"] == "string"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_hy_v4_keeps_native_tag_in_both_parameter_modes(monkeypatch, enabled, choice):
    from types import SimpleNamespace

    from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from aphrodite.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", enabled)
    parser = HYV4ToolParser.__new__(HYV4ToolParser)
    parser._extractor = SimpleNamespace(token_suffix="")
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        tool_choice=choice,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "strict": False,
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    )
    tag = parser.get_structural_tag(request)
    assert tag is not None
    Grammar.from_structural_tag(tag)
