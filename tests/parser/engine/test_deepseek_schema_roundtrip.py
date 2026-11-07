# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar-valid native values survive blocking and incremental decoding."""

import json

import pytest
from jsonschema import Draft202012Validator
from xgrammar import Grammar, get_model_structural_tag
from xgrammar.testing import _is_grammar_accept_string

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionToolsParam
from aphrodite.parser.deepseek_v4 import DeepSeekV4Parser
from aphrodite.parser.deepseek_v41 import DeepSeekV41Parser
from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import collect_tool_arguments, simulate_tool_streaming


@pytest.mark.parametrize("version", ["v4", "v4_1"])
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("constraint", ["enum", "const", "reference"])
@pytest.mark.parametrize(
    "value",
    [
        " stable",
        "stable ",
        " stable ",
        "\tstable\t",
        "stable\n",
        "stable\r\n",
        "",
        "  ",
        "\u2003stable\u00a0",
        "null",
        "2",
    ],
)
def test_constrained_string_roundtrip(version, chunk_size, constraint, value):
    prop = {"type": "string", "const": value} if constraint == "const" else {"type": "string", "enum": [value]}
    schema = {"type": "object", "properties": {"value": prop}, "required": ["value"], "additionalProperties": False}
    if constraint == "reference":
        schema["$defs"] = {"value": prop}
        schema["properties"]["value"] = {"$ref": "#/$defs/value"}
    check_roundtrip(version, chunk_size, schema, value, True, value)


@pytest.mark.parametrize("version", ["v4", "v4_1"])
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize(
    ("prop", "raw", "string", "expected"),
    [
        ({"type": ["string", "null"], "enum": ["null"]}, "null", True, "null"),
        ({"anyOf": [{"type": "string", "const": "2"}, {"type": "integer", "minimum": 5}]}, "2", True, "2"),
        ({"type": ["string", "null"]}, "null", False, None),
        ({"type": "integer"}, " \r\n42\t ", False, 42),
        (
            {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
            '{"s":"2"}',
            False,
            {"s": "2"},
        ),
    ],
)
def test_native_type_flag_is_not_reinterpreted(version, chunk_size, prop, raw, string, expected):
    schema = {"type": "object", "properties": {"value": prop}, "required": ["value"], "additionalProperties": False}
    check_roundtrip(version, chunk_size, schema, raw, string, expected)


def check_roundtrip(version, chunk_size, schema, raw, string, expected):
    space = " " if version == "v4_1" else ""
    calls = "calls" if version == "v4_1" else "tool_calls"
    text = (
        f'\n\n<｜DSML｜{space}{calls}>\n<｜DSML｜{space}invoke name="record_value">\n'
        f'<｜DSML｜{space}parameter name="value" string="{str(string).lower()}">{raw}'
        f"</｜DSML｜{space}parameter>\n</｜DSML｜{space}invoke>\n</｜DSML｜{space}{calls}>"
    )
    tool = {"type": "function", "function": {"name": "record_value", "parameters": schema, "strict": True}}
    grammar = Grammar.from_structural_tag(get_model_structural_tag("deepseek_" + version, [tool], "required", False))
    assert _is_grammar_accept_string(grammar, text)
    tools = [ChatCompletionToolsParam(**tool)]
    request = ChatCompletionRequest(model="offline", messages=[], tools=tools)
    cls = DeepSeekV41Parser if version == "v4_1" else DeepSeekV4Parser
    parser = cls(make_mock_tokenizer({}), tools, chat_template_kwargs={"enable_thinking": False})
    if chunk_size is None:
        result = parser.extract_tool_calls(text, request).tool_calls[0].function.arguments
    else:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        deltas = simulate_tool_streaming(parser, request, chunks)
        final = parser.finish_streaming()
        result = collect_tool_arguments(deltas + [(final, text)])
    assert json.loads(result) == {"value": expected}
    Draft202012Validator(schema).validate(json.loads(result))
