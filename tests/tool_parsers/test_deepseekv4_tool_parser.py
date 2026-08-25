# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4EngineToolParser."""

import json
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from aphrodite.parser.deepseek_v4 import _dsml_arg_converter
from aphrodite.tool_parsers import ToolParserManager
from aphrodite.tool_parsers.deepseekv4_engine_tool_parser import (
    DeepSeekV4EngineToolParser,
)

pytestmark = pytest.mark.skip_global_cleanup

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def make_parser(tools=None) -> DeepSeekV4EngineToolParser:
    return DeepSeekV4EngineToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items())
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(parser: DeepSeekV4EngineToolParser, full_text: str, chunk_size: int = 7):
    deltas = []
    previous_text = ""
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=make_request(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.index == tool_index and tool_call.function and tool_call.function.arguments:
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4EngineToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call("get_weather", {"location": "Beijing", "unit": "celsius"})

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_block_is_not_accepted():
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "aphrodite"}).replace("tool_calls", "function_calls")

    result = parser.extract_tool_calls(model_output, make_request())

    assert not result.tools_called
    assert result.content == model_output


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}


def test_streaming_emits_incremental_argument_chunks():
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="plan_trip",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        ),
    )
    parser = make_parser(tools=[tool])
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}plan_trip">\n'
        f'{PARAM_START}days" string="false">3{PARAM_END}\n'
        f'{PARAM_START}flexible" string="false">false{PARAM_END}\n'
        f'{PARAM_START}cities" string="false">'
        f'["Beijing","Shanghai","Tokyo","New York"]{PARAM_END}\n'
        f'{PARAM_START}notes" string="true">靠窗座位{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = stream(parser, full_text, chunk_size=4)
    arg_chunks = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert len([chunk for chunk in arg_chunks if chunk]) > 2
    assert json.loads("".join(arg_chunks)) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }


def _with_strict(
    tools: list[ChatCompletionToolsParam],
) -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type=t.type,
            function=FunctionDefinition(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
                strict=True,
            ),
        )
        for t in tools
    ]


def test_get_aphrodite_registry_structural_tag_returns_structural_tag(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser()
    strict_tools = _with_strict(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=strict_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if sample_tools:
        tool = sample_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=sample_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


def _invoked_tool_names(node, found: list[str] | None = None) -> list[str]:
    """Every tool name the tag's invoke tags can open, in order.

    Read off the structure rather than the serialized text: the DSML markers
    carry quotes, which JSON escapes, and a substring check against the escaped
    form is one backslash away from passing vacuously.
    """
    found = [] if found is None else found
    if isinstance(node, dict):
        begin = node.get("begin")
        if isinstance(begin, str) and begin.startswith(INV_START):
            found.append(begin[len(INV_START) :].split('"', 1)[0])
        for value in node.values():
            _invoked_tool_names(value, found)
    elif isinstance(node, list):
        for value in node:
            _invoked_tool_names(value, found)
    return found


def test_auto_tool_choice_constrains_names_without_strict(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    """Plain "auto" is tagged too, and the tag lists the declared tools.

    `strict` decides whether a tool's *arguments* are schema-enforced. It never
    decided which tools exist -- but "auto" used to be skipped without it,
    leaving the model free to invoke a name nobody declared.
    """
    parser = make_parser()
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="auto",
    )

    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    invoked = _invoked_tool_names(tag.model_dump())
    assert invoked == [tool.function.name for tool in sample_tools]


def test_extract_tool_calls_arguments_wrapper():
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )

    parser = DeepSeekV4EngineToolParser(mock_tokenizer, tools=[tool])
    request = MagicMock()
    request.tools = [tool]

    model_output = (
        f"{TC_START}"
        f'{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">{{"location":"Beijing"}}{PARAM_END}'
        f"{INV_END}"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"location": "Beijing"}


_ANGLE_BRACKET_TOOL = ChatCompletionToolsParam(
    function=FunctionDefinition(
        name="run_command",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
        },
    ),
)


@pytest.mark.parametrize(
    "tools",
    [[_ANGLE_BRACKET_TOOL], None],
    ids=["with_tools", "without_tools"],
)
def test_no_dsml_closing_tag_leak_in_streamed_args(tools):
    """Streaming must not leak </｜DSML｜parameter> into argument values."""
    full_text = build_tool_call("run_command", {"command": "git --version 2>&1"})
    expected = {"command": "git --version 2>&1"}

    for chunk_size in range(1, len(full_text) + 1):
        parser = make_parser(tools=tools)
        deltas = stream(parser, full_text, chunk_size=chunk_size)
        args_str = reconstruct_args(deltas)
        assert args_str, f"No args emitted at chunk_size={chunk_size}"
        assert "DSML" not in args_str, f"DSML marker leaked into args at chunk_size={chunk_size}: {args_str!r}"
        parsed = json.loads(args_str)
        assert parsed == expected, f"Args mismatch at chunk_size={chunk_size}: got {parsed!r}, expected {expected!r}"


def test_non_streaming_extract_with_angle_brackets():
    """Non-streaming extraction must correctly handle '>' in values."""
    parser = make_parser()
    full_text = build_tool_call("run_command", {"command": "git --version 2>&1"})
    result = parser.extract_tool_calls(full_text, make_request())

    assert result.tools_called
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"command": "git --version 2>&1"}
    assert "DSML" not in result.tool_calls[0].function.arguments


def test_composed_schema_converts_object_and_array_params():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "wait": {
                        "anyOf": [
                            {"type": "object"},
                            {"type": "null"},
                        ],
                    },
                    "patches": {
                        "allOf": [
                            {"type": "array", "items": {"type": "object"}},
                        ],
                    },
                },
            },
        },
    )
    parser = make_parser(tools=[tool])
    request = make_request(tools=[tool])
    model_output = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }


# xgrammar's `deepseek_xml` style writes each parameter into a slot that permits
# `[ \n\t]*` on either side of the value, so a model generating under the tool
# grammar can indent its value and never leave the grammar. Reproduced live: a
# `severity` enum came back as " \tsev2", which is not one of its four members.
PADDED_ENUM_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "file_report",
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["sev1", "sev2"]},
                "title": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["severity", "title", "count"],
        },
    },
)


def _padded_output() -> str:
    return (
        f"{TC_START}\n"
        f'{INV_START}file_report">\n'
        f'{PARAM_START}severity" string="true"> \tsev2{PARAM_END}\n'
        f'{PARAM_START}title" string="true">\t indented on purpose {PARAM_END}\n'
        f'{PARAM_START}count" string="true">\n  7\n{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )


def test_grammar_padding_is_not_folded_into_the_value():
    """Padding comes off wherever the schema says it cannot be part of a value.

    A free string keeps it: `xml_string` matches the padding as readily as the
    value, so there is no telling one from the other, and a caller asking for a
    file's contents is entitled to its leading tab.
    """
    parser = make_parser(tools=[PADDED_ENUM_TOOL])
    result = parser.extract_tool_calls(_padded_output(), make_request(tools=[PADDED_ENUM_TOOL]))

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["severity"] == "sev2"
    assert args["title"] == "\t indented on purpose "
    # `string="true"` on an integer is inside the grammar too -- the flag is a
    # free choice there, so the declared type decides, not the model.
    assert args["count"] == 7


def test_grammar_padding_is_kept_when_the_tool_is_unknown():
    """With no schema in hand there is nothing to justify editing the value."""
    parser = make_parser()
    result = parser.extract_tool_calls(_padded_output(), make_request())

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["severity"] == " \tsev2"


def test_streamed_padded_value_matches_the_whole_response():
    parser = make_parser(tools=[PADDED_ENUM_TOOL])
    whole = make_parser(tools=[PADDED_ENUM_TOOL]).extract_tool_calls(
        _padded_output(), make_request(tools=[PADDED_ENUM_TOOL])
    )
    deltas = stream(parser, _padded_output())

    streamed = reconstruct_args(deltas)
    assert json.loads(streamed) == json.loads(whole.tool_calls[0].function.arguments)


def test_partial_literals_are_withheld_until_they_parse():
    """A parameter must not be published and then rewritten.

    Mid-literal, `1.` is not a number. Rendering the fragment as the string
    "1." only to replace it with 1.5 a token later would make a streamed
    argument delta retract what an earlier one already sent, so the fragment is
    withheld instead -- the same thing the parser has always done for
    `string="false"`, now also for a value the schema types as a number.
    """
    properties = {"factor": {"type": "number"}, "mode": {"type": "string", "enum": ["fast", "slow"]}}
    raw = f'{PARAM_START}factor" string="false"> 1.5'

    seen = [json.loads(_dsml_arg_converter(raw[:n], True, properties)) for n in range(1, len(raw) + 1)]

    assert all(not isinstance(step.get("factor"), str) for step in seen), seen
    assert seen[-1] == {"factor": 1.5}
