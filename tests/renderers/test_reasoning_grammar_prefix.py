# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The reasoning form of a tool structural tag has to match the prompt.

xgrammar's reasoning form makes the segment *mandatory*: the tool tags sit
behind a `...</think>` prefix that the model has to generate before the grammar
will constrain anything. DeepSeek V4's chat-mode template hands the model an
already-closed `</think>`, so the model generates no end marker at all -- the
prefix is never satisfied, the whole reply falls through the leading free-text
span, and the tool name is unconstrained. That is how a call to a tool nobody
declared got out.

Thinking mode is the mirror image: there the model *does* write the marker, and
dropping the prefix would forbid it, killing every legitimate tool call. So the
flag has to track the resolved thinking mode in both directions, which is what
these tests pin.
"""

import json
from unittest.mock import MagicMock

import pytest

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.parser.parser_manager import ParserManager

pytestmark = pytest.mark.cpu_test

THINK_END = "</think>"


@pytest.fixture
def tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={  # type: ignore[arg-type]
                "name": "get_weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={  # type: ignore[arg-type]
                "name": "calculate_area",
                "parameters": {
                    "type": "object",
                    "properties": {"radius": {"type": "number"}},
                },
            },
        ),
    ]


def _parser(tools: list[ChatCompletionToolsParam], chat_template_kwargs: dict):
    """The composed parser the serving layer would build for DeepSeek V4."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v4",
        reasoning_parser_name="deepseek_v4",
        enable_auto_tools=True,
    )
    assert parser_cls is not None

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return parser_cls(tokenizer, tools, chat_template_kwargs=chat_template_kwargs)  # type: ignore[arg-type]


def _adjusted_tag(
    tools: list[ChatCompletionToolsParam],
    chat_template_kwargs: dict,
    tool_choice: str = "auto",
) -> dict | None:
    """The structural tag the render path installs, or None if it installs none.

    Goes through `adjust_request` rather than calling the tag builder directly,
    so the thinking mode reaching the grammar is the one the request resolved
    to -- the pairing that was wrong is exactly the one under test.
    """
    parser = _parser(tools, chat_template_kwargs)

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=tools,
        tool_choice=tool_choice,
    )
    parser.adjust_request(request)

    structured_outputs = request.structured_outputs
    if structured_outputs is None or structured_outputs.structural_tag is None:
        return None
    return json.loads(structured_outputs.structural_tag)


def _tag_for(tools: list[ChatCompletionToolsParam], chat_template_kwargs: dict) -> dict:
    tag = _adjusted_tag(tools, chat_template_kwargs)
    assert tag is not None, "expected a structural tag to be installed"
    return tag


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [{}, {"thinking": False}, {"thinking": True, "reasoning_effort": "none"}],
)
def test_chat_mode_tag_has_no_reasoning_prefix(
    tools: list[ChatCompletionToolsParam],
    chat_template_kwargs: dict,
):
    """No `</think>` is coming, so nothing may be waiting on one."""
    tag = _tag_for(tools, chat_template_kwargs)

    # The tool trigger is the top of the format, not buried behind a prefix.
    assert tag["format"]["type"] == "triggered_tags"
    assert THINK_END not in json.dumps(tag["format"]["tags"])


@pytest.mark.parametrize("chat_template_kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_thinking_mode_tag_keeps_the_reasoning_prefix(
    tools: list[ChatCompletionToolsParam],
    chat_template_kwargs: dict,
):
    """Here the model does write the marker, so the grammar must admit it."""
    tag = _tag_for(tools, chat_template_kwargs)

    assert tag["format"]["type"] == "sequence"
    prefix, suffix = tag["format"]["elements"]
    assert prefix["end"] == THINK_END
    assert suffix["type"] == "triggered_tags"


@pytest.mark.parametrize("chat_template_kwargs", [{}, {"thinking": True}])
def test_tag_names_exactly_the_declared_tools(
    tools: list[ChatCompletionToolsParam],
    chat_template_kwargs: dict,
):
    """Whatever the reasoning shape, the tool set is the declared one.

    This is the property whose absence was the bug: under `tool_choice="auto"`
    the model chooses between text and a tool call, never between a declared
    name and an invented one.
    """
    serialized = json.dumps(_tag_for(tools, chat_template_kwargs))

    for tool in tools:
        assert f'invoke name=\\"{tool.function.name}\\"' in serialized
    assert serialized.count("invoke name=") == len(tools)
    assert "totally_bogus" not in serialized


def test_tool_choice_none_installs_no_tag(tools: list[ChatCompletionToolsParam]):
    """"none" forbids the call outright, so there is nothing to constrain."""
    assert _adjusted_tag(tools, {}, tool_choice="none") is None


@pytest.mark.parametrize("response_format", [None, {"type": "text"}])
def test_auto_tolerates_a_non_constraining_reply_format(
    tools: list[ChatCompletionToolsParam],
    response_format: dict | None,
):
    """A reply format that takes no grammar does not contend for one."""
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=tools,
        tool_choice="auto",
        response_format=response_format,  # type: ignore[arg-type]
    )
    _parser(tools, {}).adjust_request(request)

    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag is not None


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_object"},
        {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": {"type": "object"}},
        },
    ],
)
def test_auto_refuses_a_competing_reply_schema(
    tools: list[ChatCompletionToolsParam],
    response_format: dict,
):
    """Now that "auto" takes the grammar, it can collide with a reply schema.

    Deliberate, and the policy `required` and named tool choice already follow:
    both constrain one decoder, so the request is refused rather than one of
    them silently winning. This is the visible consequence of tagging "auto" --
    it used to pass by quietly dropping the tool constraint, which is the very
    bug this file exists for.
    """
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=tools,
        tool_choice="auto",
        response_format=response_format,  # type: ignore[arg-type]
    )

    with pytest.raises(AphroditeValidationError, match="cannot be combined with tool calling"):
        _parser(tools, {}).adjust_request(request)
