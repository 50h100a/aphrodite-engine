# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""`strict: false` waives the schema, leaving `json_object`.

A caller who says `strict: false` is telling us not to impose their schema on
decoding -- they will validate the output themselves. So the schema stops at the
door: the constraint drops to what `json_object` asks for (valid JSON and
nothing more), the schema is never compiled into a grammar, and it is never put
to the unenforceable-keyword screen either. That last part is the reason the
flag is worth having. `dependentRequired` and friends are refused with a 400
when a caller asks for them enforced, and a caller who already said they did not
want them enforced should not be told the request is impossible.

Every route into the grammar answers the flag the same way: `response_format`
and `text.format` on both APIs, the reply slot a tool grammar carries, and the
tool parameter schemas -- whether those reach the decoder through a structural
tag or through the JSON path a tagless model falls back to. For tools, an
*unstated* `strict` takes `APHRODITE_ENFORCE_STRICT_TOOL_CALLING`, which is what
that flag has always meant; for a response format, unstated means the schema
stands, since only a caller who wrote `false` asked for the looser thing.
"""

import json

import pytest
from openai.types.responses import FunctionTool, ToolChoiceFunction

from aphrodite import envs
from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.sampling_params import StructuredOutputsParams
from aphrodite.tool_parsers.abstract_tool_parser import (
    FREEFORM_JSON_OBJECT,
    reply_schema_for_tool_grammar,
)
from aphrodite.tool_parsers.structural_tag_registry import (
    get_model_structural_tag,
    merge_reply_schema,
)
from aphrodite.tool_parsers.utils import get_json_schema_from_tools
from aphrodite.v1.structured_output.schema_features import (
    get_structured_outputs_schema_error,
    get_unenforceable_json_schema_keys,
)

pytestmark = pytest.mark.cpu_test

# `dependentRequired` is in the "no backend enforces this" row of the keyword
# table, so a schema carrying it is refused outright when it is to be enforced.
# That makes it the sharp end of the waiver: the same schema has to sail through
# once the caller says they do not want it enforced.
UNENFORCEABLE_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    "dependentRequired": {"a": ["b"]},
    "required": ["a"],
}


def _chat_request(strict: bool | None, **extra) -> ChatCompletionRequest:
    json_schema: dict = {"name": "reply", "schema": dict(UNENFORCEABLE_SCHEMA)}
    if strict is not None:
        json_schema["strict"] = strict
    return ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            **extra,
        }
    )


def _responses_request(strict: bool | None) -> ResponsesRequest:
    fmt: dict = {"type": "json_schema", "name": "reply", "schema": dict(UNENFORCEABLE_SCHEMA)}
    if strict is not None:
        fmt["strict"] = strict
    return ResponsesRequest.model_validate({"model": "m", "input": "hi", "text": {"format": fmt}})


def _chat_tool(strict: bool | None) -> ChatCompletionToolsParam:
    function: dict = {"name": "t", "parameters": dict(UNENFORCEABLE_SCHEMA)}
    if strict is not None:
        function["strict"] = strict
    return ChatCompletionToolsParam.model_validate({"type": "function", "function": function})


def _responses_tool(strict: bool | None) -> FunctionTool:
    return FunctionTool(
        type="function",
        name="t",
        parameters=dict(UNENFORCEABLE_SCHEMA),
        strict=strict,
    )


class TestResponseFormat:
    def test_non_strict_becomes_json_object(self):
        params = _chat_request(strict=False).extract_structured_outputs()
        assert params.json_object is True
        assert params.json is None

    @pytest.mark.parametrize("strict", [True, None])
    def test_otherwise_the_schema_stands(self, strict):
        params = _chat_request(strict).extract_structured_outputs()
        assert params.json == UNENFORCEABLE_SCHEMA
        assert params.json_object is None

    def test_non_strict_skips_the_unenforceability_screen(self):
        """The point of the waiver: no schema reaches the screen to fail it."""
        params = _chat_request(strict=False).extract_structured_outputs()
        assert get_structured_outputs_schema_error(params) is None

    @pytest.mark.parametrize("strict", [True, None])
    def test_an_enforced_schema_is_still_screened(self, strict):
        params = _chat_request(strict).extract_structured_outputs()
        error = get_structured_outputs_schema_error(params)
        assert error is not None
        assert "dependentRequired" in error

    def test_json_object_is_untouched(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            }
        )
        assert request.extract_structured_outputs().json_object is True


class TestResponsesTextFormat:
    """Responses states `strict` inline on the format rather than nesting it."""

    def test_non_strict_becomes_json_object(self):
        params = _responses_request(strict=False).extract_structured_outputs()
        assert params.json_object is True
        assert params.json is None

    @pytest.mark.parametrize("strict", [True, None])
    def test_otherwise_the_schema_stands(self, strict):
        params = _responses_request(strict).extract_structured_outputs()
        assert params.json == UNENFORCEABLE_SCHEMA


class TestReplySlotInToolGrammar:
    """The reply branch a tool structural tag carries alongside its calls."""

    def test_non_strict_reply_schema_is_carried_as_free_json(self):
        request = _chat_request(strict=False, tools=[_chat_tool(True)], tool_choice="auto")
        # The waived schema resolves to the same thing a `json_object` response
        # format does, which is an object of any shape rather than any JSON
        # document at all -- `json_object` is narrower than `True`.
        assert reply_schema_for_tool_grammar(request) == FREEFORM_JSON_OBJECT

    @pytest.mark.parametrize("strict", [True, None])
    def test_otherwise_the_schema_is_carried(self, strict):
        request = _chat_request(strict, tools=[_chat_tool(True)], tool_choice="auto")
        assert reply_schema_for_tool_grammar(request) == UNENFORCEABLE_SCHEMA

    def test_the_merged_tag_is_not_screened_for_the_waived_schema(self):
        request = _chat_request(strict=False, tools=[_chat_tool(False)], tool_choice="auto")
        tag = get_model_structural_tag(
            model="hermes",
            tools=request.tools,
            tool_choice="auto",
            reasoning=False,
        )
        merged = merge_reply_schema(tag, reply_schema_for_tool_grammar(request))
        assert merged is not None
        params = StructuredOutputsParams(structural_tag=json.dumps(merged.model_dump()))
        assert "dependentRequired" not in params.structural_tag
        assert get_structured_outputs_schema_error(params) is None


class TestToolParameterSchemas:
    """The JSON path taken by models whose parser has no structural tag."""

    @pytest.mark.parametrize(
        "label,tool_choice,tools_for",
        [
            ("required, chat", "required", lambda s: [_chat_tool(s)]),
            (
                "named, chat",
                ChatCompletionNamedToolChoiceParam.model_validate({"type": "function", "function": {"name": "t"}}),
                lambda s: [_chat_tool(s)],
            ),
            ("required, responses", "required", lambda s: [_responses_tool(s)]),
            (
                "forced, responses",
                ToolChoiceFunction(type="function", name="t"),
                lambda s: [_responses_tool(s)],
            ),
        ],
    )
    def test_non_strict_tool_parameters_are_waived(self, label, tool_choice, tools_for):
        schema = get_json_schema_from_tools(tool_choice=tool_choice, tools=tools_for(False))
        assert get_unenforceable_json_schema_keys(schema) == []
        assert "dependentRequired" not in json.dumps(schema)

    @pytest.mark.parametrize(
        "label,tool_choice,tools_for",
        [
            ("required, chat", "required", lambda s: [_chat_tool(s)]),
            (
                "named, chat",
                ChatCompletionNamedToolChoiceParam.model_validate({"type": "function", "function": {"name": "t"}}),
                lambda s: [_chat_tool(s)],
            ),
            ("required, responses", "required", lambda s: [_responses_tool(s)]),
            (
                "forced, responses",
                ToolChoiceFunction(type="function", name="t"),
                lambda s: [_responses_tool(s)],
            ),
        ],
    )
    def test_a_strict_tool_keeps_its_schema(self, label, tool_choice, tools_for):
        schema = get_json_schema_from_tools(tool_choice=tool_choice, tools=tools_for(True))
        assert get_unenforceable_json_schema_keys(schema) == ["dependentRequired"]

    def test_a_waived_tool_still_has_to_spell_an_object(self):
        """`strict: false` waives the shape of the arguments, not the syntax."""
        schema = get_json_schema_from_tools(
            tool_choice=ChatCompletionNamedToolChoiceParam.model_validate(
                {"type": "function", "function": {"name": "t"}}
            ),
            tools=[_chat_tool(False)],
        )
        assert schema == {"type": "object", "additionalProperties": True}

    @pytest.mark.parametrize("flag", [True, False])
    def test_an_unstated_strict_takes_the_flag(self, monkeypatch, flag):
        """The same default the structural tag path applies, so the two routes
        into the grammar cannot disagree about a tool that states nothing."""
        monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", flag, raising=False)
        schema = get_json_schema_from_tools(
            tool_choice=ChatCompletionNamedToolChoiceParam.model_validate(
                {"type": "function", "function": {"name": "t"}}
            ),
            tools=[_chat_tool(None)],
        )
        assert ("dependentRequired" in json.dumps(schema)) is flag

    def test_a_tool_stating_strict_ignores_the_flag(self, monkeypatch):
        monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", False, raising=False)
        schema = get_json_schema_from_tools(
            tool_choice=ChatCompletionNamedToolChoiceParam.model_validate(
                {"type": "function", "function": {"name": "t"}}
            ),
            tools=[_chat_tool(True)],
        )
        assert "dependentRequired" in json.dumps(schema)
