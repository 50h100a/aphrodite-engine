# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A reply schema and tool calls in one request.

One decoder runs one grammar, but a grammar can have two branches, and a tool
call is spelled differently from a reply. So the tool structural tag keeps its
tool branches and its reply branch takes the caller's schema: the model answers
either with a tool call or with a document the schema admits, and the prose that
`tool_choice: "auto"` would otherwise allow is gone -- it would have broken the
schema anyway.

Two cases do not merge. Llama spells its tool calls as bare JSON, so a JSON
reply is not a second branch but the same one. And when the tool call is forced
there is no reply left for a schema to constrain, which is not an error: the
constraint is inert, and is dropped without comment.

The rest is about the requests that get no tag at all, because their parser has
no structural tag model. There the reply schema is the only grammar and it spans
the whole reply, so the tools are gone whether or not anyone says so. These are
refused. `APHRODITE_ENFORCE_STRICT_TOOL_CALLING` is not one of those cases: it
sets the default `strict` for a tool that states none, and nothing else.
"""

import json
from unittest.mock import MagicMock, Mock

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite import envs
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.parser.parser_manager import ParserManager
from aphrodite.sampling_params import StructuredOutputsParams
from aphrodite.tool_parsers.abstract_tool_parser import (
    ToolParser,
    reject_reply_schema_without_tool_grammar,
    reject_unmergeable_reply_schema,
    reply_schema_for_tool_grammar,
)
from aphrodite.tool_parsers.structural_tag_registry import (
    get_model_structural_tag,
    merge_reply_schema,
)

pytestmark = pytest.mark.cpu_test

ENUM_SCHEMA = {
    "name": "decision",
    "schema": {"type": "string", "enum": ["approve", "reject", "escalate"]},
    "strict": True,
}
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
TOOLS = [
    ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    )
]


def _request(**kwargs):
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kwargs)


def _merge(model, reply_schema, reasoning=False):
    tag = get_model_structural_tag(model=model, tools=TOOLS, tool_choice="auto", reasoning=reasoning)
    assert tag is not None
    return merge_reply_schema(tag, reply_schema)


def _accepts(model, reply_schema, reasoning=False):
    merged = _merge(model, reply_schema, reasoning)
    assert merged is not None
    grammar = xgr.Grammar.from_structural_tag(json.dumps(merged.model_dump()))
    return lambda text: _is_grammar_accept_string(grammar, text)


# --- what the caller asked for ---------------------------------------------


def test_json_schema_reply_format_is_carried():
    request = _request(response_format={"type": "json_schema", "json_schema": ENUM_SCHEMA})

    assert reply_schema_for_tool_grammar(request) == ENUM_SCHEMA["schema"]


def test_json_object_reply_format_is_any_json():
    assert reply_schema_for_tool_grammar(_request(response_format={"type": "json_object"})) is True


def test_structured_outputs_json_is_carried():
    """The same constraint asked for directly rather than via response_format."""
    request = _request(structured_outputs=StructuredOutputsParams(json=REPLY_SCHEMA))

    assert reply_schema_for_tool_grammar(request) == REPLY_SCHEMA


@pytest.mark.parametrize("response_format", [None, {"type": "text"}])
def test_unconstraining_reply_formats_ask_for_nothing(response_format):
    """`response_format: {"type": "text"}` is what a client sends when it wants
    nothing in particular."""
    assert reply_schema_for_tool_grammar(_request(response_format=response_format)) is None


@pytest.mark.parametrize("constraint", ["regex", "choice", "grammar"])
def test_non_schema_constraints_are_refused(constraint):
    """The tag holds the reply in a slot shaped like a schema; a regex or a
    choice list has nothing to sit in it."""
    values = {"regex": r"\d+", "choice": ["a", "b"], "grammar": "root ::= \"a\""}
    request = _request(structured_outputs=StructuredOutputsParams(**{constraint: values[constraint]}))

    with pytest.raises(AphroditeValidationError) as excinfo:
        reply_schema_for_tool_grammar(request)

    assert excinfo.value.parameter == "structured_outputs"


def test_callers_own_structural_tag_is_refused():
    request = _request(response_format={"type": "structural_tag", "format": {"type": "any_text"}})

    with pytest.raises(AphroditeValidationError) as excinfo:
        reply_schema_for_tool_grammar(request)

    assert excinfo.value.parameter == "response_format"


def test_responses_api_text_format_is_carried():
    request = Mock()
    request.text = Mock()
    request.text.format = Mock(type="json_schema", schema_=REPLY_SCHEMA)
    request.structured_outputs = None
    # Route on the real type rather than on hasattr, so patch the isinstance
    # check the function makes.
    from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest

    request.__class__ = ResponsesRequest  # type: ignore[assignment]

    assert reply_schema_for_tool_grammar(request) == REPLY_SCHEMA


# --- what the merged grammar admits ----------------------------------------


@pytest.mark.parametrize("model", ["qwen_3", "hermes", "minimax", "deepseek_v4", "kimi", "glm_4_7"])
def test_marker_delimited_models_take_either_branch(model):
    accepts = _accepts(model, REPLY_SCHEMA)

    assert accepts('{"answer": "it is sunny"}')
    assert not accepts('{"answer": 5}')
    assert not accepts("It is sunny in Paris.")


def test_the_tool_branch_survives_the_merge():
    accepts = _accepts("qwen_3", REPLY_SCHEMA)

    assert accepts('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')


def test_a_reasoning_prefix_still_runs_ahead_of_the_reply():
    """The tag's reasoning form makes the segment mandatory, and the merge
    narrows what follows it rather than replacing it."""
    accepts = _accepts("qwen_3", REPLY_SCHEMA, reasoning=True)

    assert accepts('thinking</think>\n\n{"answer": "sunny"}')
    assert not accepts('{"answer": "sunny"}')


def test_harmony_constrains_the_final_channel_and_leaves_analysis_alone():
    """Harmony names the reply channel, so the schema goes inside that tag. The
    analysis channel is reasoning, not the reply, and stays free text."""
    accepts = _accepts("harmony", REPLY_SCHEMA, reasoning=True)

    assert accepts('<|channel|>final<|message|>{"answer": "sunny"}<|return|>')
    assert not accepts("<|channel|>final<|message|>It is sunny.<|return|>")
    assert accepts(
        "<|channel|>analysis<|message|>hmm<|end|>"
        '<|start|>assistant<|channel|>final<|message|>{"answer": "sunny"}<|return|>'
    )
    assert accepts(
        "<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>"
        '{"city": "Paris"}<|call|>'
    )


def test_llama_does_not_merge():
    """Its tool calls are bare JSON objects, so a JSON reply is the same branch
    and no parser downstream could say which was meant."""
    assert _merge("llama", REPLY_SCHEMA) is None


# --- forced tool calls ------------------------------------------------------


def test_a_forced_tool_call_drops_the_reply_schema_without_complaint():
    """`tool_choice` naming a tool leaves no reply for a schema to constrain, so
    the schema is inert rather than in conflict."""
    request = _request(
        tools=[TOOLS[0].model_dump()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        response_format={"type": "json_schema", "json_schema": ENUM_SCHEMA},
    )

    ToolParser(Mock()).adjust_request(request)

    assert request.response_format is None
    assert request.structured_outputs is not None
    assert request.structured_outputs.json == TOOLS[0].function.parameters


@pytest.mark.parametrize("refuse", [reject_unmergeable_reply_schema, reject_reply_schema_without_tool_grammar])
def test_the_refusals_are_value_errors_so_they_become_400s(refuse):
    """`create_error_response` maps ValueError and its subclasses to 400; a
    different base would surface as a 500."""
    assert issubclass(AphroditeValidationError, ValueError)

    with pytest.raises(AphroditeValidationError) as excinfo:
        refuse(_request(response_format={"type": "json_object"}))

    assert excinfo.value.parameter == "response_format"


# --- requests that get no tool grammar at all -------------------------------

JSON_SCHEMA_FORMAT = {"type": "json_schema", "json_schema": ENUM_SCHEMA}


def _composed_parser(tool_parser_name: str):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return parser_cls(tokenizer, TOOLS)  # type: ignore[arg-type]


def _auto_request(**kwargs):
    return _request(tools=[TOOLS[0].model_dump()], tool_choice="auto", **kwargs)


def test_a_parser_with_no_tool_grammar_refuses_a_reply_schema():
    """`granite` has no structural tag model, so nothing constrains its
    `<tool_call>` syntax and the reply schema would be the whole grammar."""
    with pytest.raises(AphroditeValidationError) as excinfo:
        _composed_parser("granite").adjust_request(_auto_request(response_format=JSON_SCHEMA_FORMAT))

    assert "no grammar for its tool calls" in str(excinfo.value)


def test_a_parser_with_no_tool_grammar_is_untouched_without_a_reply_schema():
    """Nothing is contending for the grammar, so the request goes through."""
    request = _auto_request()

    _composed_parser("granite").adjust_request(request)

    assert request.structured_outputs is None


@pytest.mark.parametrize("strict", [True, False])
def test_strict_tool_calling_only_sets_the_default_strict(monkeypatch, strict):
    """The flag is the `strict` a tool that states none takes. Off, the tool's
    arguments are free-form JSON -- the tag still spells the call, and still
    carries the reply schema beside it."""
    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", strict, raising=False)
    request = _auto_request(response_format=JSON_SCHEMA_FORMAT)

    _composed_parser("hermes").adjust_request(request)

    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag is not None
    tag = request.structured_outputs.structural_tag
    assert "get_weather" in tag
    assert ("city" in tag) is strict


def test_a_tool_stating_its_own_strict_ignores_the_flag(monkeypatch):
    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", False, raising=False)
    tool = TOOLS[0].model_dump()
    tool["function"]["strict"] = True
    request = _request(tools=[tool], tool_choice="auto")

    _composed_parser("hermes").adjust_request(request)

    assert request.structured_outputs is not None
    assert "city" in (request.structured_outputs.structural_tag or "")


def test_a_parser_that_merges_its_own_reply_schema_is_left_alone(monkeypatch):
    """Mistral builds one Lark grammar over the tools and the reply schema
    together, so the tag path must not preempt it."""
    parser = _composed_parser("granite")
    monkeypatch.setattr(type(parser._tool_parser), "merges_reply_schema", True)
    request = _auto_request(response_format=JSON_SCHEMA_FORMAT)

    parser.adjust_request(request)

    assert request.response_format is not None


def test_a_tag_already_on_the_request_is_left_alone():
    """The Cohere reasoning parser folds the tools and the reply schema into a
    tag of its own before this runs, and it runs for parsers with no tag model
    of their own. Nothing here may second-guess a grammar already settled."""
    installed = '{"type": "structural_tag", "format": {"type": "any_text"}}'
    request = _auto_request(structured_outputs=StructuredOutputsParams(structural_tag=installed))

    _composed_parser("granite").adjust_request(request)

    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag == installed


@pytest.mark.parametrize("strict", [True, False])
def test_a_forced_tool_call_drops_the_schema_whatever_the_flag_says(monkeypatch, strict):
    """The forced tag spans the whole reply either way, so the reply schema is
    inert either way -- and dropping it is not the flag's business."""
    monkeypatch.setattr(envs, "APHRODITE_ENFORCE_STRICT_TOOL_CALLING", strict, raising=False)
    request = _request(
        tools=[TOOLS[0].model_dump()],
        tool_choice="required",
        response_format=JSON_SCHEMA_FORMAT,
    )

    _composed_parser("hermes").adjust_request(request)

    assert request.response_format is None
    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag is not None
    assert "decision" not in request.structured_outputs.structural_tag
