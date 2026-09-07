# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""`tool_choice="none"` must yield a normal text reply.

"none" forbids *calling* a tool; it does not hide the tools, which stay in
the prompt exactly as OpenAI renders them. Two things follow:

1. The model must be unable to emit tool-call syntax at all.
2. If it emits one anyway, the reply must not come back empty.

Regression: with engine-backed parsers (qwen3, qwen3_coder, ...) a model
that answered with only a tool call produced whitespace-only content and no
tool call -- the call was parsed out and then discarded.
"""

import pytest

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.parser.abstract_parser import DelegatingParser
from aphrodite.parser.engine.registered_adapters import (
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
)
from aphrodite.parser.qwen3 import qwen3_config

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_media_observation",
            "description": "Report the observation requested about the supplied media.",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        },
    }
]

# What Qwen3-Coder emits when the prompt describes tools and nothing forbids
# a call: reasoning, then a tool call and nothing else.
TOOL_CALL_OUTPUT = (
    "Counting them.</think>\n\n"
    "<tool_call>\n<function=report_media_observation>\n"
    "<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def tokenizer():
    from aphrodite.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


def make_engine_parser(tokenizer):
    """An engine-backed parser pair, as `--reasoning-parser qwen3
    --tool-call-parser qwen3_coder` produces."""

    class TestParser(DelegatingParser):
        reasoning_parser_cls = Qwen3ParserReasoningAdapter
        tool_parser_cls = Qwen3ParserToolAdapter

    parser = TestParser(tokenizer, tools=TOOLS)
    assert parser._engine_based, "fixture must exercise the engine-backed path"
    return parser


def make_request(tool_choice):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "how many?"}],
        tools=TOOLS,
        tool_choice=tool_choice,
    )


class TestEntryMarkerDerivation:
    """The markers come from the transition table, not a hardcoded list."""

    def test_derived_from_tool_call_start_events(self):
        markers = qwen3_config("qwen3_coder").tool_call_entry_markers

        assert "<tool_call>" in markers
        # The `<function=` fallback is an entry point too; missing it would
        # leave a way to start a call that suppression does not cover.
        assert "<function=" in markers

    def test_markers_are_unique(self):
        markers = qwen3_config("qwen3_coder").tool_call_entry_markers

        assert len(markers) == len(set(markers))

    def test_exposed_through_tool_parser(self, tokenizer):
        parser = make_engine_parser(tokenizer)

        assert "<tool_call>" in parser._tool_parser.tool_call_entry_markers


class TestSuppression:
    """Layer 1: a tool call is made unrepresentable, not merely unwanted."""

    def test_none_suppresses_entry_markers(self, tokenizer):
        parser = make_engine_parser(tokenizer)
        request = parser.adjust_request(make_request("none"))

        assert "<tool_call>" in request.bad_words
        assert "<function=" in request.bad_words

    @pytest.mark.parametrize("tool_choice", ["auto", "required"])
    def test_other_choices_unconstrained(self, tokenizer, tool_choice):
        parser = make_engine_parser(tokenizer)
        request = parser.adjust_request(make_request(tool_choice))

        assert request.bad_words == []

    def test_no_tools_unconstrained(self, tokenizer):
        """Without tools there is no call to suppress."""
        parser = make_engine_parser(tokenizer)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert parser.adjust_request(request).bad_words == []

    def test_caller_bad_words_preserved(self, tokenizer):
        parser = make_engine_parser(tokenizer)
        request = make_request("none")
        request.bad_words = ["banana"]

        adjusted = parser.adjust_request(request)

        assert "banana" in adjusted.bad_words
        assert "<tool_call>" in adjusted.bad_words

    def test_suppression_is_idempotent(self, tokenizer):
        parser = make_engine_parser(tokenizer)
        request = parser.adjust_request(make_request("none"))
        request = parser.adjust_request(request)

        assert request.bad_words.count("<tool_call>") == 1


class TestReplyNeverEmpty:
    """Emptiness is prevented upstream, by suppression, not by re-surfacing
    the discarded call.

    Handing the raw text back would leak tool-call terminals into content,
    which `test_delegating_replay.py::test_delegating_parse_tool_choice_none`
    forbids -- for several parsers those terminals are real special tokens.
    So the guarantee is: with suppression in force the model cannot produce a
    tool-call-only reply in the first place.
    """

    def test_terminals_never_leak_into_content(self, tokenizer):
        """The stripping contract still holds for engine-backed parsers."""
        parser = make_engine_parser(tokenizer)
        request = make_request("none")

        _, content = parser.extract_reasoning(TOOL_CALL_OUTPUT, request)
        tool_calls, final = parser._extract_tool_calls(content, request, enable_auto_tools=True)

        assert tool_calls == []
        assert final is None or "<tool_call>" not in final

    def test_suppression_prevents_the_empty_reply(self, tokenizer):
        """The end-to-end guarantee: the tokens that would produce a
        tool-call-only reply are blocked before generation."""
        parser = make_engine_parser(tokenizer)
        request = parser.adjust_request(make_request("none"))

        sampling_params = request.to_sampling_params(max_tokens=64, default_sampling_params={})

        assert "<tool_call>" in sampling_params.bad_words

    def test_plain_text_reply_unaffected(self, tokenizer):
        parser = make_engine_parser(tokenizer)
        request = make_request("none")

        _, content = parser.extract_reasoning("Thinking.</think>\n\n3", request)
        tool_calls, final = parser._extract_tool_calls(content, request, enable_auto_tools=True)

        assert tool_calls == []
        assert final.strip() == "3"

    def test_auto_still_extracts_the_call(self, tokenizer):
        """The fix must not blunt normal tool calling."""
        parser = make_engine_parser(tokenizer)
        request = make_request("auto")

        _, content = parser.extract_reasoning(TOOL_CALL_OUTPUT, request)
        tool_calls, final = parser._extract_tool_calls(content, request, enable_auto_tools=True)

        assert [tc.name for tc in tool_calls] == ["report_media_observation"]
        assert final is None or "<tool_call>" not in final
