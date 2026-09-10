# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A reply schema on a harmony model belongs inside the `final` channel.

A gpt-oss reply is not a JSON document, it is a sequence of channelled
messages, and only the `final` one carries what the caller asked to be JSON.
Compiled as a plain schema the constraint spanned the whole generation, which
forbids the `<|channel|>analysis<|message|>` header the model has to open with.
The model then emitted `{` where a header belongs, the harmony parser never saw
a message it could complete, and `HarmonyParser.flush` handed the reply back
through its raw-output recovery -- so `content` came out as
`'{"a": 1}<|return|>'`, control token and all, instead of the isolated JSON.

Nothing but the reasoning-end gate stood between that and every structured
request: it withholds the bitmask until `<|channel|>final<|message|>` goes by.
That gate is off when the caller sends `include_reasoning: false`, when no
reasoning parser is configured, and when `enable_in_reasoning` is set. These
tests pin the tag that replaces it, which spells the analysis channel out as
part of the grammar and so needs no gate at all.

The second half covers what happens when the format breaks anyway. xgrammar
does not know harmony's control tokens are special -- their decoded text is
just `<|end|>` -- so a JSON string admits them like any other characters. The
harmony parser rejects one outright, and that used to leave the request as a
500.
"""

import json
from unittest.mock import Mock

import pytest

from aphrodite.config import AphroditeConfig, ModelConfig
from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.parser.harmony import HarmonyParser
from aphrodite.parser.parser_manager import ParserManager
from aphrodite.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from aphrodite.sampling_params import StructuredOutputsParams
from aphrodite.tool_parsers.abstract_tool_parser import FREEFORM_JSON_OBJECT
from aphrodite.tool_parsers.gptoss_tool_parser import GptOssToolParser
from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions
from aphrodite.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.cpu_test

# 20b and 120b share the o200k_harmony tokenizer; 20b is the cheaper pull.
MODEL = "openai/gpt-oss-20b"
VOCAB_SIZE = 201088
ENGINE_EOS_IDS = [200002, 199999, 200012]  # <|return|>, <|endoftext|>, <|call|>

REPLY_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

ANALYSIS = "<|channel|>analysis<|message|>The user wants a fact.<|end|><|start|>assistant"
FINAL = "<|channel|>final<|message|>"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def backend(tokenizer):
    model_config = Mock(spec=ModelConfig)
    model_config.try_get_generation_config = Mock(return_value={"eos_token_id": list(ENGINE_EOS_IDS)})

    config = Mock(spec=AphroditeConfig)
    config.model_config = model_config
    config.structured_outputs_config = Mock()
    config.structured_outputs_config.disable_any_whitespace = False
    config.speculative_config = None

    return XgrammarBackend(config, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)


def _parser(tokenizer, *, tool_parser: bool = True) -> HarmonyParser:
    """A HarmonyParser as the serving layer builds one.

    ``tool_parser=False`` is the plain `--reasoning-parser openai_gptoss`
    deployment: no tool parser at all, which is the case the old code could not
    reach, since the structural tag was installed from inside the tool branch.
    """
    HarmonyParser.reasoning_parser_cls = GptOssReasoningParser
    HarmonyParser.tool_parser_cls = GptOssToolParser if tool_parser else None
    return HarmonyParser(tokenizer)


def _adjusted(tokenizer, *, tool_parser: bool = True, **kwargs) -> ChatCompletionRequest:
    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}], **kwargs)
    _parser(tokenizer, tool_parser=tool_parser).adjust_request(request)
    return request


def _tag(request) -> dict:
    structured_outputs = request.structured_outputs
    assert structured_outputs is not None, "no structured outputs on the request"
    assert structured_outputs.structural_tag is not None, "no structural tag on the request"
    return json.loads(structured_outputs.structural_tag)


def _tag_named(tag: dict, begin: str) -> dict:
    """The one tag in the alternation that opens with `begin`."""
    tags = [t for t in tag["format"]["tags"] if t["begin"] == begin]
    assert len(tags) == 1, f"expected exactly one {begin!r} tag, got {len(tags)}"
    return tags[0]


class TestTheSchemaGoesInTheFinalChannel:
    def test_json_object_constrains_the_reply_and_not_the_reasoning(self, tokenizer):
        request = _adjusted(tokenizer, response_format={"type": "json_object"})

        tag = _tag(request)
        assert _tag_named(tag, FINAL)["content"] == {
            "type": "json_schema",
            "json_schema": FREEFORM_JSON_OBJECT,
            "style": "json",
            "any_order": False,
        }
        assert _tag_named(tag, "<|channel|>analysis<|message|>")["content"]["type"] == "any_text"

    def test_a_deployment_with_no_tool_parser_gets_it_too(self, tokenizer):
        """The reply schema is a grammar concern, not a tool one. Installing it
        from the tool branch meant `--reasoning-parser` on its own -- the way
        gpt-oss is usually served -- never got the tag."""
        request = _adjusted(tokenizer, tool_parser=False, response_format={"type": "json_object"})

        assert _tag_named(_tag(request), FINAL)["content"]["type"] == "json_schema"

    def test_a_deployment_with_nothing_configured_gets_it_too(self, tokenizer):
        """gpt-oss served with neither parser named used to get no Parser at
        all, so nothing read the channels back and nothing scoped the schema.
        The channel structure is the format, not an optional feature."""
        parser_cls = ParserManager.get_parser(
            tool_parser_name=None,
            reasoning_parser_name="",
            enable_auto_tools=False,
            model_name=MODEL,
            is_harmony=True,
        )
        assert parser_cls is HarmonyParser

        request = ChatCompletionRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            response_format={"type": "json_object"},
        )
        parser = parser_cls(tokenizer)
        parser.adjust_request(request)

        assert _tag_named(_tag(request), FINAL)["content"]["type"] == "json_schema"
        # And the reply comes back as the reply, not as the transcript.
        reply = ANALYSIS + FINAL + '{"answer": "Paris"}<|return|>'
        reasoning, content, _ = parser.parse(
            "", request, model_output_token_ids=tokenizer.encode(reply, add_special_tokens=False)
        )
        assert content == '{"answer": "Paris"}'
        assert reasoning is None

    def test_a_model_that_is_not_harmony_still_gets_no_parser(self):
        """The rule above is about harmony's format, not a general one: every
        other model speaks plain text when nothing is configured."""
        assert ParserManager.get_parser(model_name="some/model", is_harmony=False) is None

    def test_the_callers_schema_is_what_lands(self, tokenizer):
        request = _adjusted(
            tokenizer,
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": REPLY_SCHEMA}},
        )

        assert _tag_named(_tag(request), FINAL)["content"]["json_schema"] == REPLY_SCHEMA

    def test_a_waived_schema_falls_back_to_an_object(self, tokenizer):
        """`strict: false` waives the schema and leaves `json_object` behind."""
        request = _adjusted(
            tokenizer,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": False, "schema": REPLY_SCHEMA},
            },
        )

        assert _tag_named(_tag(request), FINAL)["content"]["json_schema"] == FREEFORM_JSON_OBJECT

    def test_structured_outputs_json_is_carried_too(self, tokenizer):
        """The same constraint asked for directly rather than via response_format."""
        request = _adjusted(tokenizer, structured_outputs=StructuredOutputsParams(json=REPLY_SCHEMA))

        assert _tag_named(_tag(request), FINAL)["content"]["json_schema"] == REPLY_SCHEMA

    def test_the_reply_ends_on_the_token_that_ends_the_turn(self, tokenizer):
        """`<|end|>` would let a second `final` message follow, and two documents
        that each satisfy a schema do not satisfy it once joined."""
        request = _adjusted(tokenizer, response_format={"type": "json_object"})

        assert _tag_named(_tag(request), FINAL)["end"] == ["<|return|>"]

    def test_the_bitmask_is_not_withheld_for_the_reasoning(self, tokenizer):
        """The tag covers the analysis channel, so there is no prelude to wait
        out -- which is the whole point of not needing the reasoning-end gate."""
        request = _adjusted(tokenizer, response_format={"type": "json_object"})

        assert request._grammar_from_tool_parser is True
        assert request.response_format is None

    def test_the_responses_api_is_scoped_the_same_way(self, tokenizer):
        request = ResponsesRequest(model=MODEL, input="Hi", text={"format": {"type": "json_object"}})
        _parser(tokenizer).adjust_request(request)

        assert _tag_named(_tag(request), FINAL)["content"]["type"] == "json_schema"
        assert request.text is None


class TestWhatIsLeftAlone:
    @pytest.mark.parametrize("response_format", [None, {"type": "text"}])
    def test_a_request_asking_for_nothing_gets_no_tag(self, tokenizer, response_format):
        request = _adjusted(tokenizer, response_format=response_format)

        structured_outputs = request.structured_outputs
        assert structured_outputs is None or structured_outputs.structural_tag is None

    @pytest.mark.parametrize(
        ("name", "value"),
        [("regex", r"\d+"), ("choice", ["a", "b"]), ("grammar", 'root ::= "a"')],
    )
    def test_a_constraint_with_no_slot_in_the_tag_stays_where_it_is(self, tokenizer, name, value):
        """A regex has nothing to sit in a schema slot. It spans the whole reply
        and has the same problem, but taking it away from callers who use it
        today is not this fix's business."""
        request = _adjusted(tokenizer, structured_outputs=StructuredOutputsParams(**{name: value}))

        assert getattr(request.structured_outputs, name) == value
        assert request.structured_outputs.structural_tag is None

    def test_the_callers_own_structural_tag_wins(self, tokenizer):
        caller_tag = '{"type": "structural_tag", "format": {"type": "any_text"}}'
        request = _adjusted(
            tokenizer,
            response_format={"type": "json_object"},
            structured_outputs=StructuredOutputsParams(structural_tag=caller_tag),
        )

        assert request.structured_outputs.structural_tag == caller_tag

    def test_a_tool_grammar_keeps_the_reply_it_already_carries(self, tokenizer):
        """`tool_choice="auto"` merges the reply into the tool tag. The reply-only
        tag must not then replace it and drop the tools."""
        request = _adjusted(tokenizer, tools=TOOLS, tool_choice="auto", response_format={"type": "json_object"})

        tag = _tag(request)
        assert _tag_named(tag, FINAL)["content"]["type"] == "json_schema"
        assert any("functions.get_weather" in t["begin"] for t in tag["format"]["tags"])

    def test_tool_choice_none_gets_the_reply_only_tag(self, tokenizer):
        """No call is going to be made, so nothing is lost by leaving no room
        for one -- and `none` asked for exactly that."""
        request = _adjusted(tokenizer, tools=TOOLS, tool_choice="none", response_format={"type": "json_object"})

        tag = _tag(request)
        assert _tag_named(tag, FINAL)["content"]["type"] == "json_schema"
        assert not any("functions." in t["begin"] for t in tag["format"]["tags"])


class TestTheTagDecodes:
    """Compiled by the real backend, walked by the real grammar."""

    @staticmethod
    def _grammar(backend, request):
        return backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            request.structured_outputs.structural_tag,
        )

    @staticmethod
    def _accepts(grammar, tokenizer, text) -> bool:
        return grammar.accept_tokens("test", tokenizer.encode(text, add_special_tokens=False))

    def test_a_harmony_reply_with_a_json_final_is_accepted(self, backend, tokenizer):
        request = _adjusted(tokenizer, response_format={"type": "json_object"})
        grammar = self._grammar(backend, request)

        assert self._accepts(grammar, tokenizer, ANALYSIS + FINAL + '{"answer": "Paris"}<|return|>')

    def test_reasoning_is_left_unconstrained(self, backend, tokenizer):
        """The bug in one line: a schema over the whole reply forbids the header
        the model must open with."""
        request = _adjusted(tokenizer, response_format={"type": "json_object"})
        grammar = self._grammar(backend, request)

        assert self._accepts(grammar, tokenizer, "<|channel|>analysis<|message|>Thinking about it.")

    def test_a_bare_json_document_is_rejected(self, backend, tokenizer):
        """What the old whole-reply grammar demanded is now the one thing that
        cannot be said: there is no message for it to be in."""
        request = _adjusted(tokenizer, response_format={"type": "json_object"})
        grammar = self._grammar(backend, request)

        assert not self._accepts(grammar, tokenizer, '{"answer": "Paris"}')

    def test_prose_in_the_final_channel_is_rejected(self, backend, tokenizer):
        request = _adjusted(tokenizer, response_format={"type": "json_object"})
        grammar = self._grammar(backend, request)

        assert not self._accepts(grammar, tokenizer, FINAL + "The capital of France is Paris.")

    def test_json_object_still_means_an_object(self, backend, tokenizer):
        """It is the object the JSON_OBJECT grammar compiles for a request with
        no tools, so a bare string is no more acceptable inside a tag than out."""
        request = _adjusted(tokenizer, response_format={"type": "json_object"})
        grammar = self._grammar(backend, request)

        assert not self._accepts(grammar, tokenizer, FINAL + '"Paris"')

    def test_the_callers_schema_is_enforced(self, backend, tokenizer):
        request = _adjusted(
            tokenizer,
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": REPLY_SCHEMA}},
        )
        grammar = self._grammar(backend, request)

        assert not self._accepts(grammar, tokenizer, FINAL + '{"unexpected"')


class TestRawOutputRecovery:
    """A stream that stops following harmony has no parse left to salvage. The
    caller gets the text; they used to get a 500."""

    # xgrammar keeps `<|end|>` in the vocabulary with its literal text, and a
    # JSON string admits that text like any other. The harmony parser refuses
    # the token, which used to escape `process_chunk` uncaught.
    DERAILED = ANALYSIS + FINAL + '{"answer": "Par<|end|>is"}<|return|>'
    # `<|call|>` is the same hazard at its worst: legal JSON string text, an
    # engine stop token, and refused by the harmony parser part-way through a
    # message rather than after one has closed.
    DERAILED_MID_MESSAGE = ANALYSIS + FINAL + '{"answer": "Par<|call|>'
    # The old symptom: a grammar applied from the first token, so no channel
    # header was ever emitted and no message could complete.
    HEADERLESS = '{"answer": "Paris"}<|return|>'
    CLEAN = ANALYSIS + FINAL + '{"answer": "Paris"}<|return|>'

    @staticmethod
    def _request():
        return ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])

    def _parse(self, tokenizer, text):
        return _parser(tokenizer).parse(
            "", self._request(), model_output_token_ids=tokenizer.encode(text, add_special_tokens=False)
        )

    def _stream(self, tokenizer, text):
        parser, request = _parser(tokenizer), self._request()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        reasoning, content = "", ""
        for index, token_id in enumerate(token_ids):
            delta = parser.parse_delta(
                tokenizer.decode([token_id]),
                [token_id],
                request,
                finished=index == len(token_ids) - 1,
            )
            if delta is None:
                continue
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        return reasoning, content

    def test_a_control_token_in_a_string_does_not_fail_the_request(self, tokenizer):
        reasoning, content, tool_calls = self._parse(tokenizer, self.DERAILED)

        assert content is not None and "Par" in content and "is" in content
        assert tool_calls is None

    def test_the_reasoning_before_the_break_survives(self, tokenizer):
        reasoning, _, _ = self._parse(tokenizer, self.DERAILED)

        assert reasoning == "The user wants a fact."

    def test_a_stream_recovers_as_well(self, tokenizer):
        reasoning, content = self._stream(tokenizer, self.DERAILED)

        assert reasoning == "The user wants a fact."
        assert "is" in content

    def test_a_stream_does_not_repeat_what_it_already_sent(self, tokenizer):
        """The recovered message carries the whole of the broken message, which
        is what a non-streaming reply needs; a stream has already sent the part
        that parsed and must only be given the tail."""
        _, content = self._stream(tokenizer, self.DERAILED)

        assert content.count('{"answer"') == 1

    def test_a_break_part_way_through_a_message_does_not_fail_it_either(self, tokenizer):
        reasoning, content, _ = self._parse(tokenizer, self.DERAILED_MID_MESSAGE)

        assert reasoning == "The user wants a fact."
        assert content == '{"answer": "Par'

    def test_what_was_read_before_the_break_is_handed_back_as_content(self, tokenizer):
        """Not as the raw bytes it arrived in: the header was already parsed
        through, so repeating it would put `<|channel|>final<|message|>` in the
        reply -- which is the thing this whole file is about."""
        _, content, _ = self._parse(tokenizer, self.DERAILED_MID_MESSAGE)

        assert FINAL not in content

    def test_a_break_part_way_through_reads_the_same_streamed(self, tokenizer):
        assert self._stream(tokenizer, self.DERAILED_MID_MESSAGE) == (
            "The user wants a fact.",
            '{"answer": "Par',
        )

    def test_a_reply_that_never_opens_a_channel_is_still_returned(self, tokenizer):
        _, content, _ = self._parse(tokenizer, self.HEADERLESS)

        assert content is not None and '"answer"' in content

    def test_a_well_formed_reply_is_untouched(self, tokenizer):
        reasoning, content, tool_calls = self._parse(tokenizer, self.CLEAN)

        assert reasoning == "The user wants a fact."
        assert content == '{"answer": "Paris"}'
        assert tool_calls is None

    def test_a_well_formed_stream_is_untouched(self, tokenizer):
        reasoning, content = self._stream(tokenizer, self.CLEAN)

        assert reasoning == "The user wants a fact."
        assert content == '{"answer": "Paris"}'
