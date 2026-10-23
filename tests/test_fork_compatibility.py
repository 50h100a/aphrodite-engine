# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for fixes translated from the older Sonar fork."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai_harmony import HarmonyError

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.completion.serving import OpenAIServingCompletion
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.logprobs import Logprob
from aphrodite.parser.abstract_parser import DelegatingParser
from aphrodite.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from aphrodite.parser.harmony import HarmonyParser
from aphrodite.renderers.base import BaseRenderer
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
from aphrodite.tool_parsers.utils import get_json_schema_from_tools
from aphrodite.v1.structured_output import StructuredOutputManager

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

SCHEMA = {"type": "object", "dependentRequired": {"a": ["b"]}}
TOOL = {"type": "function", "function": {"name": "report", "strict": False, "parameters": SCHEMA}}


@pytest.mark.parametrize("strict", [False, True, None])
def test_response_format_strictness_on_both_apis(strict):
    schema = {"name": "reply", "schema": SCHEMA}
    if strict is not None:
        schema["strict"] = strict
    chat = ChatCompletionRequest(messages=[], response_format={"type": "json_schema", "json_schema": schema})
    response = ResponsesRequest(input="hi", text={"format": {"type": "json_schema", **schema}})
    for params in (chat.extract_structured_outputs(), response.extract_structured_outputs()):
        assert params is not None
        assert params.json_object is (True if strict is False else None)
        assert params.json == (None if strict is False else SCHEMA)


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "report"}}])
def test_non_strict_tool_waives_shape_without_mutating_request(choice):
    request = ChatCompletionRequest(messages=[], tools=[copy.deepcopy(TOOL)], tool_choice=choice)
    before = request.model_dump()
    schema = get_json_schema_from_tools(request.tool_choice, request.tools, parallel_tool_calls=False)
    assert request.model_dump() == before
    if choice == "required":
        assert schema["maxItems"] == 1
        schema = schema["items"]["anyOf"][0]["properties"]["parameters"]
    assert schema == {"type": "object", "additionalProperties": True}


def test_tool_none_uses_exclusion_from_first_token_and_preserves_reply_constraint():
    class Parser(DelegatingParser):
        tool_parser_cls = Qwen3ParserToolAdapter

    tokenizer = Mock()
    tokenizer.get_vocab.return_value = {}
    tokenizer.all_special_tokens = []
    tokenizer.all_special_ids = []
    parser = Parser(tokenizer)
    request = ChatCompletionRequest(messages=[], tools=[TOOL], tool_choice="none")
    parser.adjust_request(request)
    assert request._tool_exclusion_grammar
    assert request.structured_outputs.grammar
    assert request.resolve_reasoning_ended(parser, []) is True
    grammar = request.structured_outputs.grammar
    parser.adjust_request(request)
    assert request.structured_outputs.grammar == grammar
    assert not request.bad_words

    constrained = ChatCompletionRequest(
        messages=[], tools=[TOOL], tool_choice="none", structured_outputs=StructuredOutputsParams(regex="[a-z]+")
    )
    parser.adjust_request(constrained)
    assert constrained.structured_outputs.regex == "[a-z]+"
    assert constrained.bad_words
    assert not constrained._tool_exclusion_grammar


def test_harmony_recovery_keeps_rejected_tokens_without_repeating_streamed_prefix():
    tokenizer = Mock()
    tokenizer.decode.side_effect = lambda ids: "".join({1: "a", 2: "b", 3: "c"}[i] for i in ids)
    parser = HarmonyParser(tokenizer)
    parser._parser = Mock()
    parser._parser.messages = []
    parser._parser.current_channel = "final"
    parser._parser.current_recipient = None
    parser._parser.last_content_delta = "a"
    parser._parser.process.side_effect = [None, HarmonyError("invalid header")]
    parser._is_reasoning_token = lambda *args: False
    result = parser.process_chunk([1, 2, 3])
    assert "".join(s.delta for s in result.segments) == "a"
    recovered = parser.flush()
    assert "".join(s.delta for s in recovered) == "bc"
    assert recovered[-1].completed_message.content[0].text == "abc"
    assert parser._derailed_at is None


@pytest.mark.parametrize("skip_special", [True, False])
def test_completion_logprobs_hide_special_token_width_across_chunks(skip_special):
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.return_tokens_as_token_ids = False
    tokenizer = SimpleNamespace(all_special_ids=[2])
    ids, tokens = [1, 2, 3], ["a", "<eos>", "b"]
    logprobs = [{i: Logprob(logprob=-1.0, rank=1, decoded_token=t)} for i, t in zip(ids, tokens)]
    full = serving._create_completion_logprobs(ids, logprobs, 0, tokenizer, skip_special_tokens=skip_special)
    assert full.tokens == (["a", "", "b"] if skip_special else tokens)
    offsets = []
    cursor = 0
    for i, lp in zip(ids, logprobs):
        chunk = serving._create_completion_logprobs(
            [i], [lp], 0, tokenizer, initial_text_offset=cursor, skip_special_tokens=skip_special
        )
        offsets.extend(chunk.text_offset)
        cursor = chunk.text_offset[-1] + len(chunk.tokens[-1])
    assert offsets == full.text_offset == ([0, 1, 1] if skip_special else [0, 1, 6])


def test_force_bos_preserves_offsets_and_is_idempotent(monkeypatch):
    monkeypatch.setattr("aphrodite.envs.APHRODITE_FORCE_BOS", True)
    tokenizer = SimpleNamespace(bos_token_id=9)
    ids, offsets = BaseRenderer._ensure_bos(tokenizer, [1], [(0, 3)])
    assert (ids, offsets) == ([9, 1], [(0, 0), (0, 3)])
    assert BaseRenderer._ensure_bos(tokenizer, ids, offsets) == (ids, offsets)


def test_manager_compiles_with_each_requests_backend():
    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.backend = None
    manager.backends = {}
    manager.aphrodite_config = SimpleNamespace(num_speculative_tokens=0)
    backends = {name: Mock(tokenizer=None, vocab_size=32) for name in ["xgrammar", "guidance"]}
    manager._create_backend = lambda name: backends[name]
    for name in ["xgrammar", "guidance", "xgrammar"]:
        backend = manager._get_backend(name)
        request = SimpleNamespace(
            request_id=name,
            sampling_params=SamplingParams(structured_outputs=StructuredOutputsParams(regex="a")),
            structured_output_request=SimpleNamespace(structured_output_key=(None, "a")),
        )
        assert manager._create_grammar(request, backend) is backend.compile_grammar.return_value
    assert set(manager.backends) == {"xgrammar", "guidance"}
    assert backends["xgrammar"].compile_grammar.call_count == 2
    assert backends["guidance"].compile_grammar.call_count == 1


def test_kobold_greedy_preserves_requested_choices_and_zero_temperature():
    from aphrodite.entrypoints.openai.kobold.protocol import KAIGenerationInputSchema
    from aphrodite.entrypoints.openai.kobold.serving import OpenAIServingKobold

    serving = OpenAIServingKobold.__new__(OpenAIServingKobold)
    serving.badwordsids = []
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2]
    request = KAIGenerationInputSchema(prompt="hi", n=3, temperature=0.0, max_context_length=16, max_length=4)
    params, prompt_ids = serving._prepare_engine_payload(request, tokenizer)
    assert params.n == 3
    assert params.temperature == 0.0
    assert prompt_ids == [1, 2]
