# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
import time
from concurrent.futures import Future

import pytest
from transformers import AutoTokenizer

from aphrodite.config import AphroditeConfig, StructuredOutputsConfig
from aphrodite.config.model import ModelConfig
from aphrodite.config.parallel import ParallelConfig
from aphrodite.config.speculative import SpeculativeConfig
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
from aphrodite.tokenizers import get_tokenizer
from aphrodite.v1.request import Request
from aphrodite.v1.structured_output import StructuredOutputManager
from aphrodite.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    serialize_guidance_grammar,
    validate_guidance_grammar,
)
from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "gpt2"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return get_tokenizer(
        tokenizer_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tokenizer_mode="mistral",
    )


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    structured_outputs_config = StructuredOutputsConfig(backend="guidance")
    aphrodite_config = AphroditeConfig(structured_outputs_config=structured_outputs_config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    backend = GuidanceBackend(
        aphrodite_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    dummy_wrong = tokenizer.encode('{"a"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy_wrong)
    # Rollback is done from where state was terminated, so from '}' not EOS
    grammar.rollback(len(prompt) - 1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.validate_tokens(dummy_wrong) != dummy_wrong
    assert grammar.accept_tokens("", prompt[1:])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Rollback of <= 0 should not change the terminated state
    grammar.rollback(0)
    assert grammar.is_terminated()
    grammar.rollback(-1)
    assert grammar.is_terminated()


def test_grammar_bitmask_with_specdec():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')
    aphrodite_config = AphroditeConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    structured_output_manager = StructuredOutputManager(aphrodite_config)

    for i in range(1, 2):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                json='{"type": "object"}',
            ),
        )
        sampling_params.structured_outputs._backend = "guidance"
        sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
        )

        structured_output_manager.grammar_init(request)

        def grammar_bitmask(req: Request, tokens: list[int]) -> None:
            structured_output_manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids={req.request_id: 0},
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert not req.structured_output_request.grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion():
            continue

        assert request.structured_output_request.grammar.accept_tokens(request.request_id, prompt[:i])

        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id] + prompt)  # EOS not the final token
        grammar_bitmask(request, prompt[i:])  # EOS not present
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_init_async_and_sync(async_grammar):
    """Test grammar initialization works correctly in both async and sync modes.

    This test validates that the distributed_executor_backend config option
    correctly controls whether grammar compilation happens asynchronously
    (via executor.submit) or synchronously. When set to "external_launcher",
    grammar compilation is synchronous to avoid deadlocks.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')

    # Use "external_launcher" for sync mode, None for async mode
    executor_backend = None if async_grammar else "external_launcher"
    aphrodite_config = AphroditeConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        parallel_config=ParallelConfig(distributed_executor_backend=executor_backend),
    )
    structured_output_manager = StructuredOutputManager(aphrodite_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
        ),
    )
    sampling_params.structured_outputs._backend = "guidance"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "test_request",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    structured_output_manager.grammar_init(request)

    # Check the internal _grammar type immediately after init
    # Before _check_grammar_completion is called, async mode should have a Future
    raw_grammar = request.structured_output_request._grammar
    if async_grammar:
        assert isinstance(raw_grammar, Future), "Async mode should store a Future before completion"
    else:
        assert not isinstance(raw_grammar, Future), "Sync mode should store the grammar directly, not a Future"

    # Wait for grammar to be ready (handles both async and sync cases)
    start_time = time.time()
    while not request.structured_output_request._check_grammar_completion():
        if time.time() - start_time > 5:  # 5-second timeout
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)

    # After completion, _grammar should no longer be a Future
    assert not isinstance(request.structured_output_request._grammar, Future)

    # Verify grammar is properly initialized and functional
    grammar = request.structured_output_request.grammar
    assert grammar is not None
    assert not grammar.is_terminated()

    # Verify the grammar can accept valid tokens
    assert grammar.accept_tokens(request.request_id, prompt)


@pytest.mark.parametrize(
    "request_type,grammar_spec",
    [
        pytest.param(
            StructuredOutputOptions.JSON,
            '{"type": "object"}',
            id="json",
        ),
        pytest.param(
            StructuredOutputOptions.GRAMMAR,
            'start: "hello" | "world"',
            id="lark",
        ),
    ],
)
def test_mistral_tokenizer_compile_grammar(
    mistral_tokenizer,
    request_type: StructuredOutputOptions,
    grammar_spec: str,
) -> None:
    aphrodite_config = AphroditeConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
        aphrodite_config,
        tokenizer=mistral_tokenizer,
        vocab_size=mistral_tokenizer.vocab_size,
    )
    assert backend.ll_tokenizer is mistral_tokenizer.llg_tokenizer

    grammar = backend.compile_grammar(request_type, grammar_spec)
    assert grammar is not None
    assert not grammar.is_terminated()


# --------------------------------------------------------------------------
# Structural tags: the schema has to survive the trip into llguidance.
#
# StructTag takes a JSON schema, but serialize_guidance_grammar used to hand it
# _process_schema()'s *compiled* grammar. StructTag treats any string starting
# with "{" as a schema, so llguidance read the {"grammars": ...} wrapper as one.
# It holds no schema keywords, so every tag quietly degraded to "any JSON
# value": tool arguments went unconstrained, and schema errors inside a tag
# were invisible to validate_grammar().
# --------------------------------------------------------------------------

# Plain-ASCII trigger so the gpt2 tokenizer can represent it. Real templates use
# special tokens; the trigger is literal text either way and has no bearing on
# how the argument schema is compiled.
TAG_TRIGGER = "TOOLCALL"
TAG_BEGIN = "TOOLCALL get_weather\n"
TAG_END = "\nDONE"

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["location"],
    "additionalProperties": False,
}
CONFORMING_ARGS = '{"location": "Paris", "units": "celsius"}'


def _tool_structural_tag(schema):
    return json.dumps(
        {
            "triggers": [TAG_TRIGGER],
            "structures": [{"begin": TAG_BEGIN, "schema": schema, "end": TAG_END}],
        }
    )


@pytest.fixture(scope="module")
def llg_tokenizer():
    import llguidance.hf

    return llguidance.hf.from_tokenizer(AutoTokenizer.from_pretrained(TOKENIZER), 50257)


def _tag_grammar(schema, **kwargs):
    return serialize_guidance_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _tool_structural_tag(schema),
        **kwargs,
    )


def _accepts(grammar, llg_tokenizer, payload: str) -> bool:
    """Whether the grammar permits ``payload`` as the tag's structured body."""
    import llguidance

    matcher = llguidance.LLMatcher(llg_tokenizer, grammar, log_level=0)
    assert matcher.consume_tokens(llg_tokenizer.tokenize_str(TAG_BEGIN)), "matcher rejected the tag's own begin string"
    return all(matcher.consume_tokens([token]) for token in llg_tokenizer.tokenize_str(payload))


@pytest.mark.parametrize(
    "violation,why",
    [
        ('{"location": 42}', "location must be a string"),
        ('{"units": "kelvin"}', "units outside the enum, required location missing"),
        ("[1, 2, 3]", "not an object"),
        ('"just a string"', "not an object"),
    ],
)
def test_structural_tag_enforces_argument_schema(llg_tokenizer, violation, why):
    """Tool arguments violating the schema must be blocked, not just the shape
    of the tag around them. These all decoded happily when the tag degraded to
    "any JSON value"."""
    assert not _accepts(_tag_grammar(TOOL_SCHEMA), llg_tokenizer, violation), why


def test_structural_tag_accepts_conforming_arguments(llg_tokenizer):
    """The constraint must not be so tight it rejects valid arguments."""
    assert _accepts(_tag_grammar(TOOL_SCHEMA), llg_tokenizer, CONFORMING_ARGS)


@pytest.mark.parametrize(
    "disable_any_whitespace,spaced_json_allowed",
    [(False, True), (True, False)],
)
def test_structural_tag_honours_whitespace_option(llg_tokenizer, disable_any_whitespace, spaced_json_allowed):
    """`disable_any_whitespace` reached llguidance through `defaults=` when the
    schema was pre-compiled; with a raw schema it rides along as `x-guidance`.
    Either way the option has to keep working."""
    grammar = _tag_grammar(TOOL_SCHEMA, disable_any_whitespace=disable_any_whitespace)
    assert _accepts(grammar, llg_tokenizer, CONFORMING_ARGS) is spaced_json_allowed
    # Whitespace-free JSON is legal under both settings.
    assert _accepts(grammar, llg_tokenizer, '{"location":"Paris"}')


def test_structural_tag_keeps_explicit_x_guidance(llg_tokenizer):
    """A schema carrying its own x-guidance keeps it; we only supply a default."""
    schema = copy.deepcopy(TOOL_SCHEMA)
    schema["x-guidance"] = {"whitespace_flexible": False}
    grammar = _tag_grammar(schema, disable_any_whitespace=False)
    assert not _accepts(grammar, llg_tokenizer, CONFORMING_ARGS)


def test_structural_tag_does_not_mutate_caller_schema():
    """Serializing must not write x-guidance back into the caller's dict."""
    schema = copy.deepcopy(TOOL_SCHEMA)
    _tag_grammar(schema)
    assert schema == TOOL_SCHEMA


def test_structural_tag_reports_unsatisfiable_schema(llg_tokenizer):
    """An unsatisfiable tool schema has to surface as a validation error.

    Nothing inside a tag was compiled as a schema before, so llguidance had
    nothing to object to and the request was admitted -- then died in the
    engine. A plain JSON request already caught this; the tag path must agree.
    """
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": []}},
        "required": ["mode"],
    }
    params = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=_tool_structural_tag(schema)))
    with pytest.raises(ValueError, match="[Uu]nsatisfiable"):
        validate_guidance_grammar(params, tokenizer=llg_tokenizer)


def test_structural_tag_accepts_valid_schema(llg_tokenizer):
    """Sanity: ordinary tool schemas still validate."""
    tag = _tool_structural_tag(TOOL_SCHEMA)
    validate_guidance_grammar(
        SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=tag)),
        tokenizer=llg_tokenizer,
    )


# Every test above builds the `triggers`/`structures` form, which is the only
# one serialize_guidance_grammar can compile -- and, because they all build it,
# the only one they ever exercised. Tool calls are built as the nested `format`
# form instead, so the guidance backend has never been able to serve one.


def test_structural_tag_nested_format_is_declined_not_crashed(llg_tokenizer):
    """The unsupported tag shape has to be refused as a ValueError.

    Callers separate "this backend will not take it" from "something broke" by
    the exception type: the `auto` backend catches ValueError to try the next
    backend, and the API layer turns it into a 400 naming the reason. Reading
    s_tag["triggers"] on a tag that has no such key raised KeyError, which
    matches neither -- so a request with a tool schema xgrammar happened to
    dislike came back as a 500 with an empty message.
    """
    tag = json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": TAG_BEGIN,
                "content": {"type": "json_schema", "json_schema": TOOL_SCHEMA},
                "end": TAG_END,
            },
        }
    )
    params = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=tag))
    with pytest.raises(ValueError, match="cannot compile this structural tag"):
        validate_guidance_grammar(params, tokenizer=llg_tokenizer)


def test_structural_tag_from_the_registry_is_declined_not_crashed(llg_tokenizer):
    """The same, on a tag built the way a real tool call builds one.

    Pinning the hand-written shape above is not enough on its own: if the
    builder's output ever moves, this is what notices. Whether guidance grows
    support for it or keeps refusing, the answer must stay a ValueError.
    """
    from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag

    tools = [
        ChatCompletionToolsParam.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather.",
                    "parameters": TOOL_SCHEMA,
                },
            }
        )
    ]
    structural_tag = get_model_structural_tag(
        model="deepseek_v4",
        tools=tools,
        tool_choice="required",
        reasoning=True,
    )
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(structural_tag=structural_tag.model_dump_json())
    )
    with pytest.raises(ValueError, match="cannot compile this structural tag"):
        validate_guidance_grammar(params, tokenizer=llg_tokenizer)
