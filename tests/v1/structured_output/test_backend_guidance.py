# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    process_for_additional_properties,
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
    # The rewind is counted in tokens the caller was told were accepted, which
    # here is everything past the first: the rest of the prompt, the EOS, and
    # the junk after it. Only the prompt tokens ever moved the matcher, so only
    # those are undone and it lands just after prompt[0] -- from '}' not EOS.
    grammar.rollback((len(prompt) - 1) + 1 + len(dummy_wrong))
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


@pytest.mark.parametrize(
    "tail_len,why",
    [
        (1, "EOS alone: the one case the old rollback_lag counted"),
        (4, "EOS and then some, which is what a 5-token draft window looks like"),
        (2, "no EOS at all: the matcher can stop on the closing brace"),
    ],
)
def test_rollback_does_not_rewind_tokens_the_matcher_never_took(tail_len, why):
    """A stopped matcher accepts without consuming, and must rewind by as much.

    The caller counts the tokens it was told were accepted and asks for exactly
    that many back. Tokens offered to an already-stopped matcher move nothing,
    so counting them into the rewind takes the matcher back past the start of
    the window and into text the model has already emitted -- after which the
    bitmask constrains against a prefix that is several tokens stale, and the
    reply comes apart. Only reachable under speculative decoding, where a whole
    draft window can be offered past the point the grammar was satisfied.
    """
    structured_outputs_config = StructuredOutputsConfig(backend="guidance")
    aphrodite_config = AphroditeConfig(structured_outputs_config=structured_outputs_config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = GuidanceBackend(aphrodite_config, tokenizer=tokenizer, vocab_size=50257)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

    prompt = tokenizer.encode('{"a": "b"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    # The object is complete, so the matcher is done and every further token is
    # accepted without moving it.
    assert grammar.ll_matcher.is_stopped()

    tail = ([tokenizer.eos_token_id] + prompt)[:tail_len]
    # Feed the window the way the bitmask loop does: one token per call,
    # counting each acceptance, then rewinding by the count.
    advancements = sum(1 for token in tail if grammar.accept_tokens("", [token]))
    assert advancements == tail_len
    grammar.rollback(advancements)

    # Still sitting on the closing brace, not somewhere back inside the object.
    assert grammar.ll_matcher.is_stopped(), why
    assert grammar.validate_tokens(prompt) == []


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
        sampling_params.structured_outputs._backend = "guidance"  # type: ignore[union-attr]
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
                structured_output_request_ids=[req.request_id],
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert req.structured_output_request is not None
            grammar = req.structured_output_request.grammar
            assert grammar is not None and not isinstance(grammar, Exception)
            assert not grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        assert request.structured_output_request is not None
        while not request.structured_output_request._check_grammar_completion():
            continue

        grammar = request.structured_output_request.grammar
        assert grammar is not None and not isinstance(grammar, Exception)
        assert grammar.accept_tokens(request.request_id, prompt[:i])

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
    sampling_params.structured_outputs._backend = "guidance"  # type: ignore[union-attr]
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
    assert request.structured_output_request is not None
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
    assert grammar is not None and not isinstance(grammar, Exception)
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


def test_process_for_additional_properties_preserves_literals() -> None:
    """Literal values under const/enum/default/examples must not be rewritten
    when they contain a 'properties' or 'patternProperties' key (#58695)."""
    feature = {
        "type": "Feature",
        "geometry": None,
        "properties": {"name": "HQ"},
        "patternProperties": {"^x": "y"},
    }
    schema = {
        "type": "object",
        "properties": {
            "const_field": {"const": feature},
            "enum_field": {"enum": [feature]},
            "default_field": {"type": "object", "default": feature},
            "examples_field": {"type": "object", "examples": [feature]},
            "properties": {"type": "string"},
            "const": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
            },
        },
        "$defs": {
            "item": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
            }
        },
    }

    processed = process_for_additional_properties(schema)

    assert processed["additionalProperties"] is False
    assert processed["$defs"]["item"]["additionalProperties"] is False
    assert processed["properties"]["const"]["additionalProperties"] is False
    assert "additionalProperties" not in processed["properties"]
    assert processed["properties"]["const_field"]["const"] == feature
    assert processed["properties"]["enum_field"]["enum"] == [feature]
    assert processed["properties"]["default_field"]["default"] == feature
    assert processed["properties"]["examples_field"]["examples"] == [feature]


@pytest.mark.parametrize("keyword", ["const", "enum"])
def test_disable_additional_properties_accepts_literal_with_properties_key(
    keyword: str,
) -> None:
    """GuidanceBackend with disable_additional_properties=True must accept the
    original literal value under const/enum and reject injected
    additionalProperties (#58695)."""
    feature = {"type": "Feature", "geometry": None, "properties": {"name": "HQ"}}
    rewritten = {**feature, "additionalProperties": False}
    schema = {keyword: feature if keyword == "const" else [feature]}

    aphrodite_config = AphroditeConfig(
        structured_outputs_config=StructuredOutputsConfig(
            backend="guidance",
            disable_additional_properties=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = GuidanceBackend(
        aphrodite_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON,
        json.dumps(schema),
    )

    valid_tokens = tokenizer.encode(json.dumps(feature)) + [tokenizer.eos_token_id]
    rewritten_tokens = tokenizer.encode(json.dumps(rewritten)) + [tokenizer.eos_token_id]

    assert grammar.validate_tokens(valid_tokens) == valid_tokens
    assert grammar.validate_tokens(rewritten_tokens) != rewritten_tokens
