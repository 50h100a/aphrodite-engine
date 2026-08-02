# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""grammar_bitmask under spec-decode draft padding (#44006)."""

import pytest
from transformers import AutoTokenizer

from aphrodite.config import AphroditeConfig, StructuredOutputsConfig
from aphrodite.config.model import ModelConfig
from aphrodite.config.speculative import SpeculativeConfig
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
from aphrodite.v1.request import Request
from aphrodite.v1.structured_output import StructuredOutputManager

TOKENIZER = "gpt2"
NUM_SPEC_TOKENS = 4


def _make_manager_and_request(backend: str, prompt_str: str = '{"a": "b"}'):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode(prompt_str)

    aphrodite_config = AphroditeConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS),
    )
    manager = StructuredOutputManager(aphrodite_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    sampling_params.structured_outputs._backend = backend
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "mtp_req",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    while not request.structured_output_request._check_grammar_completion():
        continue

    return tokenizer, manager, request, prompt


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_accept_tokens_stops_at_grammar_termination(backend):
    """A committed chunk can continue past the token that ends the grammar.

    Under speculative decoding a step commits several tokens at once, so the
    grammar's stop token is usually not the last one in the chunk -- EOS follows
    it. Everything after termination must be left alone: xgrammar's matcher
    warns once per token ("The matcher has terminated after accepting the stop
    token, but is trying to accept new token with id ...") when it is pushed
    past its end state, which floods the log on every structured request.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar
    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode(" ")[0]

    assert grammar.accept_tokens(request.request_id, [*prompt, eos, trailing, trailing])
    assert grammar.is_terminated()

    # Nothing is grammar-valid past the end of the grammar, so a draft arriving
    # in the same window as the stop token must not be probed either.
    assert grammar.validate_tokens([trailing]) == []

    if backend == "xgrammar":
        # The two trailing tokens were not consumed: the matcher stopped at EOS.
        assert grammar.num_processed_tokens == len(prompt) + 1


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_with_padded_invalid_drafts(backend):
    """Bitmask handles -1 padded drafts and returns N+1 rows."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend, prompt_str='{"a"')
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid_drafts = [tokenizer.encode(":")[0], tokenizer.encode(' "')[0]]
    padded = valid_drafts + [-1, -1]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_when_grammar_terminates_mid_window(backend):
    """Drafts following an EOS that terminates the grammar are a no-op."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)
    eos = tokenizer.eos_token_id
    drafts = [eos] + [tokenizer.encode(" ")[0]] * (NUM_SPEC_TOKENS - 1)

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == NUM_SPEC_TOKENS + 1
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_idempotent_across_calls(backend):
    """Repeated calls with the same input return the same bitmask."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend, prompt_str='{"a"')
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    drafts = [tokenizer.encode(":")[0], -1, -1, -1]

    first = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    second = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert first is not None and second is not None
    assert (first == second).all()
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bonus_position_constrained_after_invalid_drafts(backend):
    """Regression for #44006: bonus row stays constrained after -1 padding."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend, prompt_str='{"a"')
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid = tokenizer.encode(":")[0]
    drafts = [valid, -1, -1, -1]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1

    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_constrained_when_reasoning_ends_midwindow(backend):
    """Drafts after a mid-window reasoning-end marker stay constrained."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n")[0]

    class StubReasoner:
        def __init__(self, *_, **__):
            self.end_token_id = marker

        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = StubReasoner
    request.structured_output_request.reasoner = StubReasoner()
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode(",")[0]
    drafts = [pre, marker, post]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    assert (bitmask[0] == -1).all()
    assert (bitmask[1] == -1).all()
    assert not (bitmask[2] == -1).all()
    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_post_reasoning_end_drafts_skip_grammar_advance(backend):
    """Post-marker drafts predate the bitmask and may be grammar-invalid.

    grammar_bitmask must skip the grammar advance instead of asserting.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request(backend, prompt_str="{")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)
    assert not grammar.is_terminated()

    marker = tokenizer.encode("\n")[0]

    class StubReasoner:
        def __init__(self, *_, **__):
            self.end_token_id = marker

        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = StubReasoner
    request.structured_output_request.reasoner = StubReasoner()
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    invalid_post = tokenizer.encode("z")[0]
    drafts = [pre, marker, invalid_post]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    assert not (bitmask[2] == -1).all()
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_validate_tokens_then_bitmask_round_trip(backend):
    """validate_tokens -> pad with -1 -> grammar_bitmask must not assert."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    raw_drafts = [tokenizer.encode(",")[0], 99999, 12345, 67890]
    accepted = grammar.validate_tokens(raw_drafts)
    assert len(accepted) <= len(raw_drafts)

    padded = accepted + [-1] * (len(raw_drafts) - len(accepted))
    assert len(padded) == len(raw_drafts)

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()


class _MarkerReasoner:
    """Stub reasoner whose reasoning-end marker is a single fixed token."""

    def __init__(self, marker: int):
        self.marker = marker

    def is_reasoning_end(self, input_ids):
        return self.marker in list(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self.marker in list(delta_ids)


def _setup_json_boundary_request(backend: str):
    """Request with a plain JSON key and reasoning not yet ended."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    marker = tokenizer.encode("\n")[0]
    structured_req = request.structured_output_request
    manager.reasoner_cls = _MarkerReasoner
    structured_req.reasoner = _MarkerReasoner(marker)
    structured_req.reasoning_ended = False
    return tokenizer, manager, request, prompt, marker


def _setup_boundary_request(backend: str):
    """Request with a structural-tag key and reasoning not yet ended."""
    from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions

    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    marker = tokenizer.encode("\n")[0]
    structured_req = request.structured_output_request
    # The grammar itself is JSON (cheap to build); only the key kind matters
    # for the should_advance structural-tag branch, so pre-seed the cached
    # property.
    structured_req.__dict__["structured_output_key"] = (
        StructuredOutputOptions.STRUCTURAL_TAG,
        "",
    )
    manager.reasoner_cls = _MarkerReasoner
    structured_req.reasoner = _MarkerReasoner(marker)
    structured_req.reasoning_ended = False
    return tokenizer, manager, request, prompt, marker


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_json_grammar_advances_past_reasoning_end_in_spec_window(backend):
    """A spec window that ends reasoning must still commit its grammar content.

    The window is [reasoning, marker, "{"]. Deferring the whole step -- correct
    when a step is one token, since then the marker *is* the step -- drops that
    "{" from the matcher for good. The matcher then sits at the JSON root while
    the model has already opened the object, so the bitmask forces a second "{"
    and the following accept_tokens fails with "Failed to advance FSM".
    """
    tokenizer, manager, request, _prompt, marker = _setup_json_boundary_request(backend)
    grammar = request.structured_output_request.grammar

    pre = tokenizer.encode(" ")[0]
    open_brace = tokenizer.encode("{")[0]
    step_tokens = [pre, marker, open_brace]
    request.append_output_token_ids(step_tokens)
    # Steady-state decode invariant: the rejection adjustment in
    # update_from_output() keeps num_computed_tokens at len(all_token_ids) - 1
    # however many tokens the step accepted, so deriving the step boundary from
    # it would scan the final token only and miss the marker entirely.
    request.num_computed_tokens = len(request.all_token_ids) - 1

    assert manager.should_advance(request, step_tokens)
    advance = manager.trim_reasoning_for_advance(request, list(step_tokens))
    assert advance == [open_brace]
    assert grammar.accept_tokens(request.request_id, advance)

    # The matcher is inside the object now, so a key may start. Had the "{"
    # been dropped it would still be at the root, where only "{" is legal.
    quote = tokenizer.encode('"')[0]
    assert grammar.validate_tokens([quote]) == [quote]
    assert grammar.validate_tokens([open_brace]) == []


def test_should_advance_records_reasoning_end_index():
    """The boundary step must record where reasoning ends."""
    tokenizer, manager, request, prompt, marker = _setup_boundary_request("xgrammar")
    structured_req = request.structured_output_request

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode("{")[0]
    step_tokens = [pre, marker, post]
    request.append_output_token_ids(step_tokens)
    request.num_computed_tokens = len(request.all_token_ids) - 1

    assert manager.should_advance(request, step_tokens)
    assert structured_req.reasoning_ended
    assert structured_req.reasoning_end_token_index == len(prompt) + 1


def test_trim_reasoning_for_advance():
    """Trim drops the marker and everything before it."""
    tokenizer, manager, request, prompt, marker = _setup_boundary_request("xgrammar")
    structured_req = request.structured_output_request

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode("{")[0]

    assert manager.trim_reasoning_for_advance(request, [pre]) == [pre]

    step_tokens = [pre, marker, post]
    request.append_output_token_ids(step_tokens)
    request.num_computed_tokens = len(request.all_token_ids) - 1
    assert manager.should_advance(request, step_tokens)
    assert manager.trim_reasoning_for_advance(request, step_tokens) == [post]

    structured_req.reasoning_end_token_index = len(request.all_token_ids) - 1
    assert manager.trim_reasoning_for_advance(request, step_tokens) == []

    structured_req.reasoning_end_token_index = len(prompt) + 1
    next_step = [post, post]
    request.append_output_token_ids(next_step)
    assert manager.trim_reasoning_for_advance(request, next_step) == next_step
