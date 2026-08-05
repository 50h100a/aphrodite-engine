# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A grammar with no legal next token must end one request, not the engine."""

from unittest.mock import Mock

import pytest
import torch

from aphrodite.config import AphroditeConfig, ModelConfig, SchedulerConfig
from aphrodite.v1.engine import FinishReason
from aphrodite.v1.request import RequestStatus
from aphrodite.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

VOCAB_SIZE = 128
NUM_SPEC_TOKENS = 3


@pytest.fixture
def aphrodite_config():
    model_config = Mock(spec=ModelConfig)
    model_config.skip_tokenizer_init = True
    model_config.get_vocab_size = Mock(return_value=VOCAB_SIZE)
    model_config.is_diffusion = False

    scheduler_config = Mock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 8

    config = Mock(spec=AphroditeConfig)
    config.model_config = model_config
    config.scheduler_config = scheduler_config
    config.structured_outputs_config = Mock()
    config.structured_outputs_config.reasoning_parser = None
    config.structured_outputs_config.enable_in_reasoning = True
    config.speculative_config = Mock()
    config.speculative_config.num_speculative_tokens = NUM_SPEC_TOKENS
    config.num_speculative_tokens = NUM_SPEC_TOKENS
    return config


@pytest.fixture
def manager(aphrodite_config):
    manager = StructuredOutputManager(aphrodite_config)
    backend = Mock()
    backend.allocate_token_bitmask = Mock(
        side_effect=lambda rows, _=None: torch.zeros(rows, VOCAB_SIZE // 32, dtype=torch.int32)
    )
    manager.backends = {"xgrammar": backend}
    manager.tokenizer = Mock()
    return manager


def _request(accept_results):
    """A request whose grammar accepts drafts until `accept_results` says no."""
    grammar = Mock()
    grammar.is_terminated = Mock(return_value=False)
    grammar.accept_tokens = Mock(side_effect=accept_results)
    grammar.validate_tokens = Mock(side_effect=lambda toks: toks)
    grammar.fill_bitmask = Mock()
    grammar.rollback = Mock()

    request = Mock()
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar
    request.structured_output_request.reasoning_ended = True
    request.structured_output_request.reasoner = None
    request.structured_output_request.reasoning_parser_kwargs = None
    request.use_structured_output = True
    request.all_token_ids = [1, 2, 3]
    return request


def test_rejected_draft_does_not_raise(manager):
    """The engine core calls this unguarded, so raising kills the whole engine
    and every unrelated request sharing the batch. Drafts are speculative
    anyway: the token actually sampled may still be legal, and if it is not,
    update_from_output ends that one request with finish_reason=constraint."""
    request = _request([True, False, True])

    manager.grammar_bitmask({"req-0": request}, ["req-0"], {"req-0": [10, 11, 12]})


def test_bitmask_rows_stay_aligned_after_a_rejected_draft(manager):
    """Every scheduled position owes the runner one row. Bailing out early on
    the rejection would shift each later request's rows and silently constrain
    the wrong requests."""
    stuck = _request([False])
    healthy = _request([True, True, True])

    bitmask = manager.grammar_bitmask(
        {"req-0": stuck, "req-1": healthy},
        ["req-0", "req-1"],
        {"req-0": [10, 11, 12], "req-1": [20, 21, 22]},
    )

    # 3 speculative positions + 1 bonus, per request.
    assert bitmask.shape[0] == 2 * (NUM_SPEC_TOKENS + 1)


def test_rejected_draft_does_not_advance_the_grammar(manager):
    """The rollback at the end of the window must match the number of accepted
    advances, or the FSM is left pointing at the wrong state."""
    request = _request([True, False, True])

    manager.grammar_bitmask({"req-0": request}, ["req-0"], {"req-0": [10, 11, 12]})

    grammar = request.structured_output_request.grammar
    # Draft 1 advanced the FSM, draft 2 was rejected, and draft 3 was never
    # offered: the window stops being constrained once the grammar falls
    # behind. So exactly one advance happened and exactly one may be undone --
    # rolling back the rejected draft too would leave the FSM behind the
    # tokens the request has actually committed to.
    assert grammar.accept_tokens.call_count == 2
    assert grammar.rollback.call_count == 1
    assert grammar.rollback.call_args[0][0] == 1


def test_unconstraining_lasts_only_for_the_current_window(manager):
    """Dropping the bitmask after a bad draft must not outlive the step.

    A drafter can propose an illegal token without the target model accepting
    it; the target resamples at that position under a mask that was filled
    before the rejection, so the request carries on legally. If the drop
    persisted, every later token of that request would be generated
    unconstrained -- silently emitting output that violates the schema.
    """
    # Step 1: the second draft is rejected, so the rest of the window and only
    # the rest of the window goes unconstrained.
    request = _request([True, False])
    manager.grammar_bitmask({"req-0": request}, ["req-0"], {"req-0": [10, 11, 12]})

    grammar = request.structured_output_request.grammar
    # Rows 0 and 1 were filled from the grammar, row 2 was not; the bonus row
    # is filled again because it is derived from should_fill_bitmask, not from
    # the flag we cleared. That row constrains the token the target samples.
    assert grammar.fill_bitmask.call_count == 3

    # Step 2: a fresh call, and the grammar constrains every position again.
    grammar.fill_bitmask.reset_mock()
    grammar.accept_tokens = Mock(return_value=True)
    manager.grammar_bitmask({"req-0": request}, ["req-0"], {"req-0": [20, 21, 22]})

    assert grammar.fill_bitmask.call_count == NUM_SPEC_TOKENS + 1


def test_a_stuck_request_does_not_unconstrain_its_neighbours(manager):
    """apply_bitmask is per request; a shared one would let a single bad draft
    silently lift the schema off every other request in the batch."""
    stuck = _request([False])
    healthy = _request([True, True, True])

    manager.grammar_bitmask(
        {"req-0": stuck, "req-1": healthy},
        ["req-0", "req-1"],
        {"req-0": [10, 11, 12], "req-1": [20, 21, 22]},
    )

    assert healthy.structured_output_request.grammar.fill_bitmask.call_count == NUM_SPEC_TOKENS + 1


def test_constraint_status_maps_to_constraint_finish_reason():
    """The status is what the scheduler sets; the finish reason is what the
    client sees. A missing map entry would surface as a null finish_reason."""
    assert RequestStatus.is_finished(RequestStatus.FINISHED_CONSTRAINT)
    assert RequestStatus.get_finished_reason(RequestStatus.FINISHED_CONSTRAINT) is FinishReason.CONSTRAINT
    assert str(FinishReason.CONSTRAINT) == "constraint"
