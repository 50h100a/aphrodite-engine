# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`_bookkeeping_sync` must not write a request's token buffer past
`max_model_len`.

Under async scheduling the model runner advances `num_tokens_no_spec` once per
step and only ever re-syncs it downward, so it can reach `max_model_len` while
the scheduler still hands out one more step. That used to trip a bare `assert`
inside `sample_tokens`, which is unrecoverable: it propagates out through
`step_with_batch_queue` and kills EngineCore, taking every in-flight request and
the API server with it (see the AssertionError in error.log:

    Sampled token IDs exceed the max model length.
    Total number of tokens: 131073 > max_model_len: 131072
).

The correct behaviour is to drop the tokens that do not fit: the request is at
the context limit, so the scheduler finishes it as length-capped and those
tokens could never have been emitted.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

MAX_MODEL_LEN = 32


def _make_runner(num_tokens_no_spec: int, use_async_scheduling: bool):
    """A stub `self` carrying only what `_bookkeeping_sync` touches."""
    req_id = "req0"
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        generators={},
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
        vocab_size=128,
        num_tokens_no_spec=np.array([num_tokens_no_spec], dtype=np.int32),
        token_ids_cpu=np.zeros((1, MAX_MODEL_LEN), dtype=np.int32),
        is_token_ids=np.zeros((1, MAX_MODEL_LEN), dtype=bool),
        persistent_data={},
    )
    req_state = SimpleNamespace(output_token_ids=[], persistent_data={})
    return SimpleNamespace(
        input_batch=input_batch,
        discard_request_mask=SimpleNamespace(np=np.zeros(1, dtype=bool)),
        use_async_scheduling=use_async_scheduling,
        max_model_len=MAX_MODEL_LEN,
        requests={req_id: req_state},
        routed_experts_initialized=False,
        _get_prompt_logprobs_dict=lambda *a, **kw: {},
        _to_list=lambda t: t.tolist(),
    )


def _run(runner, sampled_token_ids):
    sampler_output = SimpleNamespace(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=None,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req0": 1},
    )
    return GPUModelRunner._bookkeeping_sync(
        runner,
        scheduler_output,
        sampler_output,
        None,  # logits
        torch.zeros(1, 1),  # hidden_states
        1,  # num_scheduled_tokens
    )


@pytest.mark.parametrize("start", [MAX_MODEL_LEN, MAX_MODEL_LEN + 3])
def test_async_bookkeeping_clamps_at_max_model_len(start):
    """No room left: the sampled placeholder is dropped, not written OOB."""
    runner = _make_runner(num_tokens_no_spec=start, use_async_scheduling=True)
    _run(runner, torch.tensor([[7]], dtype=torch.int32))

    # Must not have grown past the context window, and must not have written
    # outside `token_ids_cpu` (which is only `max_model_len` wide).
    assert runner.input_batch.num_tokens_no_spec[0] == start
    assert runner.requests["req0"].output_token_ids == []


def test_async_bookkeeping_writes_the_last_slot():
    """One slot left: it is used, and the counter lands exactly on the limit."""
    runner = _make_runner(num_tokens_no_spec=MAX_MODEL_LEN - 1, use_async_scheduling=True)
    _run(runner, torch.tensor([[7]], dtype=torch.int32))

    assert runner.input_batch.num_tokens_no_spec[0] == MAX_MODEL_LEN
    assert runner.requests["req0"].output_token_ids == [-1]
    assert runner.input_batch.is_token_ids[0, MAX_MODEL_LEN - 1]


def test_sync_bookkeeping_truncates_partial_overflow():
    """Sync scheduling with spec decode: only the tokens that fit are kept."""
    runner = _make_runner(num_tokens_no_spec=MAX_MODEL_LEN - 2, use_async_scheduling=False)
    # Three sampled tokens (bonus + 2 accepted drafts) but only two slots left.
    _run(runner, torch.tensor([[11, 12, 13]], dtype=torch.int32))

    assert runner.input_batch.num_tokens_no_spec[0] == MAX_MODEL_LEN
    assert runner.requests["req0"].output_token_ids == [11, 12]


@pytest.mark.parametrize(
    "confirmed, output_count, expected", [([20], 1, [20]), ([20], 3, [20, -1, -1]), ([], 2, [-1, -1])]
)
def test_async_resume_never_reuses_prompt_tokens_as_output(monkeypatch, confirmed, output_count, expected):
    from aphrodite.v1.worker import gpu_model_runner

    monkeypatch.setattr(gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    state = SimpleNamespace(
        output_token_ids=[], num_prompt_tokens=3, prev_num_draft_len=0, num_computed_tokens=0, block_ids=([0],)
    )
    batch = Mock()
    batch.req_id_to_index = {}
    runner = SimpleNamespace(
        requests={"r": state},
        input_batch=batch,
        speculative_config=None,
        use_async_scheduling=True,
        use_async_spec_decode=False,
        late_interaction_runner=Mock(),
        _process_encoder_cache_scheduler_output=lambda _: None,
        _may_reorder_batch=lambda _: None,
    )
    scheduled = SimpleNamespace(
        phrase_rewinds=None,
        finished_req_ids=set(),
        new_block_ids_to_zero=[],
        kv_cache_block_copies=[],
        num_scheduled_tokens={"r": 1},
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r"],
            resumed_req_ids={"r"},
            num_computed_tokens=[3],
            new_block_ids=[([1],)],
            num_output_tokens=[output_count],
            all_token_ids={"r": [10, 11, 12, *confirmed]},
        ),
    )
    GPUModelRunner._update_states(runner, scheduled)
    assert state.output_token_ids == expected
    batch.add_request.assert_called_once_with(state)
