# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise speculative bias through processor loading and sampler dispatch."""

from types import SimpleNamespace

import pytest
import torch

from aphrodite import SamplingParams
from aphrodite.v1.sample.logits_processor import build_logitsprocs
from aphrodite.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor
from aphrodite.v1.sample.logits_processor.interface import BatchUpdate
from aphrodite.v1.sample.rejection_sampler import RejectionSampler

pytestmark = pytest.mark.cpu_test


def _config():
    return SimpleNamespace(speculative_config=object(), scheduler_config=SimpleNamespace(max_num_seqs=4))


def test_speculative_bias_reaches_each_requests_target_rows():
    processors = build_logitsprocs(_config(), torch.device("cpu"), False, False)
    update = BatchUpdate(
        batch_size=4,
        removed=[],
        moved=[],
        added=[
            (0, SamplingParams(logit_bias={1: 5.0}), [], []),
            (1, SamplingParams(logit_bias={2: 50.0}), [], []),
            (2, SamplingParams(logit_bias={3: -4.0}), [], []),
            (3, SamplingParams(), [], []),
        ],
    )
    for processor in processors.all:
        processor.update_state(update)
    sampling = SimpleNamespace(
        no_penalties=True,
        bad_words_token_ids=[],
        output_token_ids=[[], [], [], []],
        allowed_token_ids_mask=None,
        logitsprocs=processors,
        thinking_budget_state_holder=None,
    )
    result = RejectionSampler.apply_logits_processors(
        SimpleNamespace(), torch.zeros(6, 4), sampling, SimpleNamespace(num_draft_tokens=[2, 0, 3, 1])
    )
    assert result.tolist() == [[0, 5, 0, 0], [0, 5, 0, 0], [0, 0, 0, -4], [0, 0, 0, -4], [0, 0, 0, -4], [0, 0, 0, 0]]


def test_bias_preserves_grammar_masked_logits():
    processor = LogitBiasLogitsProcessor(None, torch.device("cpu"), False)
    processor.biases = {0: {1: 100.0}}
    result = processor.apply_with_spec_decode(torch.tensor([[0.0, -float("inf")]]), [1])
    assert torch.isneginf(result[0, 1])


def test_custom_speculative_processors_are_still_rejected():
    with pytest.raises(ValueError, match="Custom logits processors"):
        build_logitsprocs(_config(), torch.device("cpu"), False, False, custom_logitsprocs=[LogitBiasLogitsProcessor])
