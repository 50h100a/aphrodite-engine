# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from aphrodite import SamplingParams


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(ValueError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize("n", [1, 2, 8])
def test_greedy_allows_n_greater_than_one(n: int):
    """Zero temperature no longer restricts `n` to 1; the n children are
    simply identical."""
    params = SamplingParams(n=n, temperature=0.0)
    assert params.n == n
    # Zero temperature still flattens the truncation params.
    assert (params.top_p, params.top_k, params.min_p, params.top_a) == (1.0, 0, 0.0, 0.0)


def test_greedy_with_dynatemp_allows_n_greater_than_one():
    """`temperature=0` plus dynatemp is not actually greedy at sample time."""
    params = SamplingParams(n=4, temperature=0.0, dynatemp_max=2.0)
    assert params.n == 4
