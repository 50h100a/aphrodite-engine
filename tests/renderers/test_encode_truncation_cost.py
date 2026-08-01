# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""It turns out that tokenizers don't run any faster with truncation, so
there's little point having them do so by default in `get_encode_kwargs` within
`aphrodite/renderers/params.py`.

Might as well tokenize the whole thing so any downstream error messages have
the correct token counts to hand back to the user.

These tests check whether tokenizers are still saving no compute on truncation.
"""

import time

import pytest
from transformers import AutoTokenizer

from aphrodite.renderers import TokenizeParams
from aphrodite.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from aphrodite.tokenizers.deepseek_v32 import DeepseekV32Tokenizer
from aphrodite.tokenizers.kimi_audio import KimiAudioTokenizer
from aphrodite.tokenizers.mistral import MistralTokenizer

# A bunch of tokenizers to test, split between STANDARD (uses hf/inkling) and
# NONSTANDARD (has custom logic somewhere in _APHRODITE_TOKENIZERS)
STANDARD_TOKENIZERS = [
    pytest.param("openai-community/gpt2", id="gpt2-bpe"),
    pytest.param("facebook/opt-125m", id="opt-bpe"),
    pytest.param("bert-base-uncased", id="bert-wordpiece"),
    pytest.param("NousResearch/Hermes-3-Llama-3.1-8B", id="llama3"),
    pytest.param("Qwen/Qwen3-8B", id="qwen3"),
    pytest.param("NousResearch/Llama-2-7b-hf", id="llama2-sentencepiece"),
    pytest.param("openai/gpt-oss-20b", id="gptoss-o200k"),
]
NONSTANDARD_TOKENIZERS = [
    pytest.param(DeepseekV4Tokenizer, "deepseek-ai/DeepSeek-V4-Flash", id="deepseek-v4"),
    pytest.param(DeepseekV32Tokenizer, "deepseek-ai/DeepSeek-V3.2-Exp", id="deepseek-v32"),
    pytest.param(MistralTokenizer, "mistralai/Magistral-Small-2509", id="mistral-tekken"),
    pytest.param(KimiAudioTokenizer, "moonshotai/Kimi-Audio-7B-Instruct", id="kimi-audio-tiktoken"),
]

_TEXT = " hi hello yes this is at least ten tokens." * 10_000  # ~100k tokens
_REPEATS = 3

# Truncate at 1/_TRUNC_FRACTION the actual count, and then expect the tokenizer
# to run in less than 1/_MINIMUM_SPEEDUP the time.
_TRUNC_FRACTION = 8
_MINIMUM_SPEEDUP = 2


def _best_encode_time(tokenizer, **encode_kwargs) -> tuple[float, int]:
    """Tokenize a few times and take the fastest one"""
    best = float("inf")
    count = 0
    for _ in range(_REPEATS):
        start = time.perf_counter()
        input_ids = tokenizer(_TEXT, add_special_tokens=False, **encode_kwargs)["input_ids"]
        best = min(best, time.perf_counter() - start)
        count = len(input_ids)

    return best, count


def _assert_truncation_saves_no_work(tokenizer, model: str) -> None:
    """Capping at `max_length` is not cheaper than encoding the whole prompt."""
    full_time, full_len = _best_encode_time(tokenizer, truncation=False)
    trunc_tokens = full_len // _TRUNC_FRACTION

    truncated_time, truncated_len = _best_encode_time(tokenizer, truncation=True, max_length=trunc_tokens)

    assert truncated_len == trunc_tokens, "tokenizer did not truncate according to max_length arg"

    speedup = full_time / truncated_time
    assert speedup < _MINIMUM_SPEEDUP, (
        f"{model} ({type(tokenizer).__name__}) encoded {trunc_tokens} of {full_len} tokens "
        f"{speedup:.2f}x faster than the the whole prompt,"
        f"({truncated_time * 1000:.0f}ms vs {full_time * 1000:.0f}ms). "
        f"It may be worth altering get_encode_kwargs's truncation behavior."
    )


@pytest.mark.parametrize("model", STANDARD_TOKENIZERS)
def test_standard_tokenizer_truncation_saves_no_work(model: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # offline, gated repo, ...
        pytest.skip(f"{model} unavailable: {type(exc).__name__}: {exc}")

    _assert_truncation_saves_no_work(tokenizer, model)


@pytest.mark.parametrize("tokenizer_cls,model", NONSTANDARD_TOKENIZERS)
def test_aphrodite_tokenizer_truncation_saves_no_work(tokenizer_cls, model: str):
    try:
        tokenizer = tokenizer_cls.from_pretrained(model)
    except Exception as exc:  # offline, gated repo, missing optional dep, ...
        pytest.skip(f"{model} unavailable: {type(exc).__name__}: {exc}")

    _assert_truncation_saves_no_work(tokenizer, model)


def test_length_check_encode_is_uncapped():
    """The contract the timings above justify.

    ``test_completions.py::test_text_max_length_exceeded_nonobvious`` covers the
    same decision end to end, through the renderer; this pins it at the source.
    """
    params = TokenizeParams(max_total_tokens=100, max_output_tokens=10)

    assert params.max_input_tokens == 90
    assert params.get_encode_kwargs()["truncation"] is False

    # An explicit truncate_prompt_tokens still caps, as asked.
    truncating = TokenizeParams(max_total_tokens=100, max_output_tokens=10, truncate_prompt_tokens=20)
    assert truncating.get_encode_kwargs() == {
        "truncation": True,
        "max_length": 20,
        "add_special_tokens": True,
    }
