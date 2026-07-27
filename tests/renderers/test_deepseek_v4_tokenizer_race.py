# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Regression tests for DeepSeek V4 renderer tokenizer thread safety.

``DeepseekV4Renderer`` runs ``apply_chat_template`` on its ``ThreadPoolExecutor``
while the event loop reaches the *same* tokenizer through
``InputProcessor.process_inputs`` -> ``SamplingParams.update_from_tokenizer``
(which encodes every DRY sequence breaker and bad word).

HF fast tokenizers keep truncation/padding as mutable state on the Rust
``Tokenizer``: ``set_truncation_and_padding`` calls ``no_truncation()`` /
``enable_truncation()``, both of which need an exclusive borrow. ``encode_batch``
releases the GIL while holding a shared borrow, so sharing one tokenizer across
those two threads raises ``RuntimeError: Already borrowed`` as soon as one
request encodes with truncation and another without.

The renderer must therefore hand each thread its own copy via
``maybe_make_thread_pool``, exactly as ``HfRenderer`` does.
"""

import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

import aphrodite.renderers.deepseek_v4 as deepseek_v4_renderer
from aphrodite.config import ModelConfig
from aphrodite.renderers.deepseek_v4 import DeepseekV4Renderer
from aphrodite.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from aphrodite.tokenizers.hf import ThreadSafeHFTokenizerMixin, get_cached_tokenizer

MODEL = "facebook/opt-125m"

# Long enough that encode_batch spends real time with the GIL released, which is
# the window in which the other thread's no_truncation() gets refused.
_LONG_PROMPT = "the quick brown fox " * 50_000
# What SamplingParams.update_from_tokenizer encodes per DRY sequence breaker.
_BREAKER_PROBE = "a\n"

_RACE_DURATION_S = 10.0

# DeepSeek spells its control tokens with FULLWIDTH VERTICAL LINE and
# SentencePiece's word-boundary glyph, never ASCII "|" or "_", so text a user
# types cannot forge a role boundary. Written as escapes because the ASCII
# lookalikes are indistinguishable in review and would silently break the
# assertion below (and, in the encoder itself, role separation entirely).
_BAR = "\uff5c"  # ｜ FULLWIDTH VERTICAL LINE, not ASCII "|" (U+007C)
_SPACE = "\u2581"  # ▁ LOWER ONE EIGHTH BLOCK, not ASCII "_" (U+005F)

_EXPECTED_CHAT_PROMPT = (
    f"<{_BAR}begin{_SPACE}of{_SPACE}sentence{_BAR}><{_BAR}User{_BAR}>Hello<{_BAR}Assistant{_BAR}></think>"
)


@pytest.fixture(scope="module")
def base_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture()
def config():
    model_config = ModelConfig(
        model=MODEL,
        tokenizer=MODEL,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )


def _dsv4_tokenizer(base_tokenizer):
    """A DeepSeek V4 tokenizer with its own Rust backend, as served."""
    return get_cached_tokenizer(get_deepseek_v4_tokenizer(deepcopy(base_tokenizer)))


def _run_race(tokenizer, duration_s: float = _RACE_DURATION_S) -> list[str]:
    """Drive the renderer-thread/event-loop encode pattern; collect failures."""
    errors: list[str] = []
    stop = threading.Event()

    def encode_forever(*args, **kwargs):
        try:
            while not stop.is_set():
                tokenizer.encode(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded and asserted on
            errors.append(repr(exc))
            stop.set()

    threads = [
        # Renderer worker: a long prompt, truncation enabled.
        threading.Thread(
            target=encode_forever,
            args=(_LONG_PROMPT,),
            kwargs=dict(add_special_tokens=False, truncation=True, max_length=10**6),
        ),
        # Event loop: short probe, truncation off -> no_truncation().
        threading.Thread(
            target=encode_forever,
            args=(_BREAKER_PROBE,),
            kwargs=dict(add_special_tokens=False),
        ),
    ]

    for thread in threads:
        thread.start()
    stop.wait(duration_s)
    stop.set()
    for thread in threads:
        thread.join()

    return errors


def test_renderer_pools_its_tokenizer(config, base_tokenizer):
    renderer = DeepseekV4Renderer(config, _dsv4_tokenizer(base_tokenizer))

    assert isinstance(renderer.tokenizer, ThreadSafeHFTokenizerMixin)


def test_pool_has_a_copy_per_thread(config, base_tokenizer, monkeypatch):
    config.model_config.renderer_num_workers = 3
    copies: list[int] = []
    real_maybe_make_thread_pool = deepseek_v4_renderer.maybe_make_thread_pool

    def spy(tokenizer, count=1):
        copies.append(count)
        return real_maybe_make_thread_pool(tokenizer, count)

    monkeypatch.setattr(deepseek_v4_renderer, "maybe_make_thread_pool", spy)

    DeepseekV4Renderer(config, _dsv4_tokenizer(base_tokenizer))

    # One copy per executor worker, plus one for the event loop thread.
    assert copies == [4]


def test_renderer_does_not_mutate_the_shared_tokenizer(config, base_tokenizer):
    # In production the tokenizer comes from a process-global lru_cache, so
    # pooling it in place would leak the wrapper into every other consumer.
    tokenizer = _dsv4_tokenizer(base_tokenizer)
    original_cls = type(tokenizer)

    DeepseekV4Renderer(config, tokenizer)

    assert type(tokenizer) is original_cls
    assert not isinstance(tokenizer, ThreadSafeHFTokenizerMixin)


def test_pooled_tokenizer_keeps_v4_chat_encoding(config, base_tokenizer):
    # The pool wrapper overrides apply_chat_template, so guard that requests
    # still reach encode_messages rather than falling back to an HF template.
    renderer = DeepseekV4Renderer(config, _dsv4_tokenizer(base_tokenizer))

    prompt = renderer.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt == _EXPECTED_CHAT_PROMPT


@pytest.mark.slow_test
def test_pooled_tokenizer_survives_concurrent_truncation_flips(config, base_tokenizer):
    renderer = DeepseekV4Renderer(config, _dsv4_tokenizer(base_tokenizer))

    assert _run_race(renderer.tokenizer) == []


@pytest.mark.slow_test
def test_unpooled_tokenizer_is_borrow_unsafe(base_tokenizer):
    # Negative control: without the pool the same workload raises, so the test
    # above is not passing for want of a race window.
    errors = _run_race(_dsv4_tokenizer(base_tokenizer))

    assert any("Already borrowed" in error for error in errors), (
        f"expected a borrow conflict on a shared tokenizer, got {errors}"
    )
