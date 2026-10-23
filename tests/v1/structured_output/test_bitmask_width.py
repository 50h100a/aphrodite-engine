# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The bitmask handed to the workers must be one word per 32 model-vocab ids.

Workers copy it into a device buffer preallocated at exactly that width
(`StructuredOutputsWorker.__init__`), so a mask a single word wider is a hard
shape error that takes down every worker process, not harmless padding.
Guidance sizes its bitmask from `max(vocab_size, len(tokenizer))`, which is
wider than the model vocab whenever the tokenizer carries ids the model has no
logits for -- exactly the case that crashed once `auto` started routing
fallback requests to guidance.
"""

from unittest.mock import Mock

import pytest
import torch

from aphrodite.config import AphroditeConfig, ModelConfig, SchedulerConfig
from aphrodite.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

# Deliberately not a multiple of 32: a real vocab rarely is, and the last word
# is partially used.
VOCAB_SIZE = 1000
MODEL_WORDS = (VOCAB_SIZE + 31) // 32  # 32
# What guidance would ask for when the tokenizer has ids past the model vocab.
WIDE_WORDS = MODEL_WORDS + 1


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
    config.speculative_config = None
    config.num_speculative_tokens = 0
    return config


def _backend(num_words):
    """A backend whose bitmask rows are `num_words` wide."""
    backend = Mock()
    backend.allocate_token_bitmask = Mock(
        side_effect=lambda rows, _=None: torch.zeros(rows, num_words, dtype=torch.int32)
    )
    return backend


def _manager(aphrodite_config, backends):
    manager = StructuredOutputManager(aphrodite_config)
    manager.backends = backends
    manager.tokenizer = Mock()
    return manager


def _request():
    grammar = Mock()
    grammar.is_terminated = Mock(return_value=False)
    grammar.fill_bitmask = Mock()

    request = Mock()
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar
    request.structured_output_request.reasoner = None
    request.structured_output_request.reasoning_parser_kwargs = None
    request.use_structured_output = True
    return request


def _bitmask(manager, num_requests=1):
    requests = {f"req-{i}": _request() for i in range(num_requests)}
    return manager.grammar_bitmask(requests, list(requests), {})


def test_wide_backend_is_narrowed_to_the_model_vocab(aphrodite_config):
    """The crash: guidance allocated 4041 words, the worker buffer held 4040."""
    manager = _manager(aphrodite_config, {"guidance": _backend(WIDE_WORDS)})

    bitmask = _bitmask(manager)

    assert bitmask.shape[-1] == MODEL_WORDS


def test_mixed_backends_fill_wide_but_publish_narrow(aphrodite_config):
    """Both halves matter: guidance must still get a row wide enough to fill
    (llguidance writes through a raw pointer sized from its own vocab), while
    the workers must still receive the model-vocab width."""
    manager = _manager(
        aphrodite_config,
        {"xgrammar": _backend(MODEL_WORDS), "guidance": _backend(WIDE_WORDS)},
    )

    bitmask = _bitmask(manager)

    assert manager._grammar_bitmask.shape[-1] == WIDE_WORDS
    assert bitmask.shape[-1] == MODEL_WORDS


def test_narrowing_preserves_each_rows_model_vocab_words(aphrodite_config):
    """Truncation must drop each row's tail, not restride the buffer. Repacking
    a wide row into a narrow one is where an off-by-one shifts the mask and
    constrains the wrong token ids -- silently producing schema-violating
    output instead of crashing. Needs several rows: with a single row the
    leading dim is 1, so the slice is contiguous by accident and a stride bug
    would not show."""
    manager = _manager(aphrodite_config, {"guidance": _backend(WIDE_WORDS)})

    requests = {}
    for i in range(3):
        request = _request()
        # Row i is filled with (i+1)*100 + word index, so a shifted or
        # misstrided read lands on a recognisably wrong value.
        marker = (i + 1) * 100 + torch.arange(WIDE_WORDS, dtype=torch.int32)
        request.structured_output_request.grammar.fill_bitmask = Mock(
            side_effect=lambda bitmask, idx, marker=marker: bitmask[idx].copy_(marker)
        )
        requests[f"req-{i}"] = request

    bitmask = manager.grammar_bitmask(requests, list(requests), {})

    assert bitmask.shape == (3, MODEL_WORDS)
    for i in range(3):
        expected = [(i + 1) * 100 + w for w in range(MODEL_WORDS)]
        assert bitmask[i].tolist() == expected


def test_published_bitmask_is_contiguous(aphrodite_config):
    """A column slice is non-contiguous. With an in-process executor the array
    reaches torch.from_numpy/pin_memory unchanged (no serializer to re-pack the
    rows), so the manager owes the workers a packed buffer."""
    manager = _manager(aphrodite_config, {"guidance": _backend(WIDE_WORDS)})

    bitmask = _bitmask(manager, num_requests=3)

    assert bitmask.flags.c_contiguous


def test_narrow_only_backend_is_left_alone(aphrodite_config):
    """xgrammar/outlines already allocate at the model vocab; that common case
    must stay a zero-copy view of the fill buffer."""
    manager = _manager(aphrodite_config, {"xgrammar": _backend(MODEL_WORDS)})

    bitmask = _bitmask(manager)

    assert bitmask.shape[-1] == MODEL_WORDS
    assert bitmask.flags.c_contiguous


def test_backend_narrower_than_the_model_vocab_is_rejected(aphrodite_config):
    """Silently slicing a too-narrow buffer would leave the tail of the
    vocabulary unconstrained, so this fails loudly instead."""
    manager = _manager(aphrodite_config, {"broken": _backend(MODEL_WORDS - 1)})

    with pytest.raises(AssertionError, match="does not cover the model vocabulary"):
        _bitmask(manager)


def test_published_width_matches_the_worker_buffer_formula(aphrodite_config):
    """Pins the contract to the two places that assume it:
    StructuredOutputsWorker allocates cdiv(vocab_size, 32) and gpu/warmup.py
    builds its dummy mask at (vocab_size + 31) // 32."""
    manager = _manager(aphrodite_config, {"guidance": _backend(WIDE_WORDS)})

    worker_buffer_words = (aphrodite_config.model_config.get_vocab_size() + 31) // 32

    assert _bitmask(manager).shape[-1] == worker_buffer_words
