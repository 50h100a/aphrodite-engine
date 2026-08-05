# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A grammar that will not compile must end one request, not the engine.

Two things conspired to kill an EngineCore in production:

1. With ``backend="auto"``, a schema xgrammar cannot compile is admitted by
   falling back to guidance, but the engine used to pin one backend for its
   whole lifetime (whichever the *first* structured request asked for). A
   guidance-validated request was then handed to xgrammar, which threw.
2. Async grammar compilation parks that exception in a Future, and it
   detonated at ``.result()`` inside ``Scheduler.schedule()`` -- outside any
   handler -- taking down every in-flight request with it.
"""

import json
from concurrent.futures import Future
from unittest.mock import Mock

import pytest
import torch

from aphrodite.config import AphroditeConfig, ModelConfig, SchedulerConfig
from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
from aphrodite.v1.structured_output import StructuredOutputManager
from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions
from aphrodite.v1.structured_output.request import StructuredOutputRequest

pytestmark = pytest.mark.cpu_test

VOCAB_SIZE = 128

# A `$ref` whose JSON pointer walks through an array index. xgrammar's ref
# resolver cannot index arrays, so it fails with "Cannot find field 0 in
# #/properties/x/anyOf/0". Generators that dedupe subschemas by structural
# path (zod-to-json-schema, notably) emit these routinely.
SCHEMA_XGRAMMAR_CANNOT_COMPILE = {
    "type": "object",
    "properties": {
        "x": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "y": {"$ref": "#/properties/x/anyOf/0"},
    },
    "required": ["x", "y"],
}


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
    config.parallel_config = Mock()
    config.parallel_config.distributed_executor_backend = "mp"
    return config


def _fake_backend(name: str, bitmask_words: int = VOCAB_SIZE // 32) -> Mock:
    backend = Mock()
    backend.name = name
    backend.compile_grammar = Mock(side_effect=lambda *_: Mock(name=f"{name}-grammar"))
    backend.allocate_token_bitmask = Mock(side_effect=lambda rows: torch.zeros(rows, bitmask_words, dtype=torch.int32))
    return backend


def _request(backend: str, schema=None) -> Mock:
    """A request the manager can run grammar_init() on."""
    params = StructuredOutputsParams(json=json.dumps(schema or {"type": "object"}))
    params._backend = backend
    sampling_params = Mock(spec=SamplingParams)
    sampling_params.structured_outputs = params

    request = Mock()
    request.request_id = f"req-{backend}"
    request.sampling_params = sampling_params
    request.structured_output_request = StructuredOutputRequest(params=params)
    request.use_structured_output = True
    return request


# ---------------------------------------------------------------------------
# 1. The schema itself: xgrammar declines it, guidance accepts it.
# ---------------------------------------------------------------------------


def test_xgrammar_declines_a_ref_through_an_array_index():
    """The upstream check does catch this -- so an `auto` request carrying
    this schema is routed away from xgrammar, not rejected."""
    from aphrodite.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=SCHEMA_XGRAMMAR_CANNOT_COMPILE))

    with pytest.raises(ValueError, match="Failed to transform json schema into a grammar"):
        validate_xgrammar_grammar(params)


def test_guidance_accepts_what_xgrammar_declined():
    """Which is why `auto` admits the request instead of returning 400: the
    fallback backend really can enforce this schema."""
    from aphrodite.v1.structured_output.backend_guidance import (
        is_guidance_backend_supported,
        validate_guidance_grammar,
    )

    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=SCHEMA_XGRAMMAR_CANNOT_COMPILE))

    assert is_guidance_backend_supported(SCHEMA_XGRAMMAR_CANNOT_COMPILE)
    validate_guidance_grammar(params, tokenizer=None)


# ---------------------------------------------------------------------------
# 2. Backend routing: the engine must compile with the backend the request
#    was validated against.
# ---------------------------------------------------------------------------


def test_each_request_compiles_on_its_own_backend(aphrodite_config, monkeypatch):
    """The engine used to keep whichever backend the first structured request
    asked for and compile everything with it, so a request that `auto` routed
    to guidance was handed to xgrammar and threw."""
    created = {}

    def fake_create(self, name):
        created[name] = _fake_backend(name)
        return created[name]

    monkeypatch.setattr(StructuredOutputManager, "_create_backend", fake_create)
    manager = StructuredOutputManager(aphrodite_config)
    manager._use_async_grammar_compilation = False

    xgrammar_req = _request("xgrammar")
    guidance_req = _request("guidance", SCHEMA_XGRAMMAR_CANNOT_COMPILE)
    manager.grammar_init(xgrammar_req)
    manager.grammar_init(guidance_req)

    assert set(created) == {"xgrammar", "guidance"}
    created["xgrammar"].compile_grammar.assert_called_once()
    created["guidance"].compile_grammar.assert_called_once()
    # And each got the spec belonging to its own request.
    request_type, spec = created["guidance"].compile_grammar.call_args[0]
    assert request_type is StructuredOutputOptions.JSON
    assert json.loads(spec) == SCHEMA_XGRAMMAR_CANNOT_COMPILE


def test_backends_are_reused_across_requests(aphrodite_config, monkeypatch):
    """Per-request routing must not mean per-request construction: building a
    GrammarCompiler means re-deriving TokenizerInfo over the whole vocab."""
    calls = []

    def fake_create(self, name):
        calls.append(name)
        return _fake_backend(name)

    monkeypatch.setattr(StructuredOutputManager, "_create_backend", fake_create)
    manager = StructuredOutputManager(aphrodite_config)
    manager._use_async_grammar_compilation = False

    for _ in range(3):
        manager.grammar_init(_request("xgrammar"))

    assert calls == ["xgrammar"]


def test_bitmask_is_wide_enough_for_every_backend(aphrodite_config, monkeypatch):
    """Backends disagree on vocab size (guidance rounds up to the tokenizer's
    length), and they all write into one shared bitmask. Sizing it from
    whichever backend happened to be built first would let the wider one write
    past the end of a row."""

    def fake_create(self, name):
        # 2 extra words, as a wider tokenizer would give.
        words = VOCAB_SIZE // 32 + (2 if name == "guidance" else 0)
        return _fake_backend(name, bitmask_words=words)

    monkeypatch.setattr(StructuredOutputManager, "_create_backend", fake_create)
    manager = StructuredOutputManager(aphrodite_config)
    manager._use_async_grammar_compilation = False

    narrow = _request("xgrammar")
    manager.grammar_init(narrow)
    bitmask = manager.grammar_bitmask({"req-xgrammar": narrow}, ["req-xgrammar"], {})
    assert bitmask.shape[1] == VOCAB_SIZE // 32

    # A guidance request shows up later; the shared bitmask has to grow.
    wide = _request("guidance")
    manager.grammar_init(wide)
    bitmask = manager.grammar_bitmask({"req-guidance": wide}, ["req-guidance"], {})
    assert bitmask.shape[1] == VOCAB_SIZE // 32 + 2


def test_xgrammar_fills_a_bitmask_sized_for_a_wider_backend():
    """xgrammar rejects any row that is not exactly its own vocab width, so
    sharing one bitmask with a wider backend has to narrow the view -- and the
    write still has to land in the shared tensor."""
    import xgrammar as xgr

    from aphrodite.v1.structured_output.backend_xgrammar import XgrammarGrammar

    vocab = ["{", "}", " "] + [f"t{i}" for i in range(61)]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=len(vocab))
    ctx = xgr.GrammarCompiler(tokenizer_info).compile_json_schema('{"type": "object"}')

    def new_grammar():
        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=len(vocab),
            ctx=ctx,
        )

    exact = torch.zeros(1, len(vocab) // 32, dtype=torch.int32)
    new_grammar().fill_bitmask(exact, 0)

    # Two words wider, as a guidance-sized bitmask would be.
    shared = torch.zeros(3, len(vocab) // 32 + 2, dtype=torch.int32)
    new_grammar().fill_bitmask(shared, 1)

    assert shared[1][: len(vocab) // 32].tolist() == exact[0].tolist()
    # `{` is the only legal first token, and only the addressed row is touched.
    assert exact[0][0] == 1
    assert not shared[0].any() and not shared[2].any()


# ---------------------------------------------------------------------------
# 3. Containment: a compile failure is recorded, not raised.
# ---------------------------------------------------------------------------


def _failed_future(exc: BaseException) -> Future:
    future: Future = Future()
    future.set_exception(exc)
    return future


def test_compilation_failure_does_not_escape_the_grammar_property():
    """`.grammar` is read from Scheduler.schedule(), which the engine core
    calls unguarded. Re-raising there is what killed the engine."""
    request = StructuredOutputRequest(params=StructuredOutputsParams(json="{}"))
    request.grammar = _failed_future(RuntimeError("Cannot find field 0 in #/properties/x/anyOf/0"))

    assert request.is_grammar_ready is True
    assert request.grammar is None
    assert isinstance(request.grammar_error, RuntimeError)


def test_a_pending_compilation_is_not_mistaken_for_a_failure():
    """The timeout path is the common one: the future is simply not done yet,
    and the request must stay blocked rather than be failed."""
    request = StructuredOutputRequest(params=StructuredOutputsParams(json="{}"))
    request.grammar = Future()

    assert request.is_grammar_ready is False
    assert request.grammar_error is None


def test_the_failure_survives_repeated_polling():
    """The scheduler polls once per step; consuming the error on the first
    read would leave the request blocked forever afterwards."""
    request = StructuredOutputRequest(params=StructuredOutputsParams(json="{}"))
    request.grammar = _failed_future(RuntimeError("boom"))

    assert request.is_grammar_ready is True
    for _ in range(3):
        assert request.grammar is None
        assert request.grammar_error is not None


def test_compilation_failure_reaches_the_request_not_the_caller(aphrodite_config, monkeypatch):
    """End to end through the manager: submitting to the compile thread pool
    must leave the exception on the request, ready for the scheduler to act on."""

    def fake_create(self, name):
        backend = _fake_backend(name)
        backend.compile_grammar = Mock(side_effect=RuntimeError("Cannot find field 0"))
        return backend

    monkeypatch.setattr(StructuredOutputManager, "_create_backend", fake_create)
    # Async compilation, i.e. the path that needs the thread pool.
    monkeypatch.setattr("aphrodite.v1.structured_output.cached_tokenizer_from_config", Mock())
    aphrodite_config.model_config.skip_tokenizer_init = False
    aphrodite_config.structured_outputs_config.reasoning_parser_plugin = ""
    aphrodite_config.structured_outputs_config.reasoning_parser = ""
    manager = StructuredOutputManager(aphrodite_config)
    assert manager._use_async_grammar_compilation

    request = _request("xgrammar", SCHEMA_XGRAMMAR_CANNOT_COMPILE)
    manager.grammar_init(request)

    structured_req = request.structured_output_request
    # Wait for the compile thread, without ever letting the exception through.
    while not structured_req.is_grammar_ready:
        pass

    assert structured_req.grammar is None
    assert isinstance(structured_req.grammar_error, RuntimeError)
