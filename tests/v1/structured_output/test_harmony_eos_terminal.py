# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A structural tag may name the model's EOS as one of its own terminators.

xgrammar reserves its stop tokens: they are unmasked only once the grammar is
in an accepting state, and are never offered as a terminal *inside* a rule. The
harmony tag closes a `final`-channel message with `<|return|>`, which is
gpt-oss's EOS -- so compiled against the default tokenizer info, that tag can
never be closed. The model reaches the end of its answer with the one token it
wants masked out and runs to max_tokens emitting filler, which is what a
`finish_reason: length` full of whitespace after a perfectly good reply is.

Reasoning survives this because the `analysis` tag ends on `<|end|>`, an
ordinary token, and tool calls survive because they end on `<|call|>`. Only the
reply itself is trapped, so the tests below pin all three.
"""

import json
from unittest.mock import Mock

import pytest
import torch

from aphrodite.config import AphroditeConfig, ModelConfig
from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.tool_parsers.gptoss_tool_parser import GptOssToolParser
from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag
from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions
from aphrodite.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.cpu_test

# 20b and 120b share the o200k_harmony tokenizer; 20b is the cheaper pull.
MODEL = "openai/gpt-oss-20b"
VOCAB_SIZE = 201088

RETURN = 200002  # <|return|>, and the tokenizer's eos_token_id
ENDOFTEXT = 199999
CALL = 200012
END = 200007

# generation_config.json ships all three; this is the set the engine stops on.
ENGINE_EOS_IDS = [RETURN, ENDOFTEXT, CALL]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# What the model actually emits: reason on `analysis`, close it, then answer on
# `final`. Stops one token short of the `<|return|>` under test.
REPLY_PREFIX = (
    "<|channel|>analysis<|message|>The user wants a fact.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>The capital of France is Paris."
)
TOOL_CALL_PREFIX = (
    "<|channel|>analysis<|message|>I should look this up.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather"
    '<|constrain|>json<|message|>{"city": "Paris"}'
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def backend(tokenizer):
    model_config = Mock(spec=ModelConfig)
    model_config.try_get_generation_config = Mock(return_value={"eos_token_id": list(ENGINE_EOS_IDS)})

    config = Mock(spec=AphroditeConfig)
    config.model_config = model_config
    config.structured_outputs_config = Mock()
    config.structured_outputs_config.disable_any_whitespace = False
    config.speculative_config = None

    return XgrammarBackend(config, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)


def _harmony_tag() -> str:
    """The tag the serving layer builds, preamble adjustment included."""
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=TOOLS,
        tool_choice="auto",
    )
    # HarmonyParser always asks for the reasoning form; see its
    # _grammar_needs_reasoning.
    tag = GptOssToolParser(Mock(), tools=request.tools).get_structural_tag(request, reasoning=True)
    return json.dumps(tag.model_dump())


def _walk(backend, grammar, tokenizer, text) -> torch.Tensor:
    """Feed `text` to the grammar, then return its mask over the vocabulary."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert grammar.accept_tokens("test", token_ids), f"grammar rejected {text!r}"

    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)

    import xgrammar as xgr

    logits = torch.zeros(1, VOCAB_SIZE)
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    return logits[0] > -1e30


def test_harmony_final_message_can_emit_return(backend, tokenizer):
    """The bug: `<|return|>` masked at the end of the reply, so nothing ends."""
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag())

    allowed = _walk(backend, grammar, tokenizer, REPLY_PREFIX)

    assert allowed[RETURN], "<|return|> is masked, so the reply can never terminate"
    assert grammar.accept_tokens("test", [RETURN])


def test_harmony_tool_call_still_closes(backend, tokenizer):
    """`<|call|>` was never the reserved stop token, and must stay reachable."""
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag())

    allowed = _walk(backend, grammar, tokenizer, TOOL_CALL_PREFIX)

    assert allowed[CALL]
    assert grammar.accept_tokens("test", [CALL])


def test_harmony_analysis_still_closes_on_end(backend, tokenizer):
    """Reasoning worked even with the bug present; keep it that way."""
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag())

    allowed = _walk(backend, grammar, tokenizer, "<|channel|>analysis<|message|>Thinking.")

    assert allowed[END]


def test_json_schema_still_terminates_on_eos(backend, tokenizer):
    """The other half of the split, and the reason it has to be a split.

    A completed schema leaves exactly one legal token and it must be the real
    EOS. Substituting the stop token here -- rather than only for tags that need
    it -- would hang JSON requests the same way harmony hangs today.
    """
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, schema)

    allowed = _walk(backend, grammar, tokenizer, '{"city": "Paris"}')

    assert allowed[RETURN]
    assert not allowed[ENDOFTEXT]


def test_tag_without_the_eos_keeps_the_default_compiler(backend):
    """The substitution is scoped to tags that actually need it.

    Every other registered tag -- hermes here -- terminates on ordinary text,
    so it must not pay for a second tokenizer info.
    """
    hermes_tag = get_model_structural_tag(
        model="hermes",
        tools=[ChatCompletionRequest(model="m", messages=[], tools=TOOLS).tools[0]],
        tool_choice="auto",
        reasoning=False,
    )

    chosen = backend._compiler_for_structural_tag(json.dumps(hermes_tag.model_dump()))

    assert chosen is backend.compiler


def test_harmony_tag_gets_a_substituted_compiler(backend):
    """And the harmony tag does get one, distinct from the default."""
    chosen = backend._compiler_for_structural_tag(_harmony_tag())

    assert chosen is not backend.compiler
