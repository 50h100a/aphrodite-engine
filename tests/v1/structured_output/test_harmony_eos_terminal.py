# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep current Harmony control-token behavior while porting EOS handling.

The fork's original tests used its older Harmony tag/header dialect. These
checks use current Sonar strict levels, current XGrammar-generated tool
headers, and the complete engine EOS set. They ensure final replies, tool calls,
analysis and ordinary JSON still terminate after the compiler changes.
The explicit EOS-as-tag-terminal failure is pinned independently by the
synthetic and byte-fallback cases in test_tag_stop_terminals.py.
"""

import json
from unittest.mock import Mock

import pytest
import torch

from aphrodite.config import AphroditeConfig, ModelConfig
from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag
from aphrodite.tool_parsers.tool_strict_level import ToolStrictLevel
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
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city": "Paris"}'
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def backend(tokenizer):
    model_config = Mock(spec=ModelConfig)
    model_config.hf_config = Mock(model_type="gpt_oss")
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
    tag = get_model_structural_tag(
        model="harmony",
        tools=request.tools,
        tool_choice=request.tool_choice,
        reasoning=True,
        strict_level=ToolStrictLevel.PARAMETER,
    )
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
    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag(), stop_token_ids=set(ENGINE_EOS_IDS)
    )

    allowed = _walk(backend, grammar, tokenizer, REPLY_PREFIX)

    assert allowed[RETURN], "<|return|> is masked, so the reply can never terminate"
    assert grammar.accept_tokens("test", [RETURN])


def test_harmony_tool_call_still_closes(backend, tokenizer):
    """`<|call|>` was never the reserved stop token, and must stay reachable."""
    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag(), stop_token_ids=set(ENGINE_EOS_IDS)
    )

    allowed = _walk(backend, grammar, tokenizer, TOOL_CALL_PREFIX)

    assert allowed[CALL]
    assert grammar.accept_tokens("test", [CALL])


def test_harmony_analysis_still_closes_on_end(backend, tokenizer):
    """Reasoning worked even with the bug present; keep it that way."""
    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG, _harmony_tag(), stop_token_ids=set(ENGINE_EOS_IDS)
    )

    allowed = _walk(backend, grammar, tokenizer, "<|channel|>analysis<|message|>Thinking.")

    assert allowed[END]


@pytest.mark.parametrize("unique", [False, True])
def test_harmony_final_postconditions_with_real_tokenizer(backend, tokenizer, unique):
    from aphrodite.v1.structured_output.harmony import make_harmony_json_tag
    from aphrodite.v1.structured_output.postconditions import PostconditionGrammar, maybe_wrap

    schema = (
        {"type": "array", "items": {"enum": [1, 2]}, "uniqueItems": True}
        if unique
        else {"type": "array", "items": {"type": "integer"}, "contains": {"const": 7}}
    )
    spec = make_harmony_json_tag(schema).model_dump_json()
    inner = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, spec, stop_token_ids=set(ENGINE_EOS_IDS))
    grammar = maybe_wrap(inner, StructuredOutputOptions.STRUCTURAL_TAG, spec, tokenizer, VOCAB_SIZE)
    assert isinstance(grammar, PostconditionGrammar)
    prefix = (
        "<|channel|>analysis<|message|>Quoted <|channel|>final json<|message|>[7]"
        "<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>"
    )
    allowed = _walk(backend, grammar, tokenizer, prefix + ("[1," if unique else "[1"))
    refused = "1" if unique else "]"
    (token,) = tokenizer.encode(refused, add_special_tokens=False)
    assert not allowed[token]
    assert grammar.validate_tokens([token]) == []
    completion = "2]" if unique else ",7]"
    allowed = _walk(backend, grammar, tokenizer, completion)
    assert allowed[RETURN]
    assert grammar.accept_tokens("test", [RETURN])


def test_harmony_none_with_real_tokenizer(backend, tokenizer):
    from aphrodite.parser import ParserManager

    request = ChatCompletionRequest(model="offline", messages=[], tools=TOOLS, tool_choice="none")
    cls = ParserManager.get_parser(
        tool_parser_name="openai",
        reasoning_parser_name="openai_gptoss",
        is_harmony=True,
        enable_auto_tools=True,
        model_name=MODEL,
    )
    cls(tokenizer, request.tools).adjust_request(request)
    spec = request.structured_outputs.structural_tag
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, spec, stop_token_ids=set(ENGINE_EOS_IDS))
    allowed = _walk(backend, grammar, tokenizer, REPLY_PREFIX)
    assert allowed[RETURN]
    grammar.reset()
    assert not grammar.accept_tokens("test", tokenizer.encode(TOOL_CALL_PREFIX, add_special_tokens=False))


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
