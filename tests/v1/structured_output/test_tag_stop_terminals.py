# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real matcher checks for tags that must emit an engine stop token."""

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from aphrodite.v1.structured_output.backend_types import StructuredOutputOptions
from aphrodite.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def backend():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"a": 0, "<return>": 1, "<endoftext>": 2, "<tag>": 3, "[UNK]": 4, '"': 5, "</tag>": 6},
                unk_token="[UNK]",
            )
        ),
        eos_token="<return>",
        unk_token="[UNK]",
        additional_special_tokens=["<endoftext>", "<tag>", "</tag>"],
    )
    config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="toy"),
            try_get_generation_config=lambda: {"eos_token_id": [1, 2]},
        ),
        speculative_config=None,
    )
    return XgrammarBackend(config, tokenizer=tokenizer, vocab_size=7)


def _tag(end="<return>"):
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {"type": "tag", "begin": "<tag>", "content": {"type": "const_string", "value": "a"}, "end": end},
        }
    )


def _allowed(backend, grammar, token_id):
    mask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(mask, 0)
    return bool(int(mask[0, token_id // 32]) & (1 << (token_id % 32)))


@pytest.mark.parametrize("stop_ids", [None, {1, 2}, {1}])
def test_tag_can_emit_eos_terminal(backend, stop_ids):
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _tag(), stop_token_ids=stop_ids)
    assert grammar.accept_tokens("tag", [3, 0])
    assert _allowed(backend, grammar, 1), "The tag cannot close if its EOS terminal is masked"
    assert not _allowed(backend, grammar, 2), "Alternate EOS cannot terminate an incomplete tag"
    assert grammar.accept_tokens("tag", [1])


def test_tag_can_use_the_only_engine_stop_token(backend):
    backend.aphrodite_config.model_config.try_get_generation_config = lambda: {"eos_token_id": 1}
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _tag(), stop_token_ids={1})
    assert grammar.accept_tokens("tag", [3, 0])
    assert _allowed(backend, grammar, 1)
    assert grammar.accept_tokens("tag", [1])


def test_json_and_tags_keep_independent_stop_rules(backend):
    # Compile a tag first: its EOS exception must not leak into JSON grammars.
    backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _tag(), stop_token_ids={1, 2})
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, '{"type":"string"}', stop_token_ids={1, 2})
    assert grammar.accept_tokens("json", [5, 0])
    assert not _allowed(backend, grammar, 1)
    assert not _allowed(backend, grammar, 2)
    assert not _allowed(backend, grammar, 3)  # Special tag marker cannot enter JSON strings.
    assert grammar.accept_tokens("json", [5])
    assert _allowed(backend, grammar, 1)
    assert _allowed(backend, grammar, 2)


def test_tag_without_eos_terminal_preserves_request_stops(backend):
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _tag("</tag>"), stop_token_ids={1, 2})
    assert grammar.accept_tokens("tag", [3, 0])
    assert not _allowed(backend, grammar, 1)
    assert _allowed(backend, grammar, 6)
    assert grammar.accept_tokens("tag", [6])
    assert _allowed(backend, grammar, 1)
    assert _allowed(backend, grammar, 2)


def test_missing_generation_config_does_not_break_tag_compilation(backend):
    def unavailable():
        raise OSError("generation config is unavailable")

    backend.aphrodite_config.model_config.try_get_generation_config = unavailable
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, _tag())
    assert grammar.accept_tokens("tag", [3, 0])
    assert _allowed(backend, grammar, 1)


def test_eos_terminal_compiler_preserves_byte_fallback_tokens(backend):
    import xgrammar as xgr

    vocabulary = ["<0xC3>", "<0xA9>", "<eos>", "<tag>"]
    info = xgr.TokenizerInfo(vocabulary, xgr.VocabType.BYTE_FALLBACK, stop_token_ids=[2])
    backend._tag_tokenizer_info = info
    backend.full_vocab_compiler = xgr.GrammarCompiler(info)
    backend.tokenizer = SimpleNamespace(convert_ids_to_tokens=vocabulary.__getitem__)
    backend.vocab_size = len(vocabulary)
    tag = json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "<tag>",
                "content": {"type": "const_string", "value": "é"},
                "end": "<eos>",
            },
        }
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, tag, stop_token_ids={2})
    assert grammar.accept_tokens("utf8", [3, 0, 1])
    assert _allowed(backend, grammar, 2)
    assert grammar.accept_tokens("utf8", [2])
