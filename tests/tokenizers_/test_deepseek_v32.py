# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from aphrodite.renderers.registry import RENDERER_REGISTRY
from aphrodite.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer
from aphrodite.tokenizers.registry import TokenizerRegistry

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return get_deepseek_v32_tokenizer(FakeHfTokenizer())


def _render(messages, **kwargs):
    return _tokenizer().apply_chat_template(messages, tokenize=False, **kwargs)


def _tool_call(name: str, arguments: str):
    return {"function": {"name": name, "arguments": arguments}}


def test_deepseek_v32_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v32").__name__ == ("DeepseekV32Tokenizer")
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v32").__name__ == ("DeepseekV32Renderer")


# ── Baseline renderings ───────────────────────────────────────────────
# The assistant opener used to be baked into `user_msg_template`; it is now
# appended separately so `add_generation_prompt=False` can suppress it. These
# pin that split as byte-neutral.


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, f"{BOS}a<｜User｜>hi<｜Assistant｜></think>"),
        ({"thinking": True}, f"{BOS}a<｜User｜>hi<｜Assistant｜><think>"),
        ({"enable_thinking": True}, f"{BOS}a<｜User｜>hi<｜Assistant｜><think>"),
    ],
    ids=["chat", "thinking", "enable_thinking"],
)
def test_deepseek_v32_renders_system_and_user(kwargs, expected):
    assert _render([{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}], **kwargs) == expected


def test_deepseek_v32_renders_developer_as_a_user_turn():
    prompt = _render([{"role": "developer", "content": "hi"}])

    assert prompt.startswith(f"{BOS}<｜User｜>")
    assert prompt.endswith("<｜Assistant｜></think>")
    assert "# The user's message is: hi" in prompt


def test_deepseek_v32_renders_tool_results():
    prompt = _render(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [_tool_call("f", '{"x": 1}')]},
            {"role": "tool", "content": "r1"},
        ]
    )

    assert '<｜DSML｜invoke name="f">' in prompt
    assert '<｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter>' in prompt
    assert prompt.endswith("<function_results>\n<result>r1</result>\n</function_results>\n\n</think>")


# ── add_generation_prompt ─────────────────────────────────────────────

_SYSTEM_ONLY = [
    {"role": "system", "content": "a"},
    {"role": "system", "content": "b"},
    {"role": "system", "content": "c"},
]
_SYSTEM_LAST = [
    {"role": "user", "content": "hi"},
    {"role": "system", "content": "note"},
]

# (messages, kwargs, body-without-opener, opener) -- the body is what
# `add_generation_prompt=False` renders, so both tests below share it.
_NO_TRAILING_USER = [
    pytest.param(_SYSTEM_ONLY, {}, f"{BOS}abc", "<｜Assistant｜></think>", id="system-only-chat"),
    pytest.param(
        _SYSTEM_ONLY,
        {"thinking": True},
        f"{BOS}abc",
        "<｜Assistant｜><think>",
        id="system-only-thinking",
    ),
    pytest.param(
        _SYSTEM_LAST,
        {},
        f"{BOS}<｜User｜>hi<｜Assistant｜></think>note",
        "<｜Assistant｜></think>",
        id="system-last-chat",
    ),
    pytest.param(
        _SYSTEM_LAST,
        {"thinking": True},
        f"{BOS}<｜User｜>hi<｜Assistant｜><think>note",
        "<｜Assistant｜><think>",
        id="system-last-thinking",
    ),
]


@pytest.mark.parametrize(("messages", "kwargs", "body", "opener"), _NO_TRAILING_USER)
def test_deepseek_v32_adds_generation_prefix_without_trailing_user(messages, kwargs, body, opener):
    """The reference encoder only opens the assistant turn as part of a user
    turn, so a conversation ending in a system message renders with no think
    marker -- which wedges the structured-output bitmask closed, because the
    gate reads "no marker" as "reasoning still pending"."""
    assert _render(messages, **kwargs) == body + opener


@pytest.mark.parametrize(("messages", "kwargs", "body", "opener"), _NO_TRAILING_USER)
def test_deepseek_v32_generation_prefix_can_be_suppressed(messages, kwargs, body, opener):
    assert _render(messages, add_generation_prompt=False, **kwargs) == body


def test_deepseek_v32_suppresses_generation_prefix_after_user():
    """The flag is live in both directions, not merely additive."""
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}]

    assert _render(messages, add_generation_prompt=False) == f"{BOS}a<｜User｜>hi"


def test_deepseek_v32_keeps_mid_conversation_openers_when_suppressed():
    """Only the opener that would end the prompt is governed by the flag."""
    prompt = _render(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "system", "content": "note"},
        ],
        add_generation_prompt=False,
    )

    assert prompt == f"{BOS}<｜User｜>hi<｜Assistant｜></think>yo{EOS}note"


def test_deepseek_v32_adds_exactly_one_prefix_after_user():
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}]

    assert _render(messages).count("<｜Assistant｜>") == 1


def test_deepseek_v32_closed_assistant_turn_gets_a_fresh_opener():
    prompt = _render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}])

    assert prompt == f"{BOS}<｜User｜>hi<｜Assistant｜></think>done{EOS}<｜Assistant｜></think>"


def test_deepseek_v32_completed_tool_group_needs_no_opener():
    """The last result of a call group already ends in a think marker."""
    prompt = _render(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [_tool_call("f", '{"x": 1}')]},
            {"role": "tool", "content": "r1"},
        ]
    )

    assert prompt.endswith("</function_results>\n\n</think>")
    assert prompt.count("<｜Assistant｜>") == 1


def test_deepseek_v32_partial_tool_group_gets_an_opener():
    """A group still awaiting results ends mid-block, so the opener is added."""
    prompt = _render(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [_tool_call("f", '{"x": 1}'), _tool_call("g", "{}")],
            },
            {"role": "tool", "content": "r1"},
        ]
    )

    assert prompt.endswith("<result>r1</result><｜Assistant｜></think>")


# ── continue_final_message ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, f"{BOS}<｜User｜>hi<｜Assistant｜></think>partial"),
        ({"thinking": True}, f"{BOS}<｜User｜>hi<｜Assistant｜><think>partial"),
    ],
    ids=["chat", "thinking"],
)
def test_deepseek_v32_continue_final_message_leaves_turn_open(kwargs, expected):
    """In thinking mode this used to raise: an assistant message after the last
    user was required to carry reasoning or tool calls, which a prefill never
    does."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial"},
    ]
    original = [dict(message) for message in messages]

    prompt = _render(messages, continue_final_message=True, **kwargs)

    assert prompt == expected
    assert EOS not in prompt
    assert prompt.count("<｜Assistant｜>") == 1
    # The caller's message dicts must survive untouched -- this encoder does
    # not copy them.
    assert messages == original


def test_deepseek_v32_continue_final_message_ignored_without_assistant():
    """Nothing to continue, so the request still gets a generation prompt."""
    messages = [{"role": "system", "content": "a"}]

    assert _render(messages, continue_final_message=True) == f"{BOS}a<｜Assistant｜></think>"


def test_deepseek_v32_thinking_still_requires_reasoning_when_not_a_prefill():
    """The prefill exemption must not disarm the reference check."""
    with pytest.raises(ValueError, match="without reasoning/tool_calls"):
        _render(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
            thinking=True,
        )


# ── Degenerate inputs ─────────────────────────────────────────────────


def test_deepseek_v32_accepts_an_empty_conversation():
    """`drop_thinking` used to be derived from `messages[-1]["role"]`, which
    raised IndexError here."""
    assert _render([]) == f"{BOS}<｜Assistant｜></think>"


def test_deepseek_v32_accepts_a_message_without_a_role():
    with pytest.raises(NotImplementedError, match="Unknown role"):
        _render([{"content": "a"}])
