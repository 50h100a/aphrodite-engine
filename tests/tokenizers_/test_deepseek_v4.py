# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aphrodite.entrypoints.chat_utils import parse_chat_messages
from aphrodite.renderers.registry import RENDERER_REGISTRY
from aphrodite.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from aphrodite.tokenizers.registry import TokenizerRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "deepseek_v4"


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
    return get_deepseek_v4_tokenizer(FakeHfTokenizer())


def _model_config():
    return SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
    )


def _load_reference_case(case_id: int):
    data = json.loads((FIXTURES_DIR / f"test_input_{case_id}.json").read_text())
    if isinstance(data, dict):
        return data["messages"], data.get("tools")
    return data, None


def _render_reference_case(case_id: int, **kwargs):
    messages, tools = _load_reference_case(case_id)
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    kwargs.setdefault("add_generation_prompt", False)
    return _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def test_deepseek_v4_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == ("DeepseekV4Tokenizer")
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == ("DeepseekV4Renderer")


def test_deepseek_v4_defaults_to_chat_mode():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>")


def test_deepseek_v4_uses_v4_tool_prompt_from_request_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tokenize=False,
    )

    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "</｜DSML｜tool_calls>" in prompt
    assert "function_calls" not in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜></think>")


def test_deepseek_v4_renders_parsed_history_tool_arguments():
    messages = [
        {"role": "user", "content": "List the repo"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/testbed"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file list",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Edit files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        }
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
    )

    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert 'parameter name="arguments"' not in prompt


@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
def test_deepseek_v4_accepts_openai_reasoning_effort_values(reasoning_effort):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert "Reasoning Effort: Absolute maximum" not in prompt


def test_deepseek_v4_none_reasoning_effort_disables_thinking():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_mode", "expected_effort"),
    [
        ("none", "chat", None),
        ("minimal", "thinking", "high"),
        ("low", "thinking", "high"),
        ("medium", "thinking", "high"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "max"),
        ("max", "thinking", "max"),
        ("unexpected", "thinking", "high"),
    ],
)
def test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        "aphrodite.tokenizers.deepseek_v4.encode_messages",
        fake_encode_messages,
    )

    _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_preserves_reference_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")


def test_deepseek_v4_maps_xhigh_to_reference_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")


_NO_TRAILING_USER = [
    pytest.param(
        [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "system", "content": "c"},
        ],
        "<｜begin▁of▁sentence｜>abc",
        id="system-only",
    ),
    pytest.param(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "note"},
        ],
        "<｜begin▁of▁sentence｜><｜User｜>hinote",
        id="system-last",
    ),
]


@pytest.mark.parametrize(("messages", "body"), _NO_TRAILING_USER)
@pytest.mark.parametrize(
    ("kwargs", "opener"),
    [
        ({}, "<｜Assistant｜></think>"),
        ({"thinking": True}, "<｜Assistant｜><think>"),
    ],
    ids=["chat", "thinking"],
)
def test_deepseek_v4_adds_generation_prefix_without_trailing_user(messages, body, kwargs, opener):
    """The reference encoder appends the opener only when the last message is
    user/developer (or carries a task), so these conversations used to render
    with no think marker at all -- which wedged the structured-output bitmask
    closed for the whole request, because the gate reads "no marker" as
    "reasoning still pending". `add_generation_prompt` now supplies it.

    See TestIsReasoningEnd in tests/parser/engine/test_parser_engine.py for the
    gate-side half of that fix.
    """
    prompt = _tokenizer().apply_chat_template(messages, tokenize=False, **kwargs)

    assert prompt == body + opener


@pytest.mark.parametrize(("messages", "body"), _NO_TRAILING_USER)
def test_deepseek_v4_generation_prefix_can_be_suppressed(messages, body):
    prompt = _tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    assert prompt == body


def test_deepseek_v4_suppresses_generation_prefix_after_user():
    """The flag is live in both directions, not merely additive."""
    prompt = _tokenizer().apply_chat_template(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert prompt == "<｜begin▁of▁sentence｜>a<｜User｜>hi"


def test_deepseek_v4_adds_exactly_one_prefix_after_user():
    """A conversation already in generation position must not gain a second
    opener."""
    prompt = _tokenizer().apply_chat_template(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}],
        tokenize=False,
    )

    assert prompt == "<｜begin▁of▁sentence｜>a<｜User｜>hi<｜Assistant｜></think>"


def test_deepseek_v4_keeps_reminder_after_the_opener():
    """`latest_reminder` is injected *after* the opener by design, so a
    trailing reminder must not be mistaken for a missing prefix."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "latest_reminder", "content": "r"},
        ],
        tokenize=False,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think><｜latest_reminder｜>r")
    assert prompt.count("<｜Assistant｜>") == 1


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("query", "<｜begin▁of▁sentence｜><｜User｜>hi<｜query｜>"),
        ("action", "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think><｜action｜>"),
    ],
)
def test_deepseek_v4_task_tokens_are_structural(task, expected):
    """Task special tokens mark the generation position themselves, so they are
    emitted regardless of the flag and never gain an extra opener."""
    messages = [{"role": "user", "content": "hi", "task": task}]

    assert _tokenizer().apply_chat_template(messages, tokenize=False) == expected
    assert (
        _tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        == expected
    )


def test_deepseek_v4_continue_final_message_leaves_turn_open():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial"},
    ]
    original = [dict(message) for message in messages]

    prompt = _tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>partial")
    # No closing EOS, and no second opener after the prefill.
    assert "<｜end▁of▁sentence｜>" not in prompt
    assert prompt.count("<｜Assistant｜>") == 1
    # The caller's message dicts must survive untouched.
    assert messages == original


def test_deepseek_v4_tool_result_needs_no_extra_opener():
    """Tool results are merged into a user turn, which already ends in the
    opener, so a tool-terminated conversation must not gain a second one."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
        tokenize=False,
    )

    assert prompt.endswith("<｜User｜><tool_result>r</tool_result><｜Assistant｜></think>")
    assert prompt.count("<｜Assistant｜>") == 2  # one per user turn


def test_deepseek_v4_accepts_an_empty_conversation():
    assert _tokenizer().apply_chat_template([], tokenize=False) == ("<｜begin▁of▁sentence｜><｜Assistant｜></think>")
    assert (
        _tokenizer().apply_chat_template(
            [],
            tokenize=False,
            add_generation_prompt=False,
        )
        == "<｜begin▁of▁sentence｜>"
    )


def test_deepseek_v4_closed_assistant_turn_gets_a_fresh_opener():
    """Without `continue_final_message` a trailing assistant turn is closed, so
    asking for a generation prompt opens a new one."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "done"},
        ],
        tokenize=False,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>done<｜end▁of▁sentence｜><｜Assistant｜></think>"
    )


@pytest.mark.parametrize(
    ("case_id", "kwargs"),
    [
        (1, {"thinking": True}),
        (2, {"thinking": True}),
        (3, {"thinking": True}),
        (4, {}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    """The fixtures are training-format renderings -- every one ends in a
    closed assistant turn -- so they are compared with the generation prompt
    switched off."""
    prompt = _render_reference_case(case_id, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


@pytest.mark.parametrize(
    ("case_id", "kwargs", "opener"),
    [
        (1, {"thinking": True}, "<｜Assistant｜><think>"),
        (2, {"thinking": True}, "<｜Assistant｜><think>"),
        (3, {"thinking": True}, "<｜Assistant｜><think>"),
        (4, {}, "<｜Assistant｜></think>"),
    ],
)
def test_deepseek_v4_reference_cases_gain_an_opener_when_asked(case_id, kwargs, opener):
    """Guards the `add_generation_prompt=False` default in
    `_render_reference_case`: the goldens are unchanged only because the flag
    is off, not because the flag is inert."""
    prompt = _render_reference_case(case_id, add_generation_prompt=True, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected + opener
