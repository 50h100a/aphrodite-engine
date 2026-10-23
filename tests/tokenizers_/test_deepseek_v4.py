# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aphrodite.renderers.chat_utils import parse_chat_messages
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
    # Preserve content blocks for encoding without fetching fixture media.
    return _tokenizer().apply_chat_template(
        conversation=messages,
        messages=messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def test_deepseek_v4_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == ("DeepseekV4Tokenizer")
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == ("DeepseekV4Renderer")


def test_deepseek_v4_defaults_to_thinking_with_high_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("kwargs", [{"thinking": False}, {"enable_thinking": False}])
def test_deepseek_v4_explicitly_disables_thinking(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("enable_thinking", "thinking_token"),
    [(False, "</think>"), (True, "<think>")],
)
def test_deepseek_v4_appends_assistant_transition_after_trailing_system(enable_thinking, thinking_token):
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "system", "content": "Be concise."},
        ],
        tokenize=False,
        enable_thinking=enable_thinking,
        reasoning_effort="low",
    )

    assert prompt == (
        f"<｜begin▁of▁sentence｜>You are helpful.<｜User｜>What is 2+2?Be concise.<｜Assistant｜>{thinking_token}"
    )


def test_deepseek_v4_does_not_transition_after_mid_conversation_system():
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "6"},
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "What is 5+5?"},
        ],
        tokenize=False,
        enable_thinking=False,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>What is 3+3?"
        "<｜Assistant｜></think>6<｜end▁of▁sentence｜>"
        "Answer in French.<｜User｜>What is 5+5?"
        "<｜Assistant｜></think>"
    )


@pytest.mark.parametrize(
    ("enable_thinking", "expected"),
    [
        (
            False,
            "<｜begin▁of▁sentence｜>rules<｜Assistant｜></think>answer<｜end▁of▁sentence｜>",
        ),
        (
            True,
            "<｜begin▁of▁sentence｜>rules<｜Assistant｜><think>reason</think>answer<｜end▁of▁sentence｜>",
        ),
    ],
)
def test_deepseek_v4_transitions_from_system_to_assistant(enable_thinking, expected):
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "rules"},
            {
                "role": "assistant",
                "reasoning": "reason",
                "content": "answer",
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        reasoning_effort="low",
    )

    assert prompt == expected


def test_deepseek_v4_unknown_role_raises_value_error():
    # Invalid roles are client errors: they must surface as ValueError
    # (mapped to HTTP 400 by the OpenAI serving layer), not
    # NotImplementedError (mapped to HTTP 501).
    with pytest.raises(ValueError, match="Invalid role: SYSTEM"):
        _tokenizer().apply_chat_template(
            [{"role": "SYSTEM", "content": "Hello"}],
            tokenize=False,
        )


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
    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜><think>")


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


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_prefix"),
    [
        ("low", "<｜begin▁of▁sentence｜><｜User｜>Hello"),
        ("high", "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"),
        ("max", "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"),
    ],
)
def test_deepseek_v4_renders_0731_reasoning_effort_prompts(reasoning_effort, expected_prefix):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert prompt.startswith(expected_prefix)


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
        ("minimal", "thinking", "low"),
        ("low", "thinking", "low"),
        ("medium", "thinking", "low"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "high"),
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


def test_deepseek_v4_renders_0731_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum")


def test_deepseek_v4_maps_xhigh_to_high_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")


def _render_prompt(messages, **kwargs):
    kwargs.setdefault("enable_thinking", kwargs.get("thinking", False))
    kwargs.setdefault("reasoning_effort", "low")
    kwargs.setdefault("tokenize", False)
    return _tokenizer().apply_chat_template(messages, **kwargs)


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
    prompt = _render_prompt(messages, **kwargs)

    assert prompt == body + opener


@pytest.mark.parametrize(("messages", "body"), _NO_TRAILING_USER)
def test_deepseek_v4_generation_prefix_can_be_suppressed(messages, body):
    prompt = _render_prompt(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    assert prompt == body


def test_deepseek_v4_suppresses_generation_prefix_after_user():
    """The flag is live in both directions, not merely additive."""
    prompt = _render_prompt(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert prompt == "<｜begin▁of▁sentence｜>a<｜User｜>hi"


def test_deepseek_v4_adds_exactly_one_prefix_after_user():
    """A conversation already in generation position must not gain a second
    opener."""
    prompt = _render_prompt(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}],
        tokenize=False,
    )

    assert prompt == "<｜begin▁of▁sentence｜>a<｜User｜>hi<｜Assistant｜></think>"


def test_deepseek_v4_keeps_reminder_after_the_opener():
    """`latest_reminder` is injected *after* the opener by design, so a
    trailing reminder must not be mistaken for a missing prefix."""
    prompt = _render_prompt(
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

    assert _render_prompt(messages, tokenize=False) == expected
    assert (
        _render_prompt(
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

    prompt = _render_prompt(
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
    prompt = _render_prompt(
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
    assert _render_prompt([], tokenize=False) == ("<｜begin▁of▁sentence｜><｜Assistant｜></think>")
    assert (
        _render_prompt(
            [],
            tokenize=False,
            add_generation_prompt=False,
        )
        == "<｜begin▁of▁sentence｜>"
    )


def test_deepseek_v4_closed_assistant_turn_gets_a_fresh_opener():
    """Without `continue_final_message` a trailing assistant turn is closed, so
    asking for a generation prompt opens a new one."""
    prompt = _render_prompt(
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
        (1, {"thinking": True, "reasoning_effort": "low"}),
        (2, {"thinking": True, "reasoning_effort": "low"}),
        (3, {"thinking": True, "reasoning_effort": "low"}),
        (4, {"thinking": False}),
        (5, {"thinking": False}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    """Closed assistant fixtures stay closed; the user-ended fixture opens generation."""
    prompt = _render_reference_case(
        case_id, add_generation_prompt=_load_reference_case(case_id)[0][-1]["role"] != "assistant", **kwargs
    )

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


def test_deepseek_v4_rejects_empty_developer_content():
    with pytest.raises(ValueError):
        _render_prompt(
            [{"role": "developer", "content": ""}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort="low",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking_mode": "bogus"},
        {"thinking_mode": "thinking", "reasoning_effort": "turbo"},
    ],
)
def test_deepseek_v4_encode_messages_rejects_invalid_arguments(kwargs):
    from aphrodite.tokenizers.deepseek_v4_encoding import encode_messages

    with pytest.raises(ValueError):
        encode_messages([{"role": "user", "content": "Hello"}], **kwargs)


def test_deepseek_v4_image_blocks_become_placeholders():
    prompt = _render_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first:"},
                    {"type": "image_url", "image_url": {"url": "file:///a.png"}},
                    {"type": "text", "text": "second:"},
                    {"type": "image", "source": {"data": "AAAA"}},
                ],
            }
        ],
        tokenize=False,
        thinking=False,
    )

    assert "<｜User｜>first:\n\n<｜deepseek_image｜>\n\nsecond:\n\n<｜deepseek_image｜>" in prompt


def test_deepseek_v4_image_sentinel_ids_match_tokenizer():
    """The borrowed sentinel ids must line up with the reserved
    ``<|place_holder_mm_span_XXXX|>`` tokens in the tokenizer."""
    from aphrodite.models.deepseek_v4.common.mm_preprocess import (
        IMAGE_SENTINEL_BASE_ID,
        IMAGE_SENTINEL_TOKEN_NAMES,
        image_sentinel_mask,
        validate_image_sentinel_ids,
    )

    class FakeTokenizer:
        def __init__(self, offset: int = 0) -> None:
            self.offset = offset

        def convert_tokens_to_ids(self, token: str) -> int:
            return IMAGE_SENTINEL_BASE_ID + self.offset + (IMAGE_SENTINEL_TOKEN_NAMES.index(token))

    validate_image_sentinel_ids(FakeTokenizer())  # no raise
    with pytest.raises(ValueError, match="sentinel"):
        validate_image_sentinel_ids(FakeTokenizer(offset=1))

    ids = torch.tensor(
        [
            1,
            IMAGE_SENTINEL_BASE_ID,
            IMAGE_SENTINEL_BASE_ID + 4,
            IMAGE_SENTINEL_BASE_ID + 5,
            129264,
        ]
    )
    assert image_sentinel_mask(ids).tolist() == [False, True, True, False, False]


def _request_tools():
    return [
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


def _encode_reference(messages):
    from aphrodite.tokenizers.deepseek_v4_encoding import encode_messages

    return encode_messages(messages, thinking_mode="thinking", reasoning_effort="low")


def test_deepseek_v4_attaches_request_tools_to_existing_system_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]

    prompt = _render_prompt(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference([{**messages[0], "tools": _request_tools()}, messages[1]])
    assert prompt == expected
    assert prompt.startswith("<｜begin▁of▁sentence｜>You are helpful.\n\n## Tools")
    assert prompt.count("## Tools") == 1


def test_deepseek_v4_synthetic_system_only_when_no_system_message():
    """Without a system message, request tools still get a synthetic leading
    system entry."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]

    prompt = _render_prompt(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference([{"role": "system", "tools": _request_tools()}, messages[0]])
    assert prompt == expected
    assert prompt.startswith("<｜begin▁of▁sentence｜>\n\n## Tools")


def test_deepseek_v4_request_tools_override_system_message_tools():
    message_tools = [
        {
            "type": "function",
            "function": {
                "name": "message_level_tool",
                "description": "Should not be rendered",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    system_message = {
        "role": "system",
        "content": "You are helpful.",
        "tools": message_tools,
    }
    user_message = {"role": "user", "content": "Weather in Paris?"}

    prompt = _render_prompt(
        [system_message, user_message],
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference([{**system_message, "tools": _request_tools()}, user_message])
    assert prompt == expected
    assert '"name": "get_weather"' in prompt
    assert "message_level_tool" not in prompt


def test_deepseek_v4_attaches_request_tools_to_first_system_message_only():
    """Only the first system message, at any index, gets the request tools."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "First system."},
        {"role": "system", "content": "Second system."},
        {"role": "user", "content": "Weather in Paris?"},
    ]

    prompt = _render_prompt(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference(
        [
            messages[0],
            {**messages[1], "tools": _request_tools()},
            messages[2],
            messages[3],
        ]
    )
    assert prompt == expected
    assert "First system.\n\n## Tools" in prompt
    assert prompt.count("## Tools") == 1


def test_deepseek_v4_request_tools_do_not_mutate_caller_messages():
    import copy

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    snapshot = copy.deepcopy(messages)

    _render_prompt(messages, tools=_request_tools(), tokenize=False)

    assert messages == snapshot
