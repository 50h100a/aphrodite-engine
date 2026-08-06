# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The gpt-oss render path builds its own prompt and returns early.

That made it the one path that never reached `adjust_request`, so gpt-oss
requests were sampled with no structural tag and no schema screening -- the
tool parser's grammar was built and then dropped on the floor. These tests pin
both halves: that the harmony branch asks the parsers to adjust the request,
and that doing so is what puts the tag on it.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.parser.harmony import HarmonyParser
from aphrodite.renderers.online_renderer import OnlineRenderer
from aphrodite.tool_parsers.gptoss_tool_parser import GptOssToolParser

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={  # type: ignore[arg-type]
                "name": "apply_label",
                "description": "Attach a release label to a build.",
                "parameters": {
                    "type": "object",
                    "properties": {"build": {"type": "string"}},
                    "required": ["build"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        )
    ]


def _renderer(parser) -> OnlineRenderer:
    hf_config = SimpleNamespace(model_type="gpt_oss")
    renderer = OnlineRenderer.__new__(OnlineRenderer)
    renderer.model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        tokenizer="openai/gpt-oss-20b",
    )
    renderer.renderer = MagicMock()
    renderer.renderer.get_tokenizer.return_value = MagicMock()
    renderer.parser = parser
    renderer.use_harmony = True
    renderer.enable_auto_tools = True
    renderer.exclude_tools_when_tool_choice_none = False
    return renderer


def test_harmony_branch_adjusts_the_request(monkeypatch: pytest.MonkeyPatch, tools):
    """The gpt-oss branch must hand the request to the parsers like every
    other render path does."""
    renderer = _renderer(HarmonyParser)
    monkeypatch.setattr(
        OnlineRenderer,
        "_make_request_with_harmony",
        lambda self, request, should_include_tools=True: ([], []),
    )

    seen: list[object] = []
    monkeypatch.setattr(
        OnlineRenderer,
        "adjust_request_for_parsers",
        lambda self, request, parser, chat_template_kwargs: seen.append((request, parser)),
    )

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Label build b-1."}],  # type: ignore[list-item]
        model="m",
        tools=tools,
        tool_choice="required",
    )

    import asyncio

    asyncio.run(renderer.render_chat(request))

    assert seen == [(request, HarmonyParser)]


def test_adjustment_installs_the_structural_tag(monkeypatch: pytest.MonkeyPatch, tools):
    """And the adjustment is what puts a grammar on a gpt-oss request."""
    monkeypatch.setattr(HarmonyParser, "tool_parser_cls", GptOssToolParser)
    monkeypatch.setattr(HarmonyParser, "reasoning_parser_cls", None)

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Label build b-1."}],  # type: ignore[list-item]
        model="m",
        tools=tools,
        tool_choice="required",
    )
    assert request.structured_outputs is None

    _renderer(HarmonyParser).adjust_request_for_parsers(request, HarmonyParser, None)

    assert request.structured_outputs is not None
    raw_tag = request.structured_outputs.structural_tag
    assert raw_tag is not None
    tag = json.loads(raw_tag)
    begins = [t["begin"] for t in tag["format"]["tags"]]
    assert any(b.startswith("<|channel|>commentary to=functions.apply_label") for b in begins)
    # And the gate that decides whether the bitmask is ever applied.
    assert request._grammar_from_tool_parser is True
