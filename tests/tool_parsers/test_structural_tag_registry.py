# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

import aphrodite.envs as envs
from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.parser.abstract_parser import DelegatingParser
from aphrodite.tool_parsers.abstract_tool_parser import ToolParser
from aphrodite.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from aphrodite.tool_parsers.deepseekv4_engine_tool_parser import (
    DeepSeekV4EngineToolParser,
)
from aphrodite.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser
from aphrodite.tool_parsers.deepseekv32_engine_tool_parser import (
    DeepSeekV32EngineToolParser,
)
from aphrodite.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from aphrodite.tool_parsers.gptoss_tool_parser import GptOssToolParser
from aphrodite.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from aphrodite.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from aphrodite.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from aphrodite.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from aphrodite.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from aphrodite.tool_parsers.structural_tag_registry import (
    APHRODITE_BUILTIN_STRUCTURAL_TAG_MODELS,
    SUPPORTED_STRUCTURAL_TAG_MODELS,
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS,
    _get_function_parameters,
    get_model_structural_tag,
)


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


@pytest.fixture
def sample_tools_strict() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


def test_supported_structural_tag_models_include_aphrodite_builtins():
    assert SUPPORTED_STRUCTURAL_TAG_MODELS == (
        XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | APHRODITE_BUILTIN_STRUCTURAL_TAG_MODELS
    )
    assert "hermes" in APHRODITE_BUILTIN_STRUCTURAL_TAG_MODELS


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_all_xgrammar_builtins(
    model: str,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


def test_get_model_structural_tag_supports_aphrodite_hermes(
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="hermes",
        tools=sample_tools,
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)

    # Assert the semantically meaningful structure rather than the full
    # model_dump(), which gains version-specific keys across xgrammar releases
    # (e.g. "any_order" was added to json_schema content in 0.2.3).
    dump = tag.model_dump()
    assert dump["type"] == "structural_tag"

    fmt = dump["format"]
    assert fmt["type"] == "tags_with_separator"
    assert fmt["separator"] == ""
    assert fmt["at_least_one"] is True
    assert fmt["stop_after_first"] is False

    expected_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    expected_tags = [
        ('<tool_call>\n{"name": "get_weather", "arguments": ', "}\n</tool_call>"),
        ('<tool_call>{"name": "get_weather", "arguments": ', "}</tool_call>"),
    ]
    assert len(fmt["tags"]) == len(expected_tags)
    for tag_dump, (begin, end) in zip(fmt["tags"], expected_tags):
        assert tag_dump["type"] == "tag"
        assert tag_dump["begin"] == begin
        assert tag_dump["end"] == end
        content = tag_dump["content"]
        assert content["type"] == "json_schema"
        assert content["json_schema"] == expected_schema


def test_hermes_required_tool_calls_use_empty_separator():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
    ]

    tag = get_model_structural_tag(
        model="hermes",
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )

    assert tag is not None
    assert tag.format.separator == ""


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_named_tool_choice(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice=ChatCompletionNamedToolChoiceParam(function=ChatCompletionNamedFunction(name="get_weather")),
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


@pytest.mark.parametrize(
    ("parser_cls", "model"),
    [
        (DeepSeekV3ToolParser, "deepseek_r1"),
        (DeepSeekV31ToolParser, "deepseek_v3_1"),
        (DeepSeekV32EngineToolParser, "deepseek_v3_2"),
        (DeepSeekV4EngineToolParser, "deepseek_v4"),
        (Glm47MoeModelToolParser, "glm_4_7"),
        (GptOssToolParser, "harmony"),
        (Hermes2ProToolParser, "hermes"),
        (KimiK2ToolParser, "kimi"),
        (Llama3JsonToolParser, "llama"),
        (MinimaxM2ToolParser, "minimax"),
        (Qwen3EngineToolParser, "qwen_3_coder"),
    ],
)
def test_tool_parsers_declare_matching_xgrammar_builtin_model(parser_cls, model):
    assert parser_cls.structural_tag_model == model
    assert not parser_cls.supports_required_and_named


def test_tool_parsers_without_structural_tag_support_required_and_named():
    class NonStructuralTagToolParser(ToolParser):
        pass

    assert NonStructuralTagToolParser.structural_tag_model is None
    assert NonStructuralTagToolParser.supports_required_and_named


def test_non_structural_tag_parser_uses_schema_constraints(
    sample_tools: list[ChatCompletionToolsParam],
):
    parser = ToolParser(MagicMock())
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.json is not None
    assert out.structured_outputs.structural_tag is None


def test_get_structural_tag_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "aphrodite.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = Qwen3EngineToolParser(MagicMock(), tools=sample_tools_strict)

    parser.get_structural_tag(request)

    assert captured == [False]


@pytest.mark.parametrize("has_reasoning_parser", [True, False])
def test_unified_parser_matches_reasoning_to_the_parser(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
    has_reasoning_parser: bool,
):
    """The tag is a closed alternation, so a model that reasons needs its
    reasoning segment listed or the grammar forbids its own first token."""
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "aphrodite.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)
    if has_reasoning_parser:
        parser.reasoning_parser = MagicMock(adjust_request=lambda request: request)

    parser.adjust_request(request)

    assert captured == [has_reasoning_parser]


@pytest.mark.parametrize("has_reasoning_parser", [True, False])
def test_harmony_tag_always_admits_the_analysis_channel(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
    has_reasoning_parser: bool,
):
    """The analysis channel belongs to the Harmony format, not to the reasoning
    parser. Dropping the parser stops us surfacing that segment; it does not
    stop the model emitting it, so the grammar must admit it either way."""
    from aphrodite.parser.harmony import HarmonyParser

    monkeypatch.setattr(HarmonyParser, "tool_parser_cls", GptOssToolParser)
    monkeypatch.setattr(HarmonyParser, "reasoning_parser_cls", None)

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="required",
    )
    parser = HarmonyParser(MagicMock(), tools=sample_tools_strict)
    if has_reasoning_parser:
        parser.reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    tag = out.structured_outputs.structural_tag
    assert tag is not None
    assert "<|channel|>analysis<|message|>" in tag


def test_harmony_tag_admits_the_commentary_preamble(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    """gpt-oss narrates on `commentary` with no recipient before it calls a
    tool, and HarmonyParser reads that segment back as content. The xgrammar
    template omits the form, so the grammar has to be told about it."""
    from aphrodite.parser.harmony import HarmonyParser

    monkeypatch.setattr(HarmonyParser, "tool_parser_cls", GptOssToolParser)
    monkeypatch.setattr(HarmonyParser, "reasoning_parser_cls", None)

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="required",
    )
    parser = HarmonyParser(MagicMock(), tools=sample_tools_strict)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    raw_tag = out.structured_outputs.structural_tag
    assert raw_tag is not None
    tag = json.loads(raw_tag)
    begins = [t["begin"] for t in tag["format"]["tags"]]
    assert "<|channel|>commentary<|message|>" in begins
    # The preamble is free text, not a tool call misfiled under it.
    preamble = next(t for t in tag["format"]["tags"] if t["begin"] == "<|channel|>commentary<|message|>")
    assert preamble["content"]["type"] == "any_text"


def test_structural_tag_marks_the_grammar_as_covering_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    """The tag spans the whole reply, so there is no reasoning prelude to wait
    out. Without the flag the bitmask is withheld until the reasoning parser
    sees an end marker, and a reply that is only a tool call never emits one --
    the grammar would compile and never apply."""
    from aphrodite.parser.harmony import HarmonyParser

    monkeypatch.setattr(HarmonyParser, "tool_parser_cls", GptOssToolParser)
    monkeypatch.setattr(HarmonyParser, "reasoning_parser_cls", None)

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="required",
    )
    assert request._grammar_from_tool_parser is False

    out = HarmonyParser(MagicMock(), tools=sample_tools_strict).adjust_request(request)

    assert out._grammar_from_tool_parser is True


def test_no_structural_tag_leaves_the_reasoning_gate_alone(
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    """A parser that never installs a tag must not claim to cover reasoning.
    Its schema constraint applies to the arguments alone, so the bitmask still
    has to wait for whatever comes before them."""

    class NonStructuralTagToolParser(ToolParser):
        pass

    class TestParser(DelegatingParser):
        tool_parser_cls = NonStructuralTagToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    out = TestParser(MagicMock(), tools=sample_tools_strict).adjust_request(request)

    assert out._grammar_from_tool_parser is False


def test_xgrammar_function_parameters_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[list[dict]] = []

    def fake_get_xgrammar_model_structural_tag(*, tools: list[dict], **kwargs):
        captured.append(tools)
        return None

    monkeypatch.setattr(
        "aphrodite.tool_parsers.structural_tag_registry.get_xgrammar_model_structural_tag",
        fake_get_xgrammar_model_structural_tag,
    )

    get_model_structural_tag(
        model="llama",
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert captured[0][0]["function"]["parameters"] == sample_tools_strict[0].function.parameters
    assert sample_tools_strict[0].function.parameters is not None


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_auto_tool_choice_skips_structural_tag_without_strict(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
    )

    assert tag is None


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_constrain_auto_tool_calls_builds_tag_without_strict(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
    monkeypatch: pytest.MonkeyPatch,
):
    """The operator override constrains "auto" that no client opted into."""
    monkeypatch.setattr(envs, "APHRODITE_CONSTRAIN_AUTO_TOOL_CALLS", True)

    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


def test_constrain_auto_tool_calls_still_skips_tool_choice_none(
    sample_tools: list[ChatCompletionToolsParam],
    monkeypatch: pytest.MonkeyPatch,
):
    """ "none" means no call at all, so there is nothing to constrain."""
    monkeypatch.setattr(envs, "APHRODITE_CONSTRAIN_AUTO_TOOL_CALLS", True)

    tag = get_model_structural_tag(
        model="deepseek_v4",
        tools=sample_tools,
        tool_choice="none",
        reasoning=False,
    )

    assert tag is None


def test_freeform_object_parameter_is_opened_for_xgrammar(
    monkeypatch: pytest.MonkeyPatch,
):
    """A bare ``{"type": "object"}`` compiles to a value the model can only
    leave empty, so it is handed to xgrammar with its implied default written
    out. An object that declares properties stays closed -- that closure is what
    rejects an undeclared key."""
    captured: list[list[dict]] = []

    def fake_get_xgrammar_model_structural_tag(*, tools: list[dict], **kwargs):
        captured.append(tools)
        return None

    monkeypatch.setattr(
        "aphrodite.tool_parsers.structural_tag_registry.get_xgrammar_model_structural_tag",
        fake_get_xgrammar_model_structural_tag,
    )

    params = {
        "type": "object",
        "properties": {
            "meta": {"type": "object"},
            "closed": {"type": "object", "properties": {"a": {"type": "string"}}},
            "opted_out": {"type": "object", "additionalProperties": False},
        },
        "required": ["meta"],
    }
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={"name": "blob", "strict": True, "parameters": params},
        )
    ]

    get_model_structural_tag(
        model="deepseek_v4",
        tools=tools,
        tool_choice="auto",
        reasoning=False,
    )

    sent = captured[0][0]["function"]["parameters"]["properties"]
    assert sent["meta"]["additionalProperties"] is True
    assert "additionalProperties" not in sent["closed"]
    assert sent["opted_out"]["additionalProperties"] is False
    # The request's own schema is never mutated on the way through.
    assert params["properties"]["meta"] == {"type": "object"}


def test_get_function_parameters_relaxes_function_strict_false():
    function = SimpleNamespace(
        parameters={"type": "object", "properties": {}},
        strict=False,
    )

    assert _get_function_parameters(function) is True
