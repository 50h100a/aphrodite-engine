# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stock-XGrammar native XML must decode to schema-valid, exact JSON values."""

import json

import pytest
from jsonschema import Draft202012Validator
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.parser.qwen3 import Qwen3Parser
from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag, limit_to_single_tool_call
from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import collect_tool_arguments, simulate_tool_streaming


def obj(prop, **kwargs):
    return {
        "type": "object",
        "properties": {"value": prop},
        "required": ["value"],
        "additionalProperties": False,
        **kwargs,
    }


def tools(schema, strict=True):
    return [
        ChatCompletionToolsParam(
            type="function", function={"name": "record_value", "parameters": schema, "strict": strict}
        )
    ]


def native(values, name="record_value"):
    params = "\n".join(f"<parameter={key}>\n{value}\n</parameter>" for key, value in values.items())
    body = params + "\n" if params else ""
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def grammar(schema, choice="required", reasoning=False, model="qwen_3_coder", strict=True):
    if isinstance(choice, dict):
        choice = ChatCompletionNamedToolChoiceParam.model_validate(choice)
    tag = get_model_structural_tag(model, tools(schema, strict), choice, reasoning)
    return Grammar.from_structural_tag(tag)


def parsed(schema, text, chunk_size):
    declared = tools(schema)
    request = ChatCompletionRequest(model="offline", messages=[], tools=declared)
    parser = Qwen3Parser(make_mock_tokenizer({}), declared, chat_template_kwargs={"enable_thinking": False})
    if chunk_size is None:
        result = parser.extract_tool_calls(text, request).tool_calls[0].function.arguments
    else:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        deltas = simulate_tool_streaming(parser, request, chunks)
        final = parser.finish_streaming()
        result = collect_tool_arguments(deltas + [(final, text)])
    return json.loads(result)


@pytest.mark.parametrize("model", ["qwen_3_coder", "qwen_3_5"])
@pytest.mark.parametrize("reasoning", [False, True])
def test_auto_reserves_native_markers_outside_arguments(model, reasoning):
    schema = obj({"type": "string"})
    g = grammar(schema, choice="auto", reasoning=reasoning, model=model)
    prefix = "Think.</think>\n\n" if reasoning else ""
    call = native({"value": "safe"})
    assert _is_grammar_accept_string(g, prefix + "Here is the result. " + call + " Done.")
    for stray in (
        "</function>",
        "</tool_call>",
        "</parameter>",
        "<parameter=value>",
        "<function=record_value>",
        "<tool_call>wrong",
    ):
        for text in (stray + call, call + stray):
            assert not _is_grammar_accept_string(g, prefix + text)
    # Marker-shaped string data is still data inside the schema-controlled body.
    value = "literal </tool_call> and </function>"
    assert _is_grammar_accept_string(g, prefix + native({"value": value}))
    assert parsed(schema, native({"value": value}), 1) == {"value": value}


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("constraint", ["enum", "const", "reference"])
@pytest.mark.parametrize(
    "value",
    [
        " stable",
        "stable ",
        " stable ",
        "\tstable\t",
        "stable\n",
        "stable\r\n",
        "",
        "  ",
        "\u2003stable\u00a0",
        "null",
        "2",
    ],
)
def test_exact_constrained_strings(chunk_size, constraint, value):
    prop = {
        "type": "string",
        constraint if constraint != "reference" else "enum": value if constraint == "const" else [value],
    }
    schema = obj(prop)
    if constraint == "reference":
        schema["$defs"] = {"value": prop}
        schema["properties"]["value"] = {"$ref": "#/$defs/value"}
    text = native({"value": value})
    g = grammar(schema)
    assert _is_grammar_accept_string(g, text)
    for wrong in ["\n" + value, value + "\n", " \t" + value]:
        assert not _is_grammar_accept_string(g, native({"value": wrong}))
    assert parsed(schema, text, chunk_size) == {"value": value}


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize(
    ("prop", "raw", "expected"),
    [
        ({"const": 2}, "2", 2),
        ({"enum": [False]}, "false", False),
        ({"type": "string", "minLength": 3, "maxLength": 5}, "\nx\n", "\nx\n"),
        ({"type": "string"}, "\n\r\nhello\r\n\n", "\n\r\nhello\r\n\n"),
        ({"type": ["string", "null"], "enum": ["null"]}, "null", "null"),
        ({"anyOf": [{"type": "string", "const": "2"}, {"type": "integer", "minimum": 5}]}, "2", "2"),
        ({"type": ["string", "null"], "enum": [None]}, "null", None),
        ({"type": ["string", "integer"], "enum": [2, "x"]}, "2", 2),
        ({"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}, '{"s":"2"}', {"s": "2"}),
        ({"type": "array", "items": {"type": "string"}}, '["2","null"]', ["2", "null"]),
        ({"type": "string", "pattern": "^x+$"}, "xxx", "xxx"),
    ],
)
def test_native_value_schemas(chunk_size, prop, raw, expected):
    schema = obj(prop)
    text = native({"value": raw})
    assert _is_grammar_accept_string(grammar(schema), text)
    result = parsed(schema, text, chunk_size)
    assert result == {"value": expected}
    Draft202012Validator(schema).validate(result)


@pytest.mark.parametrize("kind", ["oneOf", "anyOf"])
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("branch", ["string", "number"])
def test_later_discriminator_selects_earlier_type(kind, chunk_size, branch):
    schema = {
        "type": "object",
        kind: [
            {
                "type": "object",
                "properties": {"value": {"type": "string", "const": "2"}, "kind": {"const": "string"}},
                "required": ["value", "kind"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"value": {"type": "integer", "const": 2}, "kind": {"const": "number"}},
                "required": ["value", "kind"],
                "additionalProperties": False,
            },
        ],
    }
    text = native({"value": "2", "kind": branch})
    g = grammar(schema)
    assert _is_grammar_accept_string(g, text)
    result = parsed(schema, text, chunk_size)
    assert result == {"value": "2" if branch == "string" else 2, "kind": branch}
    Draft202012Validator(schema).validate(result)
    assert not _is_grammar_accept_string(g, native({"value": "3", "kind": branch}))


@pytest.mark.parametrize("kind", ["oneOf", "anyOf"])
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_root_alternative_with_disjoint_keys(kind, chunk_size):
    schema = {
        "type": "object",
        kind: [
            {"type": "object", "properties": {"a": {"const": "A"}}, "required": ["a"], "additionalProperties": False},
            {"type": "object", "properties": {"b": {"const": 2}}, "required": ["b"], "additionalProperties": False},
        ],
    }
    text = native({"b": "2"})
    assert _is_grammar_accept_string(grammar(schema), text)
    assert parsed(schema, text, chunk_size) == {"b": 2}


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_nested_json_reference_context(chunk_size):
    schema = obj({"$ref": "#/$defs/row"})
    schema["$defs"] = {
        "s": {"type": "string", "enum": ["2", "null"]},
        "row": {
            "type": "object",
            "properties": {"s": {"$ref": "#/$defs/s"}},
            "required": ["s"],
            "additionalProperties": False,
        },
    }
    text = native({"value": '{"s":"2"}'})
    assert _is_grammar_accept_string(grammar(schema), text)
    assert parsed(schema, text, chunk_size) == {"value": {"s": "2"}}


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": {"type": "string", "enum": ["\r\n"]}},
        {"type": "object", "properties": {"known": {"const": "ok"}}, "additionalProperties": {"const": 2}},
    ],
)
def test_dynamic_parameters_have_exact_boundaries(chunk_size, schema):
    value = "\r\n" if not schema.get("properties") else "2"
    text = native({"extra": value})
    assert _is_grammar_accept_string(grammar(schema), text)
    result = parsed(schema, text, chunk_size)
    assert result == {"extra": "\r\n" if value == "\r\n" else 2}
    Draft202012Validator(schema).validate(result)


def test_dynamic_key_cannot_overwrite_named_required_field():
    schema = {
        "type": "object",
        "properties": {"known": {"const": "ok"}},
        "required": ["known"],
        "additionalProperties": {"const": 2},
    }
    g = grammar(schema)
    text = native({"known": "ok", "extra": "2"})
    assert _is_grammar_accept_string(g, text)
    assert not _is_grammar_accept_string(g, text.replace("<parameter=extra>", "<parameter=known>"))


@pytest.mark.parametrize("choice", ["auto", "required", {"type": "function", "function": {"name": "record_value"}}])
@pytest.mark.parametrize("model", ["qwen_3_5", "qwen_3_coder"])
@pytest.mark.parametrize("reasoning", [False, True])
def test_choice_and_reasoning_envelopes(choice, model, reasoning):
    schema = obj({"const": "stable\r\n"})
    g = grammar(schema, choice, reasoning, model)
    text = native({"value": "stable\r\n"})
    prefix = "reasoning</think>\n\n" if reasoning else ""
    assert _is_grammar_accept_string(g, prefix + text)
    assert not _is_grammar_accept_string(g, prefix + native({"value": "wrong"}))
    assert not _is_grammar_accept_string(g, prefix + native({"value": "stable\r\n"}, "wrong_name"))
    if choice == "auto":
        assert _is_grammar_accept_string(g, prefix + "ordinary response")
    if not isinstance(choice, str):
        assert not _is_grammar_accept_string(g, prefix + text + text)


def test_single_call_flag_still_applies():
    tag = get_model_structural_tag("qwen_3_coder", tools(obj({"const": "x"})), "required", False)
    g = Grammar.from_structural_tag(limit_to_single_tool_call(tag))
    text = native({"value": "x"})
    assert _is_grammar_accept_string(g, text)
    assert not _is_grammar_accept_string(g, text + text)


@pytest.mark.parametrize(
    "schema",
    [
        obj({"type": "string", "minLength": "bad"}),
        obj({"enum": [1, 2], "minimum": 2}),
        obj({"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 10}]}),
        obj({"$ref": "#/$defs/missing"}),
        {"type": "object", "minProperties": 2, "additionalProperties": True},
        {"type": "object", "patternProperties": {"x": {"type": "string"}}},
    ],
)
def test_unsupported_original_schemas_rejected(schema):
    with pytest.raises(AphroditeValidationError):
        grammar(schema)
    # Explicit opt-out constrains the envelope, regardless of schema support.
    assert _is_grammar_accept_string(grammar(schema, strict=False), native({"anything": "arbitrary"}))


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_closed_object_cardinality(count):
    schema = {
        "type": "object",
        "properties": {key: {"const": "x"} for key in ("a", "b", "c")},
        "minProperties": 1,
        "maxProperties": 2,
        "additionalProperties": False,
    }
    text = native({key: "x" for key in ("a", "b", "c")[:count]})
    # Empty composition uses a single newline between function open and close.
    if not count:
        text = text.replace("record_value>\n\n</function>", "record_value>\n</function>")
    assert _is_grammar_accept_string(grammar(schema), text) == (1 <= count <= 2)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_whole_object_enum_types(chunk_size):
    schema = {"enum": [{"a": 2, "b": "null"}, {"a": 3, "b": "x"}]}
    text = native({"a": "2", "b": "null"})
    assert _is_grammar_accept_string(grammar(schema), text)
    assert parsed(schema, text, chunk_size) == {"a": 2, "b": "null"}


def test_unambiguous_string_still_streams_before_value_closes():
    schema = obj({"type": "string"})
    declared = tools(schema)
    request = ChatCompletionRequest(model="offline", messages=[], tools=declared)
    parser = Qwen3Parser(make_mock_tokenizer({}), declared, chat_template_kwargs={"enable_thinking": False})
    text = native({"value": "hello world"})
    cut = text.index("world")
    deltas = simulate_tool_streaming(parser, request, list(text[:cut]))
    prefix = collect_tool_arguments(deltas)
    assert "hello " in prefix
    assert not prefix.endswith('"}')


def test_ambiguous_root_sends_no_argument_bytes_until_discriminator_is_known():
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"value": {"const": "2"}, "kind": {"const": "s"}},
                "required": ["value", "kind"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"value": {"const": 2}, "kind": {"const": "n"}},
                "required": ["value", "kind"],
                "additionalProperties": False,
            },
        ]
    }
    declared = tools(schema)
    request = ChatCompletionRequest(model="offline", messages=[], tools=declared)
    parser = Qwen3Parser(make_mock_tokenizer({}), declared, chat_template_kwargs={"enable_thinking": False})
    text = native({"value": "2", "kind": "n"})
    cut = text.index("</function>")
    deltas = simulate_tool_streaming(parser, request, list(text[:cut]))
    assert collect_tool_arguments(deltas) == ""
    delta = parser.extract_tool_calls_streaming(text[:cut], text, text[cut:], (), (), (), request)
    assert json.loads(collect_tool_arguments([(delta, text)])) == {"value": 2, "kind": "n"}


@pytest.mark.parametrize("raw", ["x", "123456", "\u2003\u00a0"])
def test_bounded_raw_string_rejects_invalid_length(raw):
    assert not _is_grammar_accept_string(
        grammar(obj({"type": "string", "minLength": 3, "maxLength": 5})), native({"value": raw})
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_root_reference_retains_numeric_property_context(chunk_size):
    schema = {
        "$ref": "#/$defs/root",
        "$defs": {
            "root": obj({"$ref": "#/$defs/n"}),
            "n": {"const": 2},
        },
    }
    text = native({"value": "2"})
    assert _is_grammar_accept_string(grammar(schema), text)
    assert parsed(schema, text, chunk_size) == {"value": 2}


def test_opted_out_unresolvable_schema_is_only_a_hint():
    assert parsed(obj({"$ref": "#/$defs/missing"}), native({"value": "anything"}), 1) == {"value": "anything"}


def test_zero_property_bound_and_forbidden_optional_property():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "forbidden": False},
        "maxProperties": 0,
        "additionalProperties": False,
    }
    g = grammar(schema)
    assert _is_grammar_accept_string(g, native({}))
    assert not _is_grammar_accept_string(g, native({"a": "x"}))
    assert not _is_grammar_accept_string(g, native({"forbidden": "x"}))


def test_invalid_opt_out_schema_keeps_legacy_type_hints():
    from tests.tool_parsers.test_qwen3coder_tool_parser import test_extract_tool_calls_type_conversion

    # The existing compatibility fixture uses the nonstandard type 'float'.
    # It is refused for enforcement, but must remain a usable permissive hint.
    test_extract_tool_calls_type_conversion(make_mock_tokenizer({}))


def test_nested_reference_preserves_legacy_dialect_during_preflight():
    from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
    from aphrodite.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    # Draft 7 ignores the sibling minimum; losing the document dialect while
    # extracting this property would wrongly reject its valid constant 2.
    schema = obj(
        {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/n", "minimum": 5}},
            "required": ["n"],
            "additionalProperties": False,
        }
    )
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["$defs"] = {"n": {"const": 2}}
    tag = get_model_structural_tag("qwen_3_coder", tools(schema), "required", False)
    params = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=tag.model_dump_json()))
    validate_xgrammar_grammar(params)
    text = native({"value": '{"n":2}'})
    assert _is_grammar_accept_string(Grammar.from_structural_tag(tag), text)
    assert parsed(schema, text, 1) == {"value": {"n": 2}}


@pytest.mark.parametrize("key", ["enum", "const", "$id", "$ref"])
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_reference_rebasing_distinguishes_field_names_from_schema_keywords(key, chunk_size):
    schema = obj(
        {"type": "object", "properties": {key: {"$ref": "#/$defs/n"}}, "required": [key], "additionalProperties": False}
    )
    schema["$defs"] = {"n": {"const": 2}}
    text = native({"value": json.dumps({key: 2})})
    assert _is_grammar_accept_string(grammar(schema), text)
    assert parsed(schema, text, chunk_size) == {"value": {key: 2}}
