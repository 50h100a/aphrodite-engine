# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pointers, and the schemas that carry them.

A tool's parameter schema arrives as its own document, so `#/$defs/Foo`,
`#/definitions/Foo`, `#/properties/bar` and a bare `#` all read from *that*
document's root. `tool_choice: "required"` on a model with no structural tag
nests every tool's parameters inside one combined schema, which re-roots all of
them: the definitions the pointers name are no longer where the pointers say.

What that cost, before the wrapper learned to move pointers with the schema:

* `#/$defs/Foo` survived only for a Chat Completions or Responses `FunctionTool`
  whose `$defs` sat at the top level, because those were hoisted by name;
  draft-07 `definitions` and a namespaced tool's `$defs` were not, and dangled.
* Two tools could not both define a `Node`: hoisting by name made that a hard
  error on a perfectly legal request.
* `#/properties/bar` dangled, and a dangling pointer is a compile error raised
  on the backend's own thread, which reaches the caller as a 500.
* A bare `#` did not dangle. It resolved -- to the array of tool calls -- and
  compiled, and constrained the argument to the wrong schema entirely, with no
  error anywhere.
* Hoisting popped `$defs` off the caller's own tool objects, so the transform
  only worked once: a second pass over the same request produced pointers with
  nothing left to point at.

The parsers read these schemas too, to decide what type a parameter holds, and
pydantic writes every enum and nested model as a bare `{"$ref": "#/$defs/..."}`,
which declares nothing until it is followed.
"""

import copy
import json

import pytest
import xgrammar as xgr
from openai.types.responses import FunctionTool, NamespaceTool
from openai.types.responses.namespace_tool import ToolFunction

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from aphrodite.parser.deepseek_v4 import _dsml_param_value
from aphrodite.tool_parsers.utils import (
    extract_types_from_schema,
    find_tool_properties,
    get_json_schema_from_tools,
)
from aphrodite.v1.structured_output.schema_features import (
    get_schema_validation_error,
    get_unenforceable_reasons,
)

pytestmark = pytest.mark.cpu_test


# `$defs` at the top level, the shape pydantic emits.
DEFS = {
    "type": "object",
    "properties": {"n": {"$ref": "#/$defs/N"}},
    "required": ["n"],
    "$defs": {"N": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
}
# The same thing in draft-07 spelling, which is what pydantic v1 and LangChain
# still emit.
DEFINITIONS = {
    "type": "object",
    "properties": {"n": {"$ref": "#/definitions/N"}},
    "required": ["n"],
    "definitions": {"N": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
}
# A recursive schema spelled the short way, pointing at its own root.
SELF_REF = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}},
    "required": ["name"],
}
# A pointer into the schema's own body rather than into a definition.
PROPERTY_REF = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "alias": {"$ref": "#/properties/name"}},
    "required": ["name"],
}
NO_REFS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def chat_tool(name: str, parameters: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={"name": name, "parameters": copy.deepcopy(parameters), "strict": True},
    )


def compiles(schema: dict) -> bool:
    """Whether a backend can actually build a grammar from this."""
    try:
        xgr.Grammar.from_json_schema(json.dumps(schema))
    except Exception:
        return False
    return True


def accepts(schema: dict, document: object) -> bool:
    """Whether the wrapper schema admits this tool call."""
    import jsonschema

    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError:
        return False
    return True


class TestToolWrapperPointers:
    """Every pointer spelling, through the combined tool-call schema."""

    @pytest.mark.parametrize(
        "parameters,arguments",
        [
            pytest.param(DEFS, {"n": {"name": "x"}}, id="defs"),
            pytest.param(DEFINITIONS, {"n": {"name": "x"}}, id="draft-07-definitions"),
            pytest.param(SELF_REF, {"name": "x", "child": {"name": "y", "child": None}}, id="root-self-ref"),
            pytest.param(PROPERTY_REF, {"name": "x", "alias": "y"}, id="property-pointer"),
            pytest.param(NO_REFS, {"city": "SF"}, id="no-pointers"),
        ],
    )
    def test_wrapper_compiles_and_admits_the_call(self, parameters, arguments):
        schema = get_json_schema_from_tools("required", [chat_tool("a", parameters)])

        assert compiles(schema)
        assert accepts(schema, [{"name": "a", "parameters": arguments}])

    def test_bare_root_pointer_means_the_tool_schema(self):
        """`#` is the tool's own schema, not the array it was nested into.

        This is the silent one: nesting left `#` pointing at the list of tool
        calls, which compiles happily and constrains the argument to something
        the caller never wrote.
        """
        schema = get_json_schema_from_tools("required", [chat_tool("a", SELF_REF)])

        assert accepts(schema, [{"name": "a", "parameters": {"name": "x", "child": {"name": "y"}}}])
        assert not accepts(
            schema,
            [{"name": "a", "parameters": {"name": "x", "child": [{"name": "a", "parameters": {"name": "z"}}]}}],
        )

    def test_two_tools_may_define_the_same_name(self):
        """Each tool's definitions get their own slot, so neither has to win."""
        first = {"type": "object", "properties": {"x": {"$ref": "#/$defs/Item"}}, "$defs": {"Item": {"type": "string"}}}
        second = {
            "type": "object",
            "properties": {"y": {"$ref": "#/$defs/Item"}},
            "$defs": {"Item": {"type": "integer"}},
        }

        schema = get_json_schema_from_tools("required", [chat_tool("a", first), chat_tool("b", second)])

        assert compiles(schema)
        assert accepts(schema, [{"name": "a", "parameters": {"x": "text"}}])
        assert accepts(schema, [{"name": "b", "parameters": {"y": 3}}])
        assert not accepts(schema, [{"name": "b", "parameters": {"y": "text"}}])

    def test_namespaced_tool_keeps_its_definitions(self):
        namespaced = NamespaceTool(
            type="namespace",
            name="ns",
            description="a namespace",
            tools=[ToolFunction(type="function", name="inner", parameters=copy.deepcopy(DEFS), strict=True)],
        )

        schema = get_json_schema_from_tools("required", [namespaced])

        assert compiles(schema)
        assert accepts(schema, [{"name": "ns__inner", "parameters": {"n": {"name": "x"}}}])

    def test_responses_function_tool_keeps_its_definitions(self):
        tool = FunctionTool(type="function", name="a", parameters=copy.deepcopy(DEFS), strict=True)

        schema = get_json_schema_from_tools("required", [tool])

        assert compiles(schema)
        assert accepts(schema, [{"name": "a", "parameters": {"n": {"name": "x"}}}])

    def test_a_schema_without_pointers_is_still_nested_in_place(self):
        """No pointers, nothing to move: the shape callers already see stands."""
        schema = get_json_schema_from_tools("required", [chat_tool("a", NO_REFS)])

        assert "$defs" not in schema
        assert schema["items"]["anyOf"][0]["properties"]["parameters"] == NO_REFS

    def test_the_caller_s_tools_are_left_alone(self):
        tools = [chat_tool("a", DEFS)]
        before = json.dumps([tool.model_dump() for tool in tools], sort_keys=True)

        get_json_schema_from_tools("required", tools)

        assert json.dumps([tool.model_dump() for tool in tools], sort_keys=True) == before

    def test_running_twice_gives_the_same_schema(self):
        """The prompt is rendered from the same tools this reads, and `/render`
        hands the adjusted request back to be sent again. A transform that only
        worked the first time broke both."""
        tools = [chat_tool("a", DEFS)]

        first = get_json_schema_from_tools("required", tools)
        second = get_json_schema_from_tools("required", tools)

        assert first == second
        assert compiles(second)

    def test_forced_function_keeps_its_own_root(self):
        """A named tool decodes against its parameters alone, so its pointers
        were never re-rooted and must stay exactly as written."""
        from aphrodite.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionNamedToolChoiceParam,
        )

        choice = ChatCompletionNamedToolChoiceParam(type="function", function={"name": "a"})
        schema = get_json_schema_from_tools(choice, [chat_tool("a", DEFS)])

        assert schema == DEFS
        assert compiles(schema)


class TestToolPropertyResolution:
    """What the parsers see when they ask a tool what type a parameter holds."""

    PYDANTIC = {
        "$defs": {
            "Color": {"enum": ["red", "green"], "type": "string", "title": "Color"},
            "Inner": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        },
        "type": "object",
        "properties": {
            "color": {"$ref": "#/$defs/Color"},
            "inner": {"$ref": "#/$defs/Inner"},
        },
        "required": ["color", "inner"],
    }

    def test_a_property_behind_a_pointer_reports_its_type(self):
        properties = find_tool_properties([chat_tool("f", self.PYDANTIC)], "f")

        assert properties["color"]["enum"] == ["red", "green"]
        assert extract_types_from_schema(properties["inner"]) == ["object"]

    def test_an_enum_behind_a_pointer_gets_its_padding_stripped(self):
        """The grammar lets a model indent a value; the schema says whether the
        indentation is part of it. An enum member's cannot be -- but only if the
        enum is visible, and behind a `$ref` it was not, so the padding was
        handed to the caller as part of the value."""
        properties = find_tool_properties([chat_tool("f", self.PYDANTIC)], "f")

        assert _dsml_param_value("color", "true", "\n\tred\n  ", properties) == "red"

    def test_the_tool_s_own_root_may_be_a_pointer(self):
        parameters = {
            "$ref": "#/$defs/Args",
            "$defs": {"Args": {"type": "object", "properties": {"q": {"type": "integer"}}}},
        }

        properties = find_tool_properties([chat_tool("g", parameters)], "g")

        assert extract_types_from_schema(properties["q"]) == ["integer"]

    def test_a_recursive_definition_terminates(self):
        parameters = {
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/N"}},
            "$defs": {
                "N": {"type": "object", "properties": {"name": {"type": "string"}, "next": {"$ref": "#/$defs/N"}}}
            },
        }

        properties = find_tool_properties([chat_tool("r", parameters)], "r")

        # Expanded once, and the tail left as the pointer it is.
        assert properties["root"]["properties"]["name"] == {"type": "string"}
        assert properties["root"]["properties"]["next"] == {"$ref": "#/$defs/N"}

    def test_a_pointer_that_resolves_nowhere_is_left_where_it_is(self):
        """Not this layer's to complain about, and not its place to guess."""
        parameters = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Nope"}, "b": {"type": "integer"}}}

        properties = find_tool_properties([chat_tool("d", parameters)], "d")

        assert properties["a"] == {"$ref": "#/$defs/Nope"}
        assert extract_types_from_schema(properties["b"]) == ["integer"]


class TestUnresolvableReference:
    """A pointer to nothing is a bad schema, and should read as one.

    `check_schema` does not follow references, so these used to pass the screen
    and fail in the backend's compile instead -- a 500 for what is a 400.
    """

    def test_a_reachable_dangling_pointer_is_reported(self):
        schema = {"type": "object", "properties": {"n": {"$ref": "#/$defs/Nope"}}}

        assert get_schema_validation_error(schema) == "reference '#/$defs/Nope' does not resolve within the schema"
        assert not compiles(schema | {"required": ["n"]})

    def test_one_reached_through_another_pointer_is_reported(self):
        schema = {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/N"}},
            "$defs": {"N": {"type": "object", "properties": {"m": {"$ref": "#/$defs/Gone"}}}},
        }

        assert get_schema_validation_error(schema) == "reference '#/$defs/Gone' does not resolve within the schema"

    def test_one_inside_an_unreferenced_definition_is_not(self):
        """A definition nothing points at is never compiled, so a dangling
        pointer inside it costs a real request nothing."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "$defs": {"Dead": {"properties": {"x": {"$ref": "#/$defs/Nope"}}}},
        }

        assert get_schema_validation_error(schema) is None
        assert compiles(schema)

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(DEFS, id="defs"),
            pytest.param(DEFINITIONS, id="draft-07-definitions"),
            pytest.param(SELF_REF, id="root-self-ref"),
            pytest.param(PROPERTY_REF, id="property-pointer"),
            pytest.param({"$ref": "#/$defs/N", "$defs": {"N": {"$ref": "#/$defs/N"}}}, id="recursive"),
        ],
    )
    def test_a_schema_whose_pointers_resolve_passes(self, schema):
        assert get_schema_validation_error(schema) is None

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"properties": {"n": {"$ref": "http://example.com/s.json"}}}, id="external"),
            pytest.param({"properties": {"n": {"$ref": "#N"}}, "$defs": {"N": {"$anchor": "N"}}}, id="anchor"),
            pytest.param({"$id": "https://example.com/s", "properties": {"n": {"$ref": "#/$defs/Nope"}}}, id="id"),
        ],
    )
    def test_a_pointer_this_layer_cannot_place_is_left_to_the_backend(self, schema):
        assert get_schema_validation_error(schema) is None


def test_the_refusal_names_the_keyword_it_refused():
    """The decode-time layer gives up on an unfollowable reference. It used to
    say so in terms of `contains` whatever keyword had actually been refused."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/Nope"},
            "arr": {"type": "array", "items": {"enum": ["a", "b"]}, "uniqueItems": True},
        },
    }

    (reason,) = get_unenforceable_reasons(schema)

    assert "cannot follow" in reason
    # The prose says the schema "contains" a reference; the keyword named as
    # refused is the one in backticks, and it used to be the wrong one.
    assert "`contains`" not in reason
