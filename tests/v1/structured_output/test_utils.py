# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from aphrodite.v1.structured_output.backend_guidance import (
    has_guidance_unsupported_json_features,
)
from aphrodite.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)
from aphrodite.v1.structured_output.schema_features import (
    get_unenforceable_json_schema_keys,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {"type": "integer", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {"type": "number", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_object_schemas():
    return [
        {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
        {"type": "object", "patternProperties": {"^S": {"type": "string"}}},
    ]


@pytest.fixture
def supported_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
            "car_type": {"type": "string", "enum": ["sedan", "suv", "truck"]},
            "car_brand": {"type": "string", "pattern": "^[a-zA-Z]+$"},
            "short_description": {"type": "string", "maxLength": 50},
            "mileage": {"type": "number", "minimum": 0, "maximum": 1000000},
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100,
            },
            "long_description": {"type": "string", "minLength": 50, "maxLength": 2000},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
        "minProperties": 1,
        "maxProperties": 100,
    }


@pytest.mark.parametrize(
    "schema_type",
    [
        "unsupported_string_schemas",
        "unsupported_integer_schemas",
        "unsupported_number_schemas",
        "unsupported_array_schemas",
        "unsupported_object_schemas",
    ],
)
def test_unsupported_json_features_by_type(schema_type, request):
    schemas = request.getfixturevalue(schema_type)
    for schema in schemas:
        assert has_xgrammar_unsupported_json_features(schema), f"Schema should be unsupported: {schema}"


def test_supported_json_features(supported_schema):
    assert not has_xgrammar_unsupported_json_features(supported_schema), "Schema should be supported"


@pytest.mark.parametrize(
    "schema,expected",
    [
        # Rejected whatever the nesting, and with no declared "type" -- the
        # keyword alone is what the backends choke on.
        ({"type": "array", "items": {"type": "string"}, "uniqueItems": True}, ["uniqueItems"]),
        ({"items": {"type": "string"}, "uniqueItems": True}, ["uniqueItems"]),
        ({"type": "object", "properties": {"t": {"type": "array", "uniqueItems": True}}}, ["uniqueItems"]),
        ({"$defs": {"A": {"type": "array", "uniqueItems": True}}, "$ref": "#/$defs/A"}, ["uniqueItems"]),
        ({"anyOf": [{"type": "array", "uniqueItems": True}, {"type": "string"}]}, ["uniqueItems"]),
        ({"type": "array", "items": [{"uniqueItems": True}]}, ["uniqueItems"]),
        ({"type": "array", "contains": {"type": "string"}, "minContains": 1}, ["contains", "minContains"]),
        # A property *named* after a keyword is user data, not a constraint.
        ({"type": "object", "properties": {"uniqueItems": {"type": "string"}}}, []),
        ({"type": "string", "enum": ["uniqueItems", "contains"]}, []),
        ({"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 3}}}, []),
    ],
)
def test_unenforceable_json_schema_keys(schema, expected):
    assert get_unenforceable_json_schema_keys(schema) == expected


def test_guidance_unsupported_json_features():
    # llguidance reports these as "Unimplemented keys" (verified against 1.7.6).
    assert has_guidance_unsupported_json_features({"type": "array", "uniqueItems": True})
    assert has_guidance_unsupported_json_features({"type": "object", "propertyNames": {"pattern": "^a"}})
    assert has_guidance_unsupported_json_features({"not": {"type": "string"}})

    # Supported by llguidance -- must not be diverted away from guidance.
    assert not has_guidance_unsupported_json_features(
        {"type": "object", "patternProperties": {"^S": {"type": "string"}}}
    )
    assert not has_guidance_unsupported_json_features({"type": "integer", "multipleOf": 120})


def test_guidance_unsupported_json_features_supported_schema(supported_schema):
    assert not has_guidance_unsupported_json_features(supported_schema)
