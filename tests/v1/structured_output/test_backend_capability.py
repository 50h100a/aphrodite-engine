# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Which backend enforces which JSON Schema keyword, and what we do about it.

A backend that *declines* a schema is the easy case: we hear about it and fall
back. The case that reached production is the other one -- xgrammar accepts a
schema containing `allOf`, lowers the whole thing to "any JSON value", warns
only on stderr from C++, and decodes with nothing enforced at all. The reply
came back as the bare integer `-99120` against a `type: object` root, with
`required` and `additionalProperties: false` both ignored, and no error
anywhere.

So capability cannot be taken on trust from a backend's own validation. The
table in `schema_features` records what each backend was *measured* to enforce,
and `test_recorded_capability_matches_the_backends` re-measures it here: for
each keyword, a schema and an instance violating only that keyword, fed to the
real backend. Enforcement means refusing the instance. Compiling it does not
count.

When this test fails, a backend changed. Update the table -- do not relax the
test, because the table is what keeps a schema off a backend that would
silently ignore it.
"""

import json
import re
from unittest.mock import Mock

import pytest

from aphrodite.v1.structured_output.schema_features import (
    JSON_SCHEMA_BACKENDS,
    backend_conflict_message,
    get_json_schema_backend_conflict,
    get_json_schema_backends,
    get_unenforceable_json_schema_keys,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Probes: schema, an instance violating only the keyword under test, and one
# that satisfies the whole schema.
#
# The legal instance is not decoration. Without it a backend that rejects
# everything -- or that refuses the violation for an unrelated reason -- reads
# as "enforced". `additionalProperties: True` appears wherever the keyword
# under test is not itself about which properties are allowed, for the same
# reason: otherwise `additionalProperties` does the rejecting and every backend
# looks like it enforces the keyword.
# ---------------------------------------------------------------------------

_OPEN = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": True,
}


def _open(**extra):
    return dict(_OPEN, **extra)


PROBES: dict[str, tuple[dict, object, object]] = {
    "allOf": (_open(allOf=[{"required": ["b"]}]), {"a": "z"}, {"a": "z", "b": 1}),
    "if": (
        _open(
            properties={"a": {"type": "string", "enum": ["x", "y"]}, "eta": {"type": "integer"}},
            **{"if": {"properties": {"a": {"const": "x"}}, "required": ["a"]}, "then": {"required": ["eta"]}},
        ),
        {"a": "x"},
        {"a": "x", "eta": 3},
    ),
    # The legal instance carries an extra property of its own, and has to. This
    # `not` forbids one particular key, so a backend that closes the object
    # refuses the violation without reading the keyword at all; only an
    # instance that uses the freedom `not` leaves open can tell the two apart.
    # `additionalProperties: True` does not settle it, because the backends
    # that close the object are the ones that ignore it.
    "not": (_open(**{"not": {"required": ["p"]}}), {"a": "z", "p": 1}, {"a": "z", "q": 1}),
    "unevaluatedProperties": (
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "unevaluatedProperties": False},
        {"a": "z", "x": 1},
        {"a": "z"},
    ),
    "unevaluatedItems": (
        {"type": "array", "prefixItems": [{"type": "string"}], "unevaluatedItems": False},
        ["a", "b"],
        ["a"],
    ),
    "dependentRequired": (_open(dependentRequired={"a": ["b"]}), {"a": "z"}, {"a": "z", "b": 1}),
    "dependentSchemas": (_open(dependentSchemas={"a": {"required": ["b"]}}), {"a": "z"}, {"a": "z", "b": 1}),
    "propertyNames": (_open(propertyNames={"pattern": "^[ab]"}), {"a": "z", "zz": 1}, {"a": "z", "b": 1}),
    "patternProperties": (
        {"type": "object", "patternProperties": {"^x": {"type": "string"}}, "additionalProperties": False},
        {"xa": 1},
        {"xa": "s"},
    ),
    "minProperties": (_open(minProperties=2), {"a": "z"}, {"a": "z", "b": 1}),
    "uniqueItems": ({"type": "array", "items": {"type": "string"}, "uniqueItems": True}, ["a", "a"], ["a", "b"]),
    "contains": ({"type": "array", "items": {"type": "integer"}, "contains": {"const": 7}}, [1, 2], [7, 2]),
    "minContains": (
        {"type": "array", "items": {"type": "integer"}, "contains": {"const": 7}, "minContains": 2},
        [7, 1],
        [7, 7],
    ),
    "multipleOf": ({"type": "number", "multipleOf": 0.25}, 0.3, 0.25),
    "exclusiveMinimum": ({"type": "number", "exclusiveMinimum": 0}, 0, 0.5),
    "exclusiveMaximum": ({"type": "number", "exclusiveMaximum": 10}, 10, 9.5),
    "minimum": ({"type": "integer", "minimum": 1}, 0, 5),
    "maximum": ({"type": "integer", "maximum": 72}, 99, 5),
    "prefixItems": (
        {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": False},
        ["a", "b"],
        ["a", 1],
    ),
}


def _xgrammar_accepts(schema, instance):
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    return _is_grammar_accept_string(xgr.Grammar.from_json_schema(json.dumps(schema)), json.dumps(instance))


class _ByteTokenizer:
    """One token per byte, so acceptance is decided by the grammar alone."""

    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
    special_token_ids = [256]

    def __call__(self, text):
        return list(text.encode() if isinstance(text, str) else text)


def _guidance_accepts(schema, instance):
    import llguidance

    grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
    if error := llguidance.LLMatcher.validate_grammar(grammar):
        # llguidance names what it did not implement instead of degrading.
        raise RuntimeError(error)
    tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_ByteTokenizer()))
    matcher = llguidance.LLMatcher(tokenizer, grammar)
    token_ids = list(json.dumps(instance).encode())
    if matcher.try_consume_tokens(token_ids) != len(token_ids):
        return False
    return matcher.is_accepting()


def _outlines_accepts(schema, instance):
    from outlines_core import json_schema

    pattern = json_schema.build_regex_from_schema(json.dumps(schema))
    return re.fullmatch(pattern, json.dumps(instance)) is not None


def _lm_format_enforcer_accepts(schema, instance):
    from lmformatenforcer import JsonSchemaParser

    parser = JsonSchemaParser(schema)
    for char in json.dumps(instance):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


ACCEPTS = {
    "xgrammar": _xgrammar_accepts,
    "guidance": _guidance_accepts,
    "outlines": _outlines_accepts,
    "lm-format-enforcer": _lm_format_enforcer_accepts,
}


def _enforces(backend, schema, violating, legal):
    """Whether ``backend`` refuses ``violating`` while still allowing ``legal``."""
    accepts = ACCEPTS[backend]
    try:
        if accepts(schema, violating):
            return False
    except Exception:
        # Declined the schema outright. Loud, so it is not our problem here;
        # `auto` hears it and moves on.
        return False
    try:
        # Refusing the legal instance too is over-restriction, not enforcement
        # of this keyword.
        return accepts(schema, legal)
    except Exception:
        return False


def test_probe_backends_are_the_recorded_backends():
    assert set(ACCEPTS) == set(JSON_SCHEMA_BACKENDS)


@pytest.mark.parametrize("keyword", sorted(PROBES))
def test_recorded_capability_matches_the_backends(keyword):
    """The table says who enforces this keyword; ask the backends directly."""
    schema, violating, legal = PROBES[keyword]
    measured = {backend for backend in ACCEPTS if _enforces(backend, schema, violating, legal)}

    recorded = get_json_schema_backends({**schema})
    # get_json_schema_backends answers for the whole schema, and the probe
    # schemas carry only the keyword under test plus universally-supported
    # scaffolding, so the two are comparable.
    assert measured == set(recorded), (
        f"{keyword}: table says {sorted(recorded)}, backends do {sorted(measured)}. "
        "A backend changed -- update _KEYWORD_BACKENDS in schema_features."
    )


def test_xgrammar_does_not_get_a_schema_it_would_silently_ignore():
    """The production failure. xgrammar compiles this and enforces nothing."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
        "allOf": [{"required": ["a"]}],
    }

    assert "xgrammar" not in get_json_schema_backends(schema)
    assert "guidance" in get_json_schema_backends(schema)
    # And it is not rejected -- guidance can enforce it, so the request runs.
    assert get_unenforceable_json_schema_keys(schema) == []


def test_xgrammar_really_does_ignore_it():
    """Guards the claim above against xgrammar quietly gaining `allOf`: if this
    starts failing, xgrammar enforces it now and belongs back in the table."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
        "allOf": [{"required": ["a"]}],
    }

    assert _xgrammar_accepts(schema, -99120), "xgrammar now constrains `allOf`; update _KEYWORD_BACKENDS"


@pytest.mark.parametrize(
    "keyword,schema",
    [
        ("uniqueItems", {"type": "array", "items": {"type": "string"}, "uniqueItems": True}),
        ("if", {"type": "object", "if": {"required": ["a"]}, "then": {"required": ["b"]}}),
        ("dependentRequired", {"type": "object", "dependentRequired": {"a": ["b"]}}),
        ("contains", {"type": "array", "contains": {"type": "string"}}),
    ],
)
def test_keywords_no_backend_enforces_are_rejected(keyword, schema):
    """A spec is a spec: if it cannot be enforced, say so rather than decode as
    though it were."""
    assert keyword in get_unenforceable_json_schema_keys(schema)
    assert get_json_schema_backends(schema) == frozenset()


@pytest.mark.parametrize(
    "schema",
    [
        # Inert values constrain nothing, so they cost neither a backend nor a
        # rejection.
        {"type": "array", "items": {"type": "string"}, "uniqueItems": False},
        {"type": "object", "properties": {"a": {"type": "string"}}, "if": {"required": ["a"]}},
        {"type": "object", "properties": {"a": {"type": "string"}}, "allOf": []},
        {"type": "object", "properties": {"a": {"type": "string"}}, "dependentRequired": {}},
        # A property *named* after a keyword is user data, not a constraint.
        {"type": "object", "properties": {"allOf": {"type": "string"}, "uniqueItems": {"type": "string"}}},
    ],
)
def test_inert_keywords_cost_nothing(schema):
    assert get_unenforceable_json_schema_keys(schema) == []
    assert get_json_schema_backends(schema) == JSON_SCHEMA_BACKENDS


def test_schema_split_across_two_backends_is_rejected_with_both_named():
    """`allOf` is guidance-only, `unevaluatedProperties` is everyone-but-guidance.
    Each is enforceable; together they have no home, and one request decodes
    with one backend. That is not visible from either keyword alone, so it gets
    its own rejection rather than falling through as enforced."""
    schema = {
        "type": "object",
        "allOf": [{"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}],
        "unevaluatedProperties": False,
    }

    assert get_unenforceable_json_schema_keys(schema) == []
    assert get_json_schema_backends(schema) == frozenset()

    conflict = get_json_schema_backend_conflict(schema)
    assert conflict == ["allOf", "unevaluatedProperties"]
    message = backend_conflict_message(conflict)
    assert "allOf needs ['guidance']" in message
    assert "unevaluatedProperties needs" in message


def test_backend_conflict_is_not_reported_for_a_schema_with_a_home():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "allOf": [{"required": ["a"]}]}
    assert get_json_schema_backend_conflict(schema) == []


def test_backend_conflict_defers_to_the_unenforceable_report():
    """When something is unenforceable outright, that is the useful message;
    reporting a backend split on top of it would be noise."""
    schema = {"type": "array", "uniqueItems": True, "allOf": [{"minItems": 1}]}
    assert get_unenforceable_json_schema_keys(schema) == ["uniqueItems"]
    assert get_json_schema_backend_conflict(schema) == []


# ---------------------------------------------------------------------------
# Routing: what the table is for.
# ---------------------------------------------------------------------------

ORDINARY = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}
NEEDS_GUIDANCE = {**ORDINARY, "allOf": [{"required": ["a"]}]}
# No keyword routes to outlines alone any more, so there is no NEEDS_OUTLINES.
# Everything outlines enforces, xgrammar enforces too, and `auto` reaches
# xgrammar first; outlines is left for the requests the earlier two decline for
# reasons that have nothing to do with keywords.
NEEDS_NOBODY = {
    "type": "object",
    "properties": {"u": {"type": "string"}},
    "required": ["u"],
    "additionalProperties": True,
    "not": {"required": ["p"]},
}


@pytest.fixture
def route(monkeypatch):
    """Resolve a schema to the backend `auto` would pick, without needing any
    of them to be installed: each validator is stubbed to accept."""
    import aphrodite.sampling_params as sampling_params
    from aphrodite.config import ModelConfig, StructuredOutputsConfig
    from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams
    from aphrodite.v1.structured_output import backend_guidance, backend_outlines, backend_xgrammar

    monkeypatch.setattr(backend_xgrammar, "validate_xgrammar_grammar", lambda params: None)
    monkeypatch.setattr(backend_guidance, "validate_guidance_grammar", lambda params, tokenizer=None: None)
    monkeypatch.setattr(
        backend_outlines,
        "validate_structured_output_request_outlines",
        lambda params: None,
    )
    monkeypatch.setattr(sampling_params, "_get_llg_tokenizer", lambda tokenizer: None)

    model_config = Mock(spec=ModelConfig)
    model_config.is_diffusion = False

    def _route(schema=None, backend="auto", structural_tag=None):
        params = SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema, structural_tag=structural_tag)
        )
        config = Mock(spec=StructuredOutputsConfig)
        config.backend = backend
        params._validate_structured_outputs(model_config, config, Mock())
        return params.structured_outputs._backend

    return _route


def test_an_ordinary_schema_still_goes_to_xgrammar(route):
    """The fix must not cost the fast path: xgrammar stays first choice for
    everything it genuinely enforces."""
    assert route(ORDINARY) == "xgrammar"


def test_allof_is_routed_past_xgrammar_to_guidance(route):
    """The production bug, end to end. xgrammar would have taken this request
    and decoded it with nothing enforced."""
    assert route(NEEDS_GUIDANCE) == "guidance"


def test_not_is_refused_because_nothing_enforces_it(route):
    """`not` used to be recorded as outlines' and lm-format-enforcer's.

    Neither reads it. Both drop the keyword and close the object instead, which
    happens to refuse a violation of a `not` over `required` -- and refuses the
    keys `not` permits along with it, so the request was both unenforced and
    walled. With the table corrected there is nowhere to send it.
    """
    with pytest.raises(ValueError, match="cannot be enforced by structured output"):
        route(NEEDS_NOBODY)


def test_a_pinned_backend_that_would_ignore_the_schema_is_refused(route):
    """`auto` can route around xgrammar; a server pinned to it cannot, so the
    request has to be refused instead of decoded as if the schema applied."""
    with pytest.raises(ValueError, match="does not enforce every keyword"):
        route(NEEDS_GUIDANCE, backend="xgrammar")


def test_a_pinned_backend_that_can_enforce_the_schema_is_used(route):
    assert route(NEEDS_GUIDANCE, backend="guidance") == "guidance"
    assert route(ORDINARY, backend="xgrammar") == "xgrammar"


def test_requests_with_no_json_schema_are_never_refused_for_capability(route):
    """A regex or a choice reaches no JSON Schema keyword, so the capability
    check must not have an opinion about it."""
    from aphrodite.config import ModelConfig, StructuredOutputsConfig
    from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams

    model_config = Mock(spec=ModelConfig)
    model_config.is_diffusion = False
    params = SamplingParams(structured_outputs=StructuredOutputsParams(regex=r"[abc]+"))
    config = Mock(spec=StructuredOutputsConfig)
    config.backend = "xgrammar"

    params._validate_structured_outputs(model_config, config, Mock())

    assert params.structured_outputs._backend == "xgrammar"


# ---------------------------------------------------------------------------
# Structural tags: the same routing question, asked about the wrapper rather
# than the schema inside it.
#
# A tool call reaches the grammar wrapped in a structural tag, and a tag is not
# a JSON schema -- so the keyword table alone does not say where a tool request
# can go. Only xgrammar compiles the nested `format` spelling; guidance adds the
# older `structures`/`triggers` one; outlines and lm-format-enforcer compile
# neither, and do not say so when asked to validate one. Routing on keywords
# alone therefore sent tool calls to backends that accepted them in silence and
# then raised from the engine's grammar thread, which reaches the caller as a
# 500 rather than a rejection.
# ---------------------------------------------------------------------------

NESTED_TAG = json.dumps(
    {
        "type": "structural_tag",
        "format": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "begin": "<|channel|>commentary to=functions.f<|message|>",
                    "content": {"type": "json_schema", "json_schema": ORDINARY},
                    "end": "<|call|>",
                }
            ],
            "separator": "<|start|>assistant",
            "at_least_one": False,
            "stop_after_first": False,
        },
    }
)


def _tag_around(schema):
    tag = json.loads(NESTED_TAG)
    tag["format"]["tags"][0]["content"]["json_schema"] = schema
    return json.dumps(tag)


def test_a_nested_structural_tag_routes_to_xgrammar(route):
    assert route(structural_tag=NESTED_TAG) == "xgrammar"


def test_a_structures_form_tag_may_also_go_to_guidance(route):
    """The older spelling has two homes, so a schema xgrammar would ignore can
    still be routed rather than refused."""
    tag = json.dumps(
        {
            "triggers": ["<function="],
            "structures": [{"begin": "<function=f>", "schema": NEEDS_GUIDANCE, "end": "</function>"}],
        }
    )
    assert route(structural_tag=tag) == "guidance"


@pytest.mark.parametrize(
    "schema",
    [
        NEEDS_GUIDANCE,
        {**ORDINARY, "properties": {"n": {"type": "number", "multipleOf": 0.25}}, "required": ["n"]},
    ],
    ids=["allOf", "multipleOf"],
)
def test_a_tag_is_never_routed_to_a_backend_that_cannot_compile_it(route, schema):
    """A tool schema only guidance enforces, inside a tag only xgrammar reads.

    Each half has a home and they are not the same home, and a request decodes
    with one backend, so this is refused. Routing on the schema alone looked at
    the keywords, sent it to a backend that cannot read a structural tag at
    all, and the request died on the engine's grammar thread as a 500.
    """
    with pytest.raises(ValueError, match="cannot be enforced in a tool call"):
        route(structural_tag=_tag_around(schema))


def test_a_pinned_backend_that_cannot_read_the_tag_is_refused(route):
    with pytest.raises(ValueError, match="cannot compile the structural tag"):
        route(structural_tag=NESTED_TAG, backend="outlines")


def test_the_tag_check_leaves_ordinary_requests_alone(route):
    """No tag means no opinion: the plain JSON route must route as before."""
    assert route(ORDINARY) == "xgrammar"
    assert route(NEEDS_GUIDANCE) == "guidance"


@pytest.mark.parametrize("validate", ["outlines", "lm-format-enforcer"])
def test_tagless_backends_decline_a_tag_instead_of_accepting_it(validate):
    """The silence is the bug: both used to return None here and raise later,
    on the engine's grammar thread, where it is a 500."""
    from aphrodite.sampling_params import SamplingParams, StructuredOutputsParams

    if validate == "outlines":
        from aphrodite.v1.structured_output.backend_outlines import (
            validate_structured_output_request_outlines as fn,
        )
    else:
        from aphrodite.v1.structured_output.backend_lm_format_enforcer import (
            validate_structured_output_request_lm_format_enforcer as fn,
        )

    params = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=NESTED_TAG))
    with pytest.raises(ValueError, match="does not support structural tags"):
        fn(params)
