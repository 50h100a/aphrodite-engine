# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real API routes must reject generated tool grammars before sending SSE."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from aphrodite.config import StructuredOutputsConfig
from aphrodite.entrypoints.openai.chat_completion.api_router import router as chat_router
from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from aphrodite.entrypoints.openai.responses.api_router import router as responses_router
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.entrypoints.openai.responses.serving import OpenAIServingResponses
from aphrodite.entrypoints.serve.exception_handling.register import init_exception_handler
from aphrodite.inputs import tokens_input
from aphrodite.parser import ParserManager
from tests.parser.engine.conftest import make_mock_tokenizer

pytestmark = pytest.mark.cpu_test


def serving_app(route, parser_name="deepseek_v41"):
    cls = OpenAIServingChat if route == "chat" else OpenAIServingResponses
    serving = cls.__new__(cls)
    tokenizer = make_mock_tokenizer({})
    serving.model_config = SimpleNamespace(max_model_len=4096, is_diffusion=False, is_encoder_decoder=False)
    serving.renderer = SimpleNamespace(tokenizer=tokenizer, get_tokenizer=lambda: tokenizer)
    serving.engine_client = SimpleNamespace(
        errored=False,
        check_admission=Mock(),
        aphrodite_config=SimpleNamespace(structured_outputs_config=StructuredOutputsConfig()),
        generate=Mock(side_effect=AssertionError("invalid schema reached generation")),
    )
    serving.models = SimpleNamespace(model_name=lambda adapter: "offline")
    serving.has_kv_connector = False
    serving._check_model = AsyncMock(return_value=None)
    serving._maybe_get_adapters = Mock(return_value=None)
    serving._effective_chat_template_kwargs = Mock(return_value={"enable_thinking": False})
    serving._get_trace_headers = AsyncMock(return_value=None)
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.request_logger = None
    serving.parser_cls = None
    serving.parser = None
    serving.use_harmony = False
    serving.enable_store = False
    serving.tool_server = None
    serving.online_renderer = SimpleNamespace(
        validate_chat_template=Mock(return_value=None), trust_request_chat_template=True
    )
    parser_cls = ParserManager.get_parser(tool_parser_name=parser_name, enable_auto_tools=True)

    async def render(request, *args, **kwargs):
        # Exercise actual native tool-grammar injection after request parsing.
        parser_cls(tokenizer, request.tools, chat_template_kwargs={"enable_thinking": False}).adjust_request(request)
        prompt = tokens_input([1])
        return ([], [prompt]) if route == "chat" else SimpleNamespace(messages=[], engine_input=prompt)

    serving.render_chat_request = render
    serving._render_resolved_response_inputs = render
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False, sse_keep_alive_interval=0)
    app.state.enable_server_load_tracking = False
    app.state.openai_serving_chat = serving
    app.state.openai_serving_responses = serving
    app.include_router(chat_router if route == "chat" else responses_router)
    init_exception_handler(app)
    return app, serving, render


def request_body(route, prop, choice, stream, strict=True):
    parameters = {"type": "object", "properties": {"value": prop}, "required": ["value"]}
    function = {"name": "record_value", "parameters": parameters, "strict": strict}
    tool_choice = (
        {"type": "function", "function": {"name": "record_value"}}
        if choice == "named" and route == "chat"
        else {"type": "function", "name": "record_value"}
        if choice == "named"
        else choice
    )
    common = {"model": "offline", "stream": stream, "tool_choice": tool_choice}
    if route == "chat":
        return common | {
            "messages": [{"role": "user", "content": "Call record_value"}],
            "tools": [{"type": "function", "function": function}],
        }
    return common | {"input": "Call record_value", "store": False, "tools": [{"type": "function", **function}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("parser_name", ["deepseek_v41", "qwen3_xml"])
@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize(
    "prop",
    [
        {"type": "string", "minLength": "five"},
        {"type": "string", "enum": []},
        {"type": "string", "minLength": 5, "maxLength": 2},
        {"$ref": "#/$defs/missing"},
        {"type": "integer", "enum": [1, 2], "minimum": 2},
        {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 10}]},
        {"type": "string", "pattern": "^[a-z]+$", "minLength": 3},
    ],
)
async def test_invalid_or_unsupported_generated_grammar_returns_http_400(parser_name, route, stream, choice, prop):
    app, serving, _ = serving_app(route, parser_name)
    path = "/v1/chat/completions" if route == "chat" else "/v1/responses"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(path, json=request_body(route, prop, choice, stream))
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == 400
    serving.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("parser_name", ["deepseek_v41", "qwen3_xml"])
@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("strict", [True, False])
async def test_valid_constraints_and_explicit_opt_out_pass_preflight(parser_name, route, strict):
    _, serving, render = serving_app(route, parser_name)
    prop = {"type": "string", "enum": [" stable "]} if strict else {"type": "string", "minLength": "five"}
    body = request_body(route, prop, "required", True, strict=strict)
    request = ChatCompletionRequest(**body) if route == "chat" else ResponsesRequest(**body)
    await render(request)
    params = request.to_sampling_params(128, {})
    serving._validate_structured_outputs(params)
    assert params.structured_outputs._backend == "xgrammar"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": {"const": 2}}, "required": ["value"], "additionalProperties": False},
        {
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"a": {"const": "A"}},
                    "required": ["a"],
                    "additionalProperties": False,
                },
                {"type": "object", "properties": {"b": {"const": 2}}, "required": ["b"], "additionalProperties": False},
            ],
        },
        {
            "$ref": "#/$defs/root",
            "$defs": {
                "root": {
                    "type": "object",
                    "properties": {"value": {"const": "\r\n"}},
                    "required": ["value"],
                    "additionalProperties": False,
                }
            },
        },
        {"type": "object", "additionalProperties": {"type": "string", "minLength": 3, "maxLength": 5}},
    ],
)
async def test_qwen_composed_constraints_pass_real_api_preflight(route, stream, choice, schema):
    from copy import deepcopy

    _, serving, render = serving_app(route, "qwen3_xml")
    original = deepcopy(schema)
    body = request_body(route, {"type": "string"}, choice, stream)
    function = body["tools"][0]["function"] if route == "chat" else body["tools"][0]
    function["parameters"] = schema
    request = ChatCompletionRequest(**body) if route == "chat" else ResponsesRequest(**body)
    await render(request)
    params = request.to_sampling_params(128, {})
    serving._validate_structured_outputs(params)
    assert params.structured_outputs._backend == "xgrammar"
    assert schema == original
