# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from aphrodite.entrypoints.generate.base.protocol import RequestResponseMetadata
from aphrodite.entrypoints.openai.completion.protocol import CompletionRequest
from aphrodite.entrypoints.openai.completion.serving import OpenAIServingCompletion
from aphrodite.logprobs import Logprob
from aphrodite.outputs import CompletionOutput, RequestOutput

TOKENS = [(1, "abc"), (2, " de"), (3, "!")]
# Text deltas as the detokenizer streams them with stop=["\n\n"]: one char is
# held back until the request finishes, so the text lags the tokens.
TEXT_DELTAS = ["ab", "c d", "e!"]


def _logprobs(i: int) -> dict[int, Logprob]:
    token_id, token = TOKENS[i]
    return {token_id: Logprob(logprob=-1.0, rank=1, decoded_token=token)}


def _output(text: str, idxs: list[int], finished: bool) -> RequestOutput:
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=[9],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[TOKENS[i][0] for i in idxs],
                cumulative_logprob=None,
                logprobs=[_logprobs(i) for i in idxs],
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
    )


def _serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    return serving


def _request(stream: bool) -> CompletionRequest:
    return CompletionRequest(model="m", prompt="Hi", max_tokens=8, logprobs=0, stop=["\n\n"], stream=stream)


@pytest.mark.asyncio
async def test_disconnected_completion_stream_closes_engine_results():
    closed = False

    async def results():
        nonlocal closed
        try:
            yield 0, _output("ab", [0], finished=False)
            yield 0, _output("c", [], finished=True)
        finally:
            closed = True

    stream = _serving().completion_stream_generator(
        _request(stream=True),
        [{"prompt": "Hi"}],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    )
    await anext(stream)
    await stream.aclose()
    assert closed


@pytest.mark.asyncio
async def test_stream_text_offset_matches_non_stream_with_stop_holdback():
    async def results():
        for i, text in enumerate(TEXT_DELTAS):
            yield 0, _output(text, [i], finished=i == len(TEXT_DELTAS) - 1)

    serving = _serving()
    stream_offsets = []
    async for line in serving.completion_stream_generator(
        _request(stream=True),
        [{"prompt": "Hi"}],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        for choice in json.loads(data)["choices"]:
            stream_offsets += choice["logprobs"]["text_offset"]

    response = serving.request_output_to_completion_response(
        [_output("".join(TEXT_DELTAS), [0, 1, 2], finished=True)],
        _request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )
    assert response.choices[0].logprobs is not None
    assert response.choices[0].logprobs.text_offset == [0, 3, 6]
    assert stream_offsets == [0, 3, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [True, False, None])
@pytest.mark.parametrize("include_usage,force_usage", [(False, False), (True, False), (False, True)])
async def test_compact_completion_stream(compact, include_usage, force_usage):
    serving = _serving()
    serving.enable_force_include_usage = force_usage
    serving.system_fingerprint = "fingerprint"
    request = _request(stream=True)
    from aphrodite.entrypoints.generate.base.protocol import StreamOptions

    request.stream_options = StreamOptions(
        compact_stream=compact,
        include_usage=include_usage,
        continuous_usage_stats=True,
    )

    async def results():
        for i, text in enumerate(TEXT_DELTAS):
            res = _output(text, [i], finished=i == 2)
            res.num_cached_tokens = 1
            yield 0, res

    metadata = RequestResponseMetadata(request_id="r")
    lines = [
        line
        async for line in serving.completion_stream_generator(
            request,
            [{"prompt": "Hi"}],
            results(),
            "r",
            0,
            "m",
            1,
            None,
            metadata,
        )
    ]
    assert lines[-1] == "data: [DONE]\n\n"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert not any("error" in event for event in events)
    if compact:
        assert events[0] == {"usage": {"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": 1}}}
        assert all(set(event) <= {"choices", "usage"} for event in events)
        assert [event["usage"] for event in events[1:]] == [{"completion_tokens": i} for i in [1, 2, 3]]
    else:
        assert all(event["id"] == "r" and event["model"] == "m" for event in events)
        assert bool(not events[-1]["choices"]) == (include_usage or force_usage)
    choices = [choice for event in events for choice in event.get("choices", [])]
    assert "".join(choice["text"] for choice in choices) == "abc de!"
    assert [offset for choice in choices for offset in choice["logprobs"]["text_offset"]] == [0, 3, 6]
    assert choices[-1]["finish_reason"] == "stop"
    assert metadata.final_usage_info.completion_tokens == 3


@pytest.mark.asyncio
async def test_compact_completion_batched_choices():
    request = CompletionRequest(
        model="m",
        prompt=["first", "second"],
        n=2,
        stream=True,
        stream_options={"compact_stream": True},
    )

    async def results():
        for prompt_idx in [1, 0]:
            res = _output("abc", [0], finished=True)
            res.prompt_token_ids = [9] * (prompt_idx + 1)
            res.outputs.append(
                CompletionOutput(
                    index=1,
                    text="de",
                    token_ids=[2, 3],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            )
            yield prompt_idx, res

    metadata = RequestResponseMetadata(request_id="r")
    lines = [
        line
        async for line in _serving().completion_stream_generator(
            request,
            [{"prompt": "first"}, {"prompt": "second"}],
            results(),
            "r",
            0,
            "m",
            2,
            None,
            metadata,
        )
    ]
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert events[0] == {"usage": {"prompt_tokens": 2}}
    assert [event["choices"][0]["index"] for event in events[1:]] == [2, 3, 0, 1]
    assert [event["usage"]["completion_tokens"] for event in events[1:]] == [1, 2, 1, 2]
    assert metadata.final_usage_info.prompt_tokens == 3
    assert metadata.final_usage_info.completion_tokens == 6
