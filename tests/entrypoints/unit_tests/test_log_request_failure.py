# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A request refused for being wrong must not be logged as a server fault.

Streaming endpoints catch the exception and report it as an error event, so
`logger.exception` reads as the natural thing to write -- and files every 4xx
at ERROR with a full stack. A caller repeating one bad schema then fills the
log with tracebacks of the server working correctly, and buries the 5xx that
matter.
"""

import logging
from http import HTTPStatus

import pytest

from aphrodite.entrypoints.utils import log_request_failure
from aphrodite.exceptions import APHRODITENotFoundError, APHRODITEValidationError

pytestmark = pytest.mark.cpu_test

CONTEXT = "Error in chat completion stream generator."


@pytest.mark.parametrize(
    "exc",
    [
        # What a bad tool schema arrives as: validation raises a plain
        # ValueError deep in sampling_params, which create_error_response
        # classifies as a 400.
        ValueError("JSON schema is not valid: [] should be non-empty at /properties/channel/anyOf"),
        TypeError("wrong type"),
        APHRODITEValidationError("bad parameter"),
        APHRODITENotFoundError("no such model"),
    ],
)
def test_client_error_logs_one_line_without_a_stack(caplog, exc):
    with caplog.at_level(logging.DEBUG):
        try:
            raise exc
        except Exception as e:
            log_request_failure(e, CONTEXT)

    records = [r for r in caplog.records if r.name.startswith("aphrodite")]
    assert [r.levelno for r in records] == [logging.WARNING]
    # exc_info is what carries the traceback; the whole point is its absence.
    assert records[0].exc_info is None
    # The caller's own message has to survive, or the line is not worth logging.
    assert str(exc).split(":")[-1].strip()[:20] in records[0].getMessage()


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("engine died"),
        KeyError("triggers"),
        AssertionError(),
    ],
)
def test_server_error_keeps_the_traceback(caplog, exc):
    with caplog.at_level(logging.DEBUG):
        try:
            raise exc
        except Exception as e:
            log_request_failure(e, CONTEXT)

    records = [r for r in caplog.records if r.name.startswith("aphrodite")]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].exc_info is not None


def test_level_follows_the_status_the_client_is_given():
    """The two must be decided once, not judged separately in two places.

    If the log level were its own opinion about which exceptions are the
    caller's fault, it could drift from the status code -- and the log would
    then disagree with the response about whose problem the request was.
    """
    from aphrodite.entrypoints.utils import create_error_response

    for exc, expected in [
        (ValueError("bad schema"), HTTPStatus.BAD_REQUEST),
        (NotImplementedError("later"), HTTPStatus.NOT_IMPLEMENTED),
        (RuntimeError("engine died"), HTTPStatus.INTERNAL_SERVER_ERROR),
    ]:
        assert create_error_response(exc).error.code == expected.value
