# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Retry classification and deadline enforcement for media fetches."""

import errno
import socket
import time
from unittest.mock import MagicMock

import aiohttp
import pytest
import requests

import aphrodite.envs as envs
from aphrodite.connections import (
    HTTPConnection,
    _attempt_timeout,
    _is_permanent_connect_failure,
    _is_retryable,
)


def _connector_error(os_error):
    return aiohttp.ClientConnectorError(connection_key=None, os_error=os_error)


class TestRetryClassification:
    """Deterministic failures must not be retried -- retrying only burns
    the caller's deadline and produces log noise."""

    @pytest.mark.parametrize(
        "exc",
        [
            _connector_error(ConnectionRefusedError(errno.ECONNREFUSED, "refused")),
            _connector_error(OSError(errno.EHOSTUNREACH, "no route to host")),
            _connector_error(OSError(errno.ENETUNREACH, "network unreachable")),
            _connector_error(OSError(errno.EADDRNOTAVAIL, "address not available")),
            socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
            ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        ],
        ids=["refused", "hostunreach", "netunreach", "addrnotavail", "gaierror", "bare-refused"],
    )
    def test_permanent_failures_not_retried(self, exc):
        assert _is_permanent_connect_failure(exc) is True
        assert _is_retryable(exc) is False

    def test_requests_connection_error_wrapping_refused(self):
        exc = requests.exceptions.ConnectionError("failed")
        exc.__cause__ = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        assert _is_retryable(exc) is False

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError(),
            requests.exceptions.Timeout("slow"),
            _connector_error(ConnectionResetError(errno.ECONNRESET, "reset")),
            aiohttp.ServerDisconnectedError(),
        ],
        ids=["timeout", "requests-timeout", "reset", "disconnect"],
    )
    def test_transient_failures_still_retried(self, exc):
        assert _is_retryable(exc) is True

    def test_server_5xx_retried_client_4xx_not(self):
        def response_error(status):
            return aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=status, message="x")

        assert _is_retryable(response_error(503)) is True
        assert _is_retryable(response_error(404)) is False


class TestAttemptTimeout:
    def test_does_not_escalate(self):
        """Regression: the timeout used to be multiplied by 4 each attempt,
        so one slow video URL could hold a request for ~10 minutes."""
        assert _attempt_timeout(30, 45) == 30
        assert _attempt_timeout(30, 40) == 30
        assert _attempt_timeout(30, 20) == 20

    def test_clamps_to_remaining_budget(self):
        assert _attempt_timeout(30, 5) == 5
        assert _attempt_timeout(30, 0) == 0
        assert _attempt_timeout(30, -3) == 0

    def test_no_deadline_keeps_base_timeout(self):
        assert _attempt_timeout(30, None) == 30
        assert _attempt_timeout(None, None) is None


class TestDeadlineEnforcement:
    @pytest.fixture
    def hanging_server(self):
        """A socket that accepts connections but never replies."""
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(16)
        held = []

        import threading

        def accept_forever():
            while True:
                try:
                    conn, _ = srv.accept()
                    held.append(conn)
                except OSError:
                    return

        threading.Thread(target=accept_forever, daemon=True).start()
        yield srv.getsockname()[1]
        srv.close()
        for conn in held:
            conn.close()

    @pytest.mark.asyncio
    async def test_async_fetch_respects_deadline(self, hanging_server):
        conn = HTTPConnection(reuse_client=False)
        url = f"http://127.0.0.1:{hanging_server}/slow"

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await conn.async_get_bytes(url, timeout=1, deadline=3)
        elapsed = time.monotonic() - started

        assert elapsed < 3 + 2, f"exceeded deadline: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_refused_fails_immediately(self):
        """ECONNREFUSED should not consume the retry ladder at all."""
        conn = HTTPConnection(reuse_client=False)

        started = time.monotonic()
        with pytest.raises(aiohttp.ClientConnectorError):
            await conn.async_get_bytes("http://127.0.0.1:1/missing.png", timeout=5, deadline=10)
        elapsed = time.monotonic() - started

        # Previously: 3 attempts with 1s + 4s sleeps between them.
        assert elapsed < 1.0, f"retried a refused connection: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_no_deadline_still_works(self):
        conn = HTTPConnection(reuse_client=False)
        with pytest.raises(aiohttp.ClientConnectorError):
            await conn.async_get_bytes("http://127.0.0.1:1/missing.png", timeout=1)


class TestRetryLogging:
    @pytest.mark.asyncio
    async def test_single_warning_per_failed_fetch(self, caplog):
        """A caller-supplied bad URL gets one WARN, not one per attempt."""
        conn = HTTPConnection(reuse_client=False)

        with (
            caplog.at_level("WARNING", logger="aphrodite.connections"),
            pytest.raises(aiohttp.ClientConnectorError),
        ):
            await conn.async_get_bytes("http://127.0.0.1:1/x.png", timeout=1, deadline=5)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "http://127.0.0.1:1/x.png" in warnings[0].getMessage()


def test_media_fetch_env_defaults():
    """The deadlines that bound a single fetch."""
    assert envs.APHRODITE_IMAGE_FETCH_DEADLINE >= envs.APHRODITE_IMAGE_FETCH_TIMEOUT
    assert envs.APHRODITE_VIDEO_FETCH_DEADLINE >= envs.APHRODITE_VIDEO_FETCH_TIMEOUT
    assert envs.APHRODITE_AUDIO_FETCH_DEADLINE >= envs.APHRODITE_AUDIO_FETCH_TIMEOUT
    assert envs.APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS is False
