# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import errno
import functools
import socket
import time
from collections.abc import Callable, Coroutine, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import aiohttp
import requests
from urllib3.util import parse_url

import aphrodite.envs as envs
from aphrodite.logger import init_logger
from aphrodite.version import __version__ as APHRODITE_VERSION

logger = init_logger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Multiplier applied to the sleep between retry attempts: attempt N sleeps
# _RETRY_BACKOFF_FACTOR ** N seconds. The per-attempt timeout does NOT grow --
# it is bounded by whatever remains of the caller's total deadline.
_RETRY_BACKOFF_FACTOR = 4

# Don't start another attempt if less than this much budget remains; the
# attempt would time out almost immediately and only add noise.
_MIN_ATTEMPT_BUDGET = 0.5

# OS-level errors that mean "this address will not answer" rather than
# "something transient went wrong".
_FATAL_CONNECT_ERRNOS = frozenset(
    {
        errno.ECONNREFUSED,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.EADDRNOTAVAIL,
    }
)


def _is_permanent_connect_failure(exc: BaseException) -> bool:
    """Return True for connection failures that retrying cannot fix.

    A refused connection, an unroutable host or an unresolvable name will
    fail identically on the next attempt, so retrying only burns the
    caller's deadline.
    """
    # Name resolution failed.
    if isinstance(exc, socket.gaierror):
        return True
    if isinstance(exc, aiohttp.ClientConnectorDNSError):
        return True

    if isinstance(exc, aiohttp.ClientConnectorError):
        os_error = getattr(exc, "os_error", None)
        if isinstance(os_error, ConnectionRefusedError):
            return True
        for candidate in (os_error, exc):
            candidate_errno = getattr(candidate, "errno", None)
            if candidate_errno in _FATAL_CONNECT_ERRNOS:
                return True
        return False

    if isinstance(exc, ConnectionRefusedError):
        return True

    # requests wraps the underlying urllib3/socket error; walk the cause chain.
    if isinstance(exc, requests.exceptions.ConnectionError):
        cause: BaseException | None = exc
        seen = 0
        while cause is not None and seen < 10:
            if isinstance(cause, (socket.gaierror, ConnectionRefusedError)):
                return True
            if getattr(cause, "errno", None) in _FATAL_CONNECT_ERRNOS:
                return True
            cause = cause.__cause__ or cause.__context__
            seen += 1

    return False


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that are worth retrying.

    Retryable:
      - Timeouts (aiohttp, requests, stdlib)
      - Transient connection-level failures (reset, server disconnect)
      - Server errors (5xx) -- includes S3 503 SlowDown
    Not retryable:
      - Client errors (4xx) -- bad URL, auth, not-found
      - Permanent connection failures -- refused, unroutable, DNS NXDOMAIN
      - Programming errors (ValueError, TypeError, ...)
    """
    # Deterministic connection failures: retrying cannot change the outcome.
    if _is_permanent_connect_failure(exc):
        return False
    # Timeouts
    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            requests.exceptions.Timeout,
            aiohttp.ServerTimeoutError,
        ),
    ):
        return True
    # Connection-level failures
    if isinstance(
        exc,
        (
            ConnectionError,
            aiohttp.ClientConnectionError,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    # aiohttp server-side disconnects
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return True
    # requests 5xx -- raise_for_status() throws HTTPError
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None and exc.response.status_code >= 500:
        return True
    # aiohttp 5xx -- raise_for_status() throws ClientResponseError
    return isinstance(exc, aiohttp.ClientResponseError) and exc.status >= 500


def _attempt_timeout(base_timeout: float | None, remaining: float | None) -> float | None:
    """Per-attempt timeout: the base timeout, clamped to the budget left.

    Unlike the previous behaviour, this never grows across attempts -- a
    host that was too slow at N seconds does not get 4N on the next try.
    """
    if remaining is None:
        return base_timeout
    if base_timeout is None:
        return max(remaining, 0.0)
    return max(min(base_timeout, remaining), 0.0)


def _request_url(args: tuple, kwargs: dict) -> Any:
    # args[0] is `self` (bound method), args[1] is the URL
    return args[1] if len(args) > 1 else kwargs.get("url")


def _log_retry(
    args: tuple,
    kwargs: dict,
    attempt: int,
    max_retries: int,
    attempt_timeout: float | None,
    exc: Exception,
    backoff: float,
    remaining: float | None,
) -> None:
    """Per-attempt retry detail.

    Logged at DEBUG: a caller-supplied bad URL should not put one WARN line
    per attempt in the server log. The single WARN comes from
    :func:`_log_give_up` once the fetch has actually failed.
    """
    timeout_info = f"timeout={attempt_timeout:.3f}s" if attempt_timeout is not None else "no timeout"
    budget_info = f", {remaining:.3f}s budget left" if remaining is not None else ""
    logger.debug(
        "HTTP fetch failed for %s (attempt %d/%d, %s): %s -- retrying in %.3fs%s",
        _request_url(args, kwargs),
        attempt + 1,
        max_retries,
        timeout_info,
        exc,
        backoff,
        budget_info,
    )


def _log_give_up(
    args: tuple,
    kwargs: dict,
    attempts: int,
    exc: Exception,
    elapsed: float,
) -> None:
    """One WARN line per failed fetch, with the detail an operator needs."""
    logger.warning(
        "HTTP fetch failed for %s after %d attempt(s) in %.3fs: %s: %s",
        _request_url(args, kwargs),
        attempts,
        elapsed,
        type(exc).__name__,
        exc,
    )


def _sync_retry(
    fn: Callable[_P, _T],
) -> Callable[_P, _T]:
    """Add bounded retry logic to a sync method.

    The decorated method must accept ``timeout`` as a keyword argument, and
    may be given a ``deadline`` keyword: a total wall-clock budget covering
    every attempt and the sleeps between them. The per-attempt timeout never
    exceeds ``timeout`` or whatever remains of that budget, whichever is
    smaller, so a slow or hanging host cannot stall a request indefinitely.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        deadline: float | None = kwargs.pop("deadline", None)
        max_retries = max(envs.APHRODITE_MEDIA_FETCH_MAX_RETRIES, 1)
        started = time.monotonic()

        def remaining() -> float | None:
            if deadline is None:
                return None
            return deadline - (time.monotonic() - started)

        for attempt in range(max_retries):
            kwargs["timeout"] = _attempt_timeout(base_timeout, remaining())
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                attempts = attempt + 1
                left = remaining()
                if (
                    not _is_retryable(e)
                    or attempts >= max_retries
                    or (left is not None and left <= _MIN_ATTEMPT_BUDGET)
                ):
                    _log_give_up(args, kwargs, attempts, e, time.monotonic() - started)
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                if left is not None:
                    backoff = min(backoff, max(0.0, left - _MIN_ATTEMPT_BUDGET))
                _log_retry(args, kwargs, attempt, max_retries, kwargs["timeout"], e, backoff, left)
                time.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


def _async_retry(
    fn: Callable[_P, Coroutine[Any, Any, _T]],
) -> Callable[_P, Coroutine[Any, Any, _T]]:
    """Add bounded retry logic to an async method.

    The decorated method must accept ``timeout`` as a keyword argument, and
    may be given a ``deadline`` keyword: a total wall-clock budget covering
    every attempt and the sleeps between them. The per-attempt timeout never
    exceeds ``timeout`` or whatever remains of that budget, whichever is
    smaller, so a slow or hanging host cannot stall a request indefinitely.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        deadline: float | None = kwargs.pop("deadline", None)
        max_retries = max(envs.APHRODITE_MEDIA_FETCH_MAX_RETRIES, 1)
        started = time.monotonic()

        def remaining() -> float | None:
            if deadline is None:
                return None
            return deadline - (time.monotonic() - started)

        for attempt in range(max_retries):
            kwargs["timeout"] = _attempt_timeout(base_timeout, remaining())
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                attempts = attempt + 1
                left = remaining()
                if (
                    not _is_retryable(e)
                    or attempts >= max_retries
                    or (left is not None and left <= _MIN_ATTEMPT_BUDGET)
                ):
                    _log_give_up(args, kwargs, attempts, e, time.monotonic() - started)
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                if left is not None:
                    backoff = min(backoff, max(0.0, left - _MIN_ATTEMPT_BUDGET))
                _log_retry(args, kwargs, attempt, max_retries, kwargs["timeout"], e, backoff, left)
                await asyncio.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


class HTTPConnection:
    """Helper class to send HTTP requests."""

    def __init__(self, *, reuse_client: bool = True) -> None:
        super().__init__()

        self.reuse_client = reuse_client

        self._sync_client: requests.Session | None = None
        self._async_client: aiohttp.ClientSession | None = None

    def get_sync_client(self) -> requests.Session:
        if self._sync_client is None or not self.reuse_client:
            self._sync_client = requests.Session()

        return self._sync_client

    # NOTE: We intentionally use an async function even though it is not
    # required, so that the client is only accessible inside async event loop
    async def get_async_client(self) -> aiohttp.ClientSession:
        if self._async_client is None or not self.reuse_client:
            self._async_client = aiohttp.ClientSession(trust_env=True)

        return self._async_client

    def _validate_http_url(self, url: str):
        parsed_url = parse_url(url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("Invalid HTTP URL: A valid HTTP URL must have scheme 'http' or 'https'.")

    def _headers(self, **extras: str) -> MutableMapping[str, str]:
        return {"User-Agent": f"Aphrodite/{APHRODITE_VERSION}", **extras}

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = self.get_sync_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            stream=stream,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = await self.get_async_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    # NOTE: `deadline` on the two methods below is consumed by the retry
    # decorator, which uses it to bound the total time across all attempts;
    # the method bodies never see it. `validate_final_url`, when given, is
    # called with the URL actually served -- after any redirects -- so the
    # caller can re-apply host policy that the initial URL check performed.
    @_sync_retry
    def get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        deadline: float | None = None,
        allow_redirects: bool = True,
        validate_final_url: Callable[[str], None] | None = None,
    ) -> bytes:
        with self.get_response(url, timeout=timeout, allow_redirects=allow_redirects) as r:
            r.raise_for_status()
            if validate_final_url is not None:
                validate_final_url(str(r.url))

            return r.content

    @_async_retry
    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        deadline: float | None = None,
        allow_redirects: bool = True,
        validate_final_url: Callable[[str], None] | None = None,
    ) -> bytes:
        async with await self.get_async_response(url, timeout=timeout, allow_redirects=allow_redirects) as r:
            r.raise_for_status()
            if validate_final_url is not None:
                validate_final_url(str(r.real_url))

            return await r.read()

    def get_text(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.text

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.text()

    def get_json(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.json()

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.json()

    @_sync_retry
    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            with self.get_response(url, timeout=timeout) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise

    @_async_retry
    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            async with await self.get_async_response(
                url,
                timeout=timeout,
            ) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise


global_http_connection = HTTPConnection()
"""
The global [`HTTPConnection`][aphrodite.connections.HTTPConnection] instance used
by Aphrodite.
"""
