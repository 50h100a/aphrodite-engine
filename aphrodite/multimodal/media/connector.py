# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import contextlib
import hashlib
import ipaddress
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.request import url2pathname

import aiohttp
import numpy as np
import numpy.typing as npt
import requests
import torch
from PIL import Image, UnidentifiedImageError
from urllib3.util import Url, parse_url

import aphrodite.envs as envs
from aphrodite.connections import HTTPConnection, ResponseTooLargeError
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.logger import init_logger
from aphrodite.multimodal.video import get_video_loader_backend_for_processor
from aphrodite.net_policy import AddressNotAllowedError
from aphrodite.utils.mem_constants import MiB_bytes
from aphrodite.utils.registry import ExtensionManager

from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoMediaIO

logger = init_logger(__name__)

_M = TypeVar("_M")

global_thread_pool = ThreadPoolExecutor(max_workers=envs.APHRODITE_MEDIA_LOADING_THREAD_COUNT)
atexit.register(global_thread_pool.shutdown)

MEDIA_CONNECTOR_REGISTRY = ExtensionManager()

MODALITY_IO_MAP: dict[str, type[MediaIO]] = {
    "audio": AudioMediaIO,
    "image": ImageMediaIO,
    "video": VideoMediaIO,
}


# Maximum accepted length of a media URL. `data:` URLs carry their payload
# inline and are exempt.
_MAX_MEDIA_URL_LENGTH = 8192

_ALLOWED_MEDIA_URL_SCHEMES = frozenset({"http", "https", "data", "file"})


def _wrap_media_fetch_error(
    url: str,
    exc: Exception,
    parameter: str = "media_url",
) -> APHRODITEUnprocessableEntityError | Exception:
    """Convert permanent media fetch failures into 422 request errors.

    Messages produced here are returned to the API caller, so they state only
    what the caller can act on -- their URL, their field. Server-side detail
    (the underlying exception, errno, configuration) is logged instead of
    being echoed back in the response body.
    """
    # Already shaped for the caller (e.g. raised by _validate_media_url or
    # _load_file_url); don't flatten its message.
    if isinstance(exc, APHRODITEUnprocessableEntityError):
        return exc

    if isinstance(exc, ResponseTooLargeError):
        logger.warning("Rejected media URL %s: %s", url, exc)
        return APHRODITEUnprocessableEntityError(
            "Media exceeds the maximum accepted size.",
            parameter=parameter,
            value=url,
        )

    if isinstance(exc, AddressNotAllowedError):
        # Logged with the address, which is the part worth seeing: a name that
        # passed the URL check and then resolved to something non-routable is
        # the signature of a rebinding attempt, not a typo.
        logger.warning("Blocked media URL %s: %s", url, exc)
        return APHRODITEUnprocessableEntityError(
            "Media URL host is not permitted.",
            parameter=parameter,
            value=url,
        )

    if isinstance(exc, aiohttp.ClientResponseError):
        if exc.status in (408, 429):
            return exc
        if exc.status < 500:
            return APHRODITEUnprocessableEntityError(
                f"Could not fetch media from URL: HTTP {exc.status}.",
                parameter=parameter,
                value=url,
            )
        return exc

    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None:
            status_code = exc.response.status_code
            if status_code in (408, 429):
                return exc
            if status_code < 500:
                return APHRODITEUnprocessableEntityError(
                    f"Could not fetch media from URL: HTTP {status_code}.",
                    parameter=parameter,
                    value=url,
                )
        return exc

    # NOTE: must precede the connection-error branch below --
    # aiohttp.ServerTimeoutError subclasses ClientConnectionError.
    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            requests.exceptions.Timeout,
            aiohttp.ServerTimeoutError,
        ),
    ):
        # The single operator-facing WARN is emitted by the retry wrapper in
        # connections.py when it gives up; don't log the same failure twice.
        logger.debug("Media fetch timed out for %s", url)
        return APHRODITEUnprocessableEntityError(
            "Could not fetch media from URL: timed out.",
            parameter=parameter,
            value=url,
        )

    # By the time this runs the retry ladder has already given up, so from the
    # caller's point of view the URL is simply not fetchable.
    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            socket.gaierror,
            requests.exceptions.ConnectionError,
        ),
    ):
        logger.debug(
            "Media fetch failed for %s: %s: %s",
            url,
            type(exc).__name__,
            exc,
        )
        return APHRODITEUnprocessableEntityError(
            "Could not fetch media from URL: host unreachable.",
            parameter=parameter,
            value=url,
        )

    if isinstance(exc, requests.exceptions.InvalidURL):
        return APHRODITEUnprocessableEntityError(
            "Invalid media URL.",
            parameter=parameter,
            value=url,
        )

    if isinstance(exc, ValueError):
        return APHRODITEUnprocessableEntityError(
            "Invalid media URL.",
            parameter=parameter,
            value=url,
        )
    return exc


def _reject_url(reason: str, url: str, parameter: str) -> APHRODITEUnprocessableEntityError:
    return APHRODITEUnprocessableEntityError(reason, parameter=parameter, value=url)


# Declared types that say nothing about the payload. A caller that claims only
# "some bytes" is not contradicting the field it used, and generic encoders
# emit these routinely, so they are passed through to the decoder -- which
# validates the bytes themselves regardless of what the URL claimed.
_UNSPECIFIC_MEDIA_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})


def _assert_media_type_matches(
    media_type: str,
    media_io: "MediaIO[Any]",
    parameter: str,
) -> None:
    """Reject a `data:` URL whose declared type contradicts the field it came in on.

    The decoders sniff the payload, so bad bytes are already caught downstream;
    what is not caught is the label. Without this check `image_url` happily
    accepts `data:audio/wav;...`, `data:text/html;...` or an outright
    non-media-type string, as long as the bytes behind it decode as an image --
    so the declared type never has to be true, and callers that route or audit
    on it are reading a value the server never verified.
    """
    accepted = media_io.accepted_media_types
    if accepted is None:
        return

    declared = media_type.split(";", 1)[0].strip().lower()
    if declared in _UNSPECIFIC_MEDIA_TYPES:
        return
    if declared.split("/", 1)[0] in accepted:
        return

    expected = "/*, ".join(sorted(accepted)) + "/*"
    raise APHRODITEUnprocessableEntityError(
        f"Media URL declares media type {declared!r}, which is not valid for {parameter}; expected {expected}.",
        parameter=parameter,
        value=f"data:{declared[:64]};...",
    )


def _max_bytes(size_mb: int) -> int | None:
    """A per-modality size cap in MiB, or `None` if the operator lifted it."""
    return size_mb * MiB_bytes if size_mb > 0 else None


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified or ip.is_multicast


def _media_address_allowed(address: str) -> bool:
    """The `remote`/`private` split, asked of a resolved address.

    This is the same policy `_assert_host_allowed` applies to a URL, moved to
    where the address is about to be dialled. The URL check resolves a name to
    decide whether to proceed and the HTTP client then resolves it again to
    decide where to connect; a record with a short TTL can answer those two
    lookups differently, so the first is a pre-flight and this is the one that
    actually constrains the socket.
    """
    sources = envs.APHRODITE_MEDIA_ALLOWED_SOURCES
    remote_ok = "remote" in sources
    private_ok = "private" in sources
    if remote_ok and private_ok:
        return True

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        # Not something we can classify. The scheme and host checks in
        # `_validate_media_url` have already run; leave the verdict to them
        # rather than inventing one here.
        return True

    return private_ok if _is_blocked_ip(ip) else remote_ok


media_http_connection = HTTPConnection(address_policy=_media_address_allowed)
"""Connection used for media fetches.

Kept separate from `global_http_connection` in both directions. Media traffic
needs the address policy, and the other users of the global connection
(usage reporting, batch-job inputs, asset downloads) are operator-configured
rather than caller-supplied -- applying the media policy to them would break
legitimately internal endpoints, and sharing a pooled connection with them
would let media reuse a socket the policy never saw.
"""


def _allowed_sources() -> set[str]:
    """The configured media policy, logged once so it is visible in the log.

    Otherwise a mis-set policy is indistinguishable from a working one: both
    ends of the question look the same from outside, a 422.
    """
    sources = envs.APHRODITE_MEDIA_ALLOWED_SOURCES
    logger.info_once("Media sources permitted: %s", ", ".join(sorted(sources)) or "(none)")
    return sources


def _accepted_forms(sources: set[str]) -> str:
    """What a caller may send, phrased for the caller.

    Built from the policy rather than hardcoded so the remedy in an error is
    the one that will actually work on this server. It names URL forms, which
    is what the caller controls -- never the setting behind them.
    """
    forms = []
    if "remote" in sources or "private" in sources:
        forms.append("http(s)")
    if "data" in sources:
        forms.append("data:")
    if "file" in sources:
        forms.append("file:")
    if not forms:
        return ""
    if len(forms) == 1:
        return forms[0]
    return ", ".join(forms[:-1]) + " or " + forms[-1]


def _reject_source(url: str, parameter: str, sources: set[str]) -> APHRODITEUnprocessableEntityError:
    accepted = _accepted_forms(sources)
    detail = f" Provide {accepted} media instead." if accepted else ""
    return _reject_url(f"This media source is not accepted.{detail}", url, parameter)


def _host_is_private(host: str, url: str, parameter: str) -> bool:
    """Whether `host` names a non-routable address.

    Resolves names, because a public name pointing at 169.254.169.254 is the
    whole reason the distinction exists.
    """
    bare = host.strip("[]").lower()
    if bare == "localhost" or bare.endswith(".local") or bare.endswith(".localhost"):
        return True

    try:
        return _is_blocked_ip(ipaddress.ip_address(bare))
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(bare, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        logger.warning("Blocked media URL %s: name resolution failed: %s", url, exc)
        raise _reject_url("Could not fetch media from URL: host unreachable.", url, parameter) from exc

    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_blocked_ip(resolved):
            logger.warning("Media URL %s: %s resolves to non-routable %s", url, host, resolved)
            return True
    return False


def _assert_host_allowed(host: str, url: str, parameter: str) -> None:
    """Check an http(s) host against the `remote`/`private` split.

    Skipped entirely when the two are permitted alike, which is the default:
    the classification would change no outcome, and it costs a DNS lookup on
    the request path to reach it.
    """
    sources = _allowed_sources()
    remote_ok = "remote" in sources
    private_ok = "private" in sources
    if remote_ok and private_ok:
        return

    if _host_is_private(host, url, parameter):
        if not private_ok:
            logger.warning("Blocked media URL %s: host %r is not publicly routable", url, host)
            raise _reject_url("Media URL host is not permitted.", url, parameter)
    elif not remote_ok:
        logger.warning("Blocked media URL %s: host %r is not local", url, host)
        raise _reject_url("Media URL host is not permitted.", url, parameter)


def _permitted_schemes(sources: set[str]) -> frozenset[str]:
    """URL schemes the configured sources leave open.

    `remote` and `private` both speak http(s) -- they differ on which hosts,
    which `_assert_host_allowed` decides once the scheme has passed.
    """
    schemes = set()
    if "remote" in sources or "private" in sources:
        schemes |= {"http", "https"}
    if "file" in sources:
        schemes.add("file")
    if "data" in sources:
        schemes.add("data")
    return frozenset(schemes)


def _validate_media_url(url: str, parameter: str = "media_url") -> None:
    """Reject clearly broken URLs before any socket is opened.

    Raises APHRODITEUnprocessableEntityError (-> HTTP 422) so the caller gets
    a one-line reason at zero network cost.
    """
    if not isinstance(url, str) or not url.strip():
        raise _reject_url("Media URL is empty.", str(url), parameter)

    sources = _allowed_sources()
    is_data_url = url[:5].lower() == "data:"

    if not is_data_url and len(url) > _MAX_MEDIA_URL_LENGTH:
        raise _reject_url("Media URL is too long.", url[:128] + "...", parameter)

    if is_data_url:
        # Checked before the shape, so that a form the policy refuses is
        # refused whether or not it is also malformed.
        if "data" not in sources:
            logger.warning("Blocked media URL: data: source is not permitted")
            raise _reject_source("data:...", parameter, sources)
        # `_load_data_url` splits on "," then ";"; without them it would raise
        # a bare unpack ValueError with no useful message.
        spec, _, remainder = url[5:].partition(",")
        if not remainder and "," not in url[5:]:
            raise _reject_url("Malformed data: URL.", "data:...", parameter)
        if ";" not in spec:
            raise _reject_url("Malformed data: URL.", "data:...", parameter)
        return

    try:
        url_spec = parse_url(url)
    except Exception as exc:
        raise _reject_url("Invalid media URL.", url, parameter) from exc

    scheme = (url_spec.scheme or "").lower()
    if scheme not in _ALLOWED_MEDIA_URL_SCHEMES:
        raise _reject_url(
            "Unsupported media URL scheme; expected http, https, data or file.",
            url,
            parameter,
        )
    # A scheme this build understands but this server does not accept. Kept
    # apart from the check above: one says the URL is not a media URL, the
    # other says the operator does not serve media that way.
    if scheme not in _permitted_schemes(sources):
        logger.warning("Blocked media URL %s: %s: source is not permitted", url, scheme)
        raise _reject_source(url, parameter, sources)

    if scheme in ("http", "https"):
        if not url_spec.host:
            raise _reject_url("Media URL has no host.", url, parameter)
        try:
            port = url_spec.port
        except ValueError as exc:
            raise _reject_url("Media URL has an invalid port.", url, parameter) from exc
        if port is not None and not (1 <= port <= 65535):
            raise _reject_url("Media URL has an invalid port.", url, parameter)
        _assert_host_allowed(url_spec.host, url, parameter)


def merge_media_io_kwargs(
    defaults: dict[str, dict[str, Any]] | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge config-level and per-request media_io_kwargs per modality.

    Each modality key is merged using the corresponding MediaIO subclass's
    ``merge_kwargs``, which may apply modality-specific logic (e.g.
    VideoMediaIO clears cross-dependent fps/num_frames fields).
    """
    if not defaults and not overrides:
        return None
    all_keys = set(defaults or {}) | set(overrides or {})
    merged = {}
    for key in all_keys:
        io_cls = MODALITY_IO_MAP.get(key, MediaIO)
        merged[key] = io_cls.merge_kwargs(
            (defaults or {}).get(key),
            (overrides or {}).get(key),
        )
    return merged or None


@MEDIA_CONNECTOR_REGISTRY.register("http")
class MediaConnector:
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        connection: HTTPConnection | None = None,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None:
        """
        Args:
            media_io_kwargs: Additional args passed to process media
                             inputs, keyed by modalities. For example,
                             to set num_frames for video, set
                             `--media-io-kwargs '{"video":{"num_frames":40}}'`
            connection: HTTP connection client to download media contents.
                        Defaults to `media_http_connection`, which enforces
                        `APHRODITE_MEDIA_ALLOWED_SOURCES` at connect time.
                        Passing another connection opts out of that.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs that belong to this
                                   domain can be used for multi-modal inputs.
        """
        super().__init__()

        self.media_io_kwargs: dict[str, dict[str, Any]] = media_io_kwargs if media_io_kwargs else {}
        self.connection = connection if connection is not None else media_http_connection

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path).resolve()

            if not allowed_local_media_path_.exists():
                raise ValueError(
                    f"Invalid `--allowed-local-media-path`: The path {allowed_local_media_path_} does not exist."
                )
            if not allowed_local_media_path_.is_dir():
                raise ValueError(
                    f"Invalid `--allowed-local-media-path`: The path {allowed_local_media_path_} must be a directory."
                )
        else:
            allowed_local_media_path_ = None

        self.allowed_local_media_path = allowed_local_media_path_
        if allowed_media_domains is None:
            allowed_media_domains = []
        self.allowed_media_domains = allowed_media_domains

        # Media download cache (opt-in via APHRODITE_MEDIA_CACHE)
        self._media_cache_dir: str | None = None
        self._media_cache_max_bytes: int = 0
        self._media_cache_ttl_secs: float = 0
        media_cache = envs.APHRODITE_MEDIA_CACHE
        if media_cache:
            try:
                os.makedirs(media_cache, exist_ok=True)
                # Verify the directory is writable before enabling caching
                with tempfile.NamedTemporaryFile(dir=media_cache, delete=True):
                    pass
                self._media_cache_dir = media_cache
                self._media_cache_max_bytes = envs.APHRODITE_MEDIA_CACHE_MAX_SIZE_MB * 1024 * 1024
                self._media_cache_ttl_secs = envs.APHRODITE_MEDIA_CACHE_TTL_HOURS * 3600
                logger.info(
                    "Media cache enabled at %s (max %d MB, TTL %s hours)",
                    media_cache,
                    envs.APHRODITE_MEDIA_CACHE_MAX_SIZE_MB,
                    envs.APHRODITE_MEDIA_CACHE_TTL_HOURS,
                )
            except OSError:
                logger.warning(
                    "APHRODITE_MEDIA_CACHE path %s is not writable, media caching disabled",
                    media_cache,
                )

    def _get_cached_bytes(self, url: str) -> bytes | None:
        """Return cached bytes for a URL, or None if not cached/expired."""
        if not self._media_cache_dir:
            return None
        cache_path = self._media_cache_path(url)
        # Check TTL
        try:
            age = time.time() - cache_path.stat().st_mtime
        except OSError:
            return None
        if age > self._media_cache_ttl_secs:
            cache_path.unlink(missing_ok=True)
            return None
        # Touch mtime for LRU ordering
        try:
            cache_path.touch()
            return cache_path.read_bytes()
        except OSError:
            return None

    def _put_cached_bytes(self, url: str, data: bytes) -> None:
        """Store downloaded bytes and evict if over budget."""
        if not self._media_cache_dir:
            return
        cache_path = self._media_cache_path(url)
        # Atomic write via temp file + rename
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self._media_cache_dir, delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            os.rename(tmp_path, str(cache_path))
        except OSError:
            # Another process beat us or disk issue
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return
        self._maybe_evict(exclude=cache_path)

    def _maybe_evict(self, exclude: Path | None = None) -> None:
        """Evict expired entries first, then LRU until under size limit."""
        cache_dir = Path(self._media_cache_dir)  # type: ignore[arg-type]
        entries = []
        expired = []
        total_size = 0
        now = time.time()
        for f in cache_dir.iterdir():
            if f.name.startswith("."):
                continue
            try:
                stat = f.stat()
            except OSError:
                continue
            age = now - stat.st_mtime
            if age > self._media_cache_ttl_secs:
                expired.append(f)
                continue
            total_size += stat.st_size
            # Never evict the file we just wrote
            if exclude is not None and f.name == exclude.name:
                continue
            entries.append((stat.st_mtime, stat.st_size, f))

        # Evict items according to LRU policy
        entries.sort(key=lambda e: e[0], reverse=True)
        while total_size > self._media_cache_max_bytes and entries:
            mtime, size, f = entries.pop()
            expired.append(f)
            total_size -= size

        for f in expired:
            f.unlink(missing_ok=True)

    def _media_cache_path(self, url: str) -> Path:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:20]
        ext = Path(url.split("?", 1)[0]).suffix or ""
        return Path(self._media_cache_dir) / f"{url_hash}{ext}"  # type: ignore[arg-type]

    def _load_data_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        parameter: str = "media_url",
        max_bytes: int | None = None,
    ) -> _M:  # type: ignore[type-var]
        # Format per RFC 2397:
        # data:[<mediatype>][;base64],<data>
        # `_validate_media_url` has already checked the shape; re-check here
        # so direct callers get a clean message instead of an unpack error.
        try:
            data_spec, data = url[5:].split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
        except ValueError as exc:
            raise APHRODITEUnprocessableEntityError(
                "Malformed data: URL.", parameter=parameter, value="data:..."
            ) from exc

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        _assert_media_type_matches(media_type, media_io, parameter)

        # Same budget an http(s) body gets, checked from the encoded length so
        # an oversized payload is refused without allocating the decode. Four
        # base64 characters carry three bytes; padding only makes the estimate
        # generous, which is the right direction for a limit.
        if max_bytes is not None and (len(data) // 4) * 3 > max_bytes:
            raise _reject_url("Media exceeds the maximum accepted size.", "data:...", parameter)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: Url,
        media_io: MediaIO[_M],
        parameter: str = "media_url",
    ) -> _M:  # type: ignore[type-var]
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            # The caller cannot set a server CLI flag, so the flag name goes
            # to the log for whoever runs the server, not into the response.
            logger.warning(
                "Rejected file:// media URL %s; set --allowed-local-media-path to enable local files",
                url_spec.url,
            )
            raise APHRODITEUnprocessableEntityError(
                "Local file URLs are not accepted.",
                parameter=parameter,
                value=url_spec.url,
            )

        url_spec_path = url_spec.path or ""
        url_spec_netloc = url_spec.netloc or ""
        filepath = Path(url2pathname(url_spec_netloc + url_spec_path))
        if allowed_local_media_path not in filepath.resolve().parents:
            logger.warning(
                "Rejected file:// media URL %s: %s is outside --allowed-local-media-path %s",
                url_spec.url,
                filepath,
                allowed_local_media_path,
            )
            raise APHRODITEUnprocessableEntityError(
                "Local file URL is outside the permitted directory.",
                parameter=parameter,
                value=url_spec.url,
            )

        return media_io.load_file(filepath)

    def _assert_url_in_allowed_media_domains(self, url_spec: Url, parameter: str = "media_url") -> None:
        if self.allowed_media_domains and url_spec.hostname not in self.allowed_media_domains:
            logger.warning(
                "Rejected media URL %s: host %s is not in --allowed-media-domains %s",
                url_spec.url,
                url_spec.hostname,
                self.allowed_media_domains,
            )
            raise APHRODITEUnprocessableEntityError(
                "Media URL host is not permitted.",
                parameter=parameter,
                value=url_spec.url,
            )

    def _final_url_validator(self, parameter: str):
        """Re-apply the source policy to the URL actually fetched.

        Redirects are followed by default, so without this an allowed origin
        could redirect the server to a loopback or link-local address (e.g. a
        cloud metadata endpoint), or from https down to http, after the initial
        check has passed.

        NOTE: `allowed_media_domains` is deliberately NOT re-applied here.
        That list says which sites a caller may reference, and legitimate
        origins routinely redirect to a separate CDN host (github.com ->
        raw.githubusercontent.com); re-checking it would break them. The source
        policy is the check that constrains what the server may actually
        connect to, so that is the one that must survive a redirect.
        """
        sources = envs.APHRODITE_MEDIA_ALLOWED_SOURCES
        permitted = _permitted_schemes(sources)
        # Nothing to re-check when every http(s) destination is acceptable:
        # the redirect cannot reach anything the first check would have
        # refused. Keeps the default path free of a second DNS lookup.
        if {"http", "https"} <= permitted and "remote" in sources and "private" in sources:
            return None

        def _validate(final_url: str) -> None:
            spec = parse_url(final_url)
            scheme = (spec.scheme or "").lower()
            if scheme not in permitted:
                logger.warning("Blocked media redirect to %s: %s: source is not permitted", final_url, scheme)
                raise _reject_source(final_url, parameter, sources)
            if spec.host:
                _assert_host_allowed(spec.host, final_url, parameter)

        return _validate

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
        fetch_deadline: int | None = None,
        fetch_max_bytes: int | None = None,
        parameter: str = "media_url",
    ) -> _M:  # type: ignore[type-var]
        # Reject clearly broken URLs before opening any socket.
        _validate_media_url(url, parameter)

        if url[:5].lower() == "data:":
            return self._load_data_url(url, media_io, parameter, fetch_max_bytes)

        url_spec = parse_url(url)
        scheme = (url_spec.scheme or "").lower()

        if scheme in ("http", "https"):
            self._assert_url_in_allowed_media_domains(url_spec, parameter)

            cached = self._get_cached_bytes(url)
            if cached is not None:
                return media_io.load_bytes(cached)

            connection = self.connection
            try:
                data = connection.get_bytes(
                    url_spec.url,
                    timeout=fetch_timeout,
                    deadline=fetch_deadline,
                    allow_redirects=envs.APHRODITE_MEDIA_URL_ALLOW_REDIRECTS,
                    validate_final_url=self._final_url_validator(parameter),
                    max_bytes=fetch_max_bytes,
                )
            except Exception as e:
                wrapped = _wrap_media_fetch_error(url, e, parameter)
                if isinstance(wrapped, APHRODITEUnprocessableEntityError):
                    raise wrapped from e
                raise

            self._put_cached_bytes(url, data)
            return media_io.load_bytes(data)

        if scheme == "file":
            try:
                return self._load_file_url(url_spec, media_io, parameter)
            except Exception as e:
                wrapped = _wrap_media_fetch_error(url, e, parameter)
                if isinstance(wrapped, APHRODITEUnprocessableEntityError):
                    raise wrapped from e
                raise

        raise _reject_url(
            "Unsupported media URL scheme; expected http, https, data or file.",
            url,
            parameter,
        )

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
        fetch_deadline: int | None = None,
        fetch_max_bytes: int | None = None,
        parameter: str = "media_url",
    ) -> _M:
        loop = asyncio.get_running_loop()

        # Reject clearly broken URLs before opening any socket. Host policy
        # may need DNS, so run the whole check off the event loop.
        await loop.run_in_executor(global_thread_pool, _validate_media_url, url, parameter)

        if url[:5].lower() == "data:":
            future: asyncio.Future[_M] = loop.run_in_executor(
                global_thread_pool, self._load_data_url, url, media_io, parameter, fetch_max_bytes
            )
            return await future

        url_spec = parse_url(url)
        scheme = (url_spec.scheme or "").lower()

        if scheme in ("http", "https"):
            self._assert_url_in_allowed_media_domains(url_spec, parameter)

            cached = await loop.run_in_executor(global_thread_pool, self._get_cached_bytes, url)
            if cached is not None:
                future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, cached)
                return await future

            connection = self.connection
            try:
                data = await connection.async_get_bytes(
                    url_spec.url,
                    timeout=fetch_timeout,
                    deadline=fetch_deadline,
                    allow_redirects=envs.APHRODITE_MEDIA_URL_ALLOW_REDIRECTS,
                    validate_final_url=self._final_url_validator(parameter),
                    max_bytes=fetch_max_bytes,
                )
            except Exception as e:
                wrapped = _wrap_media_fetch_error(url, e, parameter)
                if isinstance(wrapped, APHRODITEUnprocessableEntityError):
                    raise wrapped from e
                raise

            await loop.run_in_executor(global_thread_pool, self._put_cached_bytes, url, data)
            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            return await future

        if scheme == "file":
            try:
                future = loop.run_in_executor(global_thread_pool, self._load_file_url, url_spec, media_io, parameter)
                return await future
            except Exception as e:
                wrapped = _wrap_media_fetch_error(url, e, parameter)
                if isinstance(wrapped, APHRODITEUnprocessableEntityError):
                    raise wrapped from e
                raise

        raise _reject_url(
            "Unsupported media URL scheme; expected http, https, data or file.",
            url,
            parameter,
        )

    def fetch_audio(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Load audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return self.load_from_url(
            audio_url,
            audio_io,
            fetch_timeout=envs.APHRODITE_AUDIO_FETCH_TIMEOUT,
            fetch_deadline=envs.APHRODITE_AUDIO_FETCH_DEADLINE,
            fetch_max_bytes=_max_bytes(envs.APHRODITE_AUDIO_FETCH_MAX_SIZE_MB),
            parameter="audio_url",
        )

    async def fetch_audio_async(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Asynchronously fetch audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return await self.load_from_url_async(
            audio_url,
            audio_io,
            fetch_timeout=envs.APHRODITE_AUDIO_FETCH_TIMEOUT,
            fetch_deadline=envs.APHRODITE_AUDIO_FETCH_DEADLINE,
            fetch_max_bytes=_max_bytes(envs.APHRODITE_AUDIO_FETCH_MAX_SIZE_MB),
            parameter="audio_url",
        )

    def fetch_image(
        self,
        image_url: str,
        *,
        image_mode: str | None = "RGB",
    ) -> Image.Image:
        """
        Load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format. Set
        `media_io_kwargs={"image": {"image_mode": None}}` to keep the
        original image mode (e.g. preserving the alpha channel).
        """
        image_io = ImageMediaIO(**({"image_mode": image_mode} | self.media_io_kwargs.get("image", {})))

        try:
            return self.load_from_url(
                image_url,
                image_io,
                fetch_timeout=envs.APHRODITE_IMAGE_FETCH_TIMEOUT,
                fetch_deadline=envs.APHRODITE_IMAGE_FETCH_DEADLINE,
                fetch_max_bytes=_max_bytes(envs.APHRODITE_IMAGE_FETCH_MAX_SIZE_MB),
                parameter="image_url",
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str | None = "RGB",
    ) -> Image.Image:
        """
        Asynchronously load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format. Set
        `media_io_kwargs={"image": {"image_mode": None}}` to keep the
        original image mode (e.g. preserving the alpha channel).
        """
        image_io = ImageMediaIO(**({"image_mode": image_mode} | self.media_io_kwargs.get("image", {})))

        try:
            return await self.load_from_url_async(
                image_url,
                image_io,
                fetch_timeout=envs.APHRODITE_IMAGE_FETCH_TIMEOUT,
                fetch_deadline=envs.APHRODITE_IMAGE_FETCH_DEADLINE,
                fetch_max_bytes=_max_bytes(envs.APHRODITE_IMAGE_FETCH_MAX_SIZE_MB),
                parameter="image_url",
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    def fetch_video(
        self,
        video_url: str,
        *,
        image_mode: str | None = "RGB",
        video_processor: str | None = None,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video from an HTTP or base64 data URL.
        """
        image_io = ImageMediaIO(**({"image_mode": image_mode} | self.media_io_kwargs.get("image", {})))
        video_io_kwargs = dict(self.media_io_kwargs.get("video", {}))
        if "video_backend" not in video_io_kwargs and (
            video_backend := get_video_loader_backend_for_processor(video_processor)
        ):
            video_io_kwargs["video_backend"] = video_backend
        video_io = VideoMediaIO(image_io, **video_io_kwargs)

        return self.load_from_url(
            video_url,
            video_io,
            fetch_timeout=envs.APHRODITE_VIDEO_FETCH_TIMEOUT,
            fetch_deadline=envs.APHRODITE_VIDEO_FETCH_DEADLINE,
            fetch_max_bytes=_max_bytes(envs.APHRODITE_VIDEO_FETCH_MAX_SIZE_MB),
            parameter="video_url",
        )

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        image_mode: str | None = "RGB",
        video_processor: str | None = None,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Asynchronously load video from an HTTP or base64 data URL.

        By default, the image is converted into RGB format. Set
        `media_io_kwargs={"image": {"image_mode": None}}` to keep the
        original image mode (e.g. preserving the alpha channel).
        """
        image_io = ImageMediaIO(**({"image_mode": image_mode} | self.media_io_kwargs.get("image", {})))
        video_io_kwargs = dict(self.media_io_kwargs.get("video", {}))
        if "video_backend" not in video_io_kwargs and (
            video_backend := get_video_loader_backend_for_processor(video_processor)
        ):
            video_io_kwargs["video_backend"] = video_backend
        video_io = VideoMediaIO(image_io, **video_io_kwargs)

        return await self.load_from_url_async(
            video_url,
            video_io,
            fetch_timeout=envs.APHRODITE_VIDEO_FETCH_TIMEOUT,
            fetch_deadline=envs.APHRODITE_VIDEO_FETCH_DEADLINE,
            fetch_max_bytes=_max_bytes(envs.APHRODITE_VIDEO_FETCH_MAX_SIZE_MB),
            parameter="video_url",
        )

    def fetch_image_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load image embedding from a URL.
        """
        image_embedding_io = ImageEmbeddingMediaIO()

        return image_embedding_io.load_base64("", data)

    def fetch_audio_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load audio embedding from a URL.
        """
        audio_embedding_io = AudioEmbeddingMediaIO()

        return audio_embedding_io.load_base64("", data)
