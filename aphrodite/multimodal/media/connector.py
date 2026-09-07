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
from aphrodite.connections import HTTPConnection, global_http_connection
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.logger import init_logger
from aphrodite.multimodal.video import get_video_loader_backend_for_processor
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


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified or ip.is_multicast


def _assert_host_allowed(host: str, url: str, parameter: str) -> None:
    """Reject loopback/private/link-local hosts when hardening is enabled.

    Off by default (APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS) so that serving media
    from localhost or the LAN keeps working.
    """
    if not envs.APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS:
        # Logged once per process so the effective policy is visible in the
        # server log -- otherwise a mis-set flag is indistinguishable from a
        # working one, since both end in a 422.
        logger.debug_once("Media URL private-host blocking is OFF (APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS=0)")
        return

    logger.info_once("Media URL private-host blocking is ON (APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS=1)")

    bare = host.strip("[]").lower()
    if bare == "localhost" or bare.endswith(".local") or bare.endswith(".localhost"):
        logger.warning("Blocked media URL %s: host %r is local", url, host)
        raise _reject_url("Media URL host is not permitted.", url, parameter)

    try:
        literal = ipaddress.ip_address(bare)
    except ValueError:
        literal = None

    if literal is not None:
        if _is_blocked_ip(literal):
            logger.warning("Blocked media URL %s: address %s is not routable", url, literal)
            raise _reject_url("Media URL host is not permitted.", url, parameter)
        return

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
            logger.warning("Blocked media URL %s: %s resolves to non-routable %s", url, host, resolved)
            raise _reject_url("Media URL host is not permitted.", url, parameter)


def _validate_media_url(url: str, parameter: str = "media_url") -> None:
    """Reject clearly broken URLs before any socket is opened.

    Raises APHRODITEUnprocessableEntityError (-> HTTP 422) so the caller gets
    a one-line reason at zero network cost.
    """
    if not isinstance(url, str) or not url.strip():
        raise _reject_url("Media URL is empty.", str(url), parameter)

    is_data_url = url[:5].lower() == "data:"

    if not is_data_url and len(url) > _MAX_MEDIA_URL_LENGTH:
        raise _reject_url("Media URL is too long.", url[:128] + "...", parameter)

    if is_data_url:
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
        connection: HTTPConnection = global_http_connection,
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
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs that belong to this
                                   domain can be used for multi-modal inputs.
        """
        super().__init__()

        self.media_io_kwargs: dict[str, dict[str, Any]] = media_io_kwargs if media_io_kwargs else {}
        self.connection = connection

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
                "Malformed data: URL.", parameter="media_url", value="data:..."
            ) from exc

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

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
                "Local file URLs are not accepted. Provide an http(s) or data: URL.",
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
        """Re-apply the private-host block to the URL actually fetched.

        Redirects are followed by default, so without this an allowed origin
        could redirect the server to a loopback or link-local address (e.g. a
        cloud metadata endpoint) after the initial check has passed.

        NOTE: `allowed_media_domains` is deliberately NOT re-applied here.
        That list says which sites a caller may reference, and legitimate
        origins routinely redirect to a separate CDN host (github.com ->
        raw.githubusercontent.com); re-checking it would break them. The
        private-host block is the check that constrains what the server may
        actually connect to, so that is the one that must survive a redirect.
        """
        if not envs.APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS:
            return None

        def _validate(final_url: str) -> None:
            spec = parse_url(final_url)
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
        parameter: str = "media_url",
    ) -> _M:  # type: ignore[type-var]
        # Reject clearly broken URLs before opening any socket.
        _validate_media_url(url, parameter)

        if url[:5].lower() == "data:":
            return self._load_data_url(url, media_io)

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
        parameter: str = "media_url",
    ) -> _M:
        loop = asyncio.get_running_loop()

        # Reject clearly broken URLs before opening any socket. Host policy
        # may need DNS, so run the whole check off the event loop.
        await loop.run_in_executor(global_thread_pool, _validate_media_url, url, parameter)

        if url[:5].lower() == "data:":
            future: asyncio.Future[_M] = loop.run_in_executor(global_thread_pool, self._load_data_url, url, media_io)
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
