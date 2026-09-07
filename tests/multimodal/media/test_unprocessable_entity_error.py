# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for unprocessable media URL error handling."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

import aphrodite.envs as envs
from aphrodite.entrypoints.serve.utils.error_response import create_error_response
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.multimodal.media import MediaConnector


class TestAphroditeUnprocessableEntityError:
    def test_creation(self):
        exc = APHRODITEUnprocessableEntityError("Test error")
        assert str(exc) == "Test error"
        assert exc.parameter is None

    def test_creation_with_parameter_and_value(self):
        exc = APHRODITEUnprocessableEntityError(
            "Test error",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )
        assert "parameter=image_url" in str(exc)
        assert "value=https://example.com/image.jpg" in str(exc)

    def test_is_value_error_subclass(self):
        exc = APHRODITEUnprocessableEntityError("Test")
        assert isinstance(exc, ValueError)


class TestMediaConnectorErrorHandling:
    @pytest.mark.asyncio
    async def test_fetch_image_async_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    @pytest.mark.asyncio
    async def test_fetch_image_async_dns_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("DNS lookup failed")

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://nonexistent.example/image.jpg")

            assert exc_info.value.parameter == "image_url"
            assert "unreachable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_async_500_preserved(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=500,
                message="Internal Server Error",
            )

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert exc_info.value.status == 500

    def test_fetch_image_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "get_bytes",
            new_callable=MagicMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    def test_fetch_image_connection_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "get_bytes",
            new_callable=MagicMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert exc_info.value.parameter == "image_url"
            assert "unreachable" in str(exc_info.value)


class TestErrorResponse:
    def test_unprocessable_entity_returns_422(self):
        exc = APHRODITEUnprocessableEntityError(
            "Failed to fetch media from URL: Cannot connect",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )

        response = create_error_response(exc)

        assert response.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert response.error.type == "UnprocessableEntityError"
        assert response.error.param == "image_url"

    def test_unprocessable_entity_message(self):
        exc = APHRODITEUnprocessableEntityError("Test error message")
        response = create_error_response(exc)

        assert response.error.message == "Test error message"
        assert response.error.code == 422


class TestMediaUrlValidation:
    """Clearly broken URLs are rejected before any socket is opened."""

    @pytest.mark.parametrize(
        ("url", "fragment"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("example.com/x.png", "scheme"),
            ("httpx://example.com/x.png", "scheme"),
            ("ftp://example.com/x.png", "scheme"),
            ("http:///x.png", "no host"),
            ("data:image/png,AAA", "data"),
            ("data:,AAA", "data"),
            ("https://example.com/" + "a" * 9000, "too long"),
        ],
    )
    def test_rejected_without_fetching(self, url, fragment):
        connector = MediaConnector()

        with (
            patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get,
            pytest.raises(APHRODITEUnprocessableEntityError) as exc_info,
        ):
            connector.fetch_image(url)

        assert fragment in str(exc_info.value).lower()
        mock_get.assert_not_called()

    def test_file_url_without_allowed_path(self):
        connector = MediaConnector()

        with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
            connector.fetch_image("file:///etc/hostname")

        assert exc_info.value.parameter == "image_url"
        assert create_error_response(exc_info.value).error.code == 422

    @pytest.mark.asyncio
    async def test_file_url_without_allowed_path_async(self):
        connector = MediaConnector()

        with pytest.raises(APHRODITEUnprocessableEntityError):
            await connector.fetch_video_async("file:///etc/hostname")

    def test_private_host_allowed_by_default(self, monkeypatch):
        """Default-off: serving media from localhost must keep working."""
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", {"remote", "private", "file", "data"})
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get:
            mock_get.return_value = b"not-an-image"
            # Reaches the decoder, which is the point: it was not pre-blocked.
            with pytest.raises(ValueError):
                connector.fetch_image("http://127.0.0.1:8000/a.png")

        mock_get.assert_called_once()

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/a.png",
            "http://localhost:8000/a.png",
            "http://169.254.169.254/latest/meta-data",
            "http://192.168.1.5/a.png",
            "http://10.0.0.1/a.png",
            "http://[::1]/a.png",
        ],
    )
    def test_private_host_blocked_when_enabled(self, monkeypatch, url):
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", {"remote", "file", "data"})
        connector = MediaConnector()

        with (
            patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get,
            pytest.raises(APHRODITEUnprocessableEntityError) as exc_info,
        ):
            connector.fetch_image(url)

        assert "not permitted" in str(exc_info.value)
        mock_get.assert_not_called()


class TestModalityParameter:
    """The 422 body must name the field the caller actually sent."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fetch", "parameter"),
        [
            ("fetch_image_async", "image_url"),
            ("fetch_video_async", "video_url"),
            ("fetch_audio_async", "audio_url"),
        ],
    )
    async def test_parameter_matches_modality(self, fetch, parameter):
        connector = MediaConnector()

        with patch.object(
            connector.connection,
            "async_get_bytes",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=404, message="Not Found"
            )

            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                await getattr(connector, fetch)("https://example.com/missing")

        assert exc_info.value.parameter == parameter


class TestMessageHygiene:
    """Response bodies are written for the API caller, not the operator.

    They must not disclose server configuration, filesystem paths, or raw
    library exception text.
    """

    # Server-side detail that must never reach a response body. The caller's
    # own URL is echoed back deliberately (via `value=`) and is not a leak.
    LEAKS = (
        "--allowed-local-media-path",
        "--allowed-media-domains",
        "ssl:default",
        "Connect call failed",
        "Traceback",
        "/root/",
        "aiohttp",
        "errno",
    )

    def _assert_clean(self, exc):
        sentence = exc.args[0]
        body = create_error_response(exc).error.message

        for leak in self.LEAKS:
            assert leak not in body, f"leaked {leak!r} in: {body}"

        # At most a statement plus a remedy, e.g.
        # "Local file URLs are not accepted."
        assert sentence.count(".") <= 2, f"too many sentences: {sentence!r}"
        assert len(sentence) < 120, f"too long for an API error: {sentence!r}"

    def test_file_url_message_is_clean(self):
        connector = MediaConnector()
        with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
            connector.fetch_image("file:///etc/passwd")
        self._assert_clean(exc_info.value)

    def test_file_url_outside_root_message_is_clean(self, tmp_path):
        connector = MediaConnector(allowed_local_media_path=str(tmp_path))
        with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
            connector.fetch_image("file:///etc/passwd")
        body = create_error_response(exc_info.value).error.message
        assert str(tmp_path) not in body, "server config path leaked into response"
        self._assert_clean(exc_info.value)

    def test_connection_error_message_is_clean(self):
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=ConnectionRefusedError(111, "Connection refused"),
            )
            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                connector.fetch_image("http://127.0.0.1:1/missing.png")

        self._assert_clean(exc_info.value)

    def test_timeout_message_is_clean(self):
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as mock_get:
            mock_get.side_effect = TimeoutError()
            with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/slow.png")

        self._assert_clean(exc_info.value)
        assert "timed out" in str(exc_info.value)


class TestEmptyMediaPart:
    """A content part with neither a URL nor a UUID is a broken request.

    Without the guard the `None` reached the multimodal parser and surfaced
    as an `assert_never` AssertionError -- HTTP 500 for what is a plain
    client mistake.
    """

    @pytest.mark.parametrize("parameter", ["image_url", "audio_url", "video_url"])
    @pytest.mark.parametrize("url", [None, "", "   "])
    def test_rejected_when_no_uuid(self, parameter, url):
        from aphrodite.entrypoints.chat_utils import _require_url_or_uuid

        with pytest.raises(APHRODITEUnprocessableEntityError) as exc_info:
            _require_url_or_uuid(url, None, parameter)

        assert exc_info.value.parameter == parameter
        assert create_error_response(exc_info.value).error.code == 422

    @pytest.mark.parametrize("parameter", ["image_url", "audio_url", "video_url"])
    def test_allowed_when_uuid_present(self, parameter):
        """Data supplied out of band via UUID is legitimate."""
        from aphrodite.entrypoints.chat_utils import _require_url_or_uuid

        _require_url_or_uuid(None, "some-uuid", parameter)

    def test_allowed_when_url_present(self):
        from aphrodite.entrypoints.chat_utils import _require_url_or_uuid

        _require_url_or_uuid("https://example.com/a.png", None, "image_url")
