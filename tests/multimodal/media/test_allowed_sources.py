# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""`APHRODITE_MEDIA_ALLOWED_SOURCES`: where the server accepts media from.

Four categories, any combination:

    remote   http(s) on a publicly routable host
    private  http(s) on a loopback/private/link-local/reserved host
    file     file:// URLs (still gated by --allowed-local-media-path)
    data     data: URLs carrying the payload inline

The default is all four, which is no policy at all and must stay exactly as
permissive -- and as cheap -- as having no policy. `data` alone refuses
retrieval entirely.
"""

import base64
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

import aphrodite.envs as envs
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.multimodal.media import MediaConnector

ALL_SOURCES = {"remote", "private", "file", "data"}

# 1x1 PNG, the smallest thing that survives the decoder.
PNG_DATA_URL = (
    "data:image/png;base64,"
    + base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000d4944415478da63fcffff3f0300050001ff9b1ce5"
            "0000000049454e44ae426082"
        )
    ).decode()
)


def resolve_env(**environ):
    """Read the variable the way a fresh process would.

    `aphrodite.envs` caches nothing here, but the resolver reads `os.environ`
    directly and emits deprecation warnings, so a subprocess keeps each case
    from leaking into the next.
    """
    code = "import aphrodite.envs as e; print(','.join(sorted(e.APHRODITE_MEDIA_ALLOWED_SOURCES)))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/root", **environ},
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return set(result.stdout.strip().split(","))


class TestResolution:
    def test_default_is_everything(self):
        """No setting must mean no restriction."""
        assert resolve_env() == ALL_SOURCES

    def test_explicit_list(self):
        assert resolve_env(APHRODITE_MEDIA_ALLOWED_SOURCES="remote,data") == {"remote", "data"}

    def test_data_only(self):
        assert resolve_env(APHRODITE_MEDIA_ALLOWED_SOURCES="data") == {"data"}

    def test_case_and_whitespace_insensitive(self):
        """The value must compare equal to the names the connector checks."""
        assert resolve_env(APHRODITE_MEDIA_ALLOWED_SOURCES=" REMOTE , Data ") == {"remote", "data"}

    def test_unknown_source_is_refused(self):
        """Loudly, rather than silently narrowing the policy."""
        with pytest.raises(RuntimeError, match="Invalid value 'remoat'"):
            resolve_env(APHRODITE_MEDIA_ALLOWED_SOURCES="remoat")

    def test_the_flag_it_replaced_is_gone(self):
        """`APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS` was `remote,file,data` by
        another name. It is removed, not aliased, so it must not resolve --
        and `validate_environ` must call it out rather than pass it over."""
        import aphrodite.envs as live

        name = "APHRODITE_MEDIA_BLOCK_PRIVATE_HOSTS"

        assert name not in live.environment_variables
        with pytest.raises(AttributeError):
            getattr(live, name)


@pytest.fixture
def sources(monkeypatch):
    def _set(*names):
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", set(names))

    return _set


class TestRetrievalRefused:
    """`data` alone: the server fetches nothing, callers inline their media."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/a.png",
            "http://example.com/a.png",
            "http://127.0.0.1:8000/a.png",
        ],
    )
    def test_http_urls_refused_without_a_socket(self, sources, url):
        sources("data")
        connector = MediaConnector()

        with (
            patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as get_bytes,
            pytest.raises(APHRODITEUnprocessableEntityError) as excinfo,
        ):
            connector.fetch_image(url)

        assert "not accepted" in str(excinfo.value)
        get_bytes.assert_not_called()

    def test_the_remedy_names_what_is_accepted(self, sources):
        """The body has to tell the caller what to send instead, and `data:`
        is all that is left."""
        sources("data")

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image("https://example.com/a.png")

        assert excinfo.value.args[0] == "This media source is not accepted. Provide data: media instead."

    def test_data_urls_still_work(self, sources):
        sources("data")

        assert MediaConnector().fetch_image(PNG_DATA_URL) is not None

    def test_file_urls_refused(self, sources, tmp_path):
        """Even with a local media path configured: the policy is the outer
        gate, `--allowed-local-media-path` the inner one."""
        sources("data")
        connector = MediaConnector(allowed_local_media_path=str(tmp_path))

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            connector.fetch_image(f"file://{tmp_path}/a.png")

        assert "not accepted" in str(excinfo.value)


class TestDataRefused:
    """The mirror case, which used to be unreachable: `_validate_media_url`
    returned from its `data:` branch before reaching the scheme check."""

    def test_data_url_refused_when_not_permitted(self, sources):
        sources("remote", "private", "file")

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image(PNG_DATA_URL)

        assert "not accepted" in str(excinfo.value)

    def test_malformed_data_url_refused_as_a_source(self, sources):
        """Refused for the reason that holds whether or not it also parses."""
        sources("remote")

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image("data:garbage")

        assert "not accepted" in str(excinfo.value)


class TestPrivateOnly:
    """`private` without `remote`: an appliance that may only read from its
    own network. The inverse of the flag this replaced."""

    def test_public_host_refused(self, sources):
        sources("private", "data")
        connector = MediaConnector()

        with (
            patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as get_bytes,
            pytest.raises(APHRODITEUnprocessableEntityError) as excinfo,
        ):
            connector.fetch_image("http://93.184.216.34/a.png")

        assert "host is not permitted" in str(excinfo.value)
        get_bytes.assert_not_called()

    def test_loopback_host_allowed(self, sources):
        sources("private", "data")
        connector = MediaConnector()

        with patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as get_bytes:
            get_bytes.return_value = b"not-an-image"
            # Reaching the decoder is the point: it was not pre-blocked.
            with pytest.raises(ValueError):
                connector.fetch_image("http://127.0.0.1:8000/a.png")

        get_bytes.assert_called_once()


class TestDefaultCostsNothing:
    """The default permits every host, so classifying one changes no outcome
    -- and reaching the classification costs a DNS lookup per request."""

    def test_no_name_resolution_on_the_default_path(self, sources):
        sources(*ALL_SOURCES)
        connector = MediaConnector()

        with (
            patch("aphrodite.multimodal.media.connector.socket.getaddrinfo") as getaddrinfo,
            patch.object(connector.connection, "get_bytes", new_callable=MagicMock) as get_bytes,
        ):
            get_bytes.return_value = b"not-an-image"
            with pytest.raises(ValueError):
                connector.fetch_image("http://example.com/a.png")

        getaddrinfo.assert_not_called()

    def test_no_redirect_validator_on_the_default_path(self, sources):
        """Nothing a redirect can reach would have been refused up front."""
        sources(*ALL_SOURCES)

        assert MediaConnector()._final_url_validator("image_url") is None


class TestRedirectsCannotEscape:
    """A policy checked only before the first request is not a policy."""

    def test_redirect_to_a_private_host_refused(self, sources):
        sources("remote", "data")
        validate = MediaConnector()._final_url_validator("image_url")

        assert validate is not None
        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            validate("http://169.254.169.254/latest/meta-data")

        assert "host is not permitted" in str(excinfo.value)

    def test_redirect_to_a_public_host_allowed(self, sources):
        sources("remote", "data")
        validate = MediaConnector()._final_url_validator("image_url")

        validate("http://93.184.216.34/a.png")
