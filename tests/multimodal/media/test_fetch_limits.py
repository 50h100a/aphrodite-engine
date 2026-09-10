# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What a media fetch is allowed to cost the server.

Two budgets that the decode-side limits do not cover. `APHRODITE_MAX_IMAGE_PIXELS`
and friends bound what a payload expands to once decoded, which says nothing
about the payload itself: the bytes are already resident by then. And the
`remote`/`private` split is decided from a URL, which says nothing about the
address the socket ends up dialling if the name is free to answer twice.
"""

import base64
import http.server
import socket
import threading
from collections.abc import Iterator
from unittest.mock import patch

import pytest

import aphrodite.envs as envs
from aphrodite.connections import ResponseTooLargeError
from aphrodite.exceptions import APHRODITEUnprocessableEntityError
from aphrodite.multimodal.media import MediaConnector
from aphrodite.multimodal.media.connector import media_http_connection

# 1x1 PNG -- the smallest thing that survives the decoder.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63fcffff3f0300050001ff9b1ce5"
    "0000000049454e44ae426082"
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()

MIB = 1 << 20
PUBLIC_IP = "93.184.216.34"
LOOPBACK_IP = "127.0.0.1"


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - silence the test log
        pass

    def do_GET(self):  # noqa: N802 - stdlib naming
        state = self.server.state  # type: ignore[attr-defined]
        state["requests"].append(self.path)

        if self.path == "/small":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG_BYTES)))
            self.end_headers()
            self.wfile.write(PNG_BYTES)
            return

        if self.path == "/declared-huge":
            # Honest about a size we will refuse: the body should never be
            # asked for, so anything written after the headers would be a bug.
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(state["size"]))
            self.end_headers()
            state["body_bytes"] += self._write_padding(state["size"])
            return

        if self.path == "/chunked-huge":
            # No Content-Length at all: only a running counter over the body
            # can stop this one.
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = b"\0" * (64 << 10)
            sent = 0
            try:
                while sent < state["size"]:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk + b"\r\n")
                    sent += len(chunk)
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Expected: the client hit its cap and hung up on us.
                pass
            state["body_bytes"] += sent
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _write_padding(self, size: int) -> int:
        chunk = b"\0" * (64 << 10)
        sent = 0
        try:
            while sent < size:
                n = min(len(chunk), size - sent)
                self.wfile.write(chunk[:n])
                sent += n
        except (BrokenPipeError, ConnectionResetError):
            pass
        return sent


class _Server:
    def __init__(self, size: int) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.state = {"requests": [], "body_bytes": 0, "size": size}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def requests(self) -> list[str]:
        return self.httpd.state["requests"]  # type: ignore[attr-defined]

    @property
    def body_bytes(self) -> int:
        return self.httpd.state["body_bytes"]  # type: ignore[attr-defined]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[_Server]:
    srv = _Server(size=4 * MIB)
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture
def cap(monkeypatch):
    """Set the image fetch cap, in MiB."""

    def _set(size_mb: int):
        monkeypatch.setattr(envs, "APHRODITE_IMAGE_FETCH_MAX_SIZE_MB", size_mb)

    return _set


@pytest.fixture
def no_retries(monkeypatch):
    """Keep a rejection from being obscured by the retry ladder."""
    monkeypatch.setattr(envs, "APHRODITE_MEDIA_FETCH_MAX_RETRIES", 1)


class TestBodySizeCap:
    def test_declared_oversize_is_refused_on_the_header(self, server, no_retries):
        """An origin honest enough to send Content-Length is taken at its word.

        Asserted at the transport, where the distinction is visible: `declared`
        says the refusal came from the header, which is to say before the body
        was requested at all.
        """
        with pytest.raises(ResponseTooLargeError) as excinfo:
            media_http_connection.get_bytes(server.url("/declared-huge"), timeout=10, max_bytes=MIB)

        assert excinfo.value.declared
        assert excinfo.value.size == 4 * MIB

    def test_declared_oversize_reaches_the_caller_as_422(self, server, cap):
        cap(1)

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image(server.url("/declared-huge"))

        assert "maximum accepted size" in str(excinfo.value)

    def test_undeclared_oversize_is_refused_mid_stream(self, server, no_retries):
        """No Content-Length to consult, so the running counter over the body
        is the only thing standing between the caller and 4 MiB of RAM.

        `declared` being false is the point: this refusal came from counting
        the body as it arrived, not from trusting a header the origin chose
        not to send.
        """
        with pytest.raises(ResponseTooLargeError) as excinfo:
            media_http_connection.get_bytes(server.url("/chunked-huge"), timeout=10, max_bytes=MIB)

        assert not excinfo.value.declared
        assert excinfo.value.limit == MIB

    def test_undeclared_oversize_reaches_the_caller_as_422(self, server, cap):
        cap(1)

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image(server.url("/chunked-huge"))

        assert "maximum accepted size" in str(excinfo.value)

    def test_a_body_under_the_cap_is_served(self, server, cap):
        cap(1)

        assert MediaConnector().fetch_image(server.url("/small")) is not None

    def test_a_cap_of_zero_lifts_the_limit(self, server, cap):
        """The operator's escape hatch has to actually let the big body through
        -- to the decoder, which then refuses it as not being an image."""
        cap(0)

        with pytest.raises(ValueError) as excinfo:
            MediaConnector().fetch_image(server.url("/chunked-huge"))

        assert "maximum accepted size" not in str(excinfo.value)
        assert server.body_bytes >= 4 * MIB

    def test_an_oversize_body_is_not_downloaded_twice(self, server, cap, monkeypatch):
        """Retrying a refusal would re-pull the body we just refused, turning
        the cap into an amplifier."""
        cap(1)
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_FETCH_MAX_RETRIES", 3)

        with pytest.raises(APHRODITEUnprocessableEntityError):
            MediaConnector().fetch_image(server.url("/chunked-huge"))

        assert server.requests == ["/chunked-huge"]

    @pytest.mark.asyncio
    async def test_the_async_path_caps_too(self, server, cap):
        """The path that actually serves API traffic."""
        cap(1)

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            await MediaConnector().fetch_image_async(server.url("/chunked-huge"))

        assert "maximum accepted size" in str(excinfo.value)
        assert server.body_bytes < 2 * MIB


class TestDataUrlSizeCap:
    """A `data:` URL spends the same memory; it just arrives already paid for."""

    def test_oversize_data_url_is_refused(self, cap):
        cap(1)
        payload = base64.b64encode(b"\0" * (2 * MIB)).decode()

        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image(f"data:image/png;base64,{payload}")

        assert "maximum accepted size" in str(excinfo.value)

    def test_data_url_under_the_cap_is_served(self, cap):
        cap(1)

        assert MediaConnector().fetch_image(PNG_DATA_URL) is not None


class TestRebinding:
    """The URL check and the connection are two different lookups."""

    @pytest.fixture
    def rebinding(self, monkeypatch):
        """Public on the first answer, loopback on every one after it."""
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", {"remote", "data"})
        answers: list[str] = []

        def fake_getaddrinfo(host, port, *args, **kwargs):
            answers.append(host)
            ip = PUBLIC_IP if len(answers) == 1 else LOOPBACK_IP
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 80))]

        with patch.object(socket, "getaddrinfo", side_effect=fake_getaddrinfo):
            yield answers

    def test_a_name_that_changes_its_answer_is_refused(self, server, rebinding, no_retries):
        """`_host_is_private` approves the public answer and the fetch proceeds;
        the address it then connects to is loopback, which this policy refuses.
        Without a check at connect time the fetch would simply succeed."""
        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            MediaConnector().fetch_image(f"http://rebind.test:{server.port}/small")

        assert "host is not permitted" in str(excinfo.value)
        # Both lookups happened: the pre-flight one and the connecting one.
        assert len(rebinding) >= 2

    @pytest.mark.asyncio
    async def test_the_async_path_is_guarded_too(self, server, rebinding, no_retries):
        with pytest.raises(APHRODITEUnprocessableEntityError) as excinfo:
            await MediaConnector().fetch_image_async(f"http://rebind.test:{server.port}/small")

        assert "host is not permitted" in str(excinfo.value)
        assert len(rebinding) >= 2

    def test_an_address_the_policy_permits_still_fetches(self, server, monkeypatch, no_retries):
        """The control for the two above.

        `private` is the permitted source here and the server is on loopback,
        so the policy has to actually classify the address and allow it --
        unlike the default, where it short-circuits before classifying and
        would pass whether or not the guard worked.
        """
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", {"private", "data"})

        assert MediaConnector().fetch_image(server.url("/small")) is not None

    @pytest.mark.asyncio
    async def test_an_address_the_policy_permits_still_fetches_async(self, server, monkeypatch, no_retries):
        monkeypatch.setattr(envs, "APHRODITE_MEDIA_ALLOWED_SOURCES", {"private", "data"})

        assert await MediaConnector().fetch_image_async(server.url("/small")) is not None
