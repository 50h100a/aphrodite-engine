# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Address policy enforced where the socket is opened, not where the URL is read.

The property under test is narrow and specific: whatever the policy is asked
about must be the address the connection actually uses. A check that runs
against its own separate lookup is not this, however correct that lookup was
at the time.
"""

import ipaddress
import socket
from unittest.mock import patch

import aiohttp
import pytest
import requests

from aphrodite.net_policy import (
    AddressNotAllowedError,
    guarded_tcp_connector,
    mount_guarded_adapter,
)

# A port nothing listens on: every test here must fail on the policy, before
# a connection is attempted. If the policy ever stops firing, the test fails
# with a connection error rather than passing by accident.
DEAD_PORT = 9

PUBLIC_IP = "93.184.216.34"
LOOPBACK_IP = "127.0.0.1"


def no_loopback(address: str) -> bool:
    return not ipaddress.ip_address(address).is_loopback


def addrinfo(ip: str, port: int) -> list[tuple]:
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


@pytest.fixture
def rebinding():
    """A name that answers publicly once, then points at loopback.

    The first answer is what a pre-flight check would see and approve; every
    answer after it is what a connection would get. A record with a short TTL
    is entitled to behave exactly like this.
    """
    answers = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        answers.append(host)
        ip = PUBLIC_IP if len(answers) == 1 else LOOPBACK_IP
        return addrinfo(ip, port or 80)

    with patch.object(socket, "getaddrinfo", side_effect=fake_getaddrinfo):
        yield answers


class TestSyncPath:
    def test_literal_ip_is_refused(self):
        session = requests.Session()
        mount_guarded_adapter(session, no_loopback)

        with pytest.raises(AddressNotAllowedError) as excinfo:
            session.get(f"http://{LOOPBACK_IP}:{DEAD_PORT}/x", timeout=5)

        assert excinfo.value.address == LOOPBACK_IP

    def test_name_resolving_to_a_refused_address_is_refused(self):
        session = requests.Session()
        mount_guarded_adapter(session, no_loopback)

        with pytest.raises(AddressNotAllowedError) as excinfo:
            session.get(f"http://localhost:{DEAD_PORT}/x", timeout=5)

        assert excinfo.value.host == "localhost"
        assert ipaddress.ip_address(excinfo.value.address).is_loopback

    def test_the_connect_lookup_is_the_one_that_counts(self, rebinding):
        """The pre-flight answer being acceptable does not make the connection
        acceptable -- only the address dialled does."""
        session = requests.Session()
        mount_guarded_adapter(session, no_loopback)

        # Stand in for a pre-flight check: it sees the public answer and is
        # satisfied, exactly as `_host_is_private` would be.
        first = socket.getaddrinfo("rebind.test", 80)
        assert first[0][4][0] == PUBLIC_IP

        with pytest.raises(AddressNotAllowedError) as excinfo:
            session.get(f"http://rebind.test:{DEAD_PORT}/x", timeout=5)

        assert excinfo.value.address == LOOPBACK_IP

    def test_a_permitted_address_connects(self):
        """Sanity: the guard refuses on policy, not on everything.

        Nothing listens on DEAD_PORT, so the dial gets refused by the kernel.
        Reaching *that* error is the pass condition -- the policy was asked
        and let the connection proceed.
        """
        seen: list[str] = []

        def allow_all(address: str) -> bool:
            seen.append(address)
            return True

        session = requests.Session()
        mount_guarded_adapter(session, allow_all)

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            session.get(f"http://{LOOPBACK_IP}:{DEAD_PORT}/x", timeout=5)

        assert not isinstance(excinfo.value, AddressNotAllowedError)
        assert seen == [LOOPBACK_IP]


@pytest.mark.asyncio
class TestAsyncPath:
    async def test_literal_ip_is_refused(self):
        """aiohttp never asks a resolver about a literal IP, so a policy that
        lived only in the resolver would not run here at all."""
        async with aiohttp.ClientSession(connector=guarded_tcp_connector(no_loopback)) as session:
            with pytest.raises(AddressNotAllowedError) as excinfo:
                await session.get(
                    f"http://{LOOPBACK_IP}:{DEAD_PORT}/x",
                    timeout=aiohttp.ClientTimeout(total=5),
                )

        assert excinfo.value.address == LOOPBACK_IP

    async def test_name_resolving_to_a_refused_address_is_refused(self):
        async with aiohttp.ClientSession(connector=guarded_tcp_connector(no_loopback)) as session:
            with pytest.raises(AddressNotAllowedError) as excinfo:
                await session.get(
                    f"http://localhost:{DEAD_PORT}/x",
                    timeout=aiohttp.ClientTimeout(total=5),
                )

        assert ipaddress.ip_address(excinfo.value.address).is_loopback

    async def test_the_connect_lookup_is_the_one_that_counts(self, rebinding):
        async with aiohttp.ClientSession(connector=guarded_tcp_connector(no_loopback)) as session:
            first = socket.getaddrinfo("rebind.test", 80)
            assert first[0][4][0] == PUBLIC_IP

            with pytest.raises(AddressNotAllowedError) as excinfo:
                await session.get(
                    f"http://rebind.test:{DEAD_PORT}/x",
                    timeout=aiohttp.ClientTimeout(total=5),
                )

        assert excinfo.value.address == LOOPBACK_IP

    async def test_a_permitted_address_connects(self):
        seen: list[str] = []

        def allow_all(address: str) -> bool:
            seen.append(address)
            return True

        async with aiohttp.ClientSession(connector=guarded_tcp_connector(allow_all)) as session:
            with pytest.raises(aiohttp.ClientConnectorError):
                await session.get(
                    f"http://{LOOPBACK_IP}:{DEAD_PORT}/x",
                    timeout=aiohttp.ClientTimeout(total=5),
                )

        assert seen == [LOOPBACK_IP]
