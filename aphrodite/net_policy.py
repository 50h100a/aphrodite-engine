# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Connection-time enforcement of an address policy.

The question that decides an SSRF outcome is not "does this hostname resolve
to something allowed" but "is the address this socket is about to connect to
allowed". Those are the same question only if nothing changes in between, and
a DNS record with a one-second TTL is free to answer differently the second
time it is asked. A pre-flight check that resolves a name, approves it, and
then hands the *name* to an HTTP client that resolves it again is guarding a
different lookup from the one that carries the request.

So the policy is applied where the address is finally known and immediately
used: inside the resolver aiohttp calls, and inside the connect path urllib3
calls. Each resolves once and connects to exactly what it resolved, which
closes the window rather than narrowing it.

A policy is a plain ``str -> bool`` over resolved addresses. It is consulted
per connection attempt, so callers can key it off configuration that changes
at runtime. Connections with no policy behave exactly as before -- nothing
here is on the default path.
"""

import socket
import sys
from collections.abc import Callable

import aiohttp
import requests
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver
from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, PoolManager
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.exceptions import (
    ConnectTimeoutError,
    LocationParseError,
    NameResolutionError,
    NewConnectionError,
)
from urllib3.util.connection import _DEFAULT_TIMEOUT, _set_socket_options, allowed_gai_family

AddressPolicy = Callable[[str], bool]
"""Given a resolved IP address, return whether connecting to it is permitted."""


class AddressNotAllowedError(Exception):
    """A resolved address was refused by the connection's address policy.

    Deliberately not an ``OSError`` or ``ConnectionError``: both aiohttp and
    urllib3 translate those into their own connection errors, which the retry
    ladder treats as transient. A refusal is not transient, and retrying it
    would hand an attacker another resolution -- another chance for a
    short-TTL record to answer differently.
    """

    def __init__(self, host: str, address: str) -> None:
        super().__init__(f"Address {address} for host {host!r} is not permitted.")
        self.host = host
        self.address = address


def _reject_disallowed(host: str, addresses: list[str], policy: AddressPolicy) -> None:
    """Refuse the host if *any* resolved address is refused.

    Fails closed rather than filtering: a name that answers with both a
    routable and a non-routable address is the shape a rebinding or
    split-horizon attack takes, not something to quietly connect to the
    acceptable half of.
    """
    for address in addresses:
        if not policy(address):
            raise AddressNotAllowedError(host, address)


def guarded_create_connection(
    address: tuple[str, int],
    timeout: object = _DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
    socket_options: list | None = None,
    *,
    policy: AddressPolicy | None = None,
) -> socket.socket:
    """``urllib3.util.connection.create_connection`` with a policy check.

    Mirrors urllib3's implementation rather than wrapping it, because the
    point is to hold on to the addresses returned by the single
    ``getaddrinfo`` call and connect to those -- wrapping would leave the
    resolution urllib3 performs unguarded.
    """
    host, port = address
    if host.startswith("["):
        host = host.strip("[]")
    err = None

    family = allowed_gai_family()

    try:
        host.encode("idna")
    except UnicodeError:
        raise LocationParseError(f"'{host}', label empty or too long") from None

    infos = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)

    if policy is not None:
        _reject_disallowed(host, [res[4][0] for res in infos], policy)

    for res in infos:
        af, socktype, proto, _canonname, sa = res
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            _set_socket_options(sock, socket_options)
            if timeout is not _DEFAULT_TIMEOUT:
                sock.settimeout(timeout)  # type: ignore[arg-type]
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            err = None
            return sock
        except OSError as e:
            err = e
            if sock is not None:
                sock.close()

    if err is not None:
        try:
            raise err
        finally:
            err = None
    raise OSError("getaddrinfo returns an empty list")


class _GuardedConnectionMixin:
    """Replaces urllib3's connect step with the policy-checked one.

    ``self.host`` is left alone, so the ``Host`` header and the TLS SNI /
    certificate check still use the name the caller asked for -- only the
    address the socket dials is constrained.
    """

    _address_policy: AddressPolicy | None = None

    def _new_conn(self) -> socket.socket:
        try:
            sock = guarded_create_connection(
                (self._dns_host, self.port),  # type: ignore[attr-defined]
                self.timeout,  # type: ignore[attr-defined]
                source_address=self.source_address,  # type: ignore[attr-defined]
                socket_options=self.socket_options,  # type: ignore[attr-defined]
                policy=self._address_policy,
            )
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e  # type: ignore[attr-defined,arg-type]
        except OSError as e:
            if isinstance(e, TimeoutError):
                raise ConnectTimeoutError(
                    self,
                    f"Connection to {self.host} timed out. (connect timeout={self.timeout})",  # type: ignore[attr-defined]
                ) from e
            raise NewConnectionError(self, f"Failed to establish a new connection: {e}") from e  # type: ignore[arg-type]

        sys.audit("http.client.connect", self, self.host, self.port)  # type: ignore[attr-defined]
        return sock


def _guarded_pool_classes(policy: AddressPolicy) -> dict[str, type]:
    http_conn = type(
        "GuardedHTTPConnection",
        (_GuardedConnectionMixin, HTTPConnection),
        {"_address_policy": staticmethod(policy)},
    )
    https_conn = type(
        "GuardedHTTPSConnection",
        (_GuardedConnectionMixin, HTTPSConnection),
        {"_address_policy": staticmethod(policy)},
    )
    return {
        "http": type("GuardedHTTPConnectionPool", (HTTPConnectionPool,), {"ConnectionCls": http_conn}),
        "https": type("GuardedHTTPSConnectionPool", (HTTPSConnectionPool,), {"ConnectionCls": https_conn}),
    }


class _GuardedPoolManager(PoolManager):
    def __init__(self, *args, address_policy: AddressPolicy, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = _guarded_pool_classes(address_policy)


class GuardedHTTPAdapter(HTTPAdapter):
    """A `requests` adapter whose connections honour an address policy."""

    def __init__(self, address_policy: AddressPolicy, **kwargs) -> None:
        # Set before super().__init__, which calls init_poolmanager.
        self._address_policy = address_policy
        super().__init__(**kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = DEFAULT_POOLBLOCK,
        **pool_kwargs,
    ) -> None:
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = _GuardedPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            address_policy=self._address_policy,
            **pool_kwargs,
        )


def mount_guarded_adapter(session: requests.Session, policy: AddressPolicy) -> None:
    """Route a session's http(s) traffic through policy-checked connections."""
    adapter = GuardedHTTPAdapter(policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)


class GuardedResolver(AbstractResolver):
    """aiohttp resolver that refuses addresses the policy rejects.

    aiohttp connects to exactly the addresses a resolver returns, so checking
    them here is checking the connection itself. Redirects get the same
    treatment for free: each hop resolves through this resolver.

    NOTE: this covers names only. aiohttp short-circuits a host that is
    already a literal IP and never calls the resolver for it, and it caches
    resolutions for `ttl_dns_cache`; `_GuardedTCPConnector` below is what
    closes both of those.
    """

    def __init__(self, policy: AddressPolicy) -> None:
        self._policy = policy
        # DefaultResolver binds to the running loop, so it cannot be built
        # until we are inside one.
        self._inner: AbstractResolver | None = None

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        if self._inner is None:
            self._inner = DefaultResolver()

        results = await self._inner.resolve(host, port, family)
        _reject_disallowed(host, [result["host"] for result in results], self._policy)
        return results

    async def close(self) -> None:
        if self._inner is not None:
            await self._inner.close()


class _GuardedTCPConnector(aiohttp.TCPConnector):
    """Applies the policy to every address the connector is about to dial.

    The resolver alone is not enough. aiohttp returns a literal-IP host
    without consulting the resolver at all, and it serves repeat lookups from
    its own DNS cache, so a policy that lives only in the resolver has two
    ways to not run. `_resolve_host` is the one funnel every connection
    passes through, cache hits and literals included.
    """

    def __init__(self, policy: AddressPolicy, **kwargs) -> None:
        super().__init__(resolver=GuardedResolver(policy), **kwargs)
        self._address_policy = policy

    async def _resolve_host(self, host: str, port: int, traces=None) -> list[ResolveResult]:
        results = await super()._resolve_host(host, port, traces=traces)
        _reject_disallowed(host, [result["host"] for result in results], self._address_policy)
        return results


def guarded_tcp_connector(policy: AddressPolicy) -> aiohttp.TCPConnector:
    """A TCPConnector that will not open a socket the policy refuses.

    Fails loudly if the aiohttp internals it hooks have moved, rather than
    quietly returning a connector that enforces less than the caller thinks.
    """
    if not hasattr(aiohttp.TCPConnector, "_resolve_host"):
        raise RuntimeError(
            "aiohttp.TCPConnector._resolve_host is missing; the address policy "
            "cannot be enforced against literal-IP hosts or cached lookups on "
            f"aiohttp {aiohttp.__version__}."
        )
    return _GuardedTCPConnector(policy)
