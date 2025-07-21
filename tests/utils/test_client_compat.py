"""
TestClient compatibility wrapper for httpx 0.28.1+ compatibility.

This module provides a working TestClient implementation that works around
the httpx 0.28.1 compatibility issue where the 'app' parameter was removed
from httpx.Client.__init__().
"""

from typing import Any, Awaitable, Callable, Dict, MutableMapping, Optional

import httpx


class CompatTestClient:
    """
    A TestClient that works with httpx 0.28.1+ by using ASGITransport explicitly.

    This class provides the same interface as starlette.testclient.TestClient
    but works around the compatibility issue with newer httpx versions.
    """

    def __init__(
        self,
        app: Callable[
            [
                MutableMapping[str, Any],
                Callable[[], Awaitable[MutableMapping[str, Any]]],
                Callable[[MutableMapping[str, Any]], Awaitable[None]],
            ],
            Awaitable[None],
        ],
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        **kwargs,
    ):
        """Initialize the compat test client."""
        self.app = app
        self.base_url = base_url
        self.raise_server_exceptions = raise_server_exceptions
        self.root_path = root_path

        # Create ASGI transport
        self.transport = httpx.ASGITransport(app=app)

        # Create httpx client with the transport
        self.client = httpx.Client(
            transport=self.transport, base_url=base_url, **kwargs
        )

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()

    def close(self):
        """Close the client."""
        if hasattr(self, "client"):
            self.client.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make a request."""
        return self.client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            data=data,
            files=files,
            headers=headers,
            cookies=cookies,
            **kwargs,
        )

    def get(self, url: str, **kwargs) -> httpx.Response:
        """Make a GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        """Make a POST request."""
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        """Make a PUT request."""
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        """Make a PATCH request."""
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make a DELETE request."""
        return self.request("DELETE", url, **kwargs)

    def head(self, url: str, **kwargs) -> httpx.Response:
        """Make a HEAD request."""
        return self.request("HEAD", url, **kwargs)

    def options(self, url: str, **kwargs) -> httpx.Response:
        """Make an OPTIONS request."""
        return self.request("OPTIONS", url, **kwargs)


def TestClient(app, **kwargs):
    """
    Factory function that creates a compatible TestClient.

    This can be used as a drop-in replacement for starlette.testclient.TestClient
    or fastapi.testclient.TestClient in environments where httpx 0.28.1+ is installed.
    """
    return CompatTestClient(app, **kwargs)
