"""Type stubs for aiohttp library."""

from typing import Any, AsyncContextManager, Dict, Optional

class ClientResponse:
    status: int

    async def __aenter__(self) -> "ClientResponse": ...
    async def __aexit__(self, *args: Any) -> None: ...

class ClientSession:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    async def __aenter__(self) -> "ClientSession": ...
    async def __aexit__(self, *args: Any) -> None: ...
    def post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> AsyncContextManager[ClientResponse]: ...
