"""TestClient compatibility wrapper for httpx 0.28+"""

from contextlib import contextmanager

import anyio
import httpx
from starlette.testclient import _TestClientTransport


class TestClient(httpx.Client):
    """TestClient compatible with httpx 0.28+"""

    def __init__(self, app, base_url="http://testserver", **kwargs):
        @contextmanager
        def portal_factory():
            with anyio.from_thread.start_blocking_portal() as portal:
                yield portal

        transport = _TestClientTransport(
            app=app,
            portal_factory=portal_factory,
            app_state={"app": app},
            client=None,  # Required for httpx 0.28+ compatibility
        )
        super().__init__(transport=transport, base_url=base_url, **kwargs)
