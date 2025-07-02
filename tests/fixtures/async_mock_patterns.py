"""
Async mock patterns for consistent asynchronous testing.

This module provides standardized patterns for mocking async operations,
ensuring consistent behavior across the test suite.
"""

import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Any, Callable, Coroutine, List, Optional, TypeVar
from unittest.mock import AsyncMock, Mock

T = TypeVar("T")


class AsyncMockFactory:
    """Factory for creating standardized async mocks."""

    @staticmethod
    def create_async_context_manager(
        return_value: Any = None, side_effect: Optional[Exception] = None
    ) -> AsyncMock:
        """Create an async context manager mock.

        Args:
            return_value: Value to return when entering context
            side_effect: Exception to raise if needed

        Returns:
            AsyncMock configured as async context manager
        """
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=return_value)
        mock.__aexit__ = AsyncMock(return_value=None)

        if side_effect:
            mock.__aenter__.side_effect = side_effect

        return mock

    @staticmethod
    def create_async_iterator(items: List[Any]) -> AsyncMock:
        """Create an async iterator mock.

        Args:
            items: List of items to yield

        Returns:
            AsyncMock configured as async iterator
        """

        async def async_generator():
            for item in items:
                yield item

        mock = AsyncMock()
        mock.__aiter__.return_value = async_generator()
        return mock

    @staticmethod
    def create_delayed_response(
        return_value: Any, delay: float = 0.1
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Create an async function that returns after a delay.

        Args:
            return_value: Value to return
            delay: Delay in seconds

        Returns:
            Async function that delays before returning
        """

        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(delay)
            return return_value

        return delayed_response

    @staticmethod
    def create_streaming_response(
        chunks: List[str], delay_between_chunks: float = 0.05
    ) -> Callable[..., AsyncMock]:
        """Create a streaming response mock.

        Args:
            chunks: List of data chunks to stream
            delay_between_chunks: Delay between chunks

        Returns:
            Async generator that yields chunks
        """

        async def stream_generator():
            for chunk in chunks:
                await asyncio.sleep(delay_between_chunks)
                yield chunk

        mock = AsyncMock()
        mock.return_value = stream_generator()
        return mock


class AsyncTestPatterns:
    """Common async test patterns."""

    @staticmethod
    async def with_timeout(
            coro: Coroutine[Any, Any, T], timeout: float = 1.0) -> T:
        """Execute coroutine with timeout.

        Args:
            coro: Coroutine to execute
            timeout: Maximum execution time

        Returns:
            Result of coroutine

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
        """
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    @asynccontextmanager
    async def assert_completes_within(timeout: float = 1.0):
        """Context manager to assert async operation completes within timeout.

        Args:
            timeout: Maximum allowed execution time

        Yields:
            None

        Raises:
            AssertionError: If operation exceeds timeout
        """
        start_time = asyncio.get_event_loop().time()
        try:
            yield
        finally:
            elapsed = asyncio.get_event_loop().time() - start_time
            assert elapsed < timeout, f"Operation took {elapsed}s, expected < {timeout}s"

    @staticmethod
    async def gather_with_errors(
            *coros: Coroutine[Any, Any, Any]) -> List[Any]:
        """Gather multiple coroutines, capturing errors.

        Args:
            *coros: Coroutines to execute

        Returns:
            List of results (successful) or exceptions (failed)
        """
        results = []
        for coro in coros:
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                results.append(e)
        return results

    @staticmethod
    def async_retry(
            max_attempts: int = 3,
            delay: float = 0.1,
            backoff: float = 2.0) -> Callable:
        """Decorator for retrying async functions.

        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier for delays

        Returns:
            Decorated function with retry logic
        """

        def decorator(
            func: Callable[..., Coroutine[Any, Any, T]],
        ) -> Callable[..., Coroutine[Any, Any, T]]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff

                raise last_exception

            return wrapper

        return decorator


class AsyncServiceMocks:
    """Pre-configured async service mocks."""

    @staticmethod
    def create_async_database_session() -> AsyncMock:
        """Create async database session mock.

        Returns:
            AsyncMock configured as database session
        """
        session = AsyncMock()

        # Query methods
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        session.scalars = AsyncMock()

        # Transaction methods
        session.begin = AsyncMockFactory.create_async_context_manager(session)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()

        # Context manager support
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock()

        return session

    @staticmethod
    def create_async_http_client() -> AsyncMock:
        """Create async HTTP client mock.

        Returns:
            AsyncMock configured as HTTP client
        """
        client = AsyncMock()

        # Response mock
        response = AsyncMock()
        response.status_code = 200
        response.json = AsyncMock(return_value={})
        response.text = AsyncMock(return_value="")
        response.raise_for_status = AsyncMock()

        # HTTP methods
        for method in ["get", "post", "put", "patch", "delete"]:
            setattr(client, method, AsyncMock(return_value=response))

        # Context manager support
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock()

        return client

    @staticmethod
    def create_async_websocket() -> AsyncMock:
        """Create async WebSocket connection mock.

        Returns:
            AsyncMock configured as WebSocket
        """
        ws = AsyncMock()

        # Connection state
        ws.closed = False
        ws.close_code = None

        # Methods
        ws.send = AsyncMock()
        ws.recv = AsyncMock(return_value='{"type": "ping"}')
        ws.close = AsyncMock(side_effect=lambda: setattr(ws, "closed", True))
        ws.ping = AsyncMock()
        ws.pong = AsyncMock()

        # Context manager support
        ws.__aenter__ = AsyncMock(return_value=ws)
        ws.__aexit__ = AsyncMock()

        return ws

    @staticmethod
    def create_async_queue(items: Optional[List[Any]] = None) -> AsyncMock:
        """Create async queue mock.

        Args:
            items: Initial items in queue

        Returns:
            AsyncMock configured as async queue
        """
        queue = AsyncMock()
        queue._items = list(items) if items else []

        async def get():
            if queue._items:
                return queue._items.pop(0)
            raise asyncio.QueueEmpty()

        async def put(item):
            queue._items.append(item)

        queue.get = AsyncMock(side_effect=get)
        queue.put = AsyncMock(side_effect=put)
        queue.empty = Mock(side_effect=lambda: len(queue._items) == 0)
        queue.qsize = Mock(side_effect=lambda: len(queue._items))

        return queue


# Convenience functions for common patterns
async def async_return(value: Any) -> Any:
    """Simple async function that returns a value."""
    return value


async def async_raise(exception: Exception) -> None:
    """Simple async function that raises an exception."""
    raise exception


def make_async(value: Any) -> Coroutine[Any, Any, Any]:
    """Convert a value into an async coroutine that returns it."""
    return async_return(value)
