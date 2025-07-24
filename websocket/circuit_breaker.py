"""
Circuit Breaker Pattern Implementation for WebSocket Connections

Provides fault tolerance and automatic recovery for WebSocket operations
using the circuit breaker pattern.
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenException(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, breaker_name: str, last_failure: Optional[datetime] = None):
        self.breaker_name = breaker_name
        self.last_failure = last_failure
        super().__init__(f"Circuit breaker '{breaker_name}' is OPEN")


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """
        Initialize circuit breaker configuration.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout: Seconds before attempting to close open circuit
            half_open_max_calls: Max calls allowed in half-open state
            excluded_exceptions: Exceptions that don't count as failures
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")

        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or ()


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Monitors failures and prevents cascading failures by temporarily
    blocking calls when a failure threshold is exceeded.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker instance
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._listeners: List[Callable] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_timeout()
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count."""
        return self._success_count

    @property
    def last_failure_time(self) -> Optional[datetime]:
        """Get time of last failure."""
        if self._last_failure_time:
            return datetime.fromtimestamp(self._last_failure_time)
        return None

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        self._check_timeout()

        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.config.timeout
            ):
                return True  # Will transition to half-open on next call
            return False
        else:  # HALF_OPEN
            return self._half_open_calls < self.config.half_open_max_calls

    def call(self, func: Callable[[], Any]) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute

        Returns:
            Result of the function

        Raises:
            CircuitOpenException: If circuit is open
            Exception: If function raises an exception
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.config.timeout
            ):
                self._transition_to(CircuitState.HALF_OPEN)

        if not self.should_allow_request():
            raise CircuitOpenException(self.name, self.last_failure_time)

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            if not isinstance(e, self.config.excluded_exceptions):
                self._record_failure()
            raise

    async def async_call(self, func: Callable[[], Any]) -> Any:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute

        Returns:
            Result of the function

        Raises:
            CircuitOpenException: If circuit is open
            Exception: If function raises an exception
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.config.timeout
            ):
                self._transition_to(CircuitState.HALF_OPEN)

        if not self.should_allow_request():
            raise CircuitOpenException(self.name, self.last_failure_time)

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = await func()
            self._record_success()
            return result
        except Exception as e:
            if not isinstance(e, self.config.excluded_exceptions):
                self._record_failure()
            raise

    def _record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self._failure_count = 0

    def _record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            self._transition_to(CircuitState.OPEN)

    def _check_timeout(self):
        """Check if timeout has expired for open circuit."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            if time.time() - self._last_failure_time >= self.config.timeout:
                # Don't transition directly to half-open, wait for next call
                pass

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._half_open_calls = 0

        # Notify listeners
        self._notify_listeners(old_state, new_state)

        logger.info(
            f"Circuit breaker '{self.name}' transitioned from {old_state.value} to {new_state.value}"
        )

    def add_listener(self, listener: Callable[[CircuitState, CircuitState, str], None]):
        """Add a state change listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[CircuitState, CircuitState, str], None]):
        """Remove a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, old_state: CircuitState, new_state: CircuitState):
        """Notify all listeners of state change."""
        for listener in self._listeners:
            try:
                listener(old_state, new_state, self.name)
            except Exception as e:
                logger.error(f"Error notifying circuit breaker listener: {e}")

    def reset(self):
        """Reset the circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_status(self) -> dict:
        """Get current status of the circuit breaker."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self._half_open_calls,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
                "half_open_max_calls": self.config.half_open_max_calls,
            },
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        return self._breakers.copy()

    def get_status_all(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# Export for database module
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitOpenException",
    "CircuitState",
    "circuit_breaker_registry"
]
