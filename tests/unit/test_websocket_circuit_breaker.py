"""Unit tests for WebSocket circuit breaker implementation."""

import asyncio
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.append("/home/green/FreeAgentics")

from websocket.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenException,
    CircuitState,
)


@pytest.mark.slow
class TestCircuitBreakerConfig(unittest.TestCase):
    """Test circuit breaker configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.success_threshold, 3)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.half_open_max_calls, 3)

    def test_config_validation(self):
        """Test configuration validation."""
        # Values must be positive
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(timeout=-1)
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(success_threshold=-1)


@pytest.mark.slow
class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Test circuit breaker functionality."""

    def setUp(self):
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,
            half_open_max_calls=2,
        )
        self.breaker = CircuitBreaker("test-breaker", self.config)

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)
        self.assertIsNone(self.breaker.last_failure_time)

    def test_successful_calls(self):
        """Test successful calls don't open circuit."""
        for _ in range(10):
            result = self.breaker.call(lambda: "success")
            self.assertEqual(result, "success")

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)

    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""

        def failing_call():
            raise Exception("Test failure")

        # First failures don't open circuit
        for i in range(self.config.failure_threshold - 1):
            with self.assertRaises(Exception):
                self.breaker.call(failing_call)
            self.assertEqual(self.breaker.state, CircuitState.CLOSED)

        # Next failure opens circuit
        with self.assertRaises(Exception):
            self.breaker.call(failing_call)
        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_open_circuit_rejects_calls(self):
        """Test open circuit rejects calls immediately."""

        # Open the circuit
        def failing_call():
            raise Exception("Test failure")

        for _ in range(self.config.failure_threshold):
            with self.assertRaises(Exception):
                self.breaker.call(failing_call)

        # Now calls should be rejected
        with self.assertRaises(CircuitOpenException) as ctx:
            self.breaker.call(lambda: "test")

        self.assertIn("is OPEN", str(ctx.exception))

    def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""

        # Open the circuit
        def failing_call():
            raise Exception("Test failure")

        for _ in range(self.config.failure_threshold):
            with self.assertRaises(Exception):
                self.breaker.call(failing_call)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Wait for timeout
        time.sleep(self.config.timeout + 0.1)

        # Circuit should allow a test call (half-open)
        self.assertTrue(self.breaker.should_allow_request())

        # Make a successful call
        result = self.breaker.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

    def test_half_open_to_closed_transition(self):
        """Test successful calls in half-open state close circuit."""
        # Open and transition to half-open
        self._open_circuit()
        time.sleep(self.config.timeout + 0.1)

        # Make successful calls to close circuit
        for i in range(self.config.success_threshold):
            result = self.breaker.call(lambda: f"success-{i}")
            self.assertEqual(result, f"success-{i}")

            if i < self.config.success_threshold - 1:
                self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)
            else:
                self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_half_open_to_open_transition(self):
        """Test failure in half-open state reopens circuit."""
        # Open and transition to half-open
        self._open_circuit()
        time.sleep(self.config.timeout + 0.1)

        # Make a successful call (enters half-open)
        self.breaker.call(lambda: "success")
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Failure should reopen
        with self.assertRaises(Exception):
            self.breaker.call(lambda: (_ for _ in ()).throw(Exception("Test")))

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_half_open_max_calls_limit(self):
        """Test half-open state respects max calls limit."""
        # Create a circuit breaker with higher success threshold
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=5,  # Higher than half_open_max_calls
            timeout=1.0,
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker("test-limit", config)

        # Open the circuit
        def failing_call():
            raise Exception("Test failure")

        for _ in range(config.failure_threshold):
            with self.assertRaises(Exception):
                breaker.call(failing_call)

        # Wait and transition to half-open
        time.sleep(config.timeout + 0.1)

        # Make max allowed calls
        for i in range(config.half_open_max_calls):
            breaker.call(lambda: "success")

        # Next call should be rejected because we hit the limit
        with self.assertRaises(CircuitOpenException):
            breaker.call(lambda: "success")

    async def test_async_call_wrapper(self):
        """Test circuit breaker with async calls."""

        async def async_success():
            await asyncio.sleep(0.01)
            return "async success"

        async def async_failure():
            await asyncio.sleep(0.01)
            raise Exception("Async failure")

        # Test successful async call
        result = await self.breaker.async_call(async_success)
        self.assertEqual(result, "async success")

        # Test failed async calls open circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(Exception):
                await self.breaker.async_call(async_failure)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Test open circuit rejects async calls
        with self.assertRaises(CircuitOpenException):
            await self.breaker.async_call(async_success)

    def test_get_status(self):
        """Test getting circuit breaker status."""
        status = self.breaker.get_status()

        self.assertEqual(status["name"], "test-breaker")
        self.assertEqual(status["state"], CircuitState.CLOSED)
        self.assertEqual(status["failure_count"], 0)
        self.assertEqual(status["success_count"], 0)
        self.assertIsNone(status["last_failure_time"])

        # Open circuit and check status
        self._open_circuit()
        status = self.breaker.get_status()

        self.assertEqual(status["state"], CircuitState.OPEN)
        self.assertEqual(
            status["failure_count"], self.config.failure_threshold
        )
        self.assertIsNotNone(status["last_failure_time"])

    def test_reset(self):
        """Test resetting circuit breaker."""
        # Open circuit
        self._open_circuit()
        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Reset
        self.breaker.reset()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)
        self.assertIsNone(self.breaker.last_failure_time)

    def test_custom_exceptions(self):
        """Test handling specific exceptions differently."""
        # Configure to ignore certain exceptions
        config = CircuitBreakerConfig(
            failure_threshold=3, excluded_exceptions=(ValueError,)
        )
        breaker = CircuitBreaker("test", config)

        # ValueError should not count as failure
        for _ in range(5):
            with self.assertRaises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Test")))

        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertEqual(breaker.failure_count, 0)

        # Other exceptions should count
        for _ in range(config.failure_threshold):
            with self.assertRaises(RuntimeError):
                breaker.call(
                    lambda: (_ for _ in ()).throw(RuntimeError("Test"))
                )

        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_listeners(self):
        """Test state change listeners."""
        state_changes = []

        def on_state_change(old_state, new_state, breaker_name):
            state_changes.append((old_state, new_state, breaker_name))

        self.breaker.add_listener(on_state_change)

        # Open circuit
        self._open_circuit()

        # Should have state change
        self.assertEqual(len(state_changes), 1)
        self.assertEqual(
            state_changes[0],
            (CircuitState.CLOSED, CircuitState.OPEN, "test-breaker"),
        )

        # Transition to half-open
        time.sleep(self.config.timeout + 0.1)
        self.breaker.call(lambda: "success")

        self.assertEqual(len(state_changes), 2)
        self.assertEqual(
            state_changes[1],
            (CircuitState.OPEN, CircuitState.HALF_OPEN, "test-breaker"),
        )

    def _open_circuit(self):
        """Helper to open the circuit breaker."""

        def failing_call():
            raise Exception("Test failure")

        for _ in range(self.config.failure_threshold):
            with self.assertRaises(Exception):
                self.breaker.call(failing_call)


if __name__ == "__main__":
    unittest.main()
