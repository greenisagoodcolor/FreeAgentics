"""Comprehensive tests for agents.error_handling module to achieve high coverage."""

import logging
from datetime import datetime, timedelta
from unittest.mock import Mock

from agents.error_handling import (
    ActionSelectionError,
    AgentError,
    ErrorHandler,
    ErrorRecoveryStrategy,
    ErrorSeverity,
    InferenceError,
    PyMDPError,
    safe_pymdp_operation,
    validate_action,
    validate_observation,
    with_error_handling,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestAgentError:
    """Test AgentError class."""

    def test_agent_error_creation(self):
        """Test AgentError creation with all fields."""
        context = {"agent_id": "123", "action": "update"}
        error = AgentError(
            message="Test error", severity=ErrorSeverity.HIGH, context=context
        )

        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == context
        assert isinstance(error.timestamp, datetime)

    def test_agent_error_defaults(self):
        """Test AgentError with default values."""
        error = AgentError("Simple error")

        assert str(error) == "Simple error"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context == {}
        assert error.timestamp is not None


class TestSpecificErrors:
    """Test specific error types."""

    def test_pymdp_error(self):
        """Test PyMDPError inherits from AgentError."""
        error = PyMDPError(
            "PyMDP failed",
            severity=ErrorSeverity.CRITICAL,
            context={"module": "inference"},
        )

        assert isinstance(error, AgentError)
        assert str(error) == "PyMDP failed"
        assert error.severity == ErrorSeverity.CRITICAL

    def test_inference_error(self):
        """Test InferenceError."""
        error = InferenceError("Inference failed", severity=ErrorSeverity.HIGH)

        assert isinstance(error, AgentError)
        assert error.severity == ErrorSeverity.HIGH

    def test_action_selection_error(self):
        """Test ActionSelectionError."""
        error = ActionSelectionError(
            "Action selection failed",
            context={"available_actions": ["up", "down"]},
        )

        assert isinstance(error, AgentError)
        assert error.context["available_actions"] == ["up", "down"]


class TestErrorRecoveryStrategy:
    """Test ErrorRecoveryStrategy class."""

    def test_initialization(self):
        """Test ErrorRecoveryStrategy initialization."""
        strategy = ErrorRecoveryStrategy(
            name="Test Strategy",
            fallback_action="stay",
            max_retries=5,
            cooldown_seconds=10,
        )

        assert strategy.name == "Test Strategy"
        assert strategy.fallback_action == "stay"
        assert strategy.max_retries == 5
        assert strategy.cooldown_seconds == 10
        assert strategy.retry_count == 0
        assert strategy.last_error_time is None

    def test_can_retry_within_limit(self):
        """Test can_retry when within retry limit."""
        strategy = ErrorRecoveryStrategy(name="Test", max_retries=3, cooldown_seconds=0)

        assert strategy.can_retry() is True

        strategy.record_error()
        assert strategy.can_retry() is True

        strategy.record_error()
        assert strategy.can_retry() is True

    def test_can_retry_exceeded_limit(self):
        """Test can_retry when retry limit exceeded."""
        strategy = ErrorRecoveryStrategy(name="Test", max_retries=2)

        strategy.record_error()
        strategy.record_error()

        assert strategy.can_retry() is False

    def test_can_retry_cooldown(self):
        """Test can_retry respects cooldown period."""
        strategy = ErrorRecoveryStrategy(
            name="Test", max_retries=5, cooldown_seconds=60
        )

        strategy.record_error()

        # Should not be able to retry immediately
        assert strategy.can_retry() is False

        # Simulate time passing
        strategy.last_error_time = datetime.now() - timedelta(seconds=61)
        assert strategy.can_retry() is True

    def test_record_error(self):
        """Test record_error updates state."""
        strategy = ErrorRecoveryStrategy(name="Test")

        assert strategy.retry_count == 0
        assert strategy.last_error_time is None

        strategy.record_error()

        assert strategy.retry_count == 1
        assert strategy.last_error_time is not None

    def test_reset(self):
        """Test reset clears error state."""
        strategy = ErrorRecoveryStrategy(name="Test")

        strategy.record_error()
        strategy.record_error()

        assert strategy.retry_count == 2
        assert strategy.last_error_time is not None

        strategy.reset()

        assert strategy.retry_count == 0
        assert strategy.last_error_time is None


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler("agent-123")

        assert handler.agent_id == "agent-123"
        assert handler.error_history == []
        assert len(handler.recovery_strategies) == 4
        assert "pymdp_failure" in handler.recovery_strategies
        assert "inference_failure" in handler.recovery_strategies
        assert "action_selection_failure" in handler.recovery_strategies
        assert "general_failure" in handler.recovery_strategies

    def test_handle_pymdp_error(self, caplog):
        """Test handling PyMDP error."""
        handler = ErrorHandler("agent-123")
        error = PyMDPError("PyMDP initialization failed")

        with caplog.at_level(logging.ERROR):
            recovery_info = handler.handle_error(error, "initialization")

        assert recovery_info["severity"] == ErrorSeverity.HIGH
        assert recovery_info["strategy_name"] == "PyMDP Failure"
        assert recovery_info["can_retry"] is True
        assert recovery_info["fallback_action"] == "stay"
        assert len(handler.error_history) == 1
        assert "agent-123" in caplog.text

    def test_handle_inference_error(self):
        """Test handling InferenceError."""
        handler = ErrorHandler("agent-123")
        error = InferenceError("Belief update failed")

        recovery_info = handler.handle_error(error, "belief_update")

        assert recovery_info["severity"] == ErrorSeverity.MEDIUM
        assert recovery_info["strategy_name"] == "Inference Failure"

    def test_handle_action_selection_error(self):
        """Test handling ActionSelectionError."""
        handler = ErrorHandler("agent-123")
        error = ActionSelectionError("No valid actions")

        recovery_info = handler.handle_error(error, "action_selection")

        assert recovery_info["severity"] == ErrorSeverity.MEDIUM
        assert recovery_info["strategy_name"] == "Action Selection Failure"

    def test_handle_general_error(self):
        """Test handling general errors."""
        handler = ErrorHandler("agent-123")
        error = RuntimeError("Unknown error")

        recovery_info = handler.handle_error(error, "unknown_operation")

        assert recovery_info["severity"] == ErrorSeverity.LOW
        assert recovery_info["strategy_name"] == "General Failure"

    def test_handle_error_with_pymdp_keyword(self):
        """Test error classification based on error message."""
        handler = ErrorHandler("agent-123")
        error = RuntimeError("PyMDP module not found")

        recovery_info = handler.handle_error(error, "import")

        # Should be classified as PyMDP error due to keyword
        assert recovery_info["severity"] == ErrorSeverity.HIGH
        assert recovery_info["strategy_name"] == "PyMDP Failure"

    def test_record_success(self):
        """Test record_success resets all strategies."""
        handler = ErrorHandler("agent-123")

        # Record some errors first
        handler.handle_error(PyMDPError("Error 1"), "op1")
        handler.handle_error(InferenceError("Error 2"), "op2")

        # Check strategies have recorded errors
        assert handler.recovery_strategies["pymdp_failure"].retry_count > 0
        assert handler.recovery_strategies["inference_failure"].retry_count > 0

        # Record success
        handler.record_success("successful_operation")

        # All strategies should be reset
        for strategy in handler.recovery_strategies.values():
            assert strategy.retry_count == 0
            assert strategy.last_error_time is None

    def test_get_error_summary_empty(self):
        """Test get_error_summary with no errors."""
        handler = ErrorHandler("agent-123")

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 0
        assert summary["recent_errors"] == []

    def test_get_error_summary_with_errors(self):
        """Test get_error_summary with error history."""
        handler = ErrorHandler("agent-123")

        # Add various errors
        handler.handle_error(PyMDPError("Error 1"), "op1")
        handler.handle_error(InferenceError("Error 2"), "op2")
        handler.handle_error(PyMDPError("Error 3"), "op3")

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 3
        assert len(summary["recent_errors"]) == 3
        assert summary["error_counts"]["PyMDPError"] == 2
        assert summary["error_counts"]["InferenceError"] == 1
        assert summary["last_error"]["error_type"] == "PyMDPError"


class TestDecorators:
    """Test decorator functions."""

    def test_with_error_handling_success(self):
        """Test with_error_handling decorator with successful execution."""

        class TestAgent:
            agent_id = "test-123"

            @with_error_handling("test_operation", fallback_result="fallback")
            def test_method(self, x):
                return x * 2

        agent = TestAgent()
        result = agent.test_method(5)

        assert result == 10
        assert hasattr(agent, "error_handler")

    def test_with_error_handling_failure(self, caplog):
        """Test with_error_handling decorator with failure."""

        class TestAgent:
            agent_id = "test-123"

            @with_error_handling("test_operation", fallback_result="fallback")
            def test_method(self):
                raise RuntimeError("Test error")

        agent = TestAgent()

        with caplog.at_level(logging.ERROR):
            result = agent.test_method()

        assert result == "fallback"
        assert "Test error" in caplog.text

    def test_with_error_handling_retry(self):
        """Test with_error_handling with retry logic."""

        class TestAgent:
            agent_id = "test-123"

            @with_error_handling("test_operation", fallback_result="fallback")
            def test_method(self):
                raise RuntimeError("Test error")

            def _fallback_test_method(self):
                return "fallback_method_result"

        agent = TestAgent()
        result = agent.test_method()

        # Should use fallback method
        assert result == "fallback_method_result"

    def test_safe_pymdp_operation_success(self):
        """Test safe_pymdp_operation decorator with success."""

        class TestAgent:
            agent_id = "test-123"
            pymdp_agent = Mock()  # Mock PyMDP agent

            @safe_pymdp_operation("test_pymdp_op", default_value=None)
            def pymdp_method(self, x):
                return x + 10

        agent = TestAgent()
        result = agent.pymdp_method(5)

        assert result == 15

    def test_safe_pymdp_operation_no_agent(self):
        """Test safe_pymdp_operation when PyMDP agent not initialized."""

        class TestAgent:
            agent_id = "test-123"
            # No pymdp_agent attribute

            @safe_pymdp_operation("test_pymdp_op", default_value="default")
            def pymdp_method(self):
                return "should_not_reach"

        agent = TestAgent()
        result = agent.pymdp_method()

        assert result == "default"

    def test_safe_pymdp_operation_with_fallback(self):
        """Test safe_pymdp_operation with fallback method."""

        class TestAgent:
            agent_id = "test-123"
            pymdp_agent = Mock()

            @safe_pymdp_operation("test_pymdp_op", default_value="default")
            def pymdp_method(self):
                raise RuntimeError("PyMDP error")

            def _fallback_pymdp_method(self):
                return "fallback_result"

        agent = TestAgent()
        result = agent.pymdp_method()

        assert result == "fallback_result"


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_observation_none(self):
        """Test validate_observation with None."""
        result = validate_observation(None)

        assert result == {"position": [0, 0], "valid": False}

    def test_validate_observation_dict(self):
        """Test validate_observation with dict."""
        obs = {"position": [1, 2], "health": 100, "energy": 50}

        result = validate_observation(obs)

        assert result["position"] == [1, 2]
        assert result["valid"] is True
        assert result["health"] == 100
        assert result["energy"] == 50

    def test_validate_observation_dict_missing_position(self):
        """Test validate_observation with dict missing position."""
        obs = {"health": 100}

        result = validate_observation(obs)

        assert result["position"] == [0, 0]
        assert result["valid"] is True
        assert result["health"] == 100

    def test_validate_observation_non_dict(self):
        """Test validate_observation with non-dict input."""
        result = validate_observation([1, 2, 3])

        assert result["observation"] == [1, 2, 3]
        assert result["position"] == [0, 0]
        assert result["valid"] is True

    def test_validate_action_none(self):
        """Test validate_action with None."""
        result = validate_action(None, ["up", "down", "left", "right"])

        assert result == "stay"

    def test_validate_action_valid_string(self):
        """Test validate_action with valid string action."""
        result = validate_action("up", ["up", "down", "left", "right"])

        assert result == "up"

    def test_validate_action_invalid_string(self):
        """Test validate_action with invalid string action."""
        result = validate_action("invalid", ["up", "down", "left", "right"])

        assert result == "stay"

    def test_validate_action_numeric(self):
        """Test validate_action with numeric action."""
        valid_actions = ["up", "down", "left", "right"]

        assert validate_action(0, valid_actions) == "up"
        assert validate_action(1, valid_actions) == "down"
        assert validate_action(2, valid_actions) == "left"
        assert validate_action(3, valid_actions) == "right"
        assert validate_action(4, valid_actions) == "up"  # Wraps around

    def test_validate_action_float(self):
        """Test validate_action with float action."""
        result = validate_action(2.7, ["up", "down", "left", "right"])

        assert result == "left"  # int(2.7) = 2

    def test_validate_action_invalid_type(self):
        """Test validate_action with invalid type."""
        result = validate_action({"action": "up"}, ["up", "down", "left", "right"])

        assert result == "stay"
