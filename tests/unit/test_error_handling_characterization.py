"""Characterization tests for agents/error_handling.py to achieve 80%+ coverage.

Following Michael Feathers' principles from "Working Effectively with Legacy Code":
- Write tests that characterize the existing behavior
- Focus on uncovered code paths
- No changes to production code unless fixing bugs
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
import time

from agents.error_handling import (
    ErrorSeverity,
    AgentError,
    PyMDPError,
    InferenceError,
    ActionSelectionError,
    ErrorRecoveryStrategy,
    ErrorHandler,
    with_error_handling,
    safe_pymdp_operation,
    validate_observation,
    validate_action,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_severity_values(self):
        """Test that severity levels have expected values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestAgentError:
    """Test AgentError exception class."""
    
    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = AgentError("Test error")
        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context == {}
        assert isinstance(error.timestamp, datetime)
    
    def test_full_initialization(self):
        """Test error with all parameters."""
        context = {"key": "value", "agent_id": "test123"}
        error = AgentError(
            "Critical failure",
            severity=ErrorSeverity.CRITICAL,
            context=context
        )
        assert str(error) == "Critical failure"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context == context
        assert error.timestamp <= datetime.now()


class TestSpecificErrors:
    """Test specific error types."""
    
    def test_pymdp_error(self):
        """Test PyMDPError inherits correctly."""
        error = PyMDPError("PyMDP failed")
        assert isinstance(error, AgentError)
        assert str(error) == "PyMDP failed"
    
    def test_inference_error(self):
        """Test InferenceError inherits correctly."""
        error = InferenceError("Inference failed")
        assert isinstance(error, AgentError)
        assert str(error) == "Inference failed"
    
    def test_action_selection_error(self):
        """Test ActionSelectionError inherits correctly."""
        error = ActionSelectionError("Action selection failed")
        assert isinstance(error, AgentError)
        assert str(error) == "Action selection failed"


class TestErrorRecoveryStrategy:
    """Test ErrorRecoveryStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ErrorRecoveryStrategy(
            name="Test Strategy",
            fallback_action="default",
            max_retries=5,
            cooldown_seconds=10
        )
        assert strategy.name == "Test Strategy"
        assert strategy.fallback_action == "default"
        assert strategy.max_retries == 5
        assert strategy.cooldown_seconds == 10
        assert strategy.retry_count == 0
        assert strategy.last_error_time is None
    
    def test_can_retry_initially(self):
        """Test can_retry returns True initially."""
        strategy = ErrorRecoveryStrategy("Test", max_retries=3)
        assert strategy.can_retry() is True
    
    def test_can_retry_respects_max_retries(self):
        """Test can_retry respects max retries."""
        strategy = ErrorRecoveryStrategy("Test", max_retries=2, cooldown_seconds=0)
        assert strategy.can_retry() is True
        
        strategy.record_error()
        assert strategy.can_retry() is True
        
        strategy.record_error()
        assert strategy.can_retry() is False  # Reached max_retries
    
    def test_can_retry_respects_cooldown(self):
        """Test can_retry respects cooldown period."""
        strategy = ErrorRecoveryStrategy("Test", cooldown_seconds=1)
        strategy.record_error()
        
        # Immediately after error, should not retry due to cooldown
        assert strategy.can_retry() is False
        
        # After cooldown period, should be able to retry
        time.sleep(1.1)  # Wait for cooldown
        assert strategy.can_retry() is True
    
    def test_record_error(self):
        """Test record_error updates state."""
        strategy = ErrorRecoveryStrategy("Test")
        assert strategy.retry_count == 0
        assert strategy.last_error_time is None
        
        strategy.record_error()
        assert strategy.retry_count == 1
        assert strategy.last_error_time is not None
        assert strategy.last_error_time <= datetime.now()
    
    def test_reset(self):
        """Test reset clears retry state."""
        strategy = ErrorRecoveryStrategy("Test")
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
        """Test error handler initialization."""
        handler = ErrorHandler("agent123")
        assert handler.agent_id == "agent123"
        assert handler.error_history == []
        assert len(handler.recovery_strategies) == 4
        assert "pymdp_failure" in handler.recovery_strategies
        assert "inference_failure" in handler.recovery_strategies
        assert "action_selection_failure" in handler.recovery_strategies
        assert "general_failure" in handler.recovery_strategies
    
    def test_handle_pymdp_error(self):
        """Test handling PyMDP-specific errors."""
        handler = ErrorHandler("agent123")
        error = PyMDPError("PyMDP initialization failed")
        
        recovery_info = handler.handle_error(error, "init_pymdp")
        
        assert recovery_info["can_retry"] is True
        assert recovery_info["fallback_action"] == "stay"
        assert recovery_info["severity"] == ErrorSeverity.HIGH
        assert recovery_info["strategy_name"] == "PyMDP Failure"
        assert recovery_info["retry_count"] == 0  # Count before increment
        
        # Check error was recorded
        assert len(handler.error_history) == 1
        error_record = handler.error_history[0]
        assert error_record["operation"] == "init_pymdp"
        assert error_record["error_type"] == "PyMDPError"
        assert error_record["severity"] == ErrorSeverity.HIGH
    
    def test_handle_inference_error(self):
        """Test handling inference errors."""
        handler = ErrorHandler("agent123")
        error = InferenceError("Inference failed")
        
        recovery_info = handler.handle_error(error, "infer")
        
        assert recovery_info["severity"] == ErrorSeverity.MEDIUM
        assert recovery_info["strategy_name"] == "Inference Failure"
        assert handler.recovery_strategies["inference_failure"].retry_count == 1
    
    def test_handle_action_selection_error(self):
        """Test handling action selection errors."""
        handler = ErrorHandler("agent123")
        error = ActionSelectionError("No valid actions")
        
        recovery_info = handler.handle_error(error, "select_action")
        
        assert recovery_info["severity"] == ErrorSeverity.MEDIUM
        assert recovery_info["strategy_name"] == "Action Selection Failure"
    
    def test_handle_general_error(self):
        """Test handling general errors."""
        handler = ErrorHandler("agent123")
        error = ValueError("Some random error")
        
        recovery_info = handler.handle_error(error, "random_op")
        
        assert recovery_info["severity"] == ErrorSeverity.LOW
        assert recovery_info["strategy_name"] == "General Failure"
    
    def test_handle_error_with_pymdp_keyword(self):
        """Test that errors containing 'pymdp' are classified correctly."""
        handler = ErrorHandler("agent123")
        error = RuntimeError("Failed to initialize pymdp agent")
        
        recovery_info = handler.handle_error(error, "init")
        
        assert recovery_info["severity"] == ErrorSeverity.HIGH
        assert recovery_info["strategy_name"] == "PyMDP Failure"
    
    def test_record_success(self):
        """Test record_success resets all strategies."""
        handler = ErrorHandler("agent123")
        
        # Record some errors first
        handler.recovery_strategies["pymdp_failure"].record_error()
        handler.recovery_strategies["inference_failure"].record_error()
        
        assert handler.recovery_strategies["pymdp_failure"].retry_count == 1
        assert handler.recovery_strategies["inference_failure"].retry_count == 1
        
        # Record success should reset all
        handler.record_success("test_operation")
        
        assert handler.recovery_strategies["pymdp_failure"].retry_count == 0
        assert handler.recovery_strategies["inference_failure"].retry_count == 0
    
    def test_get_error_summary_empty(self):
        """Test error summary when no errors."""
        handler = ErrorHandler("agent123")
        summary = handler.get_error_summary()
        
        assert summary["total_errors"] == 0
        assert summary["recent_errors"] == []
    
    def test_get_error_summary_with_errors(self):
        """Test error summary with multiple errors."""
        handler = ErrorHandler("agent123")
        
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


class TestWithErrorHandlingDecorator:
    """Test with_error_handling decorator."""
    
    def test_successful_operation(self):
        """Test decorator with successful operation."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
            
            @with_error_handling("test_op", fallback_result="fallback")
            def test_method(self, value):
                return f"success_{value}"
        
        agent = TestAgent()
        result = agent.test_method("test")
        assert result == "success_test"
        assert hasattr(agent, "error_handler")
    
    def test_operation_with_error(self):
        """Test decorator with failing operation."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
            
            @with_error_handling("test_op", fallback_result="fallback")
            def test_method(self):
                raise ValueError("Test error")
        
        agent = TestAgent()
        result = agent.test_method()
        assert result == "fallback"
        assert len(agent.error_handler.error_history) == 1
    
    def test_operation_with_retry_and_fallback_method(self):
        """Test decorator with fallback method."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                self.fallback_called = False
            
            @with_error_handling("test_op", fallback_result="default_fallback")
            def test_method(self, value):
                raise ValueError("Test error")
            
            def _fallback_test_method(self, value):
                self.fallback_called = True
                return f"fallback_{value}"
        
        agent = TestAgent()
        result = agent.test_method("test")
        assert result == "fallback_test"
        assert agent.fallback_called is True
    
    def test_operation_with_failed_retry(self):
        """Test decorator when retry also fails."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
            
            @with_error_handling("test_op", fallback_result="final_fallback")
            def test_method(self):
                raise ValueError("Test error")
            
            def _fallback_test_method(self):
                raise RuntimeError("Fallback also failed")
        
        agent = TestAgent()
        result = agent.test_method()
        assert result == "final_fallback"


class TestSafePyMDPOperation:
    """Test safe_pymdp_operation decorator."""
    
    def test_successful_pymdp_operation(self):
        """Test decorator with successful PyMDP operation."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                self.pymdp_agent = Mock()  # Mock PyMDP agent
            
            @safe_pymdp_operation("test_pymdp_op", default_value=0)
            def pymdp_method(self, value):
                return self.pymdp_agent.process(value)
        
        agent = TestAgent()
        agent.pymdp_agent.process.return_value = 42
        
        result = agent.pymdp_method(10)
        assert result == 42
        agent.pymdp_agent.process.assert_called_once_with(10)
    
    def test_pymdp_not_initialized(self):
        """Test decorator when PyMDP agent not initialized."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                # No pymdp_agent attribute
            
            @safe_pymdp_operation("test_pymdp_op", default_value=-1)
            def pymdp_method(self):
                return self.pymdp_agent.process()
        
        agent = TestAgent()
        result = agent.pymdp_method()
        assert result == -1
        assert len(agent.error_handler.error_history) == 1
        assert agent.error_handler.error_history[0]["error_type"] == "PyMDPError"
    
    def test_pymdp_operation_failure(self):
        """Test decorator when PyMDP operation fails."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                self.pymdp_agent = Mock()
            
            @safe_pymdp_operation("test_pymdp_op", default_value="safe_default")
            def pymdp_method(self):
                raise RuntimeError("PyMDP operation failed")
        
        agent = TestAgent()
        result = agent.pymdp_method()
        assert result == "safe_default"
    
    def test_pymdp_with_fallback_method(self):
        """Test PyMDP decorator with fallback method."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                self.pymdp_agent = Mock()
            
            @safe_pymdp_operation("test_pymdp_op", default_value="default")
            def pymdp_method(self, value):
                raise RuntimeError("PyMDP failed")
            
            def _fallback_pymdp_method(self, value):
                return f"fallback_{value}"
        
        agent = TestAgent()
        result = agent.pymdp_method("test")
        assert result == "fallback_test"
    
    def test_pymdp_with_failing_fallback(self):
        """Test PyMDP decorator when fallback also fails."""
        class TestAgent:
            def __init__(self):
                self.agent_id = "test123"
                self.pymdp_agent = Mock()
            
            @safe_pymdp_operation("test_pymdp_op", default_value="final_default")
            def pymdp_method(self):
                raise RuntimeError("PyMDP failed")
            
            def _fallback_pymdp_method(self):
                raise RuntimeError("Fallback also failed")
        
        agent = TestAgent()
        result = agent.pymdp_method()
        assert result == "final_default"


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_observation_none(self):
        """Test validating None observation."""
        result = validate_observation(None)
        assert result == {"position": [0, 0], "valid": False}
    
    def test_validate_observation_dict(self):
        """Test validating dict observation."""
        obs = {
            "position": [5, 10],
            "health": 100,
            "score": 42.5
        }
        result = validate_observation(obs)
        assert result["position"] == [5, 10]
        assert result["valid"] is True
        assert result["health"] == 100
        assert result["score"] == 42.5
    
    def test_validate_observation_dict_missing_position(self):
        """Test validating dict without position."""
        obs = {"health": 100}
        result = validate_observation(obs)
        assert result["position"] == [0, 0]
        assert result["valid"] is True
        assert result["health"] == 100
    
    def test_validate_observation_dict_with_invalid_fields(self):
        """Test that invalid field types are filtered out."""
        obs = {
            "position": [1, 2],
            "valid_int": 42,
            "valid_float": 3.14,
            "valid_str": "test",
            "valid_list": [1, 2, 3],
            "valid_dict": {"key": "value"},
            "invalid_func": lambda x: x,  # Should be filtered out
            "invalid_obj": object()  # Should be filtered out
        }
        result = validate_observation(obs)
        assert "valid_int" in result
        assert "valid_float" in result
        assert "valid_str" in result
        assert "valid_list" in result
        assert "valid_dict" in result
        assert "invalid_func" not in result
        assert "invalid_obj" not in result
    
    def test_validate_observation_non_dict(self):
        """Test validating non-dict observation."""
        result = validate_observation(42)
        assert result == {"observation": 42, "position": [0, 0], "valid": True}
        
        result = validate_observation("test_obs")
        assert result == {"observation": "test_obs", "position": [0, 0], "valid": True}
    
    def test_validate_action_none(self):
        """Test validating None action."""
        result = validate_action(None, ["up", "down", "left", "right", "stay"])
        assert result == "stay"
    
    def test_validate_action_valid_string(self):
        """Test validating valid string action."""
        valid_actions = ["up", "down", "left", "right", "stay"]
        assert validate_action("up", valid_actions) == "up"
        assert validate_action("down", valid_actions) == "down"
        assert validate_action("stay", valid_actions) == "stay"
    
    def test_validate_action_invalid_string(self):
        """Test validating invalid string action."""
        valid_actions = ["up", "down", "left", "right", "stay"]
        result = validate_action("invalid", valid_actions)
        assert result == "stay"
    
    def test_validate_action_numeric(self):
        """Test validating numeric action."""
        valid_actions = ["up", "down", "left", "right", "stay"]
        
        # Test in-range indices
        assert validate_action(0, valid_actions) == "up"
        assert validate_action(1, valid_actions) == "down"
        assert validate_action(4, valid_actions) == "stay"
        
        # Test out-of-range indices (should wrap around)
        assert validate_action(5, valid_actions) == "up"  # 5 % 5 = 0
        assert validate_action(7, valid_actions) == "left"  # 7 % 5 = 2
        
        # Test float values
        assert validate_action(1.7, valid_actions) == "down"  # int(1.7) = 1
    
    def test_validate_action_invalid_type(self):
        """Test validating invalid action type."""
        valid_actions = ["up", "down", "left", "right", "stay"]
        
        # Test with various invalid types
        assert validate_action([], valid_actions) == "stay"
        assert validate_action({}, valid_actions) == "stay"
        assert validate_action(object(), valid_actions) == "stay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])