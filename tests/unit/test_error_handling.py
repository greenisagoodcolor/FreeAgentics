"""Test error handling in agent operations."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent
from agents.error_handling import (
    ActionSelectionError,
    ErrorHandler,
    ErrorSeverity,
    InferenceError,
    PyMDPError,
    validate_action,
    validate_observation,
)


class TestErrorHandler:
    """Test the ErrorHandler class."""

    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler("test_agent")
        assert handler.agent_id == "test_agent"
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) > 0

    def test_pymdp_error_handling(self):
        """Test PyMDP error classification and handling."""
        handler = ErrorHandler("test_agent")

        pymdp_error = PyMDPError("PyMDP matrix inversion failed")
        recovery_info = handler.handle_error(pymdp_error, "action_selection")

        assert recovery_info["severity"] == ErrorSeverity.HIGH
        assert recovery_info["fallback_action"] == "stay"
        assert recovery_info["can_retry"] is True

    def test_error_retry_limits(self):
        """Test error retry limit enforcement."""
        handler = ErrorHandler("test_agent")

        # Simulate multiple failures
        for i in range(4):  # Max retries is 3
            recovery_info = handler.handle_error(
                PyMDPError(f"Failure {i}"), "pymdp_operation"
            )

        # Should not be able to retry after max attempts
        assert recovery_info["can_retry"] is False

    def test_error_history_tracking(self):
        """Test error history is properly tracked."""
        handler = ErrorHandler("test_agent")

        handler.handle_error(PyMDPError("Error 1"), "op1")
        handler.handle_error(InferenceError("Error 2"), "op2")

        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert len(summary["recent_errors"]) == 2
        assert "PyMDPError" in summary["error_counts"]
        assert "InferenceError" in summary["error_counts"]


class TestObservationValidation:
    """Test observation validation utilities."""

    def test_validate_none_observation(self):
        """Test validation of None observation."""
        result = validate_observation(None)
        assert result["position"] == [0, 0]
        assert result["valid"] is False

    def test_validate_dict_observation(self):
        """Test validation of dictionary observation."""
        obs = {
            "position": [1, 2],
            "temperature": 25.5,
            "invalid_field": object(),
        }
        result = validate_observation(obs)

        assert result["position"] == [1, 2]
        assert result["temperature"] == 25.5
        assert result["valid"] is True
        assert "invalid_field" not in result  # Object filtered out

    def test_validate_non_dict_observation(self):
        """Test validation of non-dictionary observation."""
        result = validate_observation("simple string")
        assert result["observation"] == "simple string"
        assert result["position"] == [0, 0]
        assert result["valid"] is True


class TestActionValidation:
    """Test action validation utilities."""

    def test_validate_string_action(self):
        """Test validation of string actions."""
        valid_actions = ["up", "down", "left", "right", "stay"]

        assert validate_action("up", valid_actions) == "up"
        assert validate_action("invalid", valid_actions) == "stay"
        assert validate_action(None, valid_actions) == "stay"

    def test_validate_numeric_action(self):
        """Test validation of numeric actions."""
        valid_actions = ["up", "down", "left", "right", "stay"]

        assert validate_action(0, valid_actions) == "up"
        assert validate_action(2.7, valid_actions) == "left"  # int(2.7) = 2
        assert validate_action(10, valid_actions) == "up"  # 10 % 5 = 0


@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="PyMDP not available")
class TestAgentErrorHandling:
    """Test error handling in actual agent operations."""

    def test_agent_error_handler_initialization(self):
        """Test that agents are initialized with error handlers."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        assert hasattr(agent, "error_handler")
        assert agent.error_handler.agent_id == "test"

    def test_step_with_invalid_observation(self):
        """Test agent step with invalid observation."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Should handle None observation gracefully
        action = agent.step(None)
        assert action in agent.actions

    def test_step_with_pymdp_failure(self):
        """Test agent step when PyMDP operations fail."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Mock PyMDP agent to raise exceptions
        mock_agent = MagicMock()
        mock_agent.infer_policies.side_effect = Exception("PyMDP failure")
        agent.pymdp_agent = mock_agent

        # Should handle PyMDP failure gracefully
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        action = agent.step(observation)

        # Should get fallback action
        assert action in agent.actions

        # Error should be recorded
        assert len(agent.error_handler.error_history) > 0

    def test_belief_update_error_handling(self):
        """Test belief update error handling."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Mock PyMDP agent to fail on belief update
        mock_agent = MagicMock()
        mock_agent.infer_states.side_effect = Exception("Inference failed")
        agent.pymdp_agent = mock_agent
        agent.current_observation = [0]

        # Should handle belief update failure gracefully
        agent.update_beliefs()  # Should not raise exception

        # Error should be recorded
        error_summary = agent.error_handler.get_error_summary()
        assert error_summary["total_errors"] > 0

    def test_action_selection_error_handling(self):
        """Test action selection error handling."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Mock PyMDP agent to fail on action selection
        mock_agent = MagicMock()
        mock_agent.infer_policies.side_effect = Exception(
            "Policy inference failed"
        )
        agent.pymdp_agent = mock_agent

        # Should handle action selection failure gracefully
        action = agent.select_action()
        assert action in agent.actions

    def test_error_status_reporting(self):
        """Test that error status is included in agent status."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Trigger an error
        agent.error_handler.handle_error(
            PyMDPError("Test error"), "test_operation"
        )

        # Check status includes error information
        status = agent.get_status()
        assert "error_summary" in status
        assert status["error_summary"]["total_errors"] == 1

    def test_concurrent_error_handling(self):
        """Test error handling with multiple agents."""
        agents = [
            BasicExplorerAgent(f"agent_{i}", f"Agent {i}", grid_size=3)
            for i in range(3)
        ]

        for agent in agents:
            agent.start()

            # Mock different failures for each agent
            mock_agent = MagicMock()
            mock_agent.infer_policies.side_effect = Exception(
                f"Failure for {agent.agent_id}"
            )
            agent.pymdp_agent = mock_agent

        # All agents should handle errors independently
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        actions = []

        for agent in agents:
            action = agent.step(observation)
            actions.append(action)

        # All should complete successfully with fallback actions
        assert len(actions) == 3
        assert all(action in agent.actions for action in actions)

        # Each should have recorded their own errors
        for agent in agents:
            assert len(agent.error_handler.error_history) > 0


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_recovery_after_success(self):
        """Test that retry counters reset after successful operations."""
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Simulate failure then success
        mock_agent = MagicMock()
        mock_agent.infer_policies.side_effect = [
            Exception("Failure"),  # First call fails
            (np.array([0.2, 0.8]), None),  # Second call succeeds
        ]
        mock_agent.sample_action.return_value = np.array(1)
        agent.pymdp_agent = mock_agent

        # First step should handle error
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        action1 = agent.step(observation)

        # Reset the mock for success case
        mock_agent.infer_policies.side_effect = None
        mock_agent.infer_policies.return_value = (np.array([0.2, 0.8]), None)

        # Second step should succeed and reset retry counters
        action2 = agent.step(observation)

        # Check that retry counters are reset
        for strategy in agent.error_handler.recovery_strategies.values():
            assert strategy.retry_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
