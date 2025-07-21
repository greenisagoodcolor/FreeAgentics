"""
Cross-Agent PyMDP Error Scenario Validation

Tests comprehensive error handling across all agent types
to ensure production robustness.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from agents.base_agent import BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.pymdp_error_handling import PyMDPErrorHandler, PyMDPErrorType
from agents.resource_collector import ResourceCollectorAgent

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class TestPyMDPErrorScenarios:
    """Test PyMDP error handling across all agent types."""

    def test_basic_explorer_numpy_edge_cases(self):
        """Test BasicExplorerAgent handles numpy edge cases."""
        agent = BasicExplorerAgent("test-explorer", "Explorer", grid_size=3)

        # Test 0-dimensional array action
        with patch.object(agent.pymdp_agent, "sample_action", return_value=np.array(2)):
            action = agent.select_action()
            assert action in ["up", "down", "left", "right", "stay"]

        # Test single-element array action
        with patch.object(agent.pymdp_agent, "sample_action", return_value=np.array([3])):
            action = agent.select_action()
            assert action in ["up", "down", "left", "right", "stay"]

        # Test multi-dimensional array (should take first element)
        with patch.object(
            agent.pymdp_agent,
            "sample_action",
            return_value=np.array([[1, 2], [3, 4]]),
        ):
            action = agent.select_action()
            assert action in ["up", "down", "left", "right", "stay"]

    def test_resource_collector_matrix_errors(self):
        """Test ResourceCollectorAgent handles matrix dimension errors."""
        agent = ResourceCollectorAgent("test-collector", "Collector", grid_size=3)

        # Test invalid observation index
        observation = {"cell_type": "unknown_type"}
        agent.perceive(observation)  # Should not crash

        # Test state inference with wrong observation shape
        if agent.pymdp_agent:
            # Mock invalid observation
            with patch.object(
                agent.pymdp_agent,
                "infer_states",
                side_effect=ValueError("Invalid observation shape"),
            ):
                agent.perceive({"cell_type": "resource"})
                # Should handle error gracefully

        # Test action selection with inference failure
        with patch.object(
            agent.pymdp_agent,
            "infer_policies",
            side_effect=RuntimeError("Inference failed"),
        ):
            action = agent.select_action()
            assert action in agent.action_map.values()  # Should use fallback

    def test_coalition_coordinator_state_errors(self):
        """Test CoalitionCoordinatorAgent handles state transition errors."""
        agent = CoalitionCoordinatorAgent("test-coordinator", "Coordinator", max_agents=4)

        # Test with invalid coalition state
        observation = {
            "visible_agents": [{"id": "agent1"}, {"id": "agent2"}],
            "coalition_status": {"invalid_coalition": {"status": "error"}},
        }
        agent.perceive(observation)  # Should handle gracefully

        # Test action selection with policy inference failure
        if agent.pymdp_agent:
            with patch.object(
                agent.pymdp_agent,
                "infer_policies",
                side_effect=ValueError("Matrix dimension mismatch"),
            ):
                action = agent.select_action()
                assert action in agent.action_map.values()  # Should use fallback

    def test_belief_update_errors_all_agents(self):
        """Test belief update error handling across all agent types."""
        agents = [
            BasicExplorerAgent("explorer", "Explorer", grid_size=3),
            ResourceCollectorAgent("collector", "Collector", grid_size=3),
            CoalitionCoordinatorAgent("coordinator", "Coordinator", max_agents=3),
        ]

        for agent in agents:
            if agent.pymdp_agent:
                # Test belief update with invalid state posterior
                with patch.object(agent.pymdp_agent, "qs", None):
                    agent.update_beliefs()  # Should not crash

                # Test belief update with NaN values
                with patch.object(agent.pymdp_agent, "qs", [np.array([np.nan, 0.5, 0.5])]):
                    agent.update_beliefs()  # Should handle NaN gracefully

    def test_free_energy_computation_errors(self):
        """Test free energy computation error handling."""
        agents = [
            BasicExplorerAgent("explorer", "Explorer", grid_size=3),
            ResourceCollectorAgent("collector", "Collector", grid_size=3),
            CoalitionCoordinatorAgent("coordinator", "Coordinator", max_agents=3),
        ]

        for agent in agents:
            # Test with no PyMDP agent
            agent.pymdp_agent = None
            fe = agent.compute_free_energy()
            assert isinstance(fe, dict)

            # Test with PyMDP but missing components
            agent._initialize_pymdp()
            if agent.pymdp_agent:
                with patch.object(agent.pymdp_agent, "qs", None):
                    fe = agent.compute_free_energy()
                    assert isinstance(fe, dict)

    def test_error_handler_integration(self):
        """Test PyMDPErrorHandler integration with agents."""
        handler = PyMDPErrorHandler("test-agent")

        # Test numpy conversion errors
        def failing_numpy_op():
            return np.array([1, 2, 3])[10]  # Index out of bounds

        success, result, error = handler.safe_execute(
            "numpy_operation", failing_numpy_op, lambda: "fallback_value"
        )

        assert not success
        assert result == "fallback_value"
        assert error is not None
        assert error.error_type == PyMDPErrorType.NUMPY_CONVERSION

    def test_matrix_validation_errors(self):
        """Test matrix validation error handling."""
        from agents.pymdp_error_handling import validate_pymdp_matrices

        # Test invalid A matrix (wrong dimensions)
        A = np.array([[0.5, 0.5]])  # Should be (num_obs, num_states)
        B = np.zeros((2, 2, 2))
        C = np.array([1.0, 0.0])
        D = np.array([0.5, 0.5])

        is_valid, msg = validate_pymdp_matrices(A, B, C, D)
        assert not is_valid
        assert "dimensions" in msg

        # Test non-normalized matrix
        A_invalid = np.array([[0.3, 0.3], [0.3, 0.3]])  # Columns don't sum to 1
        is_valid, msg = validate_pymdp_matrices(A_invalid, B, C, D)
        assert not is_valid

    def test_concurrent_agent_errors(self):
        """Test error handling in concurrent multi-agent scenarios."""
        from agents.optimized_threadpool_manager import (
            OptimizedThreadPoolManager,
        )

        manager = OptimizedThreadPoolManager(initial_workers=4)

        # Create agents with potential errors
        agents = [
            BasicExplorerAgent(f"explorer-{i}", f"Explorer-{i}", grid_size=3) for i in range(3)
        ]

        # Register agents
        for agent in agents:
            manager.register_agent(agent.agent_id, agent)

        # Create observations that might cause errors
        observations = {
            "explorer-0": {
                "position": [1, 1],
                "surroundings": np.array([[0, 0], [0, 0]]),
            },  # Wrong shape
            "explorer-1": {
                "position": [5, 5],
                "surroundings": None,
            },  # Out of bounds
            "explorer-2": {"invalid_key": "invalid_value"},  # Missing expected keys
        }

        # Execute with error handling
        results = manager.step_all_agents(observations, timeout=2.0)

        # Verify error handling
        for agent_id, result in results.items():
            assert hasattr(result, "success")
            assert hasattr(result, "error")
            # At least some should handle errors gracefully

        manager.shutdown()

    def test_edge_case_action_selection(self):
        """Test action selection edge cases."""
        agent = BasicExplorerAgent("test", "Test", grid_size=3)

        # Test with empty action map
        original_map = agent.action_map
        agent.action_map = {}

        action = agent.select_action()
        assert action == "stay"  # Default fallback

        agent.action_map = original_map

        # Test with out-of-range action index
        if agent.pymdp_agent:
            with patch.object(agent.pymdp_agent, "sample_action", return_value=99):
                action = agent.select_action()
                assert action == "stay"  # Should handle gracefully

    def test_error_recovery_and_logging(self):
        """Test error recovery and proper logging."""
        import io

        agent = ResourceCollectorAgent("test", "Test", grid_size=3)

        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("agents.resource_collector")
        logger.addHandler(handler)

        # Trigger various errors
        with patch.object(
            agent.pymdp_agent,
            "infer_policies",
            side_effect=Exception("Test error"),
        ):
            agent.select_action()

        # Check logs contain error info
        log_output = log_capture.getvalue()
        assert "Test error" in log_output or "action selection failed" in log_output

        logger.removeHandler(handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
