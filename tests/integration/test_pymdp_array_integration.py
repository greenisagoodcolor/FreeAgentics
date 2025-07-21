"""Integration test for PyMDP numpy array handling in real agent scenarios."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.base_agent import BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent


class TestPyMDPArrayIntegration:
    """Test that PyMDP array handling works in real agent scenarios."""

    def test_basic_explorer_with_array_actions(self):
        """Test BasicExplorerAgent handles PyMDP array responses."""
        agent = BasicExplorerAgent("test_explorer", "Test Explorer", grid_size=5)
        agent.start()

        # Mock PyMDP agent to return various array types
        mock_agent = MagicMock()

        # Test different PyMDP response types
        test_cases = [
            np.array(2),  # 0-d array
            np.array([2]),  # 1-d single element
            np.int32(2),  # numpy scalar
            np.int64(2),  # different numpy scalar
            2,  # regular int
        ]

        for action_response in test_cases:
            mock_agent.sample_action.return_value = action_response
            mock_agent.infer_policies.return_value = (
                np.array([0.1, 0.2, 0.6, 0.1]),
                None,
            )
            agent.pymdp_agent = mock_agent

            # This should not raise any errors
            observation = {
                "position": [2, 2],
                "surroundings": np.zeros((3, 3)),
            }
            action = agent.step(observation)

            # Should get a valid action string
            assert action in agent.actions
            assert action == "left"  # Index 2 maps to "left"

    def test_resource_collector_with_array_actions(self):
        """Test ResourceCollectorAgent handles PyMDP array responses."""
        agent = ResourceCollectorAgent("test_collector", "Test Collector", grid_size=5)
        agent.start()

        # Mock PyMDP agent
        mock_agent = MagicMock()
        mock_agent.sample_action.return_value = np.array([4])  # collect action
        mock_agent.infer_policies.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1, 0.6, 0.1]),
            None,
        )
        agent.pymdp_agent = mock_agent

        # Test step with collect action
        observation = {
            "position": [2, 2],
            "visible_cells": [{"x": 2, "y": 2, "type": "resource", "amount": 5}],
            "current_load": 3,
        }

        action = agent.step(observation)
        assert action == "collect"

    def test_coalition_coordinator_with_array_actions(self):
        """Test CoalitionCoordinatorAgent handles PyMDP array responses."""
        agent = CoalitionCoordinatorAgent(
            "test_coordinator", "Test Coordinator", max_agents=5
        )
        agent.start()

        # Mock PyMDP agent
        mock_agent = MagicMock()
        mock_agent.sample_action.return_value = np.argmax(
            np.array([0.1, 0.1, 0.1, 0.6, 0.1, 0.1])
        )  # coordinate action
        mock_agent.infer_policies.return_value = (
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            None,
        )
        agent.pymdp_agent = mock_agent

        # Test step with coordinate action
        observation = {
            "visible_agents": [
                {"id": "agent1", "position": [1, 1], "status": "active"},
                {"id": "agent2", "position": [3, 3], "status": "active"},
            ],
            "coalition_status": {},
        }

        action = agent.step(observation)
        assert action in agent.action_map.values()

    def test_policy_selection_with_arrays(self):
        """Test that policy selection metrics work with numpy arrays."""
        agent = BasicExplorerAgent("test_policy", "Test Policy", grid_size=5)
        agent.start()

        # Mock PyMDP agent with complex policy posterior
        mock_agent = MagicMock()
        q_pi = np.array([0.05, 0.15, 0.65, 0.10, 0.05])  # Policy posterior
        G = np.array([-2.1, -1.8, -0.5, -1.9, -2.0])  # Expected free energy

        mock_agent.infer_policies.return_value = (q_pi, G)
        mock_agent.sample_action.return_value = np.array(1)  # down action
        agent.pymdp_agent = mock_agent

        observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}
        action = agent.step(observation)

        # Check that metrics were computed correctly
        assert "expected_free_energy" in agent.metrics
        assert "policy_posterior" in agent.metrics
        assert "selected_policy" in agent.metrics

        # Best policy should be index 2 (highest probability)
        assert agent.metrics["selected_policy"] == 2
        assert action == "down"

    def test_free_energy_computation_with_arrays(self):
        """Test free energy computation handles numpy arrays correctly."""
        agent = BasicExplorerAgent("test_fe", "Test Free Energy", grid_size=5)
        agent.start()

        # Set up mock PyMDP agent with belief state
        mock_agent = MagicMock()

        # Mock belief state (posterior over states)
        qs = np.array([0.1, 0.3, 0.4, 0.2])  # Beliefs over 4 states
        mock_agent.qs = [qs]  # PyMDP expects list of factors

        # Mock observation model
        A = np.array(
            [
                [0.8, 0.1, 0.05, 0.05],  # P(empty|state)
                [0.1, 0.7, 0.1, 0.1],  # P(obstacle|state)
                [0.05, 0.1, 0.8, 0.05],  # P(goal|state)
                [0.03, 0.07, 0.03, 0.7],  # P(agent|state)
                [0.02, 0.03, 0.02, 0.1],  # P(out_of_bounds|state)
            ]
        )
        mock_agent.A = [A]

        # Mock prior
        D = np.array([0.25, 0.25, 0.25, 0.25])
        mock_agent.D = [D]

        agent.pymdp_agent = mock_agent
        agent.current_observation = [0]  # Observed "empty"

        # This should not raise any errors
        fe_components = agent.compute_free_energy()

        assert "total_free_energy" in fe_components
        assert "accuracy" in fe_components
        assert "complexity" in fe_components
        assert isinstance(fe_components["total_free_energy"], float)


class TestEdgeCases:
    """Test edge cases in PyMDP array handling."""

    def test_malformed_pymdp_responses(self):
        """Test handling of malformed PyMDP responses."""
        agent = BasicExplorerAgent("test_edge", "Test Edge Cases", grid_size=5)
        agent.start()

        # Mock PyMDP agent that returns problematic values
        mock_agent = MagicMock()

        # Test various edge cases
        edge_cases = [
            np.array([]),  # Empty array
            np.array([[1, 2]]),  # 2D array
            None,  # None value
            "invalid",  # String
        ]

        for edge_case in edge_cases:
            mock_agent.sample_action.return_value = edge_case
            mock_agent.infer_policies.return_value = (np.array([1.0]), None)
            agent.pymdp_agent = mock_agent

            observation = {
                "position": [2, 2],
                "surroundings": np.zeros((3, 3)),
            }

            # Should handle gracefully and fall back to non-PyMDP action selection
            action = agent.step(observation)
            assert action in agent.actions

    def test_concurrent_agent_operations(self):
        """Test that multiple agents can operate concurrently without array issues."""
        agents = [
            BasicExplorerAgent(f"explorer_{i}", f"Explorer {i}", grid_size=3)
            for i in range(3)
        ]

        for i, agent in enumerate(agents):
            agent.start()

            # Mock each agent's PyMDP with different response types
            mock_agent = MagicMock()
            mock_agent.sample_action.return_value = np.array(i % 5)  # Different actions
            mock_agent.infer_policies.return_value = (np.ones(5) / 5, None)
            agent.pymdp_agent = mock_agent

        # Run all agents simultaneously
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        actions = []

        for agent in agents:
            action = agent.step(observation)
            actions.append(action)

        # All should complete successfully
        assert len(actions) == 3
        assert all(action in agent.actions for action in actions)


if __name__ == "__main__":
    pytest.main([__file__])
