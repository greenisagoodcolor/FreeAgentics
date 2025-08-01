"""Test PyMDP Active Inference Engine Setup and Configuration (Task 44.1).

This module tests the PyMDP installation and basic configuration required for
the active inference engine implementation.

Following Nemesis Committee recommendations:
- Kent Beck: TDD with failing tests first
- Jessica Kerr: Comprehensive test observability
- Sindre Sorhus: High-quality validation and error handling
"""

from unittest.mock import patch

import numpy as np
import pytest


# Test PyMDP imports and basic functionality
def test_pymdp_imports():
    """Test that all required PyMDP components can be imported."""
    # Test core PyMDP imports
    import pymdp.maths as maths
    from pymdp import utils
    from pymdp.agent import Agent

    # Verify classes are available
    assert Agent is not None
    assert utils is not None
    assert maths is not None


def test_pymdp_agent_creation():
    """Test basic PyMDP Agent creation with minimal parameters."""
    from pymdp.agent import Agent

    # Simple A matrix (likelihood mapping states to observations)
    A = np.array(
        [
            [0.9, 0.1],  # state 0 -> obs 0 with 0.9 prob, obs 1 with 0.1 prob
            [0.1, 0.9],
        ]
    )  # state 1 -> obs 0 with 0.1 prob, obs 1 with 0.9 prob

    # Simple B matrix (transition mapping state x action -> next state)
    B = np.array(
        [
            [
                [0.8, 0.2],  # from state 0, action 0 -> state 0 with 0.8 prob
                [0.3, 0.7],
            ],  # from state 0, action 1 -> state 0 with 0.3 prob
            [
                [0.2, 0.8],  # from state 1, action 0 -> state 0 with 0.2 prob
                [0.7, 0.3],
            ],
        ]
    )  # from state 1, action 1 -> state 0 with 0.7 prob

    # Create agent with correct PyMDP API
    agent = Agent(
        A=A,
        B=B,
        num_controls=[2],  # Only num_controls is needed, not num_obs/num_states
    )

    assert agent is not None
    assert hasattr(agent, "infer_states")
    assert hasattr(agent, "infer_policies")
    assert hasattr(agent, "sample_action")


def test_pymdp_basic_inference_cycle():
    """Test basic PyMDP inference cycle: observation -> belief update -> action selection."""
    from pymdp.agent import Agent

    # Observation model
    A = np.array([[0.9, 0.1], [0.1, 0.9]])

    # Transition model
    B = np.array([[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]])

    # Initial beliefs (uniform)
    D = np.array([0.5, 0.5])

    agent = Agent(A=A, B=B, D=D, num_controls=[2])

    # Test inference cycle
    observation = [0]  # Observe state 0

    # Infer states (belief update)
    agent.infer_states(observation)

    # Infer policies (Expected Free Energy calculation)
    agent.infer_policies()

    # Sample action
    action = agent.sample_action()

    # Validate results
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert 0 <= action[0] < 2


def test_pymdp_compatibility_adapter_integration():
    """Test integration with existing PyMDPCompatibilityAdapter."""
    from pymdp.agent import Agent

    from agents.pymdp_adapter import PyMDPCompatibilityAdapter

    # Create adapter
    adapter = PyMDPCompatibilityAdapter()

    # Create test agent
    A = np.array([[0.8, 0.2], [0.2, 0.8]])
    B = np.array([[[0.6, 0.4], [0.4, 0.6]], [[0.4, 0.6], [0.6, 0.4]]])

    agent = Agent(A=A, B=B, num_controls=[2])

    # Set up agent state for action sampling
    observation = [1]
    agent.infer_states(observation)
    agent.infer_policies()

    # Test adapter action sampling
    action_int = adapter.sample_action(agent)

    assert isinstance(action_int, int)
    assert 0 <= action_int < 2


def test_gmn_pymdp_adapter_integration():
    """Test integration with existing GMN to PyMDP adapter."""
    from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp

    # Create test GMN model
    gmn_model = {
        "A": [np.array([[0.9, 0.1], [0.1, 0.9]])],
        "B": [np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.2, 0.8], [0.7, 0.3]]])],
        "C": [np.array([1.0, 0.0])],  # Preference for observation 0
        "D": [np.array([0.5, 0.5])],  # Uniform initial beliefs
    }

    # Test adaptation
    adapted = adapt_gmn_to_pymdp(gmn_model)

    assert "A" in adapted
    assert "B" in adapted
    assert "C" in adapted
    assert "D" in adapted

    # Verify single factor extraction
    assert isinstance(adapted["A"], np.ndarray)
    assert isinstance(adapted["B"], np.ndarray)
    assert isinstance(adapted["C"], np.ndarray)
    assert isinstance(adapted["D"], np.ndarray)


def test_pymdp_error_handling():
    """Test error handling for invalid PyMDP configurations."""
    from pymdp.agent import Agent

    # Test non-numpy input (this should raise TypeError)
    with pytest.raises(TypeError):
        A = [[0.5, 0.5], [0.5, 0.5]]  # List instead of numpy array
        B = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])

        Agent(A=A, B=B, num_controls=[2])


def test_pymdp_numerical_stability():
    """Test PyMDP handling of edge cases for numerical stability."""
    from pymdp.agent import Agent

    # Test with matrices close to normalization boundaries
    A = np.array([[0.999, 0.001], [0.001, 0.999]])

    B = np.array([[[0.999, 0.001], [0.001, 0.999]], [[0.001, 0.999], [0.999, 0.001]]])

    agent = Agent(A=A, B=B, num_controls=[2])

    # Test inference with extreme probabilities
    observation = [0]
    agent.infer_states(observation)
    agent.infer_policies()
    action = agent.sample_action()

    assert isinstance(action, np.ndarray)
    assert not np.isnan(action).any()
    assert not np.isinf(action).any()


def test_logging_configuration():
    """Test that logging is properly configured for PyMDP operations."""
    # Capture log output
    with patch("agents.pymdp_adapter.logger") as mock_logger:
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Verify initialization logging
        mock_logger.info.assert_called_with(
            "Initializing PyMDP compatibility adapter with strict type checking"
        )


class TestPyMDPEnvironmentConfiguration:
    """Test PyMDP environment configuration and setup."""

    def test_numpy_compatibility(self):
        """Test NumPy version compatibility with PyMDP."""
        import numpy as np

        # Test numpy array operations required by PyMDP
        a = np.array([[0.5, 0.5], [0.3, 0.7]])

        # Test matrix operations
        normalized = a / a.sum(axis=0)
        assert np.allclose(normalized.sum(axis=0), 1.0)

        # Test array indexing
        assert a[0, 1] == 0.5

        # Test shape operations
        assert a.shape == (2, 2)

    def test_memory_efficiency_setup(self):
        """Test that PyMDP setup works with memory optimization framework."""
        # This will be important for integration with existing memory optimization
        import os

        import psutil
        from pymdp.agent import Agent

        # Measure memory before agent creation
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Create agent
        A = np.array([[0.8, 0.2], [0.2, 0.8]])
        B = np.array([[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]])

        agent = Agent(A=A, B=B, num_controls=[2])

        # Measure memory after
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before

        # Ensure memory usage is reasonable (less than 10MB for simple agent)
        assert memory_used < 10 * 1024 * 1024, f"Agent used {memory_used} bytes"

        # Clean up
        del agent


# Characterization tests as recommended by Michael Feathers
class TestExistingPyMDPBehavior:
    """Characterization tests for existing PyMDP adapter behavior."""

    def test_existing_sample_action_behavior(self):
        """Characterize existing sample_action method behavior."""
        from pymdp.agent import Agent

        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create reproducible agent
        np.random.seed(42)

        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        B = np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])

        agent = Agent(A=A, B=B, num_controls=[2])

        # Set up for action sampling
        agent.infer_states([0])
        agent.infer_policies()

        # Characterize the behavior
        action = adapter.sample_action(agent)

        # Document current behavior
        assert isinstance(action, int)
        assert action in [0, 1]  # Valid action indices

    def test_existing_gmn_adapter_behavior(self):
        """Characterize existing GMN adapter behavior."""
        from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp

        # Test with minimal GMN model
        gmn_minimal = {
            "A": [np.array([[1.0, 0.0], [0.0, 1.0]])],
            "B": [np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])],
        }

        adapted = adapt_gmn_to_pymdp(gmn_minimal)

        # Document current behavior
        assert isinstance(adapted["A"], np.ndarray)
        assert isinstance(adapted["B"], np.ndarray)
        assert adapted.get("C") is None  # No preferences provided
        assert adapted.get("D") is None  # No initial beliefs provided
