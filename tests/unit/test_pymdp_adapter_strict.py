"""Test PyMDP compatibility adapter with ZERO fallbacks and strict type checking.

This module follows TDD principles to create an API compatibility adapter
that translates between PyMDP's actual API behavior and expected behavior
with ZERO fallbacks and strict type checking.

Task 1.3: Create API compatibility adapter with strict type checking
"""

from unittest.mock import Mock

import numpy as np
import pytest
from pymdp import utils

# Import PyMDP for real operations
from pymdp.agent import Agent as PyMDPAgent


class TestPyMDPAdapterStrictTypeChecking:
    """Test suite for PyMDP adapter with strict type checking - NO fallbacks."""

    def test_adapter_sample_action_returns_exact_int_type(self):
        """Adapter must convert sample_action() return to EXACT int type.

        PyMDP returns numpy.ndarray[float64] with shape (1,)
        Adapter must convert to exactly int with no graceful fallbacks.
        """
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real PyMDP agent
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent properly for PyMDP workflow
        agent.infer_states([0])
        agent.infer_policies()

        # Test adapter conversion
        result = adapter.sample_action(agent)

        # Must be EXACT int type, no numpy types allowed
        assert type(result) is int, f"Expected exact int type, got {type(result)}"
        assert result >= 0, "Action index must be non-negative"
        assert result < num_controls[0], f"Action {result} exceeds max {num_controls[0] - 1}"

    def test_adapter_sample_action_type_validation_fails_on_wrong_agent_type(
        self,
    ):
        """Adapter must raise TypeError for non-PyMDPAgent inputs - NO graceful handling."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test with wrong types - should raise TypeError immediately
        with pytest.raises(TypeError, match="Expected PyMDPAgent"):
            adapter.sample_action(None)

        with pytest.raises(TypeError, match="Expected PyMDPAgent"):
            adapter.sample_action("not an agent")

        with pytest.raises(TypeError, match="Expected PyMDPAgent"):
            adapter.sample_action(42)

    def test_adapter_sample_action_validates_numpy_return_strictly(self):
        """Adapter must validate PyMDP returns numpy.ndarray with specific properties."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Mock PyMDP agent that returns wrong types
        mock_agent = Mock(spec=PyMDPAgent)

        # Test case 1: Returns non-ndarray - should raise RuntimeError
        mock_agent.sample_action.return_value = 42  # plain int
        with pytest.raises(RuntimeError, match="expected numpy.ndarray"):
            adapter.sample_action(mock_agent)

        # Test case 2: Returns array with wrong shape - should raise RuntimeError
        mock_agent.sample_action.return_value = np.array([1, 2])  # shape (2,)
        with pytest.raises(RuntimeError, match="expected \\(1,\\)"):
            adapter.sample_action(mock_agent)

        # Test case 3: Returns array with wrong dtype - should raise RuntimeError
        mock_agent.sample_action.return_value = np.array([1], dtype=np.bool_)
        with pytest.raises(RuntimeError, match="unexpected dtype"):
            adapter.sample_action(mock_agent)

    def test_adapter_infer_policies_returns_exact_tuple_types(self):
        """Adapter must validate infer_policies() returns tuple of numpy arrays."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real PyMDP agent
        num_obs = [2]
        num_states = [2]
        num_controls = [2]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent
        agent.infer_states([0])

        # Test adapter
        q_pi, G = adapter.infer_policies(agent)

        # Must return exactly tuple of two numpy arrays
        assert isinstance(q_pi, np.ndarray), f"q_pi must be ndarray, got {type(q_pi)}"
        assert isinstance(G, np.ndarray), f"G must be ndarray, got {type(G)}"
        assert np.issubdtype(q_pi.dtype, np.floating), (
            f"q_pi must be floating type, got {q_pi.dtype}"
        )
        assert np.issubdtype(G.dtype, np.floating), f"G must be floating type, got {G.dtype}"

    def test_adapter_infer_policies_validation_fails_on_wrong_return_types(
        self,
    ):
        """Adapter must raise RuntimeError for incorrect infer_policies return types."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        mock_agent = Mock(spec=PyMDPAgent)

        # Test case 1: Returns non-tuple
        mock_agent.infer_policies.return_value = np.array([1, 2])
        with pytest.raises(RuntimeError, match="expected tuple"):
            adapter.infer_policies(mock_agent)

        # Test case 2: Returns tuple of wrong length
        mock_agent.infer_policies.return_value = (np.array([1.0]),)  # length 1
        with pytest.raises(RuntimeError, match="tuple of length 1, expected 2"):
            adapter.infer_policies(mock_agent)

        # Test case 3: Returns tuple with non-ndarray elements
        mock_agent.infer_policies.return_value = (
            [1, 2],
            [3, 4],
        )  # lists not arrays
        with pytest.raises(RuntimeError, match="expected numpy.ndarray"):
            adapter.infer_policies(mock_agent)

    def test_adapter_infer_states_validates_input_and_output_strictly(self):
        """Adapter must validate infer_states input/output with no graceful degradation."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real agent
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Test valid observation formats
        test_observations = [
            0,  # int
            [1],  # list
            np.array([2]),  # ndarray
        ]

        for obs in test_observations:
            result = adapter.infer_states(agent, obs)
            assert isinstance(result, list), f"Must return list for obs {obs}"
            assert len(result) > 0, f"Must return non-empty list for obs {obs}"

            # Validate each belief array
            for i, belief in enumerate(result):
                assert isinstance(belief, np.ndarray), f"Belief {i} must be ndarray"
                assert np.issubdtype(belief.dtype, np.floating), f"Belief {i} must be floating type"

    def test_adapter_infer_states_rejects_invalid_observation_types(self):
        """Adapter must reject invalid observation types with TypeError."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        mock_agent = Mock(spec=PyMDPAgent)

        # Test invalid observation types
        invalid_observations = [
            "string",  # string
            {"dict": "value"},  # dict
            object(),  # arbitrary object
        ]

        for invalid_obs in invalid_observations:
            with pytest.raises(TypeError, match="not supported"):
                adapter.infer_states(mock_agent, invalid_obs)

    def test_adapter_validate_agent_state_strict_requirements(self):
        """Adapter must validate agent has required attributes with no fallbacks."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test with real agent (should pass)
        num_obs = [2]
        num_states = [2]
        num_controls = [2]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        result = adapter.validate_agent_state(agent)
        assert result is True, "Valid agent should pass validation"

        # Test with agent missing required attributes
        mock_agent = Mock(spec=PyMDPAgent)
        # Mock agent that passes isinstance check but missing attribute
        mock_agent.__class__ = PyMDPAgent
        # Remove required attribute by not setting it

        with pytest.raises(RuntimeError, match="missing required attribute: A"):
            adapter.validate_agent_state(mock_agent)

    def test_adapter_safe_array_conversion_strict_no_fallbacks(self):
        """Adapter safe_array_conversion must work or fail - NO graceful degradation."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test successful conversions
        test_cases = [
            (np.array([5]), int, 5),
            (np.array([3.7]), int, 3),
            (np.array([2.5]), float, 2.5),
            (np.int64(7), int, 7),
            (5, int, 5),
            (3.14, float, 3.14),
        ]

        for value, target_type, expected in test_cases:
            result = adapter.safe_array_conversion(value, target_type)
            assert type(result) is target_type, f"Wrong type for {value}"
            assert result == expected, f"Wrong value for {value}"

    def test_adapter_safe_array_conversion_fails_on_invalid_inputs(self):
        """Adapter must raise exceptions for invalid conversions - NO graceful handling."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test invalid target types
        with pytest.raises(TypeError, match="target_type must be int or float"):
            adapter.safe_array_conversion(5, str)

        # Test empty arrays
        with pytest.raises(ValueError, match="Cannot convert empty array"):
            adapter.safe_array_conversion(np.array([]), int)

        # Test multi-element arrays
        with pytest.raises(ValueError, match="Cannot convert multi-element array"):
            adapter.safe_array_conversion(np.array([1, 2, 3]), int)

        # Test unconvertible types
        with pytest.raises(TypeError, match="Cannot convert"):
            adapter.safe_array_conversion("string", int)

    def test_adapter_handles_tuple_vs_int_return_issue(self):
        """Test that adapter correctly handles PyMDP's tuple/int return format changes.

        This specifically addresses the tuple/int return value issue from base_agent.py:397
        """
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Mock different return formats that PyMDP might use
        mock_agent = Mock(spec=PyMDPAgent)

        # Test case 1: PyMDP returns numpy array with single value (current behavior)
        mock_agent.sample_action.return_value = np.array([2], dtype=np.float64)
        result = adapter.sample_action(mock_agent)
        assert type(result) is int
        assert result == 2

        # Test case 2: PyMDP returns int64 array
        mock_agent.sample_action.return_value = np.array([1], dtype=np.int64)
        result = adapter.sample_action(mock_agent)
        assert type(result) is int
        assert result == 1

        # Test case 3: PyMDP returns float32 array
        mock_agent.sample_action.return_value = np.array([3], dtype=np.float32)
        result = adapter.sample_action(mock_agent)
        assert type(result) is int
        assert result == 3

        # Test case 4: Verify it rejects tuples (not expected from PyMDP)
        mock_agent.sample_action.return_value = (2,)
        with pytest.raises(RuntimeError, match="expected numpy.ndarray"):
            adapter.sample_action(mock_agent)

    def test_adapter_belief_update_mechanism_translation(self):
        """Test adapter translates belief update mechanisms correctly."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create agent and test belief updates
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Test belief update workflow
        observations = [0, 1, 2]

        for obs in observations:
            # Update beliefs with observation
            beliefs = adapter.infer_states(agent, obs)

            # Verify beliefs are properly formatted
            assert isinstance(beliefs, list)
            assert len(beliefs) > 0

            for belief in beliefs:
                assert isinstance(belief, np.ndarray)
                assert np.issubdtype(belief.dtype, np.floating)
                # Beliefs should sum to 1 (probability distribution)
                assert np.isclose(belief.sum(), 1.0)


class TestPyMDPAdapterRealIntegration:
    """Integration tests with real PyMDP operations - NO mocks."""

    def test_adapter_with_real_pymdp_full_workflow(self):
        """Test adapter with complete PyMDP workflow using real operations."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_controls = [4]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Validate agent state
        assert adapter.validate_agent_state(agent) is True

        # Test full workflow
        observation = 0

        # Step 1: Infer states
        beliefs = adapter.infer_states(agent, observation)
        assert len(beliefs) > 0
        assert all(isinstance(b, np.ndarray) for b in beliefs)

        # Step 2: Infer policies
        q_pi, G = adapter.infer_policies(agent)
        assert isinstance(q_pi, np.ndarray)
        assert isinstance(G, np.ndarray)

        # Step 3: Sample action
        action = adapter.sample_action(agent)
        assert type(action) is int
        assert 0 <= action < num_controls[0]

    def test_adapter_handles_different_agent_configurations(self):
        """Test adapter works with different PyMDP agent configurations."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test different sizes
        configurations = [
            ([2], [2], [2]),
            ([3], [3], [3]),
            ([5], [5], [4]),
        ]

        for num_obs, num_states, num_controls in configurations:
            A = utils.random_A_matrix(num_obs, num_states)
            B = utils.random_B_matrix(num_states, num_controls)
            agent = PyMDPAgent(A=A, B=B)

            # Test each adapter method
            assert adapter.validate_agent_state(agent) is True

            beliefs = adapter.infer_states(agent, [0])
            assert len(beliefs) > 0

            q_pi, G = adapter.infer_policies(agent)
            assert q_pi.size > 0
            assert G.size > 0

            action = adapter.sample_action(agent)
            assert 0 <= action < num_controls[0]

    def test_adapter_error_handling_real_failures(self):
        """Test adapter handles real PyMDP failures without graceful degradation."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create agent with invalid state (no initialization)
        num_obs = [2]
        num_states = [2]
        num_controls = [2]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Try to sample action without proper initialization
        # This should raise an exception since agent.q_pi is not initialized
        with pytest.raises(Exception) as exc_info:
            adapter.sample_action(agent)

        # The adapter should propagate PyMDP errors without masking them
        # We expect either RuntimeError (from adapter) or AttributeError (from PyMDP)
        assert isinstance(exc_info.value, (RuntimeError, AttributeError))

        # If it's an AttributeError from PyMDP, it should mention q_pi
        if isinstance(exc_info.value, AttributeError):
            assert "q_pi" in str(exc_info.value)

    def test_adapter_performance_with_real_operations(self):
        """Test adapter performance overhead is minimal with real operations."""
        import time

        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real agent
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent
        agent.infer_states([0])
        agent.infer_policies()

        # Time adapter operations
        start_time = time.time()

        for _ in range(10):  # Small batch for CI
            beliefs = adapter.infer_states(agent, [0])
            q_pi, G = adapter.infer_policies(agent)
            action = adapter.sample_action(agent)

        elapsed = time.time() - start_time

        # Adapter overhead should be minimal (< 1 second for 10 operations)
        assert elapsed < 1.0, f"Adapter too slow: {elapsed:.3f}s for 10 operations"

        # Validate last results
        assert type(action) is int
        assert len(beliefs) > 0
        assert isinstance(q_pi, np.ndarray)
        assert isinstance(G, np.ndarray)


class TestPyMDPAdapterDesignDecisions:
    """Test suite documenting adapter design decisions and behavior.

    This adapter follows strict principles:
    1. ZERO fallbacks - operations work or raise exceptions
    2. Strict type checking throughout
    3. Real PyMDP operations only
    4. Clear error propagation
    """

    def test_adapter_design_no_fallbacks(self):
        """Document that adapter has ZERO fallbacks - it works or raises."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # No fallback for wrong agent type
        with pytest.raises(TypeError):
            adapter.sample_action("not an agent")

        # No fallback for missing methods
        mock_obj = Mock()
        with pytest.raises(TypeError):
            adapter.sample_action(mock_obj)

        # No fallback for wrong return types
        mock_agent = Mock(spec=PyMDPAgent)
        mock_agent.sample_action.return_value = "wrong type"
        with pytest.raises(RuntimeError):
            adapter.sample_action(mock_agent)

    def test_adapter_design_strict_typing(self):
        """Document strict type checking in all conversions."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # sample_action always returns exact int type
        mock_agent = Mock(spec=PyMDPAgent)
        mock_agent.sample_action.return_value = np.array([5.9])

        result = adapter.sample_action(mock_agent)
        assert type(result) is int  # Not np.int64 or float
        assert result == 5  # Truncated, not rounded

        # safe_array_conversion enforces target types
        assert type(adapter.safe_array_conversion(5.9, int)) is int
        assert type(adapter.safe_array_conversion(5, float)) is float

    def test_adapter_design_clear_errors(self):
        """Document that errors are clear and specific."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Type errors are specific
        try:
            adapter.sample_action(None)
        except TypeError as e:
            assert "Expected PyMDPAgent" in str(e)
            assert "got <class 'NoneType'>" in str(e)

        # Runtime errors describe the issue
        mock_agent = Mock(spec=PyMDPAgent)
        mock_agent.sample_action.return_value = np.array([1, 2])  # Wrong shape

        try:
            adapter.sample_action(mock_agent)
        except RuntimeError as e:
            assert "shape (2,), expected (1,)" in str(e)

    def test_adapter_design_real_pymdp_only(self):
        """Document that adapter only works with real PyMDP operations."""
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create real PyMDP agent
        A = utils.random_A_matrix([2], [2])
        B = utils.random_B_matrix([2], [2])
        agent = PyMDPAgent(A=A, B=B)

        # All adapter methods work with real agent
        assert adapter.validate_agent_state(agent) is True

        # Real PyMDP workflow
        beliefs = adapter.infer_states(agent, 0)
        assert len(beliefs) > 0

        q_pi, G = adapter.infer_policies(agent)
        assert q_pi is not None
        assert G is not None

        # This demonstrates no mocking or fake operations
