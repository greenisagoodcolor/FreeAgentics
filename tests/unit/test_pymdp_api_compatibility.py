"""Test PyMDP API compatibility adapter with strict type checking.

This module contains failing tests that will drive the creation of an API
compatibility adapter to fix return value handling issues.

Based on Task 1.3: Create API compatibility adapter with strict type checking
"""

import numpy as np
import pytest
from pymdp import utils

# Import the actual PyMDP agent to test against real API
from pymdp.agent import Agent as PyMDPAgent


class TestPyMDPAPICompatibility:
    """Test PyMDP API compatibility for exact signature matching."""

    def test_pymdp_sample_action_return_type_unpacking_failure(self):
        """Test that reproduces the 'too many values to unpack' error.

        This test should FAIL initially, demonstrating the exact issue
        mentioned in the PRD about base_agent.py:397.

        The issue is that sample_action() returns different types than expected
        by the unpacking code.
        """
        # Create a minimal PyMDP agent setup
        num_obs = [4]
        num_states = [4]
        num_controls = [4]

        # Build minimal matrices
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = PyMDPAgent(A=A, B=B)

        # This should demonstrate what sample_action actually returns
        action_result = agent.sample_action()

        # This assertion should fail - showing actual vs expected return type
        assert isinstance(
            action_result, int
        ), f"Expected int, got {type(action_result)}: {action_result}"

    def test_pymdp_sample_action_vs_expected_unpacking(self):
        """Test the exact unpacking scenario that fails in base_agent.py.

        This simulates the error in line 899: action_idx = self.pymdp_agent.sample_action()
        and the subsequent safe_execute unpacking in lines 902-908.
        """
        # Create minimal agent
        num_obs = [4]
        num_states = [4]
        num_controls = [4]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = PyMDPAgent(A=A, B=B)

        # Follow correct PyMDP workflow
        obs = [0]
        agent.infer_states(obs)
        agent.infer_policies()

        # Simulate the exact scenario from base_agent.py
        action_idx = agent.sample_action()

        # This should fail if action_idx is not what we expect
        try:
            # Simulate safe_array_to_int call
            from agents.base_agent import safe_array_to_int

            converted = safe_array_to_int(action_idx, 4)

            # The test should document what types we're actually dealing with
            assert isinstance(
                converted, int
            ), f"safe_array_to_int should return int, got {type(converted)}"

        except Exception as e:
            pytest.fail(f"Conversion failed with actual error: {e}")

    def test_pymdp_agent_initialization_api_signature(self):
        """Test that PyMDP agent initialization matches expected API.

        This test verifies the constructor signature and required parameters.
        """
        # Test with minimal parameters
        num_obs = [2]
        num_states = [2]
        num_controls = [2]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        # This should work with current API
        agent = PyMDPAgent(A=A, B=B)

        # Verify agent has expected methods
        assert hasattr(agent, "sample_action"), "Agent should have sample_action method"
        assert hasattr(agent, "infer_states"), "Agent should have infer_states method"
        assert hasattr(agent, "step"), "Agent should have step method"

    def test_adapter_requirement_strict_type_checking(self):
        """Test that demonstrates need for strict type checking adapter.

        This test will drive the creation of the compatibility adapter
        that handles type mismatches without graceful fallbacks.
        """
        # This test should now pass since adapter is created
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Verify adapter is properly initialized
        assert adapter is not None
        assert hasattr(adapter, "sample_action")
        assert hasattr(adapter, "infer_policies")
        assert hasattr(adapter, "safe_array_conversion")

    def test_adapter_sample_action_strict_return_type(self):
        """Test adapter ensures sample_action returns exactly what we need.

        The adapter must convert PyMDP return types to exact expected types
        with no graceful fallbacks - must work or raise exceptions.
        """
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Create a real PyMDP agent for testing
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent properly
        agent.infer_states([0])
        agent.infer_policies()

        # Test adapter with real PyMDP agent
        result = adapter.sample_action(agent)

        # Must be exact int type, no numpy types allowed
        assert type(result) is int, f"Expected int, got {type(result)}"
        assert isinstance(result, int), "Result must be int instance"
        assert result >= 0, "Action index must be non-negative"

    def test_adapter_handles_different_pymdp_return_formats(self):
        """Test that adapter handles all possible PyMDP return formats.

        PyMDP can return: int, numpy.int64, numpy.array([int]), etc.
        Adapter must handle all without graceful degradation.
        """
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Test different return types using safe_array_conversion directly
        test_cases = [
            (np.int64(2), 2),
            (np.array(2), 2),  # 0-dimensional
            (np.array([2]), 2),  # 1-dimensional
            (2, 2),  # plain int
            (np.float64(2.7), 2),  # float conversion to int
        ]

        for test_input, expected in test_cases:
            result = adapter.safe_array_conversion(test_input, int)
            assert type(result) is int, f"Failed for input {test_input} (type: {type(test_input)})"
            assert result == expected, f"Expected {expected}, got {result} for input {test_input}"

    def test_real_pymdp_agent_return_types_documentation(self):
        """Document what PyMDP actually returns for debugging.

        This test captures the actual behavior to inform adapter design.
        """
        # Create real agent to see actual returns
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = PyMDPAgent(A=A, B=B)

        # Follow correct PyMDP workflow: infer_states -> infer_policies -> sample_action
        obs = [0]
        agent.infer_states(obs)

        # Must infer policies first to initialize q_pi
        q_pi, G = agent.infer_policies()

        # Now document what sample_action actually returns
        action = agent.sample_action()

        # Print for debugging (will be visible when test runs)
        print("\nPyMDP sample_action() actual return:")
        print(f"  Value: {action}")
        print(f"  Type: {type(action)}")
        print(f"  Shape: {getattr(action, 'shape', 'N/A')}")
        print(f"  Dtype: {getattr(action, 'dtype', 'N/A')}")

        # Also document infer_policies return for completeness
        print("\nPyMDP infer_policies() returns:")
        print(f"  q_pi Value: {q_pi}")
        print(f"  q_pi Type: {type(q_pi)}")
        print(f"  q_pi Shape: {getattr(q_pi, 'shape', 'N/A')}")
        print(f"  G Value: {G}")
        print(f"  G Type: {type(G)}")

        # Test will show us what we're dealing with
        # This assertion might pass or fail - we need to see what happens
        pass  # Just document, don't assert anything yet


class TestReturnValueHandlingIssues:
    """Test cases that expose return value handling problems (Task 1.4)."""

    def test_safe_execute_tuple_unpacking_compatibility(self):
        """Test the exact tuple unpacking issue from base_agent.py lines 902-908."""
        from agents.base_agent import safe_array_to_int
        from agents.pymdp_error_handling import PyMDPErrorHandler

        # Create handler like in base_agent
        handler = PyMDPErrorHandler(agent_id="test")

        # Simulate the exact call from base_agent.py
        action_idx = np.array([2])  # Typical PyMDP return

        # This is the exact pattern from lines 902-908
        success_conv, action_idx_converted, error = handler.safe_execute(
            "action_index_conversion",
            lambda: safe_array_to_int(action_idx, 4),
            lambda: 4,
        )

        # This should work - if it fails, we have tuple unpacking issues
        assert isinstance(success_conv, bool)
        assert isinstance(action_idx_converted, int)
        assert error is None or hasattr(error, "error_type")

        print(
            f"safe_execute returned: success={success_conv}, result={action_idx_converted}, error={error}"
        )

    def test_infer_policies_none_return_unpacking_issue(self):
        """Test the exact unpacking issue with infer_policies None return.

        This tests the scenario from base_agent.py lines 882-888 where
        infer_policies() might return None vs tuple, causing unpacking issues.
        """
        from agents.pymdp_error_handling import PyMDPErrorHandler

        handler = PyMDPErrorHandler(agent_id="test")

        # Test scenario 1: Normal tuple return (should work)
        def mock_infer_policies_tuple():
            return (np.array([0.5, 0.5]), np.array([-1.0, -1.0]))

        success, result, error = handler.safe_execute(
            "action_inference",
            mock_infer_policies_tuple,
            lambda: None,
        )

        assert success is True
        assert result is not None
        assert len(result) == 2

        # Test scenario 2: None return (this might cause issues in real code)
        def mock_infer_policies_none():
            return None

        success2, result2, error2 = handler.safe_execute(
            "action_inference",
            mock_infer_policies_none,
            lambda: None,
        )

        assert success2 is True
        assert result2 is None

        # The issue is when downstream code tries to unpack the None result
        # If code expects: q_pi, G = result, it will fail when result is None

        print(f"Tuple return: success={success}, result={result}")
        print(f"None return: success={success2}, result={result2}")

    def test_none_check_removal_requirement(self):
        """Test demonstrating that None checks must be removed per Task 1.4."""
        # This test documents current None checks that should be removed

        # Search for patterns like: if result is not None
        # These should be replaced with direct access that fails fast

        # This test will guide the removal of defensive None checks
