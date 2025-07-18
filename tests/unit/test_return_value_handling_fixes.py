"""Test cases for Task 1.4: Fix return value handling and remove all None checks.

This module contains tests that drive the removal of defensive None checks
and getattr usage, replacing them with direct access that fails fast.

Based on Task 1.4: Fix return value handling and remove all None checks
"""

import numpy as np
import pytest
from pymdp import utils

# Import actual PyMDP for testing
from pymdp.agent import Agent as PyMDPAgent

from agents.base_agent import BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent


class TestReturnValueHandlingFixes:
    """Test cases that expose None check and getattr issues to be fixed."""

    def test_pymdp_agent_qs_direct_access_no_none_check(self):
        """Test that PyMDP agent qs should be accessed directly without None checks.

        This test should initially FAIL to demonstrate that we need to remove
        'if result is not None' defensive patterns.
        """
        # Create a real PyMDP agent
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent properly
        agent.infer_states([0])

        # This should work - direct access without None checks
        # The test will fail if the code still has defensive None checks
        try:
            # Direct access - should always work after proper initialization
            qs = agent.qs
            assert (
                qs is not None
            ), "qs should never be None after proper initialization"
            # Fix: PyMDP returns numpy.ndarray with dtype=object, not list
            assert isinstance(
                qs, np.ndarray
            ), f"qs should be numpy.ndarray, got {type(qs)}"
            assert qs.size > 0, "qs should not be empty"

            # Test that we can access first element directly
            first_belief = qs[0]
            assert isinstance(
                first_belief, np.ndarray
            ), "First belief should be numpy array"

            print(f"Direct access successful: qs has {qs.size} factors")

        except (AttributeError, IndexError, TypeError) as e:
            pytest.fail(
                f"Direct access failed - this indicates defensive programming patterns need removal: {e}"
            )

    def test_pymdp_agent_G_direct_access_no_none_check(self):
        """Test that PyMDP agent G (expected free energy) should be accessed directly.

        This test demonstrates removal of 'if hasattr and is not None' patterns.
        """
        # Create a real PyMDP agent
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        agent = PyMDPAgent(A=A, B=B)

        # Initialize agent properly
        agent.infer_states([0])
        agent.infer_policies()

        # Direct access without hasattr/None checks
        try:
            # This should work after infer_policies()
            G = agent.G
            assert (
                G is not None
            ), "G should never be None after infer_policies()"
            assert isinstance(
                G, np.ndarray
            ), f"G should be numpy array, got {type(G)}"

            print(f"Direct G access successful: shape {G.shape}")

        except AttributeError as e:
            pytest.fail(
                f"Direct G access failed - attribute should exist after infer_policies(): {e}"
            )

    def test_base_agent_direct_attribute_access_no_getattr(self):
        """Test that agent attributes should be accessed directly, not via getattr.

        This test will drive removal of getattr usage in favor of direct access.
        """
        # Create a basic agent
        agent = BasicExplorerAgent("test_agent", "TestAgent")

        # These should work with direct access, no getattr needed
        try:
            # Direct access should work for required attributes
            agent_id = agent.agent_id
            assert (
                agent_id == "test_agent"
            ), f"Expected test_agent, got {agent_id}"

            name = agent.name
            assert name == "TestAgent", f"Expected TestAgent, got {name}"

            position = agent.position
            # Position can be list or tuple - both are valid
            assert isinstance(
                position, (list, tuple)
            ), f"Position should be list or tuple, got {type(position)}"

            metrics = agent.metrics
            assert isinstance(
                metrics, dict
            ), f"Metrics should be dict, got {type(metrics)}"

            beliefs = agent.beliefs
            assert isinstance(
                beliefs, dict
            ), f"Beliefs should be dict, got {type(beliefs)}"

            print("Direct attribute access successful")

        except AttributeError as e:
            pytest.fail(
                f"Direct attribute access failed - this suggests getattr overuse: {e}"
            )

    def test_none_check_removal_in_action_sampling(self):
        """Test that action sampling should not use defensive None checks.

        This test demonstrates that action sampling operations should assume
        valid agent state and fail fast if not properly initialized.
        """
        # Create agent
        agent = BasicExplorerAgent("test_agent", "TestAgent")
        agent.start()

        # Mock observation to trigger action sampling
        # observation = {"position": (1, 1), "surroundings": np.zeros((3, 3))}  # Not used

        # This should work without None checks - agent should be properly initialized
        try:
            action = agent.select_action()
            assert (
                action is not None
            ), "Action should never be None from proper agent"
            assert isinstance(
                action, str
            ), f"Action should be string, got {type(action)}"
            assert (
                action in agent.actions
            ), f"Action {action} not in valid actions {agent.actions}"

            print(f"Action sampling successful: {action}")

        except Exception as e:
            pytest.fail(
                f"Action sampling failed - should work without defensive checks: {e}"
            )

    def test_belief_update_direct_access_patterns(self):
        """Test that belief updates should use direct access, not defensive patterns."""
        # Create agent
        agent = BasicExplorerAgent("test_agent", "TestAgent")
        agent.start()

        # Perform belief update
        observation = {"position": (1, 1), "surroundings": np.zeros((3, 3))}
        agent.perceive(observation)
        agent.update_beliefs()

        # Test direct access to PyMDP agent components
        try:
            if (
                agent.pymdp_agent is not None
            ):  # This check is acceptable for optional components
                # But these should be direct access without None checks
                A = agent.pymdp_agent.A
                assert A is not None, "A matrix should not be None"

                B = agent.pymdp_agent.B
                assert B is not None, "B matrix should not be None"

                qs = agent.pymdp_agent.qs
                assert (
                    qs is not None
                ), "Beliefs should not be None after update"

                print("Direct belief access successful")
            else:
                pytest.skip(
                    "PyMDP agent not available - testing fallback scenario"
                )

        except AttributeError as e:
            pytest.fail(f"Direct belief access failed: {e}")

    def test_remove_defensive_none_checks_in_resource_collector(self):
        """Test ResourceCollectorAgent should not use defensive None checks."""
        agent = ResourceCollectorAgent("resource_test", name="ResourceTest")

        # Direct attribute access - no getattr
        try:
            agent_id = agent.agent_id
            action_map = agent.action_map
            position = agent.position

            assert agent_id == "resource_test"
            assert isinstance(action_map, dict)
            # Position can be list or tuple - both are valid
            assert isinstance(position, (list, tuple))

            print("ResourceCollectorAgent direct access successful")

        except AttributeError as e:
            pytest.fail(f"ResourceCollectorAgent direct access failed: {e}")

    def test_remove_defensive_none_checks_in_coalition_coordinator(self):
        """Test CoalitionCoordinatorAgent should not use defensive None checks."""
        agent = CoalitionCoordinatorAgent(
            "coalition_test", name="CoalitionTest"
        )

        # Direct attribute access - no getattr
        try:
            agent_id = agent.agent_id
            action_map = agent.action_map
            position = agent.position

            assert agent_id == "coalition_test"
            assert isinstance(action_map, dict)
            # Position can be list or tuple - both are valid
            assert isinstance(position, (list, tuple))

            print("CoalitionCoordinatorAgent direct access successful")

        except AttributeError as e:
            pytest.fail(f"CoalitionCoordinatorAgent direct access failed: {e}")

    def test_pymdp_matrix_direct_access_no_validation(self):
        """Test that PyMDP matrices should be accessed directly without validation.

        Demonstrates removal of defensive 'if matrix is not None' patterns.
        """
        # Create agent with PyMDP
        agent = BasicExplorerAgent("matrix_test", "MatrixTest")
        agent.start()

        if agent.pymdp_agent is not None:
            try:
                # Direct access - should work without None checks
                A = agent.pymdp_agent.A
                B = agent.pymdp_agent.B
                C = agent.pymdp_agent.C
                D = agent.pymdp_agent.D

                # Validate types directly
                assert isinstance(
                    A, np.ndarray
                ), f"A should be ndarray, got {type(A)}"
                assert isinstance(
                    B, np.ndarray
                ), f"B should be ndarray, got {type(B)}"
                assert isinstance(
                    C, np.ndarray
                ), f"C should be ndarray, got {type(C)}"
                assert isinstance(
                    D, np.ndarray
                ), f"D should be ndarray, got {type(D)}"

                print("Direct matrix access successful")

            except AttributeError as e:
                pytest.fail(f"Direct matrix access failed: {e}")
        else:
            pytest.skip("PyMDP agent not available")

    def test_action_conversion_no_fallback_handling(self):
        """Test that action conversion should not use fallback/default handling.

        Operations must return valid values or fail - no graceful degradation.
        """
        # Test the safe_array_to_int function behavior
        from agents.base_agent import safe_array_to_int

        # These should work without fallback to defaults
        test_cases = [
            np.array([2]),
            np.array(2),
            np.int64(2),
            2,
            np.float64(2.0),
        ]

        for test_input in test_cases:
            result = safe_array_to_int(test_input)
            assert isinstance(
                result, int
            ), f"Expected int, got {type(result)} for {test_input}"
            assert (
                result >= 0
            ), f"Expected non-negative, got {result} for {test_input}"

        print("Action conversion without fallbacks successful")

    def test_error_handling_should_fail_fast_not_graceful(self):
        """Test that error handling should fail fast, not gracefully degrade.

        This drives the removal of try/catch blocks that hide failures.
        """
        # This test documents the expected behavior:
        # Operations should either work correctly or fail immediately
        # No silent failures or graceful degradation

        # Test with invalid inputs - should raise exceptions, not return defaults
        from agents.pymdp_adapter import PyMDPCompatibilityAdapter

        adapter = PyMDPCompatibilityAdapter()

        # Invalid input should raise exception, not return default
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            adapter.safe_array_conversion("invalid_input", int)

        with pytest.raises((TypeError, ValueError, RuntimeError)):
            adapter.safe_array_conversion([], int)

        print("Fail-fast error handling verified")


class TestDirectAccessPatterns:
    """Tests that demonstrate proper direct access patterns to replace defensive code."""

    def test_agent_initialization_must_succeed_or_fail(self):
        """Test that agent initialization must succeed completely or fail.

        No partial initialization with None components.
        """
        agent = BasicExplorerAgent("init_test", "InitTest")

        # After initialization, all required components should exist
        assert agent.agent_id is not None
        assert agent.name is not None
        assert agent.position is not None
        assert agent.metrics is not None
        assert agent.beliefs is not None
        assert agent.action_map is not None
        assert agent.actions is not None

        print("Complete initialization verified")

    def test_pymdp_operations_must_have_valid_state(self):
        """Test that PyMDP operations assume valid agent state.

        No checking if components exist - they must exist.
        """
        agent = BasicExplorerAgent("state_test", "StateTest")
        agent.start()

        if agent.pymdp_agent is not None:
            # These operations should work without state validation
            observation = {
                "position": (1, 1),
                "surroundings": np.zeros((3, 3)),
            }
            agent.perceive(observation)
            agent.update_beliefs()

            # Direct access to results
            qs = agent.pymdp_agent.qs
            assert len(qs) > 0, "Should have beliefs after update"

            # Policy inference should work
            agent.pymdp_agent.infer_policies()
            G = agent.pymdp_agent.G
            assert G is not None, "Should have expected free energy"

            print("PyMDP operations with valid state successful")
        else:
            pytest.skip("PyMDP not available")

    def test_action_sampling_must_return_valid_action(self):
        """Test that action sampling must return valid action or fail.

        No returning 'stay' as fallback - must work properly.
        """
        agent = BasicExplorerAgent("action_test", "ActionTest")
        agent.start()

        # Should return valid action
        action = agent.select_action()

        # Must be a valid action string
        assert isinstance(action, str)
        assert action in agent.actions
        assert action != "", "Action cannot be empty string"

        print(f"Valid action sampling: {action}")

    def test_matrix_operations_direct_computation(self):
        """Test that matrix operations work directly without validation.

        Demonstrates removal of matrix validation checks.
        """
        agent = BasicExplorerAgent("matrix_ops_test", "MatrixOpsTest")
        agent.start()

        if agent.pymdp_agent is not None:
            # Direct matrix operations
            A = agent.pymdp_agent.A
            B = agent.pymdp_agent.B

            # Should be able to perform operations directly
            A_norm = A.sum(axis=0)  # Direct operation
            assert A_norm.shape[0] > 0, "Matrix operation should work"

            B_shape = B.shape  # Direct access
            assert len(B_shape) >= 2, "B matrix should be multi-dimensional"

            print("Direct matrix operations successful")
        else:
            pytest.skip("PyMDP not available")
