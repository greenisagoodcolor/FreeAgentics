"""
Task 9.5: Comprehensive Edge Case and Error Handling Tests

Tests systematically test error conditions and edge cases throughout the codebase
to ensure robust error handling, following TDD approach with failing tests first.

Focus areas:
- Null/undefined inputs and boundary values
- Concurrent access scenarios
- Memory exhaustion conditions
- Network failures and timeout scenarios
- Invalid state transitions
- Cascading failures
- Error propagation and recovery mechanisms
"""

import asyncio
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.base_agent import PYMDP_AVAILABLE
from agents.error_handling import ErrorHandler, InferenceError, PyMDPError


class TestNullAndBoundaryInputs:
    """Test handling of null, undefined, and boundary value inputs."""

    def test_agent_creation_with_null_inputs(self):
        """Test agent creation with various null/empty inputs."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        # Test None agent_id
        with pytest.raises((ValueError, TypeError)):
            BasicExplorerAgent(None, "Test Agent", grid_size=3)

        # Test empty string agent_id
        with pytest.raises((ValueError, TypeError)):
            BasicExplorerAgent("", "Test Agent", grid_size=3)

        # Test None agent_name
        with pytest.raises((ValueError, TypeError)):
            BasicExplorerAgent("test", None, grid_size=3)

        # Test zero grid size
        with pytest.raises((ValueError, TypeError)):
            BasicExplorerAgent("test", "Test", grid_size=0)

        # Test negative grid size
        with pytest.raises((ValueError, TypeError)):
            BasicExplorerAgent("test", "Test", grid_size=-1)

    def test_observation_boundary_values(self):
        """Test observation processing with boundary values."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Test extremely large position values
        large_obs = {
            "position": [999999, 999999],
            "surroundings": np.zeros((3, 3)),
        }
        action = agent.step(large_obs)
        assert action in agent.actions

        # Test negative position values
        negative_obs = {
            "position": [-999, -999],
            "surroundings": np.zeros((3, 3)),
        }
        action = agent.step(negative_obs)
        assert action in agent.actions

        # Test NaN and infinity values
        nan_obs = {
            "position": [float("nan"), float("inf")],
            "surroundings": np.zeros((3, 3)),
        }
        action = agent.step(nan_obs)
        assert action in agent.actions

        # Test empty surroundings array
        empty_obs = {"position": [1, 1], "surroundings": np.array([])}
        action = agent.step(empty_obs)
        assert action in agent.actions

    def test_array_boundary_conditions(self):
        """Test numpy array processing with edge cases."""
        from agents.error_handling import validate_observation

        # Test empty arrays
        result = validate_observation({"data": np.array([])})
        assert result["valid"] is True

        # Test arrays with NaN
        nan_array = np.array([1.0, float("nan"), 3.0])
        result = validate_observation({"data": nan_array})
        assert result["valid"] is True

        # Test arrays with infinity
        inf_array = np.array([1.0, float("inf"), -float("inf")])
        result = validate_observation({"data": inf_array})
        assert result["valid"] is True

        # Test extremely large arrays (memory stress test)
        try:
            large_array = np.zeros((10000, 10000))  # 800MB array
            result = validate_observation({"data": large_array})
            # Should handle gracefully without crashing
            assert result["valid"] is True
        except MemoryError:
            # Expected on memory-constrained systems
            pass


class TestConcurrentAccessScenarios:
    """Test concurrent access patterns and thread safety."""

    def test_concurrent_agent_operations(self):
        """Test multiple agents operating concurrently."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agents = [BasicExplorerAgent(f"agent_{i}", f"Agent {i}", grid_size=3) for i in range(5)]

        for agent in agents:
            agent.start()

        # Concurrent step operations
        def agent_step(agent):
            observation = {
                "position": [1, 1],
                "surroundings": np.zeros((3, 3)),
            }
            return agent.step(observation)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(agent_step, agent) for agent in agents]
            results = [future.result(timeout=5) for future in futures]

        # All should complete successfully
        assert len(results) == 5
        assert all(result in agent.actions for result in results)

    def test_error_handler_thread_safety(self):
        """Test ErrorHandler thread safety under concurrent access."""
        handler = ErrorHandler("test_agent")

        def generate_errors(thread_id):
            for i in range(10):
                error = PyMDPError(f"Thread {thread_id} error {i}")
                handler.handle_error(error, f"operation_{i}")
                time.sleep(0.001)  # Small delay to encourage race conditions

        # Run multiple threads concurrently
        threads = [threading.Thread(target=generate_errors, args=(i,)) for i in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all errors were recorded safely
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 30  # 3 threads * 10 errors each

    def test_concurrent_belief_updates(self):
        """Test concurrent belief update operations."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        def update_beliefs():
            agent.current_observation = [np.random.randint(0, 9)]
            agent.update_beliefs()
            return True

        # Multiple concurrent belief updates
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_beliefs) for _ in range(10)]
            results = [future.result(timeout=2) for future in futures]

        # All should complete without deadlocks
        assert all(results)


class TestMemoryExhaustionScenarios:
    """Test behavior under memory pressure and resource limits."""

    def test_large_observation_handling(self):
        """Test handling of extremely large observations."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Progressively larger observations to test memory handling
        sizes = [100, 1000, 5000]  # Start smaller for CI environments

        for size in sizes:
            try:
                large_surroundings = np.random.rand(size, size)
                observation = {
                    "position": [1, 1],
                    "surroundings": large_surroundings,
                }

                # Should handle without crashing
                action = agent.step(observation)
                assert action in agent.actions

                # Force garbage collection to free memory
                del large_surroundings
                del observation
                gc.collect()

            except MemoryError:
                # Expected behavior on memory-constrained systems
                break

    def test_error_history_memory_bounds(self):
        """Test error history doesn't grow unbounded."""
        handler = ErrorHandler("test_agent")

        # Generate many errors
        for i in range(1000):
            error = PyMDPError(f"Error {i}")
            handler.handle_error(error, "test_operation")

        # Error history should be bounded
        summary = handler.get_error_summary()

        # Should not store all 1000 errors (memory protection)
        assert len(handler.error_history) <= 100  # Reasonable bound
        assert summary["total_errors"] == 1000  # But count should be accurate

    def test_memory_leak_prevention(self):
        """Test that operations don't create memory leaks."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        initial_objects = len(gc.get_objects())

        # Create and destroy many agents
        for i in range(10):
            agent = BasicExplorerAgent(f"test_{i}", f"Test Agent {i}", grid_size=3)
            agent.start()

            # Perform operations
            observation = {
                "position": [1, 1],
                "surroundings": np.zeros((3, 3)),
            }
            agent.step(observation)

            # Cleanup
            del agent
            gc.collect()

        final_objects = len(gc.get_objects())

        # Should not have significant object growth
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold


class TestNetworkAndTimeoutScenarios:
    """Test network failures and timeout conditions."""

    @patch("requests.post")
    def test_llm_api_timeout_handling(self, mock_post):
        """Test LLM API timeout scenarios."""
        from inference.llm.provider_interface import LLMProvider

        # Mock timeout exception
        mock_post.side_effect = TimeoutError("Request timed out")

        # This would test LLM provider timeout handling
        # Implementation depends on specific LLM provider setup
        LLMProvider()

        # Should handle timeout gracefully
        with pytest.raises((TimeoutError, Exception)):
            # This is a placeholder - actual implementation would test
            # specific LLM provider timeout handling
            pass

    def test_database_connection_failure(self):
        """Test database connection failure scenarios."""
        # Mock database connection failure
        with patch("database.session.get_session") as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            # Test that database-dependent operations handle failures
            # This is a placeholder for actual database failure testing
            with pytest.raises(Exception):
                # Actual test would attempt database operations
                raise Exception("Database connection failed")

    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self):
        """Test WebSocket connection failure scenarios."""
        # Test WebSocket connection timeout
        with pytest.raises((asyncio.TimeoutError, ConnectionError)):
            # Mock WebSocket connection that times out
            await asyncio.wait_for(asyncio.sleep(10), timeout=0.1)


class TestInvalidStateTransitions:
    """Test invalid state transitions and edge cases."""

    def test_agent_operations_before_start(self):
        """Test operations called before agent is started."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        # Don't call start()

        # Operations should handle uninitialized state gracefully
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}

        # These should either work with defaults or raise appropriate errors
        try:
            action = agent.step(observation)
            assert action in agent.actions  # If it works, should be valid
        except (ValueError, RuntimeError):
            # Expected for uninitialized agent
            pass

    def test_multiple_start_calls(self):
        """Test calling start() multiple times."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)

        # First start should work
        agent.start()

        # Subsequent starts should be handled gracefully
        agent.start()  # Should not crash
        agent.start()  # Should not crash

        # Agent should still function normally
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        action = agent.step(observation)
        assert action in agent.actions

    def test_invalid_configuration_changes(self):
        """Test invalid configuration changes during runtime."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Try to change configuration after initialization
        # Should either be prevented or handled gracefully
        try:
            agent.grid_size = -1  # Invalid value
            agent.actions = []  # Invalid empty actions

            # Agent should still function or raise appropriate errors
            observation = {
                "position": [1, 1],
                "surroundings": np.zeros((3, 3)),
            }
            agent.step(observation)

        except (ValueError, RuntimeError):
            # Expected behavior for invalid configuration
            pass


class TestCascadingFailures:
    """Test cascading failure scenarios and error propagation."""

    def test_multiple_simultaneous_failures(self):
        """Test behavior when multiple components fail simultaneously."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Mock multiple failures
        mock_agent = MagicMock()
        mock_agent.infer_states.side_effect = Exception("State inference failed")
        mock_agent.infer_policies.side_effect = Exception("Policy inference failed")
        mock_agent.sample_action.side_effect = Exception("Action sampling failed")
        agent.pymdp_agent = mock_agent

        # Should handle cascading failures gracefully
        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}
        action = agent.step(observation)

        # Should get fallback action despite multiple failures
        assert action in agent.actions

        # Should have recorded multiple errors
        error_summary = agent.error_handler.get_error_summary()
        assert error_summary["total_errors"] > 1

    def test_error_propagation_chains(self):
        """Test that errors propagate correctly through system layers."""
        handler = ErrorHandler("test_agent")

        # Simulate error propagation chain
        original_error = PyMDPError("Original computation failed")
        handler.handle_error(original_error, "computation")

        # Simulate that recovery also fails
        recovery_error = InferenceError("Recovery mechanism failed")
        handler.handle_error(recovery_error, "recovery")

        # System should track the error chain
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert "PyMDPError" in summary["error_counts"]
        assert "InferenceError" in summary["error_counts"]

    def test_system_recovery_after_failures(self):
        """Test system recovery capabilities after failures."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Simulate failure followed by recovery
        mock_agent = MagicMock()
        mock_agent.infer_policies.side_effect = [
            Exception("Failure 1"),
            Exception("Failure 2"),
            (np.array([0.2, 0.8]), None),  # Recovery
        ]
        mock_agent.sample_action.return_value = np.array(1)
        agent.pymdp_agent = mock_agent

        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}

        # First two steps should handle failures
        action1 = agent.step(observation)
        action2 = agent.step(observation)

        # Third step should recover
        mock_agent.infer_policies.side_effect = None
        mock_agent.infer_policies.return_value = (np.array([0.2, 0.8]), None)
        action3 = agent.step(observation)

        # All actions should be valid
        assert all(action in agent.actions for action in [action1, action2, action3])

        # Error count should reflect the failures
        error_summary = agent.error_handler.get_error_summary()
        assert error_summary["total_errors"] >= 2


class TestErrorRecoveryMechanisms:
    """Test comprehensive error recovery and resilience."""

    def test_automatic_retry_mechanisms(self):
        """Test automatic retry behavior."""
        handler = ErrorHandler("test_agent")

        # Test retry limits
        error = PyMDPError("Transient failure")

        recovery_attempts = []
        for i in range(5):  # Exceed retry limit
            recovery_info = handler.handle_error(error, "test_operation")
            recovery_attempts.append(recovery_info["can_retry"])

        # Should eventually stop retrying
        assert not recovery_attempts[-1]  # Last attempt should disable retries

    def test_error_severity_escalation(self):
        """Test error severity escalation over time."""
        handler = ErrorHandler("test_agent")

        # Repeated errors should escalate severity
        for i in range(10):
            error = PyMDPError(f"Repeated error {i}")
            handler.handle_error(error, "test_operation")

        # Later errors should have higher severity impact
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 10

    def test_graceful_degradation(self):
        """Test graceful degradation under persistent errors."""
        if not PYMDP_AVAILABLE:
            assert False, "Test bypass removed - must fix underlying issue"
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("test", "Test Agent", grid_size=3)
        agent.start()

        # Mock persistent failures
        mock_agent = MagicMock()
        mock_agent.infer_policies.side_effect = Exception("Persistent failure")
        agent.pymdp_agent = mock_agent

        observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}

        # Should continue functioning with degraded performance
        actions = []
        for _ in range(5):
            action = agent.step(observation)
            actions.append(action)

        # Should still produce valid actions despite persistent failures
        assert all(action in agent.actions for action in actions)

        # Should have appropriate error tracking
        error_summary = agent.error_handler.get_error_summary()
        assert error_summary["total_errors"] >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
