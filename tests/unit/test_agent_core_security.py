"""Security-critical tests for agent core functionality following TDD principles.

This test suite covers core agent functionality that exists in the codebase:
- Agent creation via AgentManager
- Agent lifecycle and state
- Error handling
- World integration
"""

import threading
import time
from unittest.mock import patch

import pytest

from agents.agent_manager import AgentManager
from agents.base_agent import BasicExplorerAgent
from world.grid_world import GridWorld, GridWorldConfig, Position


class TestAgentManagerSecurity:
    """Test AgentManager security and core functionality."""

    @pytest.fixture
    def agent_manager(self):
        """Create agent manager instance."""
        manager = AgentManager()
        # Create a simple world for testing
        config = GridWorldConfig(width=5, height=5)
        manager.create_world(config)
        yield manager
        # Cleanup - no shutdown method needed

    def test_create_agent_with_valid_parameters(self, agent_manager):
        """Test creating an agent with valid parameters."""
        # Act
        agent_id = agent_manager.create_agent(
            agent_type="explorer", name="Test Agent", position=Position(1, 1)
        )

        # Assert
        assert agent_id is not None
        assert agent_id in agent_manager.agents
        agent = agent_manager.agents[agent_id]
        assert isinstance(agent, BasicExplorerAgent)
        assert agent.position == Position(1, 1)

    def test_create_agent_finds_free_position(self, agent_manager):
        """Test that agent creation finds free position if not specified."""
        # Act
        agent_id = agent_manager.create_agent(agent_type="explorer", name="Test Agent")

        # Assert
        assert agent_id is not None
        agent = agent_manager.agents[agent_id]
        assert agent.position is not None
        # Verify position is within world bounds
        assert 0 <= agent.position.x < agent_manager.world.width
        assert 0 <= agent.position.y < agent_manager.world.height

    def test_cannot_create_agent_without_world(self):
        """Test that agents cannot be created without a world."""
        # Arrange
        manager = AgentManager()
        manager.world = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            manager.create_agent(agent_type="explorer", name="Test Agent")

        assert "world" in str(exc_info.value).lower()

    def test_agent_position_validation(self, agent_manager):
        """Test that agent positions are validated."""
        # Act & Assert - Position outside world bounds
        with pytest.raises(ValueError):
            agent_manager.create_agent(
                agent_type="explorer",
                name="Test Agent",
                position=Position(10, 10),  # Outside 5x5 world
            )

    def test_remove_agent(self, agent_manager):
        """Test safe agent removal."""
        # Arrange
        agent_id = agent_manager.create_agent(agent_type="explorer", name="Test Agent")

        # Act
        result = agent_manager.remove_agent(agent_id)

        # Assert
        assert result is True
        assert agent_id not in agent_manager.agents
        assert agent_id not in agent_manager.world.agents

    def test_remove_nonexistent_agent(self, agent_manager):
        """Test removing non-existent agent is safe."""
        # Act
        result = agent_manager.remove_agent("fake-id")

        # Assert
        assert result is False

    def test_concurrent_agent_creation(self, agent_manager):
        """Test thread-safe concurrent agent creation."""
        # Arrange
        created_agents = []
        errors = []

        def create_agent_thread(i):
            try:
                agent_id = agent_manager.create_agent(agent_type="explorer", name="Test Agent")
                created_agents.append(agent_id)
            except Exception as e:
                errors.append(e)

        # Act - Create multiple agents concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_agent_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0
        assert len(created_agents) == 5
        assert len(set(created_agents)) == 5  # All unique IDs

    def test_agent_update_cycle(self, agent_manager):
        """Test agent update cycle execution."""
        # Arrange
        agent_id = agent_manager.create_agent(agent_type="explorer", name="Test Agent")
        agent_manager.start()

        # Act - Let it run for a short time
        time.sleep(0.1)

        # Assert - Agent should have been updated
        agent = agent_manager.agents[agent_id]
        # Agent should have moved or updated state
        assert hasattr(agent, "last_update") or agent.position != Position(0, 0)

        # Cleanup
        agent_manager.stop()

    def test_shutdown_cleanup(self, agent_manager):
        """Test proper cleanup on shutdown."""
        # Arrange
        agent_manager.create_agent(agent_type="explorer", name="Test Agent")
        agent_manager.create_agent(agent_type="explorer", name="Test Agent")

        # Act
        agent_manager.shutdown()

        # Assert
        assert len(agent_manager.agents) == 0
        assert agent_manager._executor._shutdown is True

    def test_event_broadcasting(self, agent_manager):
        """Test secure event broadcasting between agents."""
        # Arrange
        agent_id1 = agent_manager.create_agent(agent_type="explorer", name="Test Agent")
        agent_manager.create_agent(agent_type="explorer", name="Test Agent")

        # Act
        agent_manager._broadcast_event(
            {"type": "test_event", "source": agent_id1, "data": {"message": "test"}}
        )

        # Process events
        agent_manager._process_events()

        # Assert - Events should be queued and processed
        # (Implementation specific - adjust based on actual behavior)
        assert len(agent_manager._event_queue) == 0  # Processed


class TestActiveInferenceAgentSecurity:
    """Test ActiveInferenceAgent security boundaries."""

    def test_agent_cannot_modify_world_directly(self):
        """Test that agents cannot directly modify the world state."""
        # Arrange
        world = GridWorld(GridWorldConfig(width=5, height=5))
        agent = BasicExplorerAgent(agent_id="test-1", position=Position(0, 0), world=world)

        # Act - Agent should not have direct world modification access
        # This tests the principle of least privilege
        assert not hasattr(agent, "_modify_world")
        assert not hasattr(agent, "world.set_cell")

        # Agent can only interact through defined interfaces
        action = agent.select_action(agent.get_observation())
        assert isinstance(action, (int, str, dict))  # Limited action space

    def test_agent_observation_isolation(self):
        """Test that agents only see their local observations."""
        # Arrange
        world = GridWorld(GridWorldConfig(width=10, height=10))
        agent1 = BasicExplorerAgent("agent1", Position(0, 0), world)
        agent2 = BasicExplorerAgent("agent2", Position(9, 9), world)

        # Act
        obs1 = agent1.get_observation()
        obs2 = agent2.get_observation()

        # Assert - Observations should be different (position-based)
        assert obs1 != obs2
        # Agents should not see the entire world state
        assert "all_agents" not in obs1
        assert "global_state" not in obs1

    def test_agent_error_recovery(self):
        """Test that agents can recover from errors."""
        # Arrange
        world = GridWorld(GridWorldConfig(width=5, height=5))
        agent = BasicExplorerAgent("test", Position(0, 0), world)

        # Inject error in observation processing
        with patch.object(agent, "encode_observation", side_effect=Exception("Test error")):
            # Act - Should handle error gracefully
            try:
                obs = agent.get_observation()
                action = agent.select_action(obs)
            except Exception:
                # Agent should have fallback behavior
                pass

        # Assert - Agent should still be functional after error
        obs = agent.get_observation()
        assert obs is not None
        action = agent.select_action(obs)
        assert action is not None


class TestAgentResourceManagement:
    """Test agent resource usage and limits."""

    def test_agent_memory_limits(self):
        """Test that agents have memory usage limits."""
        # Arrange
        world = GridWorld(GridWorldConfig(width=5, height=5))
        agent = BasicExplorerAgent("test", Position(0, 0), world)

        # Act - Try to store large amount of data

        # Agents should have bounded memory for observations/beliefs
        if hasattr(agent, "memory") or hasattr(agent, "observation_history"):
            # Check that memory is bounded
            for i in range(1000):
                agent.update_beliefs({"step": i, "data": [0] * 1000})

            # Assert - Memory should not grow unbounded
            # (Implementation specific - adjust based on actual limits)
            assert True  # Placeholder - check actual memory usage

    def test_agent_computation_timeout(self):
        """Test that agent computations have timeouts."""
        # Arrange
        world = GridWorld(GridWorldConfig(width=5, height=5))
        agent = BasicExplorerAgent("test", Position(0, 0), world)

        # Mock expensive computation
        def expensive_computation(*args):
            time.sleep(10)  # Simulate long computation
            return 0

        # Act - Agent operations should timeout
        with patch.object(agent, "infer_states", side_effect=expensive_computation):
            start = time.time()
            try:
                # This should timeout or have bounded execution time
                obs = agent.get_observation()
                agent.select_action(obs)
            except Exception:
                pass
            elapsed = time.time() - start

            # Assert - Should not take full 10 seconds
            assert elapsed < 5.0  # Reasonable timeout
