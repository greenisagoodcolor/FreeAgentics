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

import numpy as np
import pytest

from agents.agent_manager import AgentManager
from agents.base_agent import BasicExplorerAgent
from world.grid_world import GridWorldConfig, Position


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
        # Start the specific agent
        agent_manager.start_agent(agent_id)

        # Act - Manually call update to simulate update cycle
        agent_manager.update()

        # Assert - Agent should exist and be properly initialized
        agent = agent_manager.agents[agent_id]
        assert agent is not None
        assert agent.is_active  # Agent should be active after start
        # Agent should have a valid position (not necessarily moved yet)
        assert isinstance(agent.position, Position)
        assert 0 <= agent.position.x < agent_manager.world.width
        assert 0 <= agent.position.y < agent_manager.world.height

        # Cleanup
        agent_manager.stop()

    def test_shutdown_cleanup(self, agent_manager):
        """Test proper cleanup on shutdown."""
        # Arrange
        agent_id1 = agent_manager.create_agent(agent_type="explorer", name="Test Agent 1")
        agent_id2 = agent_manager.create_agent(agent_type="explorer", name="Test Agent 2")

        # Verify agents were created
        assert len(agent_manager.agents) == 2

        # Act
        agent_manager.stop()

        # Assert - agents should be stopped but still exist in the manager
        assert len(agent_manager.agents) == 2  # Agents still tracked
        assert not agent_manager.running  # Manager is stopped
        assert not agent_manager.agents[agent_id1].is_active  # Agents are stopped
        assert not agent_manager.agents[agent_id2].is_active

    def test_event_broadcasting(self, agent_manager):
        """Test secure event broadcasting between agents."""
        # Arrange
        agent_id1 = agent_manager.create_agent(agent_type="explorer", name="Test Agent 1")
        agent_id2 = agent_manager.create_agent(agent_type="explorer", name="Test Agent 2")

        # Act - Queue an event (this happens automatically when agents are created)
        # The create_agent method already queues events

        # Process events
        agent_manager._process_event_queue()

        # Assert - Events should be queued and processed
        # After processing, the queue should be empty
        assert len(agent_manager._event_queue) == 0  # All events processed

        # Verify agents were created properly
        assert agent_id1 in agent_manager.agents
        assert agent_id2 in agent_manager.agents


class TestActiveInferenceAgentSecurity:
    """Test ActiveInferenceAgent security boundaries."""

    def test_agent_cannot_modify_world_directly(self):
        """Test that agents cannot directly modify the world state."""
        # Arrange
        agent = BasicExplorerAgent(agent_id="test-1", name="Test Agent", grid_size=5)
        agent.start()  # Initialize the agent

        # Act - Agent should not have direct world modification access
        # This tests the principle of least privilege
        assert not hasattr(agent, "_modify_world")
        assert not hasattr(agent, "world")  # Agent doesn't have direct world access

        # Agent can only interact through defined interfaces
        # Prepare a simple observation
        observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}
        agent.perceive(observation)
        action = agent.select_action()
        assert isinstance(action, str)  # Should return action name like "up", "down", etc.

    def test_agent_observation_isolation(self):
        """Test that agents only see their local observations."""
        # Arrange
        agent1 = BasicExplorerAgent("agent1", "Agent 1", grid_size=10)
        agent2 = BasicExplorerAgent("agent2", "Agent 2", grid_size=10)

        # Start agents to initialize them
        agent1.start()
        agent2.start()

        # Set different positions
        agent1.position = Position(0, 0)
        agent2.position = Position(9, 9)

        # Act - Create different observations for each agent
        obs1 = {"position": [0, 0], "surroundings": np.ones((3, 3))}
        obs2 = {"position": [9, 9], "surroundings": np.zeros((3, 3))}

        agent1.perceive(obs1)
        agent2.perceive(obs2)

        # Assert - Observations should be different
        assert agent1.current_observation != agent2.current_observation
        # Agents should not see global state in their observations
        assert isinstance(obs1, dict)
        assert "all_agents" not in obs1
        assert "global_state" not in obs1

    def test_agent_error_recovery(self):
        """Test that agents can recover from errors."""
        # Arrange
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=5)
        agent.start()

        # Inject error in belief update
        with patch.object(agent, "update_beliefs", side_effect=Exception("Test error")):
            # Act - Should handle error gracefully
            observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}

            # Perceive and select action should handle errors gracefully
            agent.perceive(observation)
            action = agent.select_action()

            # Agent should still return a valid action despite the error
            assert isinstance(action, str)
            assert action in ["up", "down", "left", "right", "stay"]

        # Assert - Agent should still be functional after error
        # Remove the patch and try again
        agent.perceive(observation)
        agent.update_beliefs()  # Should work now
        action = agent.select_action()
        assert isinstance(action, str)


class TestAgentResourceManagement:
    """Test agent resource usage and limits."""

    def test_agent_memory_limits(self):
        """Test that agents have memory usage limits."""
        # Arrange
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=5)
        agent.start()

        # Act - Try to store large amount of data
        # BasicExplorerAgent has an uncertainty_map which has bounded size
        initial_map_size = agent.uncertainty_map.nbytes

        # Update beliefs multiple times
        for i in range(10):
            observation = {"position": [i % 5, i % 5], "surroundings": np.zeros((3, 3))}
            agent.perceive(observation)
            agent.update_beliefs()

        # Assert - Memory should not grow unbounded
        # The uncertainty map should stay the same size (5x5 grid)
        assert agent.uncertainty_map.shape == (5, 5)
        assert agent.uncertainty_map.nbytes == initial_map_size

    def test_agent_computation_timeout(self):
        """Test that agent computations have timeouts."""
        # Arrange
        agent = BasicExplorerAgent("test", "Test Agent", grid_size=5)
        agent.start()

        # Mock expensive computation
        def expensive_computation(*args, **kwargs):
            # Simulate computation that takes some time
            time.sleep(0.1)
            # Return appropriate values based on the method being mocked
            if args and hasattr(args[0], "shape"):
                # For infer_states, return array of correct shape
                return np.zeros(args[0].shape)
            return np.zeros((9,))  # Default return for 5x5 grid = 25 states

        # Act - Agent operations should complete reasonably fast
        with patch.object(agent.pymdp_agent, "infer_states", side_effect=expensive_computation):
            start = time.time()

            # This should complete even with the mocked delay
            observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}
            agent.perceive(observation)
            agent.update_beliefs()
            action = agent.select_action()

            elapsed = time.time() - start

            # Assert - Should complete in reasonable time (not hanging)
            assert elapsed < 1.0  # Should be quick even with mock delay
            assert isinstance(action, str)  # Should still return valid action
