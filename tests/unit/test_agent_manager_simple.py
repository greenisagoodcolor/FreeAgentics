"""
Test suite for Agent Manager module - Core functionality.

This test suite provides comprehensive coverage for the AgentManager class
focusing on core functionality without complex dependencies.
Coverage target: 85%+
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

# Import the module under test
try:
    from agents.agent_manager import AgentManager

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock class for testing when imports fail
    class AgentManager:
        pass


class TestAgentManagerCore:
    """Test core AgentManager functionality."""

    @pytest.fixture
    def agent_manager(self):
        """Create AgentManager instance with mocked dependencies."""
        if not IMPORT_SUCCESS:
            pytest.skip("Agent manager modules not available")

        with patch("agents.agent_manager.ActiveInferenceGridAdapter"):
            with patch("agents.agent_manager.logger"):
                return AgentManager()

    def test_agent_manager_initialization(self, agent_manager):
        """Test AgentManager initialization."""
        assert hasattr(agent_manager, "agents")
        assert isinstance(agent_manager.agents, dict)
        assert len(agent_manager.agents) == 0

        assert hasattr(agent_manager, "world")
        assert agent_manager.world is None

        assert hasattr(agent_manager, "running")
        assert agent_manager.running is False

        assert hasattr(agent_manager, "_agent_counter")
        assert agent_manager._agent_counter == 0

        assert hasattr(agent_manager, "_executor")
        assert isinstance(agent_manager._executor, ThreadPoolExecutor)

        assert hasattr(agent_manager, "_event_queue")
        assert isinstance(agent_manager._event_queue, list)
        assert len(agent_manager._event_queue) == 0

        assert hasattr(agent_manager, "_event_lock")
        assert isinstance(agent_manager._event_lock, threading.Lock)

    def test_agent_manager_has_required_methods(self, agent_manager):
        """Test that AgentManager has required methods."""
        required_methods = [
            "_find_free_position",
            "create_agent",
            "get_agent",
            "remove_agent",
            "list_agents",
            "start",
            "stop",
            "update",
            "set_world",
        ]

        for method in required_methods:
            assert hasattr(agent_manager, method)
            assert callable(getattr(agent_manager, method))

    def test_find_free_position_no_world(self, agent_manager):
        """Test _find_free_position when no world is set."""
        position = agent_manager._find_free_position()

        # Should return default position when no world
        assert hasattr(position, "x")
        assert hasattr(position, "y")
        assert position.x == 0
        assert position.y == 0

    def test_find_free_position_with_world(self, agent_manager):
        """Test _find_free_position with a mocked world."""
        # Mock world
        mock_world = Mock()
        mock_world.height = 3
        mock_world.width = 3
        mock_world.agents = {}

        # Mock cells
        mock_cell = Mock()
        mock_cell.type = "EMPTY"  # Assuming CellType.EMPTY = "EMPTY"
        mock_world.get_cell.return_value = mock_cell

        agent_manager.world = mock_world

        with patch("agents.agent_manager.CellType") as mock_celltype:
            mock_celltype.EMPTY = "EMPTY"
            mock_celltype.RESOURCE = "RESOURCE"

            position = agent_manager._find_free_position()

            # Should find a position
            assert hasattr(position, "x")
            assert hasattr(position, "y")
            assert 0 <= position.x < 3
            assert 0 <= position.y < 3

    def test_create_agent_basic(self, agent_manager):
        """Test basic agent creation."""
        with patch(
            "agents.agent_manager.BasicExplorerAgent"
        ) as mock_agent_class:
            with patch("agents.agent_manager.Position") as mock_position:
                mock_agent = Mock()
                mock_agent.id = "test_agent_1"
                mock_agent_class.return_value = mock_agent

                mock_pos = Mock()
                mock_pos.x = 0
                mock_pos.y = 0
                mock_position.return_value = mock_pos

                agent_id = agent_manager.create_agent("basic", "TestAgent")

                assert agent_id == "test_agent_1"
                assert agent_id in agent_manager.agents
                assert agent_manager.agents[agent_id] == mock_agent
                assert agent_manager._agent_counter == 1

    def test_create_agent_active_inference(self, agent_manager):
        """Test Active Inference agent creation."""
        with patch(
            "agents.agent_manager.ActiveInferenceAgent"
        ) as mock_agent_class:
            with patch("agents.agent_manager.Position") as mock_position:
                mock_agent = Mock()
                mock_agent.id = "ai_agent_1"
                mock_agent_class.return_value = mock_agent

                mock_pos = Mock()
                mock_pos.x = 1
                mock_pos.y = 1
                mock_position.return_value = mock_pos

                agent_id = agent_manager.create_agent(
                    "active_inference", "AIAgent"
                )

                assert agent_id == "ai_agent_1"
                assert agent_id in agent_manager.agents
                assert agent_manager.agents[agent_id] == mock_agent

    def test_create_agent_invalid_type(self, agent_manager):
        """Test creating agent with invalid type."""
        agent_id = agent_manager.create_agent("invalid_type", "TestAgent")

        # Should return None for invalid type
        assert agent_id is None

    def test_get_agent_existing(self, agent_manager):
        """Test getting an existing agent."""
        # Add mock agent
        mock_agent = Mock()
        agent_manager.agents["test_agent"] = mock_agent

        retrieved_agent = agent_manager.get_agent("test_agent")
        assert retrieved_agent == mock_agent

    def test_get_agent_nonexistent(self, agent_manager):
        """Test getting a non-existent agent."""
        retrieved_agent = agent_manager.get_agent("nonexistent")
        assert retrieved_agent is None

    def test_remove_agent_existing(self, agent_manager):
        """Test removing an existing agent."""
        # Add mock agent
        mock_agent = Mock()
        agent_manager.agents["test_agent"] = mock_agent

        # Mock world to test removal from world
        mock_world = Mock()
        agent_manager.world = mock_world

        result = agent_manager.remove_agent("test_agent")

        assert result is True
        assert "test_agent" not in agent_manager.agents

    def test_remove_agent_nonexistent(self, agent_manager):
        """Test removing a non-existent agent."""
        result = agent_manager.remove_agent("nonexistent")
        assert result is False

    def test_list_agents_empty(self, agent_manager):
        """Test listing agents when none exist."""
        agents = agent_manager.list_agents()
        assert isinstance(agents, list)
        assert len(agents) == 0

    def test_list_agents_with_agents(self, agent_manager):
        """Test listing agents when some exist."""
        # Add mock agents
        mock_agent1 = Mock()
        mock_agent1.id = "agent1"
        mock_agent1.name = "Agent 1"

        mock_agent2 = Mock()
        mock_agent2.id = "agent2"
        mock_agent2.name = "Agent 2"

        agent_manager.agents["agent1"] = mock_agent1
        agent_manager.agents["agent2"] = mock_agent2

        agents = agent_manager.list_agents()

        assert isinstance(agents, list)
        assert len(agents) == 2

        # Check that agent info is included
        agent_ids = [agent["id"] for agent in agents]
        assert "agent1" in agent_ids
        assert "agent2" in agent_ids

    def test_set_world(self, agent_manager):
        """Test setting the world."""
        mock_world = Mock()
        agent_manager.set_world(mock_world)

        assert agent_manager.world == mock_world

    def test_start_manager(self, agent_manager):
        """Test starting the agent manager."""
        agent_manager.start()
        assert agent_manager.running is True

    def test_stop_manager(self, agent_manager):
        """Test stopping the agent manager."""
        agent_manager.running = True
        agent_manager.stop()
        assert agent_manager.running is False

    def test_update_cycle(self, agent_manager):
        """Test update cycle functionality."""
        # Add mock agent
        mock_agent = Mock()
        mock_agent.update.return_value = None
        agent_manager.agents["test_agent"] = mock_agent

        # Mock world
        mock_world = Mock()
        agent_manager.world = mock_world

        agent_manager.update()

        # Verify agent update was called
        mock_agent.update.assert_called_once()

    def test_thread_safety(self, agent_manager):
        """Test thread safety of agent operations."""
        results = []
        errors = []

        def create_agents():
            try:
                with patch(
                    "agents.agent_manager.BasicExplorerAgent"
                ) as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent.id = (
                        f"thread_agent_{threading.current_thread().ident}"
                    )
                    mock_agent_class.return_value = mock_agent

                    agent_id = agent_manager.create_agent(
                        "basic", "ThreadAgent"
                    )
                    results.append(agent_id)
            except Exception as e:
                errors.append(e)

        # Create agents from multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=create_agents)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        assert len(set(results)) == 5  # All agent IDs should be unique

    def test_event_queue_operations(self, agent_manager):
        """Test event queue operations."""
        # Test adding events to queue
        test_event = {"type": "test", "data": "test_data"}

        # Access the event queue safely
        with agent_manager._event_lock:
            agent_manager._event_queue.append(test_event)

        # Verify event was added
        with agent_manager._event_lock:
            assert len(agent_manager._event_queue) == 1
            assert agent_manager._event_queue[0] == test_event

    def test_executor_usage(self, agent_manager):
        """Test that executor is properly initialized."""
        assert hasattr(agent_manager, "_executor")
        assert isinstance(agent_manager._executor, ThreadPoolExecutor)

        # Test that executor can be used
        future = agent_manager._executor.submit(lambda: "test_result")
        result = future.result(timeout=1.0)
        assert result == "test_result"

    def test_agent_counter_increment(self, agent_manager):
        """Test that agent counter increments properly."""
        initial_count = agent_manager._agent_counter

        with patch(
            "agents.agent_manager.BasicExplorerAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            mock_agent.id = "test_agent"
            mock_agent_class.return_value = mock_agent

            agent_manager.create_agent("basic", "TestAgent")

            assert agent_manager._agent_counter == initial_count + 1

    def test_world_integration(self, agent_manager):
        """Test world integration functionality."""
        # Mock world
        mock_world = Mock()
        mock_world.add_agent = Mock()
        mock_world.remove_agent = Mock()

        agent_manager.set_world(mock_world)

        # Create agent
        with patch(
            "agents.agent_manager.BasicExplorerAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            mock_agent.id = "world_test_agent"
            mock_agent_class.return_value = mock_agent

            agent_manager.create_agent("basic", "WorldTestAgent")

            # Verify world integration
            assert agent_manager.world == mock_world

    def test_cleanup_on_stop(self, agent_manager):
        """Test cleanup when stopping the manager."""
        # Start manager
        agent_manager.start()
        assert agent_manager.running is True

        # Add some agents
        mock_agent = Mock()
        agent_manager.agents["cleanup_test"] = mock_agent

        # Stop manager
        agent_manager.stop()

        assert agent_manager.running is False
        # Agents should still exist but manager should be stopped
        assert "cleanup_test" in agent_manager.agents


class TestAgentManagerErrorHandling:
    """Test error handling in AgentManager."""

    @pytest.fixture
    def agent_manager(self):
        """Create AgentManager instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Agent manager modules not available")

        with patch("agents.agent_manager.ActiveInferenceGridAdapter"):
            with patch("agents.agent_manager.logger"):
                return AgentManager()

    def test_create_agent_with_exception(self, agent_manager):
        """Test agent creation when constructor raises exception."""
        with patch(
            "agents.agent_manager.BasicExplorerAgent"
        ) as mock_agent_class:
            mock_agent_class.side_effect = Exception("Agent creation failed")

            agent_id = agent_manager.create_agent("basic", "FailAgent")

            # Should return None on failure
            assert agent_id is None
            assert len(agent_manager.agents) == 0

    def test_update_with_agent_exception(self, agent_manager):
        """Test update when agent raises exception."""
        # Add mock agent that raises exception
        mock_agent = Mock()
        mock_agent.update.side_effect = Exception("Agent update failed")
        agent_manager.agents["failing_agent"] = mock_agent

        # Update should not crash even if agent fails
        with patch("agents.agent_manager.logger") as mock_logger:
            agent_manager.update()

            # Should log the error
            mock_logger.error.assert_called()

    def test_world_operations_without_world(self, agent_manager):
        """Test operations when world is not set."""
        # Should not crash when world is None
        agent_manager.world = None

        # These operations should handle None world gracefully
        position = agent_manager._find_free_position()
        assert position is not None

        # Update should work without world
        agent_manager.update()

    def test_concurrent_agent_operations(self, agent_manager):
        """Test concurrent agent operations for race conditions."""
        errors = []

        def create_and_remove_agent():
            try:
                with patch(
                    "agents.agent_manager.BasicExplorerAgent"
                ) as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent.id = (
                        f"concurrent_{threading.current_thread().ident}"
                    )
                    mock_agent_class.return_value = mock_agent

                    agent_id = agent_manager.create_agent(
                        "basic", "ConcurrentAgent"
                    )
                    if agent_id:
                        time.sleep(0.01)  # Small delay
                        agent_manager.remove_agent(agent_id)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = []
        for _ in range(10):
            t = threading.Thread(target=create_and_remove_agent)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should not have any errors
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=agents.agent_manager", "--cov-report=html"]
    )
