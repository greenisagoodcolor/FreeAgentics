"""Security-critical tests for agent manager following TDD principles.

This test suite covers core agent functionality:
- Agent creation and initialization
- State management and lifecycle
- Error handling and recovery
- Resource management
- Security boundaries
"""

import uuid
from unittest.mock import patch

import pytest

from agents.agent_manager import AgentManager
from database.models import AgentStatus


class TestAgentCreation:
    """Test secure agent creation and initialization."""

    @pytest.fixture
    def agent_manager(self):
        """Create agent manager instance."""
        with patch("agents.agent_manager.get_enhanced_db_manager"):
            manager = AgentManager()
            yield manager

    def test_create_agent_with_valid_config(self, agent_manager):
        """Test creating an agent with valid configuration."""
        # Arrange
        config = {
            "name": "test-agent",
            "template": "basic",
            "model_type": "exploration",
            "initial_state": 0,
            "preferences": [1.0, 0.0],
        }

        # Act
        agent_id = agent_manager.create_agent(config)

        # Assert
        assert agent_id is not None
        assert isinstance(agent_id, str)
        assert len(agent_id) == 36  # UUID format

        # Verify agent was stored
        agent = agent_manager.get_agent(agent_id)
        assert agent is not None
        assert agent.name == "test-agent"
        assert agent.status == AgentStatus.PENDING

    def test_create_agent_validates_required_fields(self, agent_manager):
        """Test that agent creation validates required fields."""
        # Arrange - Missing required 'name' field
        invalid_config = {"template": "basic"}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent_manager.create_agent(invalid_config)

        assert "name" in str(exc_info.value).lower()

    def test_create_agent_sanitizes_input(self, agent_manager):
        """Test that agent creation sanitizes dangerous input."""
        # Arrange - Config with potentially dangerous values
        config = {
            "name": "<script>alert('xss')</script>",
            "template": "basic",
            "metadata": {"sql": "'; DROP TABLE agents; --"},
        }

        # Act
        agent_id = agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)

        # Assert - Name should be sanitized
        assert "<script>" not in agent.name
        assert "DROP TABLE" not in str(agent.metadata)

    def test_create_agent_enforces_resource_limits(self, agent_manager):
        """Test that agent creation enforces resource limits."""
        # Arrange - Create max allowed agents
        max_agents = 100  # Assumed limit

        # Create agents up to limit
        for i in range(max_agents):
            config = {"name": f"agent-{i}", "template": "basic"}
            agent_manager.create_agent(config)

        # Act & Assert - Next creation should fail
        with pytest.raises(Exception) as exc_info:
            agent_manager.create_agent({"name": "overflow", "template": "basic"})

        assert "limit" in str(exc_info.value).lower()


class TestAgentLifecycle:
    """Test agent lifecycle management and state transitions."""

    @pytest.fixture
    def agent_manager(self):
        """Create agent manager with a test agent."""
        with patch("agents.agent_manager.get_enhanced_db_manager"):
            manager = AgentManager()

            # Create test agent
            config = {"name": "lifecycle-test", "template": "basic"}
            agent_id = manager.create_agent(config)

            yield manager, agent_id

    def test_agent_initialization(self, agent_manager):
        """Test agent initialization from pending to active."""
        manager, agent_id = agent_manager

        # Act
        success = manager.initialize_agent(agent_id)

        # Assert
        assert success is True
        agent = manager.get_agent(agent_id)
        assert agent.status == AgentStatus.ACTIVE
        assert agent.initialized_at is not None

    def test_agent_initialization_idempotent(self, agent_manager):
        """Test that initialization is idempotent."""
        manager, agent_id = agent_manager

        # Initialize once
        manager.initialize_agent(agent_id)
        agent1 = manager.get_agent(agent_id)
        init_time1 = agent1.initialized_at

        # Initialize again
        manager.initialize_agent(agent_id)
        agent2 = manager.get_agent(agent_id)
        init_time2 = agent2.initialized_at

        # Assert - Should not reinitialize
        assert init_time1 == init_time2
        assert agent2.status == AgentStatus.ACTIVE

    def test_agent_pause_resume(self, agent_manager):
        """Test pausing and resuming agents."""
        manager, agent_id = agent_manager

        # Initialize agent first
        manager.initialize_agent(agent_id)

        # Act - Pause
        success = manager.pause_agent(agent_id)

        # Assert
        assert success is True
        agent = manager.get_agent(agent_id)
        assert agent.status == AgentStatus.PAUSED

        # Act - Resume
        success = manager.resume_agent(agent_id)

        # Assert
        assert success is True
        agent = manager.get_agent(agent_id)
        assert agent.status == AgentStatus.ACTIVE

    def test_agent_termination(self, agent_manager):
        """Test safe agent termination."""
        manager, agent_id = agent_manager

        # Initialize agent
        manager.initialize_agent(agent_id)

        # Act
        success = manager.terminate_agent(agent_id)

        # Assert
        assert success is True
        agent = manager.get_agent(agent_id)
        assert agent.status == AgentStatus.TERMINATED
        assert agent.terminated_at is not None

        # Verify cannot perform operations on terminated agent
        assert manager.pause_agent(agent_id) is False
        assert manager.resume_agent(agent_id) is False

    def test_delete_agent_cleanup(self, agent_manager):
        """Test that agent deletion properly cleans up resources."""
        manager, agent_id = agent_manager

        # Act
        success = manager.delete_agent(agent_id)

        # Assert
        assert success is True
        assert manager.get_agent(agent_id) is None

        # Verify agent is removed from internal tracking
        assert agent_id not in manager._agents


class TestAgentMessageHandling:
    """Test secure message handling and processing."""

    @pytest.fixture
    def active_agent(self):
        """Create an active agent for testing."""
        with patch("agents.agent_manager.get_enhanced_db_manager"):
            manager = AgentManager()

            config = {"name": "message-test", "template": "basic"}
            agent_id = manager.create_agent(config)
            manager.initialize_agent(agent_id)

            yield manager, agent_id

    def test_handle_valid_observation(self, active_agent):
        """Test handling valid observations."""
        manager, agent_id = active_agent

        # Arrange
        observation = {
            "type": "state_update",
            "data": {"sensor": "temperature", "value": 25.5},
        }

        # Act
        result = manager.handle_observation(agent_id, observation)

        # Assert
        assert result is not None
        assert "action" in result
        assert "confidence" in result

    def test_handle_invalid_observation_format(self, active_agent):
        """Test that invalid observation formats are rejected."""
        manager, agent_id = active_agent

        # Arrange - Invalid observation
        invalid_observations = [
            None,
            "",
            [],
            {"no_type": "field"},
            {"type": "x" * 1000},  # Too long
        ]

        for invalid_obs in invalid_observations:
            # Act & Assert
            with pytest.raises(ValueError):
                manager.handle_observation(agent_id, invalid_obs)

    def test_handle_observation_for_paused_agent(self, active_agent):
        """Test that paused agents don't process observations."""
        manager, agent_id = active_agent

        # Pause agent
        manager.pause_agent(agent_id)

        # Arrange
        observation = {"type": "state_update", "data": {}}

        # Act
        result = manager.handle_observation(agent_id, observation)

        # Assert
        assert result is None or result.get("status") == "paused"

    def test_concurrent_message_handling(self, active_agent):
        """Test thread-safe concurrent message handling."""
        manager, agent_id = active_agent

        # Arrange
        import threading

        results = []
        errors = []

        def send_observation(obs_id):
            try:
                observation = {"type": "concurrent_test", "data": {"id": obs_id}}
                result = manager.handle_observation(agent_id, observation)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Act - Send multiple concurrent observations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=send_observation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0
        assert len(results) == 10


class TestAgentErrorHandling:
    """Test agent error handling and recovery."""

    @pytest.fixture
    def agent_with_errors(self):
        """Create agent manager with error injection."""
        with patch("agents.agent_manager.get_enhanced_db_manager"):
            manager = AgentManager()
            yield manager

    def test_handle_agent_creation_error(self, agent_with_errors):
        """Test graceful handling of agent creation errors."""
        manager = agent_with_errors

        # Arrange - Inject error
        with patch.object(
            manager, "_create_agent_instance", side_effect=Exception("Creation failed")
        ):
            # Act
            result = manager.create_agent({"name": "fail", "template": "basic"})

        # Assert
        assert result is None or "error" in result

    def test_handle_agent_not_found(self, agent_with_errors):
        """Test handling requests for non-existent agents."""
        manager = agent_with_errors
        fake_id = str(uuid.uuid4())

        # Act & Assert
        assert manager.get_agent(fake_id) is None
        assert manager.initialize_agent(fake_id) is False
        assert manager.pause_agent(fake_id) is False
        assert manager.delete_agent(fake_id) is False

    def test_recover_from_processing_error(self, agent_with_errors):
        """Test that agents can recover from processing errors."""
        manager = agent_with_errors

        # Create and initialize agent
        config = {"name": "error-recovery", "template": "basic"}
        agent_id = manager.create_agent(config)
        manager.initialize_agent(agent_id)

        # Inject processing error
        agent = manager.get_agent(agent_id)
        with patch.object(
            agent, "process_observation", side_effect=Exception("Process error")
        ):
            # Act - Should handle error gracefully
            manager.handle_observation(agent_id, {"type": "test", "data": {}})

        # Assert - Agent should still be active
        assert manager.get_agent(agent_id).status == AgentStatus.ACTIVE

        # Verify can still process after error
        result2 = manager.handle_observation(agent_id, {"type": "test2", "data": {}})
        assert result2 is not None


class TestAgentSecurity:
    """Test agent security boundaries and isolation."""

    @pytest.fixture
    def secure_manager(self):
        """Create agent manager with security focus."""
        with patch("agents.agent_manager.get_enhanced_db_manager"):
            manager = AgentManager()
            yield manager

    def test_agent_isolation(self, secure_manager):
        """Test that agents are isolated from each other."""
        # Create two agents
        config1 = {"name": "agent1", "template": "basic"}
        config2 = {"name": "agent2", "template": "basic"}

        agent1_id = secure_manager.create_agent(config1)
        agent2_id = secure_manager.create_agent(config2)

        # Initialize both
        secure_manager.initialize_agent(agent1_id)
        secure_manager.initialize_agent(agent2_id)

        # Act - Send observation to agent1
        obs = {"type": "test", "data": {"secret": "agent1-data"}}
        secure_manager.handle_observation(agent1_id, obs)

        # Assert - Agent2 should not have access to agent1's data
        agent2 = secure_manager.get_agent(agent2_id)
        assert "agent1-data" not in str(agent2.__dict__)

    def test_prevent_agent_privilege_escalation(self, secure_manager):
        """Test that agents cannot escalate privileges."""
        # Create regular agent
        config = {"name": "regular-agent", "template": "basic", "role": "user"}
        agent_id = secure_manager.create_agent(config)

        # Attempt to escalate privileges
        malicious_obs = {
            "type": "update_config",
            "data": {"role": "admin", "permissions": ["all"]},
        }

        # Act
        secure_manager.handle_observation(agent_id, malicious_obs)

        # Assert - Role should not change
        agent = secure_manager.get_agent(agent_id)
        assert agent.role != "admin"
        assert "all" not in getattr(agent, "permissions", [])

    def test_resource_exhaustion_protection(self, secure_manager):
        """Test protection against resource exhaustion attacks."""
        # Create agent
        config = {"name": "resource-test", "template": "basic"}
        agent_id = secure_manager.create_agent(config)
        secure_manager.initialize_agent(agent_id)

        # Attempt to exhaust resources with large observation
        huge_obs = {
            "type": "process",
            "data": {"array": [0] * 10_000_000},
        }  # 10M elements

        # Act - Should handle without crashing
        try:
            result = secure_manager.handle_observation(agent_id, huge_obs)
            # Should either process with limits or reject
            assert result is not None or "error" in result
        except MemoryError:
            # Acceptable - prevented memory exhaustion
            pass
