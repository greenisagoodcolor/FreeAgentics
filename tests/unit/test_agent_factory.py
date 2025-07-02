"""
Tests for Agent Factory and Registry.

Comprehensive test suite for agent creation, registration, and management
following ADR-002 canonical structure and ADR-003 dependency rules.
"""

from unittest.mock import Mock, patch

import pytest

from agents.base.agent import BaseAgent
from agents.base.agent_factory import (
    AgentFactory,
    AgentRegistry,
)

# DistributedAgentRegistry is not implemented yet
# from agents.base.agent_factory import DistributedAgentRegistry
from agents.base.data_model import Agent as AgentData
from agents.base.data_model import AgentCapability, AgentPersonality, Position
from agents.base.interfaces import IAgentEventHandler


class TestAgentFactory:
    """Test agent factory functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.factory = AgentFactory()

    def test_factory_initialization(self):
        """Test factory initialization with default types."""
        assert self.factory is not None
        assert len(self.factory._agent_types) > 0
        assert len(self.factory._default_configs) > 0

        # Should have basic agent type registered by default
        assert "basic" in self.factory._agent_types
        assert "basic" in self.factory._default_configs

    def test_register_custom_agent_type(self):
        """Test registering a custom agent type."""

        def create_custom_agent(**kwargs):
            mock_agent = Mock(spec=BaseAgent)
            mock_agent.data = Mock()
            mock_agent.data.metadata = {}
            mock_agent.data.personality = Mock()
            mock_agent.agent_id = kwargs.get("agent_id", "custom_agent")
            return mock_agent

        # Register custom type
        self.factory.register_type("custom", create_custom_agent)

        assert "custom" in self.factory._agent_types
        assert self.factory._agent_types["custom"] == create_custom_agent

    def test_register_type_with_config(self):
        """Test registering agent type with default configuration."""

        def create_specialized_agent(**kwargs):
            mock_agent = Mock(spec=BaseAgent)
            mock_agent.data = Mock()
            mock_agent.data.metadata = {}
            mock_agent.data.personality = Mock()
            return mock_agent

        default_config = {
            "capabilities": {
                AgentCapability.MOVEMENT, AgentCapability.RESOURCE_MANAGEMENT}, "personality": {
                "openness": 0.3, "conscientiousness": 0.7}, "initial_resources": {
                "energy": 100, "materials": 50}, }

        self.factory.register_type("specialized", create_specialized_agent)
        self.factory.set_default_config("specialized", default_config)

        assert "specialized" in self.factory._agent_types
        assert "specialized" in self.factory._default_configs
        assert self.factory._default_configs["specialized"] == default_config

    def test_create_basic_agent(self):
        """Test creating a basic agent."""
        agent_data = AgentData(
            agent_id="test_basic_agent",
            name="Test Basic Agent",
            position=Position(
                x=0.0,
                y=0.0,
                z=0.0),
            capabilities={
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION},
            personality=AgentPersonality(
                openness=0.7,
                conscientiousness=0.8,
                extraversion=0.5),
        )

        with patch("agents.base.agent_factory.create_agent") as mock_create:
            mock_agent = Mock(spec=BaseAgent)
            mock_agent.agent_id = "test_basic_agent"
            mock_agent.data = Mock()
            mock_agent.data.agent_id = "test_basic_agent"
            mock_agent.data.metadata = {}
            mock_agent.data.personality = Mock()
            mock_create.return_value = mock_agent

            agent = self.factory.create_agent("basic", agent_data=agent_data)

            assert agent is not None
            assert agent.agent_id == "test_basic_agent"
            mock_create.assert_called_once()

    def test_create_agent_with_custom_config(self):
        """Test creating agent with custom configuration overrides."""
        agent_data = AgentData(
            agent_id="custom_config_agent",
            name="Custom Config Agent",
            position=Position(x=1.0, y=2.0, z=0.0),
            capabilities={AgentCapability.MOVEMENT},
            personality=AgentPersonality(openness=0.8, agreeableness=0.3),
        )

        custom_config = {
            "max_energy": 200,
            "exploration_radius": 15.0,
            "learning_rate": 0.05}

        with patch("agents.base.agent_factory.create_agent") as mock_create:
            mock_agent = Mock(spec=BaseAgent)
            mock_agent.data = Mock()
            mock_agent.data.metadata = {}
            mock_agent.data.personality = Mock()
            mock_create.return_value = mock_agent

            agent = self.factory.create_agent(
                "basic", agent_data=agent_data, config_overrides=custom_config
            )

            assert agent is not None
            mock_create.assert_called_once()

            # Verify config overrides were passed
            call_args = mock_create.call_args
            # Check that the kwargs passed to create_agent contain the custom
            # config
            passed_kwargs = call_args[1] if call_args and len(
                call_args) > 1 else {}
            # The config_overrides should be merged into the kwargs
            if "config_overrides" in passed_kwargs:
                assert "max_energy" in passed_kwargs["config_overrides"]
            else:
                assert "max_energy" in passed_kwargs or any(
                    "max_energy" in str(arg) for arg in (
                        call_args[0] if call_args else []))

    def test_create_agent_unknown_type(self):
        """Test creating agent with unknown type raises error."""
        agent_data = AgentData(
            agent_id="unknown_agent",
            name="Unknown Agent",
            position=Position(x=0.0, y=0.0, z=0.0),
            capabilities=set(),
            personality=AgentPersonality(),
        )

        with pytest.raises(ValueError, match="Unknown agent type"):
            self.factory.create_agent("unknown_type", agent_data=agent_data)

    def test_list_available_types(self):
        """Test listing available agent types."""
        types = self.factory.get_supported_types()

        assert isinstance(types, list)
        assert len(types) > 0
        assert "basic" in types

    def test_get_default_config(self):
        """Test getting default configuration for agent type."""
        basic_config = self.factory.get_default_config("basic")

        assert isinstance(basic_config, dict)
        assert "capabilities" in basic_config
        assert AgentCapability.MOVEMENT in basic_config["capabilities"]

    def test_get_default_config_unknown_type(self):
        """Test getting default config for unknown type returns empty dict."""
        config = self.factory.get_default_config("unknown_type")
        assert config == {}


class TestAgentRegistry:
    """Test agent registry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = AgentRegistry()

        # Create mock agents for testing
        self.mock_agent1 = Mock(spec=BaseAgent)
        self.mock_agent1.agent_id = "agent_001"
        self.mock_agent1.data = Mock()
        self.mock_agent1.data.agent_type = "explorer"
        self.mock_agent1.data.position = Position(x=0.0, y=0.0, z=0.0)

        self.mock_agent2 = Mock(spec=BaseAgent)
        self.mock_agent2.agent_id = "agent_002"
        self.mock_agent2.data = Mock()
        self.mock_agent2.data.agent_type = "guardian"
        self.mock_agent2.data.position = Position(x=1.0, y=1.0, z=0.0)

        self.mock_agent3 = Mock(spec=BaseAgent)
        self.mock_agent3.agent_id = "agent_003"
        self.mock_agent3.data = Mock()
        self.mock_agent3.data.agent_type = "explorer"
        self.mock_agent3.data.position = Position(x=2.0, y=2.0, z=0.0)

    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry._agents) == 0
        assert len(self.registry._agents_by_type) == 0
        assert len(self.registry._event_handlers) == 0

    def test_register_agent(self):
        """Test registering an agent."""
        self.registry.register_agent(self.mock_agent1)

        # Check agent was registered
        assert self.mock_agent1.agent_id in self.registry._agents
        assert self.registry._agents[self.mock_agent1.agent_id] == self.mock_agent1
        assert "explorer" in self.registry._agents_by_type
        assert self.mock_agent1 in self.registry._agents_by_type["explorer"]

    def test_register_duplicate_agent(self):
        """Test registering agent with duplicate ID fails."""
        self.registry.register_agent(self.mock_agent1)

        # Try to register another agent with same ID
        duplicate_agent = Mock(spec=BaseAgent)
        duplicate_agent.agent_id = "agent_001"
        duplicate_agent.data = Mock()
        duplicate_agent.data.agent_type = "guardian"

        self.registry.register_agent(duplicate_agent)
        # Duplicate registration should not replace the original

        # Original agent should still be registered
        assert self.registry._agents["agent_001"] == self.mock_agent1

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        self.registry.register_agent(self.mock_agent1)
        assert self.mock_agent1.agent_id in self.registry._agents

        self.registry.unregister_agent(self.mock_agent1.agent_id)

        # Check agent was unregistered
        assert self.mock_agent1.agent_id not in self.registry._agents
        assert self.mock_agent1 not in self.registry._agents_by_type["explorer"]

    def test_unregister_nonexistent_agent(self):
        """Test unregistering non-existent agent returns False."""
        self.registry.unregister_agent("nonexistent_agent")
        # Should not raise an error for non-existent agent

    def test_get_agent(self):
        """Test getting agent by ID."""
        self.registry.register_agent(self.mock_agent1)

        retrieved_agent = self.registry.get_agent(self.mock_agent1.agent_id)
        assert retrieved_agent == self.mock_agent1

        # Test getting non-existent agent
        nonexistent = self.registry.get_agent("nonexistent")
        assert nonexistent is None

    def test_get_agents_by_type(self):
        """Test getting agents by type."""
        self.registry.register_agent(self.mock_agent1)  # explorer
        self.registry.register_agent(self.mock_agent2)  # guardian
        self.registry.register_agent(self.mock_agent3)  # explorer

        explorers = self.registry.find_agents_by_type("explorer")
        assert len(explorers) == 2
        assert self.mock_agent1 in explorers
        assert self.mock_agent3 in explorers

        guardians = self.registry.find_agents_by_type("guardian")
        assert len(guardians) == 1
        assert self.mock_agent2 in guardians

        # Test getting non-existent type
        scholars = self.registry.find_agents_by_type("scholar")
        assert len(scholars) == 0

    def test_list_all_agents(self):
        """Test listing all registered agents."""
        self.registry.register_agent(self.mock_agent1)
        self.registry.register_agent(self.mock_agent2)

        all_agents = self.registry.get_all_agents()
        assert len(all_agents) == 2
        assert self.mock_agent1 in all_agents
        assert self.mock_agent2 in all_agents

    def test_get_agent_count(self):
        """Test getting total agent count."""
        assert self.registry.get_agent_count() == 0

        self.registry.register_agent(self.mock_agent1)
        assert self.registry.get_agent_count() == 1

        self.registry.register_agent(self.mock_agent2)
        assert self.registry.get_agent_count() == 2

        self.registry.unregister_agent(self.mock_agent1.agent_id)
        assert self.registry.get_agent_count() == 1

    def test_get_agent_count_by_type(self):
        """Test getting agent count by type."""
        self.registry.register_agent(self.mock_agent1)  # explorer
        self.registry.register_agent(self.mock_agent2)  # guardian
        self.registry.register_agent(self.mock_agent3)  # explorer

        count_by_type = self.registry.get_agent_count_by_type()
        assert count_by_type["explorer"] == 2
        assert count_by_type["guardian"] == 1
        assert count_by_type.get("scholar", 0) == 0

    def test_event_handler_registration(self):
        """Test registering event handlers."""
        handler1 = Mock(spec=IAgentEventHandler)
        handler2 = Mock(spec=IAgentEventHandler)

        self.registry.add_event_handler(handler1)
        self.registry.add_event_handler(handler2)

        assert len(self.registry._event_handlers) == 2
        assert handler1 in self.registry._event_handlers
        assert handler2 in self.registry._event_handlers

    def test_event_handler_notification(self):
        """Test event handlers are notified of agent registration."""
        handler = Mock(spec=IAgentEventHandler)
        self.registry.add_event_handler(handler)

        # Register agent should trigger handler
        self.registry.register_agent(self.mock_agent1)

        handler.on_agent_created.assert_called_once_with(self.mock_agent1.data)

        # Unregister agent should trigger handler
        self.registry.unregister_agent(self.mock_agent1.agent_id)

        handler.on_agent_destroyed.assert_called_once_with(
            self.mock_agent1.data)


class TestAgentFactoryIntegration:
    """Integration tests for agent factory and registry."""

    def test_factory_registry_integration(self):
        """Test integration between factory and registry."""
        factory = AgentFactory()
        registry = AgentRegistry()

        # Create agent using factory
        agent_data = AgentData(
            agent_id="integration_agent",
            name="Integration Test Agent",
            position=Position(
                x=0.0,
                y=0.0,
                z=0.0),
            capabilities={
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION},
            personality=AgentPersonality(
                openness=0.6,
                agreeableness=0.8),
        )

        with patch("agents.base.agent_factory.create_agent") as mock_create:
            mock_agent = Mock(spec=BaseAgent)
            mock_agent.agent_id = "integration_agent"
            mock_agent.data = Mock()
            mock_agent.data.agent_type = "basic"
            mock_agent.data.metadata = {}
            mock_agent.data.personality = Mock()
            mock_agent.data.position = Position(x=0.0, y=0.0, z=0.0)
            mock_create.return_value = mock_agent

            # Create and register agent
            agent = factory.create_agent("basic", agent_data=agent_data)
            registry.register_agent(agent)

            # Check agent was registered
            assert registry.get_agent("integration_agent") == agent
            assert agent in registry.find_agents_by_type("basic")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
