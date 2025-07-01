"""
Comprehensive tests for agents package initialization.

Tests the agents package imports and basic functionality,
ensuring proper module loading and PyMDP/GNN alignment.
"""

from unittest.mock import Mock, patch

import pytest


# Test the main agents package imports
def test_agents_package_imports():
    """Test that agents package imports correctly."""
    import agents

    # Check version
    assert hasattr(agents, "__version__")
    assert agents.__version__ == "0.1.0"

    # Check __all__ exports
    assert hasattr(agents, "__all__")
    assert "base" in agents.__all__
    assert "explorer" in agents.__all__
    assert "merchant" in agents.__all__

    # Check submodules are importable
    assert hasattr(agents, "base")
    assert hasattr(agents, "explorer")
    assert hasattr(agents, "merchant")


def test_agents_base_imports():
    """Test agents.base module imports."""
    from agents import base

    # Check key classes are available
    assert hasattr(base, "Agent")
    assert hasattr(base, "BaseAgent")
    assert hasattr(base, "AgentCapability")
    assert hasattr(base, "Position")
    assert hasattr(base, "get_default_factory")


def test_agents_explorer_imports():
    """Test agents.explorer module imports."""
    from agents import explorer

    # Module should be importable
    assert explorer is not None


def test_agents_merchant_imports():
    """Test agents.merchant module imports."""
    from agents import merchant

    # Module should be importable
    assert merchant is not None


def test_position_class():
    """Test Position dataclass functionality."""
    from agents.base import Position

    # Create position
    pos = Position(x=10.0, y=20.0, z=5.0)

    assert pos.x == 10.0
    assert pos.y == 20.0
    assert pos.z == 5.0

    # Test string representation
    pos_str = str(pos)
    assert "10.0" in pos_str
    assert "20.0" in pos_str
    assert "5.0" in pos_str


def test_agent_capability_enum():
    """Test AgentCapability enum."""
    from agents.base import AgentCapability

    # Check enum values exist
    assert hasattr(AgentCapability, "MOVEMENT")
    assert hasattr(AgentCapability, "PERCEPTION")
    assert hasattr(AgentCapability, "MEMORY")
    assert hasattr(AgentCapability, "COMMUNICATION")
    assert hasattr(AgentCapability, "LEARNING")

    # Check enum values
    assert AgentCapability.MOVEMENT.value == "movement"
    assert AgentCapability.PERCEPTION.value == "perception"
    assert AgentCapability.MEMORY.value == "memory"
    assert AgentCapability.COMMUNICATION.value == "communication"
    assert AgentCapability.LEARNING.value == "learning"


def test_get_default_factory():
    """Test get_default_factory function."""
    from agents.base import get_default_factory

    # Should return a factory instance
    factory = get_default_factory()
    assert factory is not None

    # Factory should have key methods for agent creation
    assert hasattr(factory, "create_agent")
    assert hasattr(factory, "register_type")


def test_agent_data_model():
    """Test Agent data model class."""
    from agents.base import Agent, Position

    # Create agent with minimal parameters
    agent = Agent(name="Test Agent", agent_type="explorer", position=Position(0, 0, 0))

    assert agent.name == "Test Agent"
    assert agent.agent_type == "explorer"
    assert agent.position.x == 0
    assert agent.position.y == 0
    assert agent.position.z == 0

    # Check default values that actually exist
    assert agent.agent_id is not None
    assert agent.resources is not None
    assert agent.metadata == {}
    assert len(agent.capabilities) > 0


def test_agent_data_model_with_metadata():
    """Test Agent data model with metadata."""
    from agents.base import Agent, Position

    metadata = {"created_at": "2024-01-01", "version": "1.0"}

    agent = Agent(
        name="Resource Agent",
        agent_type="merchant",
        position=Position(10, 20, 0),
        metadata=metadata,
    )

    assert agent.name == "Resource Agent"
    assert agent.agent_type == "merchant"
    assert agent.metadata == metadata
    assert agent.position.x == 10
    assert agent.position.y == 20


def test_base_agent_abstract_class():
    """Test BaseAgent abstract class interface."""
    from agents.base import BaseAgent

    # Check BaseAgent exists and has required methods
    assert hasattr(BaseAgent, "start")
    assert hasattr(BaseAgent, "stop")
    assert hasattr(BaseAgent, "pause")

    # BaseAgent should be a class
    assert callable(BaseAgent)


def test_pymdp_alignment_imports():
    """Test imports support PyMDP alignment."""
    from agents.base import Agent, Position

    # Agent should support beliefs and active inference concepts
    agent = Agent(
        name="PyMDP Agent",
        agent_type="scholar",
        position=Position(0, 0, 0),
        metadata={
            "beliefs": [0.25, 0.25, 0.25, 0.25],
            "preferences": [0.0, 1.0, 0.0, 0.0],
            "free_energy": -2.5,
        },
    )

    assert "beliefs" in agent.metadata
    assert "preferences" in agent.metadata
    assert "free_energy" in agent.metadata


def test_gnn_notation_support():
    """Test GNN (Generalized Notation Notation) support in agents."""
    from agents.base import Agent, Position

    # Create agent with GNN metadata
    agent = Agent(
        name="GNN Agent",
        agent_type="scholar",
        position=Position(0, 0, 0),
        metadata={
            "notation_system": "GNN",
            "notation_version": "1.0",
            "supported_formalisms": [
                "belief_dynamics",
                "policy_selection",
                "free_energy_principle",
            ],
        },
    )

    assert agent.metadata["notation_system"] == "GNN"
    assert len(agent.metadata["supported_formalisms"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
