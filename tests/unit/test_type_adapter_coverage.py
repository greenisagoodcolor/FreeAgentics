"""Comprehensive test suite for type_adapter module to boost coverage."""

import uuid
from datetime import datetime
from enum import Enum
from unittest.mock import Mock

import pytest

from agents.type_adapter import AgentTypeAdapter, CoalitionTypeAdapter


class StatusEnum(Enum):
    """Mock status enum for testing."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class TestAgentTypeAdapter:
    """Test AgentTypeAdapter functionality."""

    def test_get_id_from_database_model(self):
        """Test getting ID from database model."""
        # Test with UUID
        agent = Mock()
        agent.id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert AgentTypeAdapter.get_id(agent) == "12345678-1234-5678-1234-567812345678"

        # Test with string ID
        agent.id = "test-id-123"
        assert AgentTypeAdapter.get_id(agent) == "test-id-123"

        # Test with integer ID
        agent.id = 12345
        assert AgentTypeAdapter.get_id(agent) == "12345"

    def test_get_id_from_in_memory_agent(self):
        """Test getting ID from in-memory agent."""
        agent = Mock()
        agent.agent_id = "memory-agent-456"
        del agent.id  # Remove id attribute
        assert AgentTypeAdapter.get_id(agent) == "memory-agent-456"

    def test_get_id_from_dict(self):
        """Test getting ID from dictionary."""
        # Test with 'id' key
        agent = {"id": "dict-id-789"}
        assert AgentTypeAdapter.get_id(agent) == "dict-id-789"

        # Test with 'agent_id' key
        agent = {"agent_id": "dict-agent-id-012"}
        assert AgentTypeAdapter.get_id(agent) == "dict-agent-id-012"

    def test_get_id_raises_attribute_error(self):
        """Test get_id raises AttributeError for invalid objects."""
        agent = Mock()
        del agent.id
        del agent.agent_id

        with pytest.raises(AttributeError) as exc:
            AgentTypeAdapter.get_id(agent)
        assert "has no 'id' or 'agent_id' attribute" in str(exc.value)

    def test_get_name(self):
        """Test getting agent name."""
        # From object
        agent = Mock()
        agent.name = "Test Agent"
        assert AgentTypeAdapter.get_name(agent) == "Test Agent"

        # From dict
        agent = {"name": "Dict Agent"}
        assert AgentTypeAdapter.get_name(agent) == "Dict Agent"

        # Missing name
        agent = Mock()
        del agent.name
        with pytest.raises(AttributeError) as exc:
            AgentTypeAdapter.get_name(agent)
        assert "has no 'name' attribute" in str(exc.value)

    def test_get_status(self):
        """Test getting agent status."""
        # From enum
        agent = Mock()
        agent.status = StatusEnum.ACTIVE
        assert AgentTypeAdapter.get_status(agent) == "active"

        # From string
        agent.status = "running"
        assert AgentTypeAdapter.get_status(agent) == "running"

        # From is_active attribute
        agent = Mock()
        agent.is_active = True
        del agent.status
        assert AgentTypeAdapter.get_status(agent) == "active"

        agent.is_active = False
        assert AgentTypeAdapter.get_status(agent) == "inactive"

        # From dict with enum
        agent = {"status": StatusEnum.INACTIVE}
        assert AgentTypeAdapter.get_status(agent) == "inactive"

        # From dict with is_active
        agent = {"is_active": True}
        assert AgentTypeAdapter.get_status(agent) == "active"

        # Unknown status
        agent = Mock()
        del agent.status
        del agent.is_active
        assert AgentTypeAdapter.get_status(agent) == "unknown"

    def test_get_position(self):
        """Test getting agent position."""
        # From object
        agent = Mock()
        agent.position = (10, 20)
        assert AgentTypeAdapter.get_position(agent) == (10, 20)

        # From dict
        agent = {"position": [5, 15]}
        assert AgentTypeAdapter.get_position(agent) == [5, 15]

        # No position
        agent = Mock()
        del agent.position
        assert AgentTypeAdapter.get_position(agent) is None

    def test_to_dict_comprehensive(self):
        """Test converting agent to dictionary."""
        # Full database model
        agent = Mock()
        agent.id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        agent.name = "Test Agent"
        agent.status = StatusEnum.ACTIVE
        agent.position = (10, 20)
        agent.template = "basic_agent"
        agent.created_at = datetime(2024, 1, 1, 12, 0, 0)
        agent.metrics = {"score": 100}

        result = AgentTypeAdapter.to_dict(agent)
        assert result["id"] == "12345678-1234-5678-1234-567812345678"
        assert result["name"] == "Test Agent"
        assert result["status"] == "active"
        assert result["position"] == "(10, 20)"
        assert result["template"] == "basic_agent"
        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["metrics"] == {"score": 100}

        # In-memory agent
        agent = Mock()
        agent.agent_id = "mem-123"
        agent.name = "Memory Agent"
        agent.is_active = True
        agent.total_steps = 50
        agent.beliefs = [1, 2, 3]
        del agent.id
        del agent.status
        del agent.position

        result = AgentTypeAdapter.to_dict(agent)
        assert result["id"] == "mem-123"
        assert result["name"] == "Memory Agent"
        assert result["status"] == "active"
        assert result["total_steps"] == 50
        assert result["has_beliefs"] == "True"

        # Minimal agent
        agent = Mock()
        del agent.id
        del agent.agent_id
        del agent.name
        del agent.status
        del agent.is_active
        del agent.position

        result = AgentTypeAdapter.to_dict(agent)
        assert result["id"] == "unknown"
        assert result["name"] == "Unknown"
        assert result["status"] == "unknown"
        assert "position" not in result


class TestCoalitionTypeAdapter:
    """Test CoalitionTypeAdapter functionality."""

    def test_get_id_from_database_model(self):
        """Test getting ID from database model."""
        # Test with UUID
        coalition = Mock()
        coalition.id = uuid.UUID("87654321-4321-8765-4321-876543218765")
        assert (
            CoalitionTypeAdapter.get_id(coalition)
            == "87654321-4321-8765-4321-876543218765"
        )

        # Test with string ID
        coalition.id = "coal-id-123"
        assert CoalitionTypeAdapter.get_id(coalition) == "coal-id-123"

    def test_get_id_from_in_memory_coalition(self):
        """Test getting ID from in-memory coalition."""
        coalition = Mock()
        coalition.coalition_id = "memory-coal-456"
        del coalition.id
        assert CoalitionTypeAdapter.get_id(coalition) == "memory-coal-456"

    def test_get_id_from_dict(self):
        """Test getting ID from dictionary."""
        # Test with 'id' key
        coalition = {"id": "dict-coal-789"}
        assert CoalitionTypeAdapter.get_id(coalition) == "dict-coal-789"

        # Test with 'coalition_id' key
        coalition = {"coalition_id": "dict-coalition-id-012"}
        assert CoalitionTypeAdapter.get_id(coalition) == "dict-coalition-id-012"

    def test_get_id_raises_attribute_error(self):
        """Test get_id raises AttributeError for invalid objects."""
        coalition = Mock()
        del coalition.id
        del coalition.coalition_id

        with pytest.raises(AttributeError) as exc:
            CoalitionTypeAdapter.get_id(coalition)
        assert "has no 'id' or 'coalition_id' attribute" in str(exc.value)

    def test_get_name(self):
        """Test getting coalition name."""
        # From object
        coalition = Mock()
        coalition.name = "Test Coalition"
        assert CoalitionTypeAdapter.get_name(coalition) == "Test Coalition"

        # From dict
        coalition = {"name": "Dict Coalition"}
        assert CoalitionTypeAdapter.get_name(coalition) == "Dict Coalition"

        # Missing name
        coalition = Mock()
        del coalition.name
        with pytest.raises(AttributeError) as exc:
            CoalitionTypeAdapter.get_name(coalition)
        assert "has no 'name' attribute" in str(exc.value)

    def test_get_members(self):
        """Test getting coalition members."""
        # In-memory coalition with members dict
        coalition = Mock()
        coalition.members = {
            "agent1": {"agent_id": "agent1", "name": "Agent One"},
            "agent2": {"agent_id": "agent2", "name": "Agent Two"},
        }
        assert CoalitionTypeAdapter.get_members(coalition) == coalition.members

        # Database model with agents relationship
        agent1 = Mock()
        agent1.id = "db-agent1"
        agent1.name = "DB Agent One"

        agent2 = Mock()
        agent2.agent_id = "db-agent2"
        agent2.name = "DB Agent Two"
        del agent2.id

        coalition = Mock()
        coalition.agents = [agent1, agent2]
        del coalition.members

        members = CoalitionTypeAdapter.get_members(coalition)
        assert "db-agent1" in members
        assert members["db-agent1"]["agent_id"] == "db-agent1"
        assert members["db-agent1"]["name"] == "DB Agent One"
        assert "db-agent2" in members
        assert members["db-agent2"]["agent_id"] == "db-agent2"
        assert members["db-agent2"]["name"] == "DB Agent Two"

        # Dict with members
        coalition = {"members": {"agent3": {"name": "Agent Three"}}}
        assert CoalitionTypeAdapter.get_members(coalition) == {
            "agent3": {"name": "Agent Three"}
        }

        # Dict with agents
        coalition = {"agents": ["agent4", "agent5"]}
        assert CoalitionTypeAdapter.get_members(coalition) == [
            "agent4",
            "agent5",
        ]

        # No members
        coalition = Mock()
        del coalition.members
        del coalition.agents
        assert CoalitionTypeAdapter.get_members(coalition) == {}

    def test_get_leader_id(self):
        """Test getting coalition leader ID."""
        # From object
        coalition = Mock()
        coalition.leader_id = "leader-123"
        assert CoalitionTypeAdapter.get_leader_id(coalition) == "leader-123"

        # None leader
        coalition.leader_id = None
        assert CoalitionTypeAdapter.get_leader_id(coalition) is None

        # From dict
        coalition = {"leader_id": "dict-leader-456"}
        assert CoalitionTypeAdapter.get_leader_id(coalition) == "dict-leader-456"

        # No leader attribute
        coalition = Mock()
        del coalition.leader_id
        assert CoalitionTypeAdapter.get_leader_id(coalition) is None

    def test_get_status(self):
        """Test getting coalition status."""
        # From enum
        coalition = Mock()
        coalition.status = StatusEnum.ACTIVE
        assert CoalitionTypeAdapter.get_status(coalition) == "active"

        # From string
        coalition.status = "forming"
        assert CoalitionTypeAdapter.get_status(coalition) == "forming"

        # From dict with enum
        coalition = {"status": StatusEnum.INACTIVE}
        assert CoalitionTypeAdapter.get_status(coalition) == "inactive"

        # Unknown status
        coalition = Mock()
        del coalition.status
        assert CoalitionTypeAdapter.get_status(coalition) == "unknown"

    def test_to_dict_comprehensive(self):
        """Test converting coalition to dictionary."""
        # Full database model
        coalition = Mock()
        coalition.id = uuid.UUID("87654321-4321-8765-4321-876543218765")
        coalition.name = "Test Coalition"
        coalition.status = StatusEnum.ACTIVE
        coalition.members = {
            "agent1": {"agent_id": "agent1", "name": "Agent One"},
            "agent2": {"agent_id": "agent2", "name": "Agent Two"},
        }
        coalition.leader_id = "agent1"
        coalition.objective = "Test objective"
        coalition.created_at = datetime(2024, 1, 1, 14, 0, 0)
        coalition.performance_score = 85.5

        result = CoalitionTypeAdapter.to_dict(coalition)
        assert result["id"] == "87654321-4321-8765-4321-876543218765"
        assert result["name"] == "Test Coalition"
        assert result["status"] == "active"
        assert result["member_count"] == "2"
        assert result["member_ids"] == "['agent1', 'agent2']"
        assert result["leader_id"] == "agent1"
        assert result["objective"] == "Test objective"
        assert result["created_at"] == "2024-01-01T14:00:00"
        assert result["performance_score"] == 85.5

        # Coalition with agents list instead of members dict
        agent1 = Mock()
        agent1.id = "list-agent1"

        coalition = Mock()
        coalition.coalition_id = "list-coal"
        coalition.name = "List Coalition"
        coalition.status = "active"
        coalition.agents = [agent1]
        del coalition.id
        del coalition.members
        del coalition.leader_id

        result = CoalitionTypeAdapter.to_dict(coalition)
        assert result["id"] == "list-coal"
        assert result["member_count"] == "1"
        assert result["member_ids"] == "['list-agent1']"

        # Minimal coalition
        coalition = Mock()
        del coalition.id
        del coalition.coalition_id
        del coalition.name
        del coalition.status
        del coalition.members
        del coalition.agents
        del coalition.leader_id

        result = CoalitionTypeAdapter.to_dict(coalition)
        assert result["id"] == "unknown"
        assert result["name"] == "Unknown"
        assert result["status"] == "unknown"
        assert result["member_count"] == "0"
        assert result["member_ids"] == "[]"
