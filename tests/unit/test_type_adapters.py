"""Test type adapters for agent and coalition objects."""

import uuid
from datetime import datetime

from agents.base_agent import BasicExplorerAgent
from agents.type_adapter import AgentTypeAdapter, CoalitionTypeAdapter
from agents.type_helpers import (
    ensure_string_id,
    get_agent_attribute,
    get_coalition_attribute,
    match_agent_id,
    safe_get_agent_id,
)
from coalitions.coalition import Coalition, CoalitionStatus
from database.models import Agent as AgentModel
from database.models import AgentStatus
from database.models import Coalition as CoalitionModel
from database.models import CoalitionStatus as DBCoalitionStatus


class TestAgentTypeAdapter:
    """Test AgentTypeAdapter functionality."""

    def test_get_id_from_database_model(self):
        """Test getting ID from database agent model."""
        db_id = uuid.uuid4()
        agent = AgentModel(
            id=db_id,
            name="Test Agent",
            template="explorer",
            status=AgentStatus.ACTIVE,
        )

        result = AgentTypeAdapter.get_id(agent)
        assert result == str(db_id)
        assert isinstance(result, str)

    def test_get_id_from_in_memory_agent(self):
        """Test getting ID from in-memory agent."""
        agent = BasicExplorerAgent("test_agent_123", "Explorer")

        result = AgentTypeAdapter.get_id(agent)
        assert result == "test_agent_123"
        assert isinstance(result, str)

    def test_get_id_from_dict(self):
        """Test getting ID from dictionary representation."""
        # Test with 'id' key
        agent_dict = {"id": "agent_456", "name": "Test"}
        result = AgentTypeAdapter.get_id(agent_dict)
        assert result == "agent_456"

        # Test with 'agent_id' key
        agent_dict = {"agent_id": "agent_789", "name": "Test"}
        result = AgentTypeAdapter.get_id(agent_dict)
        assert result == "agent_789"

    def test_get_name(self):
        """Test getting agent name from various sources."""
        # Database model
        agent_db = AgentModel(
            id=uuid.uuid4(),
            name="DB Agent",
            template="explorer",
            status=AgentStatus.ACTIVE,
        )
        assert AgentTypeAdapter.get_name(agent_db) == "DB Agent"

        # In-memory agent
        agent_mem = BasicExplorerAgent("test_id", "Memory Agent")
        assert AgentTypeAdapter.get_name(agent_mem) == "Memory Agent"

        # Dict
        agent_dict = {"name": "Dict Agent"}
        assert AgentTypeAdapter.get_name(agent_dict) == "Dict Agent"

    def test_get_status(self):
        """Test getting agent status from various sources."""
        # Database model with enum
        agent_db = AgentModel(
            id=uuid.uuid4(),
            name="Test",
            template="explorer",
            status=AgentStatus.ACTIVE,
        )
        assert AgentTypeAdapter.get_status(agent_db) == "active"

        # In-memory agent with is_active
        agent_mem = BasicExplorerAgent("test_id", "Test")
        agent_mem.is_active = True
        assert AgentTypeAdapter.get_status(agent_mem) == "active"

        agent_mem.is_active = False
        assert AgentTypeAdapter.get_status(agent_mem) == "inactive"

    def test_to_dict(self):
        """Test converting agent to dictionary."""
        # Database model
        db_id = uuid.uuid4()
        agent_db = AgentModel(
            id=db_id,
            name="Test Agent",
            template="explorer",
            status=AgentStatus.ACTIVE,
            created_at=datetime.now(),
        )

        result = AgentTypeAdapter.to_dict(agent_db)
        assert result["id"] == str(db_id)
        assert result["name"] == "Test Agent"
        assert result["status"] == "active"
        assert result["template"] == "explorer"
        assert "created_at" in result


class TestCoalitionTypeAdapter:
    """Test CoalitionTypeAdapter functionality."""

    def test_get_id_from_database_model(self):
        """Test getting ID from database coalition model."""
        db_id = uuid.uuid4()
        coalition = CoalitionModel(
            id=db_id,
            name="Test Coalition",
            objectives={"main": "Test objective"},
            status=DBCoalitionStatus.ACTIVE,
        )

        result = CoalitionTypeAdapter.get_id(coalition)
        assert result == str(db_id)
        assert isinstance(result, str)

    def test_get_id_from_in_memory_coalition(self):
        """Test getting ID from in-memory coalition."""
        coalition = Coalition(
            coalition_id="coalition_123",
            name="Test Coalition"
        )

        result = CoalitionTypeAdapter.get_id(coalition)
        assert result == "coalition_123"
        assert isinstance(result, str)

    def test_get_members(self):
        """Test getting coalition members."""
        # In-memory coalition
        coalition = Coalition("test_coalition", "Test")
        coalition.add_member("agent_1", capabilities=["explore"])
        coalition.add_member("agent_2", capabilities=["collect"])

        members = CoalitionTypeAdapter.get_members(coalition)
        assert isinstance(members, dict)
        assert "agent_1" in members
        assert "agent_2" in members

    def test_get_status(self):
        """Test getting coalition status."""
        # In-memory coalition
        coalition = Coalition("test_coalition", "Test")
        coalition.status = CoalitionStatus.ACTIVE
        assert CoalitionTypeAdapter.get_status(coalition) == "active"

        # Database model
        coalition_db = CoalitionModel(
            id=uuid.uuid4(),
            name="Test",
            objectives={"main": "Test objective"},
            status=DBCoalitionStatus.FORMING,
        )
        assert CoalitionTypeAdapter.get_status(coalition_db) == "forming"


class TestTypeHelpers:
    """Test type helper functions."""

    def test_safe_get_agent_id(self):
        """Test safe agent ID retrieval."""
        # Valid agent
        agent = BasicExplorerAgent("test_123", "Test")
        assert safe_get_agent_id(agent) == "test_123"

        # Invalid object
        assert safe_get_agent_id({}) is None
        assert safe_get_agent_id(None) is None

    def test_ensure_string_id(self):
        """Test ID conversion to string."""
        # UUID
        test_uuid = uuid.uuid4()
        assert ensure_string_id(test_uuid) == str(test_uuid)

        # Already a string
        assert ensure_string_id("test_123") == "test_123"

        # Other types
        assert ensure_string_id(123) == "123"

    def test_match_agent_id(self):
        """Test agent ID matching."""
        agent = BasicExplorerAgent("test_123", "Test")

        # String match
        assert match_agent_id(agent, "test_123") is True
        assert match_agent_id(agent, "wrong_id") is False

        # UUID match (should convert to string)
        test_uuid = uuid.uuid4()
        agent_db = AgentModel(
            id=test_uuid,
            name="Test",
            template="explorer",
            status=AgentStatus.ACTIVE,
        )
        assert match_agent_id(agent_db, test_uuid) is True
        assert match_agent_id(agent_db, str(test_uuid)) is True

    def test_get_agent_attribute(self):
        """Test getting agent attributes safely."""
        agent = BasicExplorerAgent("test_123", "Test Agent")

        # Standard attributes
        assert get_agent_attribute(agent, "name") == "Test Agent"
        assert get_agent_attribute(agent, "agent_id") == "test_123"
        assert get_agent_attribute(agent, "id") == "test_123"

        # Missing attribute with default
        assert get_agent_attribute(agent, "missing", "default") == "default"

        # Position
        agent.position = (1, 2)
        assert get_agent_attribute(agent, "position") == (1, 2)

    def test_get_coalition_attribute(self):
        """Test getting coalition attributes safely."""
        coalition = Coalition(
            "test_coalition", "Test Coalition"
        )
        coalition.add_member("agent_1")

        # Standard attributes
        assert get_coalition_attribute(coalition, "name") == "Test Coalition"
        assert (
            get_coalition_attribute(coalition, "coalition_id")
            == "test_coalition"
        )
        assert get_coalition_attribute(coalition, "id") == "test_coalition"

        # Members
        members = get_coalition_attribute(coalition, "members")
        assert isinstance(members, dict)
        assert "agent_1" in members

        # Missing attribute
        assert get_coalition_attribute(coalition, "missing", None) is None
