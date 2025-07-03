"""
Comprehensive tests for agents.base.persistence module.

Tests agent persistence functionality including CRUD operations,
serialization/deserialization, and snapshot management.
"""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from agents.base.data_model import (
    Agent,
    AgentCapability,
    AgentGoal,
    AgentPersonality,
    AgentResources,
    AgentStatus,
    Orientation,
    Position,
    ResourceAgent,
    SocialAgent,
    SocialRelationship,
)
from agents.base.persistence import AGENT_SCHEMA_VERSION, AgentPersistence, AgentSnapshot


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    session = Mock()
    session.query.return_value.filter_by.return_value.first.return_value = None
    session.query.return_value.filter_by.return_value.all.return_value = []
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.add = Mock()
    session.delete = Mock()
    return session


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    agent = Agent()
    agent.agent_id = str(uuid.uuid4())
    agent.name = "TestAgent"
    agent.agent_type = "explorer"
    agent.position = Position(x=10.0, y=20.0, z=0.0)
    agent.orientation = Orientation(w=1.0, x=0.0, y=0.0, z=0.0)
    agent.velocity = np.array([1.0, 0.0, 0.0])
    agent.status = AgentStatus.IDLE
    agent.resources = AgentResources(energy=80, health=100, memory_capacity=100, memory_used=10)
    agent.capabilities = {AgentCapability.MOVEMENT, AgentCapability.COMMUNICATION}
    agent.personality = AgentPersonality(
        openness=0.8, conscientiousness=0.6, extraversion=0.4, agreeableness=0.7, neuroticism=0.3
    )
    agent.experience_count = 100
    agent.short_term_memory = ["memory1", "memory2"]
    agent.long_term_memory = ["long_memory1"]
    agent.generative_model_params = {"param1": 0.5}
    agent.belief_state = np.array([0.1, 0.2, 0.3])
    agent.metadata = {"test_key": "test_value"}
    agent.created_at = datetime.now()

    # Add a goal
    goal = AgentGoal(
        goal_id=str(uuid.uuid4()),
        description="Test goal",
        target_position=Position(x=50.0, y=60.0, z=0.0),
        priority=1,
    )
    agent.current_goal = goal
    agent.goals = [goal]

    # Add a relationship
    relationship = SocialRelationship(
        target_agent_id=str(uuid.uuid4()),
        relationship_type="ally",
        trust_level=0.8,
        interaction_count=5,
        last_interaction=datetime.now(),
    )
    agent.relationships[relationship.target_agent_id] = relationship

    return agent


@pytest.fixture
def mock_db_agent():
    """Create a mock database agent."""
    db_agent = Mock()
    db_agent.uuid = str(uuid.uuid4())
    db_agent.name = "TestAgent"
    db_agent.type = "explorer"
    db_agent.created_at = datetime.now()
    db_agent.updated_at = datetime.now()
    db_agent.state = {
        "position": {"x": 10.0, "y": 20.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "velocity": [1.0, 0.0, 0.0],
        "status": "idle",
        "resources": {"energy": 80, "health": 100, "memory_capacity": 100, "memory_used": 10},
        "experience_count": 100,
        "short_term_memory": ["memory1", "memory2"],
        "schema_version": AGENT_SCHEMA_VERSION,
    }
    db_agent.config = {
        "capabilities": ["movement", "communication"],
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.4,
            "agreeableness": 0.7,
            "neuroticism": 0.3,
        },
        "metadata": {"test_key": "test_value"},
    }
    db_agent.beliefs = {
        "relationships": {},
        "goals": [],
        "long_term_memory": ["long_memory1"],
        "generative_model_params": {"param1": 0.5},
        "belief_state": [0.1, 0.2, 0.3],
    }
    return db_agent


class TestAgentPersistenceInitialization:
    """Test AgentPersistence initialization."""

    def test_initialization_with_session(self, mock_session):
        """Test initialization with provided session."""
        persistence = AgentPersistence(session=mock_session)

        assert persistence.session == mock_session
        assert persistence._use_external_session is True

    def test_initialization_without_session(self):
        """Test initialization without session."""
        persistence = AgentPersistence()

        assert persistence.session is None
        assert persistence._use_external_session is False

    @patch("agents.base.persistence.get_db_session")
    def test_get_session_external(self, mock_get_db, mock_session):
        """Test getting session when using external session."""
        persistence = AgentPersistence(session=mock_session)

        result = persistence._get_session()

        assert result == mock_session
        mock_get_db.assert_not_called()

    @patch("agents.base.persistence.get_db_session")
    def test_get_session_internal(self, mock_get_db):
        """Test getting session when creating internal sessions."""
        mock_get_db.return_value = Mock()
        persistence = AgentPersistence()

        result = persistence._get_session()

        mock_get_db.assert_called_once()
        assert result == mock_get_db.return_value


class TestAgentSaving:
    """Test agent saving functionality."""

    def test_save_agent_new_success(self, mock_session, sample_agent):
        """Test successfully saving a new agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        persistence = AgentPersistence(session=mock_session)

        result = persistence.save_agent(sample_agent)

        assert result is True
        mock_session.add.assert_called_once()
        # External session shouldn't auto-commit
        mock_session.commit.assert_not_called()

    def test_save_agent_update_existing(self, mock_session, sample_agent, mock_db_agent):
        """Test updating an existing agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        persistence = AgentPersistence(session=mock_session)

        result = persistence.save_agent(sample_agent, update_if_exists=True)

        assert result is True
        assert mock_db_agent.name == sample_agent.name
        assert mock_db_agent.type == sample_agent.agent_type
        mock_session.add.assert_not_called()  # Should update, not add
        # External session shouldn't auto-commit
        mock_session.commit.assert_not_called()

    def test_save_agent_exists_no_update(self, mock_session, sample_agent, mock_db_agent):
        """Test saving agent when exists and update_if_exists=False."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        persistence = AgentPersistence(session=mock_session)

        result = persistence.save_agent(sample_agent, update_if_exists=False)

        assert result is False
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    def test_save_agent_database_error(self, mock_session, sample_agent):
        """Test handling database error during save."""
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        persistence = AgentPersistence(session=mock_session)

        result = persistence.save_agent(sample_agent)

        assert result is False
        # External session shouldn't auto-rollback
        mock_session.rollback.assert_not_called()

    def test_save_agent_general_error(self, mock_session, sample_agent):
        """Test handling general error during save."""
        mock_session.query.side_effect = Exception("General error")
        persistence = AgentPersistence(session=mock_session)

        result = persistence.save_agent(sample_agent)

        assert result is False
        # External session shouldn't auto-rollback
        mock_session.rollback.assert_not_called()

    def test_save_agent_external_session_no_commit(self, mock_session, sample_agent):
        """Test that external sessions don't auto-commit."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        persistence = AgentPersistence(session=mock_session)

        persistence.save_agent(sample_agent)

        # External session shouldn't auto-commit
        mock_session.commit.assert_not_called()

    def test_save_agent_internal_session_commits(self, sample_agent):
        """Test that internal sessions auto-commit."""
        with patch("agents.base.persistence.get_db_session") as mock_get_db:
            mock_session = Mock()
            mock_session.query.return_value.filter_by.return_value.first.return_value = None
            mock_get_db.return_value = mock_session

            persistence = AgentPersistence()
            persistence.save_agent(sample_agent)

            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()


class TestAgentLoading:
    """Test agent loading functionality."""

    def test_load_agent_success(self, mock_session, mock_db_agent):
        """Test successfully loading an agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        persistence = AgentPersistence(session=mock_session)

        agent = persistence.load_agent(mock_db_agent.uuid)

        assert agent is not None
        assert agent.agent_id == mock_db_agent.uuid
        assert agent.name == mock_db_agent.name
        assert agent.agent_type == mock_db_agent.type

    def test_load_agent_not_found(self, mock_session):
        """Test loading non-existent agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        persistence = AgentPersistence(session=mock_session)

        agent = persistence.load_agent("nonexistent-id")

        assert agent is None

    def test_load_agent_error(self, mock_session):
        """Test handling error during agent loading."""
        mock_session.query.side_effect = Exception("Database error")
        persistence = AgentPersistence(session=mock_session)

        agent = persistence.load_agent("test-id")

        assert agent is None

    def test_load_agent_closes_session(self):
        """Test that internal sessions are properly closed."""
        with patch("agents.base.persistence.get_db_session") as mock_get_db:
            mock_session = Mock()
            mock_session.query.return_value.filter_by.return_value.first.return_value = None
            mock_get_db.return_value = mock_session

            persistence = AgentPersistence()
            persistence.load_agent("test-id")

            mock_session.close.assert_called_once()


class TestAgentBatchLoading:
    """Test batch agent loading functionality."""

    def test_load_all_agents_no_filters(self, mock_session):
        """Test loading all agents without filters."""
        mock_db_agents = [Mock(), Mock()]
        mock_session.query.return_value.all.return_value = mock_db_agents
        persistence = AgentPersistence(session=mock_session)

        with patch.object(persistence, "_deserialize_agent") as mock_deserialize:
            mock_deserialize.side_effect = [Mock(), Mock()]

            agents = persistence.load_all_agents()

            assert len(agents) == 2
            assert mock_deserialize.call_count == 2

    def test_load_all_agents_with_type_filter(self, mock_session):
        """Test loading agents with type filter."""
        persistence = AgentPersistence(session=mock_session)

        persistence.load_all_agents(agent_type="explorer")

        mock_session.query.return_value.filter_by.assert_called_with(type="explorer")

    def test_load_all_agents_with_status_filter(self, mock_session):
        """Test loading agents with status filter."""
        persistence = AgentPersistence(session=mock_session)

        persistence.load_all_agents(status="idle")

        # Should call filter_by twice - once for type (None), once for status
        assert mock_session.query.return_value.filter_by.call_count >= 1

    def test_load_all_agents_deserialization_error(self, mock_session):
        """Test handling deserialization errors."""
        mock_db_agent = Mock()
        mock_db_agent.uuid = "test-id"
        mock_session.query.return_value.all.return_value = [mock_db_agent]
        persistence = AgentPersistence(session=mock_session)

        with patch.object(persistence, "_deserialize_agent") as mock_deserialize:
            mock_deserialize.side_effect = Exception("Deserialization error")

            agents = persistence.load_all_agents()

            assert len(agents) == 0  # Should skip failed agents

    def test_load_all_agents_database_error(self, mock_session):
        """Test handling database error during batch loading."""
        mock_session.query.side_effect = Exception("Database error")
        persistence = AgentPersistence(session=mock_session)

        agents = persistence.load_all_agents()

        assert agents == []


class TestAgentDeletion:
    """Test agent deletion functionality."""

    def test_delete_agent_success(self, mock_session, mock_db_agent):
        """Test successfully deleting an agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        persistence = AgentPersistence(session=mock_session)

        result = persistence.delete_agent(mock_db_agent.uuid)

        assert result is True
        mock_session.delete.assert_called_once_with(mock_db_agent)
        # External session shouldn't auto-commit
        mock_session.commit.assert_not_called()

    def test_delete_agent_not_found(self, mock_session):
        """Test deleting non-existent agent."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        persistence = AgentPersistence(session=mock_session)

        result = persistence.delete_agent("nonexistent-id")

        assert result is False
        mock_session.delete.assert_not_called()

    def test_delete_agent_database_error(self, mock_session, mock_db_agent):
        """Test handling database error during deletion."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        mock_session.delete.side_effect = SQLAlchemyError("Database error")
        persistence = AgentPersistence(session=mock_session)

        result = persistence.delete_agent(mock_db_agent.uuid)

        assert result is False
        # External session shouldn't auto-rollback
        mock_session.rollback.assert_not_called()


class TestAgentSerialization:
    """Test agent serialization functionality."""

    def test_serialize_agent_basic_fields(self, sample_agent):
        """Test serialization of basic agent fields."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        # Check structure
        assert "state" in result
        assert "config" in result
        assert "beliefs" in result
        assert "location" in result

        # Check state fields
        state = result["state"]
        assert state["position"]["x"] == sample_agent.position.x
        assert state["orientation"]["w"] == sample_agent.orientation.w
        assert state["status"] == sample_agent.status.value
        assert state["schema_version"] == AGENT_SCHEMA_VERSION

    def test_serialize_agent_resources(self, sample_agent):
        """Test serialization of agent resources."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        resources = result["state"]["resources"]
        assert resources["energy"] == sample_agent.resources.energy
        assert resources["health"] == sample_agent.resources.health
        assert resources["memory_capacity"] == sample_agent.resources.memory_capacity

    def test_serialize_agent_capabilities(self, sample_agent):
        """Test serialization of agent capabilities."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        capabilities = result["config"]["capabilities"]
        expected_caps = [cap.value for cap in sample_agent.capabilities]
        assert set(capabilities) == set(expected_caps)

    def test_serialize_agent_personality(self, sample_agent):
        """Test serialization of agent personality."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        personality = result["config"]["personality"]
        assert personality["openness"] == sample_agent.personality.openness
        assert personality["conscientiousness"] == sample_agent.personality.conscientiousness

    def test_serialize_agent_relationships(self, sample_agent):
        """Test serialization of agent relationships."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        relationships = result["beliefs"]["relationships"]
        assert len(relationships) == len(sample_agent.relationships)

        for agent_id, rel_data in relationships.items():
            original_rel = sample_agent.relationships[agent_id]
            assert rel_data["target_agent_id"] == original_rel.target_agent_id
            assert rel_data["trust_level"] == original_rel.trust_level

    def test_serialize_agent_goals(self, sample_agent):
        """Test serialization of agent goals."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        goals = result["beliefs"]["goals"]
        assert len(goals) == len(sample_agent.goals)

        # Check current goal
        current_goal = result["state"]["current_goal"]
        assert current_goal["description"] == sample_agent.current_goal.description

    def test_serialize_agent_numpy_arrays(self, sample_agent):
        """Test serialization of numpy arrays."""
        persistence = AgentPersistence()

        result = persistence._serialize_agent(sample_agent)

        # Velocity should be converted to list
        velocity = result["state"]["velocity"]
        assert isinstance(velocity, list)
        assert velocity == sample_agent.velocity.tolist()

        # Belief state should be converted to list
        belief_state = result["beliefs"]["belief_state"]
        assert isinstance(belief_state, list)
        assert belief_state == sample_agent.belief_state.tolist()

    def test_serialize_agent_memory_truncation(self, sample_agent):
        """Test that memory lists are properly truncated."""
        # Add many memory items
        sample_agent.short_term_memory = [f"memory_{i}" for i in range(100)]
        sample_agent.long_term_memory = [f"long_memory_{i}" for i in range(200)]

        persistence = AgentPersistence()
        result = persistence._serialize_agent(sample_agent)

        # Short term memory should be truncated to last 50
        assert len(result["state"]["short_term_memory"]) == 50
        assert result["state"]["short_term_memory"][0] == "memory_50"

        # Long term memory should be truncated to last 100
        assert len(result["beliefs"]["long_term_memory"]) == 100
        assert result["beliefs"]["long_term_memory"][0] == "long_memory_100"


class TestAgentDeserialization:
    """Test agent deserialization functionality."""

    def test_deserialize_agent_basic_fields(self, mock_db_agent):
        """Test deserialization of basic agent fields."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert agent.agent_id == mock_db_agent.uuid
        assert agent.name == mock_db_agent.name
        assert agent.agent_type == mock_db_agent.type
        assert agent.created_at == mock_db_agent.created_at

    def test_deserialize_agent_position_and_orientation(self, mock_db_agent):
        """Test deserialization of position and orientation."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent.position, Position)
        assert agent.position.x == 10.0
        assert agent.position.y == 20.0

        assert isinstance(agent.orientation, Orientation)
        assert agent.orientation.w == 1.0

    def test_deserialize_agent_velocity_and_arrays(self, mock_db_agent):
        """Test deserialization of velocity and numpy arrays."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent.velocity, np.ndarray)
        np.testing.assert_array_equal(agent.velocity, np.array([1.0, 0.0, 0.0]))

    def test_deserialize_agent_status_and_resources(self, mock_db_agent):
        """Test deserialization of status and resources."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert agent.status == AgentStatus.IDLE
        assert isinstance(agent.resources, AgentResources)
        assert agent.resources.energy == 80

    def test_deserialize_agent_capabilities(self, mock_db_agent):
        """Test deserialization of capabilities."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent.capabilities, set)
        expected_caps = {AgentCapability.MOVEMENT, AgentCapability.COMMUNICATION}
        assert agent.capabilities == expected_caps

    def test_deserialize_agent_personality(self, mock_db_agent):
        """Test deserialization of personality."""
        persistence = AgentPersistence()

        agent = persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent.personality, AgentPersonality)
        assert agent.personality.openness == 0.8
        assert agent.personality.conscientiousness == 0.6

    def test_deserialize_agent_different_types(self, mock_db_agent):
        """Test deserialization creates correct agent subclasses."""
        persistence = AgentPersistence()

        # Test ResourceAgent
        mock_db_agent.type = "resource_management"
        agent = persistence._deserialize_agent(mock_db_agent)
        assert isinstance(agent, ResourceAgent)

        # Test SocialAgent
        mock_db_agent.type = "social_interaction"
        agent = persistence._deserialize_agent(mock_db_agent)
        assert isinstance(agent, SocialAgent)

        # Test default Agent
        mock_db_agent.type = "unknown_type"
        agent = persistence._deserialize_agent(mock_db_agent)
        assert isinstance(agent, Agent)

    def test_deserialize_agent_relationships(self, mock_db_agent):
        """Test deserialization of relationships."""
        # Add relationship data
        rel_id = str(uuid.uuid4())
        mock_db_agent.beliefs["relationships"] = {
            rel_id: {
                "target_agent_id": rel_id,
                "relationship_type": "ally",
                "trust_level": 0.8,
                "interaction_count": 5,
                "last_interaction": "2024-01-01T12:00:00",
            }
        }

        persistence = AgentPersistence()
        agent = persistence._deserialize_agent(mock_db_agent)

        assert len(agent.relationships) == 1
        rel = agent.relationships[rel_id]
        assert isinstance(rel, SocialRelationship)
        assert rel.trust_level == 0.8

    def test_deserialize_agent_goals(self, mock_db_agent):
        """Test deserialization of goals."""
        goal_data = {
            "goal_id": str(uuid.uuid4()),
            "description": "Test goal",
            "target_position": {"x": 50.0, "y": 60.0, "z": 0.0},
            "priority": 1,
        }
        mock_db_agent.beliefs["goals"] = [goal_data]

        persistence = AgentPersistence()
        agent = persistence._deserialize_agent(mock_db_agent)

        assert len(agent.goals) == 1
        goal = agent.goals[0]
        assert isinstance(goal, AgentGoal)
        assert goal.description == "Test goal"
        assert isinstance(goal.target_position, Position)

    def test_deserialize_agent_missing_fields(self, mock_db_agent):
        """Test deserialization with missing optional fields."""
        # Remove optional fields
        mock_db_agent.state = {}
        mock_db_agent.config = {}
        mock_db_agent.beliefs = {}

        persistence = AgentPersistence()
        agent = persistence._deserialize_agent(mock_db_agent)

        # Should create agent with defaults
        assert agent.agent_id == mock_db_agent.uuid
        assert agent.name == mock_db_agent.name


class TestGoalSerialization:
    """Test goal serialization helper methods."""

    def test_serialize_goal(self):
        """Test goal serialization."""
        goal = AgentGoal(
            goal_id=str(uuid.uuid4()),
            description="Test goal",
            target_position=Position(x=10.0, y=20.0, z=0.0),
            priority=1,
        )

        persistence = AgentPersistence()
        result = persistence._serialize_goal(goal)

        assert result["goal_id"] == goal.goal_id
        assert result["description"] == goal.description
        assert result["priority"] == goal.priority

    def test_deserialize_goal(self):
        """Test goal deserialization."""
        goal_id = str(uuid.uuid4())
        goal_data = {
            "goal_id": goal_id,
            "description": "Test goal",
            "target_position": {"x": 10.0, "y": 20.0, "z": 0.0},
            "priority": 1,
        }

        persistence = AgentPersistence()
        goal = persistence._deserialize_goal(goal_data)

        assert isinstance(goal, AgentGoal)
        assert goal.goal_id == goal_id
        assert goal.description == "Test goal"
        assert isinstance(goal.target_position, Position)

    def test_deserialize_goal_without_target_position(self):
        """Test goal deserialization without target position."""
        goal_data = {
            "description": "Test goal",
            "priority": 1,
            "target_position": None,
        }

        persistence = AgentPersistence()
        goal = persistence._deserialize_goal(goal_data)

        assert isinstance(goal, AgentGoal)
        assert goal.target_position is None


class TestAgentSnapshot:
    """Test AgentSnapshot functionality."""

    def test_snapshot_initialization(self, mock_session):
        """Test snapshot handler initialization."""
        persistence = AgentPersistence(session=mock_session)
        snapshot = AgentSnapshot(persistence)

        assert snapshot.persistence == persistence

    @patch("agents.base.persistence.uuid.uuid4")
    @patch("agents.base.persistence.datetime")
    def test_create_snapshot(self, mock_datetime, mock_uuid, mock_session, sample_agent):
        """Test creating agent snapshot."""
        # Setup mocks
        snapshot_id = "test-snapshot-id"
        mock_uuid.return_value = snapshot_id
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(sample_agent, "to_dict") as mock_to_dict:
            mock_to_dict.return_value = {"agent": "data"}
            with patch.object(persistence, "save_agent") as mock_save:
                mock_save.return_value = True

                result = snapshot_handler.create_snapshot(sample_agent, "Test snapshot")

                assert result == snapshot_id
                assert "snapshots" in sample_agent.metadata
                assert len(sample_agent.metadata["snapshots"]) == 1

                snapshot_data = sample_agent.metadata["snapshots"][0]
                assert snapshot_data["snapshot_id"] == snapshot_id
                assert snapshot_data["description"] == "Test snapshot"
                assert snapshot_data["agent_data"] == {"agent": "data"}

    def test_create_snapshot_limits_snapshots(self, mock_session, sample_agent):
        """Test that snapshots are limited to 10."""
        # Add 10 existing snapshots
        sample_agent.metadata["snapshots"] = [{"snapshot_id": f"snapshot_{i}"} for i in range(10)]

        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(sample_agent, "to_dict") as mock_to_dict:
            mock_to_dict.return_value = {"agent": "data"}
            with patch.object(persistence, "save_agent") as mock_save:
                mock_save.return_value = True

                snapshot_handler.create_snapshot(sample_agent, "New snapshot")

                # Should still have 10 snapshots (oldest removed)
                assert len(sample_agent.metadata["snapshots"]) == 10
                assert sample_agent.metadata["snapshots"][0]["snapshot_id"] == "snapshot_1"

    def test_restore_snapshot_success(self, mock_session, sample_agent):
        """Test successful snapshot restoration."""
        # Add snapshot to agent metadata
        snapshot_id = "test-snapshot-id"
        sample_agent.metadata["snapshots"] = [
            {
                "snapshot_id": snapshot_id,
                "agent_data": {"restored": "data"},
                "timestamp": "2024-01-01T12:00:00",
                "description": "Test snapshot",
            }
        ]

        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(persistence, "load_agent") as mock_load:
            mock_load.return_value = sample_agent
            with patch.object(Agent, "from_dict") as mock_from_dict:
                restored_agent = Mock()
                mock_from_dict.return_value = restored_agent

                result = snapshot_handler.restore_snapshot(sample_agent.agent_id, snapshot_id)

                assert result == restored_agent
                mock_from_dict.assert_called_once_with({"restored": "data"})

    def test_restore_snapshot_agent_not_found(self, mock_session):
        """Test snapshot restoration when agent not found."""
        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(persistence, "load_agent") as mock_load:
            mock_load.return_value = None

            result = snapshot_handler.restore_snapshot("nonexistent-id", "snapshot-id")

            assert result is None

    def test_restore_snapshot_not_found(self, mock_session, sample_agent):
        """Test restoration of non-existent snapshot."""
        sample_agent.metadata["snapshots"] = []

        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(persistence, "load_agent") as mock_load:
            mock_load.return_value = sample_agent

            result = snapshot_handler.restore_snapshot(
                sample_agent.agent_id, "nonexistent-snapshot"
            )

            assert result is None

    def test_list_snapshots_success(self, mock_session, sample_agent):
        """Test listing agent snapshots."""
        snapshots = [
            {
                "snapshot_id": "snap1",
                "timestamp": "2024-01-01T12:00:00",
                "description": "First snapshot",
                "agent_data": {},
            },
            {
                "snapshot_id": "snap2",
                "timestamp": "2024-01-01T13:00:00",
                "description": "Second snapshot",
                "agent_data": {},
            },
        ]
        sample_agent.metadata["snapshots"] = snapshots

        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(persistence, "load_agent") as mock_load:
            mock_load.return_value = sample_agent

            result = snapshot_handler.list_snapshots(sample_agent.agent_id)

            assert len(result) == 2
            assert result[0]["snapshot_id"] == "snap1"
            assert result[0]["description"] == "First snapshot"
            assert "agent_data" not in result[0]  # Should be filtered out

    def test_list_snapshots_agent_not_found(self, mock_session):
        """Test listing snapshots for non-existent agent."""
        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(persistence, "load_agent") as mock_load:
            mock_load.return_value = None

            result = snapshot_handler.list_snapshots("nonexistent-id")

            assert result == []


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_complete_agent_lifecycle(self, mock_session, sample_agent):
        """Test complete agent lifecycle: save, load, update, delete."""
        persistence = AgentPersistence(session=mock_session)

        # 1. Save new agent
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        result = persistence.save_agent(sample_agent)
        assert result is True

        # 2. Load agent
        mock_db_agent = Mock()
        mock_db_agent.uuid = sample_agent.agent_id
        mock_db_agent.state = {"status": "idle"}
        mock_db_agent.config = {}
        mock_db_agent.beliefs = {}
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent

        loaded_agent = persistence.load_agent(sample_agent.agent_id)
        assert loaded_agent is not None

        # 3. Update agent
        mock_session.reset_mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        result = persistence.save_agent(sample_agent, update_if_exists=True)
        assert result is True

        # 4. Delete agent
        result = persistence.delete_agent(sample_agent.agent_id)
        assert result is True

    def test_snapshot_workflow(self, mock_session, sample_agent):
        """Test complete snapshot workflow."""
        persistence = AgentPersistence(session=mock_session)
        snapshot_handler = AgentSnapshot(persistence)

        with patch.object(sample_agent, "to_dict") as mock_to_dict:
            mock_to_dict.return_value = {"original": "data"}
            with patch.object(persistence, "save_agent") as mock_save:
                mock_save.return_value = True

                # 1. Create snapshot
                snapshot_id = snapshot_handler.create_snapshot(sample_agent, "Before changes")
                assert snapshot_id is not None

                # 2. List snapshots
                with patch.object(persistence, "load_agent") as mock_load:
                    mock_load.return_value = sample_agent
                    snapshots = snapshot_handler.list_snapshots(sample_agent.agent_id)
                    assert len(snapshots) == 1

                    # 3. Restore snapshot
                    with patch.object(Agent, "from_dict") as mock_from_dict:
                        restored_agent = Mock()
                        mock_from_dict.return_value = restored_agent

                        result = snapshot_handler.restore_snapshot(
                            sample_agent.agent_id, snapshot_id
                        )
                        assert result == restored_agent


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_serialization_with_none_values(self, sample_agent):
        """Test serialization handles None values gracefully."""
        sample_agent.current_goal = None
        sample_agent.belief_state = None

        persistence = AgentPersistence()
        result = persistence._serialize_agent(sample_agent)

        assert result["state"]["current_goal"] is None
        assert "belief_state" not in result["beliefs"]

    def test_deserialization_with_missing_relationships(self, mock_db_agent):
        """Test deserialization when relationships have missing last_interaction."""
        rel_data = {
            "target_agent_id": "test-id",
            "relationship_type": "ally",
            "trust_level": 0.8,
            "interaction_count": 5,
            "last_interaction": None,
        }
        mock_db_agent.beliefs["relationships"] = {"test-id": rel_data}

        persistence = AgentPersistence()
        agent = persistence._deserialize_agent(mock_db_agent)

        rel = agent.relationships["test-id"]
        assert rel.last_interaction is None

    def test_session_management_consistency(self, mock_session, sample_agent):
        """Test that session management is consistent across operations."""
        persistence = AgentPersistence(session=mock_session)

        # All operations should use the same external session
        persistence.save_agent(sample_agent)
        persistence.load_agent(sample_agent.agent_id)
        persistence.delete_agent(sample_agent.agent_id)

        # External session should never be closed
        mock_session.close.assert_not_called()

        # Commits should not happen automatically
        assert mock_session.commit.call_count == 0
