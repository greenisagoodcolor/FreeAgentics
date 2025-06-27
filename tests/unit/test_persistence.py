"""
Module for FreeAgentics Active Inference implementation.
"""

from datetime import datetime, timedelta
from typing import Optional
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
    SocialRelationship,
)
from agents.base.persistence import AGENT_SCHEMA_VERSION, AgentPersistence, AgentSnapshot


class TestAgentPersistence:
    """Test AgentPersistence class"""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session"""
        session = Mock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit.return_value = None
        session.rollback.return_value = None
        session.close.return_value = None
        return session

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing"""
        agent = Agent(agent_id="test-agent-123", name="Test Agent", agent_type="basic")
        agent.position = Position(10.0, 20.0, 5.0)
        agent.orientation = Orientation(1.0, 0.0, 0.0, 0.0)
        agent.velocity = np.array([1.0, 2.0, 0.0])
        agent.status = AgentStatus.MOVING
        agent.personality = AgentPersonality(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.6,
            agreeableness=0.9,
            neuroticism=0.3,
        )
        agent.resources = AgentResources(
            energy=75.0, health=90.0, memory_capacity=100.0, memory_used=25.0
        )
        relationship = SocialRelationship(
            target_agent_id="other-agent-456",
            relationship_type="friend",
            trust_level=0.8,
            interaction_count=5,
            last_interaction=datetime.now(),
        )
        agent.add_relationship(relationship)
        goal = AgentGoal(
            description="Find resources",
            priority=0.8,
            target_position=Position(50.0, 50.0, 0.0),
            deadline=datetime.now() + timedelta(hours=1),
        )
        agent.add_goal(goal)
        agent.add_to_memory({"event": "found_item", "location": [30, 40]}, is_important=True)
        return agent

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_new(self, mock_get_session, mock_session, sample_agent) -> None:
        """Test saving a new agent"""
        mock_get_session.return_value = mock_session
        with patch("infrastructure.database.models.Agent") as MockDBAgent:
            persistence = AgentPersistence()
            result = persistence.save_agent(sample_agent)
            assert result is True
            assert mock_session.add.called
            assert mock_session.commit.called
            call_kwargs = MockDBAgent.call_args.kwargs
            assert call_kwargs["uuid"] == sample_agent.agent_id
            assert call_kwargs["name"] == sample_agent.name
            assert call_kwargs["type"] == sample_agent.agent_type
            assert call_kwargs["energy_level"] == 0.75
            assert call_kwargs["experience_points"] == 1

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_update_existing(self, mock_get_session, mock_session, sample_agent) -> None:
        """Test updating an existing agent"""
        mock_db_agent = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        result = persistence.save_agent(sample_agent, update_if_exists=True)
        assert result is True
        assert mock_db_agent.name == sample_agent.name
        assert mock_db_agent.type == sample_agent.agent_type
        assert mock_session.commit.called

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_exists_no_update(
        self, mock_get_session, mock_session, sample_agent
    ) -> None:
        """Test saving when agent exists but update_if_exists=False"""
        mock_db_agent = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        result = persistence.save_agent(sample_agent, update_if_exists=False)
        assert result is False
        assert not mock_session.commit.called

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_database_error(self, mock_get_session, mock_session, sample_agent) -> None:
        """Test handling database errors during save"""
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        result = persistence.save_agent(sample_agent)
        assert result is False
        assert mock_session.rollback.called

    @patch("agents.base.persistence.get_db_session")
    def test_load_agent_success(self, mock_get_session, mock_session) -> None:
        """Test loading an agent successfully"""
        mock_db_agent = Mock()
        mock_db_agent.uuid = "test-agent-123"
        mock_db_agent.name = "Test Agent"
        mock_db_agent.type = "basic"
        mock_db_agent.created_at = datetime.now()
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = {
            "position": {"x": 10.0, "y": 20.0, "z": 5.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": [1.0, 2.0, 0.0],
            "status": "moving",
            "resources": {
                "energy": 75.0,
                "health": 90.0,
                "memory_capacity": 100.0,
                "memory_used": 25.0,
            },
            "current_goal": None,
            "short_term_memory": [],
            "experience_count": 0,
            "schema_version": AGENT_SCHEMA_VERSION,
        }
        mock_db_agent.config = {
            "capabilities": ["movement", "perception", "communication"],
            "personality": {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
            "metadata": {},
        }
        mock_db_agent.beliefs = {
            "relationships": {},
            "goals": [],
            "long_term_memory": [],
            "generative_model_params": {},
        }
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        agent = persistence.load_agent("test-agent-123")
        assert agent is not None
        assert agent.agent_id == "test-agent-123"
        assert agent.name == "Test Agent"
        assert agent.position.x == 10.0
        assert agent.position.y == 20.0
        assert agent.status == AgentStatus.MOVING
        assert AgentCapability.MOVEMENT in agent.capabilities

    @patch("agents.base.persistence.get_db_session")
    def test_load_agent_not_found(self, mock_get_session, mock_session) -> None:
        """Test loading a non-existent agent"""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        agent = persistence.load_agent("non-existent-agent")
        assert agent is None

    @patch("agents.base.persistence.get_db_session")
    def test_load_all_agents(self, mock_get_session, mock_session) -> None:
        """Test loading all agents"""
        mock_db_agent1 = Mock()
        mock_db_agent1.uuid = "agent-1"
        mock_db_agent1.name = "Agent 1"
        mock_db_agent1.type = "basic"
        mock_db_agent1.created_at = datetime.now()
        mock_db_agent1.updated_at = None
        mock_db_agent1.state = {}
        mock_db_agent1.config = {}
        mock_db_agent1.beliefs = {}
        mock_db_agent2 = Mock()
        mock_db_agent2.uuid = "agent-2"
        mock_db_agent2.name = "Agent 2"
        mock_db_agent2.type = "resource_management"
        mock_db_agent2.created_at = datetime.now()
        mock_db_agent2.updated_at = None
        mock_db_agent2.state = {}
        mock_db_agent2.config = {}
        mock_db_agent2.beliefs = {}
        mock_session.query.return_value.all.return_value = [
            mock_db_agent1,
            mock_db_agent2,
        ]
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        agents = persistence.load_all_agents()
        assert len(agents) == 2
        assert agents[0].agent_id == "agent-1"
        assert agents[1].agent_id == "agent-2"
        assert isinstance(agents[1], ResourceAgent)

    @patch("agents.base.persistence.get_db_session")
    def test_delete_agent_success(self, mock_get_session, mock_session) -> None:
        """Test deleting an agent successfully"""
        mock_db_agent = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_agent
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        result = persistence.delete_agent("test-agent-123")
        assert result is True
        mock_session.delete.assert_called_with(mock_db_agent)
        assert mock_session.commit.called

    @patch("agents.base.persistence.get_db_session")
    def test_delete_agent_not_found(self, mock_get_session, mock_session) -> None:
        """Test deleting a non-existent agent"""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_get_session.return_value = mock_session
        persistence = AgentPersistence()
        result = persistence.delete_agent("non-existent-agent")
        assert result is False
        assert not mock_session.delete.called

    def test_serialize_deserialize_agent(self, sample_agent) -> None:
        """Test agent serialization and deserialization"""
        persistence = AgentPersistence()
        serialized = persistence._serialize_agent(sample_agent)
        assert "state" in serialized
        assert "config" in serialized
        assert "beliefs" in serialized
        assert serialized["state"]["position"]["x"] == 10.0
        assert serialized["state"]["status"] == "moving"
        assert set(serialized["config"]["capabilities"]) == {
            "movement",
            "perception",
            "communication",
            "memory",
            "learning",
        }
        assert len(serialized["beliefs"]["goals"]) == 1
        assert len(serialized["beliefs"]["relationships"]) == 1
        mock_db_agent = Mock()
        mock_db_agent.uuid = sample_agent.agent_id
        mock_db_agent.name = sample_agent.name
        mock_db_agent.type = sample_agent.agent_type
        mock_db_agent.created_at = sample_agent.created_at
        mock_db_agent.updated_at = sample_agent.last_updated
        mock_db_agent.state = serialized["state"]
        mock_db_agent.config = serialized["config"]
        mock_db_agent.beliefs = serialized["beliefs"]
        deserialized_agent = persistence._deserialize_agent(mock_db_agent)
        assert deserialized_agent.agent_id == sample_agent.agent_id
        assert deserialized_agent.name == sample_agent.name
        assert deserialized_agent.position.x == sample_agent.position.x
        assert deserialized_agent.status == sample_agent.status
        assert len(deserialized_agent.goals) == 1
        assert len(deserialized_agent.relationships) == 1

    def test_serialize_goal(self, sample_agent) -> None:
        """Test goal serialization"""
        persistence = AgentPersistence()
        goal = sample_agent.goals[0]
        serialized = persistence._serialize_goal(goal)
        assert serialized["description"] == goal.description
        assert serialized["priority"] == goal.priority
        assert serialized["target_position"]["x"] == 50.0
        assert serialized["deadline"] is not None
        deserialized = persistence._deserialize_goal(serialized)
        assert deserialized.description == goal.description
        assert deserialized.priority == goal.priority
        assert deserialized.target_position.x == 50.0

    def test_numpy_array_serialization(self, sample_agent) -> None:
        """Test serialization of numpy arrays"""
        persistence = AgentPersistence()
        sample_agent.belief_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        serialized = persistence._serialize_agent(sample_agent)
        assert "belief_state" in serialized["beliefs"]
        assert serialized["beliefs"]["belief_state"] == [0.1, 0.2, 0.3, 0.4, 0.5]


class TestAgentSnapshot:
    """Test AgentSnapshot class"""

    @pytest.fixture
    def mock_persistence(self):
        """Create a mock AgentPersistence"""
        persistence = Mock(spec=AgentPersistence)
        persistence.save_agent.return_value = True
        persistence.load_agent.return_value = None
        return persistence

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing"""
        agent = Agent(agent_id="test-agent-123", name="Test Agent", agent_type="basic")
        agent.position = Position(10.0, 20.0, 5.0)
        agent.orientation = Orientation(1.0, 0.0, 0.0, 0.0)
        agent.velocity = np.array([1.0, 2.0, 0.0])
        agent.status = AgentStatus.MOVING
        agent.personality = AgentPersonality(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.6,
            agreeableness=0.9,
            neuroticism=0.3,
        )
        agent.resources = AgentResources(
            energy=75.0, health=90.0, memory_capacity=100.0, memory_used=25.0
        )
        relationship = SocialRelationship(
            target_agent_id="other-agent-456",
            relationship_type="friend",
            trust_level=0.8,
            interaction_count=5,
            last_interaction=datetime.now(),
        )
        agent.add_relationship(relationship)
        goal = AgentGoal(
            description="Find resources",
            priority=0.8,
            target_position=Position(50.0, 50.0, 0.0),
            deadline=datetime.now() + timedelta(hours=1),
        )
        agent.add_goal(goal)
        agent.add_to_memory({"event": "found_item", "location": [30, 40]}, is_important=True)
        return agent

    def test_create_snapshot(self, mock_persistence, sample_agent) -> None:
        """Test creating a snapshot"""
        snapshot = AgentSnapshot(mock_persistence)
        snapshot_id = snapshot.create_snapshot(sample_agent, "Test snapshot")
        assert snapshot_id is not None
        assert "snapshots" in sample_agent.metadata
        assert len(sample_agent.metadata["snapshots"]) == 1
        assert sample_agent.metadata["snapshots"][0]["description"] == "Test snapshot"
        assert mock_persistence.save_agent.called

    def test_create_multiple_snapshots(self, mock_persistence, sample_agent) -> None:
        """Test that only last 10 snapshots are kept"""
        snapshot = AgentSnapshot(mock_persistence)
        for i in range(15):
            snapshot.create_snapshot(sample_agent, f"Snapshot {i}")
        assert len(sample_agent.metadata["snapshots"]) == 10
        assert sample_agent.metadata["snapshots"][0]["description"] == "Snapshot 5"
        assert sample_agent.metadata["snapshots"][-1]["description"] == "Snapshot 14"

    def test_restore_snapshot(self, mock_persistence, sample_agent) -> None:
        """Test restoring from a snapshot"""
        snapshot = AgentSnapshot(mock_persistence)
        snapshot_id = snapshot.create_snapshot(sample_agent, "Before changes")
        original_name = sample_agent.name
        sample_agent.name = "Modified Agent"
        sample_agent.position = Position(99.0, 99.0, 99.0)
        mock_persistence.load_agent.return_value = sample_agent
        restored_agent = snapshot.restore_snapshot(sample_agent.agent_id, snapshot_id)
        assert restored_agent is not None
        assert restored_agent.name == original_name
        assert restored_agent.position.x == 10.0

    def test_restore_nonexistent_snapshot(self, mock_persistence, sample_agent) -> None:
        """Test restoring a non-existent snapshot"""
        snapshot = AgentSnapshot(mock_persistence)
        mock_persistence.load_agent.return_value = sample_agent
        restored_agent = snapshot.restore_snapshot(sample_agent.agent_id, "fake-snapshot-id")
        assert restored_agent is None

    def test_list_snapshots(self, mock_persistence, sample_agent) -> None:
        """Test listing snapshots"""
        snapshot = AgentSnapshot(mock_persistence)
        ids = []
        for i in range(3):
            snapshot_id = snapshot.create_snapshot(sample_agent, f"Snapshot {i}")
            ids.append(snapshot_id)
        mock_persistence.load_agent.return_value = sample_agent
        snapshots_list = snapshot.list_snapshots(sample_agent.agent_id)
        assert len(snapshots_list) == 3
        for i, snap in enumerate(snapshots_list):
            assert snap["snapshot_id"] == ids[i]
            assert snap["description"] == f"Snapshot {i}"
            assert "timestamp" in snap
