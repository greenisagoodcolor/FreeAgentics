"""
Comprehensive tests for Agent Persistence Module.

Tests the functionality for saving and loading agent states to/from
the database, including serialization, deserialization, and version management.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, call, patch

import numpy as np
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
from agents.base.persistence import AGENT_SCHEMA_VERSION, AgentPersistence


class TestAgentPersistence:
    """Test AgentPersistence class."""

    def setup_method(self):
        """Set up test persistence handler."""
        self.mock_session = Mock()
        self.persistence = AgentPersistence(session=self.mock_session)

        # Create test agent
        self.test_agent = self._create_test_agent()

    def _create_test_agent(self) -> Agent:
        """Create a test agent with all fields populated."""
        agent = Agent()
        agent.agent_id = str(uuid.uuid4())
        agent.name = "TestAgent"
        agent.agent_type = "explorer"
        agent.position = Position(10.0, 20.0, 5.0)
        agent.orientation = Orientation(w=1.0, x=0.0, y=0.0, z=0.0)
        agent.velocity = np.array([1.0, 2.0, 0.0])
        agent.status = AgentStatus.IDLE
        agent.resources = AgentResources(
            energy=80.0, health=90.0, memory_capacity=100.0, memory_used=10.0
        )
        agent.capabilities = {
            AgentCapability.MOVEMENT,
            AgentCapability.PERCEPTION}
        agent.personality = AgentPersonality(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.5,
            neuroticism=0.4,
        )
        agent.current_goal = AgentGoal(
            goal_id="goal_1",
            description="Explore unknown area",
            priority=0.9,
            target_position=Position(100.0, 100.0, 0.0),
            deadline=datetime.now() + timedelta(hours=1),
        )
        agent.goals = [agent.current_goal]
        agent.short_term_memory = ["event1", "event2", "event3"]
        agent.long_term_memory = ["memory1", "memory2"]
        agent.experience_count = 42
        agent.relationships = {
            "agent_2": SocialRelationship(
                target_agent_id="agent_2",
                relationship_type="ally",
                trust_level=0.8,
                interaction_count=5,
                last_interaction=datetime.now(),
            )
        }
        agent.belief_state = np.array([0.1, 0.2, 0.3, 0.4])
        agent.generative_model_params = {"param1": 1.0, "param2": 2.0}
        agent.metadata = {"role": "scout", "team": "alpha"}
        return agent

    def test_persistence_initialization_with_session(self):
        """Test initialization with external session."""
        assert self.persistence.session == self.mock_session
        assert self.persistence._use_external_session is True

    def test_persistence_initialization_without_session(self):
        """Test initialization without external session."""
        persistence = AgentPersistence()
        assert persistence.session is None
        assert persistence._use_external_session is False

    @patch("agents.base.persistence.get_db_session")
    def test_get_session_internal(self, mock_get_db):
        """Test getting session when using internal session management."""
        mock_db_session = Mock()
        mock_get_db.return_value = mock_db_session

        persistence = AgentPersistence()
        session = persistence._get_session()

        assert session == mock_db_session
        mock_get_db.assert_called_once()

    def test_get_session_external(self):
        """Test getting session when using external session."""
        session = self.persistence._get_session()
        assert session == self.mock_session

    def test_save_agent_success_new(self):
        """Test successfully saving a new agent."""
        # Mock database query returning no existing agent
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = None

        result = self.persistence.save_agent(self.test_agent)

        assert result is True

        # Verify agent was added to session
        self.mock_session.add.assert_called_once()
        added_agent = self.mock_session.add.call_args[0][0]
        assert added_agent.uuid == self.test_agent.agent_id
        assert added_agent.name == self.test_agent.name
        assert added_agent.type == self.test_agent.agent_type
        assert added_agent.energy_level == 0.8  # 80/100
        assert added_agent.experience_points == 42

    def test_save_agent_success_update(self):
        """Test successfully updating an existing agent."""
        # Mock existing database agent
        mock_db_agent = Mock()
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_db_agent)

        result = self.persistence.save_agent(
            self.test_agent, update_if_exists=True)

        assert result is True

        # Verify agent was updated
        assert mock_db_agent.name == self.test_agent.name
        assert mock_db_agent.type == self.test_agent.agent_type
        assert mock_db_agent.energy_level == 0.8
        assert mock_db_agent.experience_points == 42

        # Verify add was not called
        self.mock_session.add.assert_not_called()

    def test_save_agent_exists_no_update(self):
        """Test saving existing agent when update_if_exists is False."""
        # Mock existing database agent
        mock_db_agent = Mock()
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_db_agent)

        result = self.persistence.save_agent(
            self.test_agent, update_if_exists=False)

        assert result is False

        # Verify nothing was added or committed
        self.mock_session.add.assert_not_called()
        self.mock_session.commit.assert_not_called()

    def test_save_agent_database_error(self):
        """Test handling database error during save."""
        self.mock_session.query.side_effect = SQLAlchemyError("Database error")

        result = self.persistence.save_agent(self.test_agent)

        assert result is False

    def test_save_agent_general_error(self):
        """Test handling general error during save."""
        self.mock_session.query.side_effect = Exception("General error")

        result = self.persistence.save_agent(self.test_agent)

        assert result is False

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_with_internal_session(self, mock_get_db):
        """Test saving agent with internal session management."""
        mock_internal_session = Mock()
        mock_get_db.return_value = mock_internal_session
        mock_internal_session.query.return_value.filter_by.return_value.first.return_value = None

        persistence = AgentPersistence()
        result = persistence.save_agent(self.test_agent)

        assert result is True
        mock_internal_session.commit.assert_called_once()
        mock_internal_session.close.assert_called_once()

    @patch("agents.base.persistence.get_db_session")
    def test_save_agent_rollback_on_error(self, mock_get_db):
        """Test rollback on error with internal session."""
        mock_internal_session = Mock()
        mock_get_db.return_value = mock_internal_session
        mock_internal_session.query.side_effect = SQLAlchemyError(
            "Database error")

        persistence = AgentPersistence()
        result = persistence.save_agent(self.test_agent)

        assert result is False
        mock_internal_session.rollback.assert_called_once()
        mock_internal_session.close.assert_called_once()

    def test_load_agent_success(self):
        """Test successfully loading an agent."""
        # Create mock database agent
        mock_db_agent = Mock()
        mock_db_agent.uuid = self.test_agent.agent_id
        mock_db_agent.name = self.test_agent.name
        mock_db_agent.type = self.test_agent.agent_type
        mock_db_agent.created_at = datetime.now()
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = {
            "position": {
                "x": 10.0,
                "y": 20.0,
                "z": 5.0},
            "orientation": {
                "w": 1.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0},
            "velocity": [
                1.0,
                2.0,
                0.0],
            "status": "idle",
            "resources": {
                "energy": 80.0,
                "health": 90.0,
                "memory_capacity": 100,
                "memory_used": 0},
            "current_goal": {
                "goal_id": "goal_1",
                "description": "Explore unknown area",
                "priority": 0.9,
                "target_position": {
                    "x": 100.0,
                    "y": 100.0,
                    "z": 0.0},
                "deadline": datetime.now().isoformat(),
            },
            "short_term_memory": [
                "event1",
                "event2"],
            "experience_count": 42,
        }
        mock_db_agent.config = {
            "capabilities": ["movement", "perception"],
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.5,
                "neuroticism": 0.4,
            },
            "metadata": {"role": "scout"},
        }
        mock_db_agent.beliefs = {
            "relationships": {},
            "goals": [],
            "long_term_memory": ["memory1"],
            "generative_model_params": {"param1": 1.0},
        }

        self.mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_db_agent)

        agent = self.persistence.load_agent(self.test_agent.agent_id)

        assert agent is not None
        assert agent.agent_id == self.test_agent.agent_id
        assert agent.name == self.test_agent.name
        assert agent.position.x == 10.0
        assert agent.position.y == 20.0
        assert agent.resources.energy == 80.0
        assert agent.experience_count == 42
        assert len(agent.short_term_memory) == 2

    def test_load_agent_not_found(self):
        """Test loading non-existent agent."""
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = None

        agent = self.persistence.load_agent("non_existent_id")

        assert agent is None

    def test_load_agent_error(self):
        """Test handling error during agent load."""
        self.mock_session.query.side_effect = Exception("Load error")

        agent = self.persistence.load_agent(self.test_agent.agent_id)

        assert agent is None

    def test_load_all_agents_success(self):
        """Test loading all agents."""
        # Create mock database agents
        mock_agents = []
        for i in range(3):
            mock_agent = Mock()
            mock_agent.uuid = f"agent_{i}"
            mock_agent.name = f"Agent{i}"
            mock_agent.type = "explorer"
            mock_agent.created_at = datetime.now()
            mock_agent.updated_at = datetime.now()
            mock_agent.state = {"position": {"x": 0, "y": 0, "z": 0}}
            mock_agent.config = {
                "capabilities": [],
                "personality": {},
                "metadata": {}}
            mock_agent.beliefs = {
                "relationships": {},
                "goals": [],
                "long_term_memory": []}
            mock_agents.append(mock_agent)

        self.mock_session.query.return_value.all.return_value = mock_agents

        agents = self.persistence.load_all_agents()

        assert len(agents) == 3
        for i, agent in enumerate(agents):
            assert agent.agent_id == f"agent_{i}"
            assert agent.name == f"Agent{i}"

    def test_load_all_agents_with_filters(self):
        """Test loading agents with type and status filters."""
        mock_query = Mock()
        self.mock_session.query.return_value = mock_query
        mock_query.filter_by.return_value = mock_query
        mock_query.all.return_value = []

        _ = self.persistence.load_all_agents(
            agent_type="explorer", status="idle")

        # Verify filter_by was called twice
        assert mock_query.filter_by.call_count == 2
        calls = mock_query.filter_by.call_args_list
        assert calls[0] == call(type="explorer")
        assert calls[1] == call(status="idle")

    def test_load_all_agents_deserialization_error(self):
        """Test handling deserialization error for some agents."""
        # Create one valid and one invalid agent
        valid_agent = Mock()
        valid_agent.uuid = "valid_agent"
        valid_agent.name = "ValidAgent"
        valid_agent.type = "explorer"
        valid_agent.created_at = datetime.now()
        valid_agent.updated_at = datetime.now()
        valid_agent.state = {"position": {"x": 0, "y": 0, "z": 0}}
        valid_agent.config = {
            "capabilities": [],
            "personality": {},
            "metadata": {}}
        valid_agent.beliefs = {
            "relationships": {},
            "goals": [],
            "long_term_memory": []}

        invalid_agent = Mock()
        invalid_agent.uuid = "invalid_agent"
        invalid_agent.state = None  # This will cause deserialization error

        self.mock_session.query.return_value.all.return_value = [
            valid_agent, invalid_agent]

        agents = self.persistence.load_all_agents()

        # Should only return the valid agent
        assert len(agents) == 1
        assert agents[0].agent_id == "valid_agent"

    def test_delete_agent_success(self):
        """Test successfully deleting an agent."""
        mock_db_agent = Mock()
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_db_agent)

        result = self.persistence.delete_agent(self.test_agent.agent_id)

        assert result is True
        self.mock_session.delete.assert_called_once_with(mock_db_agent)

    def test_delete_agent_not_found(self):
        """Test deleting non-existent agent."""
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = None

        result = self.persistence.delete_agent("non_existent_id")

        assert result is False
        self.mock_session.delete.assert_not_called()

    def test_delete_agent_database_error(self):
        """Test handling database error during delete."""
        mock_db_agent = Mock()
        self.mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_db_agent)
        self.mock_session.delete.side_effect = SQLAlchemyError("Delete error")

        result = self.persistence.delete_agent(self.test_agent.agent_id)

        assert result is False

    def test_serialize_agent_complete(self):
        """Test complete agent serialization."""
        serialized = self.persistence._serialize_agent(self.test_agent)

        # Check structure
        assert "state" in serialized
        assert "config" in serialized
        assert "beliefs" in serialized
        assert "location" in serialized

        # Check state
        state = serialized["state"]
        assert state["position"]["x"] == 10.0
        assert state["position"]["y"] == 20.0
        assert state["position"]["z"] == 5.0
        assert state["orientation"]["w"] == 1.0
        assert state["status"] == "idle"
        assert state["resources"]["energy"] == 80.0
        assert state["experience_count"] == 42
        assert state["schema_version"] == AGENT_SCHEMA_VERSION
        assert len(state["short_term_memory"]) == 3

        # Check config
        config = serialized["config"]
        assert "movement" in config["capabilities"]
        assert "perception" in config["capabilities"]
        assert config["personality"]["openness"] == 0.8
        assert config["metadata"]["role"] == "scout"

        # Check beliefs
        beliefs = serialized["beliefs"]
        assert len(beliefs["relationships"]) == 1
        assert "agent_2" in beliefs["relationships"]
        assert beliefs["relationships"]["agent_2"]["trust_level"] == 0.8
        assert len(beliefs["goals"]) == 1
        assert beliefs["belief_state"] == [0.1, 0.2, 0.3, 0.4]
        assert beliefs["generative_model_params"]["param1"] == 1.0

    def test_serialize_agent_minimal(self):
        """Test serialization with minimal agent data."""
        minimal_agent = Agent()
        minimal_agent.agent_id = "minimal"
        minimal_agent.name = "MinimalAgent"

        serialized = self.persistence._serialize_agent(minimal_agent)

        # Should not crash and provide defaults
        assert serialized["state"]["current_goal"] is None
        assert serialized["state"]["short_term_memory"] == []
        assert serialized["beliefs"]["relationships"] == {}
        assert serialized["beliefs"]["goals"] == []

    def test_deserialize_agent_basic(self):
        """Test basic agent deserialization."""
        mock_db_agent = Mock()
        mock_db_agent.uuid = "test_id"
        mock_db_agent.name = "TestAgent"
        mock_db_agent.type = "explorer"
        mock_db_agent.created_at = datetime.now()
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = {}
        mock_db_agent.config = {}
        mock_db_agent.beliefs = {}

        agent = self.persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent, Agent)
        assert agent.agent_id == "test_id"
        assert agent.name == "TestAgent"
        assert agent.agent_type == "explorer"

    def test_deserialize_resource_agent(self):
        """Test deserializing resource agent."""
        mock_db_agent = Mock()
        mock_db_agent.uuid = "resource_id"
        mock_db_agent.name = "ResourceAgent"
        mock_db_agent.type = "resource_management"
        mock_db_agent.created_at = datetime.now()
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = {}
        mock_db_agent.config = {}
        mock_db_agent.beliefs = {}

        agent = self.persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent, ResourceAgent)
        assert agent.agent_id == "resource_id"

    def test_deserialize_social_agent(self):
        """Test deserializing social agent."""
        mock_db_agent = Mock()
        mock_db_agent.uuid = "social_id"
        mock_db_agent.name = "SocialAgent"
        mock_db_agent.type = "social_interaction"
        mock_db_agent.created_at = datetime.now()
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = {}
        mock_db_agent.config = {}
        mock_db_agent.beliefs = {}

        agent = self.persistence._deserialize_agent(mock_db_agent)

        assert isinstance(agent, SocialAgent)
        assert agent.agent_id == "social_id"

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialization and deserialization preserve agent data."""
        # Serialize the test agent
        serialized = self.persistence._serialize_agent(self.test_agent)

        # Create mock DB agent with serialized data
        mock_db_agent = Mock()
        mock_db_agent.uuid = self.test_agent.agent_id
        mock_db_agent.name = self.test_agent.name
        mock_db_agent.type = self.test_agent.agent_type
        mock_db_agent.created_at = self.test_agent.created_at
        mock_db_agent.updated_at = datetime.now()
        mock_db_agent.state = serialized["state"]
        mock_db_agent.config = serialized["config"]
        mock_db_agent.beliefs = serialized["beliefs"]

        # Deserialize
        restored_agent = self.persistence._deserialize_agent(mock_db_agent)

        # Verify key properties are preserved
        assert restored_agent.agent_id == self.test_agent.agent_id
        assert restored_agent.name == self.test_agent.name
        assert restored_agent.position.x == self.test_agent.position.x
        assert restored_agent.position.y == self.test_agent.position.y
        assert restored_agent.resources.energy == self.test_agent.resources.energy
        assert restored_agent.status == self.test_agent.status
        assert len(
            restored_agent.capabilities) == len(
            self.test_agent.capabilities)
        assert restored_agent.personality.openness == self.test_agent.personality.openness

    def test_serialize_goal(self):
        """Test goal serialization."""
        goal = AgentGoal(
            goal_id="test_goal",
            description="Test exploration goal",
            priority=0.8,
            target_position=Position(50.0, 50.0, 0.0),
            deadline=datetime.now() + timedelta(hours=1),
        )

        # The method is referenced but not shown in the file snippet
        # We'll test it indirectly through agent serialization
        agent = Agent()
        agent.current_goal = goal
        agent.goals = [goal]

        serialized = self.persistence._serialize_agent(agent)

        assert serialized["state"]["current_goal"] is not None
        assert len(serialized["beliefs"]["goals"]) == 1
