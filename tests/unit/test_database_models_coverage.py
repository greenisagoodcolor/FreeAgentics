"""Comprehensive tests for database.models module to achieve high coverage."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import (
    Agent,
    AgentRole,
    AgentStatus,
    Coalition,
    CoalitionStatus,
    KnowledgeEdge,
    KnowledgeNode,
    agent_coalition_association,
)


class TestDatabaseModels:
    """Test database models comprehensively."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        SessionLocal = sessionmaker(bind=test_engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_agent_status_enum(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.STOPPED.value == "stopped"
        assert AgentStatus.ERROR.value == "error"

    def test_coalition_status_enum(self):
        """Test CoalitionStatus enum values."""
        assert CoalitionStatus.FORMING.value == "forming"
        assert CoalitionStatus.ACTIVE.value == "active"
        assert CoalitionStatus.DISBANDING.value == "disbanding"
        assert CoalitionStatus.DISSOLVED.value == "dissolved"

    def test_agent_role_enum(self):
        """Test AgentRole enum values."""
        assert AgentRole.LEADER.value == "leader"
        assert AgentRole.COORDINATOR.value == "coordinator"
        assert AgentRole.MEMBER.value == "member"
        assert AgentRole.OBSERVER.value == "observer"

    def test_agent_creation(self, test_session):
        """Test creating an agent with all fields."""
        agent_id = uuid.uuid4()
        agent = Agent(
            id=agent_id,
            name="Test Agent",
            template="explorer",
            status=AgentStatus.ACTIVE,
            gmn_spec="test_spec",
            pymdp_config={"config": "value"},
            beliefs={"belief": "value"},
            preferences={"pref": "value"},
            position=[0, 0],
            metrics={"metric": 1.0},
            parameters={"param": "value"},
            last_active=datetime.now(),
            inference_count=10,
            total_steps=100,
        )

        test_session.add(agent)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(Agent).filter_by(id=agent_id).first()
        assert retrieved is not None
        assert retrieved.name == "Test Agent"
        assert retrieved.template == "explorer"
        assert retrieved.status == AgentStatus.ACTIVE
        assert retrieved.gmn_spec == "test_spec"
        assert retrieved.pymdp_config == {"config": "value"}
        assert retrieved.beliefs == {"belief": "value"}
        assert retrieved.preferences == {"pref": "value"}
        assert retrieved.position == [0, 0]
        assert retrieved.metrics == {"metric": 1.0}
        assert retrieved.parameters == {"param": "value"}
        assert retrieved.inference_count == 10
        assert retrieved.total_steps == 100
        assert retrieved.last_active is not None

    def test_agent_defaults(self, test_session):
        """Test agent creation with default values."""
        agent = Agent(
            name="Default Agent",
            template="basic",
        )

        test_session.add(agent)
        test_session.commit()

        assert agent.status == AgentStatus.PENDING
        assert agent.pymdp_config == {}
        assert agent.beliefs == {}
        assert agent.preferences == {}
        assert agent.metrics == {}
        assert agent.parameters == {}
        assert agent.inference_count == 0
        assert agent.total_steps == 0
        assert agent.last_active is None

    def test_agent_to_dict(self, test_session):
        """Test agent to_dict method with all fields."""
        now = datetime.now()
        agent = Agent(
            name="Dict Agent",
            template="explorer",
            status=AgentStatus.ACTIVE,
            last_active=now,
            inference_count=5,
            parameters={"key": "value"},
            metrics={"score": 0.9},
            position=[1, 2],
        )

        test_session.add(agent)
        test_session.commit()

        agent_dict = agent.to_dict()

        assert agent_dict["name"] == "Dict Agent"
        assert agent_dict["template"] == "explorer"
        assert agent_dict["status"] == "active"
        assert agent_dict["inference_count"] == 5
        assert agent_dict["parameters"] == {"key": "value"}
        assert agent_dict["metrics"] == {"score": 0.9}
        assert agent_dict["position"] == [1, 2]
        assert "id" in agent_dict
        assert "created_at" in agent_dict
        assert "last_active" in agent_dict

    def test_agent_to_dict_no_last_active(self, test_session):
        """Test agent to_dict when last_active is None."""
        agent = Agent(name="No Last Active", template="basic")
        test_session.add(agent)
        test_session.commit()

        agent_dict = agent.to_dict()
        assert agent_dict["last_active"] is None

    def test_coalition_creation(self, test_session):
        """Test creating a coalition with all fields."""
        coalition_id = uuid.uuid4()
        coalition = Coalition(
            id=coalition_id,
            name="Test Coalition",
            description="Test description",
            status=CoalitionStatus.ACTIVE,
            objectives={"goal": "achieve"},
            required_capabilities=["capability1", "capability2"],
            achieved_objectives=["objective1"],
            performance_score=0.8,
            cohesion_score=0.9,
            dissolved_at=datetime.now(),
        )

        test_session.add(coalition)
        test_session.commit()

        retrieved = test_session.query(Coalition).filter_by(id=coalition_id).first()
        assert retrieved is not None
        assert retrieved.name == "Test Coalition"
        assert retrieved.description == "Test description"
        assert retrieved.status == CoalitionStatus.ACTIVE
        assert retrieved.objectives == {"goal": "achieve"}
        assert retrieved.required_capabilities == [
            "capability1",
            "capability2",
        ]
        assert retrieved.achieved_objectives == ["objective1"]
        assert retrieved.performance_score == 0.8
        assert retrieved.cohesion_score == 0.9
        assert retrieved.dissolved_at is not None

    def test_coalition_defaults(self, test_session):
        """Test coalition creation with default values."""
        coalition = Coalition(name="Default Coalition")

        test_session.add(coalition)
        test_session.commit()

        assert coalition.status == CoalitionStatus.FORMING
        assert coalition.objectives == {}
        assert coalition.required_capabilities == []
        assert coalition.achieved_objectives == []
        assert coalition.performance_score == 0.0
        assert coalition.cohesion_score == 1.0
        assert coalition.dissolved_at is None

    def test_coalition_to_dict(self, test_session):
        """Test coalition to_dict method."""
        now = datetime.now()
        coalition = Coalition(
            name="Dict Coalition",
            description="Test",
            status=CoalitionStatus.ACTIVE,
            objectives={"obj": 1},
            required_capabilities=["cap1"],
            achieved_objectives=["obj1"],
            performance_score=0.7,
            cohesion_score=0.8,
            dissolved_at=now,
        )

        test_session.add(coalition)
        test_session.commit()

        coalition_dict = coalition.to_dict()

        assert coalition_dict["name"] == "Dict Coalition"
        assert coalition_dict["description"] == "Test"
        assert coalition_dict["status"] == "active"
        assert coalition_dict["objectives"] == {"obj": 1}
        assert coalition_dict["required_capabilities"] == ["cap1"]
        assert coalition_dict["achieved_objectives"] == ["obj1"]
        assert coalition_dict["performance_score"] == 0.7
        assert coalition_dict["cohesion_score"] == 0.8
        assert coalition_dict["agent_count"] == 0
        assert "dissolved_at" in coalition_dict
        assert coalition_dict["dissolved_at"] is not None

    def test_coalition_to_dict_no_dissolved_at(self, test_session):
        """Test coalition to_dict when dissolved_at is None."""
        coalition = Coalition(name="Active Coalition")
        test_session.add(coalition)
        test_session.commit()

        coalition_dict = coalition.to_dict()
        assert coalition_dict["dissolved_at"] is None

    def test_agent_coalition_relationship(self, test_session):
        """Test many-to-many relationship between agents and coalitions."""
        # Create agents
        agent1 = Agent(name="Agent 1", template="basic")
        agent2 = Agent(name="Agent 2", template="explorer")

        # Create coalition
        coalition = Coalition(name="Test Coalition")

        test_session.add_all([agent1, agent2, coalition])
        test_session.commit()

        # Add agents to coalition
        coalition.agents.append(agent1)
        coalition.agents.append(agent2)
        test_session.commit()

        # Verify relationships
        assert len(coalition.agents) == 2
        assert agent1 in coalition.agents
        assert agent2 in coalition.agents

        assert len(agent1.coalitions) == 1
        assert coalition in agent1.coalitions

        # Test coalition to_dict shows correct agent count
        coalition_dict = coalition.to_dict()
        assert coalition_dict["agent_count"] == 2

    def test_knowledge_node_creation(self, test_session):
        """Test creating a knowledge node with all fields."""
        agent = Agent(name="Creator Agent", template="basic")
        test_session.add(agent)
        test_session.commit()

        node_id = uuid.uuid4()
        node = KnowledgeNode(
            id=node_id,
            type="concept",
            label="Test Concept",
            properties={"key": "value"},
            version=2,
            is_current=False,
            confidence=0.85,
            source="test_source",
            creator_agent_id=agent.id,
        )

        test_session.add(node)
        test_session.commit()

        retrieved = test_session.query(KnowledgeNode).filter_by(id=node_id).first()
        assert retrieved is not None
        assert retrieved.type == "concept"
        assert retrieved.label == "Test Concept"
        assert retrieved.properties == {"key": "value"}
        assert retrieved.version == 2
        assert retrieved.is_current is False
        assert retrieved.confidence == 0.85
        assert retrieved.source == "test_source"
        assert retrieved.creator_agent_id == agent.id
        assert retrieved.creator_agent == agent

    def test_knowledge_node_defaults(self, test_session):
        """Test knowledge node creation with default values."""
        node = KnowledgeNode(
            type="entity",
            label="Default Node",
        )

        test_session.add(node)
        test_session.commit()

        assert node.properties == {}
        assert node.version == 1
        assert node.is_current is True
        assert node.confidence == 1.0
        assert node.source is None
        assert node.creator_agent_id is None

    def test_knowledge_edge_creation(self, test_session):
        """Test creating a knowledge edge between nodes."""
        # Create nodes
        node1 = KnowledgeNode(type="entity", label="Node 1")
        node2 = KnowledgeNode(type="entity", label="Node 2")
        test_session.add_all([node1, node2])
        test_session.commit()

        # Create edge
        edge_id = uuid.uuid4()
        edge = KnowledgeEdge(
            id=edge_id,
            source_id=node1.id,
            target_id=node2.id,
            type="relates_to",
            properties={"strength": 0.9},
            confidence=0.95,
        )

        test_session.add(edge)
        test_session.commit()

        retrieved = test_session.query(KnowledgeEdge).filter_by(id=edge_id).first()
        assert retrieved is not None
        assert retrieved.source_id == node1.id
        assert retrieved.target_id == node2.id
        assert retrieved.type == "relates_to"
        assert retrieved.properties == {"strength": 0.9}
        assert retrieved.confidence == 0.95

    def test_knowledge_edge_defaults(self, test_session):
        """Test knowledge edge creation with default values."""
        node1 = KnowledgeNode(type="entity", label="Node 1")
        node2 = KnowledgeNode(type="entity", label="Node 2")
        test_session.add_all([node1, node2])
        test_session.commit()

        edge = KnowledgeEdge(
            source_id=node1.id,
            target_id=node2.id,
            type="links_to",
        )

        test_session.add(edge)
        test_session.commit()

        assert edge.properties == {}
        assert edge.confidence == 1.0

    def test_knowledge_node_edge_relationships(self, test_session):
        """Test relationships between knowledge nodes and edges."""
        # Create nodes
        node1 = KnowledgeNode(type="entity", label="Source")
        node2 = KnowledgeNode(type="entity", label="Target")
        node3 = KnowledgeNode(type="entity", label="Other")
        test_session.add_all([node1, node2, node3])
        test_session.commit()

        # Create edges
        edge1 = KnowledgeEdge(
            source_id=node1.id,
            target_id=node2.id,
            type="connects",
        )
        edge2 = KnowledgeEdge(
            source_id=node1.id,
            target_id=node3.id,
            type="relates",
        )
        edge3 = KnowledgeEdge(
            source_id=node2.id,
            target_id=node3.id,
            type="links",
        )
        test_session.add_all([edge1, edge2, edge3])
        test_session.commit()

        # Test outgoing edges
        assert len(node1.outgoing_edges) == 2
        assert edge1 in node1.outgoing_edges
        assert edge2 in node1.outgoing_edges

        # Test incoming edges
        assert len(node2.incoming_edges) == 1
        assert edge1 in node2.incoming_edges

        assert len(node3.incoming_edges) == 2
        assert edge2 in node3.incoming_edges
        assert edge3 in node3.incoming_edges

        # Test edge relationships
        assert edge1.source_node == node1
        assert edge1.target_node == node2

    def test_agent_knowledge_node_relationship(self, test_session):
        """Test relationship between agents and knowledge nodes they create."""
        agent = Agent(name="Knowledge Creator", template="researcher")
        test_session.add(agent)
        test_session.commit()

        # Create knowledge nodes
        node1 = KnowledgeNode(
            type="discovery",
            label="Discovery 1",
            creator_agent_id=agent.id,
        )
        node2 = KnowledgeNode(
            type="discovery",
            label="Discovery 2",
            creator_agent_id=agent.id,
        )
        test_session.add_all([node1, node2])
        test_session.commit()

        # Test relationship
        assert len(agent.knowledge_nodes) == 2
        assert node1 in agent.knowledge_nodes
        assert node2 in agent.knowledge_nodes
        assert node1.creator_agent == agent
        assert node2.creator_agent == agent

    def test_association_table_columns(self):
        """Test agent_coalition_association table has all columns."""
        # Test table columns exist
        assert "agent_id" in [c.name for c in agent_coalition_association.columns]
        assert "coalition_id" in [c.name for c in agent_coalition_association.columns]
        assert "role" in [c.name for c in agent_coalition_association.columns]
        assert "joined_at" in [c.name for c in agent_coalition_association.columns]
        assert "contribution_score" in [c.name for c in agent_coalition_association.columns]
        assert "trust_score" in [c.name for c in agent_coalition_association.columns]
