"""Pytest fixtures for test data management.

Provides reusable fixtures for common test scenarios with automatic
cleanup and transaction management.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Generator, List

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.base import Base
from database.models import (
    Agent,
    AgentStatus,
    Coalition,
    CoalitionStatus,
    KnowledgeEdge,
    KnowledgeNode,
)

from .builders import (
    AgentBuilder,
    CoalitionBuilder,
    KnowledgeEdgeBuilder,
    KnowledgeNodeBuilder,
)
from .factories import (
    AgentFactory,
    CoalitionFactory,
    KnowledgeGraphFactory,
    PerformanceDataFactory,
)


# Database fixtures
@pytest.fixture(scope="function")
def test_engine():
    """Create an in-memory SQLite engine for testing."""
    # Use StaticPool to ensure same connection across threads
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable foreign key support for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """Create a database session with automatic rollback."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()

    # Begin a transaction
    session.begin()

    yield session

    # Rollback transaction for cleanup
    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def clean_database(db_session: Session) -> Session:
    """Ensure database is clean before test."""
    # Clear all data (in order due to foreign keys)
    db_session.query(KnowledgeEdge).delete()
    db_session.query(KnowledgeNode).delete()
    db_session.execute("DELETE FROM agent_coalition")  # Association table
    db_session.query(Coalition).delete()
    db_session.query(Agent).delete()
    db_session.commit()

    return db_session


# Agent fixtures
@pytest.fixture
def agent_fixture(db_session: Session) -> Agent:
    """Create a single test agent."""
    return AgentFactory.create(
        session=db_session,
        name="TestAgent",
        template="grid_world",
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture
def agent_builder() -> AgentBuilder:
    """Provide an agent builder for custom agent creation."""
    return AgentBuilder()


@pytest.fixture
def active_agent(db_session: Session) -> Agent:
    """Create an active agent with full configuration."""
    agent = (
        AgentBuilder()
        .with_name("ActiveTestAgent")
        .active()
        .with_grid_world_config()
        .with_uniform_beliefs(num_states=5)
        .with_random_metrics()
        .with_position(5.0, 5.0)
        .build()
    )

    db_agent = Agent(**agent.dict())
    db_session.add(db_agent)
    db_session.commit()
    db_session.refresh(db_agent)

    return db_agent


@pytest.fixture
def agent_batch(db_session: Session) -> List[Agent]:
    """Create a batch of test agents."""
    return AgentFactory.create_batch(
        session=db_session,
        count=10,
        template="grid_world",
        status=AgentStatus.ACTIVE,
        distribute_positions=True,
        position_bounds={"min": [0, 0], "max": [20, 20]},
    )


@pytest.fixture
def resource_collector_agent(db_session: Session) -> Agent:
    """Create a resource collector agent."""
    agent = (
        AgentBuilder()
        .as_resource_collector()
        .with_name("ResourceCollector01")
        .active()
        .build()
    )

    db_agent = Agent(**agent.dict())
    db_session.add(db_agent)
    db_session.commit()
    db_session.refresh(db_agent)

    return db_agent


@pytest.fixture
def explorer_agent(db_session: Session) -> Agent:
    """Create an explorer agent."""
    agent = (
        AgentBuilder().as_explorer().with_name("Explorer01").active().build()
    )

    db_agent = Agent(**agent.dict())
    db_session.add(db_agent)
    db_session.commit()
    db_session.refresh(db_agent)

    return db_agent


@pytest.fixture
def coordinator_agent(db_session: Session) -> Agent:
    """Create a coordinator agent."""
    agent = (
        AgentBuilder()
        .as_coordinator()
        .with_name("Coordinator01")
        .active()
        .build()
    )

    db_agent = Agent(**agent.dict())
    db_session.add(db_agent)
    db_session.commit()
    db_session.refresh(db_agent)

    return db_agent


# Coalition fixtures
@pytest.fixture
def coalition_fixture(db_session: Session) -> Coalition:
    """Create a single test coalition."""
    return CoalitionFactory.create(
        session=db_session, name="TestCoalition", status=CoalitionStatus.ACTIVE
    )


@pytest.fixture
def coalition_builder() -> CoalitionBuilder:
    """Provide a coalition builder for custom coalition creation."""
    return CoalitionBuilder()


@pytest.fixture
def coalition_with_agents(
    db_session: Session, agent_batch: List[Agent]
) -> Coalition:
    """Create a coalition with member agents."""
    return CoalitionFactory.create(
        session=db_session,
        agents=agent_batch[:5],  # Use first 5 agents
        name="CoalitionWithMembers",
        status=CoalitionStatus.ACTIVE,
    )


@pytest.fixture
def resource_coalition(db_session: Session) -> Coalition:
    """Create a resource-focused coalition."""
    coalition = (
        CoalitionBuilder()
        .as_resource_coalition()
        .with_name("ResourceOptimizers")
        .active()
        .build()
    )

    db_coalition = Coalition(**coalition.dict())
    db_session.add(db_coalition)
    db_session.commit()
    db_session.refresh(db_coalition)

    return db_coalition


@pytest.fixture
def exploration_coalition(db_session: Session) -> Coalition:
    """Create an exploration-focused coalition."""
    coalition = (
        CoalitionBuilder()
        .as_exploration_coalition()
        .with_name("Explorers")
        .active()
        .build()
    )

    db_coalition = Coalition(**coalition.dict())
    db_session.add(db_coalition)
    db_session.commit()
    db_session.refresh(db_coalition)

    return db_coalition


@pytest.fixture
def coalition_network(db_session: Session) -> Dict[str, Any]:
    """Create a network of interconnected coalitions."""
    return CoalitionFactory.create_coalition_network(
        session=db_session,
        num_coalitions=3,
        agents_per_coalition=5,
        overlap_probability=0.2,
    )


# Knowledge graph fixtures
@pytest.fixture
def knowledge_node_fixture(db_session: Session) -> KnowledgeNode:
    """Create a single test knowledge node."""
    return KnowledgeGraphFactory.create_node(
        session=db_session, type="concept", label="TestConcept", confidence=0.9
    )


@pytest.fixture
def knowledge_node_builder() -> KnowledgeNodeBuilder:
    """Provide a knowledge node builder."""
    return KnowledgeNodeBuilder()


@pytest.fixture
def knowledge_edge_builder() -> KnowledgeEdgeBuilder:
    """Provide a knowledge edge builder."""
    return KnowledgeEdgeBuilder()


@pytest.fixture
def knowledge_graph_fixture(db_session: Session) -> Dict[str, Any]:
    """Create a test knowledge graph."""
    return KnowledgeGraphFactory.create_knowledge_graph(
        session=db_session, num_nodes=10, connectivity=0.2
    )


@pytest.fixture
def large_knowledge_graph(db_session: Session) -> Dict[str, Any]:
    """Create a large knowledge graph for performance testing."""
    return KnowledgeGraphFactory.create_knowledge_graph(
        session=db_session,
        num_nodes=100,
        connectivity=0.05,
        node_types=["concept", "entity", "fact", "observation", "inference"],
        edge_types=[
            "relates_to",
            "causes",
            "supports",
            "contradicts",
            "derived_from",
        ],
    )


@pytest.fixture
def agent_knowledge_scenario(
    db_session: Session, active_agent: Agent
) -> Dict[str, Any]:
    """Create a knowledge scenario for an agent."""
    return KnowledgeGraphFactory.create_agent_knowledge_scenario(
        session=db_session,
        agent=active_agent,
        num_observations=5,
        num_inferences=3,
    )


# Complex scenario fixtures
@pytest.fixture
def multi_agent_scenario(db_session: Session) -> Dict[str, Any]:
    """Create a complete multi-agent scenario."""
    # Create diverse agents
    agents = []
    for template in ["resource_collector", "explorer", "coordinator"]:
        batch = AgentFactory.create_batch(
            session=db_session,
            count=3,
            template=template,
            status=AgentStatus.ACTIVE,
            name_prefix=f"{template.title()}",
        )
        agents.extend(batch)

    # Create coalitions
    coalitions = []

    # Resource coalition
    resource_agents = [a for a in agents if "resource" in a.template]
    resource_coalition = CoalitionFactory.create(
        session=db_session, agents=resource_agents, name="ResourceGatherers"
    )
    coalitions.append(resource_coalition)

    # Exploration coalition
    explorer_agents = [a for a in agents if "explorer" in a.template]
    exploration_coalition = CoalitionFactory.create(
        session=db_session, agents=explorer_agents, name="Pathfinders"
    )
    coalitions.append(exploration_coalition)

    # Mixed coalition with coordinators
    coordinator_agents = [a for a in agents if "coordinator" in a.template]
    mixed_agents = (
        coordinator_agents + resource_agents[:1] + explorer_agents[:1]
    )
    mixed_coalition = CoalitionFactory.create(
        session=db_session, agents=mixed_agents, name="StrategicAlliance"
    )
    coalitions.append(mixed_coalition)

    return {
        "agents": agents,
        "coalitions": coalitions,
        "agent_types": {
            "resource_collectors": resource_agents,
            "explorers": explorer_agents,
            "coordinators": coordinator_agents,
        },
    }


@pytest.fixture
def performance_test_scenario(db_session: Session) -> Dict[str, Any]:
    """Create a performance test scenario."""
    from .schemas import PerformanceTestConfigSchema

    config = PerformanceTestConfigSchema(
        num_agents=50,
        num_coalitions=5,
        num_knowledge_nodes=100,
        knowledge_graph_connectivity=0.1,
        batch_size=50,
    )

    return PerformanceDataFactory.create_performance_scenario(
        session=db_session, config=config
    )


@pytest.fixture
def stress_test_data(db_session: Session) -> Dict[str, Any]:
    """Create stress test data."""
    return PerformanceDataFactory.create_stress_test_data(
        session=db_session, scale_factor=5  # 5x normal data volume
    )


# Mock data fixtures
@pytest.fixture
def mock_agent_data() -> Dict[str, Any]:
    """Provide mock agent data without database."""
    return AgentBuilder().as_resource_collector().build().dict()


@pytest.fixture
def mock_coalition_data() -> Dict[str, Any]:
    """Provide mock coalition data without database."""
    return CoalitionBuilder().as_exploration_coalition().build().dict()


@pytest.fixture
def mock_knowledge_node_data() -> Dict[str, Any]:
    """Provide mock knowledge node data without database."""
    return KnowledgeNodeBuilder().as_concept("TestConcept").build().dict()


# Parameterized fixtures
@pytest.fixture(
    params=["grid_world", "resource_collector", "explorer", "coordinator"]
)
def agent_by_template(request, db_session: Session) -> Agent:
    """Create agents of different templates."""
    return AgentFactory.create(
        session=db_session,
        template=request.param,
        name=f"Test{request.param.title().replace('_', '')}",
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture(params=[5, 10, 20])
def agent_batch_sized(request, db_session: Session) -> List[Agent]:
    """Create agent batches of different sizes."""
    return AgentFactory.create_batch(
        session=db_session,
        count=request.param,
        template="grid_world",
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture(params=[0.1, 0.2, 0.3])
def knowledge_graph_connectivity(
    request, db_session: Session
) -> Dict[str, Any]:
    """Create knowledge graphs with different connectivity levels."""
    return KnowledgeGraphFactory.create_knowledge_graph(
        session=db_session, num_nodes=20, connectivity=request.param
    )


# Helper fixtures
@pytest.fixture
def random_uuid() -> uuid.UUID:
    """Generate a random UUID."""
    return uuid.uuid4()


@pytest.fixture
def current_timestamp() -> datetime:
    """Get current timestamp."""
    return datetime.utcnow()


@pytest.fixture
def test_data_config() -> Dict[str, Any]:
    """Provide test data configuration."""
    return {
        "agent_templates": [
            "grid_world",
            "resource_collector",
            "explorer",
            "coordinator",
        ],
        "coalition_types": ["resource", "exploration", "defense"],
        "knowledge_node_types": [
            "concept",
            "entity",
            "fact",
            "observation",
            "inference",
        ],
        "knowledge_edge_types": [
            "relates_to",
            "causes",
            "supports",
            "contradicts",
            "derived_from",
        ],
        "default_position_bounds": {"min": [0, 0], "max": [100, 100]},
        "default_confidence_range": (0.5, 1.0),
        "default_performance_range": (0.6, 0.95),
    }
