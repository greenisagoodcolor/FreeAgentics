"""Fixed test to verify database coverage setup works."""

import os
from unittest.mock import MagicMock, patch

# Set a mock DATABASE_URL before importing database modules
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/testdb"

# Mock sqlalchemy components before import
mock_engine = MagicMock()
mock_sessionmaker = MagicMock()
mock_session = MagicMock()
mock_sessionmaker.return_value = mock_session

with patch("sqlalchemy.create_engine", return_value=mock_engine):
    with patch("sqlalchemy.orm.sessionmaker", return_value=mock_sessionmaker):
        # Import database base first
        from database.base import Base

        # Import models
        from database.models import (
            Agent,
            AgentStatus,
            Coalition,
            CoalitionStatus,
            KnowledgeEdge,
            KnowledgeNode,
        )

        # Import session components
        from database.session import get_db

        # Import types
        from database.types import GUID


def test_base_imports():
    """Test that base classes are imported."""
    assert Base is not None
    assert hasattr(Base, "metadata")


def test_model_enums():
    """Test that enum classes exist."""
    # Test AgentStatus enum
    assert AgentStatus.PENDING.value == "pending"
    assert AgentStatus.ACTIVE.value == "active"
    assert AgentStatus.PAUSED.value == "paused"

    # Test CoalitionStatus enum
    assert CoalitionStatus.FORMING.value == "forming"
    assert CoalitionStatus.ACTIVE.value == "active"


def test_agent_model():
    """Test Agent model structure."""
    assert hasattr(Agent, "__tablename__")
    assert Agent.__tablename__ == "agents"
    assert hasattr(Agent, "id")
    assert hasattr(Agent, "name")
    assert hasattr(Agent, "status")
    assert hasattr(Agent, "created_at")
    assert hasattr(Agent, "updated_at")


def test_coalition_model():
    """Test Coalition model structure."""
    assert hasattr(Coalition, "__tablename__")
    assert Coalition.__tablename__ == "coalitions"
    assert hasattr(Coalition, "id")
    assert hasattr(Coalition, "name")
    assert hasattr(Coalition, "description")
    assert hasattr(Coalition, "status")


def test_knowledge_node_model():
    """Test KnowledgeNode model structure."""
    assert hasattr(KnowledgeNode, "__tablename__")
    assert KnowledgeNode.__tablename__ == "knowledge_nodes"
    assert hasattr(KnowledgeNode, "id")
    assert hasattr(KnowledgeNode, "content")
    assert hasattr(KnowledgeNode, "agent_id")


def test_knowledge_edge_model():
    """Test KnowledgeEdge model structure."""
    assert hasattr(KnowledgeEdge, "__tablename__")
    assert KnowledgeEdge.__tablename__ == "knowledge_edges"
    assert hasattr(KnowledgeEdge, "id")
    assert hasattr(KnowledgeEdge, "source_id")
    assert hasattr(KnowledgeEdge, "target_id")
    assert hasattr(KnowledgeEdge, "weight")


def test_guid_type():
    """Test GUID custom type."""
    assert GUID is not None
    guid_instance = GUID()
    assert hasattr(guid_instance, "impl")


def test_get_db_function():
    """Test get_db function exists."""
    assert get_db is not None
    assert callable(get_db)


# BaseModel abstract test removed - no BaseModel class exists in database layer
