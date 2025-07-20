"""Characterization tests for Database Layer following Michael Feathers' principles.

Documents current behavior of:
- Model structure and relationships
- Database connection handling
- Transaction patterns
- Query optimization strategies
- JSON serialization behavior
- Error handling and rollback patterns
"""

import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from database.models import (
    Agent,
    Coalition, 
    AgentStatus,
    CoalitionStatus,
    AgentRole,
    agent_coalition_association,
    KnowledgeNode,
)
from database.base import Base
from database.connection_manager import DatabaseConnectionManager
from database.session import get_db
from database.types import GUID


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal


class TestDatabaseModelsCharacterization:
    """Characterize database model behavior."""
    
    def test_agent_model_structure(self, in_memory_db):
        """Document Agent model structure and defaults."""
        # Given
        session = in_memory_db()
        
        # When - Create agent with minimal data
        agent = Agent(
            name="Test Agent",
            template="explorer"
        )
        session.add(agent)
        session.commit()
        
        # Then - Document defaults and structure
        assert isinstance(agent.id, uuid.UUID)
        assert agent.name == "Test Agent"
        assert agent.template == "explorer"
        assert agent.status == AgentStatus.PENDING
        assert agent.gmn_spec is None
        assert agent.pymdp_config == {}
        assert agent.beliefs == {}
        assert agent.preferences == {}
        assert agent.position is None
        assert agent.metrics == {}
        assert agent.parameters == {}
        assert isinstance(agent.created_at, datetime)
        assert agent.last_active is None
        assert agent.inference_count == 0
        assert agent.total_steps == 0
        
        session.close()
        
    def test_agent_to_dict_serialization(self, in_memory_db):
        """Document agent to_dict serialization behavior."""
        # Given
        session = in_memory_db()
        agent = Agent(
            name="Serialization Test",
            template="active_inference",
            position={"x": 5, "y": 10},
            parameters={"learning_rate": 0.1},
            metrics={"accuracy": 0.85}
        )
        agent.last_active = datetime.utcnow()
        agent.inference_count = 42
        session.add(agent)
        session.commit()
        
        # When
        agent_dict = agent.to_dict()
        
        # Then - Document serialization format
        assert isinstance(agent_dict, dict)
        assert "id" in agent_dict
        assert isinstance(agent_dict["id"], str)  # UUID converted to string
        assert agent_dict["name"] == "Serialization Test"
        assert agent_dict["template"] == "active_inference"
        assert agent_dict["status"] == "pending"  # Enum value
        assert isinstance(agent_dict["created_at"], str)  # ISO format
        assert isinstance(agent_dict["last_active"], str)
        assert agent_dict["inference_count"] == 42
        assert agent_dict["parameters"] == {"learning_rate": 0.1}
        assert agent_dict["metrics"] == {"accuracy": 0.85}
        assert agent_dict["position"] == {"x": 5, "y": 10}
        
        session.close()
        
    def test_agent_coalition_relationship(self, in_memory_db):
        """Document agent-coalition many-to-many relationship."""
        # Given
        session = in_memory_db()
        
        # Create agents and coalition
        agent1 = Agent(name="Agent 1", template="explorer")
        agent2 = Agent(name="Agent 2", template="explorer")
        coalition = Coalition(
            name="Test Coalition",
            goal="Test coordination"
        )
        
        session.add_all([agent1, agent2, coalition])
        session.commit()
        
        # When - Add agents to coalition
        coalition.agents.append(agent1)
        coalition.agents.append(agent2)
        session.commit()
        
        # Then - Verify bidirectional relationship
        assert len(coalition.agents) == 2
        assert agent1 in coalition.agents
        assert agent2 in coalition.agents
        
        assert len(agent1.coalitions) == 1
        assert coalition in agent1.coalitions
        
        # Check association table
        result = session.execute(
            agent_coalition_association.select()
        ).fetchall()
        assert len(result) == 2
        
        session.close()
        
    def test_coalition_model_defaults(self, in_memory_db):
        """Document Coalition model defaults."""
        # Given
        session = in_memory_db()
        
        # When
        coalition = Coalition(
            name="Default Test",
            goal="Testing defaults"
        )
        session.add(coalition)
        session.commit()
        
        # Then
        assert isinstance(coalition.id, uuid.UUID)
        assert coalition.status == CoalitionStatus.FORMING
        assert coalition.strategy == {}
        assert coalition.performance_metrics == {}
        assert isinstance(coalition.created_at, datetime)
        assert coalition.dissolved_at is None
        
        session.close()
        
    def test_knowledge_node_structure(self, in_memory_db):
        """Document KnowledgeNode model structure."""
        # Given
        session = in_memory_db()
        
        # Create agent and node
        agent = Agent(name="Knowledge Creator", template="explorer")
        session.add(agent)
        session.commit()
        
        node = KnowledgeNode(
            node_type="concept",
            content={"concept": "test", "description": "A test concept"},
            creator_agent_id=agent.id
        )
        session.add(node)
        session.commit()
        
        # Then - Document structure
        assert isinstance(node.id, uuid.UUID)
        assert node.node_type == "concept"
        assert node.content == {"concept": "test", "description": "A test concept"}
        assert node.creator_agent_id == agent.id
        assert node.creator_agent == agent
        assert node in agent.knowledge_nodes
        
        session.close()
        


class TestDatabaseConnectionManagerCharacterization:
    """Characterize database connection management."""
    
    def test_connection_manager_initialization(self):
        """Document DatabaseConnectionManager initialization."""
        # Given
        db_url = "postgresql://test:test@localhost/test"
        
        # When
        manager = DatabaseConnectionManager(db_url)
        
        # Then
        assert manager.database_url == db_url
        assert hasattr(manager, 'engine')
        assert hasattr(manager, 'SessionLocal')
        assert manager._active_sessions == 0
        assert hasattr(manager, '_lock')
        
    def test_session_creation_pattern(self):
        """Document session creation behavior."""
        # Given
        manager = DatabaseConnectionManager("sqlite:///:memory:")
        
        # When
        session = manager.get_session()
        
        # Then
        assert isinstance(session, Session)
        assert manager._active_sessions == 1
        
        # When - Create another session
        session2 = manager.get_session()
        
        # Then
        assert session2 is not session  # Different session instances
        assert manager._active_sessions == 2
        
        # Cleanup
        session.close()
        session2.close()
        
    @patch('database.connection_manager.create_engine')
    def test_connection_pool_configuration(self, mock_create_engine):
        """Document connection pool configuration."""
        # When
        manager = DatabaseConnectionManager("postgresql://test/db")
        
        # Then - Verify pool configuration
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[1]
        
        assert call_args['pool_size'] == 10
        assert call_args['max_overflow'] == 20
        assert call_args['pool_pre_ping'] is True
        assert call_args['pool_recycle'] == 3600
        
    def test_session_context_manager(self):
        """Document session context manager behavior."""
        # Given
        manager = DatabaseConnectionManager("sqlite:///:memory:")
        Base.metadata.create_all(manager.engine)
        
        # When - Use context manager
        with manager.session_scope() as session:
            # Add test data
            agent = Agent(name="Context Test", template="test")
            session.add(agent)
            assert manager._active_sessions == 1
            
        # Then - Session closed automatically
        assert manager._active_sessions == 0
        
        # Verify data was committed
        with manager.session_scope() as session:
            agents = session.query(Agent).all()
            assert len(agents) == 1
            assert agents[0].name == "Context Test"
            
    def test_transaction_rollback_on_error(self):
        """Document rollback behavior on errors."""
        # Given
        manager = DatabaseConnectionManager("sqlite:///:memory:")
        Base.metadata.create_all(manager.engine)
        
        # When - Error occurs in transaction
        try:
            with manager.session_scope() as session:
                agent = Agent(name="Rollback Test", template="test")
                session.add(agent)
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass
            
        # Then - Transaction was rolled back
        with manager.session_scope() as session:
            agents = session.query(Agent).all()
            assert len(agents) == 0
            
    def test_concurrent_session_tracking(self):
        """Document concurrent session tracking."""
        # Given
        manager = DatabaseConnectionManager("sqlite:///:memory:")
        sessions = []
        
        # When - Create multiple sessions
        for i in range(5):
            session = manager.get_session()
            sessions.append(session)
            
        # Then
        assert manager._active_sessions == 5
        
        # When - Close some sessions
        sessions[0].close()
        sessions[2].close()
        
        # Then
        assert manager._active_sessions == 3
        
        # Cleanup
        for session in sessions:
            try:
                session.close()
            except:
                pass


class TestDatabaseQueryPatternsCharacterization:
    """Characterize common query patterns."""
    
    def test_lazy_loading_behavior(self, in_memory_db):
        """Document lazy loading of relationships."""
        # Given
        session = in_memory_db()
        
        # Create agent with coalition
        agent = Agent(name="Lazy Test", template="test")
        coalition = Coalition(name="Test Coalition", goal="Test")
        agent.coalitions.append(coalition)
        session.add(agent)
        session.commit()
        
        # Clear session to force lazy loading
        session.close()
        
        # When - New session queries agent
        session = in_memory_db()
        loaded_agent = session.query(Agent).filter_by(name="Lazy Test").first()
        
        # Then - Coalitions not loaded yet
        inspector = inspect(loaded_agent)
        assert 'coalitions' in inspector.unloaded
        
        # When - Access coalitions
        coalitions = loaded_agent.coalitions
        
        # Then - Now loaded
        assert len(coalitions) == 1
        assert 'coalitions' not in inspector.unloaded
        
        session.close()
        
    def test_bulk_insert_behavior(self, in_memory_db):
        """Document bulk insert patterns."""
        # Given
        session = in_memory_db()
        
        # When - Bulk insert agents
        agents = [
            Agent(name=f"Bulk Agent {i}", template="test")
            for i in range(100)
        ]
        session.bulk_save_objects(agents)
        session.commit()
        
        # Then
        count = session.query(Agent).count()
        assert count == 100
        
        # Note: bulk_save_objects doesn't populate IDs
        assert all(agent.id is None for agent in agents)
        
        session.close()
        
    def test_json_field_query_patterns(self, in_memory_db):
        """Document JSON field query patterns."""
        # Given
        session = in_memory_db()
        
        # Create agents with JSON data
        agent1 = Agent(
            name="JSON Agent 1",
            template="test",
            parameters={"learning_rate": 0.1, "temperature": 0.7}
        )
        agent2 = Agent(
            name="JSON Agent 2", 
            template="test",
            parameters={"learning_rate": 0.2, "temperature": 0.5}
        )
        session.add_all([agent1, agent2])
        session.commit()
        
        # When - Query by JSON field (SQLite doesn't support JSON operators)
        # This documents that JSON queries need special handling per database
        all_agents = session.query(Agent).all()
        
        # Then - Must filter in Python for SQLite
        high_lr_agents = [
            a for a in all_agents 
            if a.parameters.get("learning_rate", 0) > 0.15
        ]
        assert len(high_lr_agents) == 1
        assert high_lr_agents[0].name == "JSON Agent 2"
        
        session.close()


class TestDatabaseErrorHandlingCharacterization:
    """Characterize database error handling patterns."""
    
    def test_integrity_error_on_duplicate(self, in_memory_db):
        """Document integrity constraint behavior."""
        # Given
        session = in_memory_db()
        
        # Create agent with same ID to force integrity error
        agent_id = uuid.uuid4()
        agent1 = Agent(
            id=agent_id,
            name="First Agent",
            template="test"
        )
        session.add(agent1)
        session.commit()
        
        # When - Try to create duplicate ID
        agent2 = Agent(
            id=agent_id,  # Same ID
            name="Second Agent",
            template="test"
        )
        session.add(agent2)
        
        # Then - IntegrityError raised
        with pytest.raises(IntegrityError):
            session.commit()
            
        session.rollback()
        session.close()
        
    def test_connection_error_handling(self):
        """Document connection error handling."""
        # Given - Invalid database URL
        manager = DatabaseConnectionManager("postgresql://invalid:invalid@nohost/nodb")
        
        # When/Then - Connection errors are raised
        with pytest.raises(SQLAlchemyError):
            with manager.session_scope() as session:
                session.query(Agent).all()
                
    def test_session_cleanup_on_error(self):
        """Document session cleanup on errors."""
        # Given
        manager = DatabaseConnectionManager("sqlite:///:memory:")
        Base.metadata.create_all(manager.engine)
        
        # When - Error in session scope
        initial_sessions = manager._active_sessions
        
        try:
            with manager.session_scope() as session:
                # Verify session is tracked
                assert manager._active_sessions == initial_sessions + 1
                # Force error
                raise RuntimeError("Test error")
        except RuntimeError:
            pass
            
        # Then - Session was cleaned up
        assert manager._active_sessions == initial_sessions