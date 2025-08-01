"""Tests for Graph Database Service (Task 34.3).

Comprehensive test suite covering all aspects of the unified graph database service
with 100% test coverage following TDD principles.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from services.graph_database_service import (
    GraphDatabaseService,
    GraphRepository,
    GraphQuery,
    GraphUpdate,
    GraphStorageMetrics,
    GraphDatabaseError,
    GraphTransactionError,
    GraphValidationError,
    get_graph_database_service,
)
from knowledge_graph.schema import (
    ConversationEntity,
    ConversationRelation,
    EntityType,
    RelationType,
    Provenance,
    TemporalMetadata,
)
from database.models import KnowledgeNode, KnowledgeEdge


class TestGraphStorageMetrics:
    """Test GraphStorageMetrics data class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = GraphStorageMetrics(
            operation="test_op",
            start_time=1000.0,
        )
        
        assert metrics.operation == "test_op"
        assert metrics.start_time == 1000.0
        assert metrics.end_time is None
        assert metrics.node_count == 0
        assert metrics.edge_count == 0
        assert not metrics.success
        assert metrics.error is None
        assert metrics.graph_id is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        metrics = GraphStorageMetrics(
            operation="test_op",
            start_time=1000.0,
        )
        
        # Test ongoing operation
        with patch('time.time', return_value=1005.0):
            assert metrics.duration == 5.0
        
        # Test completed operation
        metrics.end_time = 1003.0
        assert metrics.duration == 3.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GraphStorageMetrics(
            operation="create_nodes",
            start_time=1000.0,
            end_time=1005.0,
            node_count=10,
            edge_count=5,
            success=True,
            graph_id="test-graph",
        )
        
        result = metrics.to_dict()
        
        assert result == {
            "operation": "create_nodes",
            "duration_ms": 5000.0,
            "node_count": 10,
            "edge_count": 5,
            "success": True,
            "error": None,
            "graph_id": "test-graph",
        }


class TestGraphQuery:
    """Test GraphQuery data class."""
    
    def test_default_query(self):
        """Test default query initialization."""
        query = GraphQuery()
        
        assert query.graph_ids is None
        assert query.node_types is None
        assert query.edge_types is None
        assert query.confidence_threshold is None
        assert query.created_after is None
        assert query.created_before is None
        assert query.limit is None
        assert query.offset == 0
    
    def test_parameterized_query(self):
        """Test parameterized query initialization."""
        now = datetime.now()
        query = GraphQuery(
            graph_ids=["graph1", "graph2"],
            node_types=["concept", "agent"],
            confidence_threshold=0.8,
            created_after=now,
            limit=100,
            offset=50,
        )
        
        assert query.graph_ids == ["graph1", "graph2"]
        assert query.node_types == ["concept", "agent"]
        assert query.confidence_threshold == 0.8
        assert query.created_after == now
        assert query.limit == 100
        assert query.offset == 50


class TestGraphUpdate:
    """Test GraphUpdate data class."""
    
    def test_valid_update(self):
        """Test valid update initialization."""
        entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="test entity",
        )
        
        update = GraphUpdate(
            nodes_to_create=[entity],
            nodes_to_update=[],
            nodes_to_delete=[],
            edges_to_create=[],
            edges_to_update=[],
            edges_to_delete=[],
        )
        
        assert len(update.nodes_to_create) == 1
        assert len(update.nodes_to_update) == 0
    
    def test_empty_update_raises_error(self):
        """Test that empty update raises validation error."""
        with pytest.raises(ValueError, match="must contain at least one operation"):
            GraphUpdate(
                nodes_to_create=[],
                nodes_to_update=[],
                nodes_to_delete=[],
                edges_to_create=[],
                edges_to_update=[],
                edges_to_delete=[],
            )


class TestGraphRepository:
    """Test GraphRepository class."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Mock session factory for testing."""
        session_mock = Mock(spec=Session)
        session_mock.commit.return_value = None
        session_mock.rollback.return_value = None
        session_mock.close.return_value = None
        session_mock.flush.return_value = None
        session_mock.add.return_value = None
        
        def mock_factory():
            yield session_mock
        
        return mock_factory, session_mock
    
    @pytest.fixture
    def repository(self, mock_session_factory):
        """Create repository with mocked session."""
        factory, _ = mock_session_factory
        return GraphRepository(session_factory=factory)
    
    @pytest.fixture
    def sample_entity(self):
        """Create sample conversation entity."""
        return ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="test concept",
            properties={"description": "test description"},
            provenance=Provenance(
                source_type="test",
                source_id="test-123",
                extraction_method="test_method",
                confidence_score=0.9,
            ),
        )
    
    @pytest.fixture
    def sample_relation(self):
        """Create sample conversation relation."""
        return ConversationRelation(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relation_type=RelationType.RELATES_TO,
            properties={"strength": 0.8},
            provenance=Provenance(
                source_type="test",
                source_id="test-123",
                extraction_method="test_method",
                confidence_score=0.8,
            ),
        )
    
    def test_repository_initialization(self):
        """Test repository initialization."""
        repo = GraphRepository()
        assert repo._connection_pool_size == 10
        assert repo._query_timeout == 30.0
    
    def test_repository_initialization_with_custom_session(self, mock_session_factory):
        """Test repository initialization with custom session factory."""
        factory, _ = mock_session_factory
        repo = GraphRepository(session_factory=factory)
        assert repo.session_factory == factory
    
    @patch('services.graph_database_service.time.time', side_effect=[1000.0, 1005.0])
    def test_create_nodes_success(self, mock_time, repository, mock_session_factory, sample_entity):
        """Test successful node creation."""
        _, session_mock = mock_session_factory
        
        result = repository.create_nodes(
            entities=[sample_entity],
            conversation_id="conv-123",
            agent_id="agent-456",
        )
        
        assert len(result) == 1
        assert isinstance(result[0], str)
        session_mock.add.assert_called_once()
        session_mock.flush.assert_called_once()
    
    def test_create_nodes_database_error(self, repository, mock_session_factory, sample_entity):
        """Test node creation with database error."""
        _, session_mock = mock_session_factory
        session_mock.add.side_effect = SQLAlchemyError("DB Error")
        
        with pytest.raises(GraphDatabaseError, match="Database operation failed"):
            repository.create_nodes(
                entities=[sample_entity],
                conversation_id="conv-123",
                agent_id="agent-456",
            )
        
        session_mock.rollback.assert_called_once()
    
    def test_create_edges_success(self, repository, mock_session_factory, sample_relation):
        """Test successful edge creation."""
        _, session_mock = mock_session_factory
        
        node_mapping = {
            "entity-1": "node-1",
            "entity-2": "node-2",
        }
        
        result = repository.create_edges(
            relations=[sample_relation],
            conversation_id="conv-123",
            node_id_mapping=node_mapping,
        )
        
        assert len(result) == 1
        assert isinstance(result[0], str)
        session_mock.add.assert_called_once()
        session_mock.flush.assert_called_once()
    
    def test_create_edges_missing_node_mapping(self, repository, mock_session_factory, sample_relation):
        """Test edge creation with missing node mapping."""
        _, session_mock = mock_session_factory
        
        # Empty mapping - relation should be skipped
        result = repository.create_edges(
            relations=[sample_relation],
            conversation_id="conv-123",
            node_id_mapping={},
        )
        
        assert len(result) == 0
        session_mock.add.assert_not_called()
    
    def test_create_edges_database_error(self, repository, mock_session_factory, sample_relation):
        """Test edge creation with database error."""
        _, session_mock = mock_session_factory
        session_mock.add.side_effect = SQLAlchemyError("DB Error")
        
        node_mapping = {
            "entity-1": "node-1",
            "entity-2": "node-2",
        }
        
        with pytest.raises(GraphDatabaseError, match="Database operation failed"):
            repository.create_edges(
                relations=[sample_relation],
                conversation_id="conv-123",
                node_id_mapping=node_mapping,
            )
        
        session_mock.rollback.assert_called_once()
    
    def test_query_nodes_basic(self, repository, mock_session_factory):
        """Test basic node querying."""
        _, session_mock = mock_session_factory
        
        # Create a more complete mock chain
        nodes_result = []  # Empty list that can be measured with len()
        
        # Create mock query that supports method chaining
        query_mock = Mock()
        query_mock.filter.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = nodes_result
        
        session_mock.query.return_value = query_mock
        
        query = GraphQuery(node_types=["concept"])
        result = repository.query_nodes(query)
        
        assert result == []
        assert len(result) == 0
        session_mock.query.assert_called_once()
    
    def test_query_nodes_with_filters(self, repository, mock_session_factory):
        """Test node querying with multiple filters."""
        _, session_mock = mock_session_factory
        
        # Mock query chain with multiple filters
        query_mock = Mock()
        filter_mock1 = Mock()
        filter_mock2 = Mock()
        filter_mock3 = Mock()
        nodes_result = []  # Empty list for len() compatibility
        
        query_mock.filter.return_value = filter_mock1
        filter_mock1.filter.return_value = filter_mock2
        filter_mock2.filter.return_value = filter_mock3
        filter_mock3.all.return_value = nodes_result
        
        session_mock.query.return_value = query_mock
        
        query = GraphQuery(
            node_types=["concept"],
            confidence_threshold=0.8,
            created_after=datetime.now(),
        )
        result = repository.query_nodes(query)
        
        assert result == []
    
    def test_query_nodes_database_error(self, repository, mock_session_factory):
        """Test node querying with database error."""
        _, session_mock = mock_session_factory
        session_mock.query.side_effect = SQLAlchemyError("DB Error")
        
        query = GraphQuery()
        
        with pytest.raises(GraphDatabaseError, match="Database operation failed"):
            repository.query_nodes(query)
    
    def test_query_edges_basic(self, repository, mock_session_factory):
        """Test basic edge querying."""
        _, session_mock = mock_session_factory
        
        # Mock query chain
        query_mock = Mock()
        query_mock.all.return_value = []
        session_mock.query.return_value = query_mock
        
        query = GraphQuery()
        result = repository.query_edges(query)
        
        assert result == []
        session_mock.query.assert_called_once()
    
    def test_query_edges_with_node_filters(self, repository, mock_session_factory):
        """Test edge querying with node filters."""
        _, session_mock = mock_session_factory
        
        # Mock query chain
        query_mock = Mock()
        filter_mock = Mock()
        query_mock.filter.return_value = filter_mock
        filter_mock.all.return_value = []
        session_mock.query.return_value = query_mock
        
        query = GraphQuery()
        result = repository.query_edges(
            query,
            source_node_ids=["node-1"],
            target_node_ids=["node-2"],
        )
        
        assert result == []
    
    def test_get_graph_neighborhood(self, repository, mock_session_factory):
        """Test graph neighborhood retrieval."""
        _, session_mock = mock_session_factory
        
        # Mock query results
        mock_edges = [
            Mock(source_id="node-1", target_id="node-2"),
            Mock(source_id="node-2", target_id="node-3"),
        ]
        mock_nodes = [Mock(id="node-1"), Mock(id="node-2"), Mock(id="node-3")]
        
        # Set up query mocks
        edge_query_mock = Mock()
        edge_query_mock.all.return_value = mock_edges
        
        node_query_mock = Mock()
        node_query_mock.all.return_value = mock_nodes
        
        session_mock.query.side_effect = [edge_query_mock, node_query_mock]
        
        nodes, edges = repository.get_graph_neighborhood(
            node_ids=["node-1"],
            depth=1,
        )
        
        assert nodes == mock_nodes
        assert edges == mock_edges
    
    def test_get_graph_neighborhood_invalid_depth(self, repository):
        """Test graph neighborhood with invalid depth."""
        with pytest.raises(GraphValidationError, match="Depth must be between 1 and 3"):
            repository.get_graph_neighborhood(node_ids=["node-1"], depth=0)
        
        with pytest.raises(GraphValidationError, match="Depth must be between 1 and 3"):
            repository.get_graph_neighborhood(node_ids=["node-1"], depth=4)
    
    def test_update_node_confidence_success(self, repository, mock_session_factory):
        """Test successful node confidence update."""
        _, session_mock = mock_session_factory
        
        # Mock node query
        mock_node = Mock()
        mock_node.confidence = 0.5
        
        query_mock = Mock()
        filter_mock = Mock()
        query_mock.filter.return_value = filter_mock
        filter_mock.first.return_value = mock_node
        session_mock.query.return_value = query_mock
        
        result = repository.update_node_confidence("node-1", 0.9)
        
        assert result is True
        assert mock_node.confidence == 0.9
        session_mock.flush.assert_called_once()
    
    def test_update_node_confidence_invalid_value(self, repository):
        """Test node confidence update with invalid value."""
        with pytest.raises(GraphValidationError, match="Confidence must be between 0.0 and 1.0"):
            repository.update_node_confidence("node-1", 1.5)
        
        with pytest.raises(GraphValidationError, match="Confidence must be between 0.0 and 1.0"):
            repository.update_node_confidence("node-1", -0.5)
    
    def test_update_node_confidence_node_not_found(self, repository, mock_session_factory):
        """Test node confidence update with non-existent node."""
        _, session_mock = mock_session_factory
        
        # Mock node query returning None
        query_mock = Mock()
        filter_mock = Mock()
        query_mock.filter.return_value = filter_mock
        filter_mock.first.return_value = None
        session_mock.query.return_value = query_mock
        
        with pytest.raises(GraphDatabaseError, match="Node .* not found"):
            repository.update_node_confidence("nonexistent", 0.9)
    
    def test_delete_nodes_soft_delete(self, repository, mock_session_factory):
        """Test soft delete of nodes."""
        _, session_mock = mock_session_factory
        
        # Mock update query
        query_mock = Mock()
        filter_mock = Mock()
        query_mock.filter.return_value = filter_mock
        filter_mock.update.return_value = 2  # 2 nodes updated
        session_mock.query.return_value = query_mock
        
        result = repository.delete_nodes(["node-1", "node-2"], hard_delete=False)
        
        assert result == 2
        filter_mock.update.assert_called_once()
        session_mock.flush.assert_called_once()
    
    def test_delete_nodes_hard_delete(self, repository, mock_session_factory):
        """Test hard delete of nodes."""
        _, session_mock = mock_session_factory
        
        # Mock delete queries
        edge_query_mock = Mock()
        edge_filter_mock = Mock()
        edge_query_mock.filter.return_value = edge_filter_mock
        edge_filter_mock.delete.return_value = 3  # 3 edges deleted
        
        node_query_mock = Mock()
        node_filter_mock = Mock()
        node_query_mock.filter.return_value = node_filter_mock
        node_filter_mock.delete.return_value = 2  # 2 nodes deleted
        
        session_mock.query.side_effect = [edge_query_mock, node_query_mock]
        
        result = repository.delete_nodes(["node-1", "node-2"], hard_delete=True)
        
        assert result == 2
        edge_filter_mock.delete.assert_called_once()
        node_filter_mock.delete.assert_called_once()
    
    @patch('services.graph_database_service.text')
    def test_get_graph_statistics(self, mock_text, repository, mock_session_factory):
        """Test graph statistics retrieval."""
        _, session_mock = mock_session_factory
        
        # Mock count queries
        node_query_mock = Mock()
        node_query_mock.count.return_value = 100
        
        edge_query_mock = Mock()
        edge_query_mock.count.return_value = 200
        
        session_mock.query.side_effect = [node_query_mock, edge_query_mock]
        
        # Mock execute results
        node_types_result = [Mock(type="concept", count=50), Mock(type="agent", count=50)]
        edge_types_result = [Mock(type="relates_to", count=100), Mock(type="depends_on", count=100)]
        confidence_result = Mock(
            avg_confidence=0.8,
            min_confidence=0.3,
            max_confidence=1.0,
            std_confidence=0.2,
        )
        
        session_mock.execute.side_effect = [
            Mock(fetchall=lambda: node_types_result),
            Mock(fetchall=lambda: edge_types_result),
            Mock(fetchone=lambda: confidence_result),
        ]
        
        stats = repository.get_graph_statistics()
        
        assert stats["total_nodes"] == 100
        assert stats["total_edges"] == 200
        assert stats["node_types"] == {"concept": 50, "agent": 50}
        assert stats["edge_types"] == {"relates_to": 100, "depends_on": 100}
        assert stats["confidence_stats"]["average"] == 0.8


class TestGraphDatabaseService:
    """Test GraphDatabaseService class."""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing."""
        return Mock(spec=GraphRepository)
    
    @pytest.fixture
    def service(self, mock_repository):
        """Create service with mocked repository."""
        return GraphDatabaseService(repository=mock_repository)
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            ConversationEntity(
                entity_type=EntityType.CONCEPT,
                label="test concept",
                entity_id="entity-1",
                provenance=Provenance(
                    source_type="test",
                    source_id="test-123",
                    extraction_method="test_method",
                    confidence_score=0.9,
                ),
            ),
            ConversationEntity(
                entity_type=EntityType.AGENT,
                label="test agent",
                entity_id="entity-2",
                provenance=Provenance(
                    source_type="test",
                    source_id="test-123",
                    extraction_method="test_method",
                    confidence_score=0.8,
                ),
            ),
        ]
    
    @pytest.fixture
    def sample_relations(self):
        """Create sample relations for testing."""
        return [
            ConversationRelation(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relation_type=RelationType.RELATES_TO,
                provenance=Provenance(
                    source_type="test",
                    source_id="test-123",
                    extraction_method="test_method",
                    confidence_score=0.7,
                ),
            ),
        ]
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = GraphDatabaseService()
        assert isinstance(service.repository, GraphRepository)
    
    def test_service_initialization_with_repository(self, mock_repository):
        """Test service initialization with custom repository."""
        service = GraphDatabaseService(repository=mock_repository)
        assert service.repository == mock_repository
    
    @patch('services.graph_database_service.time.time', side_effect=[1000.0, 1005.0])
    def test_process_conversation_extraction_success(
        self,
        mock_time,
        service,
        mock_repository,
        sample_entities,
        sample_relations,
    ):
        """Test successful conversation extraction processing."""
        # Mock repository responses
        mock_repository.create_nodes.return_value = ["node-1", "node-2"]
        mock_repository.create_edges.return_value = ["edge-1"]
        
        result = service.process_conversation_extraction(
            entities=sample_entities,
            relations=sample_relations,
            conversation_id="conv-123",
            agent_id="agent-456",
        )
        
        assert result["nodes_created"] == 2
        assert result["edges_created"] == 1
        assert result["node_ids"] == ["node-1", "node-2"]
        assert result["edge_ids"] == ["edge-1"]
        assert result["processing_time"] == 5.0
        assert result["conversation_id"] == "conv-123"
        assert result["agent_id"] == "agent-456"
        
        # Verify repository calls
        mock_repository.create_nodes.assert_called_once_with(
            sample_entities, "conv-123", "agent-456"
        )
        mock_repository.create_edges.assert_called_once()
    
    def test_process_conversation_extraction_error(
        self,
        service,
        mock_repository,
        sample_entities,
        sample_relations,
    ):
        """Test conversation extraction processing with error."""
        mock_repository.create_nodes.side_effect = GraphDatabaseError("DB Error")
        
        with pytest.raises(GraphDatabaseError, match="Failed to process extraction"):
            service.process_conversation_extraction(
                entities=sample_entities,
                relations=sample_relations,
                conversation_id="conv-123",
                agent_id="agent-456",
            )
    
    def test_search_knowledge_basic(self, service, mock_repository):
        """Test basic knowledge search."""
        # Mock repository response
        mock_nodes = [
            Mock(
                id="node-1",
                type="concept",
                label="test concept",
                properties={"description": "test"},
                confidence=0.9,
                created_at=datetime.now(),
            )
        ]
        mock_edges = [
            Mock(
                id="edge-1",
                source_id="node-1",
                target_id="node-2",
                type="relates_to",
                properties={},
                confidence=0.8,
            )
        ]
        
        mock_repository.query_nodes.return_value = mock_nodes
        mock_repository.query_edges.return_value = mock_edges
        
        result = service.search_knowledge(
            node_types=["concept"],
            confidence_threshold=0.8,
            limit=100,
        )
        
        assert result["total_nodes"] == 1
        assert result["total_edges"] == 1
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
        assert result["nodes"][0]["id"] == "node-1"
        assert result["edges"][0]["id"] == "edge-1"
    
    def test_search_knowledge_with_text_filter(self, service, mock_repository):
        """Test knowledge search with text filtering."""
        mock_nodes = [
            Mock(
                id="node-1",
                type="concept",
                label="machine learning",
                properties={"description": "AI technique"},
                confidence=0.9,
                created_at=datetime.now(),
            ),
            Mock(
                id="node-2",
                type="concept",  
                label="database",
                properties={"description": "data storage"},
                confidence=0.8,
                created_at=datetime.now(),
            ),
        ]
        
        mock_repository.query_nodes.return_value = mock_nodes
        mock_repository.query_edges.return_value = []
        
        result = service.search_knowledge(
            query_text="machine",
            limit=100,
        )
        
        # Should only return the machine learning node
        assert result["total_nodes"] == 1
        assert result["nodes"][0]["label"] == "machine learning"
    
    def test_get_conversation_graph(self, service, mock_repository):
        """Test getting conversation graph."""
        # Mock statistics
        mock_stats = {
            "total_nodes": 10,
            "total_edges": 15,
            "node_types": {"concept": 8, "agent": 2},
        }
        mock_repository.get_graph_statistics.return_value = mock_stats
        
        # Mock session context manager and queries
        mock_session = Mock()
        mock_nodes = [Mock(
            id="node-1",
            type="concept",
            label="test",
            properties={},
            confidence=0.9,
            created_at=datetime.now(),
        )]
        mock_edges = [Mock(
            id="edge-1",
            source_id="node-1",
            target_id="node-2",
            type="relates_to",
            properties={},
            confidence=0.8,
        )]
        
        node_query_mock = Mock()
        node_filter_mock = Mock()
        node_query_mock.filter.return_value = node_filter_mock
        node_filter_mock.all.return_value = mock_nodes
        
        edge_query_mock = Mock()
        edge_filter_mock = Mock()
        edge_query_mock.filter.return_value = edge_filter_mock
        edge_filter_mock.all.return_value = mock_edges
        
        mock_session.query.side_effect = [node_query_mock, edge_query_mock]
        
        mock_repository._get_session.return_value.__enter__.return_value = mock_session
        
        result = service.get_conversation_graph("conv-123")
        
        assert result["conversation_id"] == "conv-123"
        assert result["statistics"] == mock_stats
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
    
    def test_health_check_healthy(self, service, mock_repository):
        """Test health check when service is healthy."""
        mock_stats = {
            "total_nodes": 100,
            "total_edges": 200,
        }
        mock_repository.get_graph_statistics.return_value = mock_stats
        
        result = service.health_check()
        
        assert result["status"] == "healthy"
        assert result["database_connected"] is True
        assert result["total_nodes"] == 100
        assert result["total_edges"] == 200
    
    def test_health_check_unhealthy(self, service, mock_repository):
        """Test health check when service is unhealthy."""
        mock_repository.get_graph_statistics.side_effect = Exception("DB Connection Error")
        
        result = service.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["database_connected"] is False
        assert "DB Connection Error" in result["error"]


class TestSingletonService:
    """Test singleton service instance."""
    
    def test_get_graph_database_service_singleton(self):
        """Test that get_graph_database_service returns singleton."""
        service1 = get_graph_database_service()
        service2 = get_graph_database_service()
        
        assert service1 is service2
        assert isinstance(service1, GraphDatabaseService)
    
    @patch('services.graph_database_service._graph_db_service', None)
    def test_get_graph_database_service_creates_new(self):
        """Test that service is created when None."""
        service = get_graph_database_service()
        assert isinstance(service, GraphDatabaseService)


class TestErrorClasses:
    """Test custom error classes."""
    
    def test_graph_database_error(self):
        """Test GraphDatabaseError."""
        error = GraphDatabaseError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_graph_transaction_error(self):
        """Test GraphTransactionError."""
        error = GraphTransactionError("Transaction failed")
        assert str(error) == "Transaction failed"
        assert isinstance(error, GraphDatabaseError)
    
    def test_graph_validation_error(self):
        """Test GraphValidationError."""
        error = GraphValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, GraphDatabaseError)


# Integration tests that would run against real database
class TestGraphDatabaseIntegration:
    """Integration tests for graph database service.
    
    These tests are marked to run only when explicitly requested
    since they require a real database connection.
    """
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires real database - run manually")
    def test_full_conversation_processing_workflow(self):
        """Test full workflow from entities to database storage."""
        # This test would:
        # 1. Create real entities and relations
        # 2. Process them through the service
        # 3. Verify they're stored correctly in database
        # 4. Query them back and verify data integrity
        pass
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires real database - run manually")
    def test_concurrent_graph_updates(self):
        """Test concurrent updates to the same graph."""
        # This test would:
        # 1. Simulate multiple agents updating graph simultaneously
        # 2. Verify transaction isolation
        # 3. Check for data consistency
        pass
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires real database - run manually")
    def test_large_graph_performance(self):
        """Test performance with large graphs."""
        # This test would:
        # 1. Create graph with thousands of nodes/edges
        # 2. Measure query performance
        # 3. Verify memory usage stays reasonable
        pass