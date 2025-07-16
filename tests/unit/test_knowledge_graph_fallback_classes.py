"""Tests for knowledge_graph.fallback_classes module."""

from unittest.mock import MagicMock, patch

import pytest


class TestStorageManager:
    """Test StorageManager fallback class."""

    def test_storage_manager_creation(self):
        """Test StorageManager creation."""
        from knowledge_graph.fallback_classes import StorageManager

        backend = MagicMock()
        storage = StorageManager(backend)

        assert storage.backend == backend

    def test_storage_manager_save(self):
        """Test StorageManager save method."""
        from knowledge_graph.fallback_classes import StorageManager

        backend = MagicMock()
        storage = StorageManager(backend)

        result = storage.save(MagicMock())
        assert result is True

    def test_storage_manager_delete(self):
        """Test StorageManager delete method."""
        from knowledge_graph.fallback_classes import StorageManager

        backend = MagicMock()
        storage = StorageManager(backend)

        result = storage.delete("test_id")
        assert result is True


class TestFileStorageBackend:
    """Test FileStorageBackend fallback class."""

    def test_file_storage_backend_creation(self):
        """Test FileStorageBackend creation."""
        from knowledge_graph.fallback_classes import FileStorageBackend

        backend = FileStorageBackend("/test/path")
        assert backend is not None


class TestEventDispatcher:
    """Test EventDispatcher fallback class."""

    @pytest.mark.asyncio
    async def test_event_dispatcher_creation(self):
        """Test EventDispatcher creation."""
        from knowledge_graph.fallback_classes import EventDispatcher

        dispatcher = EventDispatcher()
        assert dispatcher is not None

    @pytest.mark.asyncio
    async def test_event_dispatcher_dispatch_event(self):
        """Test EventDispatcher dispatch_event method."""
        from knowledge_graph.fallback_classes import EventDispatcher

        dispatcher = EventDispatcher()
        event = {"type": "test", "data": "test_data"}

        # Should not raise exception
        await dispatcher.dispatch_event(event)


class TestEventFunctions:
    """Test event-related functions."""

    def test_get_event_dispatcher(self):
        """Test get_event_dispatcher function."""
        from knowledge_graph.fallback_classes import EventDispatcher, get_event_dispatcher

        result = get_event_dispatcher()
        assert isinstance(result, EventDispatcher)

    def test_create_node_event(self):
        """Test create_node_event function."""
        from knowledge_graph.fallback_classes import create_node_event

        result = create_node_event(id="test", type="concept")
        assert result == {}


class TestConversationEventIntegration:
    """Test ConversationEventIntegration fallback class."""

    def test_conversation_event_integration_creation(self):
        """Test ConversationEventIntegration creation."""
        from knowledge_graph.fallback_classes import ConversationEventIntegration

        integration = ConversationEventIntegration()
        assert integration is not None


class TestKnowledgeGraph:
    """Test KnowledgeGraph fallback class."""

    def test_knowledge_graph_creation(self):
        """Test KnowledgeGraph creation."""
        from knowledge_graph.fallback_classes import KnowledgeGraph

        graph = KnowledgeGraph()
        assert graph.graph_id == "unknown"
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_knowledge_graph_add_node(self):
        """Test KnowledgeGraph add_node method."""
        from knowledge_graph.fallback_classes import KnowledgeGraph

        graph = KnowledgeGraph()
        result = graph.add_node(MagicMock())
        assert result is True

    def test_knowledge_graph_add_edge(self):
        """Test KnowledgeGraph add_edge method."""
        from knowledge_graph.fallback_classes import KnowledgeGraph

        graph = KnowledgeGraph()
        result = graph.add_edge(MagicMock())
        assert result is True

    def test_knowledge_graph_save(self):
        """Test KnowledgeGraph save method."""
        from knowledge_graph.fallback_classes import KnowledgeGraph

        graph = KnowledgeGraph()
        # Should not raise exception
        graph.save()


class TestKnowledgeNode:
    """Test KnowledgeNode fallback class."""

    def test_knowledge_node_creation_with_defaults(self):
        """Test KnowledgeNode creation with defaults."""
        from knowledge_graph.fallback_classes import KnowledgeNode

        node = KnowledgeNode()
        assert node.id == "unknown"
        assert node.type is None
        assert node.label == ""
        assert node.properties == {}
        assert node.confidence == 1.0
        assert node.source is None

    def test_knowledge_node_creation_with_kwargs(self):
        """Test KnowledgeNode creation with kwargs."""
        from knowledge_graph.fallback_classes import KnowledgeNode

        node = KnowledgeNode(
            id="test_id",
            type="concept",
            label="Test Node",
            properties={"key": "value"},
            confidence=0.8,
            source="test_source",
        )

        assert node.id == "test_id"
        assert node.type == "concept"
        assert node.label == "Test Node"
        assert node.properties == {"key": "value"}
        assert node.confidence == 0.8
        assert node.source == "test_source"

    def test_knowledge_node_partial_kwargs(self):
        """Test KnowledgeNode creation with partial kwargs."""
        from knowledge_graph.fallback_classes import KnowledgeNode

        node = KnowledgeNode(id="test_id", type="entity")
        assert node.id == "test_id"
        assert node.type == "entity"
        assert node.label == ""  # Default
        assert node.properties == {}  # Default
        assert node.confidence == 1.0  # Default
        assert node.source is None  # Default


class TestKnowledgeEdge:
    """Test KnowledgeEdge fallback class."""

    def test_knowledge_edge_creation_with_defaults(self):
        """Test KnowledgeEdge creation with defaults."""
        from knowledge_graph.fallback_classes import KnowledgeEdge

        edge = KnowledgeEdge()
        assert edge.id == "unknown"
        assert edge.source_id is None
        assert edge.target_id is None
        assert edge.type is None
        assert edge.properties == {}
        assert edge.confidence == 1.0

    def test_knowledge_edge_creation_with_kwargs(self):
        """Test KnowledgeEdge creation with kwargs."""
        from knowledge_graph.fallback_classes import KnowledgeEdge

        edge = KnowledgeEdge(
            id="edge_id",
            source_id="source_node",
            target_id="target_node",
            type="relates_to",
            properties={"weight": 0.5},
            confidence=0.9,
        )

        assert edge.id == "edge_id"
        assert edge.source_id == "source_node"
        assert edge.target_id == "target_node"
        assert edge.type == "relates_to"
        assert edge.properties == {"weight": 0.5}
        assert edge.confidence == 0.9


class TestQueryEngine:
    """Test QueryEngine fallback class."""

    def test_query_engine_creation(self):
        """Test QueryEngine creation."""
        from knowledge_graph.fallback_classes import QueryEngine

        engine = QueryEngine()
        assert engine is not None

    def test_query_engine_execute(self):
        """Test QueryEngine execute method."""
        from knowledge_graph.fallback_classes import QueryEngine

        engine = QueryEngine()
        result = engine.execute(MagicMock())
        assert result == {}


class TestGraphSearchEngine:
    """Test GraphSearchEngine fallback class."""

    def test_graph_search_engine_creation(self):
        """Test GraphSearchEngine creation."""
        from knowledge_graph.fallback_classes import GraphSearchEngine

        engine = GraphSearchEngine()
        assert engine is not None


class TestEvolutionEngine:
    """Test EvolutionEngine fallback class."""

    def test_evolution_engine_creation(self):
        """Test EvolutionEngine creation."""
        from knowledge_graph.fallback_classes import EvolutionEngine

        engine = EvolutionEngine()
        assert engine is not None


class TestGraphQuery:
    """Test GraphQuery fallback class."""

    def test_graph_query_creation_with_defaults(self):
        """Test GraphQuery creation with defaults."""
        from knowledge_graph.fallback_classes import GraphQuery

        query = GraphQuery()
        assert query.query_type is None
        assert query.node_ids is None
        assert query.node_types is None
        assert query.node_labels is None
        assert query.node_properties is None
        assert query.edge_types is None
        assert query.source_id is None
        assert query.target_id is None
        assert query.center_id is None
        assert query.radius == 1
        assert query.confidence_threshold == 0.0
        assert query.limit is None
        assert query.order_by is None
        assert query.descending is False

    def test_graph_query_creation_with_kwargs(self):
        """Test GraphQuery creation with kwargs."""
        from knowledge_graph.fallback_classes import GraphQuery

        query = GraphQuery(
            query_type="search",
            node_ids=["node1", "node2"],
            node_types=["concept"],
            node_labels=["test"],
            node_properties={"key": "value"},
            edge_types=["relates_to"],
            source_id="source",
            target_id="target",
            center_id="center",
            radius=2,
            confidence_threshold=0.5,
            limit=10,
            order_by="confidence",
            descending=True,
        )

        assert query.query_type == "search"
        assert query.node_ids == ["node1", "node2"]
        assert query.node_types == ["concept"]
        assert query.node_labels == ["test"]
        assert query.node_properties == {"key": "value"}
        assert query.edge_types == ["relates_to"]
        assert query.source_id == "source"
        assert query.target_id == "target"
        assert query.center_id == "center"
        assert query.radius == 2
        assert query.confidence_threshold == 0.5
        assert query.limit == 10
        assert query.order_by == "confidence"
        assert query.descending is True


class TestNodeType:
    """Test NodeType fallback class."""

    def test_node_type_creation(self):
        """Test NodeType creation."""
        from knowledge_graph.fallback_classes import NodeType

        node_type = NodeType("concept")
        assert node_type.name == "concept"
        assert node_type.value == "concept"


class TestEdgeType:
    """Test EdgeType fallback class."""

    def test_edge_type_creation(self):
        """Test EdgeType creation."""
        from knowledge_graph.fallback_classes import EdgeType

        edge_type = EdgeType("relates_to")
        assert edge_type.name == "relates_to"
        assert edge_type.value == "relates_to"


class TestEventConstants:
    """Test event constants."""

    def test_event_type_constants(self):
        """Test EventType constants."""
        from knowledge_graph.fallback_classes import EventType

        assert EventType.NODE_CREATED == "node_created"

    def test_event_source_constants(self):
        """Test EventSource constants."""
        from knowledge_graph.fallback_classes import EventSource

        assert EventSource.API_REQUEST == "api_request"


class TestCreateAllFallbackClasses:
    """Test create_all_fallback_classes function."""

    @patch("knowledge_graph.fallback_classes.logging.getLogger")
    def test_create_all_fallback_classes(self, mock_get_logger):
        """Test create_all_fallback_classes function."""
        from knowledge_graph.fallback_classes import create_all_fallback_classes

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result = create_all_fallback_classes()

        # Check that logger was created
        mock_get_logger.assert_called_once_with("knowledge_graph.fallback_classes")

        # Check that all expected classes are returned
        expected_keys = {
            "logger",
            "ConversationEventIntegration",
            "EventSource",
            "EventType",
            "create_node_event",
            "get_event_dispatcher",
            "EvolutionEngine",
            "KnowledgeEdge",
            "KnowledgeGraph",
            "KnowledgeNode",
            "EdgeType",
            "GraphQuery",
            "NodeType",
            "QueryEngine",
            "GraphSearchEngine",
            "FileStorageBackend",
            "StorageManager",
        }

        assert set(result.keys()) == expected_keys
        assert result["logger"] == mock_logger

    @patch("knowledge_graph.fallback_classes.logging.getLogger")
    def test_create_all_fallback_classes_types(self, mock_get_logger):
        """Test create_all_fallback_classes returns correct types."""
        from knowledge_graph.fallback_classes import (
            ConversationEventIntegration,
            EdgeType,
            EventSource,
            EventType,
            EvolutionEngine,
            FileStorageBackend,
            GraphQuery,
            GraphSearchEngine,
            KnowledgeEdge,
            KnowledgeGraph,
            KnowledgeNode,
            NodeType,
            QueryEngine,
            StorageManager,
            create_all_fallback_classes,
            create_node_event,
            get_event_dispatcher,
        )

        result = create_all_fallback_classes()

        # Test that correct classes are returned
        assert result["ConversationEventIntegration"] == ConversationEventIntegration
        assert result["EventSource"] == EventSource
        assert result["EventType"] == EventType
        assert result["create_node_event"] == create_node_event
        assert result["get_event_dispatcher"] == get_event_dispatcher
        assert result["EvolutionEngine"] == EvolutionEngine
        assert result["KnowledgeEdge"] == KnowledgeEdge
        assert result["KnowledgeGraph"] == KnowledgeGraph
        assert result["KnowledgeNode"] == KnowledgeNode
        assert result["EdgeType"] == EdgeType
        assert result["GraphQuery"] == GraphQuery
        assert result["NodeType"] == NodeType
        assert result["QueryEngine"] == QueryEngine
        assert result["GraphSearchEngine"] == GraphSearchEngine
        assert result["FileStorageBackend"] == FileStorageBackend
        assert result["StorageManager"] == StorageManager
