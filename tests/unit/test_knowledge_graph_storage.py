"""Comprehensive tests for knowledge_graph/storage.py module."""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock database dependencies
with patch.dict(
    "sys.modules",
    {
        "database.base": MagicMock(),
        "database.session": MagicMock(),
    },
):
    from knowledge_graph.graph_engine import (
        EdgeType,
        KnowledgeEdge,
        KnowledgeGraph,
        KnowledgeNode,
        NodeType,
    )
    from knowledge_graph.storage import (
        DatabaseStorageBackend,
        EdgeModel,
        FileStorageBackend,
        GraphMetadataModel,
        NodeModel,
        PickleStorageBackend,
        StorageBackend,
        StorageManager,
    )


class TestNodeModel:
    """Test NodeModel SQLAlchemy model."""

    def test_node_model_attributes(self):
        """Test NodeModel has correct attributes."""
        node = NodeModel()

        # Test table name
        assert NodeModel.__tablename__ == "kg_nodes"

        # Test that attributes exist
        assert hasattr(node, "id")
        assert hasattr(node, "graph_id")
        assert hasattr(node, "type")
        assert hasattr(node, "label")
        assert hasattr(node, "properties")
        assert hasattr(node, "created_at")
        assert hasattr(node, "updated_at")
        assert hasattr(node, "version")

    def test_node_model_defaults(self):
        """Test NodeModel default values."""
        node = NodeModel()

        # Default version should be 1
        assert hasattr(node, "version")
        # created_at and updated_at should have default functions
        assert hasattr(node, "created_at")
        assert hasattr(node, "updated_at")

    def test_node_model_initialization(self):
        """Test NodeModel initialization with values."""
        node = NodeModel(
            id="test-node-id",
            graph_id="test-graph-id",
            type="concept",
            label="Test Node",
            properties={"key": "value"},
            version=2,
        )

        assert node.id == "test-node-id"
        assert node.graph_id == "test-graph-id"
        assert node.type == "concept"
        assert node.label == "Test Node"
        assert node.properties == {"key": "value"}
        assert node.version == 2


class TestEdgeModel:
    """Test EdgeModel SQLAlchemy model."""

    def test_edge_model_attributes(self):
        """Test EdgeModel has correct attributes."""
        edge = EdgeModel()

        # Test table name
        assert EdgeModel.__tablename__ == "kg_edges"

        # Test that attributes exist
        assert hasattr(edge, "id")
        assert hasattr(edge, "graph_id")
        assert hasattr(edge, "source_id")
        assert hasattr(edge, "target_id")
        assert hasattr(edge, "type")
        assert hasattr(edge, "properties")
        assert hasattr(edge, "weight")
        assert hasattr(edge, "created_at")
        assert hasattr(edge, "updated_at")
        assert hasattr(edge, "version")

    def test_edge_model_defaults(self):
        """Test EdgeModel default values."""
        edge = EdgeModel()

        # Default version should be 1
        assert hasattr(edge, "version")
        # Default weight should be 1.0
        assert hasattr(edge, "weight")

    def test_edge_model_initialization(self):
        """Test EdgeModel initialization with values."""
        edge = EdgeModel(
            id="test-edge-id",
            graph_id="test-graph-id",
            source_id="source-node-id",
            target_id="target-node-id",
            type="relates_to",
            properties={"strength": 0.8},
            weight=0.9,
            version=3,
        )

        assert edge.id == "test-edge-id"
        assert edge.graph_id == "test-graph-id"
        assert edge.source_id == "source-node-id"
        assert edge.target_id == "target-node-id"
        assert edge.type == "relates_to"
        assert edge.properties == {"strength": 0.8}
        assert edge.weight == 0.9
        assert edge.version == 3


class TestStorageBackend:
    """Test abstract StorageBackend class."""

    def test_abstract_class(self):
        """Test that StorageBackend is abstract."""
        from abc import ABC

        assert issubclass(StorageBackend, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            StorageBackend()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        required_methods = [
            "save_graph",
            "load_graph",
            "delete_graph",
            "list_graphs",
            "save_node",
            "load_node",
            "delete_node",
            "save_edge",
            "load_edge",
            "delete_edge",
            "query_nodes",
            "query_edges",
        ]

        for method in required_methods:
            if hasattr(StorageBackend, method):
                method_attr = getattr(StorageBackend, method)
                assert hasattr(method_attr, "__isabstractmethod__") or callable(method_attr)


class TestFileStorageBackend:
    """Test FileStorageBackend implementation."""

    def test_file_storage_initialization(self):
        """Test FileStorageBackend initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)

            assert storage.base_path == Path(temp_dir)
            assert storage.base_path.exists()

    def test_file_storage_creates_directory(self):
        """Test FileStorage creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "nonexistent"
            storage = FileStorage(str(storage_path))

            assert storage.base_path.exists()

    def test_get_graph_path(self):
        """Test _get_graph_path method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            graph_path = storage._get_graph_path("test-graph")
            expected_path = Path(temp_dir) / "test-graph.kg"

            assert graph_path == expected_path

    def test_save_and_load_graph(self):
        """Test saving and loading a complete graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create a test graph
            graph = KnowledgeGraph("test-graph")
            node1 = KnowledgeNode("node1", NodeType.CONCEPT, "Node 1", {"prop": "value1"})
            node2 = KnowledgeNode("node2", NodeType.CONCEPT, "Node 2", {"prop": "value2"})
            edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO, 0.8)

            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(edge)

            # Save graph
            success = storage.save_graph(graph)
            assert success is True

            # Load graph
            loaded_graph = storage.load_graph("test-graph")
            assert loaded_graph is not None
            assert loaded_graph.graph_id == "test-graph"
            assert len(loaded_graph.nodes) == 2
            assert len(loaded_graph.edges) == 1

    def test_save_graph_error_handling(self):
        """Test save_graph error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create a graph that will cause serialization error
            graph = KnowledgeGraph("test-graph")

            # Mock pickle.dump to raise exception
            with patch("pickle.dump", side_effect=Exception("Serialization error")):
                success = storage.save_graph(graph)
                assert success is False

    def test_load_graph_file_not_found(self):
        """Test load_graph when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            graph = storage.load_graph("nonexistent-graph")
            assert graph is None

    def test_load_graph_deserialization_error(self):
        """Test load_graph with deserialization error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create a corrupted file
            graph_path = storage._get_graph_path("corrupted-graph")
            graph_path.write_text("corrupted data")

            graph = storage.load_graph("corrupted-graph")
            assert graph is None

    def test_delete_graph(self):
        """Test deleting a graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create and save a graph
            graph = KnowledgeGraph("test-graph")
            storage.save_graph(graph)

            # Verify file exists
            graph_path = storage._get_graph_path("test-graph")
            assert graph_path.exists()

            # Delete graph
            success = storage.delete_graph("test-graph")
            assert success is True
            assert not graph_path.exists()

    def test_delete_nonexistent_graph(self):
        """Test deleting a graph that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            success = storage.delete_graph("nonexistent-graph")
            assert success is False

    def test_list_graphs(self):
        """Test listing graphs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create some graphs
            graph1 = KnowledgeGraph("graph1")
            graph2 = KnowledgeGraph("graph2")
            storage.save_graph(graph1)
            storage.save_graph(graph2)

            # Create a non-graph file
            (Path(temp_dir) / "not-a-graph.txt").write_text("test")

            graphs = storage.list_graphs()
            assert len(graphs) == 2
            assert "graph1" in graphs
            assert "graph2" in graphs

    def test_save_node(self):
        """Test saving a node."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node", {"prop": "value"})

            success = storage.save_node("test-graph", node)
            assert success is True

            # Verify node file exists
            node_path = Path(temp_dir) / "test-graph_nodes" / "node1.node"
            assert node_path.exists()

    def test_save_node_creates_directory(self):
        """Test save_node creates nodes directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node")

            success = storage.save_node("test-graph", node)
            assert success is True

            nodes_dir = Path(temp_dir) / "test-graph_nodes"
            assert nodes_dir.exists()

    def test_save_node_error_handling(self):
        """Test save_node error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node")

            # Mock pickle.dump to raise exception
            with patch("pickle.dump", side_effect=Exception("Serialization error")):
                success = storage.save_node("test-graph", node)
                assert success is False

    def test_load_node(self):
        """Test loading a node."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save a node first
            original_node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node", {"prop": "value"})
            storage.save_node("test-graph", original_node)

            # Load the node
            loaded_node = storage.load_node("test-graph", "node1")
            assert loaded_node is not None
            assert loaded_node.node_id == "node1"
            assert loaded_node.node_type == NodeType.CONCEPT
            assert loaded_node.label == "Test Node"
            assert loaded_node.properties == {"prop": "value"}

    def test_load_node_file_not_found(self):
        """Test load_node when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            node = storage.load_node("test-graph", "nonexistent-node")
            assert node is None

    def test_load_node_deserialization_error(self):
        """Test load_node with deserialization error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Create nodes directory and corrupted file
            nodes_dir = Path(temp_dir) / "test-graph_nodes"
            nodes_dir.mkdir()
            (nodes_dir / "corrupted.node").write_text("corrupted data")

            node = storage.load_node("test-graph", "corrupted")
            assert node is None

    def test_delete_node(self):
        """Test deleting a node."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save a node first
            node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node")
            storage.save_node("test-graph", node)

            # Verify file exists
            node_path = Path(temp_dir) / "test-graph_nodes" / "node1.node"
            assert node_path.exists()

            # Delete node
            success = storage.delete_node("test-graph", "node1")
            assert success is True
            assert not node_path.exists()

    def test_delete_nonexistent_node(self):
        """Test deleting a node that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            success = storage.delete_node("test-graph", "nonexistent-node")
            assert success is False

    def test_save_edge(self):
        """Test saving an edge."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO, 0.8)

            success = storage.save_edge("test-graph", edge)
            assert success is True

            # Verify edge file exists
            edge_path = Path(temp_dir) / "test-graph_edges" / "edge1.edge"
            assert edge_path.exists()

    def test_save_edge_creates_directory(self):
        """Test save_edge creates edges directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO)

            success = storage.save_edge("test-graph", edge)
            assert success is True

            edges_dir = Path(temp_dir) / "test-graph_edges"
            assert edges_dir.exists()

    def test_load_edge(self):
        """Test loading an edge."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save an edge first
            original_edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO, 0.8)
            storage.save_edge("test-graph", original_edge)

            # Load the edge
            loaded_edge = storage.load_edge("test-graph", "edge1")
            assert loaded_edge is not None
            assert loaded_edge.edge_id == "edge1"
            assert loaded_edge.source_id == "node1"
            assert loaded_edge.target_id == "node2"
            assert loaded_edge.edge_type == EdgeType.RELATES_TO
            assert loaded_edge.weight == 0.8

    def test_load_edge_file_not_found(self):
        """Test load_edge when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            edge = storage.load_edge("test-graph", "nonexistent-edge")
            assert edge is None

    def test_delete_edge(self):
        """Test deleting an edge."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save an edge first
            edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO)
            storage.save_edge("test-graph", edge)

            # Verify file exists
            edge_path = Path(temp_dir) / "test-graph_edges" / "edge1.edge"
            assert edge_path.exists()

            # Delete edge
            success = storage.delete_edge("test-graph", "edge1")
            assert success is True
            assert not edge_path.exists()

    def test_delete_nonexistent_edge(self):
        """Test deleting an edge that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            success = storage.delete_edge("test-graph", "nonexistent-edge")
            assert success is False

    def test_query_nodes(self):
        """Test querying nodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save some nodes
            node1 = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node 1", {"category": "A"})
            node2 = KnowledgeNode("node2", NodeType.CONCEPT, "Test Node 2", {"category": "B"})
            node3 = KnowledgeNode("node3", NodeType.ENTITY, "Test Entity", {"category": "A"})

            storage.save_node("test-graph", node1)
            storage.save_node("test-graph", node2)
            storage.save_node("test-graph", node3)

            # Query by type
            concept_nodes = storage.query_nodes("test-graph", {"type": NodeType.CONCEPT})
            assert len(concept_nodes) == 2

            # Query by property
            category_a_nodes = storage.query_nodes("test-graph", {"properties.category": "A"})
            assert len(category_a_nodes) == 2

    def test_query_nodes_empty_result(self):
        """Test querying nodes with no matches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            nodes = storage.query_nodes("test-graph", {"type": NodeType.CONCEPT})
            assert len(nodes) == 0

    def test_query_nodes_nonexistent_graph(self):
        """Test querying nodes from nonexistent graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            nodes = storage.query_nodes("nonexistent-graph", {"type": NodeType.CONCEPT})
            assert len(nodes) == 0

    def test_query_edges(self):
        """Test querying edges."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            # Save some edges
            edge1 = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO, 0.8)
            edge2 = KnowledgeEdge("edge2", "node2", "node3", EdgeType.CAUSES, 0.6)
            edge3 = KnowledgeEdge("edge3", "node1", "node3", EdgeType.RELATES_TO, 0.9)

            storage.save_edge("test-graph", edge1)
            storage.save_edge("test-graph", edge2)
            storage.save_edge("test-graph", edge3)

            # Query by type
            relates_edges = storage.query_edges("test-graph", {"type": EdgeType.RELATES_TO})
            assert len(relates_edges) == 2

            # Query by source
            from_node1_edges = storage.query_edges("test-graph", {"source_id": "node1"})
            assert len(from_node1_edges) == 2

    def test_query_edges_empty_result(self):
        """Test querying edges with no matches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            edges = storage.query_edges("test-graph", {"type": EdgeType.RELATES_TO})
            assert len(edges) == 0

    def test_query_edges_nonexistent_graph(self):
        """Test querying edges from nonexistent graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)

            edges = storage.query_edges("nonexistent-graph", {"type": EdgeType.RELATES_TO})
            assert len(edges) == 0


class TestSQLStorage:
    """Test SQLStorage implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return MagicMock()

    def test_sql_storage_initialization(self, mock_session):
        """Test SQLStorage initialization."""
        storage = SQLStorage(mock_session)
        assert storage.session == mock_session

    def test_save_graph(self, mock_session):
        """Test saving a graph to database."""
        storage = SQLStorage(mock_session)

        # Create a test graph
        graph = KnowledgeGraph("test-graph")
        node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node")
        edge = KnowledgeEdge("edge1", "node1", "node2", EdgeType.RELATES_TO)

        graph.add_node(node)
        graph.add_edge(edge)

        # Mock session methods
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_session.commit.return_value = None

        success = storage.save_graph(graph)
        assert success is True

        # Verify session methods were called
        assert mock_session.add.called
        assert mock_session.commit.called

    def test_save_graph_error_handling(self, mock_session):
        """Test save_graph error handling."""
        storage = SQLStorage(mock_session)

        graph = KnowledgeGraph("test-graph")

        # Mock session to raise exception
        mock_session.commit.side_effect = Exception("Database error")

        success = storage.save_graph(graph)
        assert success is False

        # Verify rollback was called
        mock_session.rollback.assert_called_once()

    def test_load_graph(self, mock_session):
        """Test loading a graph from database."""
        storage = SQLStorage(mock_session)

        # Mock database returns
        mock_node = MagicMock()
        mock_node.id = "node1"
        mock_node.type = "concept"
        mock_node.label = "Test Node"
        mock_node.properties = {"key": "value"}

        mock_edge = MagicMock()
        mock_edge.id = "edge1"
        mock_edge.source_id = "node1"
        mock_edge.target_id = "node2"
        mock_edge.type = "relates_to"
        mock_edge.weight = 0.8
        mock_edge.properties = {}

        mock_session.query.return_value.filter.return_value.all.side_effect = [
            [mock_node],
            [mock_edge],
        ]

        graph = storage.load_graph("test-graph")

        assert graph is not None
        assert graph.graph_id == "test-graph"
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1

    def test_load_graph_error_handling(self, mock_session):
        """Test load_graph error handling."""
        storage = SQLStorage(mock_session)

        # Mock session to raise exception
        mock_session.query.side_effect = Exception("Database error")

        graph = storage.load_graph("test-graph")
        assert graph is None

    def test_delete_graph(self, mock_session):
        """Test deleting a graph from database."""
        storage = SQLStorage(mock_session)

        # Mock query returns
        mock_session.query.return_value.filter.return_value.delete.return_value = 1

        success = storage.delete_graph("test-graph")
        assert success is True

        # Verify commit was called
        mock_session.commit.assert_called_once()

    def test_delete_graph_error_handling(self, mock_session):
        """Test delete_graph error handling."""
        storage = SQLStorage(mock_session)

        # Mock session to raise exception
        mock_session.query.side_effect = Exception("Database error")

        success = storage.delete_graph("test-graph")
        assert success is False

        # Verify rollback was called
        mock_session.rollback.assert_called_once()

    def test_list_graphs(self, mock_session):
        """Test listing graphs from database."""
        storage = SQLStorage(mock_session)

        # Mock query returns
        mock_session.query.return_value.distinct.return_value.all.return_value = [
            ("graph1",),
            ("graph2",),
        ]

        graphs = storage.list_graphs()
        assert len(graphs) == 2
        assert "graph1" in graphs
        assert "graph2" in graphs

    def test_list_graphs_error_handling(self, mock_session):
        """Test list_graphs error handling."""
        storage = SQLStorage(mock_session)

        # Mock session to raise exception
        mock_session.query.side_effect = Exception("Database error")

        graphs = storage.list_graphs()
        assert graphs == []

    def test_save_node(self, mock_session):
        """Test saving a node to database."""
        storage = SQLStorage(mock_session)

        node = KnowledgeNode("node1", NodeType.CONCEPT, "Test Node", {"key": "value"})

        # Mock session methods
        mock_session.query.return_value.filter.return_value.first.return_value = None

        success = storage.save_node("test-graph", node)
        assert success is True

        # Verify session methods were called
        assert mock_session.add.called
        assert mock_session.commit.called

    def test_save_node_update_existing(self, mock_session):
        """Test updating an existing node."""
        storage = SQLStorage(mock_session)

        node = KnowledgeNode("node1", NodeType.CONCEPT, "Updated Node", {"key": "new_value"})

        # Mock existing node
        mock_existing = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_existing

        success = storage.save_node("test-graph", node)
        assert success is True

        # Verify existing node was updated
        assert mock_existing.type == "concept"
        assert mock_existing.label == "Updated Node"
        assert mock_existing.properties == {"key": "new_value"}

    def test_load_node(self, mock_session):
        """Test loading a node from database."""
        storage = SQLStorage(mock_session)

        # Mock database return
        mock_node = MagicMock()
        mock_node.id = "node1"
        mock_node.type = "concept"
        mock_node.label = "Test Node"
        mock_node.properties = {"key": "value"}

        mock_session.query.return_value.filter.return_value.first.return_value = mock_node

        node = storage.load_node("test-graph", "node1")

        assert node is not None
        assert node.node_id == "node1"
        assert node.node_type == NodeType.CONCEPT
        assert node.label == "Test Node"
        assert node.properties == {"key": "value"}

    def test_load_node_not_found(self, mock_session):
        """Test loading a node that doesn't exist."""
        storage = SQLStorage(mock_session)

        mock_session.query.return_value.filter.return_value.first.return_value = None

        node = storage.load_node("test-graph", "nonexistent-node")
        assert node is None

    def test_delete_node(self, mock_session):
        """Test deleting a node from database."""
        storage = SQLStorage(mock_session)

        # Mock successful deletion
        mock_session.query.return_value.filter.return_value.delete.return_value = 1

        success = storage.delete_node("test-graph", "node1")
        assert success is True

        # Verify commit was called
        mock_session.commit.assert_called_once()

    def test_delete_node_not_found(self, mock_session):
        """Test deleting a node that doesn't exist."""
        storage = SQLStorage(mock_session)

        # Mock no deletion
        mock_session.query.return_value.filter.return_value.delete.return_value = 0

        success = storage.delete_node("test-graph", "nonexistent-node")
        assert success is False

    def test_query_nodes(self, mock_session):
        """Test querying nodes from database."""
        storage = SQLStorage(mock_session)

        # Mock database returns
        mock_node1 = MagicMock()
        mock_node1.id = "node1"
        mock_node1.type = "concept"
        mock_node1.label = "Node 1"
        mock_node1.properties = {"category": "A"}

        mock_node2 = MagicMock()
        mock_node2.id = "node2"
        mock_node2.type = "concept"
        mock_node2.label = "Node 2"
        mock_node2.properties = {"category": "A"}

        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_node1,
            mock_node2,
        ]

        nodes = storage.query_nodes("test-graph", {"type": NodeType.CONCEPT})

        assert len(nodes) == 2
        assert nodes[0].node_id == "node1"
        assert nodes[1].node_id == "node2"

    def test_query_edges(self, mock_session):
        """Test querying edges from database."""
        storage = SQLStorage(mock_session)

        # Mock database returns
        mock_edge1 = MagicMock()
        mock_edge1.id = "edge1"
        mock_edge1.source_id = "node1"
        mock_edge1.target_id = "node2"
        mock_edge1.type = "relates_to"
        mock_edge1.weight = 0.8
        mock_edge1.properties = {}

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_edge1]

        edges = storage.query_edges("test-graph", {"type": EdgeType.RELATES_TO})

        assert len(edges) == 1
        assert edges[0].edge_id == "edge1"
        assert edges[0].source_id == "node1"
        assert edges[0].target_id == "node2"
