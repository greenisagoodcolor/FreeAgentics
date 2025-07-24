"""Simple tests for knowledge_graph/storage.py module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock dependencies
with patch.dict(
    "sys.modules",
    {
        "database.base": MagicMock(),
        "database.session": MagicMock(),
    },
):
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

    def test_node_model_table_name(self):
        """Test NodeModel table name."""
        assert NodeModel.__tablename__ == "kg_nodes"

    def test_node_model_creation(self):
        """Test NodeModel can be created."""
        node = NodeModel()
        assert node is not None
        assert hasattr(node, "id")
        assert hasattr(node, "graph_id")
        assert hasattr(node, "type")
        assert hasattr(node, "label")
        assert hasattr(node, "properties")


class TestEdgeModel:
    """Test EdgeModel SQLAlchemy model."""

    def test_edge_model_table_name(self):
        """Test EdgeModel table name."""
        assert EdgeModel.__tablename__ == "kg_edges"

    def test_edge_model_creation(self):
        """Test EdgeModel can be created."""
        edge = EdgeModel()
        assert edge is not None
        assert hasattr(edge, "id")
        assert hasattr(edge, "graph_id")
        assert hasattr(edge, "source_id")
        assert hasattr(edge, "target_id")


class TestGraphMetadataModel:
    """Test GraphMetadataModel SQLAlchemy model."""

    def test_graph_metadata_model_table_name(self):
        """Test GraphMetadataModel table name."""
        assert GraphMetadataModel.__tablename__ == "kg_graph_metadata"

    def test_graph_metadata_model_creation(self):
        """Test GraphMetadataModel can be created."""
        metadata = GraphMetadataModel()
        assert metadata is not None
        assert hasattr(metadata, "graph_id")
        assert hasattr(metadata, "version")
        assert hasattr(metadata, "created_at")


class TestStorageBackend:
    """Test abstract StorageBackend class."""

    def test_abstract_class(self):
        """Test that StorageBackend is abstract."""
        from abc import ABC

        assert issubclass(StorageBackend, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            StorageBackend()


class TestFileStorageBackend:
    """Test FileStorageBackend implementation."""

    def test_file_storage_initialization(self):
        """Test FileStorageBackend initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)

            assert storage.base_path == Path(temp_dir)
            assert storage.base_path.exists()

    def test_file_storage_creates_directory(self):
        """Test FileStorageBackend creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "nonexistent"
            storage = FileStorageBackend(str(storage_path))

            assert storage.base_path.exists()

    def test_file_storage_default_path(self):
        """Test FileStorageBackend with default path."""
        with patch("pathlib.Path.mkdir"):
            storage = FileStorageBackend()
            assert storage.base_path == Path("./knowledge_graphs")

    def test_save_graph_method_exists(self):
        """Test save_graph method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "save_graph")
            assert callable(storage.save_graph)

    def test_load_graph_method_exists(self):
        """Test load_graph method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "load_graph")
            assert callable(storage.load_graph)

    def test_delete_graph_method_exists(self):
        """Test delete_graph method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "delete_graph")
            assert callable(storage.delete_graph)

    def test_list_graphs_method_exists(self):
        """Test list_graphs method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "list_graphs")
            assert callable(storage.list_graphs)

    def test_graph_exists_method_exists(self):
        """Test graph_exists method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "graph_exists")
            assert callable(storage.graph_exists)

    def test_load_metadata_method_exists(self):
        """Test _load_metadata method exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert hasattr(storage, "_load_metadata")
            assert callable(storage._load_metadata)

    def test_load_metadata_empty(self):
        """Test _load_metadata with no metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            metadata = storage._load_metadata()
            assert metadata == {}

    def test_load_metadata_invalid_json(self):
        """Test _load_metadata with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)

            # Create invalid JSON file
            metadata_path = storage.base_path / "metadata.json"
            metadata_path.write_text("invalid json")

            metadata = storage._load_metadata()
            assert metadata == {}

    def test_graph_exists_false_for_nonexistent(self):
        """Test graph_exists returns False for nonexistent graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            assert not storage.graph_exists("nonexistent-graph")

    def test_list_graphs_empty(self):
        """Test list_graphs returns empty list when no graphs exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            graphs = storage.list_graphs()
            assert graphs == []

    def test_load_graph_nonexistent_returns_none(self):
        """Test load_graph returns None for nonexistent graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            graph = storage.load_graph("nonexistent-graph")
            assert graph is None

    def test_delete_graph_nonexistent_returns_false(self):
        """Test delete_graph returns False for nonexistent graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorageBackend(temp_dir)
            result = storage.delete_graph("nonexistent-graph")
            assert result is False


class TestDatabaseStorageBackend:
    """Test DatabaseStorageBackend implementation."""

    def test_database_storage_initialization(self):
        """Test DatabaseStorageBackend initialization."""
        mock_session = MagicMock()
        storage = DatabaseStorageBackend(mock_session)
        assert storage.session == mock_session

    def test_database_storage_methods_exist(self):
        """Test DatabaseStorageBackend has required methods."""
        mock_session = MagicMock()
        storage = DatabaseStorageBackend(mock_session)

        methods = [
            "save_graph",
            "load_graph",
            "delete_graph",
            "list_graphs",
            "graph_exists",
        ]
        for method in methods:
            assert hasattr(storage, method)
            assert callable(getattr(storage, method))

    def test_save_graph_error_handling(self):
        """Test save_graph error handling."""
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("Database error")

        storage = DatabaseStorageBackend(mock_session)

        # Mock graph
        mock_graph = MagicMock()
        mock_graph.graph_id = "test-graph"
        mock_graph.version = 1
        mock_graph.created_at = MagicMock()
        mock_graph.updated_at = MagicMock()
        mock_graph.nodes = {}
        mock_graph.edges = {}

        result = storage.save_graph(mock_graph)
        assert result is False
        mock_session.rollback.assert_called_once()

    def test_load_graph_error_handling(self):
        """Test load_graph error handling."""
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database error")

        storage = DatabaseStorageBackend(mock_session)
        result = storage.load_graph("test-graph")
        assert result is None

    def test_delete_graph_error_handling(self):
        """Test delete_graph error handling."""
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database error")

        storage = DatabaseStorageBackend(mock_session)
        result = storage.delete_graph("test-graph")
        assert result is False
        mock_session.rollback.assert_called_once()

    def test_list_graphs_error_handling(self):
        """Test list_graphs error handling."""
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database error")

        storage = DatabaseStorageBackend(mock_session)
        result = storage.list_graphs()
        assert result == []

    def test_graph_exists_error_handling(self):
        """Test graph_exists error handling."""
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database error")

        storage = DatabaseStorageBackend(mock_session)
        result = storage.graph_exists("test-graph")
        assert result is False


class TestPickleStorageBackend:
    """Test PickleStorageBackend implementation."""

    def test_pickle_storage_initialization(self):
        """Test PickleStorageBackend initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = PickleStorageBackend(temp_dir)
            assert storage.base_path == Path(temp_dir)

    def test_pickle_storage_methods_exist(self):
        """Test PickleStorageBackend has required methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = PickleStorageBackend(temp_dir)

            methods = [
                "save_graph",
                "load_graph",
                "delete_graph",
                "list_graphs",
                "graph_exists",
            ]
            for method in methods:
                assert hasattr(storage, method)
                assert callable(getattr(storage, method))

    def test_pickle_storage_file_operations(self):
        """Test PickleStorageBackend file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = PickleStorageBackend(temp_dir)

            # Test list_graphs when empty
            graphs = storage.list_graphs()
            assert graphs == []

            # Test graph_exists for nonexistent graph
            exists = storage.graph_exists("nonexistent")
            assert not exists

            # Test load_graph for nonexistent graph
            graph = storage.load_graph("nonexistent")
            assert graph is None

            # Test delete_graph for nonexistent graph
            result = storage.delete_graph("nonexistent")
            assert result is False


class TestStorageManager:
    """Test StorageManager implementation."""

    def test_storage_manager_initialization(self):
        """Test StorageManager initialization."""
        mock_backend = MagicMock()
        manager = StorageManager(mock_backend)
        assert manager.backend == mock_backend

    def test_storage_manager_methods_exist(self):
        """Test StorageManager has required methods."""
        mock_backend = MagicMock()
        manager = StorageManager(mock_backend)

        methods = [
            "save_graph",
            "load_graph",
            "delete_graph",
            "list_graphs",
            "graph_exists",
        ]
        for method in methods:
            assert hasattr(manager, method)
            assert callable(getattr(manager, method))

    def test_storage_manager_delegates_to_backend(self):
        """Test StorageManager delegates calls to backend."""
        mock_backend = MagicMock()
        manager = StorageManager(mock_backend)

        # Test save_graph delegation
        mock_graph = MagicMock()
        manager.save_graph(mock_graph)
        mock_backend.save_graph.assert_called_once_with(mock_graph)

        # Test load_graph delegation
        manager.load_graph("test-id")
        mock_backend.load_graph.assert_called_once_with("test-id")

        # Test delete_graph delegation
        manager.delete_graph("test-id")
        mock_backend.delete_graph.assert_called_once_with("test-id")

        # Test list_graphs delegation
        manager.list_graphs()
        mock_backend.list_graphs.assert_called_once()

        # Test graph_exists delegation
        manager.graph_exists("test-id")
        mock_backend.graph_exists.assert_called_once_with("test-id")

    def test_storage_manager_switch_backend(self):
        """Test StorageManager switch_backend method."""
        mock_backend1 = MagicMock()
        mock_backend2 = MagicMock()

        manager = StorageManager(mock_backend1)
        assert manager.backend == mock_backend1

        manager.switch_backend(mock_backend2)
        assert manager.backend == mock_backend2

    def test_storage_manager_get_backend_info(self):
        """Test StorageManager get_backend_info method."""
        mock_backend = MagicMock()
        mock_backend.__class__.__name__ = "MockBackend"

        manager = StorageManager(mock_backend)
        info = manager.get_backend_info()

        assert "type" in info
        assert info["type"] == "MockBackend"

    def test_storage_manager_error_handling(self):
        """Test StorageManager error handling."""
        mock_backend = MagicMock()
        mock_backend.save_graph.side_effect = Exception("Backend error")

        manager = StorageManager(mock_backend)
        mock_graph = MagicMock()

        # Should not raise exception
        manager.save_graph(mock_graph)
        # The exact return value depends on implementation
        # but it should handle the error gracefully


class TestStorageIntegration:
    """Integration tests for storage components."""

    def test_file_storage_with_manager(self):
        """Test FileStorageBackend with StorageManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileStorageBackend(temp_dir)
            manager = StorageManager(backend)

            # Test that manager can use file backend
            graphs = manager.list_graphs()
            assert graphs == []

            exists = manager.graph_exists("test")
            assert not exists

    def test_database_storage_with_manager(self):
        """Test DatabaseStorageBackend with StorageManager."""
        mock_session = MagicMock()
        backend = DatabaseStorageBackend(mock_session)
        manager = StorageManager(backend)

        # Test that manager can use database backend
        mock_session.query.return_value.all.return_value = []
        graphs = manager.list_graphs()
        assert graphs == []

    def test_pickle_storage_with_manager(self):
        """Test PickleStorageBackend with StorageManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = PickleStorageBackend(temp_dir)
            manager = StorageManager(backend)

            # Test that manager can use pickle backend
            graphs = manager.list_graphs()
            assert graphs == []

            exists = manager.graph_exists("test")
            assert not exists

    def test_backend_switching(self):
        """Test switching between storage backends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_backend = FileStorageBackend(temp_dir)
            pickle_backend = PickleStorageBackend(temp_dir)

            manager = StorageManager(file_backend)
            assert manager.backend == file_backend

            manager.switch_backend(pickle_backend)
            assert manager.backend == pickle_backend
