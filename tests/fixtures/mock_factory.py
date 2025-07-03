"""
Standardized mock factory for consistent test data generation.

This module provides a centralized way to create mock objects with sensible defaults,
reducing boilerplate and ensuring consistency across tests.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import numpy as np


class MockFactory:
    """Factory for creating standardized mock objects for testing."""

    @staticmethod
    def create_agent(
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        agent_type: str = "explorer",
        status: str = "active",
        **kwargs,
    ) -> Mock:
        """Create a mock agent with sensible defaults.

        Args:
            agent_id: Agent identifier (auto-generated if not provided)
            name: Agent name (auto-generated if not provided)
            agent_type: Type of agent (explorer, guardian, merchant, scholar)
            status: Agent status (active, idle, exploring, etc.)
            **kwargs: Additional attributes to override

        Returns:
            Mock agent object with proper attribute access
        """
        if agent_id is None:
            agent_id = f"agent_{str(uuid.uuid4())[:8]}"
        if name is None:
            name = f"{agent_type.capitalize()} {agent_id[-4:]}"

        defaults = {
            "agent_id": agent_id,
            "name": name,
            "agent_type": agent_type,
            "status": status,
            "position": {"x": 0, "y": 0, "z": 0},
            "resources": {"energy": 100.0, "materials": 50, "information": 25},
            "beliefs": np.array([0.25, 0.25, 0.25, 0.25]),
            "memory": [],
            "metrics": {"steps_taken": 0, "resources_collected": 0, "interactions": 0},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        # Merge with provided kwargs
        for key, value in kwargs.items():
            defaults[key] = value

        # Create mock with proper attribute access
        mock = Mock(spec=object)
        for key, value in defaults.items():
            setattr(mock, key, value)

        # Add commonly used methods
        mock.get_position = Mock(return_value=defaults["position"])
        mock.get_beliefs = Mock(return_value=defaults["beliefs"])
        mock.update_position = Mock()
        mock.update_beliefs = Mock()

        return mock

    @staticmethod
    def create_coalition(
        coalition_id: Optional[str] = None,
        name: Optional[str] = None,
        members: Optional[List[str]] = None,
        **kwargs,
    ) -> Mock:
        """Create a mock coalition with sensible defaults.

        Args:
            coalition_id: Coalition identifier
            name: Coalition name
            members: List of member agent IDs
            **kwargs: Additional attributes

        Returns:
            Mock coalition object
        """
        if coalition_id is None:
            coalition_id = f"coalition_{str(uuid.uuid4())[:8]}"
        if name is None:
            name = f"Coalition {coalition_id[-4:]}"
        if members is None:
            members = []

        defaults = {
            "coalition_id": coalition_id,
            "name": name,
            "members": members,
            "purpose": "exploration",
            "status": "active",
            "formation_time": datetime.now(),
            "shared_resources": {"energy": 0.0, "materials": 0, "information": 0},
            "collective_beliefs": np.zeros(4),
            "success_metrics": {"tasks_completed": 0, "resources_shared": 0, "duration": 0},
        }

        defaults.update(kwargs)

        mock = Mock(spec=object)
        for key, value in defaults.items():
            setattr(mock, key, value)

        # Add coalition methods
        mock.add_member = Mock()
        mock.remove_member = Mock()
        mock.distribute_resources = Mock()
        mock.update_collective_beliefs = Mock()

        return mock

    @staticmethod
    def create_gnn_data(
        num_nodes: int = 10,
        num_edges: int = 15,
        node_features_dim: int = 32,
        edge_features_dim: int = 16,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create mock GNN data structures.

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges
            node_features_dim: Dimension of node features
            edge_features_dim: Dimension of edge features
            **kwargs: Additional graph attributes

        Returns:
            Dictionary with GNN data structures
        """
        # Generate random but consistent graph structure
        edge_index = np.random.randint(0, num_nodes, size=(2, num_edges))

        data = {
            "node_features": np.random.randn(num_nodes, node_features_dim).astype(np.float32),
            "edge_features": np.random.randn(num_edges, edge_features_dim).astype(np.float32),
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "global_features": np.random.randn(128).astype(np.float32),
            "node_labels": np.random.randint(0, 4, size=num_nodes),
            "graph_label": np.random.randint(0, 2),
        }

        data.update(kwargs)
        return data

    @staticmethod
    def create_websocket_connection(**kwargs) -> Mock:
        """Create a mock WebSocket connection.

        Returns:
            Mock WebSocket connection with async methods
        """
        defaults = {"state": "OPEN", "messages": [], "connected": True}
        defaults.update(kwargs)

        mock = AsyncMock()
        for key, value in defaults.items():
            setattr(mock, key, value)

        # Add WebSocket methods
        mock.send = AsyncMock()
        mock.recv = AsyncMock(return_value='{"type": "ping"}')
        mock.close = AsyncMock()
        mock.ping = AsyncMock()

        return mock

    @staticmethod
    def create_conversation_websocket_manager(**kwargs) -> Mock:
        """Create a mock ConversationWebSocketManager.

        Returns:
            Mock ConversationWebSocketManager with proper methods
        """
        mock = AsyncMock()

        # Add manager-specific attributes
        mock.connections = {}
        mock.subscriptions = {}
        mock.conversation_events = []

        # Add manager methods
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.broadcast = AsyncMock()
        mock.send_to_conversation = AsyncMock()
        mock.subscribe = AsyncMock()
        mock.unsubscribe = AsyncMock()
        mock.get_connection_count = Mock(return_value=0)
        mock.get_active_conversations = Mock(return_value=[])

        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(mock, key, value)

        return mock

    @staticmethod
    def create_database_manager(**kwargs) -> Mock:
        """Create a mock DatabaseManager.

        Returns:
            Mock DatabaseManager with proper methods
        """
        mock = Mock()

        # Add manager-specific methods
        mock.get_session = Mock()
        mock.get_connection = AsyncMock()
        mock.close_all_connections = Mock()
        mock.is_connected = Mock(return_value=True)

        # Set up session mock
        session_mock = MockFactory.create_database_session()
        mock.get_session.return_value = session_mock
        mock.get_connection.return_value = session_mock

        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(mock, key, value)

        return mock

    @staticmethod
    def create_database_session(**kwargs) -> Mock:
        """Create a mock database session.

        Returns:
            Mock database session with transaction support
        """
        mock = Mock()

        # Add session methods
        mock.query = Mock(return_value=mock)
        mock.filter = Mock(return_value=mock)
        mock.filter_by = Mock(return_value=mock)
        mock.first = Mock(return_value=None)
        mock.all = Mock(return_value=[])
        mock.count = Mock(return_value=0)
        mock.add = Mock()
        mock.commit = Mock()
        mock.rollback = Mock()
        mock.close = Mock()
        mock.flush = Mock()

        # Add context manager support
        mock.__enter__ = Mock(return_value=mock)
        mock.__exit__ = Mock(return_value=None)

        return mock

    @staticmethod
    def create_active_inference_data(**kwargs) -> Dict[str, Any]:
        """Create mock active inference data structures.

        Returns:
            Dictionary with active inference components
        """
        defaults = {
            "observations": np.array([1, 0, 0, 0]),  # One-hot encoded
            "beliefs": np.array([0.25, 0.25, 0.25, 0.25]),
            # Prefer first state
            "preferences": np.array([1.0, 0.0, 0.0, 0.0]),
            "free_energy": 2.5,
            # EFE for each action
            "expected_free_energy": np.array([2.0, 3.0, 2.5, 4.0]),
            "action_probabilities": np.array([0.4, 0.2, 0.3, 0.1]),
            "precision": 1.0,
            "learning_rate": 0.1,
        }
        defaults.update(kwargs)
        return defaults

    @staticmethod
    def create_api_response(
        status_code: int = 200,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        **kwargs,
    ) -> Mock:
        """Create a mock API response.

        Args:
            status_code: HTTP status code
            data: Response data
            error: Error message if applicable
            **kwargs: Additional response attributes

        Returns:
            Mock response object
        """
        if data is None:
            data = {"status": "success", "data": []}

        response_json = {
            "status_code": status_code,
            "data": data,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        response_json.update(kwargs)

        mock = Mock()
        mock.status_code = status_code
        mock.json = Mock(return_value=response_json)
        mock.text = str(response_json)
        mock.headers = {"Content-Type": "application/json"}
        mock.raise_for_status = Mock()

        if status_code >= 400:
            mock.raise_for_status.side_effect = Exception(f"HTTP {status_code} Error")

        return mock
