"""Storage layer for knowledge graph persistence.

This module provides storage backends for persisting knowledge graphs
with support for different storage systems (file, database, etc.).
"""

import json
import logging
import pickle  # nosec B403 - Required for graph serialization, only used with trusted data
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import sessionmaker

from database.base import Base
from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)

logger = logging.getLogger(__name__)


class NodeModel(Base):
    """SQLAlchemy model for knowledge nodes."""

    __tablename__ = "kg_nodes"
    __table_args__ = {'extend_existing': True}  # TODO: ARCHITECTURAL DEBT - CRITICAL: Duplicate kg_nodes table name with knowledge/models.py (see NEMESIS Committee findings)

    id = Column(String(36), primary_key=True)
    graph_id = Column(String(36), index=True)
    type = Column(String(50))
    label = Column(String(255), index=True)
    properties = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    version = Column(Integer, default=1)
    confidence = Column(Float, default=1.0)
    source = Column(String(255))


class EdgeModel(Base):
    """SQLAlchemy model for knowledge edges."""

    __tablename__ = "kg_edges"
    __table_args__ = {'extend_existing': True}  # TODO: ARCHITECTURAL DEBT - CRITICAL: Duplicate kg_edges table name with knowledge/models.py (see NEMESIS Committee findings)

    id = Column(String(36), primary_key=True)
    graph_id = Column(String(36), index=True)
    source_id = Column(String(36), ForeignKey("kg_nodes.id"))
    target_id = Column(String(36), ForeignKey("kg_nodes.id"))
    type = Column(String(50))
    properties = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    confidence = Column(Float, default=1.0)


class GraphMetadataModel(Base):
    """SQLAlchemy model for graph metadata."""

    __tablename__ = "graph_metadata"
    __table_args__ = {'extend_existing': True}  # TODO: ARCHITECTURAL DEBT - Prevent table redefinition conflicts (see NEMESIS Committee findings)

    graph_id = Column(String(36), primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    graph_metadata = Column(JSON)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save a knowledge graph.

        Args:
            graph: Graph to save

        Returns:
            True if saved successfully
        """
        pass

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load a knowledge graph.

        Args:
            graph_id: ID of graph to load

        Returns:
            Loaded graph or None if not found
        """
        pass

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """Delete a knowledge graph.

        Args:
            graph_id: ID of graph to delete

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """List available graphs.

        Returns:
            List of graph metadata
        """
        pass

    @abstractmethod
    def graph_exists(self, graph_id: str) -> bool:
        """Check if a graph exists.

        Args:
            graph_id: ID to check

        Returns:
            True if graph exists
        """
        pass


class FileStorageBackend(StorageBackend):
    """File-based storage backend."""

    def __init__(self, base_path: str = "./knowledge_graphs"):
        """Initialize file storage.

        Args:
            base_path: Base directory for storing graphs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized file storage at {self.base_path}")

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save graph to JSON file."""
        try:
            filepath = self.base_path / f"{graph.graph_id}.json"
            graph.save_to_file(str(filepath))

            # Save metadata
            metadata_path = self.base_path / "metadata.json"
            metadata = self._load_metadata()

            metadata[graph.graph_id] = {
                "graph_id": graph.graph_id,
                "version": graph.version,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load graph from JSON file."""
        try:
            filepath = self.base_path / f"{graph_id}.json"
            if not filepath.exists():
                return None

            return KnowledgeGraph.load_from_file(str(filepath))

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete graph file."""
        try:
            filepath = self.base_path / f"{graph_id}.json"
            if filepath.exists():
                filepath.unlink()

            # Update metadata
            metadata = self._load_metadata()
            if graph_id in metadata:
                del metadata[graph_id]

                metadata_path = self.base_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to delete graph: {e}")
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List available graphs."""
        metadata = self._load_metadata()
        return list(metadata.values())

    def graph_exists(self, graph_id: str) -> bool:
        """Check if graph file exists."""
        filepath = self.base_path / f"{graph_id}.json"
        return filepath.exists()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file."""
        metadata_path = self.base_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}


class DatabaseStorageBackend(StorageBackend):
    """Database-based storage backend using SQLAlchemy."""

    def __init__(self, connection_string: str = "sqlite:///knowledge_graphs.db"):
        """Initialize database storage.

        Args:
            connection_string: Database connection string
        """
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Initialized database storage: {connection_string}")

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save graph to database."""
        session = self.SessionLocal()
        try:
            # Delete existing graph data
            session.query(NodeModel).filter_by(graph_id=graph.graph_id).delete()
            session.query(EdgeModel).filter_by(graph_id=graph.graph_id).delete()
            session.query(GraphMetadataModel).filter_by(graph_id=graph.graph_id).delete()

            # Save nodes
            for node in graph.nodes.values():
                node_model = NodeModel(
                    id=node.id,
                    graph_id=graph.graph_id,
                    type=node.type.value,
                    label=node.label,
                    properties=node.properties,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    version=node.version,
                    confidence=node.confidence,
                    source=node.source,
                )
                session.add(node_model)

            # Save edges
            for edge in graph.edges.values():
                edge_model = EdgeModel(
                    id=edge.id,
                    graph_id=graph.graph_id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    type=edge.type.value,
                    properties=edge.properties,
                    created_at=edge.created_at,
                    confidence=edge.confidence,
                )
                session.add(edge_model)

            # Save metadata
            metadata_model = GraphMetadataModel(
                graph_id=graph.graph_id,
                version=graph.version,
                created_at=graph.created_at,
                updated_at=graph.updated_at,
                graph_metadata={
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
            )
            session.add(metadata_model)

            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save graph to database: {e}")
            return False

        finally:
            session.close()

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load graph from database."""
        session = self.SessionLocal()
        try:
            # Check if graph exists
            metadata = session.query(GraphMetadataModel).filter_by(graph_id=graph_id).first()

            if not metadata:
                return None

            # Create graph
            graph = KnowledgeGraph(graph_id=graph_id)

            # Load nodes
            nodes = session.query(NodeModel).filter_by(graph_id=graph_id).all()
            for node_model in nodes:
                node = KnowledgeNode(
                    id=node_model.id,
                    type=NodeType(node_model.type),
                    label=node_model.label,
                    properties=node_model.properties or {},
                    created_at=node_model.created_at,
                    updated_at=node_model.updated_at,
                    version=node_model.version,
                    confidence=node_model.confidence,
                    source=node_model.source,
                )
                graph.nodes[node.id] = node
                graph.graph.add_node(node.id, data=node)

                # Update indexes
                if node.type not in graph.type_index:
                    graph.type_index[node.type] = set()
                graph.type_index[node.type].add(node.id)

                if node.label:
                    if node.label not in graph.label_index:
                        graph.label_index[node.label] = set()
                    graph.label_index[node.label].add(node.id)

            # Load edges
            edges = session.query(EdgeModel).filter_by(graph_id=graph_id).all()
            for edge_model in edges:
                edge = KnowledgeEdge(
                    id=edge_model.id,
                    source_id=edge_model.source_id,
                    target_id=edge_model.target_id,
                    type=EdgeType(edge_model.type),
                    properties=edge_model.properties or {},
                    created_at=edge_model.created_at,
                    confidence=edge_model.confidence,
                )
                graph.edges[edge.id] = edge
                graph.graph.add_edge(edge.source_id, edge.target_id, key=edge.id, data=edge)

            # Restore metadata
            graph.version = metadata.version
            graph.created_at = metadata.created_at
            graph.updated_at = metadata.updated_at

            return graph

        except Exception as e:
            logger.error(f"Failed to load graph from database: {e}")
            return None

        finally:
            session.close()

    def delete_graph(self, graph_id: str) -> bool:
        """Delete graph from database."""
        session = self.SessionLocal()
        try:
            # Delete in order due to foreign keys
            session.query(EdgeModel).filter_by(graph_id=graph_id).delete()
            session.query(NodeModel).filter_by(graph_id=graph_id).delete()
            session.query(GraphMetadataModel).filter_by(graph_id=graph_id).delete()

            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete graph from database: {e}")
            return False

        finally:
            session.close()

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List available graphs."""
        session = self.SessionLocal()
        try:
            graphs = []

            for metadata in session.query(GraphMetadataModel).all():
                graph_info = {
                    "graph_id": metadata.graph_id,
                    "version": metadata.version,
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat(),
                }

                if metadata.graph_metadata:
                    graph_info.update(metadata.graph_metadata)

                graphs.append(graph_info)

            return graphs

        finally:
            session.close()

    def graph_exists(self, graph_id: str) -> bool:
        """Check if graph exists in database."""
        session = self.SessionLocal()
        try:
            exists = session.query(GraphMetadataModel).filter_by(graph_id=graph_id).count() > 0
            return exists

        finally:
            session.close()


class PickleStorageBackend(StorageBackend):
    """Pickle-based storage backend for fast serialization."""

    def __init__(self, base_path: str = "./knowledge_graphs_pickle"):
        """Initialize pickle storage.

        Args:
            base_path: Base directory for storing graphs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized pickle storage at {self.base_path}")

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save graph using pickle."""
        try:
            filepath = self.base_path / f"{graph.graph_id}.pkl"

            with open(filepath, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update metadata
            self._update_metadata(graph)

            return True

        except Exception as e:
            logger.error(f"Failed to pickle graph: {e}")
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load graph from pickle file."""
        try:
            filepath = self.base_path / f"{graph_id}.pkl"

            if not filepath.exists():
                return None

            with open(filepath, "rb") as f:
                return pickle.load(f)  # nosec B301 - Loading trusted graph data

        except Exception as e:
            logger.error(f"Failed to unpickle graph: {e}")
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete pickle file."""
        try:
            filepath = self.base_path / f"{graph_id}.pkl"

            if filepath.exists():
                filepath.unlink()

            # Update metadata
            metadata = self._load_metadata()
            if graph_id in metadata:
                del metadata[graph_id]
                self._save_metadata(metadata)

            return True

        except Exception as e:
            logger.error(f"Failed to delete pickle: {e}")
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List available graphs."""
        metadata = self._load_metadata()
        return list(metadata.values())

    def graph_exists(self, graph_id: str) -> bool:
        """Check if pickle file exists."""
        filepath = self.base_path / f"{graph_id}.pkl"
        return filepath.exists()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata."""
        metadata_path = self.base_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata."""
        metadata_path = self.base_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _update_metadata(self, graph: KnowledgeGraph):
        """Update metadata for a graph."""
        metadata = self._load_metadata()

        metadata[graph.graph_id] = {
            "graph_id": graph.graph_id,
            "version": graph.version,
            "created_at": graph.created_at.isoformat(),
            "updated_at": graph.updated_at.isoformat(),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        }

        self._save_metadata(metadata)


class StorageManager:
    """Manager for knowledge graph storage operations."""

    def __init__(self, backend: Optional[StorageBackend] = None):
        """Initialize storage manager.

        Args:
            backend: Storage backend to use (defaults to file storage)
        """
        self.backend = backend or FileStorageBackend()

    def save(self, graph: KnowledgeGraph) -> bool:
        """Save a knowledge graph.

        Args:
            graph: Graph to save

        Returns:
            True if saved successfully
        """
        return self.backend.save_graph(graph)

    def load(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load a knowledge graph.

        Args:
            graph_id: ID of graph to load

        Returns:
            Loaded graph or None
        """
        return self.backend.load_graph(graph_id)

    def delete(self, graph_id: str) -> bool:
        """Delete a knowledge graph.

        Args:
            graph_id: ID of graph to delete

        Returns:
            True if deleted successfully
        """
        return self.backend.delete_graph(graph_id)

    def list(self) -> List[Dict[str, Any]]:
        """List available graphs.

        Returns:
            List of graph metadata
        """
        return self.backend.list_graphs()

    def exists(self, graph_id: str) -> bool:
        """Check if a graph exists.

        Args:
            graph_id: ID to check

        Returns:
            True if graph exists
        """
        return self.backend.graph_exists(graph_id)


# Convenience aliases for backward compatibility
FileStorage = FileStorageBackend
SQLStorage = DatabaseStorageBackend
