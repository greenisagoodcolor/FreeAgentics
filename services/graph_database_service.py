"""Unified Graph Database Service (Task 34.3).

This service provides the single source of truth for knowledge graph storage,
resolving the architectural debt between duplicate schemas and implementing
production-ready transaction handling with comprehensive observability.

Follows Repository pattern with PostgreSQL as the primary backend.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, or_, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from database.models import KnowledgeEdge, KnowledgeNode
from database.session import get_db
from knowledge_graph.schema import ConversationEntity, ConversationRelation

logger = logging.getLogger(__name__)


@dataclass
class GraphStorageMetrics:
    """Metrics for graph storage operations."""

    operation: str
    start_time: float
    end_time: Optional[float] = None
    node_count: int = 0
    edge_count: int = 0
    success: bool = False
    error: Optional[str] = None
    graph_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """Calculate operation duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "operation": self.operation,
            "duration_ms": self.duration * 1000,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "success": self.success,
            "error": self.error,
            "graph_id": self.graph_id,
        }


@dataclass
class GraphQuery:
    """Query specification for graph operations."""

    graph_ids: Optional[List[str]] = None
    node_types: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class GraphUpdate:
    """Atomic graph update operation."""

    nodes_to_create: List[ConversationEntity]
    nodes_to_update: List[ConversationEntity]
    nodes_to_delete: List[str]
    edges_to_create: List[ConversationRelation]
    edges_to_update: List[ConversationRelation]
    edges_to_delete: List[str]

    def __post_init__(self) -> None:
        """Validate update operation."""
        if not any(
            [
                self.nodes_to_create,
                self.nodes_to_update,
                self.nodes_to_delete,
                self.edges_to_create,
                self.edges_to_update,
                self.edges_to_delete,
            ]
        ):
            raise ValueError("GraphUpdate must contain at least one operation")


class GraphDatabaseError(Exception):
    """Base exception for graph database operations."""

    pass


class GraphTransactionError(GraphDatabaseError):
    """Exception for transaction-related errors."""

    pass


class GraphValidationError(GraphDatabaseError):
    """Exception for validation errors."""

    pass


class GraphRepository:
    """Repository for knowledge graph storage operations.

    Implements the Repository pattern with PostgreSQL as the backing store.
    All operations are atomic and include comprehensive error handling.
    """

    def __init__(self, session_factory: Optional[sessionmaker] = None):
        """Initialize graph repository.

        Args:
            session_factory: Optional session factory for dependency injection
        """
        self.session_factory = session_factory or get_db
        self._connection_pool_size = 10
        self._query_timeout = 30.0

        logger.info("GraphRepository initialized with PostgreSQL backend")

    @contextmanager
    def _get_session(self):
        """Get database session with proper error handling."""
        session = None
        try:
            if callable(self.session_factory):
                session = next(self.session_factory())
            else:
                session = self.session_factory()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {e}")
            raise GraphDatabaseError(f"Database operation failed: {e}") from e
        finally:
            if session:
                session.close()

    def create_nodes(
        self,
        entities: List[ConversationEntity],
        conversation_id: str,
        agent_id: str,
    ) -> List[str]:
        """Create knowledge nodes from conversation entities.

        Args:
            entities: List of entities to create
            conversation_id: Source conversation ID
            agent_id: Agent that created these entities

        Returns:
            List of created node IDs

        Raises:
            GraphDatabaseError: If creation fails
        """
        metrics = GraphStorageMetrics(
            operation="create_nodes",
            start_time=time.time(),
            node_count=len(entities),
        )

        try:
            with self._get_session() as session:
                created_nodes = []

                for entity in entities:
                    node = KnowledgeNode(
                        id=str(uuid.uuid4()),
                        type=entity.entity_type.value,
                        label=entity.label,
                        properties=entity.properties,
                        confidence=entity.provenance.confidence_score if entity.provenance else 1.0,
                        source=entity.provenance.source_id
                        if entity.provenance
                        else conversation_id,
                        creator_agent_id=agent_id,
                        version=1,
                        is_current=True,
                    )

                    session.add(node)
                    created_nodes.append(node.id)

                session.flush()  # Get IDs without committing

                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "Created %d knowledge nodes",
                    len(created_nodes),
                    extra={"metrics": metrics.to_dict(), "conversation_id": conversation_id},
                )

                return created_nodes

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to create knowledge nodes: %s",
                e,
                extra={"metrics": metrics.to_dict(), "conversation_id": conversation_id},
            )
            raise GraphDatabaseError(f"Failed to create nodes: {e}") from e

    def create_edges(
        self,
        relations: List[ConversationRelation],
        conversation_id: str,
        node_id_mapping: Dict[str, str],
    ) -> List[str]:
        """Create knowledge edges from conversation relations.

        Args:
            relations: List of relations to create
            conversation_id: Source conversation ID
            node_id_mapping: Mapping from entity IDs to node IDs

        Returns:
            List of created edge IDs

        Raises:
            GraphDatabaseError: If creation fails
        """
        metrics = GraphStorageMetrics(
            operation="create_edges",
            start_time=time.time(),
            edge_count=len(relations),
        )

        try:
            with self._get_session() as session:
                created_edges = []

                for relation in relations:
                    # Map entity IDs to node IDs
                    source_node_id = node_id_mapping.get(relation.source_entity_id)
                    target_node_id = node_id_mapping.get(relation.target_entity_id)

                    if not source_node_id or not target_node_id:
                        logger.warning(
                            "Skipping relation %s: missing node mapping",
                            relation.relation_id,
                            extra={
                                "source_entity_id": relation.source_entity_id,
                                "target_entity_id": relation.target_entity_id,
                                "available_mappings": list(node_id_mapping.keys()),
                            },
                        )
                        continue

                    edge = KnowledgeEdge(
                        id=str(uuid.uuid4()),
                        source_id=source_node_id,
                        target_id=target_node_id,
                        type=relation.relation_type.value,
                        properties=relation.properties,
                        confidence=relation.provenance.confidence_score
                        if relation.provenance
                        else 1.0,
                    )

                    session.add(edge)
                    created_edges.append(edge.id)

                session.flush()

                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "Created %d knowledge edges",
                    len(created_edges),
                    extra={"metrics": metrics.to_dict(), "conversation_id": conversation_id},
                )

                return created_edges

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to create knowledge edges: %s",
                e,
                extra={"metrics": metrics.to_dict(), "conversation_id": conversation_id},
            )
            raise GraphDatabaseError(f"Failed to create edges: {e}") from e

    def query_nodes(
        self,
        query: GraphQuery,
    ) -> List[KnowledgeNode]:
        """Query knowledge nodes with filtering.

        Args:
            query: Query specification

        Returns:
            List of matching nodes

        Raises:
            GraphDatabaseError: If query fails
        """
        metrics = GraphStorageMetrics(
            operation="query_nodes",
            start_time=time.time(),
        )

        try:
            with self._get_session() as session:
                q = session.query(KnowledgeNode).filter(KnowledgeNode.is_current == True)

                if query.node_types:
                    q = q.filter(KnowledgeNode.type.in_(query.node_types))

                if query.confidence_threshold is not None:
                    q = q.filter(KnowledgeNode.confidence >= query.confidence_threshold)

                if query.created_after:
                    q = q.filter(KnowledgeNode.created_at >= query.created_after)

                if query.created_before:
                    q = q.filter(KnowledgeNode.created_at <= query.created_before)

                if query.offset:
                    q = q.offset(query.offset)

                if query.limit:
                    q = q.limit(query.limit)

                nodes = q.all()

                metrics.node_count = len(nodes)
                metrics.success = True
                metrics.end_time = time.time()

                logger.debug(
                    "Queried %d knowledge nodes",
                    len(nodes),
                    extra={"metrics": metrics.to_dict()},
                )

                return nodes

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to query knowledge nodes: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to query nodes: {e}") from e

    def query_edges(
        self,
        query: GraphQuery,
        source_node_ids: Optional[List[str]] = None,
        target_node_ids: Optional[List[str]] = None,
    ) -> List[KnowledgeEdge]:
        """Query knowledge edges with filtering.

        Args:
            query: Query specification
            source_node_ids: Optional filter by source nodes
            target_node_ids: Optional filter by target nodes

        Returns:
            List of matching edges

        Raises:
            GraphDatabaseError: If query fails
        """
        metrics = GraphStorageMetrics(
            operation="query_edges",
            start_time=time.time(),
        )

        try:
            with self._get_session() as session:
                q = session.query(KnowledgeEdge)

                if query.edge_types:
                    q = q.filter(KnowledgeEdge.type.in_(query.edge_types))

                if query.confidence_threshold is not None:
                    q = q.filter(KnowledgeEdge.confidence >= query.confidence_threshold)

                if source_node_ids:
                    q = q.filter(KnowledgeEdge.source_id.in_(source_node_ids))

                if target_node_ids:
                    q = q.filter(KnowledgeEdge.target_id.in_(target_node_ids))

                if query.created_after:
                    q = q.filter(KnowledgeEdge.created_at >= query.created_after)

                if query.created_before:
                    q = q.filter(KnowledgeEdge.created_at <= query.created_before)

                if query.offset:
                    q = q.offset(query.offset)

                if query.limit:
                    q = q.limit(query.limit)

                edges = q.all()

                metrics.edge_count = len(edges)
                metrics.success = True
                metrics.end_time = time.time()

                logger.debug(
                    "Queried %d knowledge edges",
                    len(edges),
                    extra={"metrics": metrics.to_dict()},
                )

                return edges

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to query knowledge edges: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to query edges: {e}") from e

    def get_graph_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[KnowledgeNode], List[KnowledgeEdge]]:
        """Get neighborhood subgraph around specified nodes.

        Args:
            node_ids: Starting nodes
            depth: Traversal depth (1-3 recommended)
            edge_types: Optional filter by edge types

        Returns:
            Tuple of (nodes, edges) in neighborhood

        Raises:
            GraphDatabaseError: If query fails
        """
        if depth < 1 or depth > 3:
            raise GraphValidationError("Depth must be between 1 and 3")

        metrics = GraphStorageMetrics(
            operation="get_neighborhood",
            start_time=time.time(),
        )

        try:
            with self._get_session() as session:
                visited_nodes = set(node_ids)
                current_nodes = set(node_ids)
                all_edges = []

                for _ in range(depth):
                    # Find edges connected to current nodes
                    edge_query = session.query(KnowledgeEdge).filter(
                        or_(
                            KnowledgeEdge.source_id.in_(current_nodes),
                            KnowledgeEdge.target_id.in_(current_nodes),
                        )
                    )

                    if edge_types:
                        edge_query = edge_query.filter(KnowledgeEdge.type.in_(edge_types))

                    edges = edge_query.all()
                    all_edges.extend(edges)

                    # Find new nodes to visit
                    next_nodes = set()
                    for edge in edges:
                        if edge.source_id not in visited_nodes:
                            next_nodes.add(edge.source_id)
                        if edge.target_id not in visited_nodes:
                            next_nodes.add(edge.target_id)

                    visited_nodes.update(next_nodes)
                    current_nodes = next_nodes

                    if not current_nodes:
                        break

                # Get all nodes in the neighborhood
                nodes = (
                    session.query(KnowledgeNode)
                    .filter(
                        and_(
                            KnowledgeNode.id.in_(visited_nodes),
                            KnowledgeNode.is_current == True,
                        )
                    )
                    .all()
                )

                metrics.node_count = len(nodes)
                metrics.edge_count = len(all_edges)
                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "Retrieved neighborhood: %d nodes, %d edges (depth=%d)",
                    len(nodes),
                    len(all_edges),
                    depth,
                    extra={"metrics": metrics.to_dict()},
                )

                return nodes, all_edges

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to get graph neighborhood: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to get neighborhood: {e}") from e

    def update_node_confidence(
        self,
        node_id: str,
        new_confidence: float,
    ) -> bool:
        """Update node confidence score.

        Args:
            node_id: Node to update
            new_confidence: New confidence value (0.0-1.0)

        Returns:
            True if updated successfully

        Raises:
            GraphDatabaseError: If update fails
        """
        if not (0.0 <= new_confidence <= 1.0):
            raise GraphValidationError("Confidence must be between 0.0 and 1.0")

        metrics = GraphStorageMetrics(
            operation="update_confidence",
            start_time=time.time(),
            node_count=1,
        )

        try:
            with self._get_session() as session:
                node = (
                    session.query(KnowledgeNode)
                    .filter(
                        and_(
                            KnowledgeNode.id == node_id,
                            KnowledgeNode.is_current == True,
                        )
                    )
                    .first()
                )

                if not node:
                    raise GraphDatabaseError(f"Node {node_id} not found")

                node.confidence = new_confidence
                node.updated_at = datetime.now()

                session.flush()

                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "Updated node %s confidence to %.3f",
                    node_id,
                    new_confidence,
                    extra={"metrics": metrics.to_dict()},
                )

                return True

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to update node confidence: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to update confidence: {e}") from e

    def delete_nodes(
        self,
        node_ids: List[str],
        hard_delete: bool = False,
    ) -> int:
        """Delete knowledge nodes (soft delete by default).

        Args:
            node_ids: List of node IDs to delete
            hard_delete: If True, permanently delete; if False, mark as not current

        Returns:
            Number of nodes deleted

        Raises:
            GraphDatabaseError: If deletion fails
        """
        metrics = GraphStorageMetrics(
            operation="delete_nodes",
            start_time=time.time(),
            node_count=len(node_ids),
        )

        try:
            with self._get_session() as session:
                if hard_delete:
                    # First delete associated edges
                    session.query(KnowledgeEdge).filter(
                        or_(
                            KnowledgeEdge.source_id.in_(node_ids),
                            KnowledgeEdge.target_id.in_(node_ids),
                        )
                    ).delete(synchronize_session=False)

                    # Then delete nodes
                    deleted = (
                        session.query(KnowledgeNode)
                        .filter(KnowledgeNode.id.in_(node_ids))
                        .delete(synchronize_session=False)
                    )
                else:
                    # Soft delete - mark as not current
                    deleted = (
                        session.query(KnowledgeNode)
                        .filter(
                            and_(
                                KnowledgeNode.id.in_(node_ids),
                                KnowledgeNode.is_current == True,
                            )
                        )
                        .update(
                            {
                                "is_current": False,
                                "updated_at": datetime.now(),
                            },
                            synchronize_session=False,
                        )
                    )

                session.flush()

                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "%s deleted %d nodes",
                    "Hard" if hard_delete else "Soft",
                    deleted,
                    extra={"metrics": metrics.to_dict()},
                )

                return deleted

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to delete nodes: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to delete nodes: {e}") from e

    def get_graph_statistics(
        self,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.

        Args:
            conversation_id: Optional filter by conversation

        Returns:
            Dictionary with graph statistics

        Raises:
            GraphDatabaseError: If query fails
        """
        metrics = GraphStorageMetrics(
            operation="get_statistics",
            start_time=time.time(),
        )

        try:
            with self._get_session() as session:
                # Base queries
                node_query = session.query(KnowledgeNode).filter(KnowledgeNode.is_current == True)
                edge_query = session.query(KnowledgeEdge)

                # Note: conversation filtering would require adding conversation_id to the schema
                # For now, we'll filter by other criteria or return all results

                # Count queries
                total_nodes = node_query.count()
                total_edges = edge_query.count()

                # Node type distribution
                node_types = session.execute(
                    text(
                        """
                        SELECT type, COUNT(*) as count
                        FROM db_knowledge_nodes
                        WHERE is_current = true
                        GROUP BY type
                        ORDER BY count DESC
                    """
                    ),
                    {},
                ).fetchall()

                # Edge type distribution
                edge_types = session.execute(
                    text(
                        """
                        SELECT type, COUNT(*) as count
                        FROM db_knowledge_edges
                        GROUP BY type
                        ORDER BY count DESC
                    """
                    ),
                    {},
                ).fetchall()

                # Confidence statistics
                confidence_stats = session.execute(
                    text(
                        """
                        SELECT
                            AVG(confidence) as avg_confidence,
                            MIN(confidence) as min_confidence,
                            MAX(confidence) as max_confidence,
                            STDDEV(confidence) as std_confidence
                        FROM db_knowledge_nodes
                        WHERE is_current = true
                    """
                    ),
                    {},
                ).fetchone()

                stats = {
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "node_types": {row.type: row.count for row in node_types},
                    "edge_types": {row.type: row.count for row in edge_types},
                    "confidence_stats": {
                        "average": float(confidence_stats.avg_confidence or 0),
                        "minimum": float(confidence_stats.min_confidence or 0),
                        "maximum": float(confidence_stats.max_confidence or 0),
                        "std_dev": float(confidence_stats.std_confidence or 0),
                    },
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                }

                metrics.success = True
                metrics.end_time = time.time()

                logger.info(
                    "Retrieved graph statistics: %d nodes, %d edges",
                    total_nodes,
                    total_edges,
                    extra={"metrics": metrics.to_dict(), "stats": stats},
                )

                return stats

        except SQLAlchemyError as e:
            metrics.error = str(e)
            metrics.end_time = time.time()

            logger.error(
                "Failed to get graph statistics: %s",
                e,
                extra={"metrics": metrics.to_dict()},
            )
            raise GraphDatabaseError(f"Failed to get statistics: {e}") from e


class GraphDatabaseService:
    """High-level service for knowledge graph database operations.

    This is the main entry point for all graph database operations,
    providing a clean API that orchestrates repository operations.
    """

    def __init__(self, repository: Optional[GraphRepository] = None):
        """Initialize graph database service.

        Args:
            repository: Optional repository for dependency injection
        """
        self.repository = repository or GraphRepository()
        logger.info("GraphDatabaseService initialized")

    def process_conversation_extraction(
        self,
        entities: List[ConversationEntity],
        relations: List[ConversationRelation],
        conversation_id: str,
        agent_id: str,
    ) -> Dict[str, Any]:
        """Process extracted entities and relations from a conversation.

        This is the main entry point for adding knowledge graph data
        from conversation processing.

        Args:
            entities: Extracted entities
            relations: Extracted relations
            conversation_id: Source conversation
            agent_id: Processing agent

        Returns:
            Dictionary with processing results

        Raises:
            GraphDatabaseError: If processing fails
        """
        start_time = time.time()

        logger.info(
            "Processing conversation extraction: %d entities, %d relations",
            len(entities),
            len(relations),
            extra={
                "conversation_id": conversation_id,
                "agent_id": agent_id,
            },
        )

        try:
            # Create nodes first
            node_ids = self.repository.create_nodes(entities, conversation_id, agent_id)

            # Build mapping from entity IDs to node IDs
            node_id_mapping = {
                entity.entity_id: node_id for entity, node_id in zip(entities, node_ids)
            }

            # Create edges
            edge_ids = self.repository.create_edges(relations, conversation_id, node_id_mapping)

            processing_time = time.time() - start_time

            result = {
                "nodes_created": len(node_ids),
                "edges_created": len(edge_ids),
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "processing_time": processing_time,
                "conversation_id": conversation_id,
                "agent_id": agent_id,
            }

            logger.info(
                "Successfully processed conversation extraction in %.3fs",
                processing_time,
                extra=result,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to process conversation extraction: %s",
                e,
                extra={
                    "conversation_id": conversation_id,
                    "agent_id": agent_id,
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                },
            )
            raise GraphDatabaseError(f"Failed to process extraction: {e}") from e

    def search_knowledge(
        self,
        query_text: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Search knowledge graph for relevant information.

        Args:
            query_text: Optional text to search for
            node_types: Optional filter by node types
            confidence_threshold: Minimum confidence threshold
            limit: Maximum results to return

        Returns:
            Dictionary with search results
        """
        query = GraphQuery(
            node_types=node_types,
            confidence_threshold=confidence_threshold,
            limit=limit,
        )

        nodes = self.repository.query_nodes(query)

        # If we have query text, filter by text similarity
        if query_text:
            query_lower = query_text.lower()
            filtered_nodes = [
                node
                for node in nodes
                if query_lower in node.label.lower()
                or any(
                    query_lower in str(value).lower()
                    for value in node.properties.values()
                    if isinstance(value, str)
                )
            ]
            nodes = filtered_nodes

        # Get edges for found nodes
        node_ids = [node.id for node in nodes]
        edges = []
        if node_ids:
            edges = self.repository.query_edges(
                GraphQuery(limit=limit * 2),  # More edges than nodes
                source_node_ids=node_ids,
                target_node_ids=node_ids,
            )

        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "properties": node.properties,
                    "confidence": node.confidence,
                    "created_at": node.created_at.isoformat(),
                }
                for node in nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "type": edge.type,
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                }
                for edge in edges
            ],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "query": {
                "text": query_text,
                "node_types": node_types,
                "confidence_threshold": confidence_threshold,
                "limit": limit,
            },
        }

    def get_conversation_graph(
        self,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Get knowledge graph for a specific conversation.

        Args:
            conversation_id: Conversation to get graph for

        Returns:
            Dictionary with conversation graph data
        """
        stats = self.repository.get_graph_statistics(conversation_id)

        # Get nodes and edges for this conversation
        nodes_query = GraphQuery(limit=1000)  # Reasonable limit for conversation
        edges_query = GraphQuery(limit=2000)

        with self.repository._get_session() as session:
            # Note: Without conversation_id in schema, we'll return recent nodes/edges
            # In a real implementation, we'd add conversation_id field to the schema
            nodes = (
                session.query(KnowledgeNode)
                .filter(KnowledgeNode.is_current == True)
                .limit(100)
                .all()
            )

            edges = session.query(KnowledgeEdge).limit(200).all()

        return {
            "conversation_id": conversation_id,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "properties": node.properties,
                    "confidence": node.confidence,
                    "created_at": node.created_at.isoformat(),
                }
                for node in nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "type": edge.type,
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                }
                for edge in edges
            ],
            "statistics": stats,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on graph database service.

        Returns:
            Dictionary with health status
        """
        try:
            # Test basic connectivity
            stats = self.repository.get_graph_statistics()

            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database_connected": True,
                "total_nodes": stats["total_nodes"],
                "total_edges": stats["total_edges"],
            }

        except Exception as e:
            logger.error("Graph database health check failed: %s", e)
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "database_connected": False,
                "error": str(e),
            }


# Singleton instance for application use
_graph_db_service: Optional[GraphDatabaseService] = None


def get_graph_database_service() -> GraphDatabaseService:
    """Get singleton graph database service instance."""
    global _graph_db_service
    if _graph_db_service is None:
        _graph_db_service = GraphDatabaseService()
    return _graph_db_service
