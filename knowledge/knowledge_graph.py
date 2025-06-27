"""
Knowledge Graph implementation for agent belief systems.

This module implements a graph-based knowledge representation that allows
agents to store, query, and update their beliefs about the world.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BeliefNode:
    """A node representing a belief in the agent's knowledge graph."""

    id: str
    statement: str
    confidence: float
    supporting_patterns: List[str] = field(default_factory=list)
    contradicting_patterns: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """An edge representing a relationship between beliefs."""

    id: str
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    A graph-based knowledge representation system for agents.

    This class manages beliefs, their relationships, and provides methods
    for querying and updating knowledge.
    """

    def __init__(self, agent_id: str) -> None:
        """
        Initialize the knowledge graph.

        Args:
            agent_id: Unique identifier for the agent owning this knowledge graph
        """
        self.agent_id = agent_id
        self.nodes: Dict[str, BeliefNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.node_relationships: Dict[str, Set[str]] = {}  # node_id -> connected_node_ids
        self.created_at = datetime.utcnow()

        logger.info(f"Created knowledge graph for agent {agent_id}")

    def add_belief(
        self,
        statement: str,
        confidence: float,
        supporting_patterns: Optional[List[str]] = None,
        contradicting_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BeliefNode:
        """
        Add a new belief to the knowledge graph.

        Args:
            statement: The belief statement
            confidence: Confidence level (0.0 to 1.0)
            supporting_patterns: Evidence supporting this belief
            contradicting_patterns: Evidence contradicting this belief
            metadata: Additional metadata

        Returns:
            The created BeliefNode
        """
        node_id = str(uuid.uuid4())

        node = BeliefNode(
            id=node_id,
            statement=statement,
            confidence=confidence,
            supporting_patterns=supporting_patterns or [],
            contradicting_patterns=contradicting_patterns or [],
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self.node_relationships[node_id] = set()

        logger.debug(f"Added belief node {node_id}: {statement[:50]}...")
        return node

    def update_belief(
        self,
        node_id: str,
        confidence: Optional[float] = None,
        supporting_patterns: Optional[List[str]] = None,
        contradicting_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing belief node.

        Args:
            node_id: ID of the node to update
            confidence: New confidence level
            supporting_patterns: Updated supporting patterns
            contradicting_patterns: Updated contradicting patterns
            metadata: Updated metadata

        Returns:
            True if node was updated, False if not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        if confidence is not None:
            node.confidence = confidence
        if supporting_patterns is not None:
            node.supporting_patterns = supporting_patterns
        if contradicting_patterns is not None:
            node.contradicting_patterns = contradicting_patterns
        if metadata is not None:
            node.metadata.update(metadata)

        node.updated_at = datetime.utcnow()

        logger.debug(f"Updated belief node {node_id}")
        return True

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeEdge]:
        """
        Add a relationship between two belief nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship (e.g., "supports",
                "contradicts")
            strength: Strength of the relationship (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            The created KnowledgeEdge or None if nodes don't exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge_id = str(uuid.uuid4())

        edge = KnowledgeEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {},
        )

        self.edges[edge_id] = edge
        self.node_relationships[source_id].add(target_id)
        self.node_relationships[target_id].add(source_id)

        logger.debug(f"Added relationship {relationship_type} between {source_id} and
            {target_id}")
        return edge

    def query_beliefs(
        self,
        pattern: Optional[str] = None,
        min_confidence: float = 0.0,
        max_results: Optional[int] = None,
    ) -> List[BeliefNode]:
        """
        Query beliefs based on pattern and confidence.

        Args:
            pattern: Search pattern (substring match in statement)
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results to return

        Returns:
            List of matching BeliefNode objects
        """
        results = []

        for node in self.nodes.values():
            # Check confidence threshold
            if node.confidence < min_confidence:
                continue

            # Check pattern match
            if pattern and pattern.lower() not in node.statement.lower():
                continue

            results.append(node)

        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)

        if max_results:
            results = results[:max_results]

        return results

    def get_related_beliefs(self, node_id: str) -> List[BeliefNode]:
        """
        Get all beliefs related to a given node.

        Args:
            node_id: ID of the node to find relations for

        Returns:
            List of related BeliefNode objects
        """
        if node_id not in self.node_relationships:
            return []

        related_ids = self.node_relationships[node_id]
        return [self.nodes[related_id] for related_id in related_ids if related_id in self.nodes]

    def update_from_message(self, message_content: str, sender_id: str) -> List[BeliefNode]:
        """
        Update knowledge graph based on a received message.

        Args:
            message_content: Content of the received message
            sender_id: ID of the message sender

        Returns:
            List of updated/created belief nodes
        """
        # Simple implementation - in practice would use NLP processing
        updated_nodes = []

        # Create or update belief about the message
        belief_statement = f"Agent {sender_id} communicated: {message_content}"

        # Check if similar belief exists
        existing_beliefs = (
            self.query_beliefs(pattern=f"Agent {sender_id} communicated"))

        if existing_beliefs:
            # Update existing belief
            node = existing_beliefs[0]
            node.statement = belief_statement
            node.confidence = (
                min(1.0, node.confidence + 0.1)  # Increase confidence slightly)
            node.updated_at = datetime.utcnow()
            updated_nodes.append(node)
        else:
            # Create new belief
            node = self.add_belief(
                statement=belief_statement,
                confidence=0.8,  # High confidence in direct communication
                metadata={"source": "communication", "sender": sender_id},
            )
            updated_nodes.append(node)

        return updated_nodes

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "avg_confidence": (
                sum(node.confidence for node in self.nodes.values()) / len(self.nodes)
                if self.nodes
                else 0.0
            ),
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
        }

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.node_relationships.clear()
        logger.info(f"Cleared knowledge graph for agent {self.agent_id}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Export knowledge graph to dictionary format.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "statement": node.statement,
                    "confidence": node.confidence,
                    "supporting_patterns": node.supporting_patterns,
                    "contradicting_patterns": node.contradicting_patterns,
                    "created_at": node.created_at.isoformat(),
                    "updated_at": node.updated_at.isoformat(),
                    "metadata": node.metadata,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_type": edge.relationship_type,
                    "strength": edge.strength,
                    "metadata": edge.metadata,
                }
                for edge_id, edge in self.edges.items()
            },
        }


class PatternExtractor:
    """Extract patterns and relationships from data for knowledge graphs."""

    def __init__(self, min_support: float = (
        0.1, min_confidence: float = 0.5) -> None:)
        """
        Initialize pattern extractor.

        Args:
            min_support: Minimum support threshold for patterns
            min_confidence: Minimum confidence threshold for relationships
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []

        logger.info("PatternExtractor initialized")

    def extract_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str,
        Any]]:
        """
        Extract patterns from structured data.

        Args:
            data: List of data records

        Returns:
            List of discovered patterns
        """
        patterns: List[Dict[str, Any]] = []

        # Simple pattern extraction based on frequency
        if not data:
            return patterns

        # Count attribute value frequencies
        attribute_counts: Dict[str, Dict[Any, int]] = {}
        total_records = len(data)

        for record in data:
            for key, value in record.items():
                if key not in attribute_counts:
                    attribute_counts[key] = {}

                value_str = str(value)
                if value_str not in attribute_counts[key]:
                    attribute_counts[key][value_str] = 0
                attribute_counts[key][value_str] += 1

        # Find frequent patterns
        for attribute, value_counts in attribute_counts.items():
            for value, count in value_counts.items():
                support = count / total_records

                if support >= self.min_support:
                    patterns.append(
                        {
                            "type": "frequent_value",
                            "attribute": attribute,
                            "value": value,
                            "support": support,
                            "count": count,
                        }
                    )

        self.patterns = patterns
        logger.info(f"Extracted {len(patterns)} patterns")
        return patterns

    def extract_relationships(self, data: List[Dict[str, Any]]) -> List[Dict[str,
        Any]]:
        """
        Extract relationships between attributes.

        Args:
            data: List of data records

        Returns:
            List of discovered relationships
        """
        relationships: List[Dict[str, Any]] = []

        if len(data) < 2:
            return relationships

        # Simple co-occurrence analysis
        attribute_pairs: Dict[str, Dict[str, int]] = {}
        total_records = len(data)

        for record in data:
            attributes = list(record.keys())

            # Check all pairs of attributes
            for i, attr1 in enumerate(attributes):
                for j, attr2 in enumerate(attributes[i + 1 :], i + 1):
                    pair_key = f"{attr1}|{attr2}"
                    value_pair = f"{record[attr1]}|{record[attr2]}"

                    if pair_key not in attribute_pairs:
                        attribute_pairs[pair_key] = {}

                    if value_pair not in attribute_pairs[pair_key]:
                        attribute_pairs[pair_key][value_pair] = 0
                    attribute_pairs[pair_key][value_pair] += 1

        # Calculate confidence and find strong relationships
        for pair_key, value_pairs in attribute_pairs.items():
            attr1, attr2 = pair_key.split("|")

            for value_pair, count in value_pairs.items():
                confidence = count / total_records

                if confidence >= self.min_confidence:
                    val1, val2 = value_pair.split("|")

                    relationships.append(
                        {
                            "type": "co_occurrence",
                            "attribute1": attr1,
                            "value1": val1,
                            "attribute2": attr2,
                            "value2": val2,
                            "confidence": confidence,
                            "support": count / total_records,
                            "count": count,
                        }
                    )

        self.relationships = relationships
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of extracted patterns."""
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": list(set(p["type"] for p in self.patterns)),
            "total_relationships": len(self.relationships),
            "relationship_types": list(set(r["type"] for r in self.relationships)),
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
        }

    def filter_patterns(
        self, pattern_type: Optional[str] = (
            None, min_support: Optional[float] = None)
    ) -> List[Dict[str, Any]]:
        """Filter patterns by type and support."""
        filtered = self.patterns

        if pattern_type:
            filtered = [p for p in filtered if p["type"] == pattern_type]

        if min_support is not None:
            filtered = [p for p in filtered if p["support"] >= min_support]

        return filtered
