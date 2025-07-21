"""Query engine for knowledge graph exploration and reasoning.

This module provides a powerful query interface for extracting information
from knowledge graphs using various query patterns and reasoning methods.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries supported."""

    NODE_LOOKUP = "node_lookup"  # Find specific nodes
    PATTERN_MATCH = "pattern_match"  # Match graph patterns
    PATH_QUERY = "path_query"  # Find paths between nodes
    NEIGHBORHOOD = "neighborhood"  # Get node neighborhood
    AGGREGATE = "aggregate"  # Aggregate properties
    TEMPORAL = "temporal"  # Time-based queries
    CAUSAL = "causal"  # Causal chain queries
    BELIEF = "belie"  # Agent belief queries


@dataclass
class QueryResult:
    """Result of a graph query."""

    query_type: QueryType
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edges: List[KnowledgeEdge] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def is_empty(self) -> bool:
        """Check if result is empty."""
        return not self.nodes and not self.edges and not self.paths and not self.aggregates

    def node_count(self) -> int:
        """Get number of nodes in result."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Get number of edges in result."""
        return len(self.edges)


@dataclass
class GraphQuery:
    """Specification for a graph query."""

    query_type: QueryType

    # Node lookup parameters
    node_ids: Optional[List[str]] = None
    node_types: Optional[List[NodeType]] = None
    node_labels: Optional[List[str]] = None
    node_properties: Optional[Dict[str, Any]] = None

    # Edge parameters
    edge_types: Optional[List[EdgeType]] = None
    edge_properties: Optional[Dict[str, Any]] = None

    # Path parameters
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    max_path_length: Optional[int] = None

    # Neighborhood parameters
    center_id: Optional[str] = None
    radius: int = 1

    # Temporal parameters
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Filter parameters
    confidence_threshold: float = 0.0
    source_filter: Optional[List[str]] = None

    # Limit and ordering
    limit: Optional[int] = None
    order_by: Optional[str] = None
    descending: bool = False


class QueryEngine:
    """Engine for querying knowledge graphs."""

    def __init__(self, graph: KnowledgeGraph):
        """Initialize query engine.

        Args:
            graph: Knowledge graph to query
        """
        self.graph = graph
        self.query_cache: Dict[str, QueryResult] = {}

    def execute(self, query: GraphQuery) -> QueryResult:
        """Execute a graph query.

        Args:
            query: Query specification

        Returns:
            Query results
        """
        start_time = datetime.now()

        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.query_cache:
            logger.debug("Returning cached query result")
            return self.query_cache[cache_key]

        # Route to appropriate handler
        if query.query_type == QueryType.NODE_LOOKUP:
            result = self._execute_node_lookup(query)
        elif query.query_type == QueryType.PATTERN_MATCH:
            result = self._execute_pattern_match(query)
        elif query.query_type == QueryType.PATH_QUERY:
            result = self._execute_path_query(query)
        elif query.query_type == QueryType.NEIGHBORHOOD:
            result = self._execute_neighborhood_query(query)
        elif query.query_type == QueryType.AGGREGATE:
            result = self._execute_aggregate_query(query)
        elif query.query_type == QueryType.TEMPORAL:
            result = self._execute_temporal_query(query)
        elif query.query_type == QueryType.CAUSAL:
            result = self._execute_causal_query(query)
        elif query.query_type == QueryType.BELIEF:
            result = self._execute_belief_query(query)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")

        # Calculate execution time
        end_time = datetime.now()
        result.execution_time = (end_time - start_time).total_seconds()

        # Cache result
        self.query_cache[cache_key] = result

        return result

    def _execute_node_lookup(self, query: GraphQuery) -> QueryResult:
        """Execute node lookup query."""
        result = QueryResult(query_type=QueryType.NODE_LOOKUP)

        # Start with all nodes
        candidates = list(self.graph.nodes.values())

        # Filter by IDs
        if query.node_ids:
            candidates = [n for n in candidates if n.id in query.node_ids]

        # Filter by types
        if query.node_types:
            candidates = [n for n in candidates if n.type in query.node_types]

        # Filter by labels
        if query.node_labels:
            candidates = [n for n in candidates if n.label in query.node_labels]

        # Filter by properties
        if query.node_properties:
            candidates = [
                n for n in candidates if self._match_properties(n.properties, query.node_properties)
            ]

        # Filter by confidence
        if query.confidence_threshold > 0:
            candidates = [n for n in candidates if n.confidence >= query.confidence_threshold]

        # Filter by source
        if query.source_filter:
            candidates = [n for n in candidates if n.source in query.source_filter]

        # Sort if requested
        if query.order_by:
            candidates = self._sort_nodes(candidates, query.order_by, query.descending)

        # Apply limit
        if query.limit:
            candidates = candidates[: query.limit]

        result.nodes = candidates
        return result

    def _execute_pattern_match(self, query: GraphQuery) -> QueryResult:
        """Execute pattern matching query."""
        result = QueryResult(query_type=QueryType.PATTERN_MATCH)

        # This is a simplified pattern matcher
        # In a full implementation, this would support complex graph patterns

        # Find nodes matching node criteria
        node_result = self._execute_node_lookup(query)
        matched_nodes = set(n.id for n in node_result.nodes)

        if not matched_nodes:
            return result

        # Find edges between matched nodes
        edges = []
        for edge in self.graph.edges.values():
            if edge.source_id in matched_nodes and edge.target_id in matched_nodes:
                # Check edge type filter
                if query.edge_types and edge.type not in query.edge_types:
                    continue

                # Check edge properties
                if query.edge_properties:
                    if not self._match_properties(edge.properties, query.edge_properties):
                        continue

                edges.append(edge)

        result.nodes = node_result.nodes
        result.edges = edges

        return result

    def _execute_path_query(self, query: GraphQuery) -> QueryResult:
        """Execute path finding query."""
        result = QueryResult(query_type=QueryType.PATH_QUERY)

        if not query.source_id or not query.target_id:
            return result

        # Find shortest path
        path = self.graph.find_path(query.source_id, query.target_id)
        if path:
            result.paths.append(path)

            # Collect nodes on path
            for node_id in path:
                node = self.graph.get_node(node_id)
                if node:
                    result.nodes.append(node)

            # Collect edges on path
            for i in range(len(path) - 1):
                for edge in self.graph.edges.values():
                    if edge.source_id == path[i] and edge.target_id == path[i + 1]:
                        result.edges.append(edge)

        # Find alternative paths if max_path_length specified
        if path and query.max_path_length and query.max_path_length > len(path):
            # Use NetworkX to find all simple paths up to max length
            try:
                all_paths = list(
                    nx.all_simple_paths(
                        self.graph.graph,
                        query.source_id,
                        query.target_id,
                        cutoff=query.max_path_length,
                    )
                )

                # Add unique paths
                for alt_path in all_paths:
                    if alt_path != path:
                        result.paths.append(alt_path)

                # Sort by length
                result.paths.sort(key=len)

            except nx.NetworkXNoPath:
                pass

        return result

    def _execute_neighborhood_query(self, query: GraphQuery) -> QueryResult:
        """Execute neighborhood exploration query."""
        result = QueryResult(query_type=QueryType.NEIGHBORHOOD)

        if not query.center_id or query.center_id not in self.graph.nodes:
            return result

        # Get nodes within radius
        visited = {query.center_id}
        current_level = {query.center_id}

        for _ in range(query.radius):
            next_level = set()
            for node_id in current_level:
                neighbors = self.graph.get_neighbors(node_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        next_level.add(neighbor_id)
                        visited.add(neighbor_id)

            current_level = next_level

        # Collect nodes
        for node_id in visited:
            node = self.graph.get_node(node_id)
            if node:
                # Apply filters
                if query.confidence_threshold > 0 and node.confidence < query.confidence_threshold:
                    continue
                if query.node_types and node.type not in query.node_types:
                    continue

                result.nodes.append(node)

        # Collect edges within neighborhood
        for edge in self.graph.edges.values():
            if edge.source_id in visited and edge.target_id in visited:
                if query.edge_types and edge.type not in query.edge_types:
                    continue
                result.edges.append(edge)

        return result

    def _execute_aggregate_query(self, query: GraphQuery) -> QueryResult:
        """Execute aggregation query."""
        result = QueryResult(query_type=QueryType.AGGREGATE)

        # Get nodes to aggregate
        node_result = self._execute_node_lookup(query)
        nodes = node_result.nodes

        if not nodes:
            return result

        # Basic aggregations
        result.aggregates["count"] = len(nodes)
        result.aggregates["avg_confidence"] = np.mean([n.confidence for n in nodes])
        result.aggregates["min_confidence"] = min(n.confidence for n in nodes)
        result.aggregates["max_confidence"] = max(n.confidence for n in nodes)

        # Type distribution
        type_counts: Dict[str, int] = {}
        for node in nodes:
            type_counts[node.type.value] = type_counts.get(node.type.value, 0) + 1
        result.aggregates["type_distribution"] = type_counts

        # Property aggregations
        property_stats: Dict[str, List[float]] = {}
        for node in nodes:
            for key, value in node.properties.items():
                if isinstance(value, (int, float)):
                    if key not in property_stats:
                        property_stats[key] = []
                    property_stats[key].append(value)

        for key, values in property_stats.items():
            result.aggregates[f"{key}_avg"] = np.mean(values)
            result.aggregates[f"{key}_std"] = np.std(values)

        return result

    def _execute_temporal_query(self, query: GraphQuery) -> QueryResult:
        """Execute temporal query."""
        result = QueryResult(query_type=QueryType.TEMPORAL)

        # Filter nodes by time range
        for node in self.graph.nodes.values():
            include = True

            if query.time_start and node.created_at < query.time_start:
                include = False
            if query.time_end and node.created_at > query.time_end:
                include = False

            if include:
                # Apply other filters
                if query.node_types and node.type not in query.node_types:
                    continue
                if query.confidence_threshold > 0 and node.confidence < query.confidence_threshold:
                    continue

                result.nodes.append(node)

        # Sort by time
        result.nodes.sort(key=lambda n: n.created_at)

        # Apply limit
        if query.limit:
            result.nodes = result.nodes[: query.limit]

        return result

    def _execute_causal_query(self, query: GraphQuery) -> QueryResult:
        """Execute causal chain query."""
        result = QueryResult(query_type=QueryType.CAUSAL)

        if not query.source_id:
            return result

        # Find all causal chains starting from source
        visited = set()
        chains = []

        def trace_causal_chain(node_id: str, chain: List[str]):
            if node_id in visited:
                return

            visited.add(node_id)
            chain = chain + [node_id]

            # Find causal edges
            has_causes = False
            for edge in self.graph.edges.values():
                if edge.source_id == node_id and edge.type == EdgeType.CAUSES:
                    has_causes = True
                    trace_causal_chain(edge.target_id, chain)

            if not has_causes:
                # End of chain
                chains.append(chain)

        trace_causal_chain(query.source_id, [])

        # Return chains as paths
        result.paths = chains

        # Collect all nodes in chains
        all_nodes = set()
        for chain in chains:
            all_nodes.update(chain)

        for node_id in all_nodes:
            node = self.graph.get_node(node_id)
            if node:
                result.nodes.append(node)

        return result

    def _execute_belief_query(self, query: GraphQuery) -> QueryResult:
        """Execute agent belief query."""
        result = QueryResult(query_type=QueryType.BELIEF)

        # Find belief nodes
        belief_nodes = self.graph.find_nodes_by_type(NodeType.BELIEF)

        # Filter by source (agent)
        if query.source_filter:
            belief_nodes = [n for n in belief_nodes if n.source in query.source_filter]

        # Filter by confidence
        if query.confidence_threshold > 0:
            belief_nodes = [n for n in belief_nodes if n.confidence >= query.confidence_threshold]

        # Filter by properties
        if query.node_properties:
            belief_nodes = [
                n
                for n in belief_nodes
                if self._match_properties(n.properties, query.node_properties)
            ]

        result.nodes = belief_nodes

        # Find what agents believe about
        for belief in belief_nodes:
            # Find BELIEVES edges
            for edge in self.graph.edges.values():
                if edge.source_id == belief.source and edge.target_id == belief.id:
                    if edge.type == EdgeType.BELIEVES:
                        result.edges.append(edge)

        return result

    def _match_properties(self, node_props: Dict[str, Any], filter_props: Dict[str, Any]) -> bool:
        """Check if node properties match filter."""
        for key, value in filter_props.items():
            if key not in node_props:
                return False

            if isinstance(value, dict) and "$in" in value:
                # Value in list
                if node_props[key] not in value["$in"]:
                    return False
            elif isinstance(value, dict) and "$gt" in value:
                # Greater than
                if not isinstance(node_props[key], (int, float)):
                    return False
                if node_props[key] <= value["$gt"]:
                    return False
            elif isinstance(value, dict) and "$lt" in value:
                # Less than
                if not isinstance(node_props[key], (int, float)):
                    return False
                if node_props[key] >= value["$lt"]:
                    return False
            elif isinstance(value, dict) and "$regex" in value:
                # Regex match
                import re

                if not isinstance(node_props[key], str):
                    return False
                if not re.match(value["$regex"], node_props[key]):
                    return False
            else:
                # Exact match
                if node_props[key] != value:
                    return False

        return True

    def _sort_nodes(
        self, nodes: List[KnowledgeNode], order_by: str, descending: bool
    ) -> List[KnowledgeNode]:
        """Sort nodes by specified field."""

        def get_confidence(n):
            return n.confidence

        def get_created_at(n):
            return n.created_at

        def get_updated_at(n):
            return n.updated_at

        def get_version(n):
            return n.version

        def get_property(n):
            return n.properties.get(order_by, 0)

        if order_by == "confidence":
            key_func = get_confidence
        elif order_by == "created_at":
            key_func = get_created_at
        elif order_by == "updated_at":
            key_func = get_updated_at
        elif order_by == "version":
            key_func = get_version
        else:
            # Try to sort by property
            key_func = get_property

        return sorted(nodes, key=key_func, reverse=descending)

    def _get_cache_key(self, query: GraphQuery) -> str:
        """Generate cache key for query."""
        # Simple hash of query parameters
        import hashlib
        import json

        # Convert query to dict
        query_dict = {
            "type": query.query_type.value,
            "node_ids": query.node_ids,
            "node_types": [t.value for t in query.node_types] if query.node_types else None,
            "confidence": query.confidence_threshold,
            # Add other relevant fields
        }

        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode(), usedforsecurity=False).hexdigest()

    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
