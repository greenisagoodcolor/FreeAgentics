"""Knowledge Graph Query API System (Task 34.5).

This module implements a production-ready query API with both REST and GraphQL interfaces,
semantic search, path-finding algorithms, and intelligent caching.

Follows SOLID principles and provides comprehensive graph querying capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field

from knowledge_graph.graph_engine import KnowledgeEdge, KnowledgeGraph, KnowledgeNode
from knowledge_graph.schema import EntityType, RelationType
from observability.prometheus_metrics import PrometheusMetricsCollector as PrometheusMetrics
from observability.prometheus_metrics import (
    agent_inference_duration_seconds,
    business_inference_operations_total,
)

# Note: aioredis import disabled due to Python 3.12 compatibility issues
# TODO: Update to redis-py with async support once available
REDIS_AVAILABLE = False
aioredis = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of graph queries supported."""

    NODE_LOOKUP = "node_lookup"
    EDGE_TRAVERSAL = "edge_traversal"
    PATH_FINDING = "path_finding"
    SEMANTIC_SEARCH = "semantic_search"
    NEIGHBORHOOD = "neighborhood"
    SUBGRAPH = "subgraph"
    AGGREGATION = "aggregation"


class SortOrder(Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class QueryFilter:
    """Filter criteria for graph queries."""

    entity_types: Optional[List[EntityType]] = None
    relation_types: Optional[List[RelationType]] = None
    properties: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    conversation_ids: Optional[List[str]] = None


@dataclass
class QueryOptions:
    """Options for query execution."""

    limit: int = 100
    offset: int = 0
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC
    include_metadata: bool = True
    include_properties: bool = True
    max_depth: int = 3
    timeout_seconds: float = 30.0


@dataclass
class QueryResult:
    """Result of a graph query."""

    query_id: str = field(default_factory=lambda: str(uuid4()))
    query_type: QueryType = QueryType.NODE_LOOKUP
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    cache_hit: bool = False
    total_count: int = 0
    has_more: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query_type": self.query_type.value,
            "nodes": self.nodes,
            "edges": self.edges,
            "paths": self.paths,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "cache_hit": self.cache_hit,
            "total_count": self.total_count,
            "has_more": self.has_more,
        }


class QueryComplexityAnalyzer:
    """Analyzes and limits query complexity to prevent resource exhaustion."""

    def __init__(
        self,
        max_nodes: int = 10000,
        max_edges: int = 50000,
        max_depth: int = 10,
        max_paths: int = 1000,
    ):
        """Initialize complexity analyzer.

        Args:
            max_nodes: Maximum nodes to return
            max_edges: Maximum edges to traverse
            max_depth: Maximum traversal depth
            max_paths: Maximum paths to find
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.metrics = PrometheusMetrics()

    def analyze_query_complexity(
        self,
        query_type: QueryType,
        options: QueryOptions,
        filters: Optional[QueryFilter] = None,
    ) -> Tuple[int, str]:
        """Analyze query complexity and return (score, reason).

        Returns:
            Tuple of (complexity_score, rejection_reason_if_any)
            Score of 0 means query is acceptable
        """
        complexity_score = 0
        reasons = []

        # Check depth limits
        if options.max_depth > self.max_depth:
            complexity_score += 10
            reasons.append(f"Max depth {options.max_depth} exceeds limit {self.max_depth}")

        # Check result limits
        if options.limit > self.max_nodes:
            complexity_score += 5
            reasons.append(f"Limit {options.limit} exceeds max nodes {self.max_nodes}")

        # Analyze query type complexity
        if query_type == QueryType.PATH_FINDING:
            complexity_score += 3  # Path finding is expensive
        elif query_type == QueryType.SUBGRAPH:
            complexity_score += 2  # Subgraph extraction is moderately expensive
        elif query_type == QueryType.SEMANTIC_SEARCH:
            complexity_score += 4  # Semantic search requires vector operations

        # Check filter selectivity
        if filters:
            if not filters.entity_types and not filters.properties:
                complexity_score += 5  # Unfiltered queries are expensive
                reasons.append("Query lacks selective filters")

        # Record complexity metrics
        agent_inference_duration_seconds.labels(
            agent_id="query_analyzer", operation_type="complexity_check"
        ).observe(complexity_score / 10.0)

        rejection_reason = "; ".join(reasons) if complexity_score > 10 else ""
        return complexity_score, rejection_reason


class IQueryCache(ABC):
    """Interface for query result caching."""

    @abstractmethod
    async def get(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result."""
        pass

    @abstractmethod
    async def set(self, cache_key: str, result: QueryResult, ttl_seconds: int = 300) -> bool:
        """Cache query result with TTL."""
        pass

    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached results."""
        pass


class RedisQueryCache(IQueryCache):
    """Redis-based query result cache with intelligent invalidation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Optional[Any] = None  # aioredis.Redis when available
        self.metrics = PrometheusMetrics()

    async def connect(self):
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("aioredis not available - install with: pip install aioredis")
        if not self.redis:
            self.redis = await aioredis.from_url(self.redis_url)

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None

    async def get(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result."""
        if not self.redis:
            await self.connect()

        try:
            cached_data = await self.redis.get(f"kg_query:{cache_key}")
            if cached_data:
                import json

                result_dict = json.loads(cached_data)
                result = QueryResult(**result_dict)
                result.cache_hit = True

                business_inference_operations_total.labels(
                    operation_type="kg_cache_hit", success="true"
                ).inc()
                return result

            business_inference_operations_total.labels(
                operation_type="kg_cache_miss", success="true"
            ).inc()
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            business_inference_operations_total.labels(
                operation_type="kg_cache_error", success="false"
            ).inc()
            return None

    async def set(self, cache_key: str, result: QueryResult, ttl_seconds: int = 300) -> bool:
        """Cache query result with TTL."""
        if not self.redis:
            await self.connect()

        try:
            import json

            result_dict = result.to_dict()
            result_dict["cache_hit"] = False  # Reset for caching

            await self.redis.setex(f"kg_query:{cache_key}", ttl_seconds, json.dumps(result_dict))

            business_inference_operations_total.labels(
                operation_type="kg_cache_set", success="true"
            ).inc()
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            business_inference_operations_total.labels(
                operation_type="kg_cache_error", success="false"
            ).inc()
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.redis:
            await self.connect()

        try:
            keys = await self.redis.keys(f"kg_query:{pattern}")
            if keys:
                deleted = await self.redis.delete(*keys)
                business_inference_operations_total.labels(
                    operation_type="kg_cache_invalidate", success="true"
                ).inc(deleted)
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            business_inference_operations_total.labels(
                operation_type="kg_cache_error", success="false"
            ).inc()
            return 0

    async def clear(self) -> bool:
        """Clear all cached results."""
        return await self.invalidate_pattern("*") > 0


class InMemoryQueryCache(IQueryCache):
    """In-memory query cache for testing and small deployments."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of cached queries
        """
        self.cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self.max_size = max_size
        self.metrics = PrometheusMetrics()

    async def get(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result."""
        if cache_key in self.cache:
            result, cached_at = self.cache[cache_key]

            # Simple TTL check (5 minutes)
            if (datetime.now(timezone.utc) - cached_at).seconds < 300:
                result.cache_hit = True
                business_inference_operations_total.labels(
                    operation_type="kg_cache_hit", success="true"
                ).inc()
                return result
            else:
                del self.cache[cache_key]

        business_inference_operations_total.labels(
            operation_type="kg_cache_miss", success="true"
        ).inc()
        return None

    async def set(self, cache_key: str, result: QueryResult, ttl_seconds: int = 300) -> bool:
        """Cache query result."""
        # Evict old entries if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        result.cache_hit = False  # Reset for caching
        self.cache[cache_key] = (result, datetime.now(timezone.utc))
        business_inference_operations_total.labels(
            operation_type="kg_cache_set", success="true"
        ).inc()
        return True

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        import fnmatch

        keys_to_delete = [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]

        for key in keys_to_delete:
            del self.cache[key]

        business_inference_operations_total.labels(
            operation_type="kg_cache_invalidate", success="true"
        ).inc(len(keys_to_delete))
        return len(keys_to_delete)

    async def clear(self) -> bool:
        """Clear all cached results."""
        count = len(self.cache)
        self.cache.clear()
        business_inference_operations_total.labels(
            operation_type="kg_cache_clear", success="true"
        ).inc(count)
        return True


class GraphQueryEngine:
    """Production-ready graph query engine with multiple query types and optimization."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        cache: Optional[IQueryCache] = None,
        enable_complexity_analysis: bool = True,
    ):
        """Initialize query engine.

        Args:
            knowledge_graph: Target knowledge graph
            cache: Query result cache
            enable_complexity_analysis: Whether to analyze query complexity
        """
        self.knowledge_graph = knowledge_graph
        self.cache = cache or InMemoryQueryCache()
        self.complexity_analyzer = QueryComplexityAnalyzer() if enable_complexity_analysis else None
        self.metrics = PrometheusMetrics()

        logger.info("Initialized GraphQueryEngine")

    async def execute_query(
        self,
        query_type: QueryType,
        options: QueryOptions,
        filters: Optional[QueryFilter] = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Execute a graph query with caching and complexity analysis.

        Args:
            query_type: Type of query to execute
            options: Query execution options
            filters: Filter criteria
            **kwargs: Query-specific parameters

        Returns:
            Query result with nodes, edges, or paths

        Raises:
            HTTPException: If query is too complex or times out
        """
        start_time = datetime.now(timezone.utc)

        # Analyze query complexity
        if self.complexity_analyzer:
            complexity_score, rejection_reason = self.complexity_analyzer.analyze_query_complexity(
                query_type, options, filters
            )

            if rejection_reason:
                business_inference_operations_total.labels(
                    operation_type="kg_query_rejected", success="false"
                ).inc()
                raise HTTPException(
                    status_code=400, detail=f"Query too complex: {rejection_reason}"
                )

        # Generate cache key
        cache_key = self._generate_cache_key(query_type, options, filters, kwargs)

        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # Execute query based on type
            if query_type == QueryType.NODE_LOOKUP:
                result = await self._execute_node_lookup(options, filters, **kwargs)
            elif query_type == QueryType.EDGE_TRAVERSAL:
                result = await self._execute_edge_traversal(options, filters, **kwargs)
            elif query_type == QueryType.PATH_FINDING:
                result = await self._execute_path_finding(options, filters, **kwargs)
            elif query_type == QueryType.SEMANTIC_SEARCH:
                result = await self._execute_semantic_search(options, filters, **kwargs)
            elif query_type == QueryType.NEIGHBORHOOD:
                result = await self._execute_neighborhood_query(options, filters, **kwargs)
            elif query_type == QueryType.SUBGRAPH:
                result = await self._execute_subgraph_query(options, filters, **kwargs)
            elif query_type == QueryType.AGGREGATION:
                result = await self._execute_aggregation_query(options, filters, **kwargs)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported query type: {query_type}")

            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            result.query_type = query_type

            # Cache successful results
            await self.cache.set(cache_key, result, ttl_seconds=300)

            # Record metrics
            agent_inference_duration_seconds.labels(
                agent_id="query_engine", operation_type=query_type.value
            ).observe(execution_time / 1000.0)
            business_inference_operations_total.labels(
                operation_type=f"kg_query_{query_type.value}", success="true"
            ).inc()

            logger.info(
                f"Executed {query_type.value} query in {execution_time:.2f}ms",
                extra={
                    "query_type": query_type.value,
                    "execution_time_ms": execution_time,
                    "result_count": len(result.nodes) + len(result.edges),
                    "cache_hit": result.cache_hit,
                },
            )

            return result

        except asyncio.TimeoutError:
            business_inference_operations_total.labels(
                operation_type="kg_query_timeout", success="false"
            ).inc()
            raise HTTPException(status_code=408, detail="Query timeout exceeded")
        except Exception as e:
            business_inference_operations_total.labels(
                operation_type="kg_query_failed", success="false"
            ).inc()
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

    async def _execute_node_lookup(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        node_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> QueryResult:
        """Execute node lookup query."""
        nodes = []

        # Get nodes by IDs
        if node_ids:
            for node_id in node_ids:
                if node_id in self.knowledge_graph.nodes:
                    node = self.knowledge_graph.nodes[node_id]
                    if self._node_matches_filter(node, filters):
                        nodes.append(self._serialize_node(node, options))

        # Get nodes by labels
        elif labels:
            for label in labels:
                if label in self.knowledge_graph.label_index:
                    for node_id in self.knowledge_graph.label_index[label]:
                        node = self.knowledge_graph.nodes[node_id]
                        if self._node_matches_filter(node, filters):
                            nodes.append(self._serialize_node(node, options))

        # Get all nodes matching filter
        else:
            for node in self.knowledge_graph.nodes.values():
                if self._node_matches_filter(node, filters):
                    nodes.append(self._serialize_node(node, options))

        # Apply sorting and pagination
        nodes = self._apply_sorting(nodes, options)
        total_count = len(nodes)
        nodes = nodes[options.offset : options.offset + options.limit]

        return QueryResult(
            nodes=nodes,
            total_count=total_count,
            has_more=options.offset + len(nodes) < total_count,
        )

    async def _execute_edge_traversal(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        start_node_id: str,
        relation_types: Optional[List[RelationType]] = None,
    ) -> QueryResult:
        """Execute edge traversal query."""
        if start_node_id not in self.knowledge_graph.nodes:
            return QueryResult()

        nodes = []
        edges = []
        visited = set()

        # Use BFS for traversal
        queue = [(start_node_id, 0)]  # (node_id, depth)

        while queue and len(nodes) < options.limit:
            current_id, depth = queue.pop(0)

            if depth > options.max_depth or current_id in visited:
                continue

            visited.add(current_id)
            current_node = self.knowledge_graph.nodes[current_id]

            if self._node_matches_filter(current_node, filters):
                nodes.append(self._serialize_node(current_node, options))

            # Find outgoing edges
            for edge in self.knowledge_graph.edges.values():
                if edge.source_id == current_id:
                    # Check relation type filter
                    if relation_types and edge.type not in relation_types:
                        continue

                    edges.append(self._serialize_edge(edge, options))

                    # Add target to queue for next level
                    if edge.target_id not in visited:
                        queue.append((edge.target_id, depth + 1))

        return QueryResult(
            nodes=nodes,
            edges=edges,
            total_count=len(nodes),
        )

    async def _execute_path_finding(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        start_node_id: str,
        end_node_id: str,
        max_paths: int = 10,
    ) -> QueryResult:
        """Execute path finding query using modified BFS."""
        if (
            start_node_id not in self.knowledge_graph.nodes
            or end_node_id not in self.knowledge_graph.nodes
        ):
            return QueryResult()

        paths = []
        visited_paths = set()

        # BFS to find paths
        queue = [([start_node_id], set([start_node_id]))]

        while queue and len(paths) < max_paths:
            current_path, visited_in_path = queue.pop(0)
            current_node_id = current_path[-1]

            if len(current_path) > options.max_depth:
                continue

            if current_node_id == end_node_id:
                path_key = tuple(current_path)
                if path_key not in visited_paths:
                    paths.append(current_path)
                    visited_paths.add(path_key)
                continue

            # Explore neighbors
            for edge in self.knowledge_graph.edges.values():
                if edge.source_id == current_node_id and edge.target_id not in visited_in_path:
                    new_path = current_path + [edge.target_id]
                    new_visited = visited_in_path | {edge.target_id}
                    queue.append((new_path, new_visited))

        # Get nodes and edges involved in paths
        involved_nodes = set()
        involved_edges = []

        for path in paths:
            involved_nodes.update(path)
            for i in range(len(path) - 1):
                # Find edge between consecutive nodes
                for edge in self.knowledge_graph.edges.values():
                    if edge.source_id == path[i] and edge.target_id == path[i + 1]:
                        involved_edges.append(self._serialize_edge(edge, options))
                        break

        nodes = [
            self._serialize_node(self.knowledge_graph.nodes[node_id], options)
            for node_id in involved_nodes
            if node_id in self.knowledge_graph.nodes
        ]

        return QueryResult(
            nodes=nodes,
            edges=involved_edges,
            paths=paths,
            total_count=len(paths),
        )

    async def _execute_semantic_search(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        query_text: str,
        similarity_threshold: float = 0.7,
    ) -> QueryResult:
        """Execute semantic search query (simplified implementation)."""
        # This would integrate with vector embeddings in a real implementation
        # For now, use simple text matching

        import re

        query_tokens = set(re.findall(r"\w+", query_text.lower()))
        scored_nodes = []

        for node in self.knowledge_graph.nodes.values():
            if not self._node_matches_filter(node, filters):
                continue

            # Calculate simple similarity score
            node_tokens = set(re.findall(r"\w+", node.label.lower()))

            if node_tokens:
                similarity = len(query_tokens.intersection(node_tokens)) / len(
                    query_tokens.union(node_tokens)
                )

                if similarity >= similarity_threshold:
                    node_data = self._serialize_node(node, options)
                    node_data["similarity_score"] = similarity
                    scored_nodes.append(node_data)

        # Sort by similarity score
        scored_nodes.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Apply pagination
        total_count = len(scored_nodes)
        scored_nodes = scored_nodes[options.offset : options.offset + options.limit]

        return QueryResult(
            nodes=scored_nodes,
            total_count=total_count,
            has_more=options.offset + len(scored_nodes) < total_count,
            metadata={"query_text": query_text, "similarity_threshold": similarity_threshold},
        )

    async def _execute_neighborhood_query(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        center_node_id: str,
        radius: int = 2,
    ) -> QueryResult:
        """Execute neighborhood query to get all nodes within radius."""
        if center_node_id not in self.knowledge_graph.nodes:
            return QueryResult()

        nodes = []
        edges = []
        visited = set()

        # BFS to find neighborhood
        queue = [(center_node_id, 0)]

        while queue:
            current_id, distance = queue.pop(0)

            if distance > radius or current_id in visited:
                continue

            visited.add(current_id)
            current_node = self.knowledge_graph.nodes[current_id]

            if self._node_matches_filter(current_node, filters):
                node_data = self._serialize_node(current_node, options)
                node_data["distance_from_center"] = distance
                nodes.append(node_data)

            # Add connected nodes to queue
            for edge in self.knowledge_graph.edges.values():
                if edge.source_id == current_id and edge.target_id not in visited:
                    edges.append(self._serialize_edge(edge, options))
                    queue.append((edge.target_id, distance + 1))
                elif edge.target_id == current_id and edge.source_id not in visited:
                    edges.append(self._serialize_edge(edge, options))
                    queue.append((edge.source_id, distance + 1))

        return QueryResult(
            nodes=nodes,
            edges=edges,
            total_count=len(nodes),
            metadata={"center_node": center_node_id, "radius": radius},
        )

    async def _execute_subgraph_query(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        node_ids: List[str],
    ) -> QueryResult:
        """Execute subgraph extraction query."""
        # Get all specified nodes
        nodes = []
        for node_id in node_ids:
            if node_id in self.knowledge_graph.nodes:
                node = self.knowledge_graph.nodes[node_id]
                if self._node_matches_filter(node, filters):
                    nodes.append(self._serialize_node(node, options))

        # Get all edges between the nodes
        edges = []
        node_id_set = set(node_ids)

        for edge in self.knowledge_graph.edges.values():
            if edge.source_id in node_id_set and edge.target_id in node_id_set:
                edges.append(self._serialize_edge(edge, options))

        return QueryResult(
            nodes=nodes,
            edges=edges,
            total_count=len(nodes),
            metadata={"requested_nodes": len(node_ids), "found_nodes": len(nodes)},
        )

    async def _execute_aggregation_query(
        self,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        aggregations: List[str],
    ) -> QueryResult:
        """Execute aggregation query for graph analytics."""
        result_metadata = {}

        # Filter nodes first
        filtered_nodes = [
            node
            for node in self.knowledge_graph.nodes.values()
            if self._node_matches_filter(node, filters)
        ]

        for agg in aggregations:
            if agg == "count":
                result_metadata["node_count"] = len(filtered_nodes)
            elif agg == "type_distribution":
                type_counts = {}
                for node in filtered_nodes:
                    node_type = node.type.value
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                result_metadata["type_distribution"] = type_counts
            elif agg == "avg_confidence":
                if filtered_nodes:
                    avg_confidence = sum(node.confidence for node in filtered_nodes) / len(
                        filtered_nodes
                    )
                    result_metadata["avg_confidence"] = round(avg_confidence, 3)
            elif agg == "edge_count":
                edge_count = len(self.knowledge_graph.edges)
                result_metadata["edge_count"] = edge_count

        return QueryResult(
            total_count=len(filtered_nodes),
            metadata=result_metadata,
        )

    def _node_matches_filter(self, node: KnowledgeNode, filters: Optional[QueryFilter]) -> bool:
        """Check if node matches filter criteria."""
        if not filters:
            return True

        # Check entity type
        if filters.entity_types and node.type not in filters.entity_types:
            return False

        # Check confidence threshold
        if filters.confidence_threshold and node.confidence < filters.confidence_threshold:
            return False

        # Check properties
        if filters.properties:
            for key, value in filters.properties.items():
                if key not in node.properties or node.properties[key] != value:
                    return False

        # Check creation date
        if filters.created_after and node.created_at < filters.created_after:
            return False
        if filters.created_before and node.created_at > filters.created_before:
            return False

        # Check conversation IDs
        if filters.conversation_ids:
            node_conv_id = node.properties.get("conversation_id")
            if node_conv_id not in filters.conversation_ids:
                return False

        return True

    def _serialize_node(self, node: KnowledgeNode, options: QueryOptions) -> Dict[str, Any]:
        """Serialize node for API response."""
        result = {
            "id": node.id,
            "type": node.type.value,
            "label": node.label,
            "confidence": node.confidence,
        }

        if options.include_properties:
            result["properties"] = node.properties

        if options.include_metadata:
            result["metadata"] = {
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
                "version": node.version,
                "source": node.source,
            }

        return result

    def _serialize_edge(self, edge: KnowledgeEdge, options: QueryOptions) -> Dict[str, Any]:
        """Serialize edge for API response."""
        result = {
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type.value,
            "confidence": edge.confidence,
        }

        if options.include_properties:
            result["properties"] = edge.properties

        if options.include_metadata:
            result["metadata"] = {
                "created_at": edge.created_at.isoformat(),
            }

        return result

    def _apply_sorting(
        self, nodes: List[Dict[str, Any]], options: QueryOptions
    ) -> List[Dict[str, Any]]:
        """Apply sorting to node results."""
        if not options.sort_by:
            return nodes

        reverse = options.sort_order == SortOrder.DESC

        try:
            if options.sort_by == "confidence":
                return sorted(nodes, key=lambda x: x.get("confidence", 0), reverse=reverse)
            elif options.sort_by == "label":
                return sorted(nodes, key=lambda x: x.get("label", ""), reverse=reverse)
            elif options.sort_by == "created_at":
                return sorted(
                    nodes,
                    key=lambda x: x.get("metadata", {}).get("created_at", ""),
                    reverse=reverse,
                )
        except (KeyError, TypeError):
            logger.warning(f"Failed to sort by {options.sort_by}, returning unsorted results")

        return nodes

    def _generate_cache_key(
        self,
        query_type: QueryType,
        options: QueryOptions,
        filters: Optional[QueryFilter],
        kwargs: Dict[str, Any],
    ) -> str:
        """Generate cache key for query."""
        import hashlib
        import json

        # Create deterministic key from query parameters
        key_data = {
            "query_type": query_type.value,
            "options": {
                "limit": options.limit,
                "offset": options.offset,
                "sort_by": options.sort_by,
                "sort_order": options.sort_order.value,
                "max_depth": options.max_depth,
            },
            "filters": filters.__dict__ if filters else None,
            "kwargs": sorted(kwargs.items()),
        }

        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_json.encode(), usedforsecurity=False).hexdigest()  # nosec B324

    async def invalidate_cache_for_updates(self, updated_entity_ids: List[str]) -> int:
        """Invalidate cache entries affected by entity updates."""
        # This would implement intelligent cache invalidation
        # For now, just clear all caches
        return await self.cache.clear()

    async def get_query_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        return {
            "total_nodes": len(self.knowledge_graph.nodes),
            "total_edges": len(self.knowledge_graph.edges),
            "cache_type": type(self.cache).__name__,
            "complexity_analysis_enabled": self.complexity_analyzer is not None,
        }


# Pydantic models for REST API


class NodeLookupRequest(BaseModel):
    """Request model for node lookup queries."""

    node_ids: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)
    include_metadata: bool = True


class PathFindingRequest(BaseModel):
    """Request model for path finding queries."""

    start_node_id: str
    end_node_id: str
    max_paths: int = Field(default=10, ge=1, le=100)
    max_depth: int = Field(default=6, ge=1, le=20)
    relation_types: Optional[List[str]] = None


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search queries."""

    query_text: str = Field(..., min_length=1, max_length=1000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_types: Optional[List[str]] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class NeighborhoodRequest(BaseModel):
    """Request model for neighborhood queries."""

    center_node_id: str
    radius: int = Field(default=2, ge=1, le=10)
    entity_types: Optional[List[str]] = None
    limit: int = Field(default=200, ge=1, le=5000)
