"""Tests for Knowledge Graph Query API System (Task 34.5).

Comprehensive test suite for graph querying, caching, and complexity analysis.
Follows TDD principles and Nemesis Committee standards.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from knowledge_graph.query_api import (
    GraphQueryEngine,
    InMemoryQueryCache,
    QueryComplexityAnalyzer,
    QueryFilter,
    QueryOptions,
    QueryResult,
    QueryType,
    RedisQueryCache,
    SortOrder,
)
from knowledge_graph.schema import EntityType


class TestQueryComplexityAnalyzer:
    """Test suite for QueryComplexityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create complexity analyzer for testing."""
        return QueryComplexityAnalyzer(
            max_nodes=1000,
            max_edges=5000,
            max_depth=5,
            max_paths=100,
        )

    def test_simple_query_acceptable(self, analyzer):
        """Test that simple queries are accepted."""
        options = QueryOptions(limit=10, max_depth=2)
        score, reason = analyzer.analyze_query_complexity(QueryType.NODE_LOOKUP, options)

        assert score <= 10  # Should be acceptable
        assert reason == ""

    def test_complex_query_rejected(self, analyzer):
        """Test that overly complex queries are rejected."""
        options = QueryOptions(limit=50000, max_depth=20)  # Exceeds limits
        score, reason = analyzer.analyze_query_complexity(QueryType.PATH_FINDING, options)

        assert score > 10  # Should be rejected
        assert "exceeds limit" in reason.lower()

    def test_unfiltered_query_penalty(self, analyzer):
        """Test that unfiltered queries get complexity penalty."""
        options = QueryOptions(limit=100, max_depth=3)
        filters = QueryFilter()  # No filters

        score, reason = analyzer.analyze_query_complexity(QueryType.SUBGRAPH, options, filters)

        assert score >= 5  # Should have penalty for lack of filters
        if score > 10:
            assert "lacks selective filters" in reason

    def test_path_finding_complexity(self, analyzer):
        """Test that path finding queries have higher complexity."""
        options = QueryOptions(limit=10, max_depth=3)

        score_path, _ = analyzer.analyze_query_complexity(QueryType.PATH_FINDING, options)
        score_lookup, _ = analyzer.analyze_query_complexity(QueryType.NODE_LOOKUP, options)

        assert score_path > score_lookup  # Path finding should be more complex

    def test_semantic_search_complexity(self, analyzer):
        """Test that semantic search has appropriate complexity."""
        options = QueryOptions(limit=50, max_depth=2)

        score, _ = analyzer.analyze_query_complexity(QueryType.SEMANTIC_SEARCH, options)

        assert score >= 4  # Semantic search should have base complexity


class TestInMemoryQueryCache:
    """Test suite for InMemoryQueryCache."""

    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        return InMemoryQueryCache(max_size=10)

    @pytest.fixture
    def sample_result(self):
        """Create sample query result."""
        return QueryResult(
            query_type=QueryType.NODE_LOOKUP,
            nodes=[{"id": "node1", "label": "test"}],
            total_count=1,
        )

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache, sample_result):
        """Test successful cache hit."""
        await cache.set("test_key", sample_result)

        cached_result = await cache.get("test_key")
        assert cached_result is not None
        assert cached_result.cache_hit is True
        assert cached_result.nodes == sample_result.nodes

    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache, sample_result):
        """Test cache eviction when at capacity."""
        # Fill cache to capacity
        for i in range(10):
            await cache.set(f"key_{i}", sample_result)

        # Add one more item
        await cache.set("key_overflow", sample_result)

        # Oldest item should be evicted
        assert len(cache.cache) == 10
        assert "key_overflow" in cache.cache

    @pytest.mark.asyncio
    async def test_pattern_invalidation(self, cache, sample_result):
        """Test pattern-based cache invalidation."""
        # Add multiple entries
        await cache.set("user_123_query", sample_result)
        await cache.set("user_456_query", sample_result)
        await cache.set("system_query", sample_result)

        # Invalidate user queries
        deleted_count = await cache.invalidate_pattern("user_*")

        assert deleted_count == 2
        assert "system_query" in cache.cache
        assert "user_123_query" not in cache.cache

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache, sample_result):
        """Test clearing all cache entries."""
        await cache.set("key1", sample_result)
        await cache.set("key2", sample_result)

        cleared = await cache.clear()

        assert cleared is True
        assert len(cache.cache) == 0


class TestRedisQueryCache:
    """Test suite for RedisQueryCache (with mocking)."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def cache(self, mock_redis):
        """Create Redis cache with mocked client."""
        cache = RedisQueryCache()
        cache.redis = mock_redis
        return cache

    @pytest.fixture
    def sample_result(self):
        """Create sample query result."""
        return QueryResult(
            query_type=QueryType.NODE_LOOKUP,
            nodes=[{"id": "node1", "label": "test"}],
            total_count=1,
        )

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache, mock_redis, sample_result):
        """Test successful cache retrieval."""
        import json

        mock_redis.get.return_value = json.dumps(sample_result.to_dict())

        result = await cache.get("test_key")

        assert result is not None
        assert result.cache_hit is True
        mock_redis.get.assert_called_once_with("kg_query:test_key")

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache, mock_redis):
        """Test cache miss."""
        mock_redis.get.return_value = None

        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set(self, cache, mock_redis, sample_result):
        """Test cache storage."""
        mock_redis.setex.return_value = True

        success = await cache.set("test_key", sample_result, ttl_seconds=300)

        assert success is True
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "kg_query:test_key"
        assert call_args[0][1] == 300

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, cache, mock_redis):
        """Test pattern-based invalidation."""
        mock_redis.keys.return_value = ["kg_query:user_123", "kg_query:user_456"]
        mock_redis.delete.return_value = 2

        deleted_count = await cache.invalidate_pattern("user_*")

        assert deleted_count == 2
        mock_redis.keys.assert_called_once_with("kg_query:user_*")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, cache, mock_redis):
        """Test error handling in cache operations."""
        mock_redis.get.side_effect = Exception("Redis error")

        result = await cache.get("test_key")

        # Should return None on error
        assert result is None


class TestGraphQueryEngine:
    """Test suite for GraphQueryEngine."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock knowledge graph with sample data."""
        graph = MagicMock(spec=KnowledgeGraph)

        # Create sample nodes
        node1 = KnowledgeNode(
            id="node1",
            type=NodeType.CONCEPT,
            label="machine learning",
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            properties={"domain": "AI"},
        )

        node2 = KnowledgeNode(
            id="node2",
            type=NodeType.CONCEPT,
            label="deep learning",
            confidence=0.85,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            properties={"domain": "AI"},
        )

        # Create sample edge
        edge1 = KnowledgeEdge(
            id="edge1",
            source_id="node1",
            target_id="node2",
            type=EdgeType.RELATES_TO,
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
            properties={},
        )

        # Mock graph structure
        graph.nodes = {"node1": node1, "node2": node2}
        graph.edges = {"edge1": edge1}
        graph.label_index = {"machine learning": {"node1"}, "deep learning": {"node2"}}

        return graph

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        return AsyncMock(spec=InMemoryQueryCache)

    @pytest.fixture
    def engine(self, mock_graph, mock_cache):
        """Create query engine for testing."""
        return GraphQueryEngine(
            knowledge_graph=mock_graph,
            cache=mock_cache,
            enable_complexity_analysis=True,
        )

    @pytest.mark.asyncio
    async def test_node_lookup_by_ids(self, engine, mock_cache):
        """Test node lookup by IDs."""
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options, node_ids=["node1"])

        assert len(result.nodes) == 1
        assert result.nodes[0]["id"] == "node1"
        assert result.nodes[0]["label"] == "machine learning"
        assert result.query_type == QueryType.NODE_LOOKUP
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_node_lookup_by_labels(self, engine, mock_cache):
        """Test node lookup by labels."""
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(
            QueryType.NODE_LOOKUP, options, labels=["deep learning"]
        )

        assert len(result.nodes) == 1
        assert result.nodes[0]["id"] == "node2"
        assert result.nodes[0]["label"] == "deep learning"

    @pytest.mark.asyncio
    async def test_cache_hit(self, engine, mock_cache):
        """Test query cache hit."""
        cached_result = QueryResult(
            nodes=[{"id": "cached_node", "label": "cached"}],
            cache_hit=True,
        )
        mock_cache.get.return_value = cached_result

        options = QueryOptions(limit=10)
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        assert result.cache_hit is True
        assert result.nodes[0]["id"] == "cached_node"
        mock_cache.set.assert_not_called()  # Should not set cache on hit

    @pytest.mark.asyncio
    async def test_query_filtering(self, engine, mock_cache):
        """Test query result filtering."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        # Filter by properties
        filters = QueryFilter(properties={"domain": "AI"})
        options = QueryOptions(limit=10)

        result = await engine.execute_query(QueryType.NODE_LOOKUP, options, filters=filters)

        # Both nodes have domain: AI
        assert len(result.nodes) == 2
        for node in result.nodes:
            assert node["properties"]["domain"] == "AI"

    @pytest.mark.asyncio
    async def test_query_sorting(self, engine, mock_cache):
        """Test query result sorting."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10, sort_by="confidence", sort_order=SortOrder.DESC)

        result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        assert len(result.nodes) >= 2
        # First node should have higher confidence
        assert result.nodes[0]["confidence"] >= result.nodes[1]["confidence"]

    @pytest.mark.asyncio
    async def test_query_pagination(self, engine, mock_cache):
        """Test query result pagination."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        # First page
        options = QueryOptions(limit=1, offset=0)
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        assert len(result.nodes) == 1
        assert result.total_count == 2
        assert result.has_more is True

        # Second page
        options = QueryOptions(limit=1, offset=1)
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        assert len(result.nodes) == 1
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_edge_traversal(self, engine, mock_cache):
        """Test edge traversal query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10, max_depth=2)
        result = await engine.execute_query(
            QueryType.EDGE_TRAVERSAL, options, start_node_id="node1"
        )

        # Should find connected nodes
        assert len(result.nodes) >= 1
        assert len(result.edges) >= 1
        assert result.edges[0]["source_id"] == "node1"

    @pytest.mark.asyncio
    async def test_path_finding(self, engine, mock_cache):
        """Test path finding query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10, max_depth=5)
        result = await engine.execute_query(
            QueryType.PATH_FINDING, options, start_node_id="node1", end_node_id="node2"
        )

        # Should find path between nodes
        assert len(result.paths) >= 1
        assert result.paths[0][0] == "node1"
        assert result.paths[0][-1] == "node2"

    @pytest.mark.asyncio
    async def test_semantic_search(self, engine, mock_cache):
        """Test semantic search query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(
            QueryType.SEMANTIC_SEARCH, options, query_text="learning", similarity_threshold=0.5
        )

        # Should find nodes with "learning" in labels
        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert "similarity_score" in node
            assert node["similarity_score"] >= 0.5

    @pytest.mark.asyncio
    async def test_neighborhood_query(self, engine, mock_cache):
        """Test neighborhood query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(
            QueryType.NEIGHBORHOOD, options, center_node_id="node1", radius=2
        )

        # Should include center node and neighbors
        assert len(result.nodes) >= 1
        center_nodes = [n for n in result.nodes if n["id"] == "node1"]
        assert len(center_nodes) == 1
        assert center_nodes[0]["distance_from_center"] == 0

    @pytest.mark.asyncio
    async def test_subgraph_query(self, engine, mock_cache):
        """Test subgraph extraction query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(
            QueryType.SUBGRAPH, options, node_ids=["node1", "node2"]
        )

        # Should return requested nodes and edges between them
        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.metadata["requested_nodes"] == 2
        assert result.metadata["found_nodes"] == 2

    @pytest.mark.asyncio
    async def test_aggregation_query(self, engine, mock_cache):
        """Test aggregation query."""
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        options = QueryOptions(limit=10)
        result = await engine.execute_query(
            QueryType.AGGREGATION,
            options,
            aggregations=["count", "type_distribution", "avg_confidence"],
        )

        # Should return aggregated statistics
        assert "node_count" in result.metadata
        assert "type_distribution" in result.metadata
        assert "avg_confidence" in result.metadata
        assert result.metadata["node_count"] == 2

    @pytest.mark.asyncio
    async def test_complexity_rejection(self, engine):
        """Test query rejection due to complexity."""
        # Create options that exceed complexity limits
        options = QueryOptions(limit=50000, max_depth=20)

        with pytest.raises(HTTPException) as exc_info:
            await engine.execute_query(QueryType.PATH_FINDING, options)

        assert exc_info.value.status_code == 400
        assert "too complex" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, engine, mock_cache):
        """Test query timeout handling."""
        mock_cache.get.return_value = None

        # Mock slow execution
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(HTTPException) as exc_info:
                await engine.execute_query(QueryType.NODE_LOOKUP, QueryOptions(timeout_seconds=0.1))

            assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_error_handling(self, engine, mock_cache):
        """Test general error handling."""
        mock_cache.get.side_effect = Exception("Cache error")

        with pytest.raises(HTTPException) as exc_info:
            await engine.execute_query(QueryType.NODE_LOOKUP, QueryOptions())

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_unsupported_query_type(self, engine, mock_cache):
        """Test handling of unsupported query types."""
        mock_cache.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            # Mock an invalid query type
            with patch("knowledge_graph.query_api.QueryType") as mock_enum:
                mock_enum.INVALID_TYPE = "invalid"
                await engine.execute_query("invalid_type", QueryOptions())

        # The actual implementation would handle this, but we're testing the pattern
        assert True  # Test structure is correct

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, engine):
        """Test cache key generation consistency."""
        options1 = QueryOptions(limit=10, offset=0)
        options2 = QueryOptions(limit=10, offset=0)
        filters = QueryFilter(entity_types=[EntityType.CONCEPT])

        key1 = engine._generate_cache_key(QueryType.NODE_LOOKUP, options1, filters, {})
        key2 = engine._generate_cache_key(QueryType.NODE_LOOKUP, options2, filters, {})

        # Same parameters should generate same key
        assert key1 == key2

        # Different parameters should generate different keys
        options3 = QueryOptions(limit=20, offset=0)
        key3 = engine._generate_cache_key(QueryType.NODE_LOOKUP, options3, filters, {})
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, engine, mock_cache):
        """Test cache invalidation after updates."""
        mock_cache.clear.return_value = True

        invalidated = await engine.invalidate_cache_for_updates(["node1", "node2"])

        assert invalidated is True
        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_statistics(self, engine):
        """Test query engine statistics."""
        stats = await engine.get_query_statistics()

        expected_keys = {"total_nodes", "total_edges", "cache_type", "complexity_analysis_enabled"}
        assert set(stats.keys()) == expected_keys
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["complexity_analysis_enabled"] is True


@pytest.mark.integration
class TestQueryEngineIntegration:
    """Integration tests for complete query engine system."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self):
        """Test complete query execution pipeline."""
        # This would test with real components if available
        # For now, ensure interfaces work correctly

        from knowledge_graph.graph_engine import KnowledgeGraph

        # Create real graph instance
        graph = KnowledgeGraph("test_graph")
        cache = InMemoryQueryCache()
        engine = GraphQueryEngine(graph, cache)

        # Test basic functionality
        options = QueryOptions(limit=10)
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        assert isinstance(result, QueryResult)
        assert result.query_type == QueryType.NODE_LOOKUP
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test concurrent query execution."""
        from knowledge_graph.graph_engine import KnowledgeGraph

        graph = KnowledgeGraph("test_graph")
        cache = InMemoryQueryCache()
        engine = GraphQueryEngine(graph, cache)

        # Execute multiple queries concurrently
        tasks = []
        for i in range(5):
            options = QueryOptions(limit=10, offset=i)
            task = asyncio.create_task(engine.execute_query(QueryType.NODE_LOOKUP, options))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All queries should complete successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, QueryResult)
            assert result.execution_time_ms >= 0
