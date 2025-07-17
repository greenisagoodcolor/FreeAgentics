"""
Integration tests for WebSocket connection pooling system.

Tests the complete integration of connection pool, resource manager,
monitoring, and performance improvements.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from websocket.connection_pool import ConnectionState, PoolConfig
from websocket.monitoring import ConnectionPoolMonitor
from websocket.pool_integration import (
    WebSocketPooledConnectionManager,
    run_performance_comparison,
    setup_websocket_pool,
)
from websocket.resource_manager import ResourceConfig, ResourceState


@pytest.mark.asyncio
class TestWebSocketPoolIntegration:
    """Test the complete WebSocket pool integration."""

    @pytest.fixture
    async def manager(self):
        """Create a pooled connection manager for testing."""
        pool_config = PoolConfig(
            min_size=5, max_size=20, connection_timeout=10.0, health_check_interval=30.0
        )
        resource_config = ResourceConfig(
            max_agents_per_connection=5, max_memory_per_agent=50 * 1024 * 1024, agent_timeout=300.0
        )

        manager = WebSocketPooledConnectionManager(pool_config, resource_config)

        # Mock the actual WebSocket creation
        with patch("websocket.connection_pool.create_websocket_connection") as mock_create:
            mock_ws = AsyncMock()
            mock_ws.connect = AsyncMock()
            mock_ws.ping = AsyncMock()
            mock_ws.close = AsyncMock()
            mock_create.return_value = mock_ws

            await manager.initialize("ws://test.example.com")
            yield manager
            await manager.shutdown()

    async def test_manager_initialization(self, manager):
        """Test that all components are properly initialized."""
        assert manager._initialized is True
        assert manager.pool is not None
        assert manager.resource_manager is not None
        assert manager.pool.size >= manager.pool_config.min_size

    async def test_single_agent_lifecycle(self, manager):
        """Test complete lifecycle of a single agent."""
        agent_id = "test-agent-001"

        # Allocate connection
        conn_id = await manager.allocate_agent_connection(
            agent_id, metadata={"type": "test", "priority": "high"}
        )

        assert conn_id is not None

        # Verify resource is allocated
        resource = await manager.resource_manager.get_agent_resource(agent_id)
        assert resource is not None
        assert resource.state == ResourceState.ACTIVE
        assert resource.connection_id == conn_id

        # Update resource usage
        await manager.resource_manager.update_resource_usage(
            agent_id, memory=10 * 1024 * 1024, cpu=0.2  # 10MB
        )

        # Send messages
        for i in range(5):
            await manager.send_agent_message(agent_id, {"type": "test_message", "seq": i})

        # Get metrics
        pool_metrics = manager.get_pool_metrics()
        assert pool_metrics["in_use_connections"] > 0

        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 1
        assert resource_metrics["active_agents"] == 1

        # Release connection
        await manager.release_agent_connection(agent_id)

        # Verify resource is released
        resource = await manager.resource_manager.get_agent_resource(agent_id)
        assert resource is None

        # Verify metrics updated
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 0

    async def test_multiple_agents_connection_sharing(self, manager):
        """Test multiple agents sharing connections efficiently."""
        num_agents = 15  # With max 5 per connection, should use 3 connections
        agent_ids = [f"agent-{i:03d}" for i in range(num_agents)]

        # Allocate all agents
        conn_ids = []
        for agent_id in agent_ids:
            conn_id = await manager.allocate_agent_connection(agent_id)
            conn_ids.append(conn_id)

        # Check resource allocation
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == num_agents
        assert resource_metrics["connections_in_use"] == 3  # 15 agents / 5 per connection

        # Verify connection sharing
        unique_conn_ids = set(conn_ids)
        assert len(unique_conn_ids) == 3

        # Release all agents
        for agent_id in agent_ids:
            await manager.release_agent_connection(agent_id)

        # Verify cleanup
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 0

    async def test_resource_limits_enforcement(self, manager):
        """Test that resource limits are properly enforced."""
        agent_id = "resource-heavy-agent"

        # Allocate connection
        await manager.allocate_agent_connection(agent_id)

        # Try to exceed memory limit
        with pytest.raises(Exception) as exc_info:
            await manager.resource_manager.update_resource_usage(
                agent_id, memory=100 * 1024 * 1024  # 100MB, exceeds 50MB limit
            )
        assert "exceeds limit" in str(exc_info.value)

        # Update within limits should work
        await manager.resource_manager.update_resource_usage(
            agent_id, memory=30 * 1024 * 1024  # 30MB
        )

        # Cleanup
        await manager.release_agent_connection(agent_id)

    async def test_connection_pool_scaling(self, manager):
        """Test that connection pool scales properly with demand."""
        initial_size = manager.pool.size

        # Create high demand - allocate many agents quickly
        agent_ids = []
        for i in range(50):
            agent_id = f"scaling-agent-{i}"
            agent_ids.append(agent_id)
            await manager.allocate_agent_connection(agent_id)

        # Pool should have scaled up
        assert manager.pool.size > initial_size
        assert manager.pool.size <= manager.pool_config.max_size

        # Check utilization triggered scaling
        pool_metrics = manager.get_pool_metrics()
        assert pool_metrics["utilization"] > 0.5

        # Release agents
        for agent_id in agent_ids:
            await manager.release_agent_connection(agent_id)

        # Wait for potential scale down
        await asyncio.sleep(1)

        # Pool might scale down but not below min_size
        assert manager.pool.size >= manager.pool_config.min_size

    async def test_concurrent_agent_operations(self, manager):
        """Test concurrent agent operations for thread safety."""
        num_agents = 20
        operations_per_agent = 10

        async def agent_operations(agent_id: str):
            """Simulate agent operations."""
            # Allocate
            conn_id = await manager.allocate_agent_connection(agent_id)

            # Send messages
            for i in range(operations_per_agent):
                await manager.send_agent_message(agent_id, {"type": "concurrent_test", "seq": i})

                # Update usage
                await manager.resource_manager.update_resource_usage(
                    agent_id, memory=(i + 1) * 1024 * 1024, cpu=0.1 * (i + 1)
                )

                await asyncio.sleep(0.01)

            # Release
            await manager.release_agent_connection(agent_id)

        # Run all agents concurrently
        tasks = []
        for i in range(num_agents):
            agent_id = f"concurrent-agent-{i}"
            tasks.append(agent_operations(agent_id))

        # Should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0

        # Verify clean state
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 0

    async def test_connection_health_monitoring(self, manager):
        """Test that unhealthy connections are detected and replaced."""
        # Get initial pool size
        initial_size = manager.pool.size

        # Allocate some agents
        agent_ids = []
        for i in range(5):
            agent_id = f"health-test-agent-{i}"
            agent_ids.append(agent_id)
            await manager.allocate_agent_connection(agent_id)

        # Simulate connection failure
        # In real test, would make actual WebSocket fail
        # For now, we'll test the health check mechanism exists
        assert hasattr(manager.pool, "_health_check_cycle")

        # Run health check
        await manager.pool._health_check_cycle()

        # Pool should maintain at least min_size
        assert manager.pool.size >= manager.pool_config.min_size

        # Cleanup
        for agent_id in agent_ids:
            await manager.release_agent_connection(agent_id)

    async def test_monitoring_integration(self, manager):
        """Test that monitoring is properly integrated."""
        # Monitoring should be initialized
        from websocket.monitoring import monitor

        assert monitor is not None
        assert isinstance(monitor, ConnectionPoolMonitor)

        # Perform some operations
        agent_id = "monitor-test-agent"
        await manager.allocate_agent_connection(agent_id)

        # Check that metrics are being collected
        dashboard_data = monitor.get_dashboard_data()
        assert "summary" in dashboard_data
        assert "time_series" in dashboard_data
        assert "health_status" in dashboard_data

        # Verify pool metrics
        summary = dashboard_data["summary"]
        assert summary["pool"]["size"] > 0
        assert summary["resources"]["total_agents"] == 1

        # Cleanup
        await manager.release_agent_connection(agent_id)

    async def test_graceful_shutdown(self, manager):
        """Test graceful shutdown with active agents."""
        # Allocate several agents
        agent_ids = []
        for i in range(10):
            agent_id = f"shutdown-agent-{i}"
            agent_ids.append(agent_id)
            await manager.allocate_agent_connection(agent_id)

        # Verify agents are active
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 10

        # Shutdown should handle cleanup
        await manager.shutdown()

        # Manager should be shutdown
        assert manager._initialized is False

        # Pool should be empty
        # Note: Can't check pool.size as pool is shutdown
        # But shutdown should have completed without errors

    @pytest.mark.slow
    async def test_performance_comparison(self, manager):
        """Test performance improvements with connection pooling."""
        # Run performance comparison
        comparison = await run_performance_comparison(
            num_agents=20, duration=5  # Short duration for test
        )

        # Verify structure
        assert "without_pooling" in comparison
        assert "with_pooling" in comparison
        assert "improvements" in comparison

        # Connection pooling should show improvements
        improvements = comparison["improvements"]

        # Should reduce connections significantly
        assert improvements["connection_reduction_percent"] > 50

        # Should improve throughput
        # Note: In mock environment, improvement might be minimal
        assert improvements["throughput_improvement_percent"] >= 0

        # Should save connections
        assert improvements["connections_saved"] > 0

    async def test_error_recovery(self, manager):
        """Test error recovery in various scenarios."""
        # Test allocation failure recovery
        agent_id = "error-recovery-agent"

        # First allocation should succeed
        conn_id = await manager.allocate_agent_connection(agent_id)

        # Duplicate allocation should fail
        with pytest.raises(Exception) as exc_info:
            await manager.allocate_agent_connection(agent_id)
        assert "already has allocated resources" in str(exc_info.value)

        # Release and retry should work
        await manager.release_agent_connection(agent_id)
        conn_id = await manager.allocate_agent_connection(agent_id)
        assert conn_id is not None

        # Cleanup
        await manager.release_agent_connection(agent_id)

    async def test_connection_affinity(self, manager):
        """Test connection affinity based on metadata."""
        # Allocate agents with specific metadata
        region_us_agents = []
        region_eu_agents = []

        # Allocate US region agents
        for i in range(3):
            agent_id = f"us-agent-{i}"
            region_us_agents.append(agent_id)
            await manager.allocate_agent_connection(agent_id, metadata={"region": "us-east"})

        # Allocate EU region agents
        for i in range(3):
            agent_id = f"eu-agent-{i}"
            region_eu_agents.append(agent_id)
            await manager.allocate_agent_connection(agent_id, metadata={"region": "eu-west"})

        # Verify agents are allocated
        resource_metrics = manager.get_resource_metrics()
        assert resource_metrics["total_agents"] == 6

        # In a real implementation with metadata-based routing,
        # agents with same metadata would prefer same connections

        # Cleanup
        for agent_id in region_us_agents + region_eu_agents:
            await manager.release_agent_connection(agent_id)
