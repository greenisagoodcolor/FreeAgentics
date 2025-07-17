"""
Unit tests for WebSocket resource management for agent lifecycle.

Tests resource allocation, cleanup, and lifecycle management for agents
using pooled WebSocket connections.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from websocket.connection_pool import PooledConnection, WebSocketConnectionPool
from websocket.resource_manager import (
    AgentResource,
    AgentResourceManager,
    ResourceAllocationError,
    ResourceConfig,
    ResourceLimits,
    ResourceMetrics,
    ResourceNotFoundError,
    ResourceState,
)


class TestResourceConfig:
    """Test resource management configuration."""

    def test_default_config(self):
        """Test default resource configuration."""
        config = ResourceConfig()
        assert config.max_agents_per_connection == 10
        assert config.max_memory_per_agent == 100 * 1024 * 1024  # 100MB
        assert config.max_cpu_per_agent == 1.0  # 1 CPU core
        assert config.agent_timeout == 3600.0  # 1 hour
        assert config.cleanup_interval == 60.0
        assert config.enable_resource_limits is True

    def test_custom_config(self):
        """Test custom resource configuration."""
        config = ResourceConfig(
            max_agents_per_connection=5, max_memory_per_agent=50 * 1024 * 1024, agent_timeout=1800.0
        )
        assert config.max_agents_per_connection == 5
        assert config.max_memory_per_agent == 50 * 1024 * 1024
        assert config.agent_timeout == 1800.0

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            ResourceConfig(max_agents_per_connection=-1)

        with pytest.raises(ValueError):
            ResourceConfig(max_memory_per_agent=-1)

        with pytest.raises(ValueError):
            ResourceConfig(agent_timeout=-1.0)


class TestAgentResource:
    """Test agent resource representation."""

    def test_agent_resource_creation(self):
        """Test creating an agent resource."""
        resource = AgentResource(
            agent_id="agent-123", connection_id="conn-456", allocated_at=datetime.utcnow()
        )

        assert resource.agent_id == "agent-123"
        assert resource.connection_id == "conn-456"
        assert resource.state == ResourceState.ALLOCATED
        assert resource.memory_usage == 0
        assert resource.cpu_usage == 0.0

    def test_resource_state_transitions(self):
        """Test resource state transitions."""
        resource = AgentResource(
            agent_id="agent-123", connection_id="conn-456", allocated_at=datetime.utcnow()
        )

        # Transition to active
        resource.mark_active()
        assert resource.state == ResourceState.ACTIVE
        assert resource.activated_at is not None

        # Update usage
        resource.update_usage(memory=50 * 1024 * 1024, cpu=0.5)
        assert resource.memory_usage == 50 * 1024 * 1024
        assert resource.cpu_usage == 0.5

        # Transition to idle
        resource.mark_idle()
        assert resource.state == ResourceState.IDLE

        # Transition to released
        resource.mark_released()
        assert resource.state == ResourceState.RELEASED
        assert resource.released_at is not None

    def test_resource_metadata(self):
        """Test resource metadata management."""
        resource = AgentResource(
            agent_id="agent-123", connection_id="conn-456", allocated_at=datetime.utcnow()
        )

        resource.set_metadata("type", "inference")
        resource.set_metadata("priority", "high")

        assert resource.get_metadata("type") == "inference"
        assert resource.get_metadata("priority") == "high"
        assert resource.get_metadata("nonexistent") is None


@pytest.mark.asyncio
class TestAgentResourceManager:
    """Test agent resource manager functionality."""

    @pytest.fixture
    def pool(self):
        """Create a mock connection pool."""
        pool = Mock(spec=WebSocketConnectionPool)
        pool.acquire = AsyncMock()
        pool.release = AsyncMock()
        pool.get_metrics = Mock(return_value={})
        return pool

    @pytest.fixture
    def manager(self, pool):
        """Create a resource manager with mock pool."""
        config = ResourceConfig(max_agents_per_connection=3)
        manager = AgentResourceManager(pool, config)
        return manager

    async def test_manager_initialization(self, manager, pool):
        """Test resource manager initialization."""
        assert manager.pool == pool
        assert len(manager._resources) == 0
        assert len(manager._connection_agents) == 0

    async def test_allocate_resource(self, manager, pool):
        """Test allocating resources for an agent."""
        # Mock connection
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        mock_conn.set_metadata = Mock()
        pool.acquire.return_value = mock_conn

        # Allocate resource
        resource = await manager.allocate_resource("agent-456")

        assert resource.agent_id == "agent-456"
        assert resource.connection_id == "conn-123"
        assert resource.state == ResourceState.ALLOCATED

        # Verify connection metadata was set
        mock_conn.set_metadata.assert_called()

        # Verify resource tracking
        assert "agent-456" in manager._resources
        assert "conn-123" in manager._connection_agents
        assert "agent-456" in manager._connection_agents["conn-123"]

    async def test_activate_resource(self, manager, pool):
        """Test activating an allocated resource."""
        # Setup
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        pool.acquire.return_value = mock_conn

        # Allocate and activate
        resource = await manager.allocate_resource("agent-789")
        await manager.activate_resource("agent-789")

        assert resource.state == ResourceState.ACTIVE
        assert resource.activated_at is not None

    async def test_release_resource(self, manager, pool):
        """Test releasing agent resources."""
        # Setup
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        pool.acquire.return_value = mock_conn

        # Allocate resource
        resource = await manager.allocate_resource("agent-111")

        # Release resource
        await manager.release_resource("agent-111")

        assert resource.state == ResourceState.RELEASED
        assert "agent-111" not in manager._resources
        assert len(manager._connection_agents["conn-123"]) == 0

        # Connection should be released back to pool
        pool.release.assert_called_once_with("conn-123")

    async def test_connection_sharing(self, manager, pool):
        """Test multiple agents sharing a connection."""
        # Mock connection
        mock_conn = Mock()
        mock_conn.connection_id = "conn-shared"
        mock_conn.set_metadata = Mock()
        pool.acquire.return_value = mock_conn

        # Allocate multiple agents
        resource1 = await manager.allocate_resource("agent-1")
        resource2 = await manager.allocate_resource("agent-2")
        resource3 = await manager.allocate_resource("agent-3")

        # All should share the same connection
        assert resource1.connection_id == "conn-shared"
        assert resource2.connection_id == "conn-shared"
        assert resource3.connection_id == "conn-shared"

        # Connection should be acquired only once
        assert pool.acquire.call_count == 1

        # Verify tracking
        assert len(manager._connection_agents["conn-shared"]) == 3

    async def test_connection_limit_per_agent(self, manager, pool):
        """Test respecting max agents per connection limit."""
        # Mock connections
        conn1 = Mock()
        conn1.connection_id = "conn-1"
        conn1.set_metadata = Mock()

        conn2 = Mock()
        conn2.connection_id = "conn-2"
        conn2.set_metadata = Mock()

        pool.acquire.side_effect = [conn1, conn2]

        # Allocate up to limit on first connection
        for i in range(3):  # max_agents_per_connection = 3
            await manager.allocate_resource(f"agent-{i}")

        # Next agent should get new connection
        resource = await manager.allocate_resource("agent-3")
        assert resource.connection_id == "conn-2"

        # Verify two connections were acquired
        assert pool.acquire.call_count == 2

    async def test_resource_limits_enforcement(self, manager, pool):
        """Test enforcing resource limits."""
        config = ResourceConfig(
            max_memory_per_agent=100 * 1024 * 1024,  # 100MB
            max_cpu_per_agent=1.0,
            enable_resource_limits=True,
        )
        manager = AgentResourceManager(pool, config)

        # Setup
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        pool.acquire.return_value = mock_conn

        # Allocate and update usage
        resource = await manager.allocate_resource("agent-mem")

        # Update within limits - should succeed
        await manager.update_resource_usage("agent-mem", memory=50 * 1024 * 1024, cpu=0.5)
        assert resource.memory_usage == 50 * 1024 * 1024

        # Update exceeding memory limit - should raise error
        with pytest.raises(ResourceAllocationError):
            await manager.update_resource_usage("agent-mem", memory=150 * 1024 * 1024, cpu=0.5)

        # Update exceeding CPU limit - should raise error
        with pytest.raises(ResourceAllocationError):
            await manager.update_resource_usage("agent-mem", memory=50 * 1024 * 1024, cpu=1.5)

    async def test_get_agent_connection(self, manager, pool):
        """Test getting connection for an agent."""
        # Setup
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        pool.acquire.return_value = mock_conn

        # Allocate resource
        await manager.allocate_resource("agent-xyz")

        # Get connection
        conn = await manager.get_agent_connection("agent-xyz")
        assert conn == mock_conn

        # Non-existent agent should return None
        conn = await manager.get_agent_connection("nonexistent")
        assert conn is None

    async def test_cleanup_stale_resources(self, manager, pool):
        """Test cleanup of stale/timed-out resources."""
        # Create stale resource
        mock_conn = Mock()
        mock_conn.connection_id = "conn-stale"
        pool.acquire.return_value = mock_conn

        resource = await manager.allocate_resource("agent-stale")

        # Manually set allocated_at to past
        resource.allocated_at = datetime.utcnow() - timedelta(hours=2)

        # Run cleanup (agent_timeout = 1 hour by default)
        await manager._cleanup_stale_resources()

        # Resource should be released
        assert resource.state == ResourceState.RELEASED
        assert "agent-stale" not in manager._resources
        pool.release.assert_called_with("conn-stale")

    async def test_force_release_all(self, manager, pool):
        """Test force releasing all resources."""
        # Setup multiple resources
        conns = []
        for i in range(3):
            conn = Mock()
            conn.connection_id = f"conn-{i}"
            conn.set_metadata = Mock()
            conns.append(conn)

        pool.acquire.side_effect = conns

        # Allocate multiple agents
        for i in range(3):
            await manager.allocate_resource(f"agent-{i}")

        # Force release all
        await manager.force_release_all()

        # All resources should be released
        assert len(manager._resources) == 0
        assert len(manager._connection_agents) == 0

        # All connections should be released
        assert pool.release.call_count == 3

    async def test_get_resource_metrics(self, manager, pool):
        """Test getting resource metrics."""
        # Setup resources
        mock_conn = Mock()
        mock_conn.connection_id = "conn-123"
        pool.acquire.return_value = mock_conn

        # Create resources with different states
        resource1 = await manager.allocate_resource("agent-1")
        await manager.activate_resource("agent-1")
        resource1.update_usage(memory=50 * 1024 * 1024, cpu=0.5)

        resource2 = await manager.allocate_resource("agent-2")
        resource2.update_usage(memory=30 * 1024 * 1024, cpu=0.3)

        # Get metrics
        metrics = manager.get_metrics()

        assert metrics["total_agents"] == 2
        assert metrics["active_agents"] == 1
        assert metrics["total_memory_usage"] == 80 * 1024 * 1024
        assert metrics["total_cpu_usage"] == 0.8
        assert metrics["connections_in_use"] == 1

    async def test_resource_allocation_with_preferences(self, manager, pool):
        """Test resource allocation with connection preferences."""
        # Mock connections with metadata
        conn1 = Mock()
        conn1.connection_id = "conn-region-us"
        conn1.get_metadata = Mock(return_value="us-east")
        conn1.set_metadata = Mock()

        conn2 = Mock()
        conn2.connection_id = "conn-region-eu"
        conn2.get_metadata = Mock(return_value="eu-west")
        conn2.set_metadata = Mock()

        # First call returns conn1, second returns conn2 with preference
        pool.acquire.side_effect = [conn1, conn2]

        # Allocate without preference
        resource1 = await manager.allocate_resource("agent-1")
        assert resource1.connection_id == "conn-region-us"

        # Allocate with preference (should try to get matching connection)
        resource2 = await manager.allocate_resource(
            "agent-2", prefer_metadata={"region": "eu-west"}
        )

        # Pool.acquire should be called with preference
        pool.acquire.assert_called_with(prefer_metadata={"region": "eu-west"})


class TestResourceMetrics:
    """Test resource metrics collection."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ResourceMetrics()

        assert metrics.total_allocations == 0
        assert metrics.total_releases == 0
        assert metrics.allocation_failures == 0
        assert metrics.active_agents == 0
        assert metrics.peak_agents == 0

    def test_record_allocation(self):
        """Test recording resource allocation."""
        metrics = ResourceMetrics()

        metrics.record_allocation()
        assert metrics.total_allocations == 1
        assert metrics.active_agents == 1
        assert metrics.peak_agents == 1

        metrics.record_allocation()
        assert metrics.total_allocations == 2
        assert metrics.active_agents == 2
        assert metrics.peak_agents == 2

    def test_record_release(self):
        """Test recording resource release."""
        metrics = ResourceMetrics()

        # Allocate then release
        metrics.record_allocation()
        metrics.record_allocation()
        metrics.record_release()

        assert metrics.total_releases == 1
        assert metrics.active_agents == 1
        assert metrics.peak_agents == 2  # Peak remains

    def test_record_failure(self):
        """Test recording allocation failure."""
        metrics = ResourceMetrics()

        metrics.record_allocation_failure()
        assert metrics.allocation_failures == 1

    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics = ResourceMetrics()

        metrics.record_allocation()
        metrics.record_allocation()
        metrics.record_release()
        metrics.record_allocation_failure()

        summary = metrics.get_summary()

        assert summary["total_allocations"] == 2
        assert summary["total_releases"] == 1
        assert summary["allocation_failures"] == 1
        assert summary["active_agents"] == 1
        assert summary["peak_agents"] == 2
        assert "success_rate" in summary
        assert summary["success_rate"] == 0.67  # 2/3 ~= 0.67
