"""
Unit tests for WebSocket connection pooling and resource management.

This test module follows TDD principles to define the behavior of the WebSocket
connection pool before implementation.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from websocket.connection_pool import (
    ConnectionHealthMonitor,
    ConnectionMetrics,
    ConnectionState,
    PoolConfig,
    PooledConnection,
    PoolExhaustedError,
    WebSocketConnectionPool,
)


@pytest.mark.slow
class TestPoolConfig:
    """Test connection pool configuration."""

    def test_default_config(self):
        """Test default pool configuration values."""
        config = PoolConfig()
        assert config.min_size == 5
        assert config.max_size == 100
        assert config.connection_timeout == 30.0
        assert config.health_check_interval == 30.0
        assert config.max_idle_time == 300.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.enable_auto_scaling is True

    def test_custom_config(self):
        """Test custom pool configuration."""
        config = PoolConfig(
            min_size=10,
            max_size=50,
            connection_timeout=60.0,
            health_check_interval=60.0,
        )
        assert config.min_size == 10
        assert config.max_size == 50
        assert config.connection_timeout == 60.0
        assert config.health_check_interval == 60.0

    def test_config_validation(self):
        """Test configuration validation."""
        # Min size cannot be greater than max size
        with pytest.raises(
            ValueError, match="min_size cannot be greater than max_size"
        ):
            PoolConfig(min_size=100, max_size=50)

        # Negative values should raise errors
        with pytest.raises(ValueError):
            PoolConfig(min_size=-1)

        with pytest.raises(ValueError):
            PoolConfig(connection_timeout=-1.0)


@pytest.mark.slow
class TestConnectionState:
    """Test connection state management."""

    def test_connection_state_transitions(self):
        """Test valid connection state transitions."""
        conn = PooledConnection(
            connection_id="test-1",
            websocket=Mock(),
            created_at=datetime.utcnow(),
        )

        # Initial state should be CONNECTING
        assert conn.state == ConnectionState.CONNECTING

        # Transition to CONNECTED
        conn.mark_connected()
        assert conn.state == ConnectionState.CONNECTED
        assert conn.connected_at is not None

        # Transition to IDLE
        conn.mark_idle()
        assert conn.state == ConnectionState.IDLE
        assert conn.last_used is not None

        # Transition to IN_USE
        conn.mark_in_use()
        assert conn.state == ConnectionState.IN_USE
        assert conn.use_count == 1

        # Back to IDLE
        conn.mark_idle()
        assert conn.state == ConnectionState.IDLE
        assert conn.use_count == 1

        # Transition to DISCONNECTED
        conn.mark_disconnected()
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.disconnected_at is not None

    def test_connection_metadata(self):
        """Test connection metadata management."""
        conn = PooledConnection(
            connection_id="test-1",
            websocket=Mock(),
            created_at=datetime.utcnow(),
        )

        # Add metadata
        conn.set_metadata("user_id", "user123")
        conn.set_metadata("agent_id", "agent456")

        # Retrieve metadata
        assert conn.get_metadata("user_id") == "user123"
        assert conn.get_metadata("agent_id") == "agent456"
        assert conn.get_metadata("nonexistent") is None

        # Update metadata
        conn.set_metadata("user_id", "user789")
        assert conn.get_metadata("user_id") == "user789"


@pytest.mark.asyncio
@pytest.mark.slow
class TestWebSocketConnectionPool:
    """Test WebSocket connection pool functionality."""

    async def test_pool_initialization(self):
        """Test connection pool initialization."""
        config = PoolConfig(min_size=5, max_size=20)
        pool = WebSocketConnectionPool(config)

        assert pool.size == 0
        assert pool.available_connections == 0
        assert pool.in_use_connections == 0
        assert pool.config == config

    async def test_create_connection(self):
        """Test creating a new connection."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=1, max_size=10))

        # Mock WebSocket creation
        mock_websocket = AsyncMock()
        mock_websocket.connect = AsyncMock()
        mock_websocket.ping = AsyncMock()
        mock_websocket.close = AsyncMock()

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            conn = await pool._create_connection("ws://test.example.com")

            assert conn is not None
            assert conn.state == ConnectionState.CONNECTED
            assert conn.websocket == mock_websocket
            # _create_connection doesn't add to pool, so size should still be 0
            assert pool.size == 0

    async def test_acquire_connection(self):
        """Test acquiring a connection from the pool."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=2, max_size=10))

        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.connect = AsyncMock()
        mock_websocket.ping = AsyncMock()
        mock_websocket.close = AsyncMock()

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            # Initialize pool
            await pool.initialize("ws://test.example.com")

            # Acquire connection
            conn = await pool.acquire()

            assert conn is not None
            assert conn.state == ConnectionState.IN_USE
            assert pool.in_use_connections == 1
            assert pool.available_connections == 1  # min_size=2, 1 in use

    async def test_release_connection(self):
        """Test releasing a connection back to the pool."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=1, max_size=10))

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            # Acquire and release
            conn = await pool.acquire()
            assert pool.in_use_connections == 1

            await pool.release(conn.connection_id)
            assert pool.in_use_connections == 0
            assert pool.available_connections == 1
            assert conn.state == ConnectionState.IDLE

    async def test_pool_exhaustion(self):
        """Test behavior when pool is exhausted."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=1, max_size=1))

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            # Acquire the only connection
            await pool.acquire()

            # Try to acquire another (should fail)
            with pytest.raises(PoolExhaustedError):
                await pool.acquire(timeout=0.1)

    async def test_connection_health_check(self):
        """Test connection health checking."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=1, max_size=10))

        mock_websocket = AsyncMock()
        mock_websocket.ping = AsyncMock()

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            conn = await pool.acquire()

            # Simulate successful health check
            mock_websocket.ping.return_value = True
            is_healthy = await pool._check_connection_health(conn)
            assert is_healthy is True

            # Simulate failed health check
            mock_websocket.ping.side_effect = Exception("Connection lost")
            is_healthy = await pool._check_connection_health(conn)
            assert is_healthy is False

    async def test_remove_unhealthy_connections(self):
        """Test removal of unhealthy connections."""
        # Use short health check interval so connections get checked immediately
        config = PoolConfig(min_size=2, max_size=10, health_check_interval=0.1)
        pool = WebSocketConnectionPool(config)

        # Create mock websockets
        healthy_ws = AsyncMock()
        healthy_ws.ping = AsyncMock(return_value=True)

        unhealthy_ws = AsyncMock()
        unhealthy_ws.ping = AsyncMock(side_effect=Exception("Connection lost"))
        unhealthy_ws.close = AsyncMock()

        # Create enough websockets for initial + replacement connections
        replacement_ws = AsyncMock()
        replacement_ws.ping = AsyncMock(return_value=True)
        replacement_ws.close = AsyncMock()

        websockets = [healthy_ws, unhealthy_ws, replacement_ws]

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            side_effect=websockets,
        ):
            await pool.initialize("ws://test.example.com")
            assert pool.size == 2

            # Wait a bit to make connections eligible for health check
            await asyncio.sleep(0.1)

            # Run health check
            await pool._health_check_cycle()

            # Unhealthy connection should be removed and replaced
            # The pool should maintain min_size by creating a replacement
            assert pool.size == 2  # Still 2 connections due to min_size
            unhealthy_ws.close.assert_called_once()

            # Shutdown the pool to avoid pending tasks
            await pool.shutdown()

    async def test_auto_scaling_up(self):
        """Test auto-scaling up when demand is high."""
        config = PoolConfig(
            min_size=2,
            max_size=10,
            enable_auto_scaling=True,
            scale_up_threshold=0.8,  # Scale up when 80% utilized
        )
        pool = WebSocketConnectionPool(config)

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")
            assert pool.size == 2

            # Acquire connections to trigger scaling
            await pool.acquire()
            await pool.acquire()

            # Should trigger scale up (100% utilization > 80% threshold)
            await pool._auto_scale()

            # Pool should have grown
            assert pool.size > 2
            assert pool.size <= config.max_size

    async def test_auto_scaling_down(self):
        """Test auto-scaling down when demand is low."""
        config = PoolConfig(
            min_size=2,
            max_size=10,
            enable_auto_scaling=True,
            scale_down_threshold=0.2,  # Scale down when < 20% utilized
        )
        pool = WebSocketConnectionPool(config)

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            # Start with more than min_size
            await pool.initialize("ws://test.example.com")
            for _ in range(3):  # Add extra connections
                await pool._create_and_add_connection("ws://test.example.com")

            assert pool.size == 5

            # Low utilization should trigger scale down
            await pool._auto_scale()

            # Pool should shrink but not below min_size
            assert pool.size < 5
            assert pool.size >= config.min_size

    async def test_connection_reuse_optimization(self):
        """Test connection reuse based on metadata."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=3, max_size=10))

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            # Acquire connection and set metadata
            conn1 = await pool.acquire()
            conn1.set_metadata("user_id", "user123")
            await pool.release(conn1.connection_id)

            # Acquire with preference for same user
            conn2 = await pool.acquire(prefer_metadata={"user_id": "user123"})

            # Should get the same connection
            assert conn2.connection_id == conn1.connection_id
            assert conn2.get_metadata("user_id") == "user123"

    async def test_connection_timeout(self):
        """Test connection timeout handling."""
        config = PoolConfig(connection_timeout=0.1)  # 100ms timeout
        pool = WebSocketConnectionPool(config)

        # Mock slow connection that takes longer than timeout
        async def slow_connect(url, **kwargs):
            await asyncio.sleep(1)  # Takes 1 second
            return AsyncMock()

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            side_effect=slow_connect,
        ):
            with pytest.raises(asyncio.TimeoutError):
                await pool._create_connection("ws://test.example.com")

    async def test_graceful_shutdown(self):
        """Test graceful shutdown of the pool."""
        pool = WebSocketConnectionPool(PoolConfig(min_size=3, max_size=10))

        mock_websocket = AsyncMock()
        mock_websocket.close = AsyncMock()

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            # Acquire some connections
            await pool.acquire()
            await pool.acquire()

            # Shutdown pool
            await pool.shutdown(graceful=True)

            # All connections should be closed
            assert mock_websocket.close.call_count >= 3  # At least min_size
            assert pool.size == 0


@pytest.mark.slow
class TestConnectionMetrics:
    """Test connection pool metrics collection."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ConnectionMetrics()

        assert metrics.total_connections_created == 0
        assert metrics.total_connections_destroyed == 0
        assert metrics.total_acquisitions == 0
        assert metrics.total_releases == 0
        assert metrics.failed_acquisitions == 0
        assert metrics.health_check_failures == 0
        assert metrics.average_wait_time == 0.0

    def test_record_acquisition(self):
        """Test recording connection acquisition."""
        metrics = ConnectionMetrics()

        # Record successful acquisition
        metrics.record_acquisition(wait_time=0.5, success=True)
        assert metrics.total_acquisitions == 1
        assert metrics.failed_acquisitions == 0
        assert metrics.average_wait_time == 0.5

        # Record failed acquisition
        metrics.record_acquisition(wait_time=1.0, success=False)
        assert metrics.total_acquisitions == 2
        assert metrics.failed_acquisitions == 1
        assert metrics.average_wait_time == 0.75  # (0.5 + 1.0) / 2

    def test_get_pool_utilization(self):
        """Test pool utilization calculation."""
        metrics = ConnectionMetrics()

        utilization = metrics.calculate_utilization(in_use=5, available=10, total=15)
        assert utilization == pytest.approx(0.333, rel=0.01)  # 5/15

    def test_metrics_snapshot(self):
        """Test getting metrics snapshot."""
        metrics = ConnectionMetrics()
        metrics.record_acquisition(wait_time=0.5, success=True)
        metrics.record_connection_created()
        metrics.record_health_check_failure()

        snapshot = metrics.get_snapshot()

        assert snapshot["total_connections_created"] == 1
        assert snapshot["total_acquisitions"] == 1
        assert snapshot["health_check_failures"] == 1
        assert snapshot["average_wait_time"] == 0.5
        assert "timestamp" in snapshot


@pytest.mark.asyncio
@pytest.mark.slow
class TestConnectionHealthMonitor:
    """Test connection health monitoring."""

    async def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        pool = Mock()
        monitor = ConnectionHealthMonitor(pool, check_interval=10.0)

        assert monitor.pool == pool
        assert monitor.check_interval == 10.0
        assert monitor.is_running is False

    async def test_start_monitoring(self):
        """Test starting health monitoring."""
        pool = AsyncMock()
        pool._health_check_cycle = AsyncMock()

        monitor = ConnectionHealthMonitor(pool, check_interval=0.1)

        # Start monitoring
        monitor.start()
        assert monitor.is_running is True

        # Wait for at least one health check
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop()
        assert monitor.is_running is False

        # Health check should have been called
        assert pool._health_check_cycle.call_count >= 1

    async def test_monitor_error_handling(self):
        """Test health monitor error handling."""
        pool = AsyncMock()
        pool._health_check_cycle = AsyncMock(
            side_effect=Exception("Health check error")
        )

        monitor = ConnectionHealthMonitor(pool, check_interval=0.1)

        # Start monitoring (should not crash on errors)
        monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()

        # Should have attempted health checks despite errors
        assert pool._health_check_cycle.call_count >= 1


@pytest.mark.slow
class TestConnectionPoolIntegration:
    """Integration tests for connection pool with real WebSocket-like behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent connection acquisitions."""
        config = PoolConfig(min_size=5, max_size=10)
        pool = WebSocketConnectionPool(config)

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            # Simulate concurrent acquisitions
            tasks = []
            for i in range(8):
                tasks.append(pool.acquire())

            connections = await asyncio.gather(*tasks)

            # All acquisitions should succeed
            assert len(connections) == 8
            assert pool.in_use_connections == 8

            # Pool should have scaled up
            assert pool.size >= 8

    @pytest.mark.asyncio
    async def test_connection_lifecycle_with_errors(self):
        """Test connection lifecycle with various error conditions."""
        # Use short health check interval
        config = PoolConfig(min_size=2, max_size=5, health_check_interval=0.1)
        pool = WebSocketConnectionPool(config)

        # Create websockets that will fail on ping
        healthy_ws = AsyncMock()
        healthy_ws.ping = AsyncMock(return_value=True)
        healthy_ws.close = AsyncMock()

        failing_ws = AsyncMock()
        failing_ws.ping = AsyncMock(side_effect=Exception("Connection failed"))
        failing_ws.close = AsyncMock()

        # Replacement websocket for when the failing one is removed
        replacement_ws = AsyncMock()
        replacement_ws.ping = AsyncMock(return_value=True)
        replacement_ws.close = AsyncMock()

        websockets = [healthy_ws, failing_ws, replacement_ws]

        with patch(
            "websocket.connection_pool.create_websocket_connection",
            side_effect=websockets,
        ):
            await pool.initialize("ws://test.example.com")
            assert pool.size == 2

            # Use connections
            for i in range(5):
                conn = await pool.acquire()
                await asyncio.sleep(0.01)
                await pool.release(conn.connection_id)

            # Wait to make connections eligible for health check
            await asyncio.sleep(0.1)

            # Run health check - should detect failed connection
            await pool._health_check_cycle()

            # Failed connection should be removed and replaced
            assert pool.size >= pool.config.min_size
            # The failing websocket should have been closed
            assert failing_ws.close.called

            # Shutdown the pool to avoid pending tasks
            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_performance_metrics(self):
        """Test pool performance under load."""
        config = PoolConfig(min_size=10, max_size=50)
        pool = WebSocketConnectionPool(config)

        mock_websocket = AsyncMock()
        with patch(
            "websocket.connection_pool.create_websocket_connection",
            return_value=mock_websocket,
        ):
            await pool.initialize("ws://test.example.com")

            start_time = asyncio.get_event_loop().time()

            # Simulate load
            tasks = []
            for i in range(100):

                async def acquire_and_release():
                    conn = await pool.acquire()
                    await asyncio.sleep(0.01)  # Simulate work
                    await pool.release(conn.connection_id)

                tasks.append(acquire_and_release())

            await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Get metrics
            metrics = pool.get_metrics()

            # Verify performance
            assert metrics["total_acquisitions"] == 100
            assert metrics["failed_acquisitions"] == 0
            assert metrics["average_wait_time"] < 0.1  # Should be fast
            assert duration < 2.0  # Should complete quickly with pooling
