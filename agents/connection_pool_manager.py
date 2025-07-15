"""
Enhanced Connection Pool Manager for Multi-Agent Systems

Implements WebSocket connection pooling, database connection pooling, and
resource lifecycle management to optimize multi-agent coordination efficiency.
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

import asyncpg
import psutil
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""

    # WebSocket Pool Configuration
    websocket_pool_size: int = 20
    websocket_idle_timeout: int = 300  # seconds
    websocket_max_connections_per_pool: int = 100
    websocket_reconnect_attempts: int = 3
    websocket_reconnect_delay: float = 1.0

    # Database Pool Configuration
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600  # seconds
    db_pool_pre_ping: bool = True

    # Agent Pool Configuration
    agent_pool_size: int = 50
    agent_max_concurrent: int = 20
    agent_task_timeout: int = 60  # seconds
    agent_cleanup_interval: int = 300  # seconds

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_half_open_max_calls: int = 3


@dataclass
class ConnectionMetrics:
    """Metrics for connection pool monitoring."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    max_connection_time: float = 0.0
    throughput_per_second: float = 0.0
    pool_utilization: float = 0.0

    # Historical data for trend analysis
    connection_times: List[float] = field(default_factory=list)
    error_timestamps: List[float] = field(default_factory=list)
    utilization_history: List[float] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker pattern for connection failures."""

    def __init__(self, failure_threshold: int, recovery_timeout: int, half_open_max_calls: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            if self.state == "HALF_OPEN":
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker is HALF_OPEN with max calls exceeded")
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


class WebSocketConnectionPool:
    """Enhanced WebSocket connection pool with lifecycle management."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.connection_pools: Dict[str, List[WebSocket]] = defaultdict(list)
        self.metrics = ConnectionMetrics()

        # Locks for thread safety
        self._connections_lock = threading.Lock()
        self._pools_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

        # Circuit breaker for connection failures
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_failure_threshold,
            self.config.circuit_breaker_recovery_timeout,
            self.config.circuit_breaker_half_open_max_calls,
        )

        # Background tasks
        self._cleanup_task = None
        self._metrics_task = None
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._cleanup_task = threading.Thread(target=self._cleanup_expired_connections, daemon=True)
        self._metrics_task = threading.Thread(target=self._update_metrics, daemon=True)

        self._cleanup_task.start()
        self._metrics_task.start()

    async def get_connection(self, client_id: str, pool_name: str = "default") -> WebSocket:
        """Get a connection from the pool or create a new one."""
        start_time = time.time()

        try:
            connection = await self.circuit_breaker.call(
                self._get_or_create_connection, client_id, pool_name
            )

            # Update metrics
            connection_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.connection_times.append(connection_time)
                self.metrics.avg_connection_time = sum(self.metrics.connection_times) / len(
                    self.metrics.connection_times
                )
                self.metrics.max_connection_time = max(
                    self.metrics.max_connection_time, connection_time
                )
                self.metrics.active_connections += 1

            return connection

        except Exception as e:
            with self._metrics_lock:
                self.metrics.connection_errors += 1
                self.metrics.error_timestamps.append(time.time())

            logger.error(f"Failed to get connection for {client_id}: {e}")
            raise

    async def _get_or_create_connection(self, client_id: str, pool_name: str) -> WebSocket:
        """Internal method to get or create a connection."""
        with self._connections_lock:
            if client_id in self.active_connections:
                return self.active_connections[client_id]

        with self._pools_lock:
            pool = self.connection_pools[pool_name]
            if pool:
                connection = pool.pop()
                with self._connections_lock:
                    self.active_connections[client_id] = connection
                    self.connection_metadata[client_id] = {
                        "pool_name": pool_name,
                        "created_at": time.time(),
                        "last_used": time.time(),
                    }
                return connection

        # Create new connection if pool is empty
        # Note: In real implementation, this would create WebSocket connection
        # For now, we'll return a placeholder
        logger.info(f"Creating new WebSocket connection for {client_id} in pool {pool_name}")
        return None  # Placeholder - would be actual WebSocket connection

    def return_connection(self, client_id: str):
        """Return a connection to the pool."""
        with self._connections_lock:
            if client_id in self.active_connections:
                connection = self.active_connections.pop(client_id)
                metadata = self.connection_metadata.pop(client_id, {})
                pool_name = metadata.get("pool_name", "default")

                with self._pools_lock:
                    pool = self.connection_pools[pool_name]
                    if len(pool) < self.config.websocket_pool_size:
                        pool.append(connection)

                with self._metrics_lock:
                    self.metrics.active_connections -= 1
                    self.metrics.idle_connections += 1

    def _cleanup_expired_connections(self):
        """Background task to clean up expired connections."""
        while True:
            try:
                current_time = time.time()
                expired_clients = []

                with self._connections_lock:
                    for client_id, metadata in self.connection_metadata.items():
                        if (
                            current_time - metadata.get("last_used", 0)
                            > self.config.websocket_idle_timeout
                        ):
                            expired_clients.append(client_id)

                for client_id in expired_clients:
                    self.return_connection(client_id)
                    logger.debug(f"Cleaned up expired connection for {client_id}")

                time.sleep(self.config.agent_cleanup_interval)

            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                time.sleep(60)  # Wait before retrying

    def _update_metrics(self):
        """Background task to update pool metrics."""
        while True:
            try:
                with self._metrics_lock:
                    # Update pool utilization
                    total_capacity = sum(
                        len(pool) + self.config.websocket_pool_size
                        for pool in self.connection_pools.values()
                    )
                    if total_capacity > 0:
                        used_capacity = sum(len(pool) for pool in self.connection_pools.values())
                        self.metrics.pool_utilization = used_capacity / total_capacity

                    # Update utilization history
                    self.metrics.utilization_history.append(self.metrics.pool_utilization)
                    if len(self.metrics.utilization_history) > 100:
                        self.metrics.utilization_history.pop(0)

                    # Calculate throughput
                    recent_connections = [
                        t for t in self.metrics.connection_times if time.time() - t < 60
                    ]
                    self.metrics.throughput_per_second = len(recent_connections) / 60.0

                    # Clean up old metrics data
                    cutoff_time = time.time() - 3600  # Keep 1 hour of data
                    self.metrics.connection_times = [
                        t for t in self.metrics.connection_times if t > cutoff_time
                    ]
                    self.metrics.error_timestamps = [
                        t for t in self.metrics.error_timestamps if t > cutoff_time
                    ]

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                time.sleep(60)

    def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        with self._metrics_lock:
            return self.metrics

    def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed pool status."""
        with self._connections_lock, self._pools_lock:
            return {
                "active_connections": len(self.active_connections),
                "pools": {name: len(pool) for name, pool in self.connection_pools.items()},
                "total_pools": len(self.connection_pools),
                "circuit_breaker_state": self.circuit_breaker.state,
                "metrics": self.metrics,
            }


class DatabaseConnectionPool:
    """Enhanced database connection pool with async support."""

    def __init__(self, database_url: str, config: ConnectionPoolConfig):
        self.database_url = database_url
        self.config = config
        self.pool = None
        self.metrics = ConnectionMetrics()

        self._pool_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_failure_threshold,
            self.config.circuit_breaker_recovery_timeout,
            self.config.circuit_breaker_half_open_max_calls,
        )

    async def initialize(self):
        """Initialize the database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.config.db_pool_size,
                max_size=self.config.db_pool_size + self.config.db_max_overflow,
                max_queries=50000,
                max_inactive_connection_lifetime=self.config.db_pool_recycle,
                command_timeout=self.config.db_pool_timeout,
            )
            logger.info(
                f"Database connection pool initialized with {self.config.db_pool_size} connections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        start_time = time.time()
        connection = None

        try:
            connection = await self.circuit_breaker.call(self.pool.acquire)

            connection_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.connection_times.append(connection_time)
                self.metrics.active_connections += 1

            yield connection

        except Exception as e:
            with self._metrics_lock:
                self.metrics.connection_errors += 1
                self.metrics.error_timestamps.append(time.time())

            logger.error(f"Database connection error: {e}")
            raise

        finally:
            if connection:
                await self.pool.release(connection)
                with self._metrics_lock:
                    self.metrics.active_connections -= 1

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")


class EnhancedConnectionPoolManager:
    """Main connection pool manager coordinating all connection types."""

    def __init__(self, config: ConnectionPoolConfig, database_url: str):
        self.config = config
        self.database_url = database_url

        # Initialize pools
        self.websocket_pool = WebSocketConnectionPool(config)
        self.database_pool = DatabaseConnectionPool(database_url, config)

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(config)

        # Performance metrics
        self.performance_metrics = {}
        self._metrics_lock = threading.Lock()

        logger.info("Enhanced Connection Pool Manager initialized")

    async def initialize(self):
        """Initialize all connection pools."""
        await self.database_pool.initialize()
        logger.info("All connection pools initialized")

    async def get_websocket_connection(
        self, client_id: str, pool_name: str = "default"
    ) -> WebSocket:
        """Get a WebSocket connection from the pool."""
        return await self.websocket_pool.get_connection(client_id, pool_name)

    def return_websocket_connection(self, client_id: str):
        """Return a WebSocket connection to the pool."""
        self.websocket_pool.return_connection(client_id)

    @asynccontextmanager
    async def get_database_connection(self):
        """Get a database connection from the pool."""
        async with self.database_pool.get_connection() as connection:
            yield connection

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        websocket_metrics = self.websocket_pool.get_metrics()
        database_metrics = self.database_pool.metrics
        system_metrics = self.resource_monitor.get_current_metrics()

        return {
            "websocket": websocket_metrics,
            "database": database_metrics,
            "system": system_metrics,
            "pools": {
                "websocket": self.websocket_pool.get_pool_status(),
                "database": {
                    "active_connections": database_metrics.active_connections,
                    "pool_utilization": database_metrics.pool_utilization,
                },
            },
        }

    async def close(self):
        """Close all connection pools."""
        await self.database_pool.close()
        logger.info("All connection pools closed")


class ResourceMonitor:
    """System resource monitoring for connection pools."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.metrics_history = []
        self._lock = threading.Lock()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

    def _monitor_resources(self):
        """Background thread to monitor system resources."""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()

                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_read_mb": disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                    "disk_write_mb": disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                }

                with self._lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 100 metrics (about 100 seconds)
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)

                time.sleep(1)

            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(10)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            if not self.metrics_history:
                return {}

            latest = self.metrics_history[-1]

            # Calculate averages over last 60 seconds
            recent_metrics = [m for m in self.metrics_history if time.time() - m["timestamp"] < 60]

            if recent_metrics:
                avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = latest["cpu_percent"]
                avg_memory = latest["memory_percent"]

            return {
                "current": latest,
                "avg_cpu_60s": avg_cpu,
                "avg_memory_60s": avg_memory,
                "samples": len(recent_metrics),
            }
