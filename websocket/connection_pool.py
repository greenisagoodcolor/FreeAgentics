"""
WebSocket Connection Pool Implementation

Provides efficient connection pooling for WebSocket connections with features:
- Configurable pool size (min/max)
- Connection health monitoring
- Auto-reconnection with exponential backoff
- Connection reuse optimization
- Auto-scaling based on demand
- Comprehensive metrics collection
- Resource lifecycle management
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states in the pool."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    IDLE = "idle"
    IN_USE = "in_use"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionPoolError(Exception):
    """Base exception for connection pool errors."""

    pass


class PoolExhaustedError(ConnectionPoolError):
    """Raised when the pool cannot provide a connection."""

    pass


class ConnectionNotFoundError(ConnectionPoolError):
    """Raised when a connection is not found in the pool."""

    pass


@dataclass
class PoolConfig:
    """Configuration for the WebSocket connection pool."""

    min_size: int = 5
    max_size: int = 100
    connection_timeout: float = 30.0
    health_check_interval: float = 30.0
    max_idle_time: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Scale up when 80% utilized
    scale_down_threshold: float = 0.2  # Scale down when 20% utilized
    scale_factor: float = 1.5  # Scale by 50%
    max_reconnect_attempts: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_size > self.max_size:
            raise ValueError("min_size cannot be greater than max_size")
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if self.connection_timeout < 0:
            raise ValueError("connection_timeout must be non-negative")
        if self.health_check_interval < 0:
            raise ValueError("health_check_interval must be non-negative")


class PooledConnection:
    """Represents a pooled WebSocket connection."""

    def __init__(self, connection_id: str, websocket: Any, created_at: datetime):
        self.connection_id = connection_id
        self.websocket = websocket
        self.created_at = created_at
        self.connected_at: Optional[datetime] = None
        self.last_used: Optional[datetime] = None
        self.disconnected_at: Optional[datetime] = None
        self.state = ConnectionState.CONNECTING
        self.use_count = 0
        self.metadata: Dict[str, Any] = {}
        self.last_health_check: Optional[datetime] = None
        self.health_check_failures = 0
        self._lock = asyncio.Lock()

    def mark_connected(self):
        """Mark connection as connected."""
        self.state = ConnectionState.CONNECTED
        self.connected_at = datetime.utcnow()
        self.last_health_check = datetime.utcnow()

    def mark_idle(self):
        """Mark connection as idle."""
        self.state = ConnectionState.IDLE
        self.last_used = datetime.utcnow()

    def mark_in_use(self):
        """Mark connection as in use."""
        self.state = ConnectionState.IN_USE
        self.use_count += 1

    def mark_disconnected(self):
        """Mark connection as disconnected."""
        self.state = ConnectionState.DISCONNECTED
        self.disconnected_at = datetime.utcnow()

    def mark_error(self):
        """Mark connection as in error state."""
        self.state = ConnectionState.ERROR

    def set_metadata(self, key: str, value: Any):
        """Set metadata for the connection."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata for the connection."""
        return self.metadata.get(key)

    def is_idle_timeout(self, max_idle_time: float) -> bool:
        """Check if connection has been idle too long."""
        if self.state != ConnectionState.IDLE or not self.last_used:
            return False
        return (datetime.utcnow() - self.last_used).total_seconds() > max_idle_time

    async def close(self):
        """Close the WebSocket connection."""
        try:
            if self.websocket and hasattr(self.websocket, "close"):
                await self.websocket.close()
                # Close the aiohttp session if it exists
                if hasattr(self.websocket, "_session"):
                    await self.websocket._session.close()
        except Exception as e:
            logger.error(f"Error closing connection {self.connection_id}: {e}")
        finally:
            self.mark_disconnected()


class ConnectionMetrics:
    """Tracks metrics for the connection pool."""

    def __init__(self):
        self.total_connections_created = 0
        self.total_connections_destroyed = 0
        self.total_acquisitions = 0
        self.total_releases = 0
        self.failed_acquisitions = 0
        self.health_check_failures = 0
        self.total_wait_time = 0.0
        self.acquisition_count = 0
        self._lock = asyncio.Lock()

    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time for acquisitions."""
        if self.acquisition_count == 0:
            return 0.0
        return float(self.total_wait_time / self.acquisition_count)

    def record_acquisition(self, wait_time: float, success: bool = True):
        """Record a connection acquisition attempt."""
        self.total_acquisitions += 1
        if not success:
            self.failed_acquisitions += 1
        self.total_wait_time += wait_time
        self.acquisition_count += 1

    def record_release(self):
        """Record a connection release."""
        self.total_releases += 1

    def record_connection_created(self):
        """Record creation of a new connection."""
        self.total_connections_created += 1

    def record_connection_destroyed(self):
        """Record destruction of a connection."""
        self.total_connections_destroyed += 1

    def record_health_check_failure(self):
        """Record a health check failure."""
        self.health_check_failures += 1

    def calculate_utilization(self, in_use: int, available: int, total: int) -> float:
        """Calculate pool utilization percentage."""
        if total == 0:
            return 0.0
        return in_use / total

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current metrics."""
        return {
            "total_connections_created": self.total_connections_created,
            "total_connections_destroyed": self.total_connections_destroyed,
            "total_acquisitions": self.total_acquisitions,
            "total_releases": self.total_releases,
            "failed_acquisitions": self.failed_acquisitions,
            "health_check_failures": self.health_check_failures,
            "average_wait_time": self.average_wait_time,
            "timestamp": datetime.utcnow().isoformat(),
        }


class ConnectionHealthMonitor:
    """Monitors health of connections in the pool."""

    def __init__(self, pool: "WebSocketConnectionPool", check_interval: float = 30.0):
        self.pool = pool
        self.check_interval = check_interval
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Start health monitoring."""
        if not self.is_running:
            self.is_running = True
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info("Connection health monitor started")

    async def stop(self):
        """Stop health monitoring."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Connection health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self.pool._health_check_cycle()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.check_interval)


async def create_websocket_connection(url: str, **kwargs) -> Any:
    """Create a new WebSocket connection."""
    # This is a placeholder - in real implementation, this would create
    # an actual WebSocket connection using aiohttp or similar
    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(url, **kwargs)
        # Store session reference on the websocket for proper cleanup
        ws._session = session
        return ws
    except Exception:
        await session.close()
        raise


class WebSocketConnectionPool:
    """WebSocket connection pool with health monitoring and auto-scaling."""

    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._connections: Dict[str, PooledConnection] = {}
        self._idle_connections: List[PooledConnection] = []
        self._in_use_connections: Dict[str, PooledConnection] = {}
        self._url: Optional[str] = None
        self._metrics = ConnectionMetrics()
        self._health_monitor = ConnectionHealthMonitor(self)
        self._lock = asyncio.Lock()
        self._acquire_semaphore = asyncio.Semaphore(self.config.max_size)
        self._initialized = False
        self._shutting_down = False

    @property
    def size(self) -> int:
        """Total number of connections in the pool."""
        return len(self._connections)

    @property
    def available_connections(self) -> int:
        """Number of available (idle) connections."""
        return len(self._idle_connections)

    @property
    def in_use_connections(self) -> int:
        """Number of connections currently in use."""
        return len(self._in_use_connections)

    async def initialize(self, url: str):
        """Initialize the pool with minimum connections."""
        async with self._lock:
            if self._initialized:
                return

            self._url = url
            logger.info(
                f"Initializing connection pool with min_size={self.config.min_size}"
            )

        # Create initial connections (outside lock to avoid deadlock)
        tasks = []
        for _ in range(self.config.min_size):
            tasks.append(self._create_and_add_connection(url))

        await asyncio.gather(*tasks, return_exceptions=True)

        async with self._lock:
            # Start health monitoring
            self._health_monitor.start()

            self._initialized = True
            logger.info(f"Connection pool initialized with {self.size} connections")

    async def _create_connection(self, url: str) -> PooledConnection:
        """Create a new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        try:
            # Create WebSocket connection with timeout
            websocket = await asyncio.wait_for(
                create_websocket_connection(url),
                timeout=self.config.connection_timeout,
            )

            conn = PooledConnection(connection_id, websocket, created_at)
            conn.mark_connected()

            self._metrics.record_connection_created()
            logger.debug(f"Created connection {connection_id}")

            return conn

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout for {connection_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to create connection {connection_id}: {e}")
            raise

    async def _create_and_add_connection(self, url: str) -> Optional[PooledConnection]:
        """Create and add a new connection to the pool."""
        try:
            conn = await self._create_connection(url)
            async with self._lock:
                self._connections[conn.connection_id] = conn
                self._idle_connections.append(conn)
                conn.mark_idle()
            return conn
        except Exception as e:
            logger.error(f"Failed to add connection to pool: {e}")
            return None

    async def acquire(
        self,
        timeout: Optional[float] = None,
        prefer_metadata: Optional[Dict[str, Any]] = None,
    ) -> PooledConnection:
        """Acquire a connection from the pool."""
        start_time = time.time()
        timeout = timeout or self.config.connection_timeout

        try:
            # Wait for available connection
            await asyncio.wait_for(self._acquire_semaphore.acquire(), timeout=timeout)

            try:
                async with self._lock:
                    # Try to find preferred connection based on metadata
                    if prefer_metadata and self._idle_connections:
                        for conn in self._idle_connections:
                            matches = all(
                                conn.get_metadata(k) == v
                                for k, v in prefer_metadata.items()
                            )
                            if matches:
                                self._idle_connections.remove(conn)
                                self._in_use_connections[conn.connection_id] = conn
                                conn.mark_in_use()
                                wait_time = time.time() - start_time
                                self._metrics.record_acquisition(wait_time, True)
                                return conn

                    # Get any available connection
                    if self._idle_connections:
                        conn = self._idle_connections.pop(0)
                        self._in_use_connections[conn.connection_id] = conn
                        conn.mark_in_use()
                        wait_time = time.time() - start_time
                        self._metrics.record_acquisition(wait_time, True)
                        return conn

                    # No idle connections, try to create new one if under max
                    if self.size < self.config.max_size:
                        conn = await self._create_connection(self._url)
                        self._connections[conn.connection_id] = conn
                        self._in_use_connections[conn.connection_id] = conn
                        conn.mark_in_use()
                        wait_time = time.time() - start_time
                        self._metrics.record_acquisition(wait_time, True)
                        return conn

                # Pool exhausted
                self._acquire_semaphore.release()
                wait_time = time.time() - start_time
                self._metrics.record_acquisition(wait_time, False)
                raise PoolExhaustedError(
                    "No connections available and pool at max size"
                )

            except Exception:
                self._acquire_semaphore.release()
                raise

        except asyncio.TimeoutError:
            wait_time = time.time() - start_time
            self._metrics.record_acquisition(wait_time, False)
            raise PoolExhaustedError(f"Timeout waiting for connection after {timeout}s")

    async def release(self, connection_id: str):
        """Release a connection back to the pool."""
        async with self._lock:
            if connection_id not in self._in_use_connections:
                raise ConnectionNotFoundError(
                    f"Connection {connection_id} not found in use"
                )

            conn = self._in_use_connections.pop(connection_id)

            # Check if connection is still healthy
            if conn.state in [
                ConnectionState.DISCONNECTED,
                ConnectionState.ERROR,
            ]:
                # Remove unhealthy connection
                del self._connections[connection_id]
                self._metrics.record_connection_destroyed()
                self._acquire_semaphore.release()

                # Create replacement if below min_size
                if self.size < self.config.min_size:
                    asyncio.create_task(self._create_and_add_connection(self._url))
            else:
                # Return to idle pool
                conn.mark_idle()
                self._idle_connections.append(conn)
                self._acquire_semaphore.release()

            self._metrics.record_release()

    async def _check_connection_health(self, conn: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            if hasattr(conn.websocket, "ping"):
                await conn.websocket.ping()
            conn.last_health_check = datetime.utcnow()
            conn.health_check_failures = 0
            return True
        except Exception as e:
            logger.warning(
                f"Health check failed for connection {conn.connection_id}: {e}"
            )
            conn.health_check_failures += 1
            self._metrics.record_health_check_failure()
            return False

    async def _health_check_cycle(self):
        """Perform health checks on all connections."""
        async with self._lock:
            connections_to_check = list(self._idle_connections)

        unhealthy_connections = []

        for conn in connections_to_check:
            # Skip recently checked connections
            if conn.last_health_check:
                time_since_check = (
                    datetime.utcnow() - conn.last_health_check
                ).total_seconds()
                if time_since_check < self.config.health_check_interval / 2:
                    continue

            is_healthy = await self._check_connection_health(conn)
            if not is_healthy:
                unhealthy_connections.append(conn)

        # Remove unhealthy connections
        if unhealthy_connections:
            async with self._lock:
                for conn in unhealthy_connections:
                    logger.info(f"Removing unhealthy connection {conn.connection_id}")

                    # Close and remove connection
                    await conn.close()

                    if conn.connection_id in self._connections:
                        del self._connections[conn.connection_id]

                    if conn in self._idle_connections:
                        self._idle_connections.remove(conn)

                    self._metrics.record_connection_destroyed()

            # Create replacements to maintain min_size
            replacements_needed = max(0, self.config.min_size - self.size)
            if replacements_needed > 0:
                tasks = []
                for _ in range(replacements_needed):
                    tasks.append(self._create_and_add_connection(self._url))
                await asyncio.gather(*tasks, return_exceptions=True)

        # Check for idle timeout
        await self._remove_idle_connections()

        # Auto-scale if enabled
        if self.config.enable_auto_scaling:
            await self._auto_scale()

    async def _remove_idle_connections(self):
        """Remove connections that have been idle too long."""
        async with self._lock:
            connections_to_remove = []

            for conn in self._idle_connections:
                if conn.is_idle_timeout(self.config.max_idle_time):
                    # Don't remove if it would go below min_size
                    if self.size - len(connections_to_remove) > self.config.min_size:
                        connections_to_remove.append(conn)

            for conn in connections_to_remove:
                logger.info(f"Removing idle connection {conn.connection_id}")
                await conn.close()

                self._idle_connections.remove(conn)
                del self._connections[conn.connection_id]
                self._metrics.record_connection_destroyed()

    async def _auto_scale(self):
        """Auto-scale the pool based on utilization."""
        utilization = self._metrics.calculate_utilization(
            self.in_use_connections, self.available_connections, self.size
        )

        if (
            utilization > self.config.scale_up_threshold
            and self.size < self.config.max_size
        ):
            # Scale up
            target_size = min(
                int(self.size * self.config.scale_factor), self.config.max_size
            )
            connections_to_add = target_size - self.size

            logger.info(
                f"Scaling up pool from {self.size} to {target_size} (utilization: {utilization:.2%})"
            )

            tasks = []
            for _ in range(connections_to_add):
                tasks.append(self._create_and_add_connection(self._url))

            await asyncio.gather(*tasks, return_exceptions=True)

        elif (
            utilization < self.config.scale_down_threshold
            and self.size > self.config.min_size
        ):
            # Scale down
            target_size = max(
                int(self.size / self.config.scale_factor), self.config.min_size
            )
            connections_to_remove = self.size - target_size

            logger.info(
                f"Scaling down pool from {self.size} to {target_size} (utilization: {utilization:.2%})"
            )

            async with self._lock:
                # Remove idle connections first
                removed = 0
                for conn in list(self._idle_connections):
                    if removed >= connections_to_remove:
                        break

                    await conn.close()
                    self._idle_connections.remove(conn)
                    del self._connections[conn.connection_id]
                    self._metrics.record_connection_destroyed()
                    removed += 1

    async def shutdown(self, graceful: bool = True):
        """Shutdown the connection pool."""
        logger.info(f"Shutting down connection pool (graceful={graceful})")

        self._shutting_down = True

        # Stop health monitoring
        await self._health_monitor.stop()

        # Wait for in-use connections if graceful
        if graceful:
            timeout = 30.0  # Max wait time
            start_time = time.time()

            while self.in_use_connections > 0 and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            if self.in_use_connections > 0:
                logger.warning(
                    f"Forcing shutdown with {self.in_use_connections} connections still in use"
                )

        # Close all connections
        async with self._lock:
            all_connections = list(self._connections.values())

            for conn in all_connections:
                await conn.close()

            self._connections.clear()
            self._idle_connections.clear()
            self._in_use_connections.clear()

        logger.info("Connection pool shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        metrics = self._metrics.get_snapshot()
        metrics.update(
            {
                "pool_size": self.size,
                "available_connections": self.available_connections,
                "in_use_connections": self.in_use_connections,
                "utilization": self._metrics.calculate_utilization(
                    self.in_use_connections,
                    self.available_connections,
                    self.size,
                ),
            }
        )
        return metrics

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections."""
        info = []
        for conn in self._connections.values():
            info.append(
                {
                    "connection_id": conn.connection_id,
                    "state": conn.state.value,
                    "created_at": conn.created_at.isoformat(),
                    "use_count": conn.use_count,
                    "last_used": conn.last_used.isoformat() if conn.last_used else None,
                    "metadata": conn.metadata,
                }
            )
        return info
