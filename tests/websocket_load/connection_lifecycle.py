"""Connection lifecycle management utilities for WebSocket load testing."""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .client_manager import WebSocketClient, WebSocketClientManager
from .message_generators import MessageGenerator
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class ConnectionLifecycleManager:
    """Manages WebSocket connection lifecycles with various patterns."""

    def __init__(
        self,
        client_manager: WebSocketClientManager,
        metrics_collector: MetricsCollector,
    ):
        """Initialize connection lifecycle manager."""
        self.client_manager = client_manager
        self.metrics = metrics_collector
        self.connection_states: Dict[str, ConnectionState] = {}
        self.reconnection_attempts: Dict[str, int] = {}
        self.lifecycle_tasks: Dict[str, asyncio.Task] = {}

        # Lifecycle hooks
        self.on_state_change: Optional[Callable] = None

    async def manage_connection_lifecycle(
        self,
        client: WebSocketClient,
        pattern: str = "persistent",
        **kwargs,
    ):
        """Manage a client's connection lifecycle based on pattern."""
        client_id = client.client_id
        self.connection_states[client_id] = ConnectionState.IDLE

        try:
            if pattern == "persistent":
                await self._persistent_connection(client, **kwargs)
            elif pattern == "intermittent":
                await self._intermittent_connection(client, **kwargs)
            elif pattern == "bursty":
                await self._bursty_connection(client, **kwargs)
            elif pattern == "failover":
                await self._failover_connection(client, **kwargs)
            else:
                raise ValueError(f"Unknown connection pattern: {pattern}")

        except asyncio.CancelledError:
            logger.info(f"Lifecycle management cancelled for {client_id}")
            raise
        except Exception as e:
            logger.error(f"Lifecycle error for {client_id}: {e}")
            self._update_state(client_id, ConnectionState.FAILED)

    async def _persistent_connection(
        self,
        client: WebSocketClient,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
        activity_generator: Optional[MessageGenerator] = None,
        activity_interval: float = 1.0,
    ):
        """Maintain a persistent connection with automatic reconnection."""
        client_id = client.client_id
        self.reconnection_attempts[client_id] = 0

        while self.reconnection_attempts[client_id] < max_reconnect_attempts:
            try:
                # Connect
                self._update_state(client_id, ConnectionState.CONNECTING)
                success = await client.connect()

                if success:
                    self._update_state(client_id, ConnectionState.CONNECTED)
                    self.reconnection_attempts[client_id] = 0
                    self.metrics.record_connection_attempt(True)

                    # Start activity if generator provided
                    if activity_generator:
                        activity_task = asyncio.create_task(
                            self._generate_activity(
                                client, activity_generator, activity_interval
                            )
                        )

                    # Wait for disconnection
                    while client.is_connected:
                        await asyncio.sleep(1.0)

                    # Clean up activity task
                    if activity_generator and "activity_task" in locals():
                        activity_task.cancel()
                        try:
                            await activity_task
                        except asyncio.CancelledError:
                            pass

                    self._update_state(client_id, ConnectionState.DISCONNECTED)

                else:
                    self.metrics.record_connection_attempt(False)
                    self.reconnection_attempts[client_id] += 1

                # Wait before reconnecting
                if (
                    self.reconnection_attempts[client_id]
                    < max_reconnect_attempts
                ):
                    self._update_state(client_id, ConnectionState.RECONNECTING)
                    await asyncio.sleep(reconnect_interval)

            except Exception as e:
                logger.error(
                    f"Persistent connection error for {client_id}: {e}"
                )
                self.reconnection_attempts[client_id] += 1

                if (
                    self.reconnection_attempts[client_id]
                    < max_reconnect_attempts
                ):
                    self._update_state(client_id, ConnectionState.RECONNECTING)
                    await asyncio.sleep(reconnect_interval)

        self._update_state(client_id, ConnectionState.FAILED)
        logger.error(f"Max reconnection attempts reached for {client_id}")

    async def _intermittent_connection(
        self,
        client: WebSocketClient,
        connect_duration: Tuple[float, float] = (10.0, 60.0),
        disconnect_duration: Tuple[float, float] = (5.0, 30.0),
        activity_generator: Optional[MessageGenerator] = None,
        activity_interval: float = 2.0,
        cycles: int = 10,
    ):
        """Create intermittent connections that connect and disconnect periodically."""
        client_id = client.client_id

        for cycle in range(cycles):
            try:
                # Random connection duration
                connect_time = random.uniform(*connect_duration)

                # Connect
                self._update_state(client_id, ConnectionState.CONNECTING)
                success = await client.connect()

                if success:
                    self._update_state(client_id, ConnectionState.CONNECTED)
                    self.metrics.record_connection_attempt(True)

                    # Stay connected for duration
                    start_time = time.time()

                    # Start activity if generator provided
                    if activity_generator:
                        activity_task = asyncio.create_task(
                            self._generate_activity(
                                client, activity_generator, activity_interval
                            )
                        )

                    # Wait for connection duration
                    await asyncio.sleep(connect_time)

                    # Clean up activity task
                    if activity_generator and "activity_task" in locals():
                        activity_task.cancel()
                        try:
                            await activity_task
                        except asyncio.CancelledError:
                            pass

                    # Disconnect
                    self._update_state(
                        client_id, ConnectionState.DISCONNECTING
                    )
                    await client.disconnect()

                    # Record connection duration
                    duration = time.time() - start_time
                    self.metrics.record_connection_closed(duration)

                else:
                    self.metrics.record_connection_attempt(False)

                # Disconnect period
                if cycle < cycles - 1:
                    self._update_state(client_id, ConnectionState.DISCONNECTED)
                    disconnect_time = random.uniform(*disconnect_duration)
                    await asyncio.sleep(disconnect_time)

            except Exception as e:
                logger.error(
                    f"Intermittent connection error for {client_id}: {e}"
                )
                self.metrics.record_error("connection")

        self._update_state(client_id, ConnectionState.DISCONNECTED)

    async def _bursty_connection(
        self,
        client: WebSocketClient,
        burst_size: Tuple[int, int] = (10, 50),
        burst_interval: float = 0.1,
        idle_duration: Tuple[float, float] = (5.0, 20.0),
        message_generator: Optional[MessageGenerator] = None,
        cycles: int = 10,
    ):
        """Create bursty traffic patterns with periods of high activity."""
        client_id = client.client_id

        # Connect once
        self._update_state(client_id, ConnectionState.CONNECTING)
        success = await client.connect()

        if not success:
            self.metrics.record_connection_attempt(False)
            self._update_state(client_id, ConnectionState.FAILED)
            return

        self._update_state(client_id, ConnectionState.CONNECTED)
        self.metrics.record_connection_attempt(True)

        try:
            for cycle in range(cycles):
                # Burst phase
                self._update_state(client_id, ConnectionState.ACTIVE)
                burst_count = random.randint(*burst_size)

                for _ in range(burst_count):
                    if message_generator:
                        message = message_generator.generate()
                        await client.send_message(message)
                        self.metrics.record_message_sent(
                            message.get("type", "unknown"), len(str(message))
                        )

                    await asyncio.sleep(burst_interval)

                # Idle phase
                if cycle < cycles - 1:
                    self._update_state(client_id, ConnectionState.CONNECTED)
                    idle_time = random.uniform(*idle_duration)
                    await asyncio.sleep(idle_time)

        except Exception as e:
            logger.error(f"Bursty connection error for {client_id}: {e}")
            self.metrics.record_error("send")
        finally:
            # Disconnect
            self._update_state(client_id, ConnectionState.DISCONNECTING)
            await client.disconnect()
            self._update_state(client_id, ConnectionState.DISCONNECTED)

    async def _failover_connection(
        self,
        client: WebSocketClient,
        primary_url: str,
        backup_urls: List[str],
        health_check_interval: float = 5.0,
        failover_threshold: int = 3,
        activity_generator: Optional[MessageGenerator] = None,
    ):
        """Implement failover behavior between primary and backup servers."""
        client_id = client.client_id
        current_url_index = -1  # Start with primary
        consecutive_failures = 0

        urls = [primary_url] + backup_urls

        while True:
            try:
                # Select URL (primary or backup)
                if consecutive_failures >= failover_threshold:
                    current_url_index = (current_url_index + 1) % len(urls)
                    consecutive_failures = 0
                    logger.info(
                        f"Failing over {client_id} to {urls[current_url_index]}"
                    )

                # Update client URL
                if current_url_index >= 0:
                    client.base_url = urls[current_url_index]

                # Connect
                self._update_state(client_id, ConnectionState.CONNECTING)
                success = await client.connect()

                if success:
                    self._update_state(client_id, ConnectionState.CONNECTED)
                    self.metrics.record_connection_attempt(True)
                    consecutive_failures = 0

                    # Health check loop
                    while client.is_connected:
                        try:
                            # Send ping for health check
                            await client.send_message({"type": "ping"})
                            await asyncio.sleep(health_check_interval)

                        except Exception:
                            consecutive_failures += 1
                            if consecutive_failures >= failover_threshold:
                                logger.warning(
                                    f"Health check failures for {client_id}, initiating failover"
                                )
                                await client.disconnect()
                                break

                else:
                    self.metrics.record_connection_attempt(False)
                    consecutive_failures += 1

                # Wait before retry/failover
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failover connection error for {client_id}: {e}")
                consecutive_failures += 1
                await asyncio.sleep(1.0)

    async def _generate_activity(
        self,
        client: WebSocketClient,
        generator: MessageGenerator,
        interval: float,
    ):
        """Generate periodic activity for a connected client."""
        try:
            while client.is_connected:
                message = generator.generate()
                await client.send_message(message)
                self.metrics.record_message_sent(
                    message.get("type", "unknown"), len(str(message))
                )
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"Activity generation error for {client.client_id}: {e}"
            )
            self.metrics.record_error("send")

    def _update_state(self, client_id: str, state: ConnectionState):
        """Update connection state and trigger callback."""
        old_state = self.connection_states.get(client_id)
        self.connection_states[client_id] = state

        logger.debug(f"Client {client_id} state: {old_state} -> {state}")

        if self.on_state_change:
            asyncio.create_task(
                self.on_state_change(client_id, old_state, state)
            )

    async def start_lifecycle_management(
        self,
        clients: List[WebSocketClient],
        pattern: str = "persistent",
        **kwargs,
    ):
        """Start lifecycle management for multiple clients."""
        tasks = []

        for client in clients:
            task = asyncio.create_task(
                self.manage_connection_lifecycle(client, pattern, **kwargs)
            )
            self.lifecycle_tasks[client.client_id] = task
            tasks.append(task)

        return tasks

    async def stop_all_lifecycles(self):
        """Stop all lifecycle management tasks."""
        for task in self.lifecycle_tasks.values():
            task.cancel()

        if self.lifecycle_tasks:
            await asyncio.gather(
                *self.lifecycle_tasks.values(), return_exceptions=True
            )

        self.lifecycle_tasks.clear()

    def get_connection_states(self) -> Dict[str, str]:
        """Get current connection states for all clients."""
        return {
            client_id: state.value
            for client_id, state in self.connection_states.items()
        }

    def get_state_distribution(self) -> Dict[str, int]:
        """Get distribution of connection states."""
        distribution = {}

        for state in ConnectionState:
            count = sum(
                1 for s in self.connection_states.values() if s == state
            )
            if count > 0:
                distribution[state.value] = count

        return distribution


class ConnectionPool:
    """Manages a pool of reusable WebSocket connections."""

    def __init__(
        self,
        client_manager: WebSocketClientManager,
        min_size: int = 10,
        max_size: int = 100,
        acquire_timeout: float = 5.0,
    ):
        """Initialize connection pool."""
        self.client_manager = client_manager
        self.min_size = min_size
        self.max_size = max_size
        self.acquire_timeout = acquire_timeout

        self.available: asyncio.Queue[WebSocketClient] = asyncio.Queue()
        self.in_use: Set[str] = set()
        self.total_created = 0

        self._pool_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self):
        """Initialize the connection pool with minimum connections."""
        # Create initial connections
        clients = await self.client_manager.create_clients(self.min_size)

        # Connect them
        results = await self.client_manager.connect_clients(clients)

        # Add successful connections to pool
        for client in clients:
            if results.get(client.client_id, False):
                await self.available.put(client)
                self.total_created += 1

        # Start pool maintenance
        self._running = True
        self._pool_task = asyncio.create_task(self._maintain_pool())

        logger.info(
            f"Connection pool initialized with {self.available.qsize()} connections"
        )

    async def acquire(self) -> Optional[WebSocketClient]:
        """Acquire a connection from the pool."""
        try:
            # Try to get an available connection
            client = await asyncio.wait_for(
                self.available.get(), timeout=self.acquire_timeout
            )

            # Verify it's still connected
            if client.is_connected:
                self.in_use.add(client.client_id)
                return client
            else:
                # Connection is dead, create a new one if possible
                if self.total_created < self.max_size:
                    return await self._create_new_connection()
                else:
                    return None

        except asyncio.TimeoutError:
            # No available connections, try to create new one
            if self.total_created < self.max_size:
                return await self._create_new_connection()
            else:
                logger.warning("Connection pool exhausted")
                return None

    async def release(self, client: WebSocketClient):
        """Release a connection back to the pool."""
        if client.client_id in self.in_use:
            self.in_use.remove(client.client_id)

            if client.is_connected:
                await self.available.put(client)
            else:
                # Connection is dead, don't return to pool
                self.total_created -= 1

    async def _create_new_connection(self) -> Optional[WebSocketClient]:
        """Create a new connection for the pool."""
        clients = await self.client_manager.create_clients(1)
        client = clients[0]

        success = await client.connect()
        if success:
            self.total_created += 1
            self.in_use.add(client.client_id)
            return client
        else:
            return None

    async def _maintain_pool(self):
        """Maintain minimum pool size and health."""
        while self._running:
            try:
                # Check pool health every 5 seconds
                await asyncio.sleep(5.0)

                # Calculate current pool size
                current_size = self.available.qsize() + len(self.in_use)

                # Create new connections if below minimum
                if current_size < self.min_size:
                    needed = self.min_size - current_size
                    clients = await self.client_manager.create_clients(needed)

                    for client in clients:
                        success = await client.connect()
                        if success:
                            await self.available.put(client)
                            self.total_created += 1

                # TODO: Implement connection health checks and cleanup

            except Exception as e:
                logger.error(f"Pool maintenance error: {e}")

    async def shutdown(self):
        """Shutdown the connection pool."""
        self._running = False

        if self._pool_task:
            self._pool_task.cancel()
            try:
                await self._pool_task
            except asyncio.CancelledError:
                pass

        # Disconnect all connections
        all_clients = []

        # Get all available connections
        while not self.available.empty():
            try:
                client = self.available.get_nowait()
                all_clients.append(client)
            except asyncio.QueueEmpty:
                break

        # Disconnect all
        for client in all_clients:
            if client.is_connected:
                await client.disconnect()

        logger.info("Connection pool shut down")
