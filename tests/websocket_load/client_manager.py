"""WebSocket Client Manager for handling multiple concurrent connections."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a WebSocket connection."""

    connection_id: str
    connected_at: float = field(default_factory=time.time)
    disconnected_at: Optional[float] = None
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: List[str] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    @property
    def connection_duration(self) -> float:
        """Get connection duration in seconds."""
        if self.disconnected_at:
            return self.disconnected_at - self.connected_at
        return time.time() - self.connected_at

    @property
    def average_latency(self) -> float:
        """Get average message latency."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def throughput_sent(self) -> float:
        """Get outgoing throughput in messages per second."""
        duration = self.connection_duration
        return self.messages_sent / duration if duration > 0 else 0.0

    @property
    def throughput_received(self) -> float:
        """Get incoming throughput in messages per second."""
        duration = self.connection_duration
        return self.messages_received / duration if duration > 0 else 0.0


class WebSocketClient:
    """Individual WebSocket client with metrics tracking."""

    def __init__(
        self,
        client_id: str,
        base_url: str = "ws://localhost:8000",
        endpoint: str = "/api/v1/ws",
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
    ):
        """Initialize WebSocket client."""
        self.client_id = client_id
        self.base_url = base_url
        self.endpoint = endpoint
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.stats = ConnectionStats(connection_id=client_id)
        self.is_connected = False
        self.pending_messages: Dict[
            str, float
        ] = {}  # message_id -> timestamp for latency tracking

        # Callbacks
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close

        # Message queue for sending
        self.send_queue: asyncio.Queue = asyncio.Queue()
        self.receive_task: Optional[asyncio.Task] = None
        self.send_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            url = f"{self.base_url}{self.endpoint}/{self.client_id}"
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.stats.connected_at = time.time()

            # Start message handling tasks
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.send_task = asyncio.create_task(self._send_loop())

            logger.debug(f"Client {self.client_id} connected to {url}")
            return True

        except Exception as e:
            logger.error(f"Client {self.client_id} connection failed: {e}")
            self.stats.errors.append(f"Connection failed: {str(e)}")
            return False

    async def disconnect(self):
        """Close WebSocket connection."""
        self.is_connected = False

        # Cancel tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.send_task:
            self.send_task.cancel()

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.stats.disconnected_at = time.time()

        if self.on_close:
            await self.on_close(self)

        logger.debug(f"Client {self.client_id} disconnected")

    async def send_message(self, message: Dict[str, Any], track_latency: bool = True):
        """Queue a message for sending."""
        if not self.is_connected:
            raise RuntimeError(f"Client {self.client_id} is not connected")

        # Add message ID for latency tracking
        if track_latency and "id" not in message:
            message["id"] = str(uuid.uuid4())
            self.pending_messages[message["id"]] = time.time()

        await self.send_queue.put(message)

    async def _send_loop(self):
        """Process outgoing messages from queue."""
        while self.is_connected:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.send_queue.get(), timeout=0.1)

                if self.websocket:
                    message_str = json.dumps(message)
                    await self.websocket.send(message_str)

                    self.stats.messages_sent += 1
                    self.stats.bytes_sent += len(message_str.encode())

                    logger.debug(
                        f"Client {self.client_id} sent: {message.get('type', 'unknown')}"
                    )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Client {self.client_id} send error: {e}")
                self.stats.errors.append(f"Send error: {str(e)}")
                if self.on_error:
                    await self.on_error(self, e)

    async def _receive_loop(self):
        """Process incoming messages."""
        while self.is_connected:
            try:
                if self.websocket:
                    message_str = await self.websocket.recv()
                    message = json.loads(message_str)

                    self.stats.messages_received += 1
                    self.stats.bytes_received += len(message_str.encode())

                    # Track latency if this is a response to our message
                    if "id" in message and message["id"] in self.pending_messages:
                        latency = time.time() - self.pending_messages[message["id"]]
                        self.stats.latencies.append(latency)
                        del self.pending_messages[message["id"]]

                    logger.debug(
                        f"Client {self.client_id} received: {message.get('type', 'unknown')}"
                    )

                    if self.on_message:
                        await self.on_message(self, message)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client {self.client_id} connection closed")
                break
            except Exception as e:
                logger.error(f"Client {self.client_id} receive error: {e}")
                self.stats.errors.append(f"Receive error: {str(e)}")
                if self.on_error:
                    await self.on_error(self, e)


class WebSocketClientManager:
    """Manages multiple WebSocket clients for load testing."""

    def __init__(
        self,
        base_url: str = "ws://localhost:8000",
        endpoint: str = "/api/v1/ws",
        client_prefix: str = "load_test",
    ):
        """Initialize client manager."""
        self.base_url = base_url
        self.endpoint = endpoint
        self.client_prefix = client_prefix
        self.clients: Dict[str, WebSocketClient] = {}
        self.active_clients: Set[str] = set()

        # Global callbacks
        self.global_on_message: Optional[Callable] = None
        self.global_on_error: Optional[Callable] = None
        self.global_on_close: Optional[Callable] = None

    async def create_clients(
        self,
        count: int,
        stagger_delay: float = 0.0,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
    ) -> List[WebSocketClient]:
        """Create multiple WebSocket clients."""
        created_clients = []

        for i in range(count):
            client_id = f"{self.client_prefix}_{uuid.uuid4().hex[:8]}_{i}"

            client = WebSocketClient(
                client_id=client_id,
                base_url=self.base_url,
                endpoint=self.endpoint,
                on_message=on_message or self.global_on_message,
                on_error=on_error or self.global_on_error,
                on_close=on_close or self.global_on_close,
            )

            self.clients[client_id] = client
            created_clients.append(client)

            # Stagger client creation if requested
            if stagger_delay > 0 and i < count - 1:
                await asyncio.sleep(stagger_delay)

        logger.info(f"Created {count} WebSocket clients")
        return created_clients

    async def connect_clients(
        self,
        clients: Optional[List[WebSocketClient]] = None,
        concurrent_limit: int = 50,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, bool]:
        """Connect multiple clients with concurrency control."""
        if clients is None:
            clients = list(self.clients.values())

        results = {}

        # Process clients in batches to avoid overwhelming the server
        for i in range(0, len(clients), concurrent_limit):
            batch = clients[i : i + concurrent_limit]
            tasks = []

            for client in batch:
                task = self._connect_with_retry(client, retry_count, retry_delay)
                tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for client, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Client {client.client_id} connection failed: {result}"
                    )
                    results[client.client_id] = False
                else:
                    results[client.client_id] = result
                    if result:
                        self.active_clients.add(client.client_id)

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Connected {successful}/{len(clients)} clients successfully")

        return results

    async def _connect_with_retry(
        self,
        client: WebSocketClient,
        retry_count: int,
        retry_delay: float,
    ) -> bool:
        """Connect a client with retry logic."""
        for attempt in range(retry_count):
            try:
                success = await client.connect()
                if success:
                    return True

                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(
                    f"Client {client.client_id} connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        return False

    async def disconnect_all(self):
        """Disconnect all active clients."""
        tasks = []

        for client_id in list(self.active_clients):
            client = self.clients.get(client_id)
            if client and client.is_connected:
                tasks.append(client.disconnect())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.active_clients.clear()
        logger.info("Disconnected all clients")

    async def broadcast_message(
        self, message: Dict[str, Any], clients: Optional[List[str]] = None
    ):
        """Send a message to multiple clients."""
        if clients is None:
            clients = list(self.active_clients)

        tasks = []
        for client_id in clients:
            client = self.clients.get(client_id)
            if client and client.is_connected:
                tasks.append(client.send_message(message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = sum(1 for r in results if isinstance(r, Exception))
            if errors:
                logger.warning(
                    f"Failed to send message to {errors}/{len(tasks)} clients"
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics for all clients."""
        total_clients = len(self.clients)
        active_clients = len(self.active_clients)

        total_messages_sent = 0
        total_messages_received = 0
        total_bytes_sent = 0
        total_bytes_received = 0
        total_errors = 0
        all_latencies = []

        for client in self.clients.values():
            total_messages_sent += client.stats.messages_sent
            total_messages_received += client.stats.messages_received
            total_bytes_sent += client.stats.bytes_sent
            total_bytes_received += client.stats.bytes_received
            total_errors += len(client.stats.errors)
            all_latencies.extend(client.stats.latencies)

        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        return {
            "total_clients": total_clients,
            "active_clients": active_clients,
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "total_errors": total_errors,
            "average_latency_ms": avg_latency * 1000,
            "latency_p50_ms": self._percentile(all_latencies, 0.5) * 1000
            if all_latencies
            else 0,
            "latency_p95_ms": self._percentile(all_latencies, 0.95) * 1000
            if all_latencies
            else 0,
            "latency_p99_ms": self._percentile(all_latencies, 0.99) * 1000
            if all_latencies
            else 0,
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """Get a specific client by ID."""
        return self.clients.get(client_id)

    def get_active_clients(self) -> List[WebSocketClient]:
        """Get all active clients."""
        return [self.clients[cid] for cid in self.active_clients if cid in self.clients]
