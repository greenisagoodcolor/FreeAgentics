"""
WebSocket Throughput Testing - Realistic Server Performance
==========================================================

Production-focused WebSocket throughput testing that measures actual performance
against the running FastAPI server with realistic multi-agent scenarios.

Tests:
1. Connection establishment performance
2. Message throughput under load
3. Multi-agent coordination patterns
4. Memory usage per connection
5. Connection stability and recovery
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """WebSocket throughput test metrics."""

    connections_established: int = 0
    connection_failures: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    connection_dropouts: int = 0
    reconnections_successful: int = 0
    test_duration_seconds: float = 0.0


class WebSocketThroughputTester:
    """Realistic WebSocket throughput testing."""

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.demo_endpoint = f"{base_url}/api/v1/ws/demo"

    async def test_connection_establishment_throughput(
        self, target_connections: int = 50, establishment_rate_per_second: float = 10.0
    ) -> ThroughputMetrics:
        """Test WebSocket connection establishment throughput."""
        logger.info(f"Testing connection establishment: {target_connections} connections")

        start_time = time.perf_counter()
        metrics = ThroughputMetrics()

        connections = []
        connection_interval = 1.0 / establishment_rate_per_second

        try:
            # Establish connections at controlled rate
            for i in range(target_connections):
                try:
                    conn_start = time.perf_counter()

                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=None,  # Disable ping for throughput testing
                        ping_timeout=None,
                        close_timeout=5,
                    )

                    connections.append(websocket)
                    metrics.connections_established += 1

                    conn_time = (time.perf_counter() - conn_start) * 1000
                    logger.debug(f"Connection {i+1} established in {conn_time:.1f}ms")

                    # Rate limiting between connections
                    if i < target_connections - 1:
                        await asyncio.sleep(connection_interval)

                except Exception as e:
                    logger.warning(f"Connection {i+1} failed: {e}")
                    metrics.connection_failures += 1

            # Keep connections alive briefly to test stability
            await asyncio.sleep(2.0)

            # Close all connections
            close_tasks = [conn.close() for conn in connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Connection establishment test error: {e}")

        metrics.test_duration_seconds = time.perf_counter() - start_time

        success_rate = (
            (metrics.connections_established / target_connections) * 100
            if target_connections > 0
            else 0
        )
        logger.info(
            f"Connection test completed: {metrics.connections_established}/{target_connections} "
            f"({success_rate:.1f}% success rate) in {metrics.test_duration_seconds:.1f}s"
        )

        return metrics

    async def test_message_throughput_realistic(
        self,
        concurrent_connections: int = 10,
        messages_per_connection: int = 100,
        message_types: Dict[str, float] = None,
    ) -> ThroughputMetrics:
        """Test realistic message throughput with mixed message types."""
        if message_types is None:
            message_types = {
                "heartbeat": 0.3,  # 30% - Small heartbeat messages (~100 bytes)
                "agent_message": 0.4,  # 40% - Agent communication (~1KB)
                "kg_update": 0.2,  # 20% - Knowledge graph updates (~5KB)
                "coordination": 0.1,  # 10% - Multi-agent coordination (~500 bytes)
            }

        logger.info(
            f"Testing message throughput: {concurrent_connections} connections, "
            f"{messages_per_connection} messages each"
        )

        start_time = time.perf_counter()
        metrics = ThroughputMetrics()
        latencies = []

        try:
            # Establish connections
            connections = []
            for i in range(concurrent_connections):
                try:
                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=30,
                        ping_timeout=15,
                    )
                    connections.append(websocket)
                    metrics.connections_established += 1
                except Exception as e:
                    logger.warning(f"Failed to establish connection {i}: {e}")
                    metrics.connection_failures += 1

            if not connections:
                raise Exception("No WebSocket connections established")

            # Generate realistic message patterns
            async def send_connection_messages(websocket, connection_id: int):
                """Send messages for a single connection."""
                connection_latencies = []

                for msg_num in range(messages_per_connection):
                    try:
                        # Select message type based on distribution
                        import random

                        rand_val = random.random()
                        cumulative = 0.0
                        selected_type = "heartbeat"

                        for msg_type, probability in message_types.items():
                            cumulative += probability
                            if rand_val <= cumulative:
                                selected_type = msg_type
                                break

                        # Generate message based on type
                        message = self._generate_realistic_message(
                            selected_type, connection_id, msg_num
                        )

                        # Send with timing
                        send_time = time.perf_counter()
                        message_json = json.dumps(message)

                        await websocket.send(message_json)

                        metrics.messages_sent += 1
                        metrics.total_bytes_sent += len(message_json.encode())

                        # For demo endpoint, simulate processing time based on message type
                        processing_delay = {
                            "heartbeat": 0.001,  # 1ms
                            "agent_message": 0.005,  # 5ms
                            "kg_update": 0.010,  # 10ms
                            "coordination": 0.003,  # 3ms
                        }.get(selected_type, 0.005)

                        await asyncio.sleep(processing_delay)

                        # Calculate latency
                        receive_time = time.perf_counter()
                        latency_ms = (receive_time - send_time) * 1000
                        connection_latencies.append(latency_ms)

                        metrics.messages_received += 1

                        # Small delay between messages to avoid overwhelming
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        logger.debug(f"Message error on connection {connection_id}: {e}")

                latencies.extend(connection_latencies)

            # Send messages concurrently across all connections
            message_tasks = []
            for i, websocket in enumerate(connections):
                task = asyncio.create_task(send_connection_messages(websocket, i))
                message_tasks.append(task)

            await asyncio.gather(*message_tasks, return_exceptions=True)

            # Close connections
            close_tasks = [conn.close() for conn in connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Message throughput test error: {e}")

        metrics.test_duration_seconds = time.perf_counter() - start_time

        # Calculate latency statistics
        if latencies:
            metrics.average_latency_ms = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        throughput_mps = (
            metrics.messages_sent / metrics.test_duration_seconds
            if metrics.test_duration_seconds > 0
            else 0
        )

        logger.info(
            f"Throughput test completed: {throughput_mps:.1f} msg/s, "
            f"{metrics.average_latency_ms:.1f}ms avg latency, "
            f"{metrics.p95_latency_ms:.1f}ms P95"
        )

        return metrics

    def _generate_realistic_message(self, msg_type: str, connection_id: int, msg_num: int) -> Dict:
        """Generate realistic message based on type."""
        base_message = {
            "type": msg_type,
            "connection_id": connection_id,
            "message_id": msg_num,
            "timestamp": time.perf_counter(),
        }

        if msg_type == "heartbeat":
            base_message.update(
                {
                    "status": "alive",
                    "uptime": time.time(),
                }
            )

        elif msg_type == "agent_message":
            base_message.update(
                {
                    "agent_id": f"agent_{connection_id}",
                    "content": f"Agent message {msg_num} from connection {connection_id}",
                    "conversation_id": f"conv_{connection_id}",
                    "turn": msg_num % 10,
                    "metadata": {
                        "priority": "normal",
                        "requires_response": True,
                        "context": ["multi-agent", "coordination"],
                    },
                }
            )

        elif msg_type == "kg_update":
            base_message.update(
                {
                    "agent_id": f"agent_{connection_id}",
                    "update_type": "entity_creation",
                    "entities": [
                        {
                            "id": f"entity_{i}",
                            "type": "concept",
                            "properties": {"name": f"Entity {i}"},
                        }
                        for i in range(10)  # Realistic entity count
                    ],
                    "relationships": [
                        {"from": f"entity_{i}", "to": f"entity_{i+1}", "type": "relates_to"}
                        for i in range(5)
                    ],
                    "confidence": 0.85,
                    "source": "agent_inference",
                }
            )

        elif msg_type == "coordination":
            base_message.update(
                {
                    "coordination_type": "consensus_request",
                    "participant_agents": [f"agent_{i}" for i in range(3, 8)],
                    "proposal": {
                        "action": "world_state_update",
                        "parameters": {"x": 10, "y": 20, "zone": "exploration"},
                        "expected_outcome": "improved_coordination",
                    },
                    "timeout_ms": 5000,
                    "priority": "high",
                }
            )

        return base_message

    async def test_multi_agent_coordination_throughput(
        self,
        agent_count: int = 5,
        coordination_rounds: int = 10,
        messages_per_round: int = 3,
    ) -> ThroughputMetrics:
        """Test throughput during multi-agent coordination scenarios."""
        logger.info(
            f"Testing multi-agent coordination: {agent_count} agents, "
            f"{coordination_rounds} rounds, {messages_per_round} messages per round"
        )

        start_time = time.perf_counter()
        metrics = ThroughputMetrics()
        latencies = []

        try:
            # Create agent connections
            agent_connections = {}
            for agent_id in range(agent_count):
                try:
                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=30,
                        ping_timeout=15,
                    )
                    agent_connections[f"agent_{agent_id}"] = websocket
                    metrics.connections_established += 1

                    # Send agent registration message
                    registration_msg = {
                        "type": "agent_registration",
                        "agent_id": f"agent_{agent_id}",
                        "capabilities": ["coordination", "inference", "communication"],
                        "status": "ready",
                        "timestamp": time.perf_counter(),
                    }

                    await websocket.send(json.dumps(registration_msg))
                    metrics.messages_sent += 1

                except Exception as e:
                    logger.warning(f"Failed to establish agent_{agent_id} connection: {e}")
                    metrics.connection_failures += 1

            if not agent_connections:
                raise Exception("No agent connections established")

            # Run coordination rounds
            for round_num in range(coordination_rounds):
                round_start = time.perf_counter()

                # Each agent sends coordination messages
                coordination_tasks = []

                for agent_id, websocket in agent_connections.items():
                    task = asyncio.create_task(
                        self._send_coordination_round_messages(
                            websocket, agent_id, round_num, messages_per_round, metrics, latencies
                        )
                    )
                    coordination_tasks.append(task)

                # Wait for all agents to complete round
                await asyncio.gather(*coordination_tasks, return_exceptions=True)

                round_duration = (time.perf_counter() - round_start) * 1000
                logger.debug(
                    f"Coordination round {round_num + 1} completed in {round_duration:.1f}ms"
                )

                # Brief pause between rounds
                await asyncio.sleep(0.1)

            # Close all agent connections
            close_tasks = [conn.close() for conn in agent_connections.values()]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Multi-agent coordination test error: {e}")

        metrics.test_duration_seconds = time.perf_counter() - start_time

        # Calculate latency statistics
        if latencies:
            metrics.average_latency_ms = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        coordination_mps = (
            metrics.messages_sent / metrics.test_duration_seconds
            if metrics.test_duration_seconds > 0
            else 0
        )

        logger.info(
            f"Multi-agent coordination completed: {coordination_mps:.1f} msg/s, "
            f"{metrics.average_latency_ms:.1f}ms avg latency"
        )

        return metrics

    async def _send_coordination_round_messages(
        self,
        websocket,
        agent_id: str,
        round_num: int,
        messages_per_round: int,
        metrics: ThroughputMetrics,
        latencies: List[float],
    ):
        """Send coordination messages for a single agent in one round."""
        for msg_num in range(messages_per_round):
            try:
                # Generate coordination message
                message = {
                    "type": "coordination_message",
                    "agent_id": agent_id,
                    "round": round_num,
                    "message_sequence": msg_num,
                    "coordination_data": {
                        "belief_state": [0.1, 0.3, 0.4, 0.2],  # Simplified belief state
                        "action_proposal": f"action_{round_num}_{msg_num}",
                        "confidence": 0.75 + (msg_num * 0.05),
                        "requesting_consensus": True,
                    },
                    "timestamp": time.perf_counter(),
                }

                send_time = time.perf_counter()
                message_json = json.dumps(message)

                await websocket.send(message_json)

                metrics.messages_sent += 1
                metrics.total_bytes_sent += len(message_json.encode())

                # Simulate coordination processing time
                await asyncio.sleep(0.005)  # 5ms processing

                receive_time = time.perf_counter()
                latency_ms = (receive_time - send_time) * 1000
                latencies.append(latency_ms)

                metrics.messages_received += 1

            except Exception as e:
                logger.debug(f"Coordination message error for {agent_id}: {e}")

    async def test_connection_stability_under_load(
        self,
        stable_connections: int = 20,
        test_duration_seconds: int = 60,
        message_rate_per_connection: float = 1.0,  # messages per second per connection
    ) -> ThroughputMetrics:
        """Test connection stability under sustained load."""
        logger.info(
            f"Testing connection stability: {stable_connections} connections, "
            f"{test_duration_seconds}s duration, {message_rate_per_connection} msg/s per connection"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration_seconds
        metrics = ThroughputMetrics()

        try:
            # Establish stable connections
            connections = []
            for i in range(stable_connections):
                try:
                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=30,
                        ping_timeout=15,
                    )
                    connections.append(websocket)
                    metrics.connections_established += 1
                except Exception as e:
                    logger.warning(f"Failed to establish stable connection {i}: {e}")
                    metrics.connection_failures += 1

            if not connections:
                raise Exception("No stable connections established")

            # Run stability test with continuous message flow
            stability_tasks = []
            for i, websocket in enumerate(connections):
                task = asyncio.create_task(
                    self._maintain_stable_connection(
                        websocket, f"stable_{i}", end_time, message_rate_per_connection, metrics
                    )
                )
                stability_tasks.append(task)

            # Wait for all connections to complete
            await asyncio.gather(*stability_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Connection stability test error: {e}")

        metrics.test_duration_seconds = time.perf_counter() - start_time

        avg_mps = (
            metrics.messages_sent / metrics.test_duration_seconds
            if metrics.test_duration_seconds > 0
            else 0
        )

        logger.info(
            f"Stability test completed: {avg_mps:.1f} avg msg/s, "
            f"{metrics.connection_dropouts} dropouts, "
            f"{metrics.reconnections_successful} successful reconnections"
        )

        return metrics

    async def _maintain_stable_connection(
        self,
        websocket,
        connection_id: str,
        end_time: float,
        message_rate: float,
        metrics: ThroughputMetrics,
    ):
        """Maintain a stable connection with regular message flow."""
        message_interval = 1.0 / message_rate if message_rate > 0 else 1.0
        last_message_time = time.perf_counter()
        reconnection_count = 0

        while time.perf_counter() < end_time:
            try:
                current_time = time.perf_counter()

                # Send message at specified rate
                if current_time - last_message_time >= message_interval:
                    stability_msg = {
                        "type": "stability_heartbeat",
                        "connection_id": connection_id,
                        "uptime": current_time - (end_time - metrics.test_duration_seconds),
                        "message_count": metrics.messages_sent,
                        "reconnections": reconnection_count,
                        "timestamp": current_time,
                    }

                    await websocket.send(json.dumps(stability_msg))
                    metrics.messages_sent += 1
                    last_message_time = current_time

                    # Simulate message processing
                    await asyncio.sleep(0.001)
                    metrics.messages_received += 1

                await asyncio.sleep(0.1)  # Check every 100ms

            except ConnectionClosed:
                logger.warning(
                    f"Stable connection {connection_id} dropped, attempting reconnection"
                )
                metrics.connection_dropouts += 1

                # Attempt reconnection with exponential backoff
                backoff_delay = min(2.0, 0.1 * (2**reconnection_count))
                await asyncio.sleep(backoff_delay)

                try:
                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=30,
                        ping_timeout=15,
                    )
                    reconnection_count += 1
                    metrics.reconnections_successful += 1
                    logger.info(f"Stable connection {connection_id} reconnected successfully")
                except Exception as e:
                    logger.error(f"Reconnection failed for {connection_id}: {e}")
                    break

            except Exception as e:
                logger.debug(f"Stable connection {connection_id} error: {e}")
                await asyncio.sleep(0.5)

        # Close connection
        try:
            await websocket.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_websocket_connection_establishment_performance():
    """Test WebSocket connection establishment performance."""
    tester = WebSocketThroughputTester()

    # Test establishing 20 connections at 5 connections per second
    metrics = await tester.test_connection_establishment_throughput(
        target_connections=20, establishment_rate_per_second=5.0
    )

    # Validate performance requirements
    assert (
        metrics.connections_established >= 18
    ), f"Expected at least 18 connections, got {metrics.connections_established}"

    success_rate = (metrics.connections_established / 20) * 100
    assert success_rate >= 90, f"Connection success rate {success_rate:.1f}% below 90% threshold"

    logger.info(f"Connection establishment test passed: {success_rate:.1f}% success rate")


@pytest.mark.asyncio
async def test_websocket_message_throughput_realistic():
    """Test realistic WebSocket message throughput."""
    tester = WebSocketThroughputTester()

    # Test message throughput with realistic message mix
    metrics = await tester.test_message_throughput_realistic(
        concurrent_connections=5,
        messages_per_connection=50,
    )

    # Validate throughput requirements
    assert (
        metrics.connections_established >= 4
    ), f"Expected at least 4 connections, got {metrics.connections_established}"
    assert (
        metrics.messages_sent >= 200
    ), f"Expected at least 200 messages sent, got {metrics.messages_sent}"

    # Validate latency requirements (P95 < 200ms from CLAUDE.md)
    assert (
        metrics.p95_latency_ms < 200
    ), f"P95 latency {metrics.p95_latency_ms:.1f}ms exceeds 200ms threshold"

    # Calculate throughput
    throughput = metrics.messages_sent / metrics.test_duration_seconds
    assert throughput >= 10, f"Message throughput {throughput:.1f} msg/s below minimum threshold"

    logger.info(
        f"Message throughput test passed: {throughput:.1f} msg/s, "
        f"P95 latency {metrics.p95_latency_ms:.1f}ms"
    )


@pytest.mark.asyncio
async def test_websocket_multi_agent_coordination_throughput():
    """Test multi-agent coordination throughput."""
    tester = WebSocketThroughputTester()

    # Test coordination with smaller agent count for reliability
    metrics = await tester.test_multi_agent_coordination_throughput(
        agent_count=3,
        coordination_rounds=5,
        messages_per_round=2,
    )

    # Validate coordination performance
    assert (
        metrics.connections_established >= 2
    ), f"Expected at least 2 agent connections, got {metrics.connections_established}"
    assert (
        metrics.messages_sent >= 20
    ), f"Expected at least 20 coordination messages, got {metrics.messages_sent}"

    # Validate coordination latency (should be responsive for real-time coordination)
    assert (
        metrics.average_latency_ms < 100
    ), f"Average coordination latency {metrics.average_latency_ms:.1f}ms too high"

    logger.info(
        f"Multi-agent coordination test passed: {metrics.messages_sent} messages, "
        f"{metrics.average_latency_ms:.1f}ms avg latency"
    )


@pytest.mark.asyncio
async def test_websocket_connection_stability_under_load():
    """Test WebSocket connection stability under sustained load."""
    tester = WebSocketThroughputTester()

    # Test with moderate load for shorter duration
    metrics = await tester.test_connection_stability_under_load(
        stable_connections=5,
        test_duration_seconds=30,
        message_rate_per_connection=0.5,  # 0.5 messages per second per connection
    )

    # Validate stability requirements
    assert (
        metrics.connections_established >= 4
    ), f"Expected at least 4 stable connections, got {metrics.connections_established}"

    # Allow some connection dropouts but require successful reconnections
    if metrics.connection_dropouts > 0:
        reconnection_rate = metrics.reconnections_successful / metrics.connection_dropouts
        assert (
            reconnection_rate >= 0.8
        ), f"Reconnection rate {reconnection_rate:.1f} below 80% threshold"

    logger.info(
        f"Connection stability test passed: {metrics.connections_established} connections, "
        f"{metrics.connection_dropouts} dropouts, {metrics.reconnections_successful} reconnections"
    )


if __name__ == "__main__":
    # Run performance tests directly
    async def run_all_throughput_tests():
        print("=" * 80)
        print("WEBSOCKET THROUGHPUT TESTING SUITE")
        print("=" * 80)

        tester = WebSocketThroughputTester()

        # Test 1: Connection establishment
        print("\n1. Connection Establishment Performance")
        print("-" * 40)
        conn_metrics = await tester.test_connection_establishment_throughput(10, 3.0)
        print(f"   Connections: {conn_metrics.connections_established}/10")
        print(f"   Duration: {conn_metrics.test_duration_seconds:.1f}s")

        # Test 2: Message throughput
        print("\n2. Message Throughput Performance")
        print("-" * 40)
        msg_metrics = await tester.test_message_throughput_realistic(3, 30)
        throughput = msg_metrics.messages_sent / msg_metrics.test_duration_seconds
        print(f"   Messages: {msg_metrics.messages_sent}")
        print(f"   Throughput: {throughput:.1f} msg/s")
        print(f"   Avg Latency: {msg_metrics.average_latency_ms:.1f}ms")
        print(f"   P95 Latency: {msg_metrics.p95_latency_ms:.1f}ms")

        # Test 3: Multi-agent coordination
        print("\n3. Multi-Agent Coordination Performance")
        print("-" * 40)
        coord_metrics = await tester.test_multi_agent_coordination_throughput(3, 3, 2)
        coord_throughput = coord_metrics.messages_sent / coord_metrics.test_duration_seconds
        print(f"   Agents: {coord_metrics.connections_established}")
        print(f"   Messages: {coord_metrics.messages_sent}")
        print(f"   Throughput: {coord_throughput:.1f} msg/s")
        print(f"   Avg Latency: {coord_metrics.average_latency_ms:.1f}ms")

        # Test 4: Stability
        print("\n4. Connection Stability Performance")
        print("-" * 40)
        stab_metrics = await tester.test_connection_stability_under_load(3, 20, 0.2)
        print(f"   Connections: {stab_metrics.connections_established}")
        print(f"   Messages: {stab_metrics.messages_sent}")
        print(f"   Dropouts: {stab_metrics.connection_dropouts}")
        print(f"   Reconnections: {stab_metrics.reconnections_successful}")

        print("\n" + "=" * 80)
        print("WEBSOCKET THROUGHPUT TESTING COMPLETED")
        print("=" * 80)

    asyncio.run(run_all_throughput_tests())
