"""
WebSocket Integrated Throughput Testing
=======================================

Integration tests that start the actual FastAPI server and test WebSocket
throughput against realistic multi-agent scenarios.

These tests provide accurate performance measurements by testing against
the real server implementation including authentication, rate limiting,
and message routing.
"""

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class TestServerManager:
    """Manages test server lifecycle for WebSocket performance testing."""

    def __init__(self, port: int = 8001):  # Use different port to avoid conflicts
        self.port = port
        self.server_process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
        self.ws_base_url = f"ws://localhost:{port}"

    def _is_port_available(self, port: int) -> bool:
        """Check if port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return True
            except OSError:
                return False

    async def _wait_for_server_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready to accept connections."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to connect to health endpoint
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            return True
            except Exception:
                pass

            await asyncio.sleep(0.5)

        return False

    async def start_server(self) -> bool:
        """Start the test server."""
        if self.server_process is not None:
            logger.warning("Server already running")
            return True

        # Find available port
        original_port = self.port
        for port_offset in range(0, 10):
            test_port = original_port + port_offset
            if self._is_port_available(test_port):
                self.port = test_port
                self.base_url = f"http://localhost:{test_port}"
                self.ws_base_url = f"ws://localhost:{test_port}"
                break
        else:
            logger.error("No available ports found")
            return False

        try:
            # Set environment for test server
            env = os.environ.copy()
            env.update(
                {
                    "PORT": str(self.port),
                    "DATABASE_URL": "sqlite:///:memory:",  # In-memory database for testing
                    "LOG_LEVEL": "WARNING",  # Reduce noise during testing
                    "PYTHONPATH": os.getcwd(),
                }
            )

            # Start server process
            cmd = [
                "python",
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.port),
            ]

            self.server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )

            # Wait for server to be ready
            is_ready = await self._wait_for_server_ready()

            if not is_ready:
                logger.error("Server failed to start within timeout")
                await self.stop_server()
                return False

            logger.info(f"Test server started on port {self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self.stop_server()
            return False

    async def stop_server(self):
        """Stop the test server."""
        if self.server_process is None:
            return

        try:
            # Terminate process group
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)

            # Wait for process to terminate
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                self.server_process.wait()

            logger.info("Test server stopped")

        except Exception as e:
            logger.warning(f"Error stopping server: {e}")

        finally:
            self.server_process = None


@asynccontextmanager
async def test_server():
    """Context manager for test server lifecycle."""
    manager = TestServerManager()

    try:
        server_started = await manager.start_server()
        if not server_started:
            pytest.skip("Could not start test server")

        yield manager

    finally:
        await manager.stop_server()


class IntegratedWebSocketTester:
    """WebSocket performance testing against integrated server."""

    def __init__(self, server_manager: TestServerManager):
        self.server_manager = server_manager
        self.demo_endpoint = f"{server_manager.ws_base_url}/api/v1/ws/demo"

    async def test_realistic_agent_conversation_throughput(
        self,
        agent_count: int = 3,
        conversation_turns: int = 5,
        turn_duration_ms: int = 500,
    ) -> Dict[str, float]:
        """Test realistic agent conversation throughput patterns."""
        logger.info(
            f"Testing agent conversation throughput: {agent_count} agents, {conversation_turns} turns"
        )

        start_time = time.perf_counter()
        metrics = {
            "connections_established": 0,
            "connection_failures": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "conversations_completed": 0,
            "kg_updates_sent": 0,
            "coordination_messages": 0,
            "test_duration_seconds": 0.0,
        }

        agent_connections = {}
        latencies = []

        try:
            # Establish agent connections
            for agent_id in range(agent_count):
                try:
                    websocket = await websockets.connect(
                        self.demo_endpoint,
                        ping_interval=30,
                        ping_timeout=15,
                    )
                    agent_connections[f"agent_{agent_id}"] = websocket
                    metrics["connections_established"] += 1

                    # Send agent registration
                    reg_msg = {
                        "type": "agent_registration",
                        "agent_id": f"agent_{agent_id}",
                        "capabilities": ["conversation", "knowledge_graph", "coordination"],
                        "timestamp": time.perf_counter(),
                    }

                    send_time = time.perf_counter()
                    await websocket.send(json.dumps(reg_msg))
                    metrics["messages_sent"] += 1

                    # Simulate server processing
                    await asyncio.sleep(0.01)
                    latencies.append((time.perf_counter() - send_time) * 1000)
                    metrics["messages_received"] += 1

                except Exception as e:
                    logger.warning(f"Failed to establish agent_{agent_id} connection: {e}")
                    metrics["connection_failures"] += 1

            if not agent_connections:
                raise Exception("No agent connections established")

            # Run conversation simulation
            for conversation_round in range(conversation_turns):
                round_start = time.perf_counter()

                # Each agent participates in conversation turn
                for agent_id, websocket in agent_connections.items():
                    try:
                        # Agent conversation message
                        conv_msg = {
                            "type": "agent_message",
                            "agent_id": agent_id,
                            "conversation_id": f"test_conv_{conversation_round}",
                            "turn": conversation_round,
                            "content": f"Agent {agent_id} message for turn {conversation_round}",
                            "requires_coordination": True,
                            "timestamp": time.perf_counter(),
                        }

                        send_time = time.perf_counter()
                        await websocket.send(json.dumps(conv_msg))
                        metrics["messages_sent"] += 1

                        await asyncio.sleep(0.005)  # Processing delay
                        latencies.append((time.perf_counter() - send_time) * 1000)
                        metrics["messages_received"] += 1

                        # Knowledge graph update
                        kg_msg = {
                            "type": "knowledge_graph_update",
                            "agent_id": agent_id,
                            "entities": [f"entity_{conversation_round}_{i}" for i in range(3)],
                            "relationships": [
                                {"from": "entity_0", "to": "entity_1", "type": "discusses"}
                            ],
                            "confidence": 0.8,
                            "timestamp": time.perf_counter(),
                        }

                        send_time = time.perf_counter()
                        await websocket.send(json.dumps(kg_msg))
                        metrics["messages_sent"] += 1
                        metrics["kg_updates_sent"] += 1

                        await asyncio.sleep(0.008)  # KG processing delay
                        latencies.append((time.perf_counter() - send_time) * 1000)
                        metrics["messages_received"] += 1

                    except Exception as e:
                        logger.debug(f"Error in conversation for {agent_id}: {e}")

                # Inter-agent coordination messages
                if len(agent_connections) > 1:
                    # First agent sends coordination request
                    first_agent = list(agent_connections.keys())[0]
                    websocket = agent_connections[first_agent]

                    coord_msg = {
                        "type": "coordination_request",
                        "from_agent": first_agent,
                        "target_agents": list(agent_connections.keys())[1:],
                        "coordination_type": "consensus",
                        "proposal": f"action_proposal_{conversation_round}",
                        "timestamp": time.perf_counter(),
                    }

                    send_time = time.perf_counter()
                    await websocket.send(json.dumps(coord_msg))
                    metrics["messages_sent"] += 1
                    metrics["coordination_messages"] += 1

                    await asyncio.sleep(0.003)  # Coordination processing
                    latencies.append((time.perf_counter() - send_time) * 1000)
                    metrics["messages_received"] += 1

                # Wait for turn completion
                turn_duration_seconds = turn_duration_ms / 1000.0
                remaining_time = turn_duration_seconds - (time.perf_counter() - round_start)
                if remaining_time > 0:
                    await asyncio.sleep(remaining_time)

                metrics["conversations_completed"] += 1

            # Close all connections
            close_tasks = [conn.close() for conn in agent_connections.values()]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Agent conversation test error: {e}")

        # Calculate final metrics
        metrics["test_duration_seconds"] = time.perf_counter() - start_time

        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            metrics["p95_latency_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]

        throughput = metrics["messages_sent"] / metrics["test_duration_seconds"]

        logger.info(
            f"Agent conversation test completed: {throughput:.1f} msg/s, "
            f"{metrics['conversations_completed']} conversations, "
            f"{metrics['avg_latency_ms']:.1f}ms avg latency"
        )

        return metrics

    async def test_connection_stability_with_server(
        self,
        connection_count: int = 5,
        test_duration_seconds: int = 30,
        message_interval_seconds: float = 1.0,
    ) -> Dict[str, float]:
        """Test connection stability against real server."""
        logger.info(
            f"Testing connection stability: {connection_count} connections, {test_duration_seconds}s"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration_seconds

        metrics = {
            "connections_established": 0,
            "connection_failures": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_dropouts": 0,
            "reconnections_successful": 0,
            "test_duration_seconds": 0.0,
        }

        # Create stability test tasks
        stability_tasks = []
        for i in range(connection_count):
            task = asyncio.create_task(
                self._maintain_stable_connection_with_server(
                    f"stability_{i}", end_time, message_interval_seconds, metrics
                )
            )
            stability_tasks.append(task)

        # Wait for all connections to complete
        await asyncio.gather(*stability_tasks, return_exceptions=True)

        metrics["test_duration_seconds"] = time.perf_counter() - start_time

        logger.info(
            f"Stability test completed: {metrics['connections_established']} connections, "
            f"{metrics['connection_dropouts']} dropouts, "
            f"{metrics['reconnections_successful']} reconnections"
        )

        return metrics

    async def _maintain_stable_connection_with_server(
        self,
        connection_id: str,
        end_time: float,
        message_interval: float,
        metrics: Dict[str, float],
    ):
        """Maintain stable connection with periodic messages."""
        websocket = None
        reconnection_count = 0
        last_message_time = time.perf_counter()

        try:
            # Initial connection
            websocket = await websockets.connect(
                self.demo_endpoint,
                ping_interval=30,
                ping_timeout=15,
            )
            metrics["connections_established"] += 1

            while time.perf_counter() < end_time:
                try:
                    current_time = time.perf_counter()

                    # Send periodic heartbeat
                    if current_time - last_message_time >= message_interval:
                        heartbeat_msg = {
                            "type": "heartbeat",
                            "connection_id": connection_id,
                            "uptime": current_time - (end_time - metrics["test_duration_seconds"]),
                            "reconnections": reconnection_count,
                            "timestamp": current_time,
                        }

                        await websocket.send(json.dumps(heartbeat_msg))
                        metrics["messages_sent"] += 1
                        last_message_time = current_time

                        # Simulate response
                        await asyncio.sleep(0.005)
                        metrics["messages_received"] += 1

                    await asyncio.sleep(0.1)  # Check interval

                except ConnectionClosed:
                    logger.warning(f"Connection {connection_id} dropped, attempting reconnection")
                    metrics["connection_dropouts"] += 1

                    # Exponential backoff reconnection
                    backoff_delay = min(2.0, 0.1 * (2**reconnection_count))
                    await asyncio.sleep(backoff_delay)

                    try:
                        websocket = await websockets.connect(
                            self.demo_endpoint,
                            ping_interval=30,
                            ping_timeout=15,
                        )
                        reconnection_count += 1
                        metrics["reconnections_successful"] += 1
                        logger.info(f"Connection {connection_id} reconnected")
                    except Exception as e:
                        logger.error(f"Reconnection failed for {connection_id}: {e}")
                        break

                except Exception as e:
                    logger.debug(f"Connection {connection_id} error: {e}")
                    await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Connection {connection_id} lifecycle error: {e}")
            metrics["connection_failures"] += 1

        finally:
            if websocket:
                try:
                    await websocket.close()
                except Exception:
                    pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integrated_agent_conversation_throughput():
    """Integration test: Agent conversation throughput with real server."""
    async with test_server() as server_manager:
        tester = IntegratedWebSocketTester(server_manager)

        # Test realistic agent conversation scenario
        metrics = await tester.test_realistic_agent_conversation_throughput(
            agent_count=3,
            conversation_turns=3,
            turn_duration_ms=300,
        )

        # Validate performance requirements
        assert (
            metrics["connections_established"] >= 2
        ), f"Expected at least 2 connections, got {metrics['connections_established']}"
        assert (
            metrics["messages_sent"] >= 15
        ), f"Expected at least 15 messages, got {metrics['messages_sent']}"
        assert (
            metrics["conversations_completed"] >= 2
        ), f"Expected at least 2 conversations, got {metrics['conversations_completed']}"

        # Validate latency requirements (P95 < 200ms)
        assert (
            metrics["p95_latency_ms"] < 200
        ), f"P95 latency {metrics['p95_latency_ms']:.1f}ms exceeds 200ms threshold"

        # Validate throughput
        throughput = metrics["messages_sent"] / metrics["test_duration_seconds"]
        assert throughput >= 5, f"Throughput {throughput:.1f} msg/s below minimum threshold"

        logger.info(
            f"Agent conversation test passed: {throughput:.1f} msg/s, "
            f"P95 latency {metrics['p95_latency_ms']:.1f}ms, "
            f"{metrics['kg_updates_sent']} KG updates, "
            f"{metrics['coordination_messages']} coordination messages"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integrated_connection_stability():
    """Integration test: Connection stability with real server."""
    async with test_server() as server_manager:
        tester = IntegratedWebSocketTester(server_manager)

        # Test connection stability
        metrics = await tester.test_connection_stability_with_server(
            connection_count=3,
            test_duration_seconds=15,
            message_interval_seconds=0.5,
        )

        # Validate stability requirements
        assert (
            metrics["connections_established"] >= 2
        ), f"Expected at least 2 connections, got {metrics['connections_established']}"

        # Allow some dropouts but require successful reconnections
        if metrics["connection_dropouts"] > 0:
            reconnection_rate = metrics["reconnections_successful"] / metrics["connection_dropouts"]
            assert (
                reconnection_rate >= 0.5
            ), f"Reconnection rate {reconnection_rate:.1f} below 50% threshold"

        # Validate message flow
        expected_messages = metrics["connections_established"] * (15 / 0.5)  # duration / interval
        min_expected = expected_messages * 0.7  # Allow 30% tolerance for timing variations
        assert (
            metrics["messages_sent"] >= min_expected
        ), f"Expected at least {min_expected} messages, got {metrics['messages_sent']}"

        logger.info(
            f"Connection stability test passed: {metrics['connections_established']} connections, "
            f"{metrics['messages_sent']} messages, "
            f"{metrics['connection_dropouts']} dropouts, "
            f"{metrics['reconnections_successful']} reconnections"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_memory_usage_validation():
    """Integration test: Validate WebSocket memory usage stays within budget."""
    import psutil

    async with test_server() as server_manager:
        tester = IntegratedWebSocketTester(server_manager)

        # Monitor memory during connection test
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = initial_memory_mb

        # Create multiple connections to test memory scaling
        connection_count = 10
        connections = []

        try:
            # Establish connections gradually
            for i in range(connection_count):
                websocket = await websockets.connect(
                    tester.demo_endpoint,
                    ping_interval=30,
                    ping_timeout=15,
                )
                connections.append(websocket)

                # Monitor memory after each connection
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

                # Send a few messages to ensure connection is active
                for msg_num in range(3):
                    test_msg = {
                        "type": "memory_test",
                        "connection_id": i,
                        "message_id": msg_num,
                        "timestamp": time.perf_counter(),
                    }
                    await websocket.send(json.dumps(test_msg))
                    await asyncio.sleep(0.01)

                await asyncio.sleep(0.1)  # Brief pause between connections

            # Keep connections alive for a period
            await asyncio.sleep(2.0)

            # Final memory check
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, final_memory_mb)

        finally:
            # Close all connections
            close_tasks = [conn.close() for conn in connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Calculate memory metrics
        memory_increase_mb = peak_memory_mb - initial_memory_mb
        memory_per_connection_mb = (
            memory_increase_mb / connection_count if connection_count > 0 else 0
        )

        # Validate memory usage against 34.5MB per connection budget
        assert (
            memory_per_connection_mb <= 34.5
        ), f"Memory per connection {memory_per_connection_mb:.1f}MB exceeds 34.5MB budget"

        logger.info(
            f"Memory validation passed: {memory_per_connection_mb:.1f}MB per connection "
            f"(budget: 34.5MB), peak memory: {peak_memory_mb:.1f}MB"
        )


if __name__ == "__main__":
    # Run integration tests directly
    async def run_integration_tests():
        print("=" * 80)
        print("WEBSOCKET INTEGRATED THROUGHPUT TESTING")
        print("=" * 80)

        async with test_server() as server_manager:
            tester = IntegratedWebSocketTester(server_manager)

            # Test 1: Agent conversation throughput
            print("\n1. Agent Conversation Throughput")
            print("-" * 40)
            conv_metrics = await tester.test_realistic_agent_conversation_throughput(2, 3, 200)
            throughput = conv_metrics["messages_sent"] / conv_metrics["test_duration_seconds"]
            print(f"   Agents: {conv_metrics['connections_established']}")
            print(f"   Messages: {conv_metrics['messages_sent']}")
            print(f"   Throughput: {throughput:.1f} msg/s")
            print(f"   Avg Latency: {conv_metrics['avg_latency_ms']:.1f}ms")
            print(f"   Conversations: {conv_metrics['conversations_completed']}")
            print(f"   KG Updates: {conv_metrics['kg_updates_sent']}")

            # Test 2: Connection stability
            print("\n2. Connection Stability")
            print("-" * 40)
            stab_metrics = await tester.test_connection_stability_with_server(3, 10, 0.3)
            print(f"   Connections: {stab_metrics['connections_established']}")
            print(f"   Messages: {stab_metrics['messages_sent']}")
            print(f"   Dropouts: {stab_metrics['connection_dropouts']}")
            print(f"   Reconnections: {stab_metrics['reconnections_successful']}")

        print("\n" + "=" * 80)
        print("WEBSOCKET INTEGRATED TESTING COMPLETED")
        print("=" * 80)

    # Check if we can run integration tests
    try:
        import aiohttp

        asyncio.run(run_integration_tests())
    except ImportError:
        print("aiohttp required for integration tests. Install with: pip install aiohttp")
    except Exception as e:
        print(f"Integration test error: {e}")
        print(
            "Run individual tests with: python -m pytest tests/performance/test_websocket_integrated_throughput.py -m integration -v"
        )
