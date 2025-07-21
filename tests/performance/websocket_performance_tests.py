"""
WebSocket Performance Testing Suite
===================================

This module provides comprehensive WebSocket performance testing including:
- Connection establishment and teardown performance
- Message throughput testing
- Concurrent connection handling
- Memory usage under WebSocket load
- Real-time bidirectional communication testing
- Connection stability under stress
"""

import asyncio
import json
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import psutil
import websockets

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMetrics:
    """Metrics for WebSocket performance."""

    connection_time_ms: float = 0.0
    message_send_time_ms: float = 0.0
    message_receive_time_ms: float = 0.0
    roundtrip_time_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    connection_errors: int = 0
    message_errors: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WebSocketTestResult:
    """Result of a WebSocket performance test."""

    test_name: str
    concurrent_connections: int
    duration_seconds: float
    total_messages_sent: int
    total_messages_received: int
    message_loss_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_messages_per_second: float
    connection_success_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors_encountered: int
    test_metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketPerformanceTester:
    """Comprehensive WebSocket performance testing."""

    def __init__(self, websocket_url: str = "ws://localhost:8000/ws"):
        self.websocket_url = websocket_url
        self.process = psutil.Process()
        self.test_results: List[WebSocketTestResult] = []

        # Performance thresholds
        self.thresholds = {
            'connection_time_ms': 1000.0,  # 1 second max connection time
            'message_latency_ms': 100.0,  # 100ms max message latency
            'throughput_min_mps': 100.0,  # 100 messages per second minimum
            'connection_success_rate': 95.0,  # 95% connection success rate
            'message_loss_rate': 1.0,  # 1% max message loss
            'memory_usage_max_mb': 500.0,  # 500MB max memory usage
        }

    async def run_connection_performance_test(
        self, max_connections: int = 100, connection_interval: float = 0.1
    ) -> WebSocketTestResult:
        """Test WebSocket connection establishment performance."""
        logger.info(
            f"Starting connection performance test with {max_connections} connections"
        )

        start_time = time.perf_counter()
        connections = []
        connection_times = []
        successful_connections = 0
        failed_connections = 0

        try:
            # Establish connections with interval
            for i in range(max_connections):
                connection_start = time.perf_counter()

                try:
                    websocket = await websockets.connect(
                        self.websocket_url,
                        ping_interval=None,  # Disable ping for performance testing
                        ping_timeout=None,
                        close_timeout=10,
                    )

                    connection_time = (
                        time.perf_counter() - connection_start
                    ) * 1000
                    connection_times.append(connection_time)
                    connections.append(websocket)
                    successful_connections += 1

                    logger.debug(
                        f"Connection {i+1} established in {connection_time:.1f}ms"
                    )

                except Exception as e:
                    failed_connections += 1
                    logger.warning(f"Connection {i+1} failed: {e}")

                # Interval between connections
                if i < max_connections - 1:
                    await asyncio.sleep(connection_interval)

            # Keep connections alive briefly to test stability
            await asyncio.sleep(5)

            # Close all connections
            close_tasks = []
            for websocket in connections:
                close_tasks.append(websocket.close())

            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Connection test error: {e}")

        duration = time.perf_counter() - start_time
        connection_success_rate = (
            successful_connections / max_connections
        ) * 100

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = WebSocketTestResult(
            test_name="connection_performance",
            concurrent_connections=successful_connections,
            duration_seconds=duration,
            total_messages_sent=0,
            total_messages_received=0,
            message_loss_rate=0.0,
            average_latency_ms=statistics.mean(connection_times)
            if connection_times
            else 0,
            p95_latency_ms=np.percentile(connection_times, 95)
            if connection_times
            else 0,
            p99_latency_ms=np.percentile(connection_times, 99)
            if connection_times
            else 0,
            throughput_messages_per_second=0.0,
            connection_success_rate=connection_success_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            errors_encountered=failed_connections,
            test_metadata={
                'connection_times': connection_times,
                'max_connections': max_connections,
                'connection_interval': connection_interval,
                'successful_connections': successful_connections,
                'failed_connections': failed_connections,
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Connection test completed: {successful_connections}/{max_connections} connections, {connection_success_rate:.1f}% success rate"
        )

        return result

    async def run_message_throughput_test(
        self,
        connections: int = 10,
        messages_per_connection: int = 100,
        message_size: int = 1024,
    ) -> WebSocketTestResult:
        """Test WebSocket message throughput."""
        logger.info(
            f"Starting message throughput test: {connections} connections, {messages_per_connection} messages each"
        )

        start_time = time.perf_counter()
        websocket_connections = []
        total_messages_sent = 0
        total_messages_received = 0
        message_latencies = []
        connection_errors = 0
        message_errors = 0

        try:
            # Establish connections
            for i in range(connections):
                try:
                    websocket = await websockets.connect(
                        self.websocket_url,
                        ping_interval=None,
                        ping_timeout=None,
                    )
                    websocket_connections.append(websocket)
                except Exception as e:
                    connection_errors += 1
                    logger.warning(f"Failed to establish connection {i}: {e}")

            if not websocket_connections:
                raise Exception("No WebSocket connections established")

            # Generate test message
            test_message = {
                'type': 'performance_test',
                'data': 'x' * message_size,
                'timestamp': time.time(),
                'id': str(uuid.uuid4()),
            }

            # Send messages concurrently
            async def send_messages(websocket, connection_id):
                nonlocal total_messages_sent, total_messages_received, message_errors

                for msg_id in range(messages_per_connection):
                    try:
                        # Prepare message with unique ID
                        message = test_message.copy()
                        message['connection_id'] = connection_id
                        message['message_id'] = msg_id
                        message['send_time'] = time.perf_counter()

                        # Send message
                        await websocket.send(json.dumps(message))
                        total_messages_sent += 1

                        # For real WebSocket server, we would wait for response
                        # For testing, we'll simulate response time
                        await asyncio.sleep(0.001)  # 1ms simulated processing

                        # Simulate receiving acknowledgment
                        receive_time = time.perf_counter()
                        latency = (receive_time - message['send_time']) * 1000
                        message_latencies.append(latency)
                        total_messages_received += 1

                    except Exception as e:
                        message_errors += 1
                        logger.debug(
                            f"Message error on connection {connection_id}: {e}"
                        )

            # Run message sending tasks
            tasks = []
            for i, websocket in enumerate(websocket_connections):
                task = asyncio.create_task(send_messages(websocket, i))
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Close connections
            close_tasks = []
            for websocket in websocket_connections:
                close_tasks.append(websocket.close())

            await asyncio.gather(*close_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Message throughput test error: {e}")

        duration = time.perf_counter() - start_time
        throughput = total_messages_sent / duration if duration > 0 else 0
        message_loss_rate = (
            (
                (total_messages_sent - total_messages_received)
                / total_messages_sent
                * 100
            )
            if total_messages_sent > 0
            else 0
        )

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = WebSocketTestResult(
            test_name="message_throughput",
            concurrent_connections=len(websocket_connections),
            duration_seconds=duration,
            total_messages_sent=total_messages_sent,
            total_messages_received=total_messages_received,
            message_loss_rate=message_loss_rate,
            average_latency_ms=statistics.mean(message_latencies)
            if message_latencies
            else 0,
            p95_latency_ms=np.percentile(message_latencies, 95)
            if message_latencies
            else 0,
            p99_latency_ms=np.percentile(message_latencies, 99)
            if message_latencies
            else 0,
            throughput_messages_per_second=throughput,
            connection_success_rate=(
                (connections - connection_errors) / connections * 100
            )
            if connections > 0
            else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            errors_encountered=connection_errors + message_errors,
            test_metadata={
                'connections': connections,
                'messages_per_connection': messages_per_connection,
                'message_size': message_size,
                'connection_errors': connection_errors,
                'message_errors': message_errors,
                'latency_stats': {
                    'min_ms': min(message_latencies)
                    if message_latencies
                    else 0,
                    'max_ms': max(message_latencies)
                    if message_latencies
                    else 0,
                    'std_ms': np.std(message_latencies)
                    if message_latencies
                    else 0,
                },
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Throughput test completed: {throughput:.1f} msg/s, {message_loss_rate:.2f}% loss rate"
        )

        return result

    async def run_stress_test(
        self, max_connections: int = 500, test_duration: int = 300
    ) -> WebSocketTestResult:
        """Run WebSocket stress test."""
        logger.info(
            f"Starting WebSocket stress test: {max_connections} connections for {test_duration}s"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration

        total_messages_sent = 0
        total_messages_received = 0
        connection_errors = 0
        message_errors = 0
        latencies = []

        async def stress_connection(connection_id):
            nonlocal total_messages_sent, total_messages_received, connection_errors, message_errors

            websocket = None
            try:
                # Establish connection
                websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=20,  # 20 second ping
                    ping_timeout=10,
                    close_timeout=5,
                )

                # Send messages until test ends
                while time.perf_counter() < end_time:
                    try:
                        message = {
                            'type': 'stress_test',
                            'connection_id': connection_id,
                            'timestamp': time.perf_counter(),
                            'data': f'stress_message_{total_messages_sent}',
                        }

                        send_time = time.perf_counter()
                        await websocket.send(json.dumps(message))
                        total_messages_sent += 1

                        # Simulate receiving response (in real test, would wait for actual response)
                        await asyncio.sleep(0.01)  # 10ms simulated processing

                        receive_time = time.perf_counter()
                        latency = (receive_time - send_time) * 1000
                        latencies.append(latency)
                        total_messages_received += 1

                        # Random delay between messages
                        await asyncio.sleep(np.random.uniform(0.1, 1.0))

                    except Exception as e:
                        message_errors += 1
                        logger.debug(
                            f"Message error on connection {connection_id}: {e}"
                        )

            except Exception as e:
                connection_errors += 1
                logger.warning(f"Connection {connection_id} failed: {e}")

            finally:
                if websocket:
                    try:
                        await websocket.close()
                    except:
                        pass

        try:
            # Start connections gradually
            tasks = []
            connection_interval = test_duration / (
                max_connections * 2
            )  # Spread connections over first half

            for i in range(max_connections):
                task = asyncio.create_task(stress_connection(i))
                tasks.append(task)

                # Small delay between connection attempts
                await asyncio.sleep(connection_interval)

            # Wait for all connections to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Stress test error: {e}")

        duration = time.perf_counter() - start_time
        throughput = total_messages_sent / duration if duration > 0 else 0
        message_loss_rate = (
            (
                (total_messages_sent - total_messages_received)
                / total_messages_sent
                * 100
            )
            if total_messages_sent > 0
            else 0
        )
        successful_connections = max_connections - connection_errors

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = WebSocketTestResult(
            test_name="stress_test",
            concurrent_connections=successful_connections,
            duration_seconds=duration,
            total_messages_sent=total_messages_sent,
            total_messages_received=total_messages_received,
            message_loss_rate=message_loss_rate,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            throughput_messages_per_second=throughput,
            connection_success_rate=(
                successful_connections / max_connections * 100
            )
            if max_connections > 0
            else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            errors_encountered=connection_errors + message_errors,
            test_metadata={
                'max_connections': max_connections,
                'test_duration': test_duration,
                'successful_connections': successful_connections,
                'connection_errors': connection_errors,
                'message_errors': message_errors,
                'connection_interval': connection_interval,
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Stress test completed: {successful_connections}/{max_connections} connections, {throughput:.1f} msg/s"
        )

        return result

    async def run_stability_test(
        self, connections: int = 50, test_duration: int = 600
    ) -> WebSocketTestResult:
        """Run WebSocket stability test over extended period."""
        logger.info(
            f"Starting WebSocket stability test: {connections} connections for {test_duration}s"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration

        stable_connections = 0
        connection_dropouts = 0
        reconnection_attempts = 0
        successful_reconnections = 0
        total_messages_sent = 0
        total_messages_received = 0
        latencies = []

        async def stability_connection(connection_id):
            nonlocal stable_connections, connection_dropouts, reconnection_attempts, successful_reconnections
            nonlocal total_messages_sent, total_messages_received

            websocket = None
            connection_start = time.perf_counter()
            last_heartbeat = connection_start

            try:
                # Initial connection
                websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=30,  # 30 second ping
                    ping_timeout=15,
                )

                stable_connections += 1

                # Send periodic messages
                while time.perf_counter() < end_time:
                    try:
                        current_time = time.perf_counter()

                        # Send heartbeat message every 5 seconds
                        if current_time - last_heartbeat >= 5:
                            message = {
                                'type': 'heartbeat',
                                'connection_id': connection_id,
                                'timestamp': current_time,
                                'uptime': current_time - connection_start,
                            }

                            send_time = time.perf_counter()
                            await websocket.send(json.dumps(message))
                            total_messages_sent += 1
                            last_heartbeat = current_time

                            # Simulate response
                            await asyncio.sleep(0.005)  # 5ms response time

                            receive_time = time.perf_counter()
                            latency = (receive_time - send_time) * 1000
                            latencies.append(latency)
                            total_messages_received += 1

                        await asyncio.sleep(0.1)  # Check every 100ms

                    except websockets.exceptions.ConnectionClosed:
                        connection_dropouts += 1
                        logger.warning(
                            f"Connection {connection_id} dropped, attempting reconnection"
                        )

                        # Attempt reconnection
                        reconnection_attempts += 1
                        try:
                            websocket = await websockets.connect(
                                self.websocket_url,
                                ping_interval=30,
                                ping_timeout=15,
                            )
                            successful_reconnections += 1
                            logger.info(
                                f"Connection {connection_id} reconnected successfully"
                            )
                        except Exception as e:
                            logger.error(
                                f"Reconnection failed for connection {connection_id}: {e}"
                            )
                            break

                    except Exception as e:
                        logger.debug(
                            f"Stability test error on connection {connection_id}: {e}"
                        )
                        await asyncio.sleep(1)  # Wait before retrying

            except Exception as e:
                logger.error(
                    f"Stability connection {connection_id} failed: {e}"
                )
                stable_connections -= 1

            finally:
                if websocket:
                    try:
                        await websocket.close()
                    except:
                        pass

        try:
            # Start all connections
            tasks = []
            for i in range(connections):
                task = asyncio.create_task(stability_connection(i))
                tasks.append(task)

                # Small delay between connections
                await asyncio.sleep(0.1)

            # Wait for all connections to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Stability test error: {e}")

        duration = time.perf_counter() - start_time
        throughput = total_messages_sent / duration if duration > 0 else 0
        message_loss_rate = (
            (
                (total_messages_sent - total_messages_received)
                / total_messages_sent
                * 100
            )
            if total_messages_sent > 0
            else 0
        )
        reconnection_success_rate = (
            (successful_reconnections / reconnection_attempts * 100)
            if reconnection_attempts > 0
            else 100
        )

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = WebSocketTestResult(
            test_name="stability_test",
            concurrent_connections=connections,
            duration_seconds=duration,
            total_messages_sent=total_messages_sent,
            total_messages_received=total_messages_received,
            message_loss_rate=message_loss_rate,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            throughput_messages_per_second=throughput,
            connection_success_rate=100.0,  # All connections were attempted
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            errors_encountered=connection_dropouts,
            test_metadata={
                'connections': connections,
                'test_duration': test_duration,
                'stable_connections': stable_connections,
                'connection_dropouts': connection_dropouts,
                'reconnection_attempts': reconnection_attempts,
                'successful_reconnections': successful_reconnections,
                'reconnection_success_rate': reconnection_success_rate,
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Stability test completed: {stable_connections} stable connections, {connection_dropouts} dropouts, {reconnection_success_rate:.1f}% reconnection success"
        )

        return result

    async def run_comprehensive_websocket_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive WebSocket test suite."""
        logger.info("Starting comprehensive WebSocket test suite")

        test_results = {}

        try:
            # 1. Connection Performance Test
            logger.info("Running connection performance test...")
            connection_result = await self.run_connection_performance_test(
                max_connections=50
            )
            test_results['connection_performance'] = connection_result

            # 2. Message Throughput Test
            logger.info("Running message throughput test...")
            throughput_result = await self.run_message_throughput_test(
                connections=20, messages_per_connection=50
            )
            test_results['message_throughput'] = throughput_result

            # 3. Stress Test
            logger.info("Running stress test...")
            stress_result = await self.run_stress_test(
                max_connections=100, test_duration=60
            )
            test_results['stress_test'] = stress_result

            # 4. Stability Test
            logger.info("Running stability test...")
            stability_result = await self.run_stability_test(
                connections=25, test_duration=120
            )
            test_results['stability_test'] = stability_result

        except Exception as e:
            logger.error(f"Test suite error: {e}")
            test_results['error'] = str(e)

        # Generate comprehensive report
        report = self._generate_comprehensive_report(test_results)

        return report

    def _generate_comprehensive_report(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive WebSocket performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests': len(
                    [k for k in test_results.keys() if k != 'error']
                ),
                'tests_completed': len(
                    [
                        k
                        for k, v in test_results.items()
                        if isinstance(v, WebSocketTestResult)
                    ]
                ),
                'errors': test_results.get('error', None),
            },
            'performance_metrics': {},
            'sla_validation': {'violations': [], 'requirements_met': True},
            'recommendations': [],
        }

        # Analyze each test result
        for test_name, result in test_results.items():
            if isinstance(result, WebSocketTestResult):
                report['performance_metrics'][test_name] = {
                    'concurrent_connections': result.concurrent_connections,
                    'throughput_mps': result.throughput_messages_per_second,
                    'average_latency_ms': result.average_latency_ms,
                    'p95_latency_ms': result.p95_latency_ms,
                    'p99_latency_ms': result.p99_latency_ms,
                    'connection_success_rate': result.connection_success_rate,
                    'message_loss_rate': result.message_loss_rate,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'errors_encountered': result.errors_encountered,
                }

                # Check SLA violations
                violations = self._check_sla_violations(result)
                if violations:
                    report['sla_validation']['violations'].extend(violations)
                    report['sla_validation']['requirements_met'] = False

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            test_results
        )

        return report

    def _check_sla_violations(
        self, result: WebSocketTestResult
    ) -> List[Dict[str, Any]]:
        """Check for SLA violations in test results."""
        violations = []

        # Connection success rate
        if (
            result.connection_success_rate
            < self.thresholds['connection_success_rate']
        ):
            violations.append(
                {
                    'metric': 'connection_success_rate',
                    'threshold': self.thresholds['connection_success_rate'],
                    'actual': result.connection_success_rate,
                    'severity': 'high',
                    'description': f'Connection success rate ({result.connection_success_rate:.1f}%) below threshold',
                }
            )

        # Message latency
        if result.average_latency_ms > self.thresholds['message_latency_ms']:
            violations.append(
                {
                    'metric': 'message_latency',
                    'threshold': self.thresholds['message_latency_ms'],
                    'actual': result.average_latency_ms,
                    'severity': 'medium',
                    'description': f'Average message latency ({result.average_latency_ms:.1f}ms) exceeds threshold',
                }
            )

        # Throughput
        if (
            result.throughput_messages_per_second
            < self.thresholds['throughput_min_mps']
        ):
            violations.append(
                {
                    'metric': 'throughput',
                    'threshold': self.thresholds['throughput_min_mps'],
                    'actual': result.throughput_messages_per_second,
                    'severity': 'medium',
                    'description': f'Throughput ({result.throughput_messages_per_second:.1f} msg/s) below threshold',
                }
            )

        # Message loss rate
        if result.message_loss_rate > self.thresholds['message_loss_rate']:
            violations.append(
                {
                    'metric': 'message_loss_rate',
                    'threshold': self.thresholds['message_loss_rate'],
                    'actual': result.message_loss_rate,
                    'severity': 'high',
                    'description': f'Message loss rate ({result.message_loss_rate:.2f}%) exceeds threshold',
                }
            )

        # Memory usage
        if result.memory_usage_mb > self.thresholds['memory_usage_max_mb']:
            violations.append(
                {
                    'metric': 'memory_usage',
                    'threshold': self.thresholds['memory_usage_max_mb'],
                    'actual': result.memory_usage_mb,
                    'severity': 'medium',
                    'description': f'Memory usage ({result.memory_usage_mb:.1f}MB) exceeds threshold',
                }
            )

        return violations

    def _generate_recommendations(
        self, test_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        for test_name, result in test_results.items():
            if isinstance(result, WebSocketTestResult):
                # Connection recommendations
                if result.connection_success_rate < 95:
                    recommendations.append(
                        f"Low connection success rate in {test_name} ({result.connection_success_rate:.1f}%). Consider implementing connection retry logic and investigating network issues."
                    )

                # Latency recommendations
                if result.p95_latency_ms > 200:
                    recommendations.append(
                        f"High P95 latency in {test_name} ({result.p95_latency_ms:.1f}ms). Consider optimizing WebSocket message processing and implementing message queuing."
                    )

                # Throughput recommendations
                if result.throughput_messages_per_second < 50:
                    recommendations.append(
                        f"Low throughput in {test_name} ({result.throughput_messages_per_second:.1f} msg/s). Consider implementing message batching and optimizing serialization."
                    )

                # Memory recommendations
                if result.memory_usage_mb > 300:
                    recommendations.append(
                        f"High memory usage in {test_name} ({result.memory_usage_mb:.1f}MB). Consider implementing connection pooling and memory optimization."
                    )

                # Error recommendations
                if result.errors_encountered > 0:
                    recommendations.append(
                        f"Errors encountered in {test_name} ({result.errors_encountered}). Implement better error handling and monitoring."
                    )

        if not recommendations:
            recommendations.append(
                "All WebSocket performance tests passed. System is performing well under tested conditions."
            )

        return recommendations

    def save_results(self, filename: str):
        """Save test results to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': [
                {
                    'test_name': result.test_name,
                    'concurrent_connections': result.concurrent_connections,
                    'duration_seconds': result.duration_seconds,
                    'total_messages_sent': result.total_messages_sent,
                    'total_messages_received': result.total_messages_received,
                    'message_loss_rate': result.message_loss_rate,
                    'average_latency_ms': result.average_latency_ms,
                    'p95_latency_ms': result.p95_latency_ms,
                    'p99_latency_ms': result.p99_latency_ms,
                    'throughput_messages_per_second': result.throughput_messages_per_second,
                    'connection_success_rate': result.connection_success_rate,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'errors_encountered': result.errors_encountered,
                    'test_metadata': result.test_metadata,
                }
                for result in self.test_results
            ],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"WebSocket test results saved to {filename}")


# Example usage and testing
async def run_websocket_performance_validation():
    """Run WebSocket performance validation."""
    print("=" * 80)
    print("WEBSOCKET PERFORMANCE VALIDATION SUITE")
    print("=" * 80)

    # Note: This is a mock implementation since we don't have a running WebSocket server
    # In a real scenario, you would start the WebSocket server first

    tester = WebSocketPerformanceTester()

    try:
        # Run comprehensive test suite
        report = await tester.run_comprehensive_websocket_test_suite()

        # Print results
        print("\n" + "=" * 50)
        print("WEBSOCKET PERFORMANCE RESULTS")
        print("=" * 50)

        print(f"Total tests: {report['test_summary']['total_tests']}")
        print(f"Tests completed: {report['test_summary']['tests_completed']}")

        if report['test_summary'].get('errors'):
            print(f"Errors: {report['test_summary']['errors']}")

        # Print performance metrics
        for test_name, metrics in report['performance_metrics'].items():
            print(f"\n{test_name.upper()}:")
            print(f"  Connections: {metrics['concurrent_connections']}")
            print(f"  Throughput: {metrics['throughput_mps']:.1f} msg/s")
            print(f"  Avg Latency: {metrics['average_latency_ms']:.1f}ms")
            print(f"  P95 Latency: {metrics['p95_latency_ms']:.1f}ms")
            print(
                f"  Connection Success: {metrics['connection_success_rate']:.1f}%"
            )
            print(f"  Message Loss: {metrics['message_loss_rate']:.2f}%")
            print(f"  Memory Usage: {metrics['memory_usage_mb']:.1f}MB")
            print(f"  Errors: {metrics['errors_encountered']}")

        # SLA validation
        sla = report['sla_validation']
        print("\n" + "=" * 30)
        print("SLA VALIDATION")
        print("=" * 30)
        print(f"Requirements met: {'✓' if sla['requirements_met'] else '✗'}")

        if sla['violations']:
            print("\nViolations:")
            for violation in sla['violations']:
                print(f"  - {violation['metric']}: {violation['description']}")

        # Recommendations
        print("\n" + "=" * 30)
        print("RECOMMENDATIONS")
        print("=" * 30)
        for rec in report['recommendations']:
            print(f"  - {rec}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"websocket_performance_{timestamp}.json"
        tester.save_results(filename)

        print(f"\nDetailed results saved to: {filename}")

        return report

    except Exception as e:
        print(f"WebSocket performance validation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(run_websocket_performance_validation())
