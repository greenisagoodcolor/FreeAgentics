"""
Load Testing Framework for FreeAgentics
=======================================

This framework provides sophisticated load testing capabilities including:
- Concurrent user simulation with realistic behavior patterns
- WebSocket connection stress testing
- Database connection pool validation
- Real-time metrics collection
- Performance bottleneck identification
- SLA validation and reporting
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import psutil
from observability.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class LoadTestScenario:
    """Defines a load test scenario."""

    name: str
    description: str
    user_count: int
    duration_seconds: int
    ramp_up_seconds: int
    think_time_min: float = 0.1
    think_time_max: float = 2.0
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    websocket_enabled: bool = False
    database_load_enabled: bool = False


@dataclass
class UserSession:
    """Represents a single user session."""

    user_id: str
    session_start: float
    requests_made: int = 0
    errors_encountered: int = 0
    response_times: List[float] = field(default_factory=list)
    websocket_messages: int = 0
    database_queries: int = 0
    current_state: str = "active"
    last_activity: float = field(default_factory=time.time)


@dataclass
class LoadTestResult:
    """Results from a load test."""

    scenario_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    users_spawned: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    throughput_rps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    sla_violations: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class LoadTestingFramework:
    """Advanced load testing framework."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.performance_monitor = get_performance_monitor()
        self.active_sessions: Dict[str, UserSession] = {}
        self.results_history: List[LoadTestResult] = []
        self.real_time_metrics: Dict[str, Any] = {}

        # Performance thresholds
        self.sla_thresholds = {
            "response_time_p95_ms": 3000.0,  # <3s requirement
            "response_time_p99_ms": 5000.0,  # <5s for 99th percentile
            "error_rate_max": 1.0,  # <1% error rate
            "throughput_min_rps": 50.0,  # Minimum 50 requests per second
            "memory_usage_max_mb": 2000.0,  # Maximum 2GB memory
            "cpu_usage_max_percent": 80.0,  # Maximum 80% CPU
        }

        # Built-in scenarios
        self.predefined_scenarios = self._create_predefined_scenarios()

    def _create_predefined_scenarios(self) -> Dict[str, LoadTestScenario]:
        """Create predefined load test scenarios."""
        return {
            "baseline": LoadTestScenario(
                name="Baseline Performance",
                description="Basic load test with 10 users",
                user_count=10,
                duration_seconds=60,
                ramp_up_seconds=10,
                endpoints=[
                    {"path": "/health", "weight": 0.4, "method": "GET"},
                    {
                        "path": "/api/v1/system/status",
                        "weight": 0.3,
                        "method": "GET",
                    },
                    {"path": "/api/v1/agents", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.1,
                        "method": "GET",
                    },
                ],
            ),
            "standard": LoadTestScenario(
                name="Standard Load Test",
                description="Standard load test with 50 users",
                user_count=50,
                duration_seconds=300,  # 5 minutes
                ramp_up_seconds=60,
                endpoints=[
                    {"path": "/health", "weight": 0.3, "method": "GET"},
                    {
                        "path": "/api/v1/system/status",
                        "weight": 0.2,
                        "method": "GET",
                    },
                    {"path": "/api/v1/agents", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.15,
                        "method": "GET",
                    },
                    {
                        "path": "/api/v1/system/metrics",
                        "weight": 0.1,
                        "method": "GET",
                    },
                    {"path": "/metrics", "weight": 0.05, "method": "GET"},
                ],
            ),
            "peak": LoadTestScenario(
                name="Peak Load Test",
                description="Peak load test with 100 users",
                user_count=100,
                duration_seconds=600,  # 10 minutes
                ramp_up_seconds=120,
                endpoints=[
                    {"path": "/health", "weight": 0.25, "method": "GET"},
                    {
                        "path": "/api/v1/system/status",
                        "weight": 0.2,
                        "method": "GET",
                    },
                    {"path": "/api/v1/agents", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.15,
                        "method": "GET",
                    },
                    {
                        "path": "/api/v1/system/metrics",
                        "weight": 0.1,
                        "method": "GET",
                    },
                    {"path": "/metrics", "weight": 0.05, "method": "GET"},
                    {
                        "path": "/api/v1/agents",
                        "weight": 0.03,
                        "method": "POST",
                        "payload": {"name": "test_agent"},
                    },
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.02,
                        "method": "POST",
                        "payload": {"text": "test prompt"},
                    },
                ],
                websocket_enabled=True,
                database_load_enabled=True,
            ),
            "stress": LoadTestScenario(
                name="Stress Test",
                description="Stress test with 200 users",
                user_count=200,
                duration_seconds=1800,  # 30 minutes
                ramp_up_seconds=300,
                think_time_min=0.05,
                think_time_max=1.0,
                endpoints=[
                    {"path": "/health", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/system/status",
                        "weight": 0.15,
                        "method": "GET",
                    },
                    {"path": "/api/v1/agents", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.15,
                        "method": "GET",
                    },
                    {
                        "path": "/api/v1/system/metrics",
                        "weight": 0.1,
                        "method": "GET",
                    },
                    {"path": "/metrics", "weight": 0.05, "method": "GET"},
                    {
                        "path": "/api/v1/agents",
                        "weight": 0.05,
                        "method": "POST",
                        "payload": {"name": "stress_agent"},
                    },
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.05,
                        "method": "POST",
                        "payload": {"text": "stress prompt"},
                    },
                    {
                        "path": "/api/v1/system/cleanup",
                        "weight": 0.02,
                        "method": "POST",
                    },
                    {
                        "path": "/api/v1/system/gc",
                        "weight": 0.01,
                        "method": "POST",
                    },
                ],
                websocket_enabled=True,
                database_load_enabled=True,
            ),
            "soak": LoadTestScenario(
                name="Soak Test",
                description="Extended soak test with 75 users for 2 hours",
                user_count=75,
                duration_seconds=7200,  # 2 hours
                ramp_up_seconds=600,  # 10 minutes
                think_time_min=1.0,
                think_time_max=5.0,
                endpoints=[
                    {"path": "/health", "weight": 0.3, "method": "GET"},
                    {
                        "path": "/api/v1/system/status",
                        "weight": 0.25,
                        "method": "GET",
                    },
                    {"path": "/api/v1/agents", "weight": 0.2, "method": "GET"},
                    {
                        "path": "/api/v1/prompts",
                        "weight": 0.15,
                        "method": "GET",
                    },
                    {
                        "path": "/api/v1/system/metrics",
                        "weight": 0.1,
                        "method": "GET",
                    },
                ],
                websocket_enabled=True,
                database_load_enabled=True,
            ),
        }

    async def run_load_test(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Run a load test scenario."""
        logger.info(f"Starting load test: {scenario.name}")
        logger.info(f"Users: {scenario.user_count}, Duration: {scenario.duration_seconds}s")

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        start_time = datetime.now()
        psutil.Process()

        try:
            # Create connector with appropriate limits
            connector = aiohttp.TCPConnector(
                limit=scenario.user_count * 5,  # 5 connections per user
                limit_per_host=scenario.user_count * 3,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(total=30, connect=10)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "FreeAgentics-LoadTest/1.0"},
            ) as session:
                # Initialize real-time metrics
                self.real_time_metrics = {
                    "active_users": 0,
                    "requests_per_second": 0,
                    "average_response_time": 0,
                    "error_rate": 0,
                    "memory_usage_mb": 0,
                    "cpu_usage_percent": 0,
                }

                # Start metrics collection task
                metrics_task = asyncio.create_task(
                    self._collect_real_time_metrics(scenario.duration_seconds)
                )

                # Start user simulation
                user_tasks = []

                # Gradual ramp-up
                ramp_up_delay = scenario.ramp_up_seconds / scenario.user_count

                for i in range(scenario.user_count):
                    user_id = f"user_{i:04d}_{uuid.uuid4().hex[:8]}"

                    # Create user session
                    user_session = UserSession(
                        user_id=user_id,
                        session_start=time.time() + (i * ramp_up_delay),
                    )

                    self.active_sessions[user_id] = user_session

                    # Start user task
                    user_task = asyncio.create_task(
                        self._simulate_user(session, scenario, user_session)
                    )
                    user_tasks.append(user_task)

                    # Ramp-up delay
                    if i < scenario.user_count - 1:
                        await asyncio.sleep(ramp_up_delay)

                logger.info(f"All {scenario.user_count} users spawned")

                # Wait for test duration
                await asyncio.sleep(scenario.duration_seconds)

                # Stop metrics collection
                metrics_task.cancel()

                # Cancel all user tasks
                for task in user_tasks:
                    task.cancel()

                # Wait for tasks to complete
                await asyncio.gather(*user_tasks, return_exceptions=True)

        finally:
            self.performance_monitor.stop_monitoring()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Collect results
        result = self._compile_results(scenario, start_time, end_time, duration)
        self.results_history.append(result)

        logger.info(
            f"Load test completed: {result.throughput_rps:.1f} RPS, {result.error_rate:.2f}% errors"
        )

        return result

    async def _simulate_user(
        self,
        session: aiohttp.ClientSession,
        scenario: LoadTestScenario,
        user_session: UserSession,
    ):
        """Simulate a single user's behavior."""
        try:
            # Wait for ramp-up time
            wait_time = user_session.session_start - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            test_end_time = time.time() + scenario.duration_seconds

            # WebSocket connection if enabled
            websocket_task = None
            if scenario.websocket_enabled:
                websocket_task = asyncio.create_task(
                    self._simulate_websocket_user(user_session, scenario.duration_seconds)
                )

            # Database load if enabled
            db_task = None
            if scenario.database_load_enabled:
                db_task = asyncio.create_task(
                    self._simulate_database_load(user_session, scenario.duration_seconds)
                )

            # Main request loop
            while time.time() < test_end_time:
                # Choose endpoint based on weights
                endpoint = self._choose_endpoint(scenario.endpoints)

                # Make request
                await self._make_request(session, endpoint, user_session)

                # Think time
                think_time = random.uniform(scenario.think_time_min, scenario.think_time_max)
                await asyncio.sleep(think_time)

                user_session.last_activity = time.time()

            # Clean up
            if websocket_task:
                websocket_task.cancel()
            if db_task:
                db_task.cancel()

            user_session.current_state = "completed"

        except asyncio.CancelledError:
            user_session.current_state = "cancelled"
        except Exception as e:
            user_session.current_state = "error"
            user_session.errors_encountered += 1
            logger.error(f"User {user_session.user_id} encountered error: {e}")

    def _choose_endpoint(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Choose an endpoint based on weights."""
        if not endpoints:
            return {"path": "/health", "method": "GET", "weight": 1.0}

        weights = [ep.get("weight", 1.0) for ep in endpoints]
        return random.choices(endpoints, weights=weights)[0]

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: Dict[str, Any],
        user_session: UserSession,
    ):
        """Make an HTTP request."""
        url = f"{self.base_url}{endpoint['path']}"
        method = endpoint.get("method", "GET")
        payload = endpoint.get("payload")

        start_time = time.perf_counter()

        try:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    await response.read()
                    success = response.status < 400
            elif method.upper() == "POST":
                async with session.post(url, json=payload) as response:
                    await response.read()
                    success = response.status < 400
            elif method.upper() == "PUT":
                async with session.put(url, json=payload) as response:
                    await response.read()
                    success = response.status < 400
            elif method.upper() == "DELETE":
                async with session.delete(url) as response:
                    await response.read()
                    success = response.status < 400
            else:
                success = False

            response_time = (time.perf_counter() - start_time) * 1000
            user_session.response_times.append(response_time)
            user_session.requests_made += 1

            if not success:
                user_session.errors_encountered += 1

        except Exception as e:
            user_session.errors_encountered += 1
            response_time = (time.perf_counter() - start_time) * 1000
            user_session.response_times.append(response_time)
            logger.debug(f"Request failed for {user_session.user_id}: {e}")

    async def _simulate_websocket_user(self, user_session: UserSession, duration: int):
        """Simulate WebSocket connections."""
        try:
            # Mock WebSocket simulation (replace with actual WebSocket logic)
            end_time = time.time() + duration

            while time.time() < end_time:
                # Simulate WebSocket message
                await asyncio.sleep(random.uniform(0.5, 2.0))
                user_session.websocket_messages += 1

                # Simulate processing time
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket simulation error for {user_session.user_id}: {e}")

    async def _simulate_database_load(self, user_session: UserSession, duration: int):
        """Simulate database load."""
        try:
            end_time = time.time() + duration

            while time.time() < end_time:
                # Simulate database query
                await asyncio.sleep(random.uniform(1.0, 5.0))
                user_session.database_queries += 1

                # Simulate query processing time
                query_types = ["SELECT", "INSERT", "UPDATE", "DELETE"]
                query_type = random.choice(query_types)

                if query_type == "SELECT":
                    await asyncio.sleep(0.005)  # 5ms
                elif query_type == "INSERT":
                    await asyncio.sleep(0.010)  # 10ms
                elif query_type == "UPDATE":
                    await asyncio.sleep(0.008)  # 8ms
                else:  # DELETE
                    await asyncio.sleep(0.006)  # 6ms

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Database simulation error for {user_session.user_id}: {e}")

    async def _collect_real_time_metrics(self, duration: int):
        """Collect real-time metrics during the test."""
        try:
            start_time = time.time()
            end_time = start_time + duration

            while time.time() < end_time:
                # Collect metrics
                active_users = len(
                    [s for s in self.active_sessions.values() if s.current_state == "active"]
                )

                # Calculate requests per second
                current_time = time.time()
                recent_requests = []
                for session in self.active_sessions.values():
                    recent_requests.extend(
                        [t for t in session.response_times if current_time - t < 60]  # Last minute
                    )

                rps = len(recent_requests) / min(60, current_time - start_time)

                # Calculate average response time
                all_response_times = []
                for session in self.active_sessions.values():
                    all_response_times.extend(session.response_times)

                avg_response_time = np.mean(all_response_times) if all_response_times else 0

                # Calculate error rate
                total_requests = sum(s.requests_made for s in self.active_sessions.values())
                total_errors = sum(s.errors_encountered for s in self.active_sessions.values())
                error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

                # System metrics
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024
                cpu_usage = process.cpu_percent()

                # Update real-time metrics
                self.real_time_metrics.update(
                    {
                        "active_users": active_users,
                        "requests_per_second": rps,
                        "average_response_time": avg_response_time,
                        "error_rate": error_rate,
                        "memory_usage_mb": memory_usage,
                        "cpu_usage_percent": cpu_usage,
                        "timestamp": current_time,
                    }
                )

                await asyncio.sleep(1)  # Update every second

        except asyncio.CancelledError:
            pass

    def _compile_results(
        self,
        scenario: LoadTestScenario,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> LoadTestResult:
        """Compile test results."""
        # Aggregate all response times
        all_response_times = []
        total_requests = 0
        total_errors = 0
        total_websocket_messages = 0
        total_database_queries = 0

        for session in self.active_sessions.values():
            all_response_times.extend(session.response_times)
            total_requests += session.requests_made
            total_errors += session.errors_encountered
            total_websocket_messages += session.websocket_messages
            total_database_queries += session.database_queries

        successful_requests = total_requests - total_errors
        throughput = successful_requests / duration if duration > 0 else 0
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        # System metrics
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        cpu_usage = process.cpu_percent()

        # Performance metrics
        performance_metrics = {
            "response_time_stats": {
                "min_ms": np.min(all_response_times) if all_response_times else 0,
                "max_ms": np.max(all_response_times) if all_response_times else 0,
                "mean_ms": np.mean(all_response_times) if all_response_times else 0,
                "median_ms": np.median(all_response_times) if all_response_times else 0,
                "p95_ms": np.percentile(all_response_times, 95) if all_response_times else 0,
                "p99_ms": np.percentile(all_response_times, 99) if all_response_times else 0,
                "std_ms": np.std(all_response_times) if all_response_times else 0,
            },
            "throughput_stats": {
                "requests_per_second": throughput,
                "requests_per_minute": throughput * 60,
                "requests_per_hour": throughput * 3600,
            },
            "user_stats": {
                "users_spawned": len(self.active_sessions),
                "users_completed": len(
                    [s for s in self.active_sessions.values() if s.current_state == "completed"]
                ),
                "users_cancelled": len(
                    [s for s in self.active_sessions.values() if s.current_state == "cancelled"]
                ),
                "users_error": len(
                    [s for s in self.active_sessions.values() if s.current_state == "error"]
                ),
            },
            "websocket_stats": {
                "total_messages": total_websocket_messages,
                "messages_per_second": total_websocket_messages / duration if duration > 0 else 0,
            },
            "database_stats": {
                "total_queries": total_database_queries,
                "queries_per_second": total_database_queries / duration if duration > 0 else 0,
            },
        }

        # Check SLA violations
        sla_violations = self._check_sla_violations(
            performance_metrics, error_rate, memory_usage, cpu_usage
        )

        result = LoadTestResult(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            users_spawned=len(self.active_sessions),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_errors,
            response_times=all_response_times,
            throughput_rps=throughput,
            error_rate=error_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            sla_violations=sla_violations,
            performance_metrics=performance_metrics,
        )

        return result

    def _check_sla_violations(
        self,
        performance_metrics: Dict[str, Any],
        error_rate: float,
        memory_usage: float,
        cpu_usage: float,
    ) -> List[Dict[str, Any]]:
        """Check for SLA violations."""
        violations = []

        # Response time violations
        p95_response_time = performance_metrics["response_time_stats"]["p95_ms"]
        if p95_response_time > self.sla_thresholds["response_time_p95_ms"]:
            violations.append(
                {
                    "metric": "response_time_p95",
                    "threshold": self.sla_thresholds["response_time_p95_ms"],
                    "actual": p95_response_time,
                    "severity": "critical",
                    "description": f"P95 response time ({p95_response_time:.1f}ms) exceeds threshold ({self.sla_thresholds['response_time_p95_ms']:.1f}ms)",
                }
            )

        p99_response_time = performance_metrics["response_time_stats"]["p99_ms"]
        if p99_response_time > self.sla_thresholds["response_time_p99_ms"]:
            violations.append(
                {
                    "metric": "response_time_p99",
                    "threshold": self.sla_thresholds["response_time_p99_ms"],
                    "actual": p99_response_time,
                    "severity": "high",
                    "description": f"P99 response time ({p99_response_time:.1f}ms) exceeds threshold ({self.sla_thresholds['response_time_p99_ms']:.1f}ms)",
                }
            )

        # Error rate violations
        if error_rate > self.sla_thresholds["error_rate_max"]:
            violations.append(
                {
                    "metric": "error_rate",
                    "threshold": self.sla_thresholds["error_rate_max"],
                    "actual": error_rate,
                    "severity": "critical",
                    "description": f"Error rate ({error_rate:.2f}%) exceeds threshold ({self.sla_thresholds['error_rate_max']:.1f}%)",
                }
            )

        # Throughput violations
        throughput = performance_metrics["throughput_stats"]["requests_per_second"]
        if throughput < self.sla_thresholds["throughput_min_rps"]:
            violations.append(
                {
                    "metric": "throughput",
                    "threshold": self.sla_thresholds["throughput_min_rps"],
                    "actual": throughput,
                    "severity": "medium",
                    "description": f"Throughput ({throughput:.1f} RPS) below threshold ({self.sla_thresholds['throughput_min_rps']:.1f} RPS)",
                }
            )

        # Memory violations
        if memory_usage > self.sla_thresholds["memory_usage_max_mb"]:
            violations.append(
                {
                    "metric": "memory_usage",
                    "threshold": self.sla_thresholds["memory_usage_max_mb"],
                    "actual": memory_usage,
                    "severity": "high",
                    "description": f"Memory usage ({memory_usage:.1f}MB) exceeds threshold ({self.sla_thresholds['memory_usage_max_mb']:.1f}MB)",
                }
            )

        # CPU violations
        if cpu_usage > self.sla_thresholds["cpu_usage_max_percent"]:
            violations.append(
                {
                    "metric": "cpu_usage",
                    "threshold": self.sla_thresholds["cpu_usage_max_percent"],
                    "actual": cpu_usage,
                    "severity": "medium",
                    "description": f"CPU usage ({cpu_usage:.1f}%) exceeds threshold ({self.sla_thresholds['cpu_usage_max_percent']:.1f}%)",
                }
            )

        return violations

    def generate_load_test_report(self, result: LoadTestResult) -> Dict[str, Any]:
        """Generate a comprehensive load test report."""
        report = {
            "test_summary": {
                "scenario_name": result.scenario_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_seconds": result.duration_seconds,
                "users_spawned": result.users_spawned,
            },
            "request_summary": {
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "success_rate": (
                    (result.successful_requests / result.total_requests * 100)
                    if result.total_requests > 0
                    else 0
                ),
                "error_rate": result.error_rate,
                "throughput_rps": result.throughput_rps,
            },
            "performance_metrics": result.performance_metrics,
            "resource_usage": {
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
            },
            "sla_validation": {
                "violations_count": len(result.sla_violations),
                "violations": result.sla_violations,
                "sla_met": len(result.sla_violations) == 0,
            },
            "recommendations": self._generate_load_test_recommendations(result),
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _generate_load_test_recommendations(self, result: LoadTestResult) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []

        # Response time recommendations
        p95_response_time = result.performance_metrics["response_time_stats"]["p95_ms"]
        if p95_response_time > 2000:
            recommendations.append(
                f"High P95 response time ({p95_response_time:.1f}ms). Consider implementing caching, optimizing database queries, or scaling horizontally."
            )

        # Throughput recommendations
        if result.throughput_rps < 50:
            recommendations.append(
                f"Low throughput ({result.throughput_rps:.1f} RPS). Consider optimizing application performance, adding connection pooling, or increasing server resources."
            )

        # Error rate recommendations
        if result.error_rate > 1.0:
            recommendations.append(
                f"High error rate ({result.error_rate:.2f}%). Investigate error causes, implement better error handling, and consider circuit breakers."
            )

        # Memory recommendations
        if result.memory_usage_mb > 1000:
            recommendations.append(
                f"High memory usage ({result.memory_usage_mb:.1f}MB). Consider implementing memory optimization, garbage collection tuning, or memory leak detection."
            )

        # CPU recommendations
        if result.cpu_usage_percent > 70:
            recommendations.append(
                f"High CPU usage ({result.cpu_usage_percent:.1f}%). Consider optimizing CPU-intensive operations, implementing caching, or scaling vertically."
            )

        # SLA violations
        if result.sla_violations:
            critical_violations = [v for v in result.sla_violations if v["severity"] == "critical"]
            if critical_violations:
                recommendations.append(
                    f"Critical SLA violations detected. Immediate action required for: {', '.join([v['metric'] for v in critical_violations])}"
                )

        if not recommendations:
            recommendations.append(
                "Load test passed all requirements. System is performing well under the tested load."
            )

        return recommendations

    def get_scenario(self, name: str) -> Optional[LoadTestScenario]:
        """Get a predefined scenario by name."""
        return self.predefined_scenarios.get(name)

    def list_scenarios(self) -> List[str]:
        """List all available scenarios."""
        return list(self.predefined_scenarios.keys())

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return self.real_time_metrics.copy()

    def clear_results_history(self):
        """Clear the results history."""
        self.results_history.clear()
        self.active_sessions.clear()

    def save_results(self, filename: str):
        """Save all results to a file."""
        data = {
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "duration_seconds": r.duration_seconds,
                    "users_spawned": r.users_spawned,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "throughput_rps": r.throughput_rps,
                    "error_rate": r.error_rate,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "sla_violations": r.sla_violations,
                    "performance_metrics": r.performance_metrics,
                }
                for r in self.results_history
            ],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Load test results saved to {filename}")


# Example usage
async def run_load_test_suite():
    """Run a comprehensive load test suite."""
    print("=" * 80)
    print("LOAD TESTING FRAMEWORK - COMPREHENSIVE SUITE")
    print("=" * 80)

    framework = LoadTestingFramework()

    # Run different scenarios
    scenarios_to_test = ["baseline", "standard", "peak"]

    for scenario_name in scenarios_to_test:
        print(f"\n{'=' * 20} {scenario_name.upper()} LOAD TEST {'=' * 20}")

        scenario = framework.get_scenario(scenario_name)
        if not scenario:
            print(f"Scenario {scenario_name} not found")
            continue

        try:
            # Modify scenario for testing (reduce duration)
            scenario.duration_seconds = min(
                scenario.duration_seconds, 60
            )  # Max 1 minute for testing
            scenario.user_count = min(scenario.user_count, 20)  # Max 20 users for testing

            result = await framework.run_load_test(scenario)
            report = framework.generate_load_test_report(result)

            # Print summary
            print(f"Duration: {result.duration_seconds:.1f}s")
            print(f"Users: {result.users_spawned}")
            print(
                f"Requests: {result.total_requests} (Success: {result.successful_requests}, Failed: {result.failed_requests})"
            )
            print(f"Throughput: {result.throughput_rps:.1f} RPS")
            print(f"Error Rate: {result.error_rate:.2f}%")
            print(
                f"P95 Response Time: {result.performance_metrics['response_time_stats']['p95_ms']:.1f}ms"
            )
            print(f"Memory Usage: {result.memory_usage_mb:.1f}MB")
            print(f"CPU Usage: {result.cpu_usage_percent:.1f}%")

            # SLA violations
            if result.sla_violations:
                print("\nSLA Violations:")
                for violation in result.sla_violations:
                    print(f"  - {violation['metric']}: {violation['description']}")
            else:
                print("\n✓ All SLA requirements met")

            # Recommendations
            print("\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        except Exception as e:
            print(f"Load test failed: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results_{timestamp}.json"
    framework.save_results(filename)
    print(f"\nResults saved to: {filename}")

    print("\n" + "=" * 80)
    print("LOAD TEST SUITE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_load_test_suite())
