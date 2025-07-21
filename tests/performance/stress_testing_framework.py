"""
Stress Testing Framework with Failure Recovery
==============================================

This module provides comprehensive stress testing capabilities including:
- Progressive load increase until system breaking point
- Failure recovery validation and graceful degradation
- System resilience testing under extreme conditions
- Resource exhaustion testing
- Circuit breaker and failover validation
- Recovery time measurement after failures
- Chaos engineering principles for fault injection
"""

import asyncio
import json
import logging
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import psutil

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures to inject for testing."""

    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_REFUSED = "connection_refused"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    DATABASE_UNAVAILABLE = "database_unavailable"
    SERVICE_OVERLOAD = "service_overload"
    PARTIAL_FAILURE = "partial_failure"


@dataclass
class StressTestPhase:
    """Represents a phase in the stress test."""

    name: str
    duration_seconds: int
    concurrent_users: int
    requests_per_second: int
    failure_injection: Optional[FailureType] = None
    failure_rate: float = 0.0
    recovery_expected: bool = False


@dataclass
class StressTestResult:
    """Result of a stress test."""

    test_name: str
    start_time: datetime
    end_time: datetime
    phases_completed: int
    breaking_point_users: int
    breaking_point_rps: float
    max_response_time_ms: float
    failure_recovery_time_ms: float
    system_degradation_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    memory_peak_mb: float
    cpu_peak_percent: float
    recovery_success: bool
    graceful_degradation: bool
    test_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureInjectionResult:
    """Result of failure injection."""

    failure_type: FailureType
    injection_time: datetime
    detection_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    affected_requests: int = 0
    system_response: str = ""
    recovery_successful: bool = False


class StressTestingFramework:
    """Advanced stress testing framework with failure recovery validation."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.process = psutil.Process()
        self.test_results: List[StressTestResult] = []
        self.failure_injections: List[FailureInjectionResult] = []

        # Stress test configuration
        self.breaking_point_threshold = (
            50.0  # 50% error rate indicates breaking point
        )
        self.recovery_timeout = 30.0  # 30 seconds max recovery time
        self.degradation_threshold = 200.0  # 200ms response time degradation

        # System health monitoring
        self.baseline_metrics = None
        self.current_metrics = {}
        self.monitoring_active = False

        # Predefined stress test scenarios
        self.stress_scenarios = self._create_stress_scenarios()

    def _create_stress_scenarios(self) -> Dict[str, List[StressTestPhase]]:
        """Create predefined stress test scenarios."""
        return {
            "progressive_load": [
                StressTestPhase("warmup", 30, 10, 5),
                StressTestPhase("light_load", 60, 25, 15),
                StressTestPhase("medium_load", 90, 50, 30),
                StressTestPhase("heavy_load", 120, 100, 60),
                StressTestPhase("extreme_load", 150, 200, 100),
                StressTestPhase("breaking_point", 180, 500, 200),
            ],
            "failure_recovery": [
                StressTestPhase("baseline", 60, 50, 25),
                StressTestPhase(
                    "network_failure",
                    90,
                    50,
                    25,
                    FailureType.NETWORK_TIMEOUT,
                    0.2,
                ),
                StressTestPhase(
                    "recovery_test", 120, 50, 25, recovery_expected=True
                ),
                StressTestPhase(
                    "database_failure",
                    90,
                    50,
                    25,
                    FailureType.DATABASE_UNAVAILABLE,
                    0.3,
                ),
                StressTestPhase(
                    "final_recovery", 60, 50, 25, recovery_expected=True
                ),
            ],
            "resource_exhaustion": [
                StressTestPhase("baseline", 60, 20, 10),
                StressTestPhase(
                    "memory_stress",
                    120,
                    50,
                    25,
                    FailureType.MEMORY_EXHAUSTION,
                    0.1,
                ),
                StressTestPhase(
                    "cpu_stress", 120, 75, 40, FailureType.CPU_SPIKE, 0.1
                ),
                StressTestPhase(
                    "combined_stress",
                    180,
                    100,
                    60,
                    FailureType.SERVICE_OVERLOAD,
                    0.2,
                ),
                StressTestPhase(
                    "recovery_validation", 90, 30, 15, recovery_expected=True
                ),
            ],
            "chaos_engineering": [
                StressTestPhase("stable_baseline", 60, 30, 20),
                StressTestPhase(
                    "random_failures",
                    180,
                    50,
                    30,
                    FailureType.PARTIAL_FAILURE,
                    0.15,
                ),
                StressTestPhase(
                    "cascade_failure",
                    120,
                    75,
                    45,
                    FailureType.SERVICE_OVERLOAD,
                    0.3,
                ),
                StressTestPhase(
                    "recovery_chaos", 150, 40, 25, recovery_expected=True
                ),
                StressTestPhase("stability_test", 90, 30, 20),
            ],
        }

    async def start_system_monitoring(self):
        """Start continuous system monitoring."""
        self.monitoring_active = True

        # Capture baseline metrics
        self.baseline_metrics = await self._capture_system_metrics()

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(
            self._monitor_system_health()
        )

        logger.info("System monitoring started")

    async def stop_system_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False

        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("System monitoring stopped")

    async def _monitor_system_health(self):
        """Monitor system health continuously."""
        try:
            while self.monitoring_active:
                self.current_metrics = await self._capture_system_metrics()
                await asyncio.sleep(1)  # Monitor every second
        except asyncio.CancelledError:
            pass

    async def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics."""
        return {
            'timestamp': time.time(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'open_files': len(self.process.open_files()),
            'threads': self.process.num_threads(),
        }

    async def run_stress_test_scenario(
        self, scenario_name: str
    ) -> StressTestResult:
        """Run a complete stress test scenario."""
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Unknown stress test scenario: {scenario_name}")

        phases = self.stress_scenarios[scenario_name]
        logger.info(
            f"Starting stress test scenario: {scenario_name} with {len(phases)} phases"
        )

        # Start monitoring
        await self.start_system_monitoring()

        start_time = datetime.now()

        try:
            # Execute each phase
            phases_completed = 0
            breaking_point_users = 0
            breaking_point_rps = 0
            max_response_time = 0
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            failure_recovery_time = 0
            recovery_success = True
            graceful_degradation = True

            for phase in phases:
                logger.info(
                    f"Executing phase: {phase.name} ({phase.concurrent_users} users, {phase.requests_per_second} RPS)"
                )

                phase_result = await self._execute_stress_phase(phase)

                # Update totals
                total_requests += phase_result['total_requests']
                successful_requests += phase_result['successful_requests']
                failed_requests += phase_result['failed_requests']
                max_response_time = max(
                    max_response_time, phase_result['max_response_time']
                )

                # Check for breaking point
                error_rate = (
                    (
                        phase_result['failed_requests']
                        / phase_result['total_requests']
                        * 100
                    )
                    if phase_result['total_requests'] > 0
                    else 0
                )
                if error_rate > self.breaking_point_threshold:
                    breaking_point_users = phase.concurrent_users
                    breaking_point_rps = phase.requests_per_second
                    logger.warning(
                        f"Breaking point reached at {breaking_point_users} users, {breaking_point_rps} RPS"
                    )

                # Check for recovery
                if phase.recovery_expected:
                    recovery_result = await self._validate_recovery(phase)
                    if not recovery_result['recovered']:
                        recovery_success = False
                        logger.error(f"Recovery failed in phase: {phase.name}")

                    failure_recovery_time += recovery_result['recovery_time']

                # Check for graceful degradation
                if phase_result['avg_response_time'] > (
                    self.baseline_metrics.get('avg_response_time', 0)
                    + self.degradation_threshold
                ):
                    if (
                        error_rate > 20
                    ):  # More than 20% errors indicates non-graceful degradation
                        graceful_degradation = False

                phases_completed += 1

                # Break if system is completely broken
                if error_rate > 90:  # 90% error rate means system is down
                    logger.error(
                        f"System completely broken in phase: {phase.name}"
                    )
                    break

                # Small delay between phases
                await asyncio.sleep(5)

            end_time = datetime.now()

            # Calculate system degradation rate
            if self.baseline_metrics and self.current_metrics:
                degradation_rate = self._calculate_degradation_rate()
            else:
                degradation_rate = 0

            # System metrics
            memory_peak = max(
                self.current_metrics.get('memory_mb', 0),
                self.baseline_metrics.get('memory_mb', 0),
            )
            cpu_peak = max(
                self.current_metrics.get('cpu_percent', 0),
                self.baseline_metrics.get('cpu_percent', 0),
            )

            error_rate = (
                (failed_requests / total_requests * 100)
                if total_requests > 0
                else 0
            )

            result = StressTestResult(
                test_name=scenario_name,
                start_time=start_time,
                end_time=end_time,
                phases_completed=phases_completed,
                breaking_point_users=breaking_point_users,
                breaking_point_rps=breaking_point_rps,
                max_response_time_ms=max_response_time,
                failure_recovery_time_ms=failure_recovery_time,
                system_degradation_rate=degradation_rate,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                error_rate=error_rate,
                memory_peak_mb=memory_peak,
                cpu_peak_percent=cpu_peak,
                recovery_success=recovery_success,
                graceful_degradation=graceful_degradation,
                test_metadata={
                    'phases': [phase.name for phase in phases],
                    'failure_injections': len(self.failure_injections),
                    'monitoring_duration': (
                        end_time - start_time
                    ).total_seconds(),
                },
            )

            self.test_results.append(result)

            logger.info(
                f"Stress test completed: {phases_completed}/{len(phases)} phases, {error_rate:.1f}% error rate"
            )

            return result

        finally:
            await self.stop_system_monitoring()

    async def _execute_stress_phase(
        self, phase: StressTestPhase
    ) -> Dict[str, Any]:
        """Execute a single stress test phase."""
        start_time = time.perf_counter()
        end_time = start_time + phase.duration_seconds

        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []

        # Failure injection if specified
        if phase.failure_injection:
            await self._inject_failure(
                phase.failure_injection, phase.failure_rate
            )

        async def stress_user(user_id: int):
            nonlocal total_requests, successful_requests, failed_requests

            try:
                # Create session for this user
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    while time.perf_counter() < end_time:
                        # Calculate delay to maintain target RPS
                        delay = (
                            1.0
                            / phase.requests_per_second
                            * phase.concurrent_users
                        )

                        # Make request
                        request_start = time.perf_counter()
                        try:
                            # Choose random endpoint
                            endpoint = random.choice(
                                [
                                    '/health',
                                    '/api/v1/system/status',
                                    '/api/v1/agents',
                                    '/metrics',
                                ]
                            )

                            async with session.get(
                                f'{self.base_url}{endpoint}'
                            ) as response:
                                await response.read()

                                request_time = (
                                    time.perf_counter() - request_start
                                ) * 1000
                                response_times.append(request_time)

                                if response.status < 400:
                                    successful_requests += 1
                                else:
                                    failed_requests += 1

                                total_requests += 1

                        except Exception as e:
                            failed_requests += 1
                            total_requests += 1
                            logger.debug(
                                f"Request failed for user {user_id}: {e}"
                            )

                        # Wait before next request
                        await asyncio.sleep(delay + random.uniform(0, 0.1))

            except Exception as e:
                logger.error(f"Stress user {user_id} failed: {e}")

        # Start all users
        tasks = []
        for i in range(phase.concurrent_users):
            task = asyncio.create_task(stress_user(i))
            tasks.append(task)

        # Wait for phase to complete
        await asyncio.sleep(phase.duration_seconds)

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for cleanup
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate phase metrics
        avg_response_time = (
            statistics.mean(response_times) if response_times else 0
        )
        max_response_time = max(response_times) if response_times else 0

        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'response_times': response_times,
        }

    async def _inject_failure(
        self, failure_type: FailureType, failure_rate: float
    ):
        """Inject a specific type of failure."""
        logger.info(
            f"Injecting failure: {failure_type.value} at {failure_rate*100}% rate"
        )

        injection_result = FailureInjectionResult(
            failure_type=failure_type, injection_time=datetime.now()
        )

        try:
            if failure_type == FailureType.NETWORK_TIMEOUT:
                await self._inject_network_timeout(failure_rate)
            elif failure_type == FailureType.MEMORY_EXHAUSTION:
                await self._inject_memory_exhaustion(failure_rate)
            elif failure_type == FailureType.CPU_SPIKE:
                await self._inject_cpu_spike(failure_rate)
            elif failure_type == FailureType.SERVICE_OVERLOAD:
                await self._inject_service_overload(failure_rate)
            elif failure_type == FailureType.PARTIAL_FAILURE:
                await self._inject_partial_failure(failure_rate)
            else:
                logger.warning(
                    f"Failure type {failure_type.value} not implemented"
                )

            injection_result.system_response = "Failure injected successfully"

        except Exception as e:
            injection_result.system_response = f"Failure injection error: {e}"
            logger.error(f"Failed to inject {failure_type.value}: {e}")

        self.failure_injections.append(injection_result)

    async def _inject_network_timeout(self, failure_rate: float):
        """Simulate network timeout failures."""
        # In a real scenario, this would configure network delays or drop packets
        # For testing, we'll simulate by adding delays to some requests
        await asyncio.sleep(1)  # Simulate configuration time
        logger.debug(
            f"Network timeout simulation configured at {failure_rate*100}% rate"
        )

    async def _inject_memory_exhaustion(self, failure_rate: float):
        """Simulate memory exhaustion."""
        # Create memory pressure
        memory_hogs = []
        try:
            for i in range(int(50 * failure_rate)):  # Scale with failure rate
                # Create large objects to consume memory
                large_object = np.random.rand(10000, 100)  # ~80MB per object
                memory_hogs.append(large_object)
                await asyncio.sleep(0.1)  # Small delay between allocations

            logger.debug(
                f"Memory exhaustion simulation: allocated {len(memory_hogs)} large objects"
            )

            # Keep memory allocated for a while
            await asyncio.sleep(30)

        finally:
            # Clean up
            memory_hogs.clear()
            import gc

            gc.collect()

    async def _inject_cpu_spike(self, failure_rate: float):
        """Simulate CPU spike."""

        def cpu_intensive_work():
            # CPU-intensive calculation
            result = 0
            for i in range(int(1000000 * failure_rate)):
                result += i**2
            return result

        # Run CPU-intensive work in thread pool
        with ThreadPoolExecutor(max_workers=int(4 * failure_rate)) as executor:
            tasks = [
                executor.submit(cpu_intensive_work)
                for _ in range(int(10 * failure_rate))
            ]

            # Wait a bit for CPU spike
            await asyncio.sleep(20)

            # Wait for tasks to complete
            for task in tasks:
                try:
                    task.result(timeout=30)
                except Exception as e:
                    logger.debug(f"CPU spike task failed: {e}")

    async def _inject_service_overload(self, failure_rate: float):
        """Simulate service overload."""
        # Simulate by creating many concurrent requests
        overload_requests = int(100 * failure_rate)

        async def overload_request():
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f'{self.base_url}/health'
                    ) as response:
                        await response.read()
            except Exception:
                pass  # Expected to fail under overload

        tasks = [overload_request() for _ in range(overload_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(
            f"Service overload simulation: sent {overload_requests} concurrent requests"
        )

    async def _inject_partial_failure(self, failure_rate: float):
        """Simulate partial system failure."""
        # Simulate by randomly failing some operations
        failure_duration = 30  # 30 seconds of partial failure

        logger.debug(
            f"Partial failure simulation: {failure_rate*100}% failure rate for {failure_duration}s"
        )

        # In a real scenario, this would configure service to fail certain requests
        await asyncio.sleep(failure_duration)

    async def _validate_recovery(
        self, phase: StressTestPhase
    ) -> Dict[str, Any]:
        """Validate system recovery after failure."""
        recovery_start = time.perf_counter()
        recovery_timeout = 60  # 60 seconds max recovery time

        logger.info(f"Validating recovery for phase: {phase.name}")

        recovered = False
        recovery_time = 0

        # Test recovery by making health checks
        async with aiohttp.ClientSession() as session:
            end_time = time.perf_counter() + recovery_timeout

            while time.perf_counter() < end_time:
                try:
                    async with session.get(
                        f'{self.base_url}/health'
                    ) as response:
                        if response.status == 200:
                            recovery_time = (
                                time.perf_counter() - recovery_start
                            ) * 1000
                            recovered = True
                            logger.info(
                                f"System recovered in {recovery_time:.1f}ms"
                            )
                            break
                except Exception:
                    pass

                await asyncio.sleep(1)  # Check every second

        if not recovered:
            recovery_time = recovery_timeout * 1000
            logger.error(f"System did not recover within {recovery_timeout}s")

        return {
            'recovered': recovered,
            'recovery_time': recovery_time,
            'recovery_timeout': recovery_timeout,
        }

    def _calculate_degradation_rate(self) -> float:
        """Calculate system degradation rate compared to baseline."""
        if not self.baseline_metrics or not self.current_metrics:
            return 0

        # Calculate degradation across multiple metrics
        degradation_factors = []

        # Memory degradation
        baseline_memory = self.baseline_metrics.get('memory_mb', 0)
        current_memory = self.current_metrics.get('memory_mb', 0)
        if baseline_memory > 0:
            memory_degradation = (
                current_memory - baseline_memory
            ) / baseline_memory
            degradation_factors.append(memory_degradation)

        # CPU degradation
        baseline_cpu = self.baseline_metrics.get('cpu_percent', 0)
        current_cpu = self.current_metrics.get('cpu_percent', 0)
        if baseline_cpu > 0:
            cpu_degradation = (current_cpu - baseline_cpu) / baseline_cpu
            degradation_factors.append(cpu_degradation)

        # Network connections degradation
        baseline_connections = self.baseline_metrics.get(
            'network_connections', 0
        )
        current_connections = self.current_metrics.get(
            'network_connections', 0
        )
        if baseline_connections > 0:
            conn_degradation = (
                current_connections - baseline_connections
            ) / baseline_connections
            degradation_factors.append(conn_degradation)

        # Return average degradation rate
        if degradation_factors:
            return (
                statistics.mean(degradation_factors) * 100
            )  # Convert to percentage
        return 0

    def generate_stress_test_report(
        self, result: StressTestResult
    ) -> Dict[str, Any]:
        """Generate comprehensive stress test report."""
        report = {
            'test_summary': {
                'scenario_name': result.test_name,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration_seconds': (
                    result.end_time - result.start_time
                ).total_seconds(),
                'phases_completed': result.phases_completed,
            },
            'breaking_point_analysis': {
                'breaking_point_users': result.breaking_point_users,
                'breaking_point_rps': result.breaking_point_rps,
                'breaking_point_reached': result.breaking_point_users > 0,
                'system_limits': {
                    'max_concurrent_users': result.breaking_point_users
                    if result.breaking_point_users > 0
                    else 'Not reached',
                    'max_requests_per_second': result.breaking_point_rps
                    if result.breaking_point_rps > 0
                    else 'Not reached',
                },
            },
            'failure_recovery_analysis': {
                'recovery_time_ms': result.failure_recovery_time_ms,
                'recovery_success': result.recovery_success,
                'graceful_degradation': result.graceful_degradation,
                'system_degradation_rate': result.system_degradation_rate,
                'failure_injections': len(self.failure_injections),
            },
            'performance_metrics': {
                'total_requests': result.total_requests,
                'successful_requests': result.successful_requests,
                'failed_requests': result.failed_requests,
                'error_rate': result.error_rate,
                'max_response_time_ms': result.max_response_time_ms,
                'success_rate': (
                    (result.successful_requests / result.total_requests) * 100
                )
                if result.total_requests > 0
                else 0,
            },
            'resource_utilization': {
                'memory_peak_mb': result.memory_peak_mb,
                'cpu_peak_percent': result.cpu_peak_percent,
                'baseline_comparison': {
                    'memory_increase': result.memory_peak_mb
                    - (
                        self.baseline_metrics.get('memory_mb', 0)
                        if self.baseline_metrics
                        else 0
                    ),
                    'cpu_increase': result.cpu_peak_percent
                    - (
                        self.baseline_metrics.get('cpu_percent', 0)
                        if self.baseline_metrics
                        else 0
                    ),
                },
            },
            'resilience_assessment': {
                'system_resilience_score': self._calculate_resilience_score(
                    result
                ),
                'failure_tolerance': 'High'
                if result.recovery_success and result.graceful_degradation
                else 'Medium'
                if result.recovery_success
                else 'Low',
                'recovery_capability': 'Excellent'
                if result.failure_recovery_time_ms < 5000
                else 'Good'
                if result.failure_recovery_time_ms < 30000
                else 'Poor',
            },
            'recommendations': self._generate_stress_test_recommendations(
                result
            ),
            'failure_details': [
                {
                    'failure_type': inj.failure_type.value,
                    'injection_time': inj.injection_time.isoformat(),
                    'recovery_time': inj.recovery_time.isoformat()
                    if inj.recovery_time
                    else None,
                    'system_response': inj.system_response,
                    'recovery_successful': inj.recovery_successful,
                }
                for inj in self.failure_injections
            ],
        }

        return report

    def _calculate_resilience_score(self, result: StressTestResult) -> float:
        """Calculate a resilience score based on test results."""
        score = 100.0  # Start with perfect score

        # Deduct points for high error rate
        if result.error_rate > 10:
            score -= min(result.error_rate * 2, 40)  # Max 40 points deduction

        # Deduct points for failed recovery
        if not result.recovery_success:
            score -= 30

        # Deduct points for non-graceful degradation
        if not result.graceful_degradation:
            score -= 20

        # Deduct points for slow recovery
        if result.failure_recovery_time_ms > 30000:  # >30s recovery
            score -= 20
        elif result.failure_recovery_time_ms > 10000:  # >10s recovery
            score -= 10

        # Deduct points for high resource usage
        if result.memory_peak_mb > 2000:  # >2GB
            score -= 10
        if result.cpu_peak_percent > 90:  # >90% CPU
            score -= 10

        return max(score, 0)  # Ensure score is not negative

    def _generate_stress_test_recommendations(
        self, result: StressTestResult
    ) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []

        # Breaking point recommendations
        if result.breaking_point_users > 0:
            recommendations.append(
                f"System breaking point reached at {result.breaking_point_users} concurrent users. Consider horizontal scaling or load balancing."
            )

        # Error rate recommendations
        if result.error_rate > 10:
            recommendations.append(
                f"High error rate ({result.error_rate:.1f}%). Implement circuit breakers and better error handling."
            )

        # Recovery recommendations
        if not result.recovery_success:
            recommendations.append(
                "System failed to recover from injected failures. Implement health checks and automatic recovery mechanisms."
            )

        # Response time recommendations
        if result.max_response_time_ms > 10000:  # >10s response time
            recommendations.append(
                f"Very slow response times detected ({result.max_response_time_ms:.1f}ms). Implement request timeouts and caching."
            )

        # Resource usage recommendations
        if result.memory_peak_mb > 1500:  # >1.5GB
            recommendations.append(
                f"High memory usage ({result.memory_peak_mb:.1f}MB). Implement memory management and garbage collection optimization."
            )

        if result.cpu_peak_percent > 85:  # >85% CPU
            recommendations.append(
                f"High CPU usage ({result.cpu_peak_percent:.1f}%). Optimize CPU-intensive operations and consider load balancing."
            )

        # Graceful degradation recommendations
        if not result.graceful_degradation:
            recommendations.append(
                "System does not degrade gracefully under load. Implement rate limiting and request prioritization."
            )

        # General resilience recommendations
        resilience_score = self._calculate_resilience_score(result)
        if resilience_score < 70:
            recommendations.append(
                f"Low resilience score ({resilience_score:.1f}). Implement comprehensive fault tolerance and monitoring."
            )

        if not recommendations:
            recommendations.append(
                "System demonstrated excellent resilience under stress. Continue monitoring and testing regularly."
            )

        return recommendations

    def save_results(self, filename: str):
        """Save stress test results to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': [
                {
                    'test_name': result.test_name,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'phases_completed': result.phases_completed,
                    'breaking_point_users': result.breaking_point_users,
                    'breaking_point_rps': result.breaking_point_rps,
                    'max_response_time_ms': result.max_response_time_ms,
                    'failure_recovery_time_ms': result.failure_recovery_time_ms,
                    'system_degradation_rate': result.system_degradation_rate,
                    'total_requests': result.total_requests,
                    'successful_requests': result.successful_requests,
                    'failed_requests': result.failed_requests,
                    'error_rate': result.error_rate,
                    'memory_peak_mb': result.memory_peak_mb,
                    'cpu_peak_percent': result.cpu_peak_percent,
                    'recovery_success': result.recovery_success,
                    'graceful_degradation': result.graceful_degradation,
                    'test_metadata': result.test_metadata,
                }
                for result in self.test_results
            ],
            'failure_injections': [
                {
                    'failure_type': inj.failure_type.value,
                    'injection_time': inj.injection_time.isoformat(),
                    'system_response': inj.system_response,
                    'recovery_successful': inj.recovery_successful,
                }
                for inj in self.failure_injections
            ],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Stress test results saved to {filename}")


# Example usage
async def run_stress_test_validation():
    """Run comprehensive stress test validation."""
    print("=" * 80)
    print("STRESS TESTING FRAMEWORK WITH FAILURE RECOVERY")
    print("=" * 80)

    framework = StressTestingFramework()

    # Test scenarios
    scenarios_to_test = [
        'progressive_load',
        'failure_recovery',
        'resource_exhaustion',
    ]

    for scenario_name in scenarios_to_test:
        print(f"\n{'='*20} {scenario_name.upper()} STRESS TEST {'='*20}")

        try:
            # Run stress test scenario
            result = await framework.run_stress_test_scenario(scenario_name)
            report = framework.generate_stress_test_report(result)

            # Print summary
            print(
                f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s"
            )
            print(f"Phases completed: {result.phases_completed}")
            print(
                f"Breaking point: {result.breaking_point_users} users at {result.breaking_point_rps} RPS"
            )
            print(f"Total requests: {result.total_requests}")
            print(f"Error rate: {result.error_rate:.1f}%")
            print(f"Max response time: {result.max_response_time_ms:.1f}ms")
            print(
                f"Recovery success: {'✓' if result.recovery_success else '✗'}"
            )
            print(
                f"Graceful degradation: {'✓' if result.graceful_degradation else '✗'}"
            )
            print(f"Memory peak: {result.memory_peak_mb:.1f}MB")
            print(f"CPU peak: {result.cpu_peak_percent:.1f}%")

            # Resilience assessment
            resilience = report['resilience_assessment']
            print(
                f"Resilience score: {resilience['system_resilience_score']:.1f}/100"
            )
            print(f"Failure tolerance: {resilience['failure_tolerance']}")
            print(f"Recovery capability: {resilience['recovery_capability']}")

            # Recommendations
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

        except Exception as e:
            print(f"Stress test {scenario_name} failed: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stress_test_results_{timestamp}.json"
    framework.save_results(filename)
    print(f"\nResults saved to: {filename}")

    print("\n" + "=" * 80)
    print("STRESS TEST VALIDATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_stress_test_validation())
