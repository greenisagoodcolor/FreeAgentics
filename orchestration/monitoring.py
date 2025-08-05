"""Orchestration Monitoring and Health Checking.

Provides comprehensive monitoring, metrics collection, and health checking
for the conversation orchestration system. Integrates with existing
Prometheus metrics and adds orchestration-specific observability.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from threading import Lock

from observability.prometheus_metrics import PrometheusMetricsCollector
from .errors import categorize_error

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a component."""

    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_rate: float = 0.0
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "error_rate": self.error_rate,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration operations."""

    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

    # Timing metrics
    avg_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0

    # Error metrics
    error_count_by_type: Dict[str, int] = field(default_factory=dict)
    error_count_by_component: Dict[str, int] = field(default_factory=dict)

    # Resource metrics
    active_conversations: int = 0
    peak_active_conversations: int = 0

    # Retry metrics
    total_retries: int = 0
    retry_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        success_rate = self.successful_executions / max(1, self.total_executions)
        failure_rate = self.failed_executions / max(1, self.total_executions)

        return {
            "execution_metrics": {
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "failed_executions": self.failed_executions,
                "success_rate": success_rate,
                "failure_rate": failure_rate,
            },
            "timing_metrics": {
                "avg_execution_time_ms": self.avg_execution_time_ms,
                "p95_execution_time_ms": self.p95_execution_time_ms,
                "p99_execution_time_ms": self.p99_execution_time_ms,
            },
            "error_metrics": {
                "by_type": self.error_count_by_type,
                "by_component": self.error_count_by_component,
            },
            "resource_metrics": {
                "active_conversations": self.active_conversations,
                "peak_active_conversations": self.peak_active_conversations,
            },
            "retry_metrics": {
                "total_retries": self.total_retries,
                "retry_success_rate": self.retry_success_rate,
            },
        }


class HealthChecker:
    """Health checker for orchestration components.

    Monitors component health and provides unified health status
    with detailed component-level information.
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        unhealthy_threshold: int = 3,
        degraded_threshold: float = 0.1,  # 10% error rate
    ):
        self.check_interval_seconds = check_interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.degraded_threshold = degraded_threshold

        self._component_health: Dict[str, ComponentHealth] = {}
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._check_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = Lock()

    async def start(self) -> None:
        """Start health checking."""
        if self._is_running:
            return

        self._is_running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop health checking."""
        if not self._is_running:
            return

        self._is_running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("Health checker stopped")

    def register_component(self, name: str) -> None:
        """Register a component for health checking."""
        with self._lock:
            if name not in self._component_health:
                self._component_health[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(timezone.utc),
                )
                logger.debug(f"Registered component for health checking: {name}")

    def update_component_health(
        self,
        name: str,
        status: HealthStatus,
        response_time_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
        message: Optional[str] = None,
        **metadata,
    ) -> None:
        """Update component health status."""
        with self._lock:
            if name not in self._component_health:
                self.register_component(name)

            self._component_health[name] = ComponentHealth(
                name=name,
                status=status,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time_ms,
                error_rate=error_rate or 0.0,
                message=message,
                metadata=metadata,
            )

            # Add to history
            self._health_history[name].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": status.value,
                    "response_time_ms": response_time_ms,
                    "error_rate": error_rate,
                }
            )

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        with self._lock:
            return self._component_health.get(name)

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            if not self._component_health:
                return {
                    "overall_status": HealthStatus.UNKNOWN.value,
                    "components": {},
                    "summary": {
                        "total_components": 0,
                        "healthy": 0,
                        "degraded": 0,
                        "unhealthy": 0,
                        "unknown": 0,
                    },
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

            # Count component statuses
            status_counts = defaultdict(int)
            for health in self._component_health.values():
                status_counts[health.status] += 1

            # Determine overall status
            if status_counts[HealthStatus.UNHEALTHY] > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif status_counts[HealthStatus.DEGRADED] > 0:
                overall_status = HealthStatus.DEGRADED
            elif status_counts[HealthStatus.HEALTHY] == len(self._component_health):
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.DEGRADED

            return {
                "overall_status": overall_status.value,
                "components": {
                    name: health.to_dict() for name, health in self._component_health.items()
                },
                "summary": {
                    "total_components": len(self._component_health),
                    "healthy": status_counts[HealthStatus.HEALTHY],
                    "degraded": status_counts[HealthStatus.DEGRADED],
                    "unhealthy": status_counts[HealthStatus.UNHEALTHY],
                    "unknown": status_counts[HealthStatus.UNKNOWN],
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

    async def _health_check_loop(self) -> None:
        """Background health checking loop."""
        while self._is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error

    async def _perform_health_checks(self) -> None:
        """Perform health checks for all registered components."""
        # This is a placeholder - actual health checks would be implemented
        # by the orchestrator based on component-specific logic
        pass


class MetricsCollector:
    """Collector for orchestration metrics.

    Tracks execution metrics, timing, errors, and resource usage
    with sliding window calculations for rates and percentiles.
    """

    def __init__(
        self,
        window_size: int = 1000,
        prometheus_collector: Optional[PrometheusMetricsCollector] = None,
    ):
        self.window_size = window_size
        self.prometheus_collector = prometheus_collector or PrometheusMetricsCollector()

        # Metrics storage
        self._metrics = OrchestrationMetrics()
        self._execution_times: deque = deque(maxlen=window_size)
        self._error_history: deque = deque(maxlen=window_size)
        self._lock = Lock()

        # Active conversation tracking
        self._active_conversations: Set[str] = set()

    def record_execution_start(self, conversation_id: str) -> None:
        """Record the start of an execution."""
        with self._lock:
            self._active_conversations.add(conversation_id)
            self._metrics.active_conversations = len(self._active_conversations)

            if self._metrics.active_conversations > self._metrics.peak_active_conversations:
                self._metrics.peak_active_conversations = self._metrics.active_conversations

    def record_execution_end(
        self,
        conversation_id: str,
        success: bool,
        execution_time_ms: float,
        error: Optional[Exception] = None,
        retry_count: int = 0,
    ) -> None:
        """Record the completion of an execution."""
        with self._lock:
            # Remove from active conversations
            self._active_conversations.discard(conversation_id)
            self._metrics.active_conversations = len(self._active_conversations)

            # Update execution counts
            self._metrics.total_executions += 1
            if success:
                self._metrics.successful_executions += 1
            else:
                self._metrics.failed_executions += 1

            # Update timing metrics
            self._execution_times.append(execution_time_ms)
            self._update_timing_metrics()

            # Update error metrics
            if error:
                error_type = categorize_error(error)
                self._metrics.error_count_by_type[error_type] = (
                    self._metrics.error_count_by_type.get(error_type, 0) + 1
                )

                # Extract component from error context if available
                component = "unknown"
                if hasattr(error, "context") and error.context:
                    component = error.context.component or "unknown"

                self._metrics.error_count_by_component[component] = (
                    self._metrics.error_count_by_component.get(component, 0) + 1
                )

                self._error_history.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": error_type,
                        "component": component,
                        "message": str(error),
                    }
                )

            # Update retry metrics
            if retry_count > 0:
                self._metrics.total_retries += retry_count
                # Update retry success rate (simplified calculation)
                if success:
                    # This is a rough approximation - more sophisticated tracking
                    # would be needed for precise retry success rates
                    total_retry_attempts = self._metrics.total_retries
                    if total_retry_attempts > 0:
                        self._metrics.retry_success_rate = min(
                            1.0, self._metrics.successful_executions / total_retry_attempts
                        )

        # Update Prometheus metrics
        if self.prometheus_collector:
            # Use the available function for recording business operations
            from observability.prometheus_metrics import record_business_inference_operation

            record_business_inference_operation(
                operation_type="conversation_orchestration",
                success=success,
            )

    def record_component_error(self, component: str, error: Exception) -> None:
        """Record an error for a specific component."""
        with self._lock:
            error_type = categorize_error(error)

            self._metrics.error_count_by_component[component] = (
                self._metrics.error_count_by_component.get(component, 0) + 1
            )

            self._metrics.error_count_by_type[error_type] = (
                self._metrics.error_count_by_type.get(error_type, 0) + 1
            )

    def get_metrics(self) -> OrchestrationMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return OrchestrationMetrics(
                total_executions=self._metrics.total_executions,
                successful_executions=self._metrics.successful_executions,
                failed_executions=self._metrics.failed_executions,
                avg_execution_time_ms=self._metrics.avg_execution_time_ms,
                p95_execution_time_ms=self._metrics.p95_execution_time_ms,
                p99_execution_time_ms=self._metrics.p99_execution_time_ms,
                error_count_by_type=self._metrics.error_count_by_type.copy(),
                error_count_by_component=self._metrics.error_count_by_component.copy(),
                active_conversations=self._metrics.active_conversations,
                peak_active_conversations=self._metrics.peak_active_conversations,
                total_retries=self._metrics.total_retries,
                retry_success_rate=self._metrics.retry_success_rate,
            )

    def get_error_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent error history."""
        with self._lock:
            return list(self._error_history)[-limit:]

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._metrics = OrchestrationMetrics()
            self._execution_times.clear()
            self._error_history.clear()
            self._active_conversations.clear()

    def _update_timing_metrics(self) -> None:
        """Update timing metrics from execution times window."""
        if not self._execution_times:
            return

        times = sorted(self._execution_times)
        n = len(times)

        # Average
        self._metrics.avg_execution_time_ms = sum(times) / n

        # Percentiles
        if n >= 2:
            p95_index = int(n * 0.95)
            self._metrics.p95_execution_time_ms = times[min(p95_index, n - 1)]

            p99_index = int(n * 0.99)
            self._metrics.p99_execution_time_ms = times[min(p99_index, n - 1)]
