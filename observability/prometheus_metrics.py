"""Prometheus metrics exposition for FreeAgentics multi-agent system.

This module provides native Prometheus metrics exposition with comprehensive
agent coordination, belief system, and performance monitoring capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from observability.performance_metrics import performance_tracker

logger = logging.getLogger(__name__)

# Custom registry for FreeAgentics metrics
freeagentics_registry = CollectorRegistry()

# ============================================================================
# AGENT COORDINATION METRICS
# ============================================================================

agent_coordination_requests_total = Counter(
    "freeagentics_agent_coordination_requests_total",
    "Total number of agent coordination requests",
    ["agent_id", "coordination_type", "status"],
    registry=freeagentics_registry,
)

agent_coordination_errors_total = Counter(
    "freeagentics_agent_coordination_errors_total",
    "Total number of agent coordination errors",
    ["agent_id", "error_type", "severity"],
    registry=freeagentics_registry,
)

agent_coordination_concurrent_sessions = Gauge(
    "freeagentics_agent_coordination_concurrent_sessions",
    "Number of concurrent agent coordination sessions",
    ["coordination_type"],
    registry=freeagentics_registry,
)

agent_coordination_duration_seconds = Histogram(
    "freeagentics_agent_coordination_duration_seconds",
    "Duration of agent coordination operations",
    ["agent_id", "coordination_type"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=freeagentics_registry,
)

# ============================================================================
# BELIEF SYSTEM METRICS
# ============================================================================

belief_state_updates_total = Counter(
    "freeagentics_belief_state_updates_total",
    "Total number of belief state updates",
    ["agent_id", "update_type", "success"],
    registry=freeagentics_registry,
)

belief_convergence_time_seconds = Histogram(
    "freeagentics_belief_convergence_time_seconds",
    "Time taken for belief convergence",
    ["agent_id"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=freeagentics_registry,
)

belief_accuracy_ratio = Gauge(
    "freeagentics_belief_accuracy_ratio",
    "Ratio of accurate beliefs vs total beliefs",
    ["agent_id"],
    registry=freeagentics_registry,
)

belief_free_energy_current = Gauge(
    "freeagentics_belief_free_energy_current",
    "Current free energy value for agent beliefs",
    ["agent_id"],
    registry=freeagentics_registry,
)

belief_prediction_errors_total = Counter(
    "freeagentics_belief_prediction_errors_total",
    "Total number of belief prediction errors",
    ["agent_id", "error_magnitude"],
    registry=freeagentics_registry,
)

# ============================================================================
# AGENT PERFORMANCE METRICS
# ============================================================================

agent_inference_duration_seconds = Histogram(
    "freeagentics_agent_inference_duration_seconds",
    "Duration of agent inference operations",
    ["agent_id", "operation_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=freeagentics_registry,
)

agent_steps_total = Counter(
    "freeagentics_agent_steps_total",
    "Total number of agent steps executed",
    ["agent_id", "step_type", "success"],
    registry=freeagentics_registry,
)

agent_errors_total = Counter(
    "freeagentics_agent_errors_total",
    "Total number of agent errors",
    ["agent_id", "error_type", "severity"],
    registry=freeagentics_registry,
)

agent_memory_usage_bytes = Gauge(
    "freeagentics_agent_memory_usage_bytes",
    "Memory usage per agent in bytes",
    ["agent_id"],
    registry=freeagentics_registry,
)

agent_cpu_usage_percent = Gauge(
    "freeagentics_agent_cpu_usage_percent",
    "CPU usage per agent as percentage",
    ["agent_id"],
    registry=freeagentics_registry,
)

# ============================================================================
# SYSTEM METRICS
# ============================================================================

system_active_agents_total = Gauge(
    "freeagentics_system_active_agents_total",
    "Total number of active agents",
    registry=freeagentics_registry,
)

system_coalitions_total = Gauge(
    "freeagentics_system_coalitions_total",
    "Total number of active coalitions",
    registry=freeagentics_registry,
)

system_knowledge_graph_nodes_total = Gauge(
    "freeagentics_system_knowledge_graph_nodes_total",
    "Total number of knowledge graph nodes",
    registry=freeagentics_registry,
)

system_knowledge_graph_edges_total = Gauge(
    "freeagentics_system_knowledge_graph_edges_total",
    "Total number of knowledge graph edges",
    registry=freeagentics_registry,
)

system_throughput_operations_per_second = Gauge(
    "freeagentics_system_throughput_operations_per_second",
    "System throughput in operations per second",
    ["operation_type"],
    registry=freeagentics_registry,
)

system_memory_usage_bytes = Gauge(
    "freeagentics_system_memory_usage_bytes",
    "Total system memory usage in bytes",
    registry=freeagentics_registry,
)

system_cpu_usage_percent = Gauge(
    "freeagentics_system_cpu_usage_percent",
    "System CPU usage as percentage",
    registry=freeagentics_registry,
)

# ============================================================================
# BUSINESS METRICS
# ============================================================================

business_inference_operations_total = Counter(
    "freeagentics_business_inference_operations_total",
    "Total number of inference operations",
    ["operation_type", "success"],
    registry=freeagentics_registry,
)

business_user_interactions_total = Counter(
    "freeagentics_business_user_interactions_total",
    "Total number of user interactions",
    ["interaction_type", "outcome"],
    registry=freeagentics_registry,
)

business_response_quality_score = Gauge(
    "freeagentics_business_response_quality_score",
    "Quality score of system responses",
    ["response_type"],
    registry=freeagentics_registry,
)

# ============================================================================
# SECURITY METRICS
# ============================================================================

security_authentication_attempts_total = Counter(
    "freeagentics_security_authentication_attempts_total",
    "Total number of authentication attempts",
    ["method", "outcome"],
    registry=freeagentics_registry,
)

security_anomaly_detections_total = Counter(
    "freeagentics_security_anomaly_detections_total",
    "Total number of security anomaly detections",
    ["anomaly_type", "severity"],
    registry=freeagentics_registry,
)

security_access_violations_total = Counter(
    "freeagentics_security_access_violations_total",
    "Total number of access violations",
    ["violation_type", "resource"],
    registry=freeagentics_registry,
)

# ============================================================================
# BUILD AND DEPLOYMENT METRICS
# ============================================================================

build_info = Info(
    "freeagentics_build_info",
    "Build information for FreeAgentics",
    registry=freeagentics_registry,
)

deployment_info = Info(
    "freeagentics_deployment_info",
    "Deployment information for FreeAgentics",
    registry=freeagentics_registry,
)

# ============================================================================
# METRICS COLLECTION CLASS
# ============================================================================


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics state."""

    timestamp: datetime
    active_agents: int
    total_inferences: int
    total_belief_updates: int
    avg_inference_time_ms: float
    avg_memory_usage_mb: float
    avg_cpu_usage_percent: float
    system_throughput: float
    free_energy_avg: float


class PrometheusMetricsCollector:
    """Collects and exposes FreeAgentics metrics in Prometheus format."""

    def __init__(self):
        self.last_collection_time = time.time()
        self.collection_interval = 15.0  # seconds
        self.running = False
        self.collection_task = None

        # Initialize build info
        self._initialize_build_info()

        logger.info("🎯 Prometheus metrics collector initialized")

    def _initialize_build_info(self):
        """Initialize build and deployment information."""
        try:
            import os

            build_info.info(
                {
                    "version": os.getenv("FREEAGENTICS_VERSION", "0.0.1-dev"),
                    "commit": os.getenv("GIT_COMMIT", "unknown"),
                    "branch": os.getenv("GIT_BRANCH", "unknown"),
                    "build_date": os.getenv("BUILD_DATE", "unknown"),
                    "python_version": os.getenv("PYTHON_VERSION", "unknown"),
                }
            )

            deployment_info.info(
                {
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "deployment_date": datetime.now().isoformat(),
                    "kubernetes_namespace": os.getenv(
                        "KUBERNETES_NAMESPACE", "default"
                    ),
                    "pod_name": os.getenv("POD_NAME", "unknown"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to initialize build info: {e}")

    async def start_collection(self):
        """Start automatic metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("📊 Prometheus metrics collection started")

    async def stop_collection(self):
        """Stop automatic metrics collection."""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Prometheus metrics collection stopped")

    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_metrics(self):
        """Collect and update all metrics."""
        try:
            # Get performance tracker snapshot
            snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )

            # Update system metrics
            system_active_agents_total.set(snapshot.active_agents)
            system_memory_usage_bytes.set(
                snapshot.memory_usage_mb * 1024 * 1024
            )
            system_cpu_usage_percent.set(snapshot.cpu_usage_percent)

            # Update throughput metrics
            system_throughput_operations_per_second.labels(
                operation_type="inference"
            ).set(snapshot.agent_throughput)

            system_throughput_operations_per_second.labels(
                operation_type="belief_update"
            ).set(snapshot.belief_updates_per_sec)

            # Update belief system metrics
            if snapshot.free_energy_avg > 0:
                belief_free_energy_current.labels(agent_id="system").set(
                    snapshot.free_energy_avg
                )

            # Collect agent-specific metrics
            await self._collect_agent_metrics()

            # Collect system resource metrics
            await self._collect_system_resources()

            # Update collection timestamp
            self.last_collection_time = time.time()

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")

    async def _collect_agent_metrics(self):
        """Collect metrics for individual agents."""
        try:
            # Get agent metrics from performance tracker
            for (
                agent_id,
                agent_buffers,
            ) in performance_tracker.agent_metrics.items():
                # Inference time metrics
                inference_stats = agent_buffers["inference_times"].get_stats(
                    60
                )
                if inference_stats["count"] > 0:
                    # Update histogram with average (approximation)
                    agent_inference_duration_seconds.labels(
                        agent_id=agent_id, operation_type="inference"
                    ).observe(
                        inference_stats["avg"] / 1000.0
                    )  # Convert ms to seconds

                # Memory usage (approximation based on system metrics)
                memory_estimate = (
                    performance_tracker.memory_usage.get_stats(60)["latest"]
                    * 1024
                    * 1024
                ) / max(1, len(performance_tracker.agent_metrics))
                agent_memory_usage_bytes.labels(agent_id=agent_id).set(
                    memory_estimate
                )

                # CPU usage (approximation)
                cpu_estimate = performance_tracker.cpu_usage.get_stats(60)[
                    "latest"
                ] / max(1, len(performance_tracker.agent_metrics))
                agent_cpu_usage_percent.labels(agent_id=agent_id).set(
                    cpu_estimate
                )

                # Error metrics
                error_stats = agent_buffers["error_counts"].get_stats(60)
                if error_stats["count"] > 0:
                    agent_errors_total.labels(
                        agent_id=agent_id,
                        error_type="inference",
                        severity="warning",
                    ).inc(error_stats["count"])

        except Exception as e:
            logger.error(f"Failed to collect agent metrics: {e}")

    async def _collect_system_resources(self):
        """Collect system resource metrics."""
        try:
            import psutil

            # System memory
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.set(memory.used)

            # System CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            system_cpu_usage_percent.set(cpu_percent)

        except Exception as e:
            logger.debug(f"Failed to collect system resources: {e}")

    def get_metrics_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        try:
            # Get current values from gauges
            active_agents = int(system_active_agents_total._value.get())

            # Get counters
            total_inferences = sum(
                counter._value.get()  
                for counter in business_inference_operations_total._metrics.values()
            )

            total_belief_updates = sum(
                counter._value.get()  
                for counter in belief_state_updates_total._metrics.values()
            )

            return MetricsSnapshot(
                timestamp=datetime.now(),
                active_agents=active_agents,
                total_inferences=total_inferences,
                total_belief_updates=total_belief_updates,
                avg_inference_time_ms=0.0,  # Would need to calculate from histogram
                avg_memory_usage_mb=system_memory_usage_bytes._value.get()
                / (1024 * 1024),
                avg_cpu_usage_percent=system_cpu_usage_percent._value.get(),
                system_throughput=0.0,  # Would need to calculate from gauge
                free_energy_avg=0.0,  # Would need to get from belief gauge
            )
        except Exception as e:
            logger.error(f"Failed to get metrics snapshot: {e}")
            return MetricsSnapshot(
                timestamp=datetime.now(),
                active_agents=0,
                total_inferences=0,
                total_belief_updates=0,
                avg_inference_time_ms=0.0,
                avg_memory_usage_mb=0.0,
                avg_cpu_usage_percent=0.0,
                system_throughput=0.0,
                free_energy_avg=0.0,
            )

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus exposition format."""
        return generate_latest(freeagentics_registry).decode("utf-8")

    def get_content_type(self) -> str:
        """Get the content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST


# ============================================================================
# CONVENIENCE FUNCTIONS FOR METRICS RECORDING
# ============================================================================


def record_agent_coordination_request(
    agent_id: str, coordination_type: str, status: str
):
    """Record an agent coordination request."""
    agent_coordination_requests_total.labels(
        agent_id=agent_id, coordination_type=coordination_type, status=status
    ).inc()


def record_agent_coordination_error(
    agent_id: str, error_type: str, severity: str
):
    """Record an agent coordination error."""
    agent_coordination_errors_total.labels(
        agent_id=agent_id, error_type=error_type, severity=severity
    ).inc()


def record_agent_coordination_duration(
    agent_id: str, coordination_type: str, duration_seconds: float
):
    """Record agent coordination duration."""
    agent_coordination_duration_seconds.labels(
        agent_id=agent_id, coordination_type=coordination_type
    ).observe(duration_seconds)


def record_belief_state_update(agent_id: str, update_type: str, success: bool):
    """Record a belief state update."""
    belief_state_updates_total.labels(
        agent_id=agent_id,
        update_type=update_type,
        success="true" if success else "false",
    ).inc()


def record_belief_convergence_time(
    agent_id: str, convergence_time_seconds: float
):
    """Record belief convergence time."""
    belief_convergence_time_seconds.labels(agent_id=agent_id).observe(
        convergence_time_seconds
    )


def update_belief_accuracy_ratio(agent_id: str, accuracy_ratio: float):
    """Update belief accuracy ratio."""
    belief_accuracy_ratio.labels(agent_id=agent_id).set(accuracy_ratio)


def update_belief_free_energy(agent_id: str, free_energy: float):
    """Update belief free energy."""
    belief_free_energy_current.labels(agent_id=agent_id).set(free_energy)


def record_agent_inference_duration(
    agent_id: str, operation_type: str, duration_seconds: float
):
    """Record agent inference duration."""
    agent_inference_duration_seconds.labels(
        agent_id=agent_id, operation_type=operation_type
    ).observe(duration_seconds)


def record_agent_step(agent_id: str, step_type: str, success: bool):
    """Record an agent step."""
    agent_steps_total.labels(
        agent_id=agent_id,
        step_type=step_type,
        success="true" if success else "false",
    ).inc()


def record_agent_error(agent_id: str, error_type: str, severity: str):
    """Record an agent error."""
    agent_errors_total.labels(
        agent_id=agent_id, error_type=error_type, severity=severity
    ).inc()


def record_business_inference_operation(operation_type: str, success: bool):
    """Record a business inference operation."""
    business_inference_operations_total.labels(
        operation_type=operation_type, success="true" if success else "false"
    ).inc()


def record_user_interaction(interaction_type: str, outcome: str):
    """Record a user interaction."""
    business_user_interactions_total.labels(
        interaction_type=interaction_type, outcome=outcome
    ).inc()


def update_response_quality_score(response_type: str, score: float):
    """Update response quality score."""
    business_response_quality_score.labels(response_type=response_type).set(
        score
    )


def record_authentication_attempt(method: str, outcome: str):
    """Record an authentication attempt."""
    security_authentication_attempts_total.labels(
        method=method, outcome=outcome
    ).inc()


def record_security_anomaly(anomaly_type: str, severity: str):
    """Record a security anomaly detection."""
    security_anomaly_detections_total.labels(
        anomaly_type=anomaly_type, severity=severity
    ).inc()


def record_access_violation(violation_type: str, resource: str):
    """Record an access violation."""
    security_access_violations_total.labels(
        violation_type=violation_type, resource=resource
    ).inc()


# ============================================================================
# GLOBAL METRICS COLLECTOR INSTANCE
# ============================================================================

prometheus_collector = PrometheusMetricsCollector()


async def start_prometheus_metrics_collection():
    """Start Prometheus metrics collection."""
    await prometheus_collector.start_collection()


async def stop_prometheus_metrics_collection():
    """Stop Prometheus metrics collection."""
    await prometheus_collector.stop_collection()


def get_prometheus_metrics() -> str:
    """Get all metrics in Prometheus format."""
    return prometheus_collector.get_prometheus_metrics()


def get_prometheus_content_type() -> str:
    """Get Prometheus content type."""
    return prometheus_collector.get_content_type()
