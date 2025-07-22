"""Production-Grade Application Performance Monitoring (APM) for FreeAgentics.

This module provides comprehensive APM capabilities including:
- Real-time performance tracking and analytics
- Distributed transaction tracing
- Application topology mapping
- Performance profiling and optimization
- SLA monitoring and alerting
- Resource utilization analytics
- Error tracking and analysis
- Business metrics correlation
"""

import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    open_files: int
    load_average: Tuple[float, float, float]


@dataclass
class TransactionTrace:
    """Distributed transaction trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    component: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "started"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    agent_id: Optional[str] = None
    error: Optional[str] = None

    def finish(self, status: str = "completed", error: Optional[str] = None):
        """Finish the trace span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        if error:
            self.error = error
            self.status = "error"


@dataclass
class ServiceTopology:
    """Service topology representation."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def add_service(self, service_name: str, metadata: Optional[Dict] = None):
        """Add a service to the topology."""
        self.nodes[service_name] = {
            "service": service_name,
            "metadata": metadata or {},
            "first_seen": time.time(),
            "last_seen": time.time(),
            "request_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0,
        }
        self.last_updated = time.time()

    def add_edge(self, source: str, target: str, operation: str):
        """Add an edge between services."""
        edge = {
            "source": source,
            "target": target,
            "operation": operation,
            "weight": 1,
            "last_seen": time.time(),
        }

        # Check if edge exists and update weight
        existing = next(
            (
                e
                for e in self.edges
                if e["source"] == source and e["target"] == target and e["operation"] == operation
            ),
            None,
        )

        if existing:
            existing["weight"] += 1
            existing["last_seen"] = time.time()
        else:
            self.edges.append(edge)

        self.last_updated = time.time()


class ProductionAPM:
    """Production-grade APM system for comprehensive performance monitoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production APM system.

        Args:
            config: APM configuration options
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.sampling_rate = self.config.get("sampling_rate", 1.0)  # 100% by default
        self.trace_buffer_size = self.config.get("trace_buffer_size", 10000)
        self.metrics_interval = self.config.get("metrics_interval", 30)  # seconds

        # Storage for traces and metrics
        self.traces: Dict[str, List[TransactionTrace]] = {}
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.service_topology = ServiceTopology()
        self.active_traces: Dict[str, TransactionTrace] = {}
        self.sla_violations: List[Dict[str, Any]] = []

        # Performance thresholds
        self.thresholds = {
            "response_time_p95": 500,  # ms
            "error_rate": 5.0,  # %
            "cpu_usage": 80.0,  # %
            "memory_usage": 85.0,  # %
            "disk_usage": 90.0,  # %
            "coordination_timeout": 10000,  # ms
        }

        # Metrics collection task
        self.metrics_task: Optional[asyncio.Task] = None
        self.running = False

        # Statistics
        self.stats = {
            "traces_collected": 0,
            "performance_snapshots": 0,
            "sla_violations": 0,
            "services_discovered": 0,
            "start_time": time.time(),
        }

        logger.info("Production APM system initialized")

    async def start(self):
        """Start APM data collection."""
        if not self.enabled or self.running:
            return

        self.running = True
        self.stats["start_time"] = time.time()

        # Start metrics collection task
        self.metrics_task = asyncio.create_task(self._collect_system_metrics())

        logger.info("Production APM system started")

    async def stop(self):
        """Stop APM data collection."""
        if not self.running:
            return

        self.running = False

        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass

        logger.info("Production APM system stopped")

    @asynccontextmanager
    async def trace_transaction(
        self,
        operation: str,
        component: str = "api",
        agent_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing transactions.

        Args:
            operation: Operation being traced
            component: Component name
            agent_id: Associated agent ID
            parent_trace_id: Parent trace ID for distributed tracing
            tags: Additional tags

        Yields:
            TransactionTrace: The trace object
        """
        if not self.enabled or not self._should_sample():
            # Create a no-op trace
            trace = TransactionTrace(
                trace_id="noop",
                span_id="noop",
                parent_span_id=None,
                operation=operation,
                component=component,
                start_time=time.time(),
                agent_id=agent_id,
                tags=tags or {},
            )
            yield trace
            return

        trace_id = str(uuid4())
        span_id = str(uuid4())

        trace = TransactionTrace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_trace_id,
            operation=operation,
            component=component,
            start_time=time.time(),
            agent_id=agent_id,
            tags=tags or {},
        )

        self.active_traces[trace_id] = trace

        try:
            yield trace
            trace.finish("completed")
        except Exception as e:
            trace.finish("error", str(e))
            raise
        finally:
            # Store completed trace
            await self._store_trace(trace)

            # Remove from active traces
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]

    async def record_business_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        """Record business metric for correlation with performance data.

        Args:
            metric_name: Name of the business metric
            value: Metric value
            tags: Additional tags
            timestamp: Metric timestamp
        """
        business_metric = {
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": timestamp or time.time(),
            "component": "business",
        }

        # Store business metric
        if "business_metrics" not in self.traces:
            self.traces["business_metrics"] = []

        self.traces["business_metrics"].append(business_metric)

        # Keep only recent metrics
        cutoff = time.time() - 3600  # 1 hour
        self.traces["business_metrics"] = [
            m for m in self.traces["business_metrics"] if m["timestamp"] > cutoff
        ]

    async def check_sla_violations(self) -> List[Dict[str, Any]]:
        """Check for SLA violations and return violations list.

        Returns:
            List of SLA violations
        """
        violations = []
        now = time.time()

        # Check response time violations
        recent_traces = self._get_recent_traces(300)  # Last 5 minutes
        if recent_traces:
            response_times = [t.duration_ms for t in recent_traces if t.duration_ms]
            if response_times:
                p95 = self._calculate_percentile(response_times, 95)
                if p95 > self.thresholds["response_time_p95"]:
                    violations.append(
                        {
                            "type": "response_time_p95",
                            "threshold": self.thresholds["response_time_p95"],
                            "actual": p95,
                            "timestamp": now,
                            "affected_operations": list(set(t.operation for t in recent_traces)),
                        }
                    )

        # Check error rate violations
        if recent_traces:
            error_count = sum(1 for t in recent_traces if t.status == "error")
            error_rate = (error_count / len(recent_traces)) * 100
            if error_rate > self.thresholds["error_rate"]:
                violations.append(
                    {
                        "type": "error_rate",
                        "threshold": self.thresholds["error_rate"],
                        "actual": error_rate,
                        "timestamp": now,
                        "error_count": error_count,
                        "total_requests": len(recent_traces),
                    }
                )

        # Check system resource violations
        if self.performance_snapshots:
            latest_snapshot = self.performance_snapshots[-1]

            if latest_snapshot.cpu_percent > self.thresholds["cpu_usage"]:
                violations.append(
                    {
                        "type": "cpu_usage",
                        "threshold": self.thresholds["cpu_usage"],
                        "actual": latest_snapshot.cpu_percent,
                        "timestamp": now,
                    }
                )

            if latest_snapshot.memory_percent > self.thresholds["memory_usage"]:
                violations.append(
                    {
                        "type": "memory_usage",
                        "threshold": self.thresholds["memory_usage"],
                        "actual": latest_snapshot.memory_percent,
                        "timestamp": now,
                        "memory_mb": latest_snapshot.memory_mb,
                    }
                )

        # Store violations
        for violation in violations:
            self.sla_violations.append(violation)
            self.stats["sla_violations"] += 1

        # Keep only recent violations
        cutoff = time.time() - 86400  # 24 hours
        self.sla_violations = [v for v in self.sla_violations if v["timestamp"] > cutoff]

        return violations

    def get_performance_analytics(self, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get comprehensive performance analytics.

        Args:
            duration_seconds: Time window for analysis

        Returns:
            Performance analytics data
        """
        cutoff = time.time() - duration_seconds
        recent_traces = self._get_recent_traces(duration_seconds)
        recent_snapshots = [s for s in self.performance_snapshots if s.timestamp > cutoff]

        analytics = {
            "time_window": duration_seconds,
            "timestamp": time.time(),
            "request_analytics": self._analyze_requests(recent_traces),
            "performance_analytics": self._analyze_performance(recent_snapshots),
            "error_analytics": self._analyze_errors(recent_traces),
            "topology_analytics": self._analyze_topology(),
            "sla_analytics": self._analyze_sla_compliance(),
            "business_metrics": self._get_business_metrics(duration_seconds),
        }

        return analytics

    def get_service_topology(self) -> Dict[str, Any]:
        """Get current service topology.

        Returns:
            Service topology data
        """
        return {
            "nodes": list(self.service_topology.nodes.values()),
            "edges": self.service_topology.edges,
            "last_updated": self.service_topology.last_updated,
            "node_count": len(self.service_topology.nodes),
            "edge_count": len(self.service_topology.edges),
        }

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics.

        Returns:
            Real-time metrics
        """
        now = time.time()
        recent_traces = self._get_recent_traces(60)  # Last minute

        # Current system performance
        try:
            current_perf = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0),
                "process_count": len(psutil.pids()),
                "timestamp": now,
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            current_perf = {"error": str(e), "timestamp": now}

        # Active traces
        active_count = len(self.active_traces)

        # Request rate (requests per minute)
        request_rate = len(recent_traces)

        # Error rate
        error_count = sum(1 for t in recent_traces if t.status == "error")
        error_rate = (error_count / len(recent_traces)) * 100 if recent_traces else 0

        # Average response time
        response_times = [t.duration_ms for t in recent_traces if t.duration_ms]
        avg_response_time = statistics.mean(response_times) if response_times else 0

        return {
            "timestamp": now,
            "system_performance": current_perf,
            "application_metrics": {
                "active_traces": active_count,
                "request_rate_per_minute": request_rate,
                "error_rate_percent": error_rate,
                "avg_response_time_ms": avg_response_time,
                "total_requests": len(recent_traces),
                "error_count": error_count,
            },
            "sla_status": {
                "violations_last_hour": len(
                    [v for v in self.sla_violations if v["timestamp"] > now - 3600]
                ),
                "critical_violations": len(
                    [
                        v
                        for v in self.sla_violations
                        if v["timestamp"] > now - 3600
                        and v["type"] in ["error_rate", "response_time_p95"]
                    ]
                ),
            },
        }

    async def generate_performance_report(
        self, duration_hours: int = 24, include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Args:
            duration_hours: Report time window in hours
            include_recommendations: Whether to include recommendations

        Returns:
            Performance report
        """
        duration_seconds = duration_hours * 3600
        analytics = self.get_performance_analytics(duration_seconds)

        report = {
            "report_id": str(uuid4()),
            "generated_at": datetime.now().isoformat(),
            "time_window_hours": duration_hours,
            "analytics": analytics,
            "system_stats": self.stats.copy(),
            "health_score": self._calculate_health_score(analytics),
        }

        if include_recommendations:
            report["recommendations"] = await self._generate_recommendations(analytics)

        return report

    def _should_sample(self) -> bool:
        """Determine if current request should be sampled."""
        import random

        return random.random() < self.sampling_rate

    async def _store_trace(self, trace: TransactionTrace):
        """Store completed trace."""
        operation = trace.operation

        if operation not in self.traces:
            self.traces[operation] = []

        self.traces[operation].append(trace)
        self.stats["traces_collected"] += 1

        # Keep buffer size manageable
        if len(self.traces[operation]) > self.trace_buffer_size:
            self.traces[operation] = self.traces[operation][-self.trace_buffer_size :]

        # Update service topology
        await self._update_topology(trace)

    async def _update_topology(self, trace: TransactionTrace):
        """Update service topology based on trace."""
        service = trace.component

        # Add service node
        if service not in self.service_topology.nodes:
            self.service_topology.add_service(
                service,
                {"agent_id": trace.agent_id, "first_operation": trace.operation},
            )

        # Update service stats
        node = self.service_topology.nodes[service]
        node["last_seen"] = time.time()
        node["request_count"] += 1

        if trace.status == "error":
            node["error_count"] += 1

        if trace.duration_ms:
            current_avg = node["avg_response_time"]
            count = node["request_count"]
            node["avg_response_time"] = ((current_avg * (count - 1)) + trace.duration_ms) / count

        # Add edges for agent coordination
        if trace.agent_id and trace.parent_span_id:
            self.service_topology.add_edge(f"agent-{trace.agent_id}", service, trace.operation)

    async def _collect_system_metrics(self):
        """Collect system performance metrics periodically."""
        while self.running:
            try:
                # Get system metrics
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                    disk_io_read=(
                        psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0
                    ),
                    disk_io_write=(
                        psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0
                    ),
                    network_bytes_sent=psutil.net_io_counters().bytes_sent,
                    network_bytes_recv=psutil.net_io_counters().bytes_recv,
                    process_count=len(psutil.pids()),
                    thread_count=sum(
                        p.num_threads()
                        for p in psutil.process_iter(["num_threads"])
                        if p.info["num_threads"]
                    ),
                    open_files=(
                        len(psutil.Process().open_files())
                        if hasattr(psutil.Process(), "open_files")
                        else 0
                    ),
                    load_average=(
                        psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)
                    ),
                )

                self.performance_snapshots.append(snapshot)
                self.stats["performance_snapshots"] += 1

                # Keep only recent snapshots
                cutoff = time.time() - 86400  # 24 hours
                self.performance_snapshots = [
                    s for s in self.performance_snapshots if s.timestamp > cutoff
                ]

                # Check for SLA violations
                await self.check_sla_violations()

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            await asyncio.sleep(self.metrics_interval)

    def _get_recent_traces(self, duration_seconds: int) -> List[TransactionTrace]:
        """Get traces from the last N seconds."""
        cutoff = time.time() - duration_seconds
        recent_traces = []

        for operation_traces in self.traces.values():
            if isinstance(operation_traces, list):
                recent_traces.extend(
                    [
                        t
                        for t in operation_traces
                        if isinstance(t, TransactionTrace) and t.start_time > cutoff
                    ]
                )

        return recent_traces

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index == int(index):
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _analyze_requests(self, traces: List[TransactionTrace]) -> Dict[str, Any]:
        """Analyze request patterns."""
        if not traces:
            return {"error": "No traces available"}

        response_times = [t.duration_ms for t in traces if t.duration_ms]
        operations = [t.operation for t in traces]

        return {
            "total_requests": len(traces),
            "unique_operations": len(set(operations)),
            "response_time_stats": {
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p90": self._calculate_percentile(response_times, 90),
                "p95": self._calculate_percentile(response_times, 95),
                "p99": self._calculate_percentile(response_times, 99),
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
            },
            "operation_breakdown": {
                op: len([t for t in traces if t.operation == op]) for op in set(operations)
            },
        }

    def _analyze_performance(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze system performance."""
        if not snapshots:
            return {"error": "No performance snapshots available"}

        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]

        return {
            "snapshot_count": len(snapshots),
            "cpu_stats": {
                "mean": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "p95": self._calculate_percentile(cpu_values, 95),
            },
            "memory_stats": {
                "mean": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "p95": self._calculate_percentile(memory_values, 95),
            },
            "load_average": snapshots[-1].load_average if snapshots else (0, 0, 0),
        }

    def _analyze_errors(self, traces: List[TransactionTrace]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_traces = [t for t in traces if t.status == "error"]

        if not error_traces:
            return {"total_errors": 0, "error_rate": 0.0}

        error_rate = (len(error_traces) / len(traces)) * 100 if traces else 0

        return {
            "total_errors": len(error_traces),
            "error_rate": error_rate,
            "error_breakdown": {
                error: len([t for t in error_traces if t.error == error])
                for error in set(t.error for t in error_traces if t.error)
            },
            "error_operations": {
                op: len([t for t in error_traces if t.operation == op])
                for op in set(t.operation for t in error_traces)
            },
        }

    def _analyze_topology(self) -> Dict[str, Any]:
        """Analyze service topology."""
        return {
            "services": len(self.service_topology.nodes),
            "connections": len(self.service_topology.edges),
            "most_active_services": sorted(
                self.service_topology.nodes.items(),
                key=lambda x: x[1]["request_count"],
                reverse=True,
            )[:5],
            "highest_error_services": sorted(
                [
                    (name, node)
                    for name, node in self.service_topology.nodes.items()
                    if node["error_count"] > 0
                ],
                key=lambda x: x[1]["error_count"] / max(x[1]["request_count"], 1),
                reverse=True,
            )[:5],
        }

    def _analyze_sla_compliance(self) -> Dict[str, Any]:
        """Analyze SLA compliance."""
        recent_violations = [
            v for v in self.sla_violations if v["timestamp"] > time.time() - 86400  # Last 24 hours
        ]

        violation_types = {}
        for violation in recent_violations:
            v_type = violation["type"]
            if v_type not in violation_types:
                violation_types[v_type] = []
            violation_types[v_type].append(violation)

        return {
            "total_violations": len(recent_violations),
            "violation_types": {
                v_type: len(violations) for v_type, violations in violation_types.items()
            },
            "compliance_score": max(0, 100 - (len(recent_violations) * 2)),  # Simple scoring
            "critical_violations": len(
                [v for v in recent_violations if v["type"] in ["error_rate", "response_time_p95"]]
            ),
        }

    def _get_business_metrics(self, duration_seconds: int) -> Dict[str, Any]:
        """Get business metrics for the time window."""
        if "business_metrics" not in self.traces:
            return {}

        cutoff = time.time() - duration_seconds
        recent_metrics = [m for m in self.traces["business_metrics"] if m["timestamp"] > cutoff]

        metrics_by_name = {}
        for metric in recent_metrics:
            name = metric["metric_name"]
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(metric["value"])

        return {
            metric_name: {
                "count": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
            }
            for metric_name, values in metrics_by_name.items()
        }

    def _calculate_health_score(self, analytics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0

        # Penalize based on error rate
        error_rate = analytics.get("error_analytics", {}).get("error_rate", 0)
        score -= error_rate * 2  # Each 1% error rate reduces score by 2

        # Penalize based on response time
        p95_response = (
            analytics.get("request_analytics", {}).get("response_time_stats", {}).get("p95", 0)
        )
        if p95_response > self.thresholds["response_time_p95"]:
            score -= (p95_response - self.thresholds["response_time_p95"]) / 10

        # Penalize based on resource usage
        cpu_p95 = analytics.get("performance_analytics", {}).get("cpu_stats", {}).get("p95", 0)
        if cpu_p95 > self.thresholds["cpu_usage"]:
            score -= (cpu_p95 - self.thresholds["cpu_usage"]) / 2

        # Penalize based on SLA violations
        violations = analytics.get("sla_analytics", {}).get("total_violations", 0)
        score -= violations * 3

        return max(0.0, min(100.0, score))

    async def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance recommendations based on analytics."""
        recommendations = []

        # High error rate recommendation
        error_rate = analytics.get("error_analytics", {}).get("error_rate", 0)
        if error_rate > 2:
            recommendations.append(
                {
                    "type": "error_rate",
                    "priority": "high",
                    "title": "High Error Rate Detected",
                    "description": f"Current error rate is {error_rate:.1f}%, exceeding recommended 2%",
                    "recommendations": [
                        "Review error logs for common failure patterns",
                        "Implement circuit breakers for external dependencies",
                        "Add retry logic with exponential backoff",
                        "Consider horizontal scaling if capacity-related",
                    ],
                }
            )

        # High response time recommendation
        p95_response = (
            analytics.get("request_analytics", {}).get("response_time_stats", {}).get("p95", 0)
        )
        if p95_response > 300:
            recommendations.append(
                {
                    "type": "response_time",
                    "priority": "medium",
                    "title": "High Response Time Detected",
                    "description": f"P95 response time is {p95_response:.1f}ms, consider optimization",
                    "recommendations": [
                        "Profile slow operations for bottlenecks",
                        "Implement caching for frequently accessed data",
                        "Optimize database queries and add indexes",
                        "Consider async processing for heavy operations",
                    ],
                }
            )

        # High CPU usage recommendation
        cpu_p95 = analytics.get("performance_analytics", {}).get("cpu_stats", {}).get("p95", 0)
        if cpu_p95 > 70:
            recommendations.append(
                {
                    "type": "cpu_usage",
                    "priority": "medium",
                    "title": "High CPU Usage Detected",
                    "description": f"P95 CPU usage is {cpu_p95:.1f}%, consider scaling",
                    "recommendations": [
                        "Scale horizontally by adding more instances",
                        "Optimize CPU-intensive algorithms",
                        "Implement CPU usage limits per request",
                        "Consider moving heavy processing to background jobs",
                    ],
                }
            )

        return recommendations


# Global APM instance
production_apm = ProductionAPM()


# Convenience functions
async def start_production_apm():
    """Start the production APM system."""
    await production_apm.start()


async def stop_production_apm():
    """Stop the production APM system."""
    await production_apm.stop()


async def trace_transaction(operation: str, **kwargs):
    """Create a transaction trace context."""
    return production_apm.trace_transaction(operation, **kwargs)


async def check_sla_violations():
    """Check for SLA violations."""
    return await production_apm.check_sla_violations()


def get_real_time_metrics():
    """Get real-time performance metrics."""
    return production_apm.get_real_time_metrics()


def get_performance_analytics(duration_seconds: int = 3600):
    """Get performance analytics."""
    return production_apm.get_performance_analytics(duration_seconds)


async def generate_performance_report(duration_hours: int = 24):
    """Generate comprehensive performance report."""
    return await production_apm.generate_performance_report(duration_hours)
