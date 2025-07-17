"""
WebSocket Connection Pool Monitoring

Provides monitoring, metrics collection, and dashboard endpoints for
WebSocket connection pool and resource management.
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from websocket.connection_pool import ConnectionState, WebSocketConnectionPool
from websocket.resource_manager import AgentResourceManager, ResourceState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring/websocket", tags=["websocket-monitoring"])


class MetricSnapshot(BaseModel):
    """A single metric snapshot."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeSeriesMetric:
    """Stores time series data for a metric."""

    def __init__(self, max_size: int = 1000, window_minutes: int = 60):
        self.max_size = max_size
        self.window_minutes = window_minutes
        self.data: deque[MetricSnapshot] = deque(maxlen=max_size)

    def add_value(self, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a new value to the time series."""
        snapshot = MetricSnapshot(value=value, metadata=metadata or {})
        self.data.append(snapshot)

    def get_recent(self, minutes: Optional[int] = None) -> List[MetricSnapshot]:
        """Get recent data points within the specified time window."""
        if minutes is None:
            minutes = self.window_minutes

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [s for s in self.data if s.timestamp > cutoff_time]

    def get_stats(self, minutes: Optional[int] = None) -> Dict[str, float]:
        """Get statistics for recent data."""
        recent = self.get_recent(minutes)
        if not recent:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "current": 0.0, "count": 0}

        values = [s.value for s in recent]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "current": recent[-1].value if recent else 0.0,
            "count": len(values),
        }


class ConnectionPoolMonitor:
    """Monitors WebSocket connection pool metrics."""

    def __init__(self, pool: WebSocketConnectionPool, resource_manager: AgentResourceManager):
        self.pool = pool
        self.resource_manager = resource_manager

        # Time series metrics
        self.pool_size = TimeSeriesMetric()
        self.pool_utilization = TimeSeriesMetric()
        self.available_connections = TimeSeriesMetric()
        self.in_use_connections = TimeSeriesMetric()
        self.acquisition_wait_time = TimeSeriesMetric()
        self.health_check_failures = TimeSeriesMetric()
        self.total_agents = TimeSeriesMetric()
        self.active_agents = TimeSeriesMetric()
        self.memory_usage = TimeSeriesMetric()
        self.cpu_usage = TimeSeriesMetric()

        # Event logs
        self.events: deque[Dict[str, Any]] = deque(maxlen=1000)

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._collection_interval = 10.0  # Collect metrics every 10 seconds

    def start(self):
        """Start monitoring."""
        if not self._running:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("Connection pool monitoring started")

    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Connection pool monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._collection_interval)

    async def _collect_metrics(self):
        """Collect current metrics from pool and resource manager."""
        # Pool metrics
        pool_metrics = self.pool.get_metrics()
        self.pool_size.add_value(pool_metrics.get("pool_size", 0))
        self.pool_utilization.add_value(pool_metrics.get("utilization", 0))
        self.available_connections.add_value(pool_metrics.get("available_connections", 0))
        self.in_use_connections.add_value(pool_metrics.get("in_use_connections", 0))
        self.acquisition_wait_time.add_value(pool_metrics.get("average_wait_time", 0))
        self.health_check_failures.add_value(pool_metrics.get("health_check_failures", 0))

        # Resource manager metrics
        resource_metrics = self.resource_manager.get_metrics()
        self.total_agents.add_value(resource_metrics.get("total_agents", 0))
        self.active_agents.add_value(resource_metrics.get("active_agents", 0))
        self.memory_usage.add_value(resource_metrics.get("total_memory_usage", 0))
        self.cpu_usage.add_value(resource_metrics.get("total_cpu_usage", 0))

    def log_event(
        self, event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a monitoring event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "description": description,
            "metadata": metadata or {},
        }
        self.events.append(event)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "summary": self._get_summary(),
            "time_series": self._get_time_series_data(),
            "connection_details": self._get_connection_details(),
            "resource_details": self._get_resource_details(),
            "recent_events": list(self.events)[-50:],  # Last 50 events
            "health_status": self._get_health_status(),
        }

    def _get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        pool_metrics = self.pool.get_metrics()
        resource_metrics = self.resource_manager.get_metrics()

        return {
            "pool": {
                "size": pool_metrics.get("pool_size", 0),
                "available": pool_metrics.get("available_connections", 0),
                "in_use": pool_metrics.get("in_use_connections", 0),
                "utilization": round(pool_metrics.get("utilization", 0) * 100, 1),
                "total_acquisitions": pool_metrics.get("total_acquisitions", 0),
                "failed_acquisitions": pool_metrics.get("failed_acquisitions", 0),
            },
            "resources": {
                "total_agents": resource_metrics.get("total_agents", 0),
                "active_agents": resource_metrics.get("active_agents", 0),
                "total_memory_mb": round(
                    resource_metrics.get("total_memory_usage", 0) / (1024 * 1024), 2
                ),
                "total_cpu_cores": round(resource_metrics.get("total_cpu_usage", 0), 2),
                "connections_in_use": resource_metrics.get("connections_in_use", 0),
            },
        }

    def _get_time_series_data(self) -> Dict[str, Any]:
        """Get time series data for charts."""
        return {
            "pool_size": {
                "data": [
                    {"t": s.timestamp.isoformat(), "v": s.value}
                    for s in self.pool_size.get_recent(30)
                ],
                "stats": self.pool_size.get_stats(30),
            },
            "pool_utilization": {
                "data": [
                    {"t": s.timestamp.isoformat(), "v": s.value * 100}
                    for s in self.pool_utilization.get_recent(30)
                ],
                "stats": self.pool_utilization.get_stats(30),
            },
            "active_agents": {
                "data": [
                    {"t": s.timestamp.isoformat(), "v": s.value}
                    for s in self.active_agents.get_recent(30)
                ],
                "stats": self.active_agents.get_stats(30),
            },
            "memory_usage_mb": {
                "data": [
                    {"t": s.timestamp.isoformat(), "v": s.value / (1024 * 1024)}
                    for s in self.memory_usage.get_recent(30)
                ],
                "stats": {
                    k: v / (1024 * 1024) if k != "count" else v
                    for k, v in self.memory_usage.get_stats(30).items()
                },
            },
        }

    def _get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed connection information."""
        connections = self.pool.get_connection_info()

        # Enhance with agent information
        for conn in connections:
            conn_id = conn["connection_id"]
            # Get agents on this connection
            agents = []
            for resource in self.resource_manager.get_resource_info():
                if resource["connection_id"] == conn_id:
                    agents.append(
                        {
                            "agent_id": resource["agent_id"],
                            "state": resource["state"],
                            "memory_mb": round(resource["memory_usage"] / (1024 * 1024), 2),
                            "cpu": round(resource["cpu_usage"], 2),
                        }
                    )
            conn["agents"] = agents
            conn["agent_count"] = len(agents)

        return connections

    def _get_resource_details(self) -> List[Dict[str, Any]]:
        """Get detailed resource information."""
        return self.resource_manager.get_resource_info()

    def _get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        pool_metrics = self.pool.get_metrics()
        resource_metrics = self.resource_manager.get_metrics()

        # Calculate health score (0-100)
        health_score = 100
        issues = []

        # Check pool utilization
        utilization = pool_metrics.get("utilization", 0)
        if utilization > 0.9:
            health_score -= 20
            issues.append("Pool utilization very high (>90%)")
        elif utilization > 0.8:
            health_score -= 10
            issues.append("Pool utilization high (>80%)")

        # Check failed acquisitions
        failed_rate = pool_metrics.get("failed_acquisitions", 0) / max(
            1, pool_metrics.get("total_acquisitions", 1)
        )
        if failed_rate > 0.1:
            health_score -= 30
            issues.append(f"High failure rate ({failed_rate:.1%})")
        elif failed_rate > 0.05:
            health_score -= 15
            issues.append(f"Moderate failure rate ({failed_rate:.1%})")

        # Check health check failures
        recent_health_failures = self.health_check_failures.get_stats(10)
        if recent_health_failures["max"] > 5:
            health_score -= 20
            issues.append("Frequent health check failures")

        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        else:
            status = "critical"

        return {
            "status": status,
            "score": health_score,
            "issues": issues,
            "last_check": datetime.utcnow().isoformat(),
        }


# Global monitor instance (to be initialized with pool and resource manager)
monitor: Optional[ConnectionPoolMonitor] = None


def initialize_monitor(pool: WebSocketConnectionPool, resource_manager: AgentResourceManager):
    """Initialize the global monitor instance."""
    global monitor
    monitor = ConnectionPoolMonitor(pool, resource_manager)
    monitor.start()
    logger.info("WebSocket monitoring initialized")


# API Endpoints


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    return monitor.get_dashboard_data()


@router.get("/metrics")
async def get_current_metrics():
    """Get current metrics snapshot."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    summary = monitor._get_summary()
    health = monitor._get_health_status()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pool_metrics": summary["pool"],
        "resource_metrics": summary["resources"],
        "health": health,
    }


@router.get("/connections")
async def get_connection_details():
    """Get detailed information about all connections."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    return {
        "connections": monitor._get_connection_details(),
        "total": len(monitor._get_connection_details()),
    }


@router.get("/resources")
async def get_resource_details():
    """Get detailed information about all agent resources."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    resources = monitor._get_resource_details()

    # Group by state
    by_state = {}
    for resource in resources:
        state = resource["state"]
        if state not in by_state:
            by_state[state] = []
        by_state[state].append(resource)

    return {"resources": resources, "by_state": by_state, "total": len(resources)}


@router.get("/timeseries/{metric}")
async def get_metric_timeseries(metric: str, minutes: int = 30):
    """Get time series data for a specific metric."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    valid_metrics = [
        "pool_size",
        "pool_utilization",
        "available_connections",
        "in_use_connections",
        "acquisition_wait_time",
        "health_check_failures",
        "total_agents",
        "active_agents",
        "memory_usage",
        "cpu_usage",
    ]

    if metric not in valid_metrics:
        raise HTTPException(
            status_code=400, detail=f"Invalid metric. Valid metrics: {valid_metrics}"
        )

    metric_obj = getattr(monitor, metric)

    return {
        "metric": metric,
        "window_minutes": minutes,
        "data": [
            {"timestamp": s.timestamp.isoformat(), "value": s.value}
            for s in metric_obj.get_recent(minutes)
        ],
        "stats": metric_obj.get_stats(minutes),
    }


@router.get("/events")
async def get_recent_events(limit: int = 100):
    """Get recent monitoring events."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    events = list(monitor.events)
    if limit:
        events = events[-limit:]

    return {"events": events, "total": len(events)}


@router.post("/events")
async def log_monitoring_event(
    event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None
):
    """Log a custom monitoring event."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    monitor.log_event(event_type, description, metadata)

    return {"status": "event_logged", "timestamp": datetime.utcnow().isoformat()}


# Performance benchmarking endpoint


class BenchmarkRequest(BaseModel):
    """Request for performance benchmark."""

    concurrent_agents: int = Field(default=10, ge=1, le=1000)
    duration_seconds: int = Field(default=60, ge=10, le=600)
    message_rate: float = Field(default=1.0, ge=0.1, le=100.0)


class BenchmarkResult(BaseModel):
    """Result of performance benchmark."""

    total_agents: int
    duration: float
    total_messages: int
    successful_acquisitions: int
    failed_acquisitions: int
    avg_acquisition_time: float
    avg_message_latency: float
    peak_pool_size: int
    peak_memory_mb: float
    errors: List[str]


@router.post("/benchmark", response_model=BenchmarkResult)
async def run_performance_benchmark(request: BenchmarkRequest):
    """Run a performance benchmark on the connection pool."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")

    logger.info(
        f"Starting benchmark: {request.concurrent_agents} agents for {request.duration_seconds}s"
    )

    start_time = time.time()
    errors = []
    message_count = 0
    acquisition_times = []

    # Start monitoring peak values
    initial_pool_size = monitor.pool.size
    peak_pool_size = initial_pool_size
    peak_memory = 0

    async def agent_workload(agent_id: str):
        """Simulate agent workload."""
        nonlocal message_count, peak_pool_size, peak_memory

        try:
            # Allocate resources
            acq_start = time.time()
            resource = await monitor.resource_manager.allocate_resource(agent_id)
            acquisition_times.append(time.time() - acq_start)

            # Activate agent
            await monitor.resource_manager.activate_resource(agent_id)

            # Simulate work
            work_start = time.time()
            while (time.time() - work_start) < request.duration_seconds:
                # Simulate sending message
                await asyncio.sleep(1.0 / request.message_rate)
                message_count += 1

                # Update metrics
                memory = 10 * 1024 * 1024  # 10MB per agent
                cpu = 0.1
                await monitor.resource_manager.update_resource_usage(
                    agent_id, memory=memory, cpu=cpu
                )

                # Track peaks
                current_pool_size = monitor.pool.size
                if current_pool_size > peak_pool_size:
                    peak_pool_size = current_pool_size

                current_memory = sum(
                    r.memory_usage for r in monitor.resource_manager._resources.values()
                )
                if current_memory > peak_memory:
                    peak_memory = current_memory

            # Release resources
            await monitor.resource_manager.release_resource(agent_id)

        except Exception as e:
            errors.append(f"Agent {agent_id}: {str(e)}")

    # Run concurrent agents
    tasks = []
    for i in range(request.concurrent_agents):
        agent_id = f"benchmark-agent-{i}"
        tasks.append(agent_workload(agent_id))

    # Wait for all agents to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    # Calculate results
    duration = time.time() - start_time
    pool_metrics = monitor.pool.get_metrics()

    result = BenchmarkResult(
        total_agents=request.concurrent_agents,
        duration=duration,
        total_messages=message_count,
        successful_acquisitions=len(acquisition_times),
        failed_acquisitions=len(errors),
        avg_acquisition_time=(
            sum(acquisition_times) / len(acquisition_times) if acquisition_times else 0
        ),
        avg_message_latency=duration / message_count if message_count else 0,
        peak_pool_size=peak_pool_size,
        peak_memory_mb=peak_memory / (1024 * 1024),
        errors=errors[:10],  # First 10 errors
    )

    # Log benchmark event
    monitor.log_event(
        "benchmark_completed",
        f"Benchmark with {request.concurrent_agents} agents",
        {"duration": duration, "messages": message_count, "errors": len(errors)},
    )

    logger.info(f"Benchmark completed: {message_count} messages in {duration:.2f}s")

    return result
