"""Real-time monitoring endpoints for agents and system performance."""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class MetricPoint(BaseModel):
    """Single metric data point."""

    timestamp: float = Field(default_factory=time.time)
    value: float
    agent_id: Optional[str] = None
    metric_type: str


class MonitoringSession(BaseModel):
    """Monitoring session configuration."""

    session_id: str
    client_id: str
    metrics: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    sample_rate: float = 1.0  # Seconds between samples
    buffer_size: int = 1000


class MetricsCollector:
    """Collects and manages system and agent metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        # Metric buffers: metric_type -> deque of MetricPoints
        self.metrics: Dict[str, Deque[MetricPoint]] = {}
        self.buffer_size = 10000

        # Performance counters
        self.counters = {
            "total_inferences": 0,
            "active_agents": 0,
            "messages_processed": 0,
            "errors": 0,
        }

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default metric types."""
        default_metrics = [
            "cpu_usage",
            "memory_usage",
            "inference_rate",
            "agent_count",
            "world_updates",
            "message_throughput",
        ]

        for metric in default_metrics:
            self.metrics[metric] = deque(maxlen=self.buffer_size)

    def record_metric(
        self, metric_type: str, value: float, agent_id: Optional[str] = None
    ):
        """Record a metric value."""
        if metric_type not in self.metrics:
            self.metrics[metric_type] = deque(maxlen=self.buffer_size)

        point = MetricPoint(
            value=value, agent_id=agent_id, metric_type=metric_type
        )

        self.metrics[metric_type].append(point)

    def get_metrics(
        self,
        metric_type: str,
        duration: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for a specific type and optional time window."""
        if metric_type not in self.metrics:
            return []

        metrics = list(self.metrics[metric_type])

        # Filter by time window if specified
        if duration:
            cutoff_time = time.time() - duration
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        # Filter by agent if specified
        if agent_id:
            metrics = [m for m in metrics if m.agent_id == agent_id]

        return [m.dict() for m in metrics]

    def get_summary(
        self, metric_type: str, duration: float = 60.0
    ) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        metrics = self.get_metrics(metric_type, duration)

        if not metrics:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0,
            }

        values = [m["value"] for m in metrics]

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }

    def increment_counter(self, counter: str, amount: int = 1):
        """Increment a performance counter."""
        if counter in self.counters:
            self.counters[counter] += amount

    def get_counters(self) -> Dict[str, int]:
        """Get all performance counters."""
        return self.counters.copy()


# Global metrics collector
metrics_collector = MetricsCollector()


class MonitoringManager:
    """Manages monitoring sessions and metric streaming."""

    def __init__(self, collector: MetricsCollector):
        """Initialize monitoring manager."""
        self.collector = collector
        self.sessions: Dict[str, MonitoringSession] = {}
        self.active_streams: Dict[str, asyncio.Task] = {}

    async def start_session(
        self, websocket: WebSocket, session: MonitoringSession
    ):
        """Start a monitoring session."""
        self.sessions[session.session_id] = session

        # Start metric streaming task
        task = asyncio.create_task(self._stream_metrics(websocket, session))
        self.active_streams[session.session_id] = task

        logger.info(f"Started monitoring session {session.session_id}")

    async def stop_session(self, session_id: str):
        """Stop a monitoring session."""
        if session_id in self.active_streams:
            task = self.active_streams[session_id]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            del self.active_streams[session_id]

        if session_id in self.sessions:
            del self.sessions[session_id]

        logger.info(f"Stopped monitoring session {session_id}")

    async def _stream_metrics(
        self, websocket: WebSocket, session: MonitoringSession
    ):
        """Stream metrics to a WebSocket client."""
        try:
            while True:
                # Collect metrics for this session
                metrics_data = {}

                for metric_type in session.metrics:
                    # Get recent metrics
                    metrics = self.collector.get_metrics(
                        metric_type,
                        duration=session.sample_rate * 10,  # Last 10 samples
                    )

                    # Filter by agents if specified
                    if session.agents:
                        metrics = [
                            m
                            for m in metrics
                            if not m.get("agent_id")
                            or m["agent_id"] in session.agents
                        ]

                    if metrics:
                        metrics_data[metric_type] = metrics[-1]  # Latest value

                # Send metrics update
                message = {
                    "type": "metrics_update",
                    "session_id": session.session_id,
                    "timestamp": time.time(),
                    "metrics": metrics_data,
                    "counters": self.collector.get_counters(),
                }

                await websocket.send_json(message)

                # Wait for next sample
                await asyncio.sleep(session.sample_rate)

        except asyncio.CancelledError:
            logger.info(
                f"Metric streaming cancelled for session {session.session_id}"
            )
        except Exception as e:
            logger.error(f"Error streaming metrics: {e}")


# Global monitoring manager
monitoring_manager = MonitoringManager(metrics_collector)


@router.websocket("/ws/monitor/{client_id}")
async def monitor_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time monitoring."""
    await websocket.accept()

    session_id = f"monitor_{client_id}_{int(time.time())}"
    session = None

    try:
        while True:
            # Receive configuration or commands
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "start_monitoring":
                    # Start monitoring session
                    config = message.get("config", {})
                    session = MonitoringSession(
                        session_id=session_id,
                        client_id=client_id,
                        metrics=config.get(
                            "metrics", ["cpu_usage", "memory_usage"]
                        ),
                        agents=config.get("agents", []),
                        sample_rate=config.get("sample_rate", 1.0),
                        buffer_size=config.get("buffer_size", 1000),
                    )

                    await monitoring_manager.start_session(websocket, session)

                    await websocket.send_json(
                        {
                            "type": "monitoring_started",
                            "session_id": session_id,
                            "config": session.dict(),
                        }
                    )

                elif msg_type == "stop_monitoring":
                    # Stop monitoring session
                    await monitoring_manager.stop_session(session_id)

                    await websocket.send_json(
                        {
                            "type": "monitoring_stopped",
                            "session_id": session_id,
                        }
                    )

                elif msg_type == "get_summary":
                    # Get metric summary
                    metric_type = message.get("metric_type")
                    duration = message.get("duration", 60.0)

                    if metric_type:
                        summary = metrics_collector.get_summary(
                            metric_type, duration
                        )

                        await websocket.send_json(
                            {
                                "type": "metric_summary",
                                "metric_type": metric_type,
                                "summary": summary,
                            }
                        )

                elif msg_type == "ping":
                    # Health check
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": time.time(),
                        }
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                    }
                )
            except Exception as e:
                logger.error(f"Error handling monitoring message: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )

    except WebSocketDisconnect:
        if session_id:
            await monitoring_manager.stop_session(session_id)
        logger.info(f"Monitoring client {client_id} disconnected")


# REST endpoints for metric access
@router.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    duration: Optional[float] = 60.0,
    agent_id: Optional[str] = None,
):
    """Get metrics for a specific type."""
    metrics = metrics_collector.get_metrics(metric_type, duration, agent_id)
    summary = metrics_collector.get_summary(metric_type, duration)

    return {
        "metric_type": metric_type,
        "duration": duration,
        "agent_id": agent_id,
        "summary": summary,
        "data": metrics,
    }


@router.get("/metrics/types")
async def get_metric_types():
    """Get available metric types."""
    return {
        "metric_types": list(metrics_collector.metrics.keys()),
        "counters": list(metrics_collector.counters.keys()),
    }


@router.get("/metrics/counters")
async def get_counters():
    """Get performance counters."""
    return metrics_collector.get_counters()


@router.get("/beliefs/stats")
async def get_all_belief_stats():
    """Get belief monitoring statistics for all agents."""
    try:
        from observability.belief_monitoring import get_all_belief_statistics

        return get_all_belief_statistics()
    except ImportError:
        return {"error": "Belief monitoring not available"}


@router.get("/beliefs/stats/{agent_id}")
async def get_agent_belief_stats(agent_id: str):
    """Get belief monitoring statistics for a specific agent."""
    try:
        from observability.belief_monitoring import get_belief_statistics

        return get_belief_statistics(agent_id)
    except ImportError:
        return {"error": "Belief monitoring not available"}


@router.post("/beliefs/reset")
async def reset_belief_monitoring():
    """Reset belief monitoring for all agents."""
    try:
        from observability.belief_monitoring import belief_monitoring_hooks

        belief_monitoring_hooks.reset_all()
        return {"message": "Belief monitoring reset successfully"}
    except ImportError:
        return {"error": "Belief monitoring not available"}


@router.post("/beliefs/reset/{agent_id}")
async def reset_agent_belief_monitoring(agent_id: str):
    """Reset belief monitoring for a specific agent."""
    try:
        from observability.belief_monitoring import belief_monitoring_hooks

        belief_monitoring_hooks.reset_agent_monitor(agent_id)
        return {"message": f"Belief monitoring reset for agent {agent_id}"}
    except ImportError:
        return {"error": "Belief monitoring not available"}


@router.get("/coordination/stats")
async def get_coordination_stats():
    """Get coordination statistics for all agents."""
    try:
        from observability.coordination_metrics import (
            get_system_coordination_report,
        )

        return get_system_coordination_report()
    except ImportError:
        return {"error": "Coordination metrics not available"}


@router.get("/coordination/stats/{agent_id}")
async def get_agent_coordination_stats(agent_id: str):
    """Get coordination statistics for a specific agent."""
    try:
        from observability.coordination_metrics import (
            get_agent_coordination_stats,
        )

        return get_agent_coordination_stats(agent_id)
    except ImportError:
        return {"error": "Coordination metrics not available"}


@router.get("/coordination/coalitions")
async def get_coalition_statistics():
    """Get coalition statistics."""
    try:
        from observability.coordination_metrics import coordination_metrics

        return coordination_metrics.get_coalition_statistics()
    except ImportError:
        return {"error": "Coordination metrics not available"}


# Simulation metrics recording functions
async def record_agent_metric(
    agent_id: str,
    metric_type: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Record an agent-specific metric."""
    # Note: metadata parameter is accepted for compatibility but currently unused
    metrics_collector.record_metric(metric_type, value, agent_id)


async def record_system_metric(
    metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None
):
    """Record a system-wide metric."""
    # Note: metadata parameter is accepted for compatibility but currently unused
    metrics_collector.record_metric(metric_type, value)


async def increment_counter(counter: str, amount: int = 1):
    """Increment a performance counter."""
    metrics_collector.increment_counter(counter, amount)
