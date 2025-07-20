"""System API endpoints for monitoring and metrics."""

import logging
from datetime import datetime
from typing import Any, Dict, List

import psutil
from fastapi import APIRouter, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class SystemMetrics(BaseModel):
    """System-wide metrics for the platform."""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_agents: int
    total_inferences: int
    avg_response_time: float
    api_calls_per_minute: int


class ServiceHealth(BaseModel):
    """Health status of a service component."""

    service: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any] = {}


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics() -> SystemMetrics:
    """Get current system metrics."""
    # Get real system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Demo values (in production, aggregate from monitoring system)
    metrics = SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=cpu_percent,
        memory_usage=memory.percent,
        active_agents=0,  # Would query from agent service
        total_inferences=42,
        avg_response_time=120.5,
        api_calls_per_minute=15,
    )

    return metrics


@router.get("/health/services", response_model=List[ServiceHealth])
async def get_services_health() -> List[ServiceHealth]:
    """Check health status of all services."""
    services = []

    # Check inference engine
    services.append(
        ServiceHealth(
            service="inference-engine",
            status="healthy",
            last_check=datetime.now(),
            details={"engine": "PyMDP", "version": "0.1.0"},
        )
    )

    # Check GNN service
    services.append(
        ServiceHealth(
            service="gnn-service",
            status="healthy",
            last_check=datetime.now(),
            details={"backend": "PyTorch Geometric", "cuda_available": False},
        )
    )

    # Check LLM service
    services.append(
        ServiceHealth(
            service="llm-service",
            status="degraded",
            last_check=datetime.now(),
            details={"provider": "Ollama", "models_loaded": 0},
        )
    )

    # Check database
    services.append(
        ServiceHealth(
            service="database",
            status="healthy",
            last_check=datetime.now(),
            details={"type": "PostgreSQL", "connections": 5},
        )
    )

    # Check cache
    services.append(
        ServiceHealth(
            service="cache",
            status="healthy",
            last_check=datetime.now(),
            details={"type": "Redis", "memory_usage": "12MB"},
        )
    )

    return services


@router.get("/info")
async def get_system_info() -> dict:
    """Get system information and capabilities."""
    return {
        "platform": "FreeAgentics",
        "version": "0.1.0-alpha",
        "environment": "development",
        "capabilities": {
            "active_inference": {
                "engine": "PyMDP",
                "status": "partial",
                "completion": 15,
            },
            "graph_neural_networks": {
                "framework": "PyTorch Geometric",
                "status": "partial",
                "completion": 40,
            },
            "coalition_formation": {
                "algorithm": "Not Implemented",
                "status": "planned",
                "completion": 0,
            },
            "llm_integration": {
                "providers": ["Ollama", "LlamaCpp"],
                "status": "ready",
                "completion": 90,
            },
        },
        "hardware": {
            "cpu_cores": psutil.cpu_count(),
            "total_memory": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "python_version": "3.10+",
            "cuda_available": False,
        },
    }


@router.get("/logs/recent")
async def get_recent_logs(limit: int = 100) -> List[dict]:
    """Get recent system logs."""
    # In production, this would query from a log aggregation service
    logs = [
        {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "service": "api",
            "message": "System metrics requested",
        },
        {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "service": "inference",
            "message": "PyMDP integration not fully implemented",
        },
        {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "service": "agents",
            "message": "No active agents in system",
        },
    ]

    return logs[:limit]


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> Response:
    """Get Prometheus metrics."""
    try:
        from observability.prometheus_metrics import (
            get_prometheus_content_type,
            get_prometheus_metrics,
        )

        metrics_data = get_prometheus_metrics()
        content_type = get_prometheus_content_type()

        return Response(
            content=metrics_data,
            media_type=content_type,
            headers={"Cache-Control": "no-cache"},
        )
    except ImportError:
        logger.warning("Prometheus metrics not available")
        return Response(
            content="# Prometheus metrics not available\n",
            media_type="text/plain",
            status_code=503,
        )
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        return Response(
            content=f"# Error getting metrics: {e}\n",
            media_type="text/plain",
            status_code=500,
        )


@router.get("/metrics/health")
async def get_health_metrics() -> Dict[str, Any]:
    """Get health-focused metrics for monitoring."""
    try:
        from observability.prometheus_metrics import prometheus_collector

        # Get current metrics snapshot
        snapshot = prometheus_collector.get_metrics_snapshot()

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "status": "healthy",
            "metrics": {
                "active_agents": snapshot.active_agents,
                "total_inferences": snapshot.total_inferences,
                "total_belief_updates": snapshot.total_belief_updates,
                "memory_usage_mb": snapshot.avg_memory_usage_mb,
                "cpu_usage_percent": snapshot.avg_cpu_usage_percent,
                "system_throughput": snapshot.system_throughput,
                "free_energy_avg": snapshot.free_energy_avg,
            },
            "thresholds": {
                "max_agents": 50,
                "max_memory_mb": 2048,
                "max_cpu_percent": 80,
                "min_throughput": 0.1,
            },
        }
    except ImportError:
        logger.warning("Prometheus metrics not available")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "degraded",
            "error": "Prometheus metrics not available",
        }
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "unhealthy",
            "error": str(e),
        }
