"""Real-time monitoring dashboard API endpoints."""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from observability.monitoring_dashboard import (
    get_dashboard_data,
    get_metric_time_series,
    monitoring_dashboard,
    record_dashboard_event,
    start_monitoring_dashboard,
    stop_monitoring_dashboard,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.on_event("startup")
async def startup_event():
    """Start monitoring dashboard on API startup."""
    try:
        await start_monitoring_dashboard()
        logger.info("Started monitoring dashboard")
    except Exception as e:
        logger.error(f"Failed to start monitoring dashboard: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """Stop monitoring dashboard on API shutdown."""
    try:
        await stop_monitoring_dashboard()
        logger.info("Stopped monitoring dashboard")
    except Exception as e:
        logger.error(f"Failed to stop monitoring dashboard: {e}")


@router.get("/")
async def get_dashboard():
    """Get current dashboard data.

    Returns complete dashboard snapshot including:
    - System-wide metrics
    - Individual agent dashboards
    - Active coalitions
    - Recent events
    - Current alerts
    """
    try:
        dashboard_data = get_dashboard_data()

        if not dashboard_data:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Dashboard not available",
                    "message": "Dashboard is still initializing, please try again",
                },
            )

        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_name}")
async def get_metric_history(
    metric_name: str, duration_minutes: int = Query(default=5, ge=1, le=60)
):
    """Get time series data for a specific metric.

    Args:
        metric_name: Name of the metric (e.g., 'avg_inference_time', 'system_cpu')
        duration_minutes: Duration of history to return (1-60 minutes)

    Returns:
        Time series data with timestamps and values
    """
    try:
        time_series = get_metric_time_series(metric_name, duration_minutes)

        if not time_series:
            raise HTTPException(
                status_code=404, detail=f"Metric '{metric_name}' not found or has no history"
            )

        return time_series

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metric history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    level: Optional[str] = Query(default=None, regex="^(warning|critical)$"),
    limit: int = Query(default=50, ge=1, le=100),
):
    """Get recent alerts.

    Args:
        level: Filter by alert level ('warning' or 'critical')
        limit: Maximum number of alerts to return

    Returns:
        List of recent alerts
    """
    try:
        # Get alerts from history
        all_alerts = list(monitoring_dashboard.alert_history)

        # Filter by level if specified
        if level:
            all_alerts = [a for a in all_alerts if a.get("level") == level]

        # Apply limit
        alerts = all_alerts[-limit:]

        return {"total": len(all_alerts), "returned": len(alerts), "alerts": alerts}

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(
    event_type: Optional[str] = Query(default=None), limit: int = Query(default=20, ge=1, le=100)
):
    """Get recent dashboard events.

    Args:
        event_type: Filter by event type
        limit: Maximum number of events to return

    Returns:
        List of recent events
    """
    try:
        # Get events from history
        all_events = list(monitoring_dashboard.event_history)

        # Filter by type if specified
        if event_type:
            all_events = [e for e in all_events if e.get("type") == event_type]

        # Apply limit
        events = all_events[-limit:]

        return {"total": len(all_events), "returned": len(events), "events": events}

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def record_event(event_data: Dict):
    """Record a custom dashboard event.

    Args:
        event_data: Event data with 'type' and optional 'data' fields

    Returns:
        Success confirmation
    """
    try:
        event_type = event_data.get("type")
        if not event_type:
            raise HTTPException(status_code=400, detail="Event type is required")

        data = event_data.get("data", {})

        record_dashboard_event(event_type, data)

        return {"status": "success", "message": "Event recorded"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def dashboard_health():
    """Check dashboard health status.

    Returns:
        Dashboard health information
    """
    try:
        dashboard_data = get_dashboard_data()

        health = {
            "status": "healthy" if dashboard_data else "initializing",
            "running": monitoring_dashboard.running,
            "update_interval": monitoring_dashboard.update_interval,
            "has_data": dashboard_data is not None,
        }

        if dashboard_data:
            health["last_update"] = dashboard_data.get("timestamp")
            health["agent_count"] = len(dashboard_data.get("agent_dashboards", {}))
            health["alert_count"] = len(dashboard_data.get("alerts", []))

        return health

    except Exception as e:
        logger.error(f"Failed to check dashboard health: {e}")
        return {"status": "error", "error": str(e), "running": False}
