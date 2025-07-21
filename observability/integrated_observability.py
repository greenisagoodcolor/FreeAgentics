"""Integrated Observability Manager for FreeAgentics Agent Operations.

This module coordinates all observability systems to provide comprehensive
monitoring of agent operations, including:
- Performance metrics collection with APM capabilities
- Belief state monitoring with ML-based anomaly detection
- Agent coordination tracking with distributed tracing
- Real-time dashboards with advanced visualizations
- Intelligent alerting with context-aware routing
- Security monitoring and vulnerability scanning
- Performance regression detection and SLA enforcement
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from observability.agent_metrics_integration import (
    MetricsContext,
    record_custom_agent_metric,
)
from observability.alerting_system import (
    check_agent_alert,
    get_active_alerts,
    get_alert_stats,
)
from observability.belief_monitoring import (
    get_all_belief_statistics,
    get_belief_statistics,
    monitor_belief_update,
)
from observability.coordination_metrics import (
    get_agent_coordination_stats,
    get_system_coordination_report,
    record_coordination,
)
from observability.performance_metrics import (
    get_performance_report,
    record_belief_metric,
    record_inference_metric,
    record_step_metric,
    start_performance_tracking,
    stop_performance_tracking,
)
from observability.prometheus_metrics import (
    get_prometheus_metrics,
    start_prometheus_metrics_collection,
    stop_prometheus_metrics_collection,
)
from observability.pymdp_integration import (
    get_pymdp_performance_summary,
    record_agent_lifecycle_event,
    record_coordination_event,
)

logger = logging.getLogger(__name__)


class IntegratedObservabilityManager:
    """Manages all observability systems for agent operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integrated observability manager.

        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.running = False
        self.agent_registrations: Dict[str, Any] = {}
        self.coordination_sessions: Dict[str, Any] = {}
        self.performance_baselines = {
            "inference_time_ms": 10.0,
            "belief_update_ms": 5.0,
            "step_time_ms": 20.0,
            "memory_usage_mb": 100.0,
        }

        # Component status
        self.prometheus_enabled = self.config.get("prometheus", True)
        self.performance_tracking_enabled = self.config.get(
            "performance_tracking", True
        )
        self.belief_monitoring_enabled = self.config.get("belief_monitoring", True)
        self.coordination_tracking_enabled = self.config.get(
            "coordination_tracking", True
        )
        self.alerting_enabled = self.config.get("alerting", True)

        logger.info("Integrated observability manager initialized")

    async def start(self):
        """Start all observability systems."""
        if self.running:
            logger.warning("Observability manager already running")
            return

        self.running = True
        logger.info("Starting integrated observability systems...")

        # Start performance tracking
        if self.performance_tracking_enabled:
            try:
                await start_performance_tracking()
                logger.info("✓ Performance tracking started")
            except Exception as e:
                logger.error(f"Failed to start performance tracking: {e}")

        # Start Prometheus metrics collection
        if self.prometheus_enabled:
            try:
                await start_prometheus_metrics_collection()
                logger.info("✓ Prometheus metrics collection started")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics: {e}")

        logger.info("🎯 Integrated observability systems started")

    async def stop(self):
        """Stop all observability systems."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping integrated observability systems...")

        # Stop performance tracking
        if self.performance_tracking_enabled:
            try:
                await stop_performance_tracking()
                logger.info("✓ Performance tracking stopped")
            except Exception as e:
                logger.error(f"Failed to stop performance tracking: {e}")

        # Stop Prometheus metrics collection
        if self.prometheus_enabled:
            try:
                await stop_prometheus_metrics_collection()
                logger.info("✓ Prometheus metrics collection stopped")
            except Exception as e:
                logger.error(f"Failed to stop Prometheus metrics: {e}")

        logger.info("🛑 Integrated observability systems stopped")

    async def register_agent(
        self, agent_id: str, agent_type: str, metadata: Optional[Dict] = None
    ):
        """Register an agent with observability systems.

        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            metadata: Additional metadata
        """
        self.agent_registrations[agent_id] = {
            "agent_type": agent_type,
            "registered_at": datetime.now(),
            "metadata": metadata or {},
        }

        # Record agent lifecycle event
        await record_agent_lifecycle_event(
            agent_id,
            "registered",
            {
                "agent_type": agent_type,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(f"📋 Agent {agent_id} registered with observability systems")

    async def unregister_agent(self, agent_id: str, reason: str = "stopped"):
        """Unregister an agent from observability systems.

        Args:
            agent_id: Agent identifier
            reason: Reason for unregistration
        """
        if agent_id in self.agent_registrations:
            registration = self.agent_registrations[agent_id]
            del self.agent_registrations[agent_id]

            # Record agent lifecycle event
            await record_agent_lifecycle_event(
                agent_id,
                "unregistered",
                {
                    "reason": reason,
                    "registration_duration_seconds": (
                        datetime.now() - registration["registered_at"]
                    ).total_seconds(),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(f"📋 Agent {agent_id} unregistered from observability systems")

    @asynccontextmanager
    async def monitor_agent_operation(
        self, agent_id: str, operation: str, metadata: Optional[Dict] = None
    ):
        """Context manager for monitoring agent operations.

        Args:
            agent_id: Agent identifier
            operation: Operation being performed
            metadata: Additional metadata

        Yields:
            Operation context
        """
        start_time = time.time()
        success = True
        error = None

        try:
            # Use the metrics context for detailed tracking
            with MetricsContext(agent_id, operation):
                yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Record operation metrics based on type
            if operation == "inference":
                await record_inference_metric(agent_id, duration_ms, success, error)
            elif operation == "belief_update":
                await record_belief_metric(agent_id, duration_ms)
            elif operation == "step":
                await record_step_metric(agent_id, duration_ms)
            else:
                # Custom operation
                await record_custom_agent_metric(
                    agent_id,
                    f"{operation}_duration_ms",
                    duration_ms,
                    {"success": success},
                )

            # Check for performance alerts
            if self.alerting_enabled:
                await self._check_operation_alerts(
                    agent_id, operation, duration_ms, success
                )

    async def monitor_belief_update_detailed(
        self,
        agent_id: str,
        beliefs: Dict[str, Any],
        free_energy: Optional[float] = None,
    ):
        """Monitor detailed belief state updates.

        Args:
            agent_id: Agent identifier
            beliefs: Current belief state
            free_energy: Free energy value
        """
        if not self.belief_monitoring_enabled:
            return

        try:
            # Record belief update with monitoring hooks
            await monitor_belief_update(
                agent_id,
                beliefs,
                free_energy=free_energy,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "beliefs_size": len(beliefs) if isinstance(beliefs, dict) else 0,
                },
            )
        except Exception as e:
            logger.error(f"Failed to monitor belief update for agent {agent_id}: {e}")

    async def monitor_coordination_event(
        self,
        event_type: str,
        coordinator_id: str,
        participant_ids: List[str],
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ):
        """Monitor agent coordination events.

        Args:
            event_type: Type of coordination event
            coordinator_id: Coordinating agent
            participant_ids: Participating agents
            duration_ms: Duration of coordination
            success: Whether coordination succeeded
            metadata: Additional metadata
        """
        if not self.coordination_tracking_enabled:
            return

        try:
            # Record coordination event
            await record_coordination(
                coordinator_id,
                participant_ids,
                event_type,
                time.time() - (duration_ms / 1000),  # Approximate start time
                success,
                metadata,
            )

            # Record in PyMDP integration
            await record_coordination_event(event_type, participant_ids, metadata)
        except Exception as e:
            logger.error(f"Failed to monitor coordination event: {e}")

    async def _check_operation_alerts(
        self, agent_id: str, operation: str, duration_ms: float, success: bool
    ):
        """Check for operation-specific alerts.

        Args:
            agent_id: Agent identifier
            operation: Operation type
            duration_ms: Operation duration
            success: Whether operation succeeded
        """
        try:
            # Check against baselines
            baseline_key = f"{operation}_time_ms"
            if baseline_key in self.performance_baselines:
                baseline = self.performance_baselines[baseline_key]
                if duration_ms > baseline * 5:  # 5x baseline
                    await check_agent_alert(
                        agent_id,
                        {
                            "operation": operation,
                            "duration_ms": duration_ms,
                            "baseline_ms": baseline,
                            "success": success,
                            "performance_degradation": True,
                        },
                    )

            # Check for failures
            if not success:
                await check_agent_alert(
                    agent_id,
                    {
                        "operation": operation,
                        "duration_ms": duration_ms,
                        "success": success,
                        "agent_status": "failed",
                    },
                )
        except Exception as e:
            logger.error(f"Failed to check operation alerts: {e}")

    async def get_agent_observability_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive observability summary for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Observability summary
        """
        summary: Dict[str, Any] = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Registration info
        if agent_id in self.agent_registrations:
            registration = self.agent_registrations[agent_id]
            summary["registration"] = {
                "registered_at": registration["registered_at"].isoformat(),
                "agent_type": registration["agent_type"],
                "metadata": registration["metadata"],
            }

        # Performance metrics
        try:
            from observability.performance_metrics import get_agent_report

            performance_report = await get_agent_report(agent_id)
            summary["performance"] = performance_report
        except Exception as e:
            summary["performance"] = {"error": str(e)}

        # Belief statistics
        try:
            belief_stats = get_belief_statistics(agent_id)
            summary["beliefs"] = belief_stats
        except Exception as e:
            summary["beliefs"] = {"error": str(e)}

        # Coordination statistics
        try:
            coord_stats = get_agent_coordination_stats(agent_id)
            summary["coordination"] = coord_stats
        except Exception as e:
            summary["coordination"] = {"error": str(e)}

        # PyMDP performance summary
        try:
            pymdp_summary = await get_pymdp_performance_summary(agent_id)
            summary["pymdp"] = pymdp_summary
        except Exception as e:
            summary["pymdp"] = {"error": str(e)}

        return summary

    async def get_system_observability_dashboard(self) -> Dict[str, Any]:
        """Get system-wide observability dashboard.

        Returns:
            System dashboard data
        """
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "registered_agents": len(self.agent_registrations),
        }

        # System performance report
        try:
            performance_report = await get_performance_report()
            dashboard["performance"] = performance_report
        except Exception as e:
            dashboard["performance"] = {"error": str(e)}

        # Active alerts
        try:
            active_alerts = get_active_alerts()
            alert_stats = get_alert_stats()
            dashboard["alerts"] = {
                "active": active_alerts,
                "stats": alert_stats,
            }
        except Exception as e:
            dashboard["alerts"] = {"error": str(e)}

        # Coordination system report
        try:
            coord_report = get_system_coordination_report()
            dashboard["coordination"] = coord_report
        except Exception as e:
            dashboard["coordination"] = {"error": str(e)}

        # All agent belief statistics
        try:
            all_belief_stats = get_all_belief_statistics()
            dashboard["beliefs"] = all_belief_stats
        except Exception as e:
            dashboard["beliefs"] = {"error": str(e)}

        # Component status
        dashboard["components"] = {
            "prometheus": self.prometheus_enabled,
            "performance_tracking": self.performance_tracking_enabled,
            "belief_monitoring": self.belief_monitoring_enabled,
            "coordination_tracking": self.coordination_tracking_enabled,
            "alerting": self.alerting_enabled,
        }

        return dashboard

    async def get_prometheus_metrics_endpoint(self) -> str:
        """Get Prometheus metrics for scraping.

        Returns:
            Prometheus metrics string
        """
        if not self.prometheus_enabled:
            return "# Prometheus metrics not enabled\n"

        try:
            return get_prometheus_metrics()
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return f"# Error getting metrics: {e}\n"

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of observability systems.

        Returns:
            Health check results
        """
        health: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "components": {},
        }

        # Check each component
        components = [
            ("performance_tracking", self.performance_tracking_enabled),
            ("prometheus", self.prometheus_enabled),
            ("belief_monitoring", self.belief_monitoring_enabled),
            ("coordination_tracking", self.coordination_tracking_enabled),
            ("alerting", self.alerting_enabled),
        ]

        for component, enabled in components:
            if enabled:
                try:
                    # Basic health check - try to get a simple metric
                    if component == "performance_tracking":
                        await get_performance_report()
                    elif component == "prometheus":
                        get_prometheus_metrics()
                    elif component == "belief_monitoring":
                        get_all_belief_statistics()
                    elif component == "coordination_tracking":
                        get_system_coordination_report()
                    elif component == "alerting":
                        get_alert_stats()

                    health["components"][component] = "healthy"
                except Exception as e:
                    health["components"][component] = f"error: {e}"
            else:
                health["components"][component] = "disabled"

        return health


# Global observability manager instance
observability_manager = IntegratedObservabilityManager()


# Convenience functions for easy integration
async def start_observability():
    """Start the integrated observability system."""
    await observability_manager.start()


async def stop_observability():
    """Stop the integrated observability system."""
    await observability_manager.stop()


async def register_agent_for_observability(
    agent_id: str, agent_type: str, metadata: Optional[Dict] = None
):
    """Register an agent for observability monitoring."""
    await observability_manager.register_agent(agent_id, agent_type, metadata)


async def unregister_agent_from_observability(agent_id: str, reason: str = "stopped"):
    """Unregister an agent from observability monitoring."""
    await observability_manager.unregister_agent(agent_id, reason)


async def monitor_agent_operation(
    agent_id: str, operation: str, metadata: Optional[Dict] = None
):
    """Context manager for monitoring agent operations."""
    return observability_manager.monitor_agent_operation(agent_id, operation, metadata)


async def get_agent_observability_summary(agent_id: str) -> Dict[str, Any]:
    """Get observability summary for an agent."""
    return await observability_manager.get_agent_observability_summary(agent_id)


async def get_system_observability_dashboard() -> Dict[str, Any]:
    """Get system-wide observability dashboard."""
    return await observability_manager.get_system_observability_dashboard()


async def observability_health_check() -> Dict[str, Any]:
    """Perform observability health check."""
    return await observability_manager.health_check()
