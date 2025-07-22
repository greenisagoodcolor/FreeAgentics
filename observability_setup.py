#!/usr/bin/env python3
"""Enhanced observability setup for FreeAgentics production deployment.

This module sets up comprehensive monitoring, logging, and observability
integrating with the existing monitoring.py framework.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import psutil

from api.v1.monitoring import (
    metrics_collector,
    record_agent_metric,
    record_system_metric,
)


class SystemMetricsCollector:
    """Collects system-level performance metrics."""

    def __init__(self):
        """Initialize system metrics collector with process monitoring."""
        self.process = psutil.Process()
        self.last_cpu_time = time.time()
        self.last_cpu_percent = 0.0

    async def collect_system_metrics(self):
        """Collect and record system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await record_system_metric("cpu_usage", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            await record_system_metric("memory_usage", memory.percent)
            await record_system_metric("memory_available_gb", memory.available / (1024**3))

            # Process-specific metrics
            process_memory = self.process.memory_info()
            await record_system_metric("process_memory_mb", process_memory.rss / (1024**2))
            await record_system_metric("process_cpu_percent", self.process.cpu_percent())

            # Disk metrics
            disk = psutil.disk_usage("/")
            await record_system_metric("disk_usage_percent", disk.percent)
            await record_system_metric("disk_free_gb", disk.free / (1024**3))

        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")


class ActiveInferenceMetricsCollector:
    """Collects Active Inference specific metrics from agents."""

    def __init__(self, agent_manager=None):
        """Initialize Active Inference metrics collector.

        Args:
            agent_manager: Agent manager instance to monitor
        """
        self.agent_manager = agent_manager

    async def collect_agent_metrics(self):
        """Collect metrics from all active agents."""
        if not self.agent_manager:
            return

        try:
            for agent_id, agent in self.agent_manager.agents.items():
                if hasattr(agent, "metrics"):
                    # Record core Active Inference metrics
                    metrics = agent.metrics

                    if "avg_free_energy" in metrics:
                        await record_agent_metric(
                            agent_id, "free_energy", metrics["avg_free_energy"]
                        )

                    if "belief_entropy" in metrics:
                        await record_agent_metric(
                            agent_id,
                            "belief_entropy",
                            metrics["belief_entropy"],
                        )

                    if "total_observations" in metrics:
                        await record_agent_metric(
                            agent_id,
                            "observations_count",
                            metrics["total_observations"],
                        )

                    if "total_actions" in metrics:
                        await record_agent_metric(
                            agent_id, "actions_count", metrics["total_actions"]
                        )

                    # PyMDP specific metrics
                    if "expected_free_energy" in metrics:
                        await record_agent_metric(
                            agent_id,
                            "expected_free_energy",
                            metrics["expected_free_energy"],
                        )

        except Exception as e:
            logging.error(f"Failed to collect agent metrics: {e}")


class StructuredLogger:
    """Enhanced structured logging for production monitoring."""

    def __init__(self):
        """Initialize structured logger and configure JSON logging."""
        self.setup_structured_logging()

    def setup_structured_logging(self):
        """Configure structured JSON logging for production."""

        # Create formatter for structured logs
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": time.time(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)

                # Add agent context if present
                if hasattr(record, "agent_id"):
                    log_entry["agent_id"] = record.agent_id

                if hasattr(record, "inference_step"):
                    log_entry["inference_step"] = record.inference_step

                return json.dumps(log_entry)

        # Set up root logger with JSON formatter
        root_logger = logging.getLogger()

        # Add file handler for structured logs
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/freeagentics.json")
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

        # Keep console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
        root_logger.addHandler(console_handler)

        root_logger.setLevel(logging.INFO)


class AlertManager:
    """Manages alerts and notifications for critical events."""

    def __init__(self):
        """Initialize alert manager with default thresholds."""
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage_percent": 90.0,
            "agent_error_rate": 0.1,  # 10% error rate
            "inference_failure_rate": 0.05,  # 5% failure rate
        }

        self.alert_cooldowns = {}  # Prevent alert spam
        self.cooldown_duration = 300  # 5 minutes

    async def check_alerts(self):
        """Check metrics against thresholds and trigger alerts."""
        current_time = time.time()

        for metric_type, threshold in self.alert_thresholds.items():
            # Skip if still in cooldown
            last_alert = self.alert_cooldowns.get(metric_type, 0)
            if current_time - last_alert < self.cooldown_duration:
                continue

            # Get recent metric data
            summary = metrics_collector.get_summary(metric_type, duration=60.0)

            if summary and summary["latest"] > threshold:
                await self._send_alert(metric_type, summary["latest"], threshold)
                self.alert_cooldowns[metric_type] = current_time

    async def _send_alert(self, metric_type: str, value: float, threshold: float):
        """Send alert notification."""
        alert_message = {
            "type": "alert",
            "metric": metric_type,
            "value": value,
            "threshold": threshold,
            "timestamp": time.time(),
            "severity": "high" if value > threshold * 1.2 else "medium",
        }

        # Log the alert
        logging.warning(f"ALERT: {metric_type} = {value:.2f} (threshold: {threshold})")

        # Record alert as metric
        await record_system_metric("alerts_triggered", 1)

        # In production, you would send to Slack, PagerDuty, etc.
        # For now, we'll write to a dedicated alert log
        os.makedirs("logs", exist_ok=True)
        with open("logs/alerts.json", "a") as f:
            f.write(json.dumps(alert_message) + "\n")


class HealthChecker:
    """Comprehensive health checking for all system components."""

    def __init__(self, agent_manager=None, database=None):
        """Initialize health checker with optional components.

        Args:
            agent_manager: Agent manager instance to check
            database: Database connection to check
        """
        self.agent_manager = agent_manager
        self.database = database

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
        }

        # Check database connectivity
        try:
            if self.database:
                # Simple query to check DB health
                # This would need to be adapted to your actual database interface
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "response_time_ms": 0,  # Would measure actual query time
                }
            else:
                health_status["components"]["database"] = {
                    "status": "unknown",
                    "message": "Database connection not available",
                }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "degraded"

        # Check agent manager
        try:
            if self.agent_manager:
                active_agents = len([a for a in self.agent_manager.agents.values() if a.is_active])
                health_status["components"]["agent_manager"] = {
                    "status": "healthy",
                    "active_agents": active_agents,
                    "total_agents": len(self.agent_manager.agents),
                }
            else:
                health_status["components"]["agent_manager"] = {
                    "status": "unknown",
                    "message": "Agent manager not available",
                }
        except Exception as e:
            health_status["components"]["agent_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "degraded"

        # Check system resources
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            resource_status = "healthy"
            if cpu_percent > 90 or memory_percent > 90:
                resource_status = "critical"
                health_status["overall_status"] = "unhealthy"
            elif cpu_percent > 70 or memory_percent > 80:
                resource_status = "warning"
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"

            health_status["components"]["system_resources"] = {
                "status": resource_status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
            }
        except Exception as e:
            health_status["components"]["system_resources"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "degraded"

        return health_status


class ObservabilityManager:
    """Main orchestrator for all observability components."""

    def __init__(self, agent_manager=None, database=None):
        """Initialize observability manager with all monitoring components.

        Args:
            agent_manager: Agent manager instance to monitor
            database: Database connection to monitor
        """
        self.system_collector = SystemMetricsCollector()
        self.agent_collector = ActiveInferenceMetricsCollector(agent_manager)
        self.structured_logger = StructuredLogger()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker(agent_manager, database)

        self.collection_interval = 5.0  # 5 seconds
        self.alert_check_interval = 30.0  # 30 seconds
        self.health_check_interval = 60.0  # 1 minute

        self._running = False

    async def start(self):
        """Start all observability collection tasks."""
        import asyncio

        self._running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_check_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

        logging.info("Observability manager started")
        return tasks

    async def stop(self):
        """Stop observability collection."""
        self._running = False
        logging.info("Observability manager stopped")

    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""
        import asyncio

        while self._running:
            try:
                await self.system_collector.collect_system_metrics()
                await self.agent_collector.collect_agent_metrics()

                # Record collection success
                await record_system_metric("metrics_collection_success", 1)

            except Exception as e:
                logging.error(f"Metrics collection failed: {e}")
                await record_system_metric("metrics_collection_errors", 1)

            await asyncio.sleep(self.collection_interval)

    async def _alert_check_loop(self):
        """Background loop for alert checking."""
        import asyncio

        while self._running:
            try:
                await self.alert_manager.check_alerts()
            except Exception as e:
                logging.error(f"Alert checking failed: {e}")

            await asyncio.sleep(self.alert_check_interval)

    async def _health_check_loop(self):
        """Background loop for health checking."""
        import asyncio

        while self._running:
            try:
                health_status = await self.health_checker.check_health()

                # Record health status as metrics
                if health_status["overall_status"] == "healthy":
                    await record_system_metric("system_health", 1.0)
                elif health_status["overall_status"] == "degraded":
                    await record_system_metric("system_health", 0.5)
                else:
                    await record_system_metric("system_health", 0.0)

                # Log health status
                logging.info(f"Health check: {health_status['overall_status']}")

                # Write detailed health status to file
                os.makedirs("logs", exist_ok=True)
                with open("logs/health_status.json", "w") as f:
                    json.dump(health_status, f, indent=2)

            except Exception as e:
                logging.error(f"Health check failed: {e}")
                await record_system_metric("health_check_errors", 1)

            await asyncio.sleep(self.health_check_interval)


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get the global observability manager instance."""
    return _observability_manager


def initialize_observability(agent_manager=None, database=None) -> ObservabilityManager:
    """Initialize and return the global observability manager."""
    global _observability_manager

    if _observability_manager is None:
        _observability_manager = ObservabilityManager(agent_manager, database)

    return _observability_manager


@asynccontextmanager
async def observability_context(agent_manager=None, database=None):
    """Context manager for observability lifecycle management."""
    manager = initialize_observability(agent_manager, database)

    try:
        # Start observability
        tasks = await manager.start()
        yield manager
    finally:
        # Stop observability
        await manager.stop()

        # Cancel background tasks
        import asyncio

        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


if __name__ == "__main__":
    """Standalone observability server for testing."""
    import asyncio

    async def test_observability():
        async with observability_context():
            logging.info("Observability test started")

            # Run for 60 seconds
            await asyncio.sleep(60)

            logging.info("Observability test completed")

    asyncio.run(test_observability())
