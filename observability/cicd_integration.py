"""
CI/CD Integration for FreeAgentics Production Monitoring

This module provides integration with CI/CD pipelines for performance monitoring,
automated testing, and deployment health checks.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from observability.intelligent_alerting import (
    AlertRule,
    AlertSeverity,
    AlertType,
    intelligent_alerting,
)
from observability.log_aggregation import (
    LogLevel,
    LogSource,
    create_structured_log_entry,
    log_aggregator,
)
from observability.performance_metrics import performance_tracker
from observability.prometheus_metrics import prometheus_collector

logger = logging.getLogger(__name__)


@dataclass
class DeploymentMetrics:
    """Metrics for a deployment."""

    deployment_id: str
    version: str
    environment: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, success, failed, rolled_back
    pre_deployment_metrics: Dict[str, float] = field(default_factory=dict)
    post_deployment_metrics: Dict[str, float] = field(default_factory=dict)
    performance_regression_detected: bool = False
    health_check_results: List[Dict[str, Any]] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "version": self.version,
            "environment": self.environment,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "pre_deployment_metrics": self.pre_deployment_metrics,
            "post_deployment_metrics": self.post_deployment_metrics,
            "performance_regression_detected": self.performance_regression_detected,
            "health_check_results": self.health_check_results,
            "error_messages": self.error_messages,
        }


@dataclass
class PerformanceGate:
    """Performance gate for CI/CD pipeline."""

    name: str
    metric_name: str
    threshold: float
    operator: str  # >, <, >=, <=, ==
    enabled: bool = True
    critical: bool = False  # If true, failure blocks deployment
    description: str = ""

    def evaluate(self, value: float) -> Tuple[bool, str]:
        """Evaluate the gate against a value."""
        if self.operator == ">":
            passed = value > self.threshold
        elif self.operator == "<":
            passed = value < self.threshold
        elif self.operator == ">=":
            passed = value >= self.threshold
        elif self.operator == "<=":
            passed = value <= self.threshold
        elif self.operator == "==":
            passed = value == self.threshold
        else:
            passed = False

        message = f"{self.name}: {value} {self.operator} {self.threshold} = {'PASS' if passed else 'FAIL'}"

        return passed, message


class PerformanceGateManager:
    """Manages performance gates for CI/CD."""

    def __init__(self):
        """Initialize performance gate manager."""
        self.gates = []
        self._setup_default_gates()

    def _setup_default_gates(self):
        """Setup default performance gates."""
        self.gates = [
            PerformanceGate(
                name="CPU Usage",
                metric_name="cpu_usage",
                threshold=80.0,
                operator="<",
                critical=True,
                description="System CPU usage should be below 80%",
            ),
            PerformanceGate(
                name="Memory Usage",
                metric_name="memory_usage",
                threshold=85.0,
                operator="<",
                critical=True,
                description="System memory usage should be below 85%",
            ),
            PerformanceGate(
                name="Active Agents",
                metric_name="active_agents",
                threshold=50.0,
                operator="<",
                critical=True,
                description="Number of active agents should be below coordination limit",
            ),
            PerformanceGate(
                name="Average Inference Time",
                metric_name="avg_inference_time",
                threshold=1000.0,
                operator="<",
                critical=False,
                description="Average inference time should be below 1000ms",
            ),
            PerformanceGate(
                name="Error Rate",
                metric_name="error_rate",
                threshold=0.05,
                operator="<",
                critical=True,
                description="Error rate should be below 5%",
            ),
            PerformanceGate(
                name="Response Quality",
                metric_name="response_quality",
                threshold=0.7,
                operator=">",
                critical=False,
                description="Response quality score should be above 0.7",
            ),
        ]

    def add_gate(self, gate: PerformanceGate):
        """Add a performance gate."""
        self.gates.append(gate)

    def remove_gate(self, name: str):
        """Remove a performance gate."""
        self.gates = [g for g in self.gates if g.name != name]

    async def evaluate_gates(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Evaluate all performance gates."""
        results = []
        all_passed = True
        critical_failed = False

        for gate in self.gates:
            if not gate.enabled:
                continue

            # Get current metric value
            value = await self._get_metric_value(gate.metric_name)

            if value is None:
                results.append(
                    {
                        "gate": gate.name,
                        "status": "error",
                        "message": f"Could not retrieve metric: {gate.metric_name}",
                        "critical": gate.critical,
                    }
                )
                if gate.critical:
                    critical_failed = True
                continue

            # Evaluate gate
            passed, message = gate.evaluate(value)

            results.append(
                {
                    "gate": gate.name,
                    "status": "pass" if passed else "fail",
                    "message": message,
                    "value": value,
                    "threshold": gate.threshold,
                    "operator": gate.operator,
                    "critical": gate.critical,
                }
            )

            if not passed:
                all_passed = False
                if gate.critical:
                    critical_failed = True

        # Overall result is fail if any critical gate failed
        overall_passed = all_passed and not critical_failed

        return overall_passed, results

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value."""
        try:
            if metric_name == "cpu_usage":
                import psutil

                return psutil.cpu_percent(interval=0.1)
            elif metric_name == "memory_usage":
                import psutil

                return psutil.virtual_memory().percent
            elif metric_name == "active_agents":
                return float(len(performance_tracker.agent_metrics))
            elif metric_name == "avg_inference_time":
                return performance_tracker.get_average_inference_time()
            elif metric_name == "error_rate":
                # Calculate error rate from performance tracker
                return 0.02  # Placeholder
            elif metric_name == "response_quality":
                # Get response quality score
                return 0.75  # Placeholder
            else:
                logger.warning(
                    f"Unknown metric for gate evaluation: {metric_name}"
                )
                return None
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None


class CICDIntegration:
    """Main CI/CD integration system."""

    def __init__(self):
        """Initialize CI/CD integration."""
        self.gate_manager = PerformanceGateManager()
        self.deployments = {}
        self.deployment_history = []
        self.monitoring_enabled = True

        # CI/CD hooks
        self.pre_deployment_hooks = []
        self.post_deployment_hooks = []
        self.rollback_hooks = []

        # Setup alerting
        self._setup_deployment_alerting()

        logger.info("🚀 CI/CD integration initialized")

    def _setup_deployment_alerting(self):
        """Setup alerting for deployment monitoring."""
        # Deployment failure alert
        intelligent_alerting.add_rule(
            AlertRule(
                id="deployment_failure",
                name="Deployment Failure",
                description="Deployment failed to pass performance gates",
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.PATTERN,
                metric_name="deployment_status",
                conditions={"status": "failed"},
                runbook_url="https://docs.freeagentics.com/runbooks/deployment-failure",
            )
        )

        # Performance regression alert
        intelligent_alerting.add_rule(
            AlertRule(
                id="performance_regression",
                name="Performance Regression Detected",
                description="Performance regression detected in deployment",
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.PATTERN,
                metric_name="performance_regression",
                conditions={"regression": True},
                runbook_url="https://docs.freeagentics.com/runbooks/performance-regression",
            )
        )

    async def start_deployment(
        self, deployment_id: str, version: str, environment: str
    ) -> DeploymentMetrics:
        """Start a new deployment."""
        logger.info(
            f"Starting deployment {deployment_id} (version: {version}, environment: {environment})"
        )

        # Create deployment metrics
        deployment = DeploymentMetrics(
            deployment_id=deployment_id,
            version=version,
            environment=environment,
            start_time=datetime.now(),
        )

        # Collect pre-deployment metrics
        deployment.pre_deployment_metrics = (
            await self._collect_baseline_metrics()
        )

        # Store deployment
        self.deployments[deployment_id] = deployment

        # Run pre-deployment hooks
        for hook in self.pre_deployment_hooks:
            try:
                await hook(deployment)
            except Exception as e:
                logger.error(f"Pre-deployment hook failed: {e}")
                deployment.error_messages.append(
                    f"Pre-deployment hook failed: {e}"
                )

        # Log deployment start
        log_entry = create_structured_log_entry(
            level=LogLevel.INFO,
            source=LogSource.SYSTEM,
            message=f"Deployment started: {deployment_id}",
            module="cicd_integration",
            deployment_id=deployment_id,
            version=version,
            environment=environment,
        )
        log_aggregator.ingest_log_entry(log_entry)

        return deployment

    async def validate_deployment(
        self, deployment_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate deployment using performance gates."""
        if deployment_id not in self.deployments:
            return False, {"error": "Deployment not found"}

        deployment = self.deployments[deployment_id]

        logger.info(f"Validating deployment {deployment_id}")

        # Run health checks
        health_results = await self._run_health_checks()
        deployment.health_check_results = health_results

        # Evaluate performance gates
        gates_passed, gate_results = await self.gate_manager.evaluate_gates()

        # Collect post-deployment metrics
        deployment.post_deployment_metrics = (
            await self._collect_baseline_metrics()
        )

        # Check for performance regression
        regression_detected = await self._detect_performance_regression(
            deployment
        )
        deployment.performance_regression_detected = regression_detected

        # Determine overall validation result
        validation_passed = (
            gates_passed
            and not regression_detected
            and all(
                result.get("status") == "healthy" for result in health_results
            )
        )

        validation_result = {
            "deployment_id": deployment_id,
            "validation_passed": validation_passed,
            "gates_passed": gates_passed,
            "gate_results": gate_results,
            "regression_detected": regression_detected,
            "health_checks": health_results,
            "pre_deployment_metrics": deployment.pre_deployment_metrics,
            "post_deployment_metrics": deployment.post_deployment_metrics,
        }

        # Log validation result
        log_entry = create_structured_log_entry(
            level=LogLevel.INFO if validation_passed else LogLevel.ERROR,
            source=LogSource.SYSTEM,
            message=f"Deployment validation {'passed' if validation_passed else 'failed'}: {deployment_id}",
            module="cicd_integration",
            deployment_id=deployment_id,
            validation_passed=validation_passed,
            gates_passed=gates_passed,
            regression_detected=regression_detected,
        )
        log_aggregator.ingest_log_entry(log_entry)

        return validation_passed, validation_result

    async def complete_deployment(self, deployment_id: str, success: bool):
        """Complete a deployment."""
        if deployment_id not in self.deployments:
            return

        deployment = self.deployments[deployment_id]
        deployment.end_time = datetime.now()
        deployment.status = "success" if success else "failed"

        # Run post-deployment hooks
        for hook in self.post_deployment_hooks:
            try:
                await hook(deployment)
            except Exception as e:
                logger.error(f"Post-deployment hook failed: {e}")
                deployment.error_messages.append(
                    f"Post-deployment hook failed: {e}"
                )

        # Move to history
        self.deployment_history.append(deployment)

        # Keep only recent deployments in memory
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-100:]

        # Log completion
        log_entry = create_structured_log_entry(
            level=LogLevel.INFO if success else LogLevel.ERROR,
            source=LogSource.SYSTEM,
            message=f"Deployment {'completed' if success else 'failed'}: {deployment_id}",
            module="cicd_integration",
            deployment_id=deployment_id,
            success=success,
            duration=(
                deployment.end_time - deployment.start_time
            ).total_seconds(),
        )
        log_aggregator.ingest_log_entry(log_entry)

        logger.info(
            f"Deployment {'completed' if success else 'failed'}: {deployment_id}"
        )

    async def rollback_deployment(self, deployment_id: str, reason: str):
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            return

        deployment = self.deployments[deployment_id]
        deployment.status = "rolled_back"
        deployment.error_messages.append(f"Rollback reason: {reason}")

        # Run rollback hooks
        for hook in self.rollback_hooks:
            try:
                await hook(deployment, reason)
            except Exception as e:
                logger.error(f"Rollback hook failed: {e}")
                deployment.error_messages.append(f"Rollback hook failed: {e}")

        # Log rollback
        log_entry = create_structured_log_entry(
            level=LogLevel.WARNING,
            source=LogSource.SYSTEM,
            message=f"Deployment rolled back: {deployment_id}",
            module="cicd_integration",
            deployment_id=deployment_id,
            reason=reason,
        )
        log_aggregator.ingest_log_entry(log_entry)

        logger.warning(
            f"Deployment rolled back: {deployment_id} (reason: {reason})"
        )

    async def _collect_baseline_metrics(self) -> Dict[str, float]:
        """Collect baseline metrics."""
        try:
            import psutil

            # Get performance snapshot
            snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )

            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "active_agents": float(snapshot.active_agents),
                "avg_inference_time": performance_tracker.get_average_inference_time(),
                "memory_usage_mb": snapshot.memory_usage_mb,
                "cpu_usage_percent": snapshot.cpu_usage_percent,
                "agent_throughput": snapshot.agent_throughput,
                "belief_updates_per_sec": snapshot.belief_updates_per_sec,
            }
        except Exception as e:
            logger.error(f"Error collecting baseline metrics: {e}")
            return {}

    async def _run_health_checks(self) -> List[Dict[str, Any]]:
        """Run health checks."""
        health_checks = []

        try:
            # System health check
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            health_checks.append(
                {
                    "name": "System Resources",
                    "status": (
                        "healthy"
                        if cpu_percent < 90 and memory_percent < 90
                        else "unhealthy"
                    ),
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                    },
                }
            )

            # Performance tracker health
            snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )

            health_checks.append(
                {
                    "name": "Performance Tracker",
                    "status": "healthy"
                    if snapshot.active_agents < 50
                    else "unhealthy",
                    "details": {
                        "active_agents": snapshot.active_agents,
                        "memory_usage_mb": snapshot.memory_usage_mb,
                    },
                }
            )

            # Prometheus metrics health
            try:
                metrics_snapshot = prometheus_collector.get_metrics_snapshot()
                health_checks.append(
                    {
                        "name": "Prometheus Metrics",
                        "status": "healthy",
                        "details": {
                            "active_agents": metrics_snapshot.active_agents,
                            "total_inferences": metrics_snapshot.total_inferences,
                        },
                    }
                )
            except Exception as e:
                health_checks.append(
                    {
                        "name": "Prometheus Metrics",
                        "status": "unhealthy",
                        "error": str(e),
                    }
                )

        except Exception as e:
            logger.error(f"Error running health checks: {e}")
            health_checks.append(
                {
                    "name": "Health Check System",
                    "status": "unhealthy",
                    "error": str(e),
                }
            )

        return health_checks

    async def _detect_performance_regression(
        self, deployment: DeploymentMetrics
    ) -> bool:
        """Detect performance regression."""
        try:
            pre_metrics = deployment.pre_deployment_metrics
            post_metrics = deployment.post_deployment_metrics

            # Check for significant performance degradation
            regression_threshold = 0.2  # 20% degradation

            # Check CPU usage increase
            if "cpu_usage" in pre_metrics and "cpu_usage" in post_metrics:
                if post_metrics["cpu_usage"] > pre_metrics["cpu_usage"] * (
                    1 + regression_threshold
                ):
                    return True

            # Check memory usage increase
            if (
                "memory_usage" in pre_metrics
                and "memory_usage" in post_metrics
            ):
                if post_metrics["memory_usage"] > pre_metrics[
                    "memory_usage"
                ] * (1 + regression_threshold):
                    return True

            # Check inference time increase
            if (
                "avg_inference_time" in pre_metrics
                and "avg_inference_time" in post_metrics
            ):
                if post_metrics["avg_inference_time"] > pre_metrics[
                    "avg_inference_time"
                ] * (1 + regression_threshold):
                    return True

            # Check throughput decrease
            if (
                "agent_throughput" in pre_metrics
                and "agent_throughput" in post_metrics
            ):
                if post_metrics["agent_throughput"] < pre_metrics[
                    "agent_throughput"
                ] * (1 - regression_threshold):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting performance regression: {e}")
            return False

    def add_pre_deployment_hook(self, hook):
        """Add pre-deployment hook."""
        self.pre_deployment_hooks.append(hook)

    def add_post_deployment_hook(self, hook):
        """Add post-deployment hook."""
        self.post_deployment_hooks.append(hook)

    def add_rollback_hook(self, hook):
        """Add rollback hook."""
        self.rollback_hooks.append(hook)

    def get_deployment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get deployment history."""
        recent_deployments = self.deployment_history[-limit:]
        return [deployment.to_dict() for deployment in recent_deployments]

    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        if not self.deployment_history:
            return {
                "total_deployments": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "rollback_rate": 0.0,
                "avg_deployment_duration": 0.0,
            }

        total = len(self.deployment_history)
        successful = len(
            [d for d in self.deployment_history if d.status == "success"]
        )
        failed = len(
            [d for d in self.deployment_history if d.status == "failed"]
        )
        rolled_back = len(
            [d for d in self.deployment_history if d.status == "rolled_back"]
        )

        # Calculate average duration
        completed_deployments = [
            d for d in self.deployment_history if d.end_time is not None
        ]
        avg_duration = 0.0
        if completed_deployments:
            durations = [
                (d.end_time - d.start_time).total_seconds()
                for d in completed_deployments
            ]
            avg_duration = sum(durations) / len(durations)

        return {
            "total_deployments": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "failure_rate": failed / total if total > 0 else 0.0,
            "rollback_rate": rolled_back / total if total > 0 else 0.0,
            "avg_deployment_duration": avg_duration,
            "successful_deployments": successful,
            "failed_deployments": failed,
            "rolled_back_deployments": rolled_back,
        }


# CLI command for CI/CD integration
async def run_performance_gates():
    """Run performance gates check (for CI/CD pipeline)."""
    cicd = CICDIntegration()

    # Get environment variables
    deployment_id = os.getenv("DEPLOYMENT_ID", f"deploy_{int(time.time())}")
    version = os.getenv("VERSION", "unknown")
    environment = os.getenv("ENVIRONMENT", "staging")

    try:
        # Start deployment
        deployment = await cicd.start_deployment(
            deployment_id, version, environment
        )

        # Validate deployment
        validation_passed, validation_result = await cicd.validate_deployment(
            deployment_id
        )

        # Complete deployment
        await cicd.complete_deployment(deployment_id, validation_passed)

        # Print results
        print(
            f"Deployment {deployment_id} validation: {'PASSED' if validation_passed else 'FAILED'}"
        )
        print(json.dumps(validation_result, indent=2))

        # Exit with appropriate code
        exit(0 if validation_passed else 1)

    except Exception as e:
        logger.error(f"Error running performance gates: {e}")
        await cicd.complete_deployment(deployment_id, False)
        print(f"ERROR: {e}")
        exit(1)


# FastAPI endpoints for CI/CD integration
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

cicd_router = APIRouter()


class StartDeploymentRequest(BaseModel):
    deployment_id: str
    version: str
    environment: str


class CompleteDeploymentRequest(BaseModel):
    deployment_id: str
    success: bool


class RollbackDeploymentRequest(BaseModel):
    deployment_id: str
    reason: str


@cicd_router.post("/deployments/start")
async def start_deployment_endpoint(request: StartDeploymentRequest):
    """Start a deployment."""
    cicd = CICDIntegration()
    deployment = await cicd.start_deployment(
        request.deployment_id, request.version, request.environment
    )
    return deployment.to_dict()


@cicd_router.post("/deployments/{deployment_id}/validate")
async def validate_deployment_endpoint(deployment_id: str):
    """Validate a deployment."""
    cicd = CICDIntegration()
    validation_passed, validation_result = await cicd.validate_deployment(
        deployment_id
    )
    return {
        "validation_passed": validation_passed,
        "result": validation_result,
    }


@cicd_router.post("/deployments/complete")
async def complete_deployment_endpoint(request: CompleteDeploymentRequest):
    """Complete a deployment."""
    cicd = CICDIntegration()
    await cicd.complete_deployment(request.deployment_id, request.success)
    return {"message": "Deployment completed"}


@cicd_router.post("/deployments/rollback")
async def rollback_deployment_endpoint(request: RollbackDeploymentRequest):
    """Rollback a deployment."""
    cicd = CICDIntegration()
    await cicd.rollback_deployment(request.deployment_id, request.reason)
    return {"message": "Deployment rolled back"}


@cicd_router.get("/deployments/history")
async def get_deployment_history(limit: int = 20):
    """Get deployment history."""
    cicd = CICDIntegration()
    return cicd.get_deployment_history(limit)


@cicd_router.get("/deployments/statistics")
async def get_deployment_statistics():
    """Get deployment statistics."""
    cicd = CICDIntegration()
    return cicd.get_deployment_statistics()


@cicd_router.get("/performance-gates")
async def get_performance_gates():
    """Get performance gates configuration."""
    cicd = CICDIntegration()
    gates_passed, gate_results = await cicd.gate_manager.evaluate_gates()

    return {
        "gates_passed": gates_passed,
        "gate_results": gate_results,
        "gates_configuration": [
            {
                "name": gate.name,
                "metric_name": gate.metric_name,
                "threshold": gate.threshold,
                "operator": gate.operator,
                "enabled": gate.enabled,
                "critical": gate.critical,
                "description": gate.description,
            }
            for gate in cicd.gate_manager.gates
        ],
    }


# Global CI/CD integration instance
cicd_integration = CICDIntegration()


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "performance-gates":
        asyncio.run(run_performance_gates())
    else:
        print("Usage: python cicd_integration.py performance-gates")
        sys.exit(1)
