"""
Distributed Tracing Integration for FreeAgentics Production Monitoring

This module provides integration between the distributed tracing system and
other monitoring components like Prometheus, logging, and alerting.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from observability.distributed_tracing import (
    DistributedTracer,
    Trace,
    TraceSpan,
    get_distributed_tracer,
    trace_span,
)
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
from observability.prometheus_metrics import (
    record_agent_coordination_duration,
    record_agent_coordination_error,
    record_agent_coordination_request,
    record_agent_error,
    record_agent_inference_duration,
    record_agent_step,
)

logger = logging.getLogger(__name__)


@dataclass
class TracingMetrics:
    """Metrics collected from distributed tracing."""

    total_traces: int = 0
    completed_traces: int = 0
    failed_traces: int = 0
    avg_trace_duration: float = 0.0
    traces_by_service: Dict[str, int] = field(default_factory=dict)
    spans_by_operation: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0


class TracingAnalyzer:
    """Analyzer for distributed tracing data."""

    def __init__(self, tracer: DistributedTracer):
        """Initialize tracing analyzer."""
        self.tracer = tracer
        self.analysis_window = 3600  # 1 hour
        self.last_analysis = datetime.now()

    def analyze_traces(self, time_window: int = 3600) -> TracingMetrics:
        """Analyze traces within time window."""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)

        # Get traces within time window
        relevant_traces = []
        for trace in self.tracer.traces.values():
            if trace.start_time >= cutoff_time.timestamp():
                relevant_traces.append(trace)

        if not relevant_traces:
            return TracingMetrics()

        # Calculate metrics
        metrics = TracingMetrics()
        metrics.total_traces = len(relevant_traces)

        completed_traces = [t for t in relevant_traces if t.end_time is not None]
        failed_traces = [t for t in relevant_traces if t.error_count > 0]

        metrics.completed_traces = len(completed_traces)
        metrics.failed_traces = len(failed_traces)
        metrics.error_rate = len(failed_traces) / len(relevant_traces) if relevant_traces else 0.0

        # Duration metrics
        if completed_traces:
            durations = [t.duration_ms for t in completed_traces if t.duration_ms is not None]
            if durations:
                metrics.avg_trace_duration = sum(durations) / len(durations)
                sorted_durations = sorted(durations)
                metrics.p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
                metrics.p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]

        # Service and operation metrics
        for trace in relevant_traces:
            for span in trace.spans.values():
                metrics.traces_by_service[span.service_name] = (
                    metrics.traces_by_service.get(span.service_name, 0) + 1
                )
                metrics.spans_by_operation[span.operation_name] = (
                    metrics.spans_by_operation.get(span.operation_name, 0) + 1
                )

        return metrics

    def detect_performance_issues(self, metrics: TracingMetrics) -> List[Dict[str, Any]]:
        """Detect performance issues from tracing metrics."""
        issues = []

        # High error rate
        if metrics.error_rate > 0.1:  # 10% error rate
            issues.append(
                {
                    "type": "high_error_rate",
                    "severity": "high",
                    "message": f"High error rate detected: {metrics.error_rate:.2%}",
                    "value": metrics.error_rate,
                    "threshold": 0.1,
                }
            )

        # Slow traces
        if metrics.p95_duration > 5000:  # 5 seconds
            issues.append(
                {
                    "type": "slow_traces",
                    "severity": "medium",
                    "message": f"Slow traces detected: P95 = {metrics.p95_duration:.0f}ms",
                    "value": metrics.p95_duration,
                    "threshold": 5000,
                }
            )

        # Very slow traces
        if metrics.p99_duration > 10000:  # 10 seconds
            issues.append(
                {
                    "type": "very_slow_traces",
                    "severity": "high",
                    "message": f"Very slow traces detected: P99 = {metrics.p99_duration:.0f}ms",
                    "value": metrics.p99_duration,
                    "threshold": 10000,
                }
            )

        return issues


class TracingIntegration:
    """Integration layer for distributed tracing with monitoring systems."""

    def __init__(self, tracer: DistributedTracer):
        """Initialize tracing integration."""
        self.tracer = tracer
        self.analyzer = TracingAnalyzer(tracer)
        self.running = False
        self.integration_task = None
        self.metrics_export_task = None

        # Setup span hooks
        self._setup_span_hooks()

        # Setup alerting rules
        self._setup_alerting_rules()

        logger.info("🔗 Distributed tracing integration initialized")

    def _setup_span_hooks(self):
        """Setup hooks for span lifecycle events."""
        # Hook into span completion to export metrics
        original_finish_span = self.tracer.finish_span

        def hooked_finish_span(span: TraceSpan, status: str = "ok", error: Optional[str] = None):
            """Hooked finish_span to export metrics."""
            # Call original method
            original_finish_span(span, status, error)

            # Export metrics
            asyncio.create_task(self._export_span_metrics(span))

        self.tracer.finish_span = hooked_finish_span

    def _setup_alerting_rules(self):
        """Setup alerting rules for tracing metrics."""
        # High error rate alert
        intelligent_alerting.add_rule(
            AlertRule(
                id="tracing_high_error_rate",
                name="High Trace Error Rate",
                description="High error rate detected in distributed traces",
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.THRESHOLD,
                metric_name="trace_error_rate",
                threshold_value=0.1,
                threshold_operator=">",
                time_window=300,
                evaluation_frequency=60,
                runbook_url="https://docs.freeagentics.com/runbooks/trace-errors",
            )
        )

        # Slow traces alert
        intelligent_alerting.add_rule(
            AlertRule(
                id="tracing_slow_traces",
                name="Slow Distributed Traces",
                description="P95 trace duration exceeds threshold",
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.THRESHOLD,
                metric_name="trace_p95_duration",
                threshold_value=5000,
                threshold_operator=">",
                time_window=600,
                evaluation_frequency=120,
                runbook_url="https://docs.freeagentics.com/runbooks/slow-traces",
            )
        )

    async def start(self):
        """Start tracing integration."""
        if self.running:
            logger.warning("Tracing integration already running")
            return

        self.running = True

        # Start integration task
        self.integration_task = asyncio.create_task(self._integration_loop())

        # Start metrics export task
        self.metrics_export_task = asyncio.create_task(self._metrics_export_loop())

        logger.info("🚀 Distributed tracing integration started")

    async def stop(self):
        """Stop tracing integration."""
        if not self.running:
            logger.warning("Tracing integration not running")
            return

        self.running = False

        # Stop tasks
        if self.integration_task:
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass

        if self.metrics_export_task:
            self.metrics_export_task.cancel()
            try:
                await self.metrics_export_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Distributed tracing integration stopped")

    async def _integration_loop(self):
        """Main integration loop."""
        while self.running:
            try:
                await self._perform_integration_tasks()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tracing integration loop: {e}")
                await asyncio.sleep(60)

    async def _metrics_export_loop(self):
        """Metrics export loop."""
        while self.running:
            try:
                await self._export_tracing_metrics()
                await asyncio.sleep(30)  # Export every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")
                await asyncio.sleep(30)

    async def _perform_integration_tasks(self):
        """Perform integration tasks."""
        # Analyze traces
        metrics = self.analyzer.analyze_traces()

        # Detect issues
        issues = self.analyzer.detect_performance_issues(metrics)

        # Log issues
        for issue in issues:
            log_entry = create_structured_log_entry(
                level=LogLevel.WARNING if issue["severity"] == "medium" else LogLevel.ERROR,
                source=LogSource.OBSERVABILITY,
                message=issue["message"],
                module="tracing_integration",
                issue_type=issue["type"],
                severity=issue["severity"],
                value=issue["value"],
                threshold=issue["threshold"],
            )
            log_aggregator.ingest_log_entry(log_entry)

        # Update trace statistics
        stats = self.tracer.get_trace_stats()

        # Log trace statistics
        if stats["total_traces"] > 0:
            logger.info(f"Trace statistics: {json.dumps(stats, indent=2)}")

    async def _export_tracing_metrics(self):
        """Export tracing metrics to Prometheus."""
        try:
            # Get trace statistics
            stats = self.tracer.get_trace_stats()

            # Export to custom metrics (if metrics exporter is available)
            try:
                from observability.metrics_exporter import get_metrics_exporter

                exporter = get_metrics_exporter()

                # Record custom metrics
                exporter.record_custom_application_event(
                    "trace_statistics", "distributed_tracing", "exported"
                )

            except ImportError:
                pass

            # Export trace metrics
            if stats["total_traces"] > 0:
                logger.debug(f"Exported tracing metrics: {json.dumps(stats)}")

        except Exception as e:
            logger.error(f"Error exporting tracing metrics: {e}")

    async def _export_span_metrics(self, span: TraceSpan):
        """Export metrics for a completed span."""
        try:
            # Determine metric type based on operation
            if span.operation_name.startswith("coordination_"):
                # Agent coordination metrics
                status = "success" if span.status == "ok" else "failure"
                coordination_type = span.operation_name.replace("coordination_", "")

                record_agent_coordination_request(
                    agent_id=span.tags.get("agent_id", "unknown"),
                    coordination_type=coordination_type,
                    status=status,
                )

                if span.duration_ms:
                    record_agent_coordination_duration(
                        agent_id=span.tags.get("agent_id", "unknown"),
                        coordination_type=coordination_type,
                        duration_seconds=span.duration_ms / 1000.0,
                    )

                if span.status == "error":
                    record_agent_coordination_error(
                        agent_id=span.tags.get("agent_id", "unknown"),
                        error_type=span.error or "unknown",
                        severity="high",
                    )

            elif span.operation_name.startswith("inference_"):
                # Agent inference metrics
                if span.duration_ms:
                    record_agent_inference_duration(
                        agent_id=span.tags.get("agent_id", "unknown"),
                        operation_type=span.operation_name,
                        duration_seconds=span.duration_ms / 1000.0,
                    )

                # Record agent step
                record_agent_step(
                    agent_id=span.tags.get("agent_id", "unknown"),
                    step_type=span.operation_name,
                    success=span.status == "ok",
                )

                if span.status == "error":
                    record_agent_error(
                        agent_id=span.tags.get("agent_id", "unknown"),
                        error_type=span.error or "unknown",
                        severity="medium",
                    )

            # Log span completion
            log_entry = create_structured_log_entry(
                level=LogLevel.INFO if span.status == "ok" else LogLevel.ERROR,
                source=LogSource.OBSERVABILITY,
                message=f"Span completed: {span.operation_name}",
                module="distributed_tracing",
                trace_id=span.trace_id,
                span_id=span.span_id,
                operation_name=span.operation_name,
                service_name=span.service_name,
                duration_ms=span.duration_ms,
                status=span.status,
                agent_id=span.tags.get("agent_id"),
                **span.tags,
            )
            log_aggregator.ingest_log_entry(log_entry)

        except Exception as e:
            logger.error(f"Error exporting span metrics: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "running": self.running,
            "tracer_stats": self.tracer.get_trace_stats(),
            "analyzer_metrics": (
                self.analyzer.analyze_traces().to_dict()
                if hasattr(self.analyzer.analyze_traces(), "to_dict")
                else {}
            ),
            "integration_health": "healthy" if self.running else "stopped",
        }


from typing import List as ListType

# FastAPI endpoints for tracing integration
from fastapi import APIRouter, Query

tracing_router = APIRouter()


@tracing_router.get("/traces")
async def get_traces(limit: int = Query(100, ge=1, le=1000)):
    """Get recent traces."""
    tracer = get_distributed_tracer()
    completed_traces = tracer.get_completed_traces(limit)

    return {
        "traces": [trace.to_dict() for trace in completed_traces],
        "total": len(completed_traces),
        "active_traces": len(tracer.get_active_traces()),
    }


@tracing_router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get specific trace."""
    tracer = get_distributed_tracer()
    trace = tracer.get_trace(trace_id)

    if not trace:
        return {"error": "Trace not found"}

    return {"trace": trace.to_dict(), "spans": [span.to_dict() for span in trace.spans.values()]}


@tracing_router.get("/traces/stats")
async def get_trace_stats():
    """Get tracing statistics."""
    tracer = get_distributed_tracer()
    stats = tracer.get_trace_stats()

    # Get integration status if available
    integration_status = {}
    try:
        integration_status = tracing_integration.get_integration_status()
    except:
        pass

    return {"tracer_stats": stats, "integration_status": integration_status}


@tracing_router.get("/traces/analysis")
async def get_trace_analysis(time_window: int = Query(3600, ge=60, le=86400)):
    """Get trace analysis."""
    tracer = get_distributed_tracer()
    analyzer = TracingAnalyzer(tracer)

    metrics = analyzer.analyze_traces(time_window)
    issues = analyzer.detect_performance_issues(metrics)

    return {
        "metrics": {
            "total_traces": metrics.total_traces,
            "completed_traces": metrics.completed_traces,
            "failed_traces": metrics.failed_traces,
            "error_rate": metrics.error_rate,
            "avg_trace_duration": metrics.avg_trace_duration,
            "p95_duration": metrics.p95_duration,
            "p99_duration": metrics.p99_duration,
            "traces_by_service": metrics.traces_by_service,
            "spans_by_operation": metrics.spans_by_operation,
        },
        "issues": issues,
        "time_window": time_window,
    }


# Global tracing integration instance
tracing_integration = TracingIntegration(get_distributed_tracer())


async def start_tracing_integration():
    """Start tracing integration."""
    await tracing_integration.start()


async def stop_tracing_integration():
    """Stop tracing integration."""
    await tracing_integration.stop()


# Utility functions for application integration
async def trace_agent_operation(
    agent_id: str, operation_name: str, operation_func, *args, **kwargs
):
    """Trace an agent operation with automatic metrics export."""
    tracer = get_distributed_tracer()

    async with trace_span(tracer, operation_name, service_name=f"agent-{agent_id}") as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("operation", operation_name)

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log(f"Operation {operation_name} completed successfully")
            return result
        except Exception as e:
            span.add_log(f"Operation {operation_name} failed: {e}", level="error")
            raise


async def trace_coordination_operation(
    agent_id: str, coordination_type: str, operation_func, *args, **kwargs
):
    """Trace coordination operation with automatic metrics export."""
    tracer = get_distributed_tracer()

    async with trace_span(
        tracer, f"coordination_{coordination_type}", service_name=f"agent-{agent_id}"
    ) as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("coordination_type", coordination_type)

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log(f"Coordination {coordination_type} completed successfully")
            return result
        except Exception as e:
            span.add_log(f"Coordination {coordination_type} failed: {e}", level="error")
            raise


async def trace_inference_operation(
    agent_id: str, inference_type: str, operation_func, *args, **kwargs
):
    """Trace inference operation with automatic metrics export."""
    tracer = get_distributed_tracer()

    async with trace_span(
        tracer, f"inference_{inference_type}", service_name=f"agent-{agent_id}"
    ) as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("inference_type", inference_type)

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log(f"Inference {inference_type} completed successfully")
            return result
        except Exception as e:
            span.add_log(f"Inference {inference_type} failed: {e}", level="error")
            raise
