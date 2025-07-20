"""
Distributed Tracing System for FreeAgentics Multi-Agent Platform.

This module implements distributed tracing capabilities to track requests
across multiple agents, services, and coordination operations.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    error: Optional[str] = None

    def finish(self, status: str = "ok", error: Optional[str] = None):
        """Finish the span with timing and status."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error

    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value

    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": level,
            **kwargs,
        }
        self.logs.append(log_entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class Trace:
    """Represents a complete distributed trace."""

    trace_id: str
    spans: Dict[str, TraceSpan] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    service_names: Set[str] = field(default_factory=set)
    operation_count: int = 0
    error_count: int = 0

    def add_span(self, span: TraceSpan):
        """Add a span to the trace."""
        self.spans[span.span_id] = span
        self.service_names.add(span.service_name)
        self.operation_count += 1

        if span.status == "error":
            self.error_count += 1

    def get_root_span(self) -> Optional[TraceSpan]:
        """Get the root span of the trace."""
        for span in self.spans.values():
            if span.parent_span_id is None:
                return span
        return None

    def get_child_spans(self, parent_span_id: str) -> List[TraceSpan]:
        """Get all child spans of a parent span."""
        return [
            span
            for span in self.spans.values()
            if span.parent_span_id == parent_span_id
        ]

    def finish(self):
        """Finish the trace by calculating total duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.spans.values()],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "service_names": list(self.service_names),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
        }


class DistributedTracer:
    """Main distributed tracing implementation."""

    def __init__(
        self, service_name: str = "freeagentics", max_traces: int = 1000
    ):
        """Initialize the distributed tracer."""
        self.service_name = service_name
        self.max_traces = max_traces
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.trace_ttl = 3600  # 1 hour
        self.running = False
        self.cleanup_task = None

        logger.info(
            f"🔍 Distributed tracer initialized for service: {service_name}"
        )

    async def start(self):
        """Start the distributed tracer."""
        if self.running:
            logger.warning("Distributed tracer already running")
            return

        self.running = True

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("🔍 Distributed tracer started")

    async def stop(self):
        """Stop the distributed tracer."""
        if not self.running:
            logger.warning("Distributed tracer not running")
            return

        self.running = False

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("🔍 Distributed tracer stopped")

    def start_trace(
        self, operation_name: str, service_name: Optional[str] = None
    ) -> TraceSpan:
        """Start a new trace with a root span."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            service_name=service_name or self.service_name,
        )

        # Create new trace
        trace = Trace(trace_id=trace_id)
        trace.add_span(span)

        self.traces[trace_id] = trace
        self.active_spans[span_id] = span

        return span

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None,
        service_name: Optional[str] = None,
    ) -> TraceSpan:
        """Start a new span within an existing trace."""
        if parent_span is None:
            # Start a new trace if no parent
            return self.start_trace(operation_name, service_name)

        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=parent_span.trace_id,
            span_id=span_id,
            parent_span_id=parent_span.span_id,
            operation_name=operation_name,
            service_name=service_name or self.service_name,
        )

        # Add to existing trace
        if parent_span.trace_id in self.traces:
            self.traces[parent_span.trace_id].add_span(span)

        self.active_spans[span_id] = span

        return span

    def finish_span(
        self, span: TraceSpan, status: str = "ok", error: Optional[str] = None
    ):
        """Finish a span."""
        span.finish(status, error)

        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]

        # Check if trace is complete (no more active spans)
        trace = self.traces.get(span.trace_id)
        if trace:
            active_spans_in_trace = [
                s
                for s in self.active_spans.values()
                if s.trace_id == span.trace_id
            ]

            if not active_spans_in_trace:
                trace.finish()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)

    def get_active_traces(self) -> List[Trace]:
        """Get all active traces."""
        return [
            trace for trace in self.traces.values() if trace.end_time is None
        ]

    def get_completed_traces(self, limit: int = 100) -> List[Trace]:
        """Get completed traces."""
        completed = [
            trace
            for trace in self.traces.values()
            if trace.end_time is not None
        ]
        return sorted(completed, key=lambda t: t.end_time or 0, reverse=True)[
            :limit
        ]

    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        total_traces = len(self.traces)
        active_traces = len(self.get_active_traces())
        completed_traces = total_traces - active_traces

        # Calculate average duration for completed traces
        completed = [
            t for t in self.traces.values() if t.duration_ms is not None
        ]
        avg_duration = (
            sum(t.duration_ms for t in completed if t.duration_ms is not None)
            / len(completed)
            if completed
            else 0
        )

        return {
            "total_traces": total_traces,
            "active_traces": active_traces,
            "completed_traces": completed_traces,
            "active_spans": len(self.active_spans),
            "avg_duration_ms": avg_duration,
            "services": list(
                set().union(*(t.service_names for t in self.traces.values()))
            ),
            "error_rate": (
                sum(t.error_count for t in completed)
                / sum(t.operation_count for t in completed)
                if completed
                else 0
            ),
        }

    async def _cleanup_loop(self):
        """Background loop to clean up old traces."""
        while self.running:
            try:
                await self._cleanup_old_traces()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trace cleanup: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _cleanup_old_traces(self):
        """Clean up old traces to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - self.trace_ttl

        # Find traces to remove
        traces_to_remove = []
        for trace_id, trace in self.traces.items():
            if trace.start_time < cutoff_time:
                traces_to_remove.append(trace_id)

        # Remove old traces
        for trace_id in traces_to_remove:
            del self.traces[trace_id]

            # Remove associated active spans
            spans_to_remove = [
                span_id
                for span_id, span in self.active_spans.items()
                if span.trace_id == trace_id
            ]
            for span_id in spans_to_remove:
                del self.active_spans[span_id]

        # Limit total traces
        if len(self.traces) > self.max_traces:
            # Remove oldest traces
            sorted_traces = sorted(
                self.traces.items(), key=lambda x: x[1].start_time
            )
            to_remove = sorted_traces[: len(self.traces) - self.max_traces]

            for trace_id, _ in to_remove:
                del self.traces[trace_id]

        if traces_to_remove:
            logger.debug(f"Cleaned up {len(traces_to_remove)} old traces")


@asynccontextmanager
async def trace_span(
    tracer: DistributedTracer,
    operation_name: str,
    parent_span: Optional[TraceSpan] = None,
    service_name: Optional[str] = None,
):
    """Context manager for tracing operations."""
    span = tracer.start_span(operation_name, parent_span, service_name)

    try:
        yield span
        tracer.finish_span(span, "ok")
    except Exception as e:
        tracer.finish_span(span, "error", str(e))
        raise


class AgentTracingMixin:
    """Mixin class to add tracing capabilities to agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = None
        self.current_span = None

    def set_tracer(self, tracer: DistributedTracer):
        """Set the distributed tracer for this agent."""
        self.tracer = tracer

    async def trace_operation(
        self, operation_name: str, operation_func, *args, **kwargs
    ):
        """Trace an agent operation."""
        if not self.tracer:
            return await operation_func(*args, **kwargs)

        async with trace_span(
            self.tracer,
            operation_name,
            self.current_span,
            service_name=f"agent-{getattr(self, 'agent_id', 'unknown')}",
        ) as span:
            span.add_tag("agent_id", getattr(self, "agent_id", "unknown"))
            span.add_tag("operation", operation_name)

            try:
                result = await operation_func(*args, **kwargs)
                span.add_log(
                    f"Operation {operation_name} completed successfully"
                )
                return result
            except Exception as e:
                span.add_log(
                    f"Operation {operation_name} failed: {e}", level="error"
                )
                raise


# Global distributed tracer instance
_distributed_tracer = None


def get_distributed_tracer(
    service_name: str = "freeagentics",
) -> DistributedTracer:
    """Get the global distributed tracer instance."""
    global _distributed_tracer

    if _distributed_tracer is None:
        _distributed_tracer = DistributedTracer(service_name)

    return _distributed_tracer


async def start_distributed_tracing(service_name: str = "freeagentics"):
    """Start the global distributed tracing system."""
    tracer = get_distributed_tracer(service_name)
    await tracer.start()
    return tracer


async def stop_distributed_tracing():
    """Stop the global distributed tracing system."""
    global _distributed_tracer

    if _distributed_tracer:
        await _distributed_tracer.stop()
        _distributed_tracer = None


# Convenience functions for common tracing operations
async def trace_agent_coordination(
    tracer: DistributedTracer,
    agent_id: str,
    coordination_type: str,
    operation_func,
    *args,
    **kwargs,
):
    """Trace agent coordination operations."""
    operation_name = f"coordination_{coordination_type}"

    async with trace_span(
        tracer, operation_name, service_name=f"agent-{agent_id}"
    ) as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("coordination_type", coordination_type)

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log(f"Coordination {coordination_type} completed")
            return result
        except Exception as e:
            span.add_log(
                f"Coordination {coordination_type} failed: {e}", level="error"
            )
            raise


async def trace_belief_update(
    tracer: DistributedTracer, agent_id: str, operation_func, *args, **kwargs
):
    """Trace belief system updates."""
    operation_name = "belief_update"

    async with trace_span(
        tracer, operation_name, service_name=f"agent-{agent_id}"
    ) as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("operation_type", "belief_update")

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log("Belief update completed")
            return result
        except Exception as e:
            span.add_log(f"Belief update failed: {e}", level="error")
            raise


async def trace_inference_step(
    tracer: DistributedTracer,
    agent_id: str,
    step_type: str,
    operation_func,
    *args,
    **kwargs,
):
    """Trace inference steps."""
    operation_name = f"inference_{step_type}"

    async with trace_span(
        tracer, operation_name, service_name=f"agent-{agent_id}"
    ) as span:
        span.add_tag("agent_id", agent_id)
        span.add_tag("step_type", step_type)

        try:
            result = await operation_func(*args, **kwargs)
            span.add_log(f"Inference step {step_type} completed")
            return result
        except Exception as e:
            span.add_log(
                f"Inference step {step_type} failed: {e}", level="error"
            )
            raise
