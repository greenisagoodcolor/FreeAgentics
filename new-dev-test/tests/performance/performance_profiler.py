"""Component Performance Profiler.

This module provides detailed performance profiling capabilities for different
system components, including code-level profiling, memory analysis, and
bottleneck identification.
"""

import asyncio
import cProfile
import functools
import json
import logging
import pstats
import threading
import time
import tracemalloc
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

from tests.performance.unified_metrics_collector import MetricSource, MetricType, record_metric

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session."""

    component: str
    operation: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_time_seconds: float
    memory_start_mb: float
    memory_peak_mb: float
    memory_allocated_mb: float
    call_count: int
    function_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory_allocations: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ComponentProfiler:
    """Performance profiler for system components."""

    def __init__(
        self,
        enable_cpu_profiling: bool = True,
        enable_memory_profiling: bool = True,
        profile_async: bool = True,
        top_functions: int = 50,
        memory_snapshot_interval: float = 0.1,
    ):
        """Initialize the profiler.

        Args:
            enable_cpu_profiling: Enable CPU profiling
            enable_memory_profiling: Enable memory profiling
            profile_async: Enable async profiling support
            top_functions: Number of top functions to report
            memory_snapshot_interval: Interval for memory snapshots (seconds)
        """
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_memory_profiling = enable_memory_profiling
        self.profile_async = profile_async
        self.top_functions = top_functions
        self.memory_snapshot_interval = memory_snapshot_interval

        # Profile storage
        self._profiles: Dict[str, List[ProfileResult]] = defaultdict(list)
        self._active_profiles: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Component-specific settings
        self._component_settings: Dict[str, Dict[str, Any]] = {
            "database": {
                "track_queries": True,
                "slow_query_threshold_ms": 100,
                "track_connections": True,
            },
            "websocket": {
                "track_messages": True,
                "track_connections": True,
                "latency_buckets": [1, 5, 10, 25, 50, 100, 250, 500, 1000],
            },
            "agent": {
                "track_inference": True,
                "track_beliefs": True,
                "track_actions": True,
                "memory_limit_mb": 50,
            },
            "inference": {
                "track_matrix_ops": True,
                "track_cache_hits": True,
                "profile_algorithms": True,
            },
        }

    @contextmanager
    def profile_component(self, component: str, operation: str, metadata: Dict[str, Any] = None):
        """Profile a component operation."""
        profile_id = f"{component}.{operation}.{time.time()}"

        # Start profiling
        cpu_profiler = None
        if self.enable_cpu_profiling:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()

        memory_snapshot = None
        if self.enable_memory_profiling:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            memory_snapshot = tracemalloc.take_snapshot()

        process = psutil.Process()
        start_time = datetime.now()
        start_cpu_time = process.cpu_times().user + process.cpu_times().system
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB

        with self._lock:
            self._active_profiles[profile_id] = {
                "component": component,
                "operation": operation,
                "metadata": metadata or {},
                "start_time": start_time,
                "cpu_profiler": cpu_profiler,
                "memory_snapshot": memory_snapshot,
                "memory_samples": [(start_time, start_memory)],
                "peak_memory": start_memory,
            }

        try:
            yield profile_id
        finally:
            # Stop profiling
            end_time = datetime.now()
            end_cpu_time = process.cpu_times().user + process.cpu_times().system
            end_memory = process.memory_info().rss / (1024 * 1024)

            if cpu_profiler:
                cpu_profiler.disable()

            memory_snapshot_end = None
            if self.enable_memory_profiling and tracemalloc.is_tracing():
                memory_snapshot_end = tracemalloc.take_snapshot()

            # Create profile result
            with self._lock:
                profile_data = self._active_profiles.pop(profile_id, {})

            result = self._analyze_profile(
                component=component,
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                cpu_time_start=start_cpu_time,
                cpu_time_end=end_cpu_time,
                memory_start=start_memory,
                memory_end=end_memory,
                memory_peak=profile_data.get("peak_memory", end_memory),
                cpu_profiler=cpu_profiler,
                memory_snapshot_start=profile_data.get("memory_snapshot"),
                memory_snapshot_end=memory_snapshot_end,
                metadata=profile_data.get("metadata", {}),
            )

            # Store result
            with self._lock:
                self._profiles[component].append(result)

            # Record metrics
            self._record_profile_metrics(result)

    @asynccontextmanager
    async def profile_component_async(
        self, component: str, operation: str, metadata: Dict[str, Any] = None
    ):
        """Async version of profile_component."""
        if not self.profile_async:
            async with self.profile_component(component, operation, metadata) as profile_id:
                yield profile_id
            return

        profile_id = f"{component}.{operation}.{time.time()}"

        # Start profiling in a thread-safe way
        loop = asyncio.get_event_loop()

        # Initialize profiling
        await loop.run_in_executor(
            None,
            self._start_profile,
            profile_id,
            component,
            operation,
            metadata,
        )

        # Start memory monitoring task
        memory_task = asyncio.create_task(self._monitor_memory_async(profile_id))

        try:
            yield profile_id
        finally:
            # Stop memory monitoring
            memory_task.cancel()
            try:
                await memory_task
            except asyncio.CancelledError:
                pass

            # Finalize profiling
            result = await loop.run_in_executor(None, self._stop_profile, profile_id)

            if result:
                # Record metrics
                self._record_profile_metrics(result)

    def _start_profile(
        self,
        profile_id: str,
        component: str,
        operation: str,
        metadata: Optional[Dict[str, Any]],
    ):
        """Start profiling (thread-safe)."""
        cpu_profiler = None
        if self.enable_cpu_profiling:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()

        memory_snapshot = None
        if self.enable_memory_profiling:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            memory_snapshot = tracemalloc.take_snapshot()

        process = psutil.Process()
        start_time = datetime.now()
        start_cpu_time = process.cpu_times().user + process.cpu_times().system
        start_memory = process.memory_info().rss / (1024 * 1024)

        with self._lock:
            self._active_profiles[profile_id] = {
                "component": component,
                "operation": operation,
                "metadata": metadata or {},
                "start_time": start_time,
                "start_cpu_time": start_cpu_time,
                "cpu_profiler": cpu_profiler,
                "memory_snapshot": memory_snapshot,
                "memory_samples": [(start_time, start_memory)],
                "peak_memory": start_memory,
                "start_memory": start_memory,
            }

    def _stop_profile(self, profile_id: str) -> Optional[ProfileResult]:
        """Stop profiling and analyze results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                return None

            profile_data = self._active_profiles.pop(profile_id)

        # Stop CPU profiler
        cpu_profiler = profile_data.get("cpu_profiler")
        if cpu_profiler:
            cpu_profiler.disable()

        # Get final measurements
        process = psutil.Process()
        end_time = datetime.now()
        end_cpu_time = process.cpu_times().user + process.cpu_times().system
        end_memory = process.memory_info().rss / (1024 * 1024)

        memory_snapshot_end = None
        if self.enable_memory_profiling and tracemalloc.is_tracing():
            memory_snapshot_end = tracemalloc.take_snapshot()

        # Analyze profile
        result = self._analyze_profile(
            component=profile_data["component"],
            operation=profile_data["operation"],
            start_time=profile_data["start_time"],
            end_time=end_time,
            cpu_time_start=profile_data["start_cpu_time"],
            cpu_time_end=end_cpu_time,
            memory_start=profile_data["start_memory"],
            memory_end=end_memory,
            memory_peak=profile_data["peak_memory"],
            cpu_profiler=cpu_profiler,
            memory_snapshot_start=profile_data.get("memory_snapshot"),
            memory_snapshot_end=memory_snapshot_end,
            metadata=profile_data.get("metadata", {}),
        )

        # Store result
        with self._lock:
            self._profiles[profile_data["component"]].append(result)

        return result

    async def _monitor_memory_async(self, profile_id: str):
        """Monitor memory usage during async profiling."""
        process = psutil.Process()

        while True:
            try:
                await asyncio.sleep(self.memory_snapshot_interval)

                current_memory = process.memory_info().rss / (1024 * 1024)
                current_time = datetime.now()

                with self._lock:
                    if profile_id in self._active_profiles:
                        profile = self._active_profiles[profile_id]
                        profile["memory_samples"].append((current_time, current_memory))
                        profile["peak_memory"] = max(profile["peak_memory"], current_memory)
                    else:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                break

    def _analyze_profile(
        self,
        component: str,
        operation: str,
        start_time: datetime,
        end_time: datetime,
        cpu_time_start: float,
        cpu_time_end: float,
        memory_start: float,
        memory_end: float,
        memory_peak: float,
        cpu_profiler: Optional[cProfile.Profile],
        memory_snapshot_start: Optional[Any],
        memory_snapshot_end: Optional[Any],
        metadata: Dict[str, Any],
    ) -> ProfileResult:
        """Analyze profiling data and create result."""
        duration = (end_time - start_time).total_seconds()
        cpu_time = cpu_time_end - cpu_time_start

        result = ProfileResult(
            component=component,
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            cpu_time_seconds=cpu_time,
            memory_start_mb=memory_start,
            memory_peak_mb=memory_peak,
            memory_allocated_mb=memory_end - memory_start,
            call_count=1,
        )

        # Analyze CPU profile
        if cpu_profiler:
            stats = pstats.Stats(cpu_profiler)
            stats.strip_dirs()
            stats.sort_stats("cumulative")

            # Get top functions
            function_list = stats.get_stats_profile().func_profiles
            for func, (cc, nc, tt, ct, callers) in list(function_list.items())[
                : self.top_functions
            ]:
                func_name = f"{func[0]}:{func[1]}:{func[2]}"
                result.function_stats[func_name] = {
                    "calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "time_per_call": tt / nc if nc > 0 else 0,
                }

        # Analyze memory profile
        if memory_snapshot_start and memory_snapshot_end:
            top_stats = memory_snapshot_end.compare_to(memory_snapshot_start, "lineno")

            for stat in top_stats[:20]:  # Top 20 memory allocations
                result.memory_allocations.append(
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_mb": stat.size_diff / (1024 * 1024),
                        "count": stat.count_diff,
                    }
                )

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(result, metadata)
        result.bottlenecks = bottlenecks

        # Generate recommendations
        recommendations = self._generate_recommendations(result, bottlenecks)
        result.recommendations = recommendations

        return result

    def _identify_bottlenecks(
        self, result: ProfileResult, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # CPU bottlenecks
        if result.function_stats:
            total_time = sum(f["cumulative_time"] for f in result.function_stats.values())
            for func_name, stats in result.function_stats.items():
                if stats["cumulative_time"] > total_time * 0.1:  # >10% of total time
                    bottlenecks.append(
                        {
                            "type": "cpu",
                            "severity": (
                                "high" if stats["cumulative_time"] > total_time * 0.25 else "medium"
                            ),
                            "component": result.component,
                            "function": func_name,
                            "time_percent": (stats["cumulative_time"] / total_time) * 100,
                            "calls": stats["calls"],
                        }
                    )

        # Memory bottlenecks
        if result.memory_allocated_mb > 100:  # >100MB allocated
            bottlenecks.append(
                {
                    "type": "memory",
                    "severity": "high" if result.memory_allocated_mb > 500 else "medium",
                    "component": result.component,
                    "allocated_mb": result.memory_allocated_mb,
                    "peak_mb": result.memory_peak_mb,
                }
            )

        # Component-specific bottlenecks
        settings = self._component_settings.get(result.component, {})

        if result.component == "database":
            # Check for slow queries
            if metadata.get("query_time_ms", 0) > settings.get("slow_query_threshold_ms", 100):
                bottlenecks.append(
                    {
                        "type": "database_query",
                        "severity": "high",
                        "query_time_ms": metadata["query_time_ms"],
                        "query_type": metadata.get("query_type", "unknown"),
                    }
                )

        elif result.component == "agent":
            # Check for memory limit
            if result.memory_peak_mb > settings.get("memory_limit_mb", 50):
                bottlenecks.append(
                    {
                        "type": "agent_memory",
                        "severity": "high",
                        "peak_memory_mb": result.memory_peak_mb,
                        "limit_mb": settings["memory_limit_mb"],
                    }
                )

        return bottlenecks

    def _generate_recommendations(
        self, result: ProfileResult, bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "cpu":
                func = bottleneck["function"]
                time_pct = bottleneck["time_percent"]
                recommendations.append(
                    f"Function '{func}' consumes {time_pct:.1f}% of CPU time. "
                    "Consider optimizing or caching its results."
                )

            elif bottleneck["type"] == "memory":
                allocated = bottleneck["allocated_mb"]
                recommendations.append(
                    f"High memory allocation detected ({allocated:.1f}MB). "
                    "Consider using generators or processing data in chunks."
                )

            elif bottleneck["type"] == "database_query":
                query_time = bottleneck["query_time_ms"]
                recommendations.append(
                    f"Slow database query detected ({query_time}ms). "
                    "Consider adding indexes or optimizing the query."
                )

            elif bottleneck["type"] == "agent_memory":
                peak = bottleneck["peak_memory_mb"]
                limit = bottleneck["limit_mb"]
                recommendations.append(
                    f"Agent memory usage ({peak:.1f}MB) exceeds limit ({limit}MB). "
                    "Consider reducing belief state size or using compression."
                )

        # General recommendations based on metrics
        if result.cpu_time_seconds > result.duration_seconds * 0.8:
            recommendations.append(
                "High CPU utilization detected. Consider using async operations "
                "or distributing work across multiple processes."
            )

        if result.memory_allocated_mb < 0:
            recommendations.append(
                "Memory was freed during operation. This is good for long-running processes."
            )

        return recommendations

    def _record_profile_metrics(self, result: ProfileResult):
        """Record profiling metrics to the unified collector."""
        # Record to unified metrics
        record_metric(
            f"{result.operation}_duration_seconds",
            result.duration_seconds,
            MetricSource.SYSTEM,
            MetricType.HISTOGRAM,
            tags={"component": result.component},
        )

        record_metric(
            f"{result.operation}_cpu_time_seconds",
            result.cpu_time_seconds,
            MetricSource.SYSTEM,
            MetricType.HISTOGRAM,
            tags={"component": result.component},
        )

        record_metric(
            f"{result.operation}_memory_peak_mb",
            result.memory_peak_mb,
            MetricSource.SYSTEM,
            MetricType.GAUGE,
            tags={"component": result.component},
        )

        record_metric(
            f"{result.operation}_memory_allocated_mb",
            result.memory_allocated_mb,
            MetricSource.SYSTEM,
            MetricType.GAUGE,
            tags={"component": result.component},
        )

    def get_component_summary(
        self, component: str, operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary for a component."""
        with self._lock:
            profiles = self._profiles.get(component, [])

            if operation:
                profiles = [p for p in profiles if p.operation == operation]

            if not profiles:
                return {"error": f"No profiles found for {component}/{operation}"}

            # Calculate statistics
            durations = [p.duration_seconds for p in profiles]
            cpu_times = [p.cpu_time_seconds for p in profiles]
            memory_peaks = [p.memory_peak_mb for p in profiles]
            memory_allocated = [p.memory_allocated_mb for p in profiles]

            # Aggregate bottlenecks
            all_bottlenecks = []
            for profile in profiles:
                all_bottlenecks.extend(profile.bottlenecks)

            # Count bottleneck types
            bottleneck_counts = defaultdict(int)
            for b in all_bottlenecks:
                bottleneck_counts[b["type"]] += 1

            return {
                "component": component,
                "operation": operation,
                "profile_count": len(profiles),
                "duration_stats": {
                    "min": np.min(durations),
                    "max": np.max(durations),
                    "avg": np.mean(durations),
                    "std": np.std(durations),
                    "p50": np.percentile(durations, 50),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99),
                },
                "cpu_time_stats": {
                    "min": np.min(cpu_times),
                    "max": np.max(cpu_times),
                    "avg": np.mean(cpu_times),
                    "total": np.sum(cpu_times),
                },
                "memory_peak_stats": {
                    "min": np.min(memory_peaks),
                    "max": np.max(memory_peaks),
                    "avg": np.mean(memory_peaks),
                },
                "memory_allocated_stats": {
                    "min": np.min(memory_allocated),
                    "max": np.max(memory_allocated),
                    "avg": np.mean(memory_allocated),
                    "total": np.sum(memory_allocated),
                },
                "bottleneck_summary": dict(bottleneck_counts),
                "latest_profile": profiles[-1].start_time.isoformat(),
            }

    def export_profiles(
        self,
        filepath: Path,
        component: Optional[str] = None,
        format: str = "json",
    ):
        """Export profiling data."""
        with self._lock:
            if component:
                profiles_to_export = {component: self._profiles.get(component, [])}
            else:
                profiles_to_export = dict(self._profiles)

        if format == "json":
            # Convert to JSON-serializable format
            export_data = {
                "export_time": datetime.now().isoformat(),
                "components": {},
            }

            for comp, profiles in profiles_to_export.items():
                export_data["components"][comp] = []
                for profile in profiles:
                    profile_data = {
                        "operation": profile.operation,
                        "start_time": profile.start_time.isoformat(),
                        "end_time": profile.end_time.isoformat(),
                        "duration_seconds": profile.duration_seconds,
                        "cpu_time_seconds": profile.cpu_time_seconds,
                        "memory_start_mb": profile.memory_start_mb,
                        "memory_peak_mb": profile.memory_peak_mb,
                        "memory_allocated_mb": profile.memory_allocated_mb,
                        "call_count": profile.call_count,
                        "function_stats": profile.function_stats,
                        "memory_allocations": profile.memory_allocations,
                        "bottlenecks": profile.bottlenecks,
                        "recommendations": profile.recommendations,
                    }
                    export_data["components"][comp].append(profile_data)

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_profiles(self, component: Optional[str] = None):
        """Clear stored profiles."""
        with self._lock:
            if component:
                self._profiles[component] = []
            else:
                self._profiles.clear()


# Global profiler instance
component_profiler = ComponentProfiler()


def profile_operation(component: str, operation: str):
    """Decorator for profiling functions."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with component_profiler.profile_component_async(component, operation):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with component_profiler.profile_component(component, operation):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
