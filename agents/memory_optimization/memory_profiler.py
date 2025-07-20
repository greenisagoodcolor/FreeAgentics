#!/usr/bin/env python3
"""Advanced Memory Profiler for Multi-Agent Systems.

This module provides comprehensive memory profiling capabilities to identify
memory hotspots, track allocations, and optimize memory usage patterns.

Key features:
- Per-agent memory tracking
- Allocation pattern analysis
- Memory leak detection
- Real-time memory visualization
- Allocation stack trace tracking
- Memory usage prediction
"""

import gc
import logging
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""

    timestamp: float
    total_memory_mb: float
    rss_mb: float  # Resident set size
    vms_mb: float  # Virtual memory size
    available_mb: float
    percent_used: float
    gc_stats: Dict[str, int]
    top_allocations: List[Tuple[str, int]]  # (traceback, size)
    agent_memory: Dict[str, float]  # agent_id -> memory_mb


@dataclass
class AllocationPattern:
    """Pattern of memory allocations."""

    location: str  # File:line or function name
    count: int = 0
    total_size: int = 0
    current_size: int = 0
    peak_size: int = 0
    allocations: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        """Initialize allocation pattern."""
        if not hasattr(self, "location") or not self.location:
            self.location = "unknown"

    def record_allocation(self, size: int):
        """Record a new allocation."""
        self.count += 1
        self.total_size += size
        self.current_size += size
        self.peak_size = max(self.peak_size, self.current_size)
        self.allocations.append((time.time(), size))

    def record_deallocation(self, size: int):
        """Record a deallocation."""
        self.current_size = max(0, self.current_size - size)


class AdvancedMemoryProfiler:
    """Advanced memory profiler for multi-agent systems."""

    def __init__(
        self,
        enable_tracemalloc: bool = True,
        snapshot_interval: float = 10.0,
        max_snapshots: int = 100,
        track_allocations: bool = True,
        allocation_threshold_kb: int = 100,  # Track allocations > 100KB
    ):
        """Initialize the memory profiler.

        Args:
            enable_tracemalloc: Enable Python memory allocation tracking
            snapshot_interval: Interval between automatic snapshots
            max_snapshots: Maximum number of snapshots to keep
            track_allocations: Track individual allocations
            allocation_threshold_kb: Minimum allocation size to track
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.track_allocations = track_allocations
        self.allocation_threshold = allocation_threshold_kb * 1024

        # Memory tracking
        self.snapshots = deque(maxlen=max_snapshots)
        self.allocation_patterns: Dict[str, AllocationPattern] = defaultdict(
            lambda: AllocationPattern("unknown")
        )
        self._agent_memory: Dict[str, float] = {}
        self._agent_objects: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )

        # Thread safety
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Memory leak detection
        self._allocation_history: Dict[str, List[float]] = defaultdict(list)
        self._leak_candidates: Set[str] = set()

        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc:
            if not tracemalloc.is_tracing():
                tracemalloc.start(10)  # Keep 10 frames of traceback

        # Process handle for system memory stats
        self.process = psutil.Process() if psutil else None

        logger.info("Initialized advanced memory profiler")

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped memory monitoring")

    def _monitor_loop(self):
        """Execute main monitoring loop."""
        while self._monitoring:
            try:
                self.take_snapshot()
                self._detect_memory_leaks()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")

    def register_agent(self, agent_id: str, agent_object: Any):
        """Register an agent for memory tracking.

        Args:
            agent_id: Agent identifier
            agent_object: Agent object reference
        """
        with self._lock:
            self._agent_objects[agent_id] = agent_object
            self._agent_memory[agent_id] = 0.0

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from memory tracking.

        Args:
            agent_id: Agent identifier
        """
        with self._lock:
            self._agent_objects.pop(agent_id, None)
            self._agent_memory.pop(agent_id, None)

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile memory usage of a specific operation.

        Args:
            operation_name: Name of the operation

        Yields:
            MemorySnapshot: Snapshot before operation
        """
        # Take snapshot before
        snapshot_before = self.take_snapshot(tag=f"{operation_name}_start")

        if self.enable_tracemalloc:
            tracemalloc.start()
            snapshot_tracemalloc_before = tracemalloc.take_snapshot()

        try:
            yield snapshot_before
        finally:
            # Take snapshot after
            snapshot_after = self.take_snapshot(tag=f"{operation_name}_end")

            if self.enable_tracemalloc:
                snapshot_tracemalloc_after = tracemalloc.take_snapshot()

                # Compare snapshots
                top_stats = snapshot_tracemalloc_after.compare_to(
                    snapshot_tracemalloc_before, "lineno"
                )

                # Log significant allocations
                for stat in top_stats[:10]:
                    if stat.size_diff > self.allocation_threshold:
                        logger.info(
                            f"{operation_name} allocation: {stat} "
                            f"(+{stat.size_diff / 1024:.1f} KB)"
                        )

            # Calculate memory delta
            memory_delta = (
                snapshot_after.total_memory_mb
                - snapshot_before.total_memory_mb
            )
            logger.info(
                f"{operation_name} memory delta: {memory_delta:+.1f} MB"
            )

    def take_snapshot(self, tag: Optional[str] = None) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            tag: Optional tag for the snapshot

        Returns:
            MemorySnapshot instance
        """
        with self._lock:
            # Get process memory info
            if self.process:
                mem_info = self.process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)

                # Get system memory
                vm = psutil.virtual_memory()
                available_mb = vm.available / (1024 * 1024)
                percent_used = vm.percent
            else:
                rss_mb = vms_mb = available_mb = percent_used = 0.0

            # Get GC stats
            gc_stats = {f"gen{i}_count": gc.get_count()[i] for i in range(3)}
            gc_stats["objects"] = len(gc.get_objects())

            # Get top allocations if tracemalloc is enabled
            top_allocations = []
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("traceback")

                for stat in top_stats[:20]:  # Top 20 allocations
                    if stat.size > self.allocation_threshold:
                        # Format traceback
                        tb_lines = []
                        for frame in stat.traceback:
                            tb_lines.append(f"{frame.filename}:{frame.lineno}")
                        location = " -> ".join(tb_lines[-3:])  # Last 3 frames
                        top_allocations.append((location, stat.size))

                        # Track allocation pattern
                        if location not in self.allocation_patterns:
                            self.allocation_patterns[
                                location
                            ] = AllocationPattern(location)
                        self.allocation_patterns[location].record_allocation(
                            stat.size
                        )

            # Calculate per-agent memory
            agent_memory = {}
            for agent_id, agent_obj in self._agent_objects.items():
                try:
                    # Estimate agent memory usage
                    agent_size = self._estimate_object_size(agent_obj)
                    agent_memory[agent_id] = agent_size / (1024 * 1024)
                except Exception as e:
                    logger.debug(
                        f"Failed to estimate size for agent {agent_id}: {e}"
                    )

            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_memory_mb=rss_mb,
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                available_mb=available_mb,
                percent_used=percent_used,
                gc_stats=gc_stats,
                top_allocations=top_allocations,
                agent_memory=agent_memory,
            )

            # Store snapshot
            self.snapshots.append(snapshot)

            # Update agent memory tracking
            self._agent_memory.update(agent_memory)

            return snapshot

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate the memory size of an object.

        Args:
            obj: Object to measure

        Returns:
            Estimated size in bytes
        """
        size = sys.getsizeof(obj)

        # Add sizes of common attributes
        if hasattr(obj, "__dict__"):
            for _attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, (list, dict, np.ndarray)):
                    size += sys.getsizeof(attr_value)
                    if isinstance(attr_value, np.ndarray):
                        size += attr_value.nbytes

        return size

    def _detect_memory_leaks(self):
        """Detect potential memory leaks based on allocation patterns."""
        with self._lock:
            current_time = time.time()

            for location, pattern in self.allocation_patterns.items():
                # Skip if not enough data
                if pattern.count < 10:
                    continue

                # Check for continuously growing memory
                recent_allocs = [
                    size
                    for ts, size in pattern.allocations
                    if current_time - ts < 60  # Last minute
                ]

                if len(recent_allocs) >= 5:
                    # Calculate trend
                    avg_size = sum(recent_allocs) / len(recent_allocs)
                    if pattern.current_size > avg_size * 2:
                        # Potential leak - size growing significantly
                        self._leak_candidates.add(location)
                        logger.warning(
                            f"Potential memory leak at {location}: "
                            f"current={pattern.current_size / 1024:.1f}KB, "
                            f"avg={avg_size / 1024:.1f}KB"
                        )

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report.

        Returns:
            Dictionary containing memory analysis
        """
        with self._lock:
            if not self.snapshots:
                return {"error": "No snapshots available"}

            latest = self.snapshots[-1]
            first = self.snapshots[0]

            # Calculate trends
            memory_growth = latest.total_memory_mb - first.total_memory_mb
            time_elapsed = latest.timestamp - first.timestamp
            growth_rate = (
                memory_growth / (time_elapsed / 3600)
                if time_elapsed > 0
                else 0
            )

            # Analyze allocation patterns
            top_allocators = sorted(
                self.allocation_patterns.items(),
                key=lambda x: x[1].current_size,
                reverse=True,
            )[:10]

            # Agent memory analysis
            agent_stats = {
                "count": len(self._agent_memory),
                "total_mb": sum(self._agent_memory.values()),
                "avg_mb": (
                    sum(self._agent_memory.values()) / len(self._agent_memory)
                    if self._agent_memory
                    else 0
                ),
                "per_agent": dict(self._agent_memory),
            }

            return {
                "summary": {
                    "current_memory_mb": latest.total_memory_mb,
                    "memory_growth_mb": memory_growth,
                    "growth_rate_mb_per_hour": growth_rate,
                    "gc_objects": latest.gc_stats.get("objects", 0),
                    "potential_leaks": len(self._leak_candidates),
                },
                "agent_memory": agent_stats,
                "top_allocations": [
                    {
                        "location": location,
                        "current_size_mb": pattern.current_size
                        / (1024 * 1024),
                        "peak_size_mb": pattern.peak_size / (1024 * 1024),
                        "allocation_count": pattern.count,
                    }
                    for location, pattern in top_allocators
                ],
                "memory_timeline": [
                    {
                        "timestamp": s.timestamp,
                        "memory_mb": s.total_memory_mb,
                        "available_mb": s.available_mb,
                    }
                    for s in list(self.snapshots)[-20:]  # Last 20 snapshots
                ],
                "leak_candidates": list(self._leak_candidates),
            }

    def optimize_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Analyze and suggest optimizations for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Optimization suggestions
        """
        with self._lock:
            agent_obj = self._agent_objects.get(agent_id)
            if not agent_obj:
                return {"error": f"Agent {agent_id} not found"}

            suggestions = []

            # Check belief state size
            if hasattr(agent_obj, "beliefs") and isinstance(
                agent_obj.beliefs, np.ndarray
            ):
                belief_size_mb = agent_obj.beliefs.nbytes / (1024 * 1024)
                if belief_size_mb > 5.0:
                    suggestions.append(
                        {
                            "type": "belief_compression",
                            "reason": f"Large belief state: {belief_size_mb:.1f}MB",
                            "suggestion": "Enable belief compression or use sparse representations",
                        }
                    )

            # Check action history
            if (
                hasattr(agent_obj, "action_history")
                and len(agent_obj.action_history) > 1000
            ):
                suggestions.append(
                    {
                        "type": "history_pruning",
                        "reason": f"Long action history: {len(agent_obj.action_history)} items",
                        "suggestion": "Implement circular buffer or periodic pruning",
                    }
                )

            # Check for large cached computations
            cache_attrs = [
                attr for attr in dir(agent_obj) if "cache" in attr.lower()
            ]
            for attr in cache_attrs:
                try:
                    cache_obj = getattr(agent_obj, attr)
                    if hasattr(cache_obj, "__len__") and len(cache_obj) > 100:
                        suggestions.append(
                            {
                                "type": "cache_optimization",
                                "reason": f"Large cache {attr}: {len(cache_obj)} items",
                                "suggestion": "Implement LRU cache or size limits",
                            }
                        )
                except Exception as e:
                    # Skip inaccessible attributes - may be properties with side effects
                    logger.debug(
                        f"Could not inspect cache attribute {attr}: {e}"
                    )

            return {
                "agent_id": agent_id,
                "current_memory_mb": self._agent_memory.get(agent_id, 0.0),
                "suggestions": suggestions,
            }

    def shutdown(self):
        """Shutdown the memory profiler."""
        self.stop_monitoring()

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

        logger.info("Memory profiler shutdown complete")


# Global profiler instance
_global_profiler: Optional[AdvancedMemoryProfiler] = None


def get_memory_profiler() -> AdvancedMemoryProfiler:
    """Get the global memory profiler instance.

    Returns:
        Global AdvancedMemoryProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = AdvancedMemoryProfiler()
    return _global_profiler
