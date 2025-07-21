#!/usr/bin/env python3
"""Enhanced Memory Profiler combining memory_profiler, tracemalloc, and pympler.

This module provides comprehensive memory profiling capabilities by integrating
multiple memory profiling tools to provide deep insights into memory usage patterns,
identify hotspots, and optimize the 34.5MB per agent memory limit.

Key features:
- Integration of memory_profiler, tracemalloc, and pympler
- Memory hotspot identification and analysis
- Memory leak detection and tracking
- Agent-specific memory profiling
- Comprehensive reporting and visualization
"""

import gc
import logging
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    from memory_profiler import memory_usage, profile
except ImportError:
    memory_usage = None
    profile = None

try:
    from pympler import asizeof, muppy, summary, tracker
except ImportError:
    asizeof = None
    muppy = None
    summary = None
    tracker = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryHotspot:
    """Memory hotspot information."""

    location: str
    size_mb: float
    count: int
    type: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class AgentMemoryProfile:
    """Agent-specific memory profile."""

    agent_id: str
    baseline_memory_mb: float
    current_memory_mb: float
    peak_memory_mb: float
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    deallocations: List[Dict[str, Any]] = field(default_factory=list)
    growth_rate_mb_per_hour: float = 0.0
    last_updated: float = field(default_factory=time.time)


class EnhancedMemoryProfiler:
    """Enhanced memory profiler combining multiple profiling tools."""

    def __init__(
        self,
        enable_tracemalloc: bool = True,
        enable_memory_profiler: bool = True,
        enable_pympler: bool = True,
        snapshot_interval: float = 10.0,
        hotspot_threshold_mb: float = 1.0,
        leak_detection_window: int = 10,
    ):
        """Initialize the enhanced memory profiler.

        Args:
            enable_tracemalloc: Enable Python tracemalloc
            enable_memory_profiler: Enable memory_profiler library
            enable_pympler: Enable pympler library
            snapshot_interval: Interval between automatic snapshots
            hotspot_threshold_mb: Minimum size to consider as hotspot
            leak_detection_window: Number of snapshots for leak detection
        """
        self.enable_tracemalloc = enable_tracemalloc and True  # Check availability
        self.enable_memory_profiler = enable_memory_profiler and memory_usage is not None
        self.enable_pympler = enable_pympler and muppy is not None

        self.snapshot_interval = snapshot_interval
        self.hotspot_threshold_mb = hotspot_threshold_mb
        self.leak_detection_window = leak_detection_window

        # Profiling state
        self.snapshots = deque(maxlen=100)
        self.hotspots = []
        self.agent_profiles: Dict[str, AgentMemoryProfile] = {}

        # Tool-specific trackers
        self.tracemalloc_enabled = False
        self.memory_profiler_enabled = False
        self.pympler_enabled = False
        self.pympler_tracker = None

        # Thread safety
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Memory leak detection
        self._allocation_trends: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=leak_detection_window)
        )
        self._suspected_leaks: Set[str] = set()

        logger.info(
            f"Initialized EnhancedMemoryProfiler with tools: "
            f"tracemalloc={self.enable_tracemalloc}, "
            f"memory_profiler={self.enable_memory_profiler}, "
            f"pympler={self.enable_pympler}"
        )

    def start_profiling(self, tools: List[str] = None):
        """Start profiling with specified tools.

        Args:
            tools: List of tools to enable. If None, use all available.
        """
        with self._lock:
            if tools is None:
                tools = []
                if self.enable_tracemalloc:
                    tools.append("tracemalloc")
                if self.enable_memory_profiler:
                    tools.append("memory_profiler")
                if self.enable_pympler:
                    tools.append("pympler")

            # Start tracemalloc
            if "tracemalloc" in tools and self.enable_tracemalloc:
                if not tracemalloc.is_tracing():
                    tracemalloc.start(10)  # Keep 10 frames
                self.tracemalloc_enabled = True
                logger.info("Started tracemalloc profiling")

            # Enable memory_profiler
            if "memory_profiler" in tools and self.enable_memory_profiler:
                self.memory_profiler_enabled = True
                logger.info("Enabled memory_profiler")

            # Start pympler
            if "pympler" in tools and self.enable_pympler:
                if tracker:
                    self.pympler_tracker = tracker.SummaryTracker()
                self.pympler_enabled = True
                logger.info("Started pympler tracking")

    def stop_profiling(self):
        """Stop all profiling."""
        with self._lock:
            # Stop monitoring
            if self._monitoring:
                self._monitoring = False
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5.0)

            # Stop tracemalloc
            if self.tracemalloc_enabled and tracemalloc.is_tracing():
                tracemalloc.stop()
            self.tracemalloc_enabled = False

            # Disable other tools
            self.memory_profiler_enabled = False
            self.pympler_enabled = False
            self.pympler_tracker = None

            logger.info("Stopped all memory profiling")

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        with self._lock:
            if self._monitoring:
                return

            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Started continuous memory monitoring")

    def _monitor_loop(self):
        """Execute main monitoring loop."""
        while self._monitoring:
            try:
                self.take_snapshot("auto_monitor")
                self.analyze_hotspots()
                self._detect_memory_leaks()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a comprehensive memory snapshot.

        Args:
            label: Label for the snapshot

        Returns:
            Dictionary containing snapshot data from all enabled tools
        """
        with self._lock:
            snapshot = {
                "timestamp": time.time(),
                "label": label,
                "tracemalloc": None,
                "memory_profiler": None,
                "pympler": None,
                "gc_stats": self._get_gc_stats(),
            }

            # Tracemalloc snapshot
            if self.tracemalloc_enabled and tracemalloc.is_tracing():
                try:
                    tm_snapshot = tracemalloc.take_snapshot()
                    top_stats = tm_snapshot.statistics("lineno")

                    current, peak = tracemalloc.get_traced_memory()

                    # Get top allocations
                    top_allocations = []
                    for stat in top_stats[:20]:
                        if stat.size > self.hotspot_threshold_mb * 1024 * 1024:
                            frame = stat.traceback[0]
                            top_allocations.append(
                                {
                                    "file": frame.filename,
                                    "line": frame.lineno,
                                    "size_mb": stat.size / 1024 / 1024,
                                    "count": stat.count,
                                    "average_size": (
                                        stat.size / stat.count if stat.count > 0 else 0
                                    ),
                                }
                            )

                    snapshot["tracemalloc"] = {
                        "current_mb": current / 1024 / 1024,
                        "peak_mb": peak / 1024 / 1024,
                        "total_size": sum(stat.size for stat in top_stats) / 1024 / 1024,
                        "total_count": sum(stat.count for stat in top_stats),
                        "top_allocations": top_allocations[:10],
                    }
                except Exception as e:
                    logger.error(f"Error capturing tracemalloc snapshot: {e}")

            # Memory profiler data
            if self.memory_profiler_enabled and psutil:
                try:
                    process = psutil.Process()
                    mem_info = process.memory_info()

                    snapshot["memory_profiler"] = {
                        "rss_mb": mem_info.rss / 1024 / 1024,
                        "vms_mb": mem_info.vms / 1024 / 1024,
                        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                        "percent": process.memory_percent(),
                        "num_threads": process.num_threads(),
                    }
                except Exception as e:
                    logger.error(f"Error capturing memory_profiler data: {e}")

            # Pympler data
            if self.pympler_enabled and muppy:
                try:
                    all_objects = muppy.get_objects()
                    sum_obj = summary.summarize(all_objects)

                    # Get type statistics
                    type_stats = {}
                    for row in sum_obj[:20]:  # Top 20 types
                        type_name = row[0]
                        count = row[1]
                        size = row[2]
                        type_stats[type_name] = {
                            "count": count,
                            "size": size,
                            "size_mb": size / 1024 / 1024,
                        }

                    snapshot["pympler"] = {
                        "total_objects": len(all_objects),
                        "total_size_mb": sum(row[2] for row in sum_obj) / 1024 / 1024,
                        "type_stats": type_stats,
                    }

                    # Update tracker if available
                    if self.pympler_tracker:
                        self.pympler_tracker.print_diff()

                except Exception as e:
                    logger.error(f"Error capturing pympler data: {e}")

            self.snapshots.append(snapshot)
            return snapshot

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        return {
            "collections": {f"gen{i}": gc.get_count()[i] for i in range(3)},
            "collected": gc.get_stats(),
            "garbage": len(gc.garbage),
            "objects": len(gc.get_objects()),
        }

    def analyze_hotspots(self) -> List[MemoryHotspot]:
        """Analyze memory hotspots from all profiling data.

        Returns:
            List of identified memory hotspots
        """
        with self._lock:
            hotspots = []

            if not self.snapshots:
                return hotspots

            latest = self.snapshots[-1]

            # Analyze tracemalloc hotspots
            if latest["tracemalloc"] and "top_allocations" in latest["tracemalloc"]:
                for alloc in latest["tracemalloc"]["top_allocations"]:
                    if alloc["size_mb"] >= self.hotspot_threshold_mb:
                        hotspots.append(
                            MemoryHotspot(
                                location=f"{alloc['file']}:{alloc['line']}",
                                size_mb=alloc["size_mb"],
                                count=alloc["count"],
                                type="allocation",
                                details={
                                    "average_size": alloc["average_size"],
                                    "file": alloc["file"],
                                    "line": alloc["line"],
                                },
                            )
                        )

            # Analyze pympler type hotspots
            if latest["pympler"] and "type_stats" in latest["pympler"]:
                for type_name, stats in latest["pympler"]["type_stats"].items():
                    if stats["size_mb"] >= self.hotspot_threshold_mb:
                        hotspots.append(
                            MemoryHotspot(
                                location=f"Type: {type_name}",
                                size_mb=stats["size_mb"],
                                count=stats["count"],
                                type="object_type",
                                details=stats,
                            )
                        )

            # Sort by size
            self.hotspots = sorted(hotspots, key=lambda x: x.size_mb, reverse=True)

            # Track allocation trends for leak detection
            for hotspot in self.hotspots:
                self._allocation_trends[hotspot.location].append(hotspot.size_mb)

            return self.hotspots

    def _detect_memory_leaks(self):
        """Detect potential memory leaks based on trends."""
        with self._lock:
            self._suspected_leaks.clear()

            for location, sizes in self._allocation_trends.items():
                if len(sizes) < self.leak_detection_window // 2:
                    continue

                # Check for consistent growth
                growth_count = sum(1 for i in range(1, len(sizes)) if sizes[i] > sizes[i - 1])

                if growth_count >= len(sizes) * 0.7:  # 70% growth trend
                    # Calculate growth rate
                    if len(sizes) >= 2:
                        growth_rate = (sizes[-1] - sizes[0]) / len(sizes)
                        if growth_rate > 0.1:  # Growing by >0.1MB per snapshot
                            self._suspected_leaks.add(location)
                            logger.warning(
                                f"Suspected memory leak at {location}: "
                                f"growth rate {growth_rate:.2f} MB/snapshot"
                            )

    def register_agent(self, agent_id: str, agent_obj: Any):
        """Register an agent for memory tracking.

        Args:
            agent_id: Agent identifier
            agent_obj: Agent object to track
        """
        with self._lock:
            # Take initial snapshot
            initial_memory = self._estimate_agent_memory(agent_obj)

            self.agent_profiles[agent_id] = AgentMemoryProfile(
                agent_id=agent_id,
                baseline_memory_mb=initial_memory,
                current_memory_mb=initial_memory,
                peak_memory_mb=initial_memory,
            )

            logger.info(
                f"Registered agent {agent_id} with baseline memory" f" {initial_memory:.2f} MB"
            )

    def update_agent_memory(self, agent_id: str, agent_obj: Any):
        """Update memory tracking for an agent.

        Args:
            agent_id: Agent identifier
            agent_obj: Agent object
        """
        with self._lock:
            if agent_id not in self.agent_profiles:
                self.register_agent(agent_id, agent_obj)
                return

            profile = self.agent_profiles[agent_id]
            current_memory = self._estimate_agent_memory(agent_obj)

            profile.current_memory_mb = current_memory
            profile.peak_memory_mb = max(profile.peak_memory_mb, current_memory)

            # Calculate growth rate
            time_diff = time.time() - profile.last_updated
            if time_diff > 0:
                memory_diff = current_memory - profile.baseline_memory_mb
                profile.growth_rate_mb_per_hour = (memory_diff / time_diff) * 3600

            profile.last_updated = time.time()

    def _estimate_agent_memory(self, agent_obj: Any) -> float:
        """Estimate memory usage of an agent object.

        Args:
            agent_obj: Agent object to measure

        Returns:
            Estimated memory usage in MB
        """
        if self.enable_pympler and asizeof:
            try:
                return asizeof.asizeof(agent_obj) / 1024 / 1024
            except Exception:  # nosec B110 # Safe fallback to simpler memory estimation
                pass

        # Fallback to simple estimation
        total_bytes = sys.getsizeof(agent_obj)

        # Add major attributes
        for attr in [
            "beliefs",
            "action_history",
            "observations",
            "transition_matrix",
        ]:
            if hasattr(agent_obj, attr):
                value = getattr(agent_obj, attr)
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                else:
                    total_bytes += sys.getsizeof(value)

        return total_bytes / 1024 / 1024

    def compare_snapshots(self, idx1: int, idx2: int) -> Dict[str, Any]:
        """Compare two snapshots to identify changes.

        Args:
            idx1: First snapshot index
            idx2: Second snapshot index

        Returns:
            Comparison results
        """
        with self._lock:
            if idx1 >= len(self.snapshots) or idx2 >= len(self.snapshots):
                return {"error": "Invalid snapshot indices"}

            snap1 = self.snapshots[idx1]
            snap2 = self.snapshots[idx2]

            comparison = {
                "time_diff": snap2["timestamp"] - snap1["timestamp"],
                "label1": snap1["label"],
                "label2": snap2["label"],
            }

            # Compare tracemalloc
            if snap1["tracemalloc"] and snap2["tracemalloc"]:
                tm1 = snap1["tracemalloc"]
                tm2 = snap2["tracemalloc"]
                comparison["tracemalloc_diff"] = {
                    "current_diff": tm2["current_mb"] - tm1["current_mb"],
                    "peak_diff": tm2["peak_mb"] - tm1["peak_mb"],
                    "total_size_diff": tm2["total_size"] - tm1["total_size"],
                    "total_count_diff": tm2["total_count"] - tm1["total_count"],
                }

            # Compare memory_profiler
            if snap1["memory_profiler"] and snap2["memory_profiler"]:
                mp1 = snap1["memory_profiler"]
                mp2 = snap2["memory_profiler"]
                comparison["memory_profiler_diff"] = {
                    "rss_diff": mp2["rss_mb"] - mp1["rss_mb"],
                    "vms_diff": mp2["vms_mb"] - mp1["vms_mb"],
                    "percent_diff": mp2["percent"] - mp1["percent"],
                }

            # Compare pympler
            if snap1["pympler"] and snap2["pympler"]:
                pm1 = snap1["pympler"]
                pm2 = snap2["pympler"]
                comparison["pympler_diff"] = {
                    "object_count_diff": pm2["total_objects"] - pm1["total_objects"],
                    "total_size_diff": pm2["total_size_mb"] - pm1["total_size_mb"],
                }

            return comparison

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager to profile a specific operation.

        Args:
            operation_name: Name of the operation to profile
        """
        # Take before snapshot
        before_snapshot = self.take_snapshot(f"{operation_name}_start")

        # Start detailed profiling if available
        if self.enable_memory_profiler and memory_usage:
            pass  # Memory profiling enabled but not actively used in this context

        start_time = time.time()

        try:
            yield before_snapshot
        finally:
            # Take after snapshot
            end_time = time.time()
            self.take_snapshot(f"{operation_name}_end")

            # Log results
            duration = end_time - start_time
            comparison = self.compare_snapshots(-2, -1)

            logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")

            if comparison.get("tracemalloc_diff"):
                tm_diff = comparison["tracemalloc_diff"]
                logger.info(f"  Tracemalloc: {tm_diff['current_diff']:+.2f} MB")

            if comparison.get("memory_profiler_diff"):
                mp_diff = comparison["memory_profiler_diff"]
                logger.info(f"  RSS: {mp_diff['rss_diff']:+.2f} MB")

    def _generate_header_section(self) -> List[str]:
        """Generate the header section of the report."""
        report = ["=" * 60]
        report.append("Enhanced Memory Profiling Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        return report

    def _generate_tools_section(self) -> List[str]:
        """Generate the profiling tools status section."""
        report = ["## Profiling Tools"]
        report.append(
            f"- Tracemalloc: {'Enabled' if self.tracemalloc_enabled else
            'Disabled'}"
        )
        report.append(
            f"- Memory Profiler: {'Enabled' if self.memory_profiler_enabled else
                'Disabled'}"
        )
        report.append(f"- Pympler: {'Enabled' if self.pympler_enabled else 'Disabled'}")
        report.append("")
        return report

    def _generate_summary_section(self) -> List[str]:
        """Generate the summary statistics section."""
        report = ["## Summary Statistics"]
        report.append(f"Total snapshots: {len(self.snapshots)}")

        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            duration = (last["timestamp"] - first["timestamp"]) / 60
            report.append(f"Profiling duration: {duration:.1f} minutes")

            # Memory growth
            if first["tracemalloc"] and last["tracemalloc"]:
                current_growth = (
                    last["tracemalloc"]["current_mb"] - first["tracemalloc"]["current_mb"]
                )
                peak_growth = last["tracemalloc"]["peak_mb"] - first["tracemalloc"]["peak_mb"]
                report.append(f"Tracemalloc current growth: {current_growth:+.2f} MB")
                report.append(f"Tracemalloc peak growth: {peak_growth:+.2f} MB")

            if first["memory_profiler"] and last["memory_profiler"]:
                rss_growth = last["memory_profiler"]["rss_mb"] - first["memory_profiler"]["rss_mb"]
                report.append(f"RSS memory growth: {rss_growth:+.2f} MB")

        report.append("")
        return report

    def _generate_hotspots_section(self) -> List[str]:
        """Generate the memory hotspots section."""
        if not self.hotspots:
            return []

        report = ["## Memory Hotspots"]
        report.append(
            f"Found {len(self.hotspots)} hotspots (threshold: {self.hotspot_threshold_mb} MB)"
        )
        report.append("")

        for i, hotspot in enumerate(self.hotspots[:10], 1):
            report.append(f"{i}. {hotspot.location}")
            report.append(f"   Size: {hotspot.size_mb:.2f} MB")
            report.append(f"   Count: {hotspot.count:,}")
            report.append(f"   Type: {hotspot.type}")
            if hotspot.details:
                for key, value in hotspot.details.items():
                    if key not in ["size_mb", "count"]:
                        report.append(f"   {key}: {value}")
            report.append("")
        return report

    def _generate_leaks_section(self) -> List[str]:
        """Generate the suspected memory leaks section."""
        if not self._suspected_leaks:
            return []

        report = ["## Suspected Memory Leaks"]
        report.append(f"Found {len(self._suspected_leaks)} potential leaks")
        report.append("")

        for location in sorted(self._suspected_leaks):
            if location in self._allocation_trends:
                sizes = list(self._allocation_trends[location])
                if len(sizes) >= 2:
                    growth = sizes[-1] - sizes[0]
                    report.append(f"- {location}")
                    report.append(f"  Initial: {sizes[0]:.2f} MB")
                    report.append(f"  Current: {sizes[-1]:.2f} MB")
                    report.append(f"  Growth: {growth:+.2f} MB")
                    report.append("")
        return report

    def _generate_agents_section(self) -> List[str]:
        """Generate the agent memory profiles section."""
        if not self.agent_profiles:
            return []

        report = ["## Agent Memory Profiles"]
        report.append(f"Tracking {len(self.agent_profiles)} agents")
        report.append("")

        # Sort by current memory usage
        sorted_agents = sorted(
            self.agent_profiles.items(),
            key=lambda x: x[1].current_memory_mb,
            reverse=True,
        )

        for agent_id, profile in sorted_agents[:10]:
            report.append(f"Agent: {agent_id}")
            report.append(f"  Baseline: {profile.baseline_memory_mb:.2f} MB")
            report.append(f"  Current: {profile.current_memory_mb:.2f} MB")
            report.append(f"  Peak: {profile.peak_memory_mb:.2f} MB")

            if profile.growth_rate_mb_per_hour != 0:
                report.append(f"  Growth rate: {profile.growth_rate_mb_per_hour:+.2f} MB/hour")
            report.append("")
        return report

    def _generate_snapshots_section(self) -> List[str]:
        """Generate the recent snapshots section."""
        report = ["## Recent Snapshots"]
        for snap in list(self.snapshots)[-5:]:
            report.append(
                f"\nSnapshot: {snap['label']} @ {datetime.fromtimestamp(snap['timestamp']).strftime('%H:%M:%S')}"
            )

            if snap["tracemalloc"]:
                tm = snap["tracemalloc"]
                report.append(
                    f"  Tracemalloc: {tm['current_mb']:.2f} MB current, "
                    f"{tm['peak_mb']:.2f} MB peak"
                )

            if snap["memory_profiler"]:
                mp = snap["memory_profiler"]
                report.append(
                    f"  Memory Profiler: {mp['rss_mb']:.2f} MB RSS, "
                    f"{mp['percent']:.1f}% of total"
                )

            if snap["pympler"]:
                pm = snap["pympler"]
                report.append(
                    f"  Pympler: {pm['total_objects']:,} objects, "
                    f"{pm['total_size_mb']:.2f} MB total"
                )
        return report

    def generate_report(self) -> str:
        """Generate comprehensive memory profiling report.

        Returns:
            Formatted report string
        """
        with self._lock:
            if not self.snapshots:
                return "No profiling data available"

            report = []
            report.extend(self._generate_header_section())
            report.extend(self._generate_tools_section())
            report.extend(self._generate_summary_section())
            report.extend(self._generate_hotspots_section())
            report.extend(self._generate_leaks_section())
            report.extend(self._generate_agents_section())
            report.extend(self._generate_snapshots_section())

            report.append("")
            report.append("=" * 60)

            return "\n".join(report)

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get memory optimization recommendations based on profiling data.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Check for large allocations
        for hotspot in self.hotspots[:10]:
            if hotspot.size_mb > 10.0:
                recommendations.append(
                    {
                        "type": "large_allocation",
                        "location": hotspot.location,
                        "size_mb": hotspot.size_mb,
                        "recommendation": "Consider using memory-mapped files or "
                        "lazy loading for large data structures",
                    }
                )

        # Check for memory leaks
        for location in self._suspected_leaks:
            recommendations.append(
                {
                    "type": "memory_leak",
                    "location": location,
                    "recommendation": "Potential memory leak detected. Check for circular references or unbounded growth",
                }
            )

        # Check agent memory usage
        for agent_id, profile in self.agent_profiles.items():
            if profile.current_memory_mb > 15.0:  # Above target
                recommendations.append(
                    {
                        "type": "agent_memory",
                        "agent_id": agent_id,
                        "current_mb": profile.current_memory_mb,
                        "recommendation": "Agent exceeds memory target. Enable belief compression and memory optimization",
                    }
                )

        return recommendations


# Global profiler instance
_global_profiler: Optional[EnhancedMemoryProfiler] = None


def get_enhanced_profiler() -> EnhancedMemoryProfiler:
    """Get the global enhanced memory profiler instance.

    Returns:
        Global EnhancedMemoryProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = EnhancedMemoryProfiler()
    return _global_profiler
