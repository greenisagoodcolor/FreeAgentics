"""
Memory Usage Optimization and Garbage Collection Management.

This module implements comprehensive memory optimizations:
1. Memory pooling for frequently allocated objects
2. Object reuse and recycling patterns
3. Memory profiling and leak detection
4. Garbage collection tuning and monitoring
5. Memory usage analysis and reporting
6. Weak reference management
7. Memory-efficient data structures
8. Cache-aware data organization
"""

import gc
import logging
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

from observability.performance_monitor import get_performance_monitor

try:
    import pympler
    from pympler import muppy, summary, tracker

    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Pympler not available, memory profiling will be limited")

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    # System memory
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float

    # Process memory
    process_rss_mb: float
    process_vms_mb: float
    process_shared_mb: float
    process_percent: float

    # Python memory
    python_objects: int
    python_memory_mb: float

    # Garbage collection
    gc_collections: Tuple[int, int, int]
    gc_collected: int
    gc_uncollectable: int

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryLeak:
    """Memory leak detection result."""

    object_type: str
    count: int
    growth_rate: float
    memory_mb: float
    severity: str  # 'low', 'medium', 'high'
    first_seen: datetime
    last_seen: datetime = field(default_factory=datetime.now)


class ObjectPool:
    """Generic object pool for memory optimization."""

    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 100,
        reset_func: Optional[Callable[[Any], None]] = None,
    ):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self.pool: deque[Any] = deque()
        self.created_count = 0
        self.reused_count = 0
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire an object from the pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                return obj
            else:
                obj = self.factory()
                self.created_count += 1
                return obj

    def release(self, obj):
        """Release an object back to the pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object if reset function provided
                if self.reset_func:
                    self.reset_func(obj)
                self.pool.append(obj)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "max_size": self.max_size,
                "created_count": self.created_count,
                "reused_count": self.reused_count,
                "reuse_rate": (self.reused_count / max(self.created_count + self.reused_count, 1))
                * 100,
            }


class MemoryProfiler:
    """Memory profiling and leak detection."""

    def __init__(self, monitoring_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

        # Memory tracking
        self.memory_history: deque[MemoryStats] = deque(maxlen=1000)
        self.object_tracking: defaultdict[type, List[Any]] = defaultdict(list)
        self.leak_detections: List[MemoryLeak] = []

        # Garbage collection tracking
        self.gc_stats = {
            "collections": [0, 0, 0],
            "collected": 0,
            "uncollectable": 0,
        }

        # Object counting
        self.object_counts: defaultdict[type, int] = defaultdict(int)
        self.object_sizes: defaultdict[type, int] = defaultdict(int)

        # Pympler integration
        if PYMPLER_AVAILABLE:
            self.tracker = tracker.SummaryTracker()
        else:
            self.tracker = None

        logger.info("Memory profiler initialized")

    def start_monitoring(self):
        """Start memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                stats = self._collect_memory_stats()
                self.memory_history.append(stats)
                self._detect_memory_leaks()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_memory_stats(self) -> MemoryStats:
        """Collect comprehensive memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = self.process.memory_info()

        # Python objects
        python_objects = len(gc.get_objects())

        # Garbage collection stats
        gc_stats = gc.get_stats()
        collections = tuple(stat["collections"] for stat in gc_stats)

        # Python memory estimation
        python_memory_mb = sys.getsizeof(gc.get_objects()) / 1024 / 1024

        return MemoryStats(
            total_memory_mb=system_memory.total / 1024 / 1024,
            used_memory_mb=system_memory.used / 1024 / 1024,
            available_memory_mb=system_memory.available / 1024 / 1024,
            memory_percent=system_memory.percent,
            process_rss_mb=process_memory.rss / 1024 / 1024,
            process_vms_mb=process_memory.vms / 1024 / 1024,
            process_shared_mb=getattr(process_memory, "shared", 0) / 1024 / 1024,
            process_percent=self.process.memory_percent(),
            python_objects=python_objects,
            python_memory_mb=python_memory_mb,
            gc_collections=collections,
            gc_collected=sum(stat.get("collected", 0) for stat in gc_stats),
            gc_uncollectable=sum(stat.get("uncollectable", 0) for stat in gc_stats),
        )

    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        if len(self.memory_history) < 10:
            return

        # Analyze object count trends
        if PYMPLER_AVAILABLE:
            self._detect_leaks_with_pympler()
        else:
            self._detect_leaks_basic()

    def _detect_leaks_with_pympler(self):
        """Detect leaks using Pympler."""
        try:
            # Get current object summary
            all_objects = muppy.get_objects()
            current_summary = summary.summarize(all_objects)

            # Check for object type growth
            for item in current_summary:
                obj_type = item[0]
                count = item[1]
                size = item[2]

                # Track object growth
                self.object_counts[obj_type] = count
                self.object_sizes[obj_type] = size

                # Simple leak detection based on growth
                if len(self.object_tracking[obj_type]) > 0:
                    previous_count = self.object_tracking[obj_type][-1]
                    growth_rate = (count - previous_count) / max(previous_count, 1)

                    if growth_rate > 0.1 and count > 1000:  # 10% growth with significant count
                        leak = MemoryLeak(
                            object_type=obj_type,
                            count=count,
                            growth_rate=growth_rate,
                            memory_mb=size / 1024 / 1024,
                            severity="high" if growth_rate > 0.5 else "medium",
                            first_seen=datetime.now(),
                        )
                        self.leak_detections.append(leak)
                        logger.warning(
                            f"Potential memory leak detected: {obj_type} - {count} objects, {growth_rate:.1%} growth"
                        )

                self.object_tracking[obj_type].append(count)
                if len(self.object_tracking[obj_type]) > 100:
                    self.object_tracking[obj_type].popleft()

        except Exception as e:
            logger.error(f"Pympler leak detection error: {e}")

    def _detect_leaks_basic(self):
        """Basic leak detection without Pympler."""
        # Count objects by type
        objects = gc.get_objects()
        type_counts = defaultdict(int)

        for obj in objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1

        # Check for growth patterns
        for obj_type, count in type_counts.items():
            if len(self.object_tracking[obj_type]) > 0:
                previous_count = self.object_tracking[obj_type][-1]
                growth_rate = (count - previous_count) / max(previous_count, 1)

                if growth_rate > 0.2 and count > 500:  # 20% growth with significant count
                    leak = MemoryLeak(
                        object_type=obj_type,
                        count=count,
                        growth_rate=growth_rate,
                        memory_mb=0,  # Size not available in basic mode
                        severity="high" if growth_rate > 0.5 else "medium",
                        first_seen=datetime.now(),
                    )
                    self.leak_detections.append(leak)
                    logger.warning(
                        f"Potential memory leak detected: {obj_type} - {count} objects, {growth_rate:.1%} growth"
                    )

            self.object_tracking[obj_type].append(count)
            if len(self.object_tracking[obj_type]) > 100:
                self.object_tracking[obj_type].popleft()

    def get_current_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        if self.memory_history:
            return self.memory_history[-1]
        return self._collect_memory_stats()

    def get_memory_history(self, minutes: int = 30) -> List[MemoryStats]:
        """Get memory history for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [stats for stats in self.memory_history if stats.timestamp >= cutoff_time]

    def get_leak_detections(self) -> List[MemoryLeak]:
        """Get detected memory leaks."""
        return list(self.leak_detections)

    def clear_leak_detections(self):
        """Clear leak detection history."""
        self.leak_detections.clear()

    def take_snapshot(self) -> Dict[str, Any]:
        """Take a memory snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "memory_stats": self.get_current_stats(),
            "top_objects": self._get_top_objects(),
            "gc_stats": self._get_gc_stats(),
        }

        if PYMPLER_AVAILABLE:
            snapshot["pympler_summary"] = self._get_pympler_summary()

        return snapshot

    def _get_top_objects(self) -> List[Dict[str, Any]]:
        """Get top objects by count."""
        objects = gc.get_objects()
        type_counts: defaultdict[str, int] = defaultdict(int)

        for obj in objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1

        # Sort by count and return top 20
        top_objects = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return [{"type": obj_type, "count": count} for obj_type, count in top_objects]

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        gc_stats = gc.get_stats()

        return {
            "collections": [stat["collections"] for stat in gc_stats],
            "collected": [stat.get("collected", 0) for stat in gc_stats],
            "uncollectable": [stat.get("uncollectable", 0) for stat in gc_stats],
            "thresholds": gc.get_threshold(),
            "counts": gc.get_count(),
        }

    def _get_pympler_summary(self) -> List[Dict[str, Any]]:
        """Get Pympler memory summary."""
        if not PYMPLER_AVAILABLE:
            return []

        try:
            all_objects = muppy.get_objects()
            summ = summary.summarize(all_objects)

            return [
                {
                    "type": item[0],
                    "count": item[1],
                    "size_mb": item[2] / 1024 / 1024,
                }
                for item in summ[:20]  # Top 20
            ]
        except Exception as e:
            logger.error(f"Pympler summary error: {e}")
            return []


class GarbageCollectionTuner:
    """Garbage collection optimization and tuning."""

    def __init__(self):
        self.original_thresholds = gc.get_threshold()
        self.tuning_enabled = False
        self.collection_times = deque(maxlen=100)
        self.performance_monitor = get_performance_monitor()

    def enable_tuning(self, workload_type: str = "mixed"):
        """Enable GC tuning for specific workload."""
        if self.tuning_enabled:
            return

        self.tuning_enabled = True

        # Adjust thresholds based on workload
        if workload_type == "high_allocation":
            # More frequent gen0 collection, less frequent gen1/gen2
            gc.set_threshold(500, 5, 5)
        elif workload_type == "long_running":
            # Less frequent collection for long-running processes
            gc.set_threshold(1000, 15, 15)
        elif workload_type == "memory_constrained":
            # More aggressive collection
            gc.set_threshold(400, 5, 5)
        else:
            # Balanced approach
            gc.set_threshold(700, 10, 10)

        logger.info(f"GC tuning enabled for {workload_type} workload")

    def disable_tuning(self):
        """Disable GC tuning and restore original thresholds."""
        if not self.tuning_enabled:
            return

        gc.set_threshold(*self.original_thresholds)
        self.tuning_enabled = False
        logger.info("GC tuning disabled, original thresholds restored")

    def force_collection(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Force garbage collection and measure performance."""
        start_time = time.perf_counter()

        if generation is None:
            collected = gc.collect()
        else:
            collected = gc.collect(generation)

        collection_time = time.perf_counter() - start_time
        self.collection_times.append(collection_time)

        result = {
            "collected": collected,
            "collection_time": collection_time,
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Forced GC collection: {collected} objects collected in {collection_time:.3f}s"
        )
        return result

    def get_tuning_stats(self) -> Dict[str, Any]:
        """Get GC tuning statistics."""
        avg_collection_time = (
            sum(self.collection_times) / len(self.collection_times) if self.collection_times else 0
        )

        return {
            "tuning_enabled": self.tuning_enabled,
            "current_thresholds": gc.get_threshold(),
            "original_thresholds": self.original_thresholds,
            "collection_count": len(self.collection_times),
            "avg_collection_time": avg_collection_time,
            "max_collection_time": max(self.collection_times) if self.collection_times else 0,
            "gc_stats": gc.get_stats(),
        }


class MemoryOptimizer:
    """Main memory optimization coordinator."""

    def __init__(self, monitoring_interval: float = 30.0):
        self.profiler = MemoryProfiler(monitoring_interval)
        self.gc_tuner = GarbageCollectionTuner()
        self.object_pools: Dict[str, ObjectPool] = {}
        self.weak_refs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.performance_monitor = get_performance_monitor()

        # Optimization settings
        self.auto_gc_enabled = True
        self.memory_threshold_mb = 1000  # Trigger optimization at 1GB
        self.leak_detection_enabled = True

        logger.info("Memory optimizer initialized")

    def start_optimization(self, workload_type: str = "mixed"):
        """Start memory optimization."""
        # Start profiling
        self.profiler.start_monitoring()

        # Enable GC tuning
        self.gc_tuner.enable_tuning(workload_type)

        # Create common object pools
        self._create_default_pools()

        logger.info(f"Memory optimization started for {workload_type} workload")

    def stop_optimization(self):
        """Stop memory optimization."""
        self.profiler.stop_monitoring()
        self.gc_tuner.disable_tuning()
        logger.info("Memory optimization stopped")

    def _create_default_pools(self):
        """Create default object pools for common types."""
        # Pool for numpy arrays
        self.object_pools["numpy_array"] = ObjectPool(
            factory=lambda: np.zeros(100, dtype=np.float32),
            max_size=50,
            reset_func=lambda arr: arr.fill(0),
        )

        # Pool for dictionaries
        self.object_pools["dict"] = ObjectPool(
            factory=dict, max_size=100, reset_func=lambda d: d.clear()
        )

        # Pool for lists
        self.object_pools["list"] = ObjectPool(
            factory=list, max_size=100, reset_func=lambda lst: lst.clear()
        )

    def get_object_pool(self, pool_name: str) -> Optional[ObjectPool]:
        """Get object pool by name."""
        return self.object_pools.get(pool_name)

    def create_object_pool(
        self,
        name: str,
        factory: Callable[[], Any],
        max_size: int = 100,
        reset_func: Optional[Callable[[Any], None]] = None,
    ):
        """Create a new object pool."""
        self.object_pools[name] = ObjectPool(factory, max_size, reset_func)

    def register_weak_ref(self, obj):
        """Register object for weak reference tracking."""
        self.weak_refs.add(obj)

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.profiler.get_current_stats()
        if stats:
            return stats.process_rss_mb > self.memory_threshold_mb
        return False

    def optimize_memory(self):
        """Perform memory optimization actions."""
        if self.check_memory_pressure():
            logger.info("Memory pressure detected, performing optimization")

            # Force garbage collection
            gc_result = self.gc_tuner.force_collection()

            # Clear weak references
            alive_refs = len(self.weak_refs)
            self.weak_refs = weakref.WeakSet()

            # Log optimization results
            logger.info(
                f"Memory optimization complete: {gc_result['collected']} objects collected, "
                f"{alive_refs} weak references cleared"
            )

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report."""
        current_stats = self.profiler.get_current_stats()
        memory_history = self.profiler.get_memory_history(30)  # Last 30 minutes
        leak_detections = self.profiler.get_leak_detections()

        # Calculate trends
        if len(memory_history) >= 2:
            memory_trend = (
                memory_history[-1].process_rss_mb - memory_history[0].process_rss_mb
            ) / len(memory_history)
        else:
            memory_trend = 0

        # Object pool stats
        pool_stats = {name: pool.get_stats() for name, pool in self.object_pools.items()}

        return {
            "timestamp": datetime.now().isoformat(),
            "current_stats": current_stats,
            "memory_trend_mb_per_interval": memory_trend,
            "leak_detections": len(leak_detections),
            "gc_tuning": self.gc_tuner.get_tuning_stats(),
            "object_pools": pool_stats,
            "weak_refs_count": len(self.weak_refs),
            "memory_pressure": self.check_memory_pressure(),
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        stats = self.profiler.get_current_stats()
        if stats:
            # Memory usage recommendations
            if stats.process_rss_mb > 1000:
                recommendations.append(
                    "High memory usage detected. Consider implementing object pooling."
                )

            if stats.python_objects > 1000000:
                recommendations.append(
                    "High object count. Consider using generators and iterators."
                )

            # GC recommendations
            gc_stats = self.gc_tuner.get_tuning_stats()
            if gc_stats["avg_collection_time"] > 0.1:
                recommendations.append("Long GC pauses detected. Consider tuning GC thresholds.")

            # Leak detection recommendations
            leaks = self.profiler.get_leak_detections()
            if leaks:
                recommendations.append(
                    f"Memory leaks detected in {len(leaks)} object types. Investigation recommended."
                )

        return recommendations

    @contextmanager
    def memory_tracking(self, operation_name: str):
        """Context manager for tracking memory usage of operations."""
        start_stats = self.profiler.get_current_stats()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_stats = self.profiler.get_current_stats()
            end_time = time.perf_counter()

            if start_stats and end_stats:
                memory_delta = end_stats.process_rss_mb - start_stats.process_rss_mb
                time_delta = end_time - start_time

                logger.info(
                    f"Memory tracking - {operation_name}: "
                    f"{memory_delta:.2f}MB memory change, "
                    f"{time_delta:.3f}s duration"
                )


# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer instance."""
    return memory_optimizer


def start_memory_optimization(workload_type: str = "mixed"):
    """Start memory optimization."""
    memory_optimizer.start_optimization(workload_type)


def stop_memory_optimization():
    """Stop memory optimization."""
    memory_optimizer.stop_optimization()


def get_memory_report() -> Dict[str, Any]:
    """Get comprehensive memory report."""
    return memory_optimizer.get_comprehensive_report()


def force_memory_optimization():
    """Force memory optimization."""
    memory_optimizer.optimize_memory()


# Example usage and benchmarking
def benchmark_memory_optimization():
    """Benchmark memory optimization features."""
    print("=" * 80)
    print("MEMORY OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Start optimization
    start_memory_optimization("high_allocation")

    try:
        # Simulate memory-intensive operations
        print("\nSimulating memory-intensive operations...")

        # Create many objects
        objects = []
        for i in range(10000):
            obj = {
                "id": i,
                "data": np.random.rand(100),
                "metadata": {"created": datetime.now(), "processed": False},
            }
            objects.append(obj)

        # Get memory stats
        stats = memory_optimizer.get_comprehensive_report()

        print(f"Current memory usage: {stats['current_stats'].process_rss_mb:.2f}MB")
        print(f"Object pools: {len(stats['object_pools'])}")
        print(f"Memory pressure: {stats['memory_pressure']}")

        # Force optimization
        print("\nForcing memory optimization...")
        force_memory_optimization()

        # Get updated stats
        final_stats = memory_optimizer.get_comprehensive_report()
        print(f"Final memory usage: {final_stats['current_stats'].process_rss_mb:.2f}MB")

        # Print recommendations
        print("\nOptimization recommendations:")
        for rec in final_stats["optimization_recommendations"]:
            print(f"  - {rec}")

    finally:
        stop_memory_optimization()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_memory_optimization()
