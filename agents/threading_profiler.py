"""Threading Performance Profiler for Multi-Agent Systems.

Comprehensive profiling tool to identify threading bottlenecks and optimization
opportunities in the FreeAgentics multi-agent system.
"""

import cProfile
import gc
import io
import logging
import pstats
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ThreadMetrics:
    """Metrics for individual thread performance."""

    thread_id: int
    name: str
    total_time: float = 0.0
    cpu_time: float = 0.0
    wait_time: float = 0.0
    lock_acquisitions: int = 0
    lock_wait_time: float = 0.0
    context_switches: int = 0
    tasks_completed: int = 0

    def __post_init__(self):
        """Initialize timing metrics after dataclass initialization."""
        self.start_time = time.perf_counter()
        self.start_cpu_time = time.process_time()


@dataclass
class LockMetrics:
    """Metrics for lock contention analysis."""

    lock_id: str
    acquisitions: int = 0
    contentions: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    holders: Dict[int, int] = field(default_factory=dict)  # thread_id -> count


@dataclass
class ThreadingProfile:
    """Complete threading performance profile."""

    total_duration: float
    thread_metrics: Dict[int, ThreadMetrics]
    lock_metrics: Dict[str, LockMetrics]
    pool_metrics: Dict[str, Any]
    memory_metrics: Dict[str, Any]
    gil_contention: float
    optimal_thread_count: int
    bottlenecks: List[str]
    recommendations: List[str]


class InstrumentedLock:
    """Instrumented lock for profiling lock contention."""

    def __init__(self, lock_id: str, profiler: "ThreadingProfiler"):
        """Initialize instrumented lock.

        Args:
            lock_id: Unique identifier for this lock.
            profiler: ThreadingProfiler instance for recording metrics.
        """
        self._lock = threading.RLock()
        self.lock_id = lock_id
        self.profiler = profiler

    def acquire(self, blocking: bool = True, timeout: float = -1):
        """Acquire the lock while recording profiling metrics.

        Args:
            blocking: Whether to block until lock is available.
            timeout: Maximum time to wait for lock acquisition.

        Returns:
            bool: True if lock was acquired, False otherwise.
        """
        thread_id = threading.get_ident()
        start_time = time.perf_counter()

        acquired = self._lock.acquire(blocking, timeout)

        wait_time = time.perf_counter() - start_time

        if acquired:
            self.profiler._record_lock_acquisition(
                self.lock_id, thread_id, wait_time, wait_time > 0.001
            )

        return acquired

    def release(self):
        """Release the lock."""
        self._lock.release()

    def __enter__(self):
        """Enter context manager by acquiring lock."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager by releasing lock."""
        self.release()


class ThreadingProfiler:
    """Comprehensive threading performance profiler."""

    def __init__(self):
        """Initialize the threading profiler."""
        self.thread_metrics: Dict[int, ThreadMetrics] = {}
        self.lock_metrics: Dict[str, LockMetrics] = {}
        self._metrics_lock = threading.Lock()
        self.start_time = None
        self.end_time = None

    def create_instrumented_lock(self, lock_id: str) -> InstrumentedLock:
        """Create an instrumented lock for profiling."""
        with self._metrics_lock:
            if lock_id not in self.lock_metrics:
                self.lock_metrics[lock_id] = LockMetrics(lock_id)
        return InstrumentedLock(lock_id, self)

    def _record_lock_acquisition(
        self,
        lock_id: str,
        thread_id: int,
        wait_time: float,
        was_contended: bool,
    ):
        """Record lock acquisition metrics."""
        with self._metrics_lock:
            if lock_id not in self.lock_metrics:
                self.lock_metrics[lock_id] = LockMetrics(lock_id)

            metrics = self.lock_metrics[lock_id]
            metrics.acquisitions += 1
            metrics.total_wait_time += wait_time
            metrics.max_wait_time = max(metrics.max_wait_time, wait_time)

            if was_contended:
                metrics.contentions += 1

            if thread_id not in metrics.holders:
                metrics.holders[thread_id] = 0
            metrics.holders[thread_id] += 1

            # Update thread metrics
            if thread_id in self.thread_metrics:
                self.thread_metrics[thread_id].lock_acquisitions += 1
                self.thread_metrics[thread_id].lock_wait_time += wait_time

    @contextmanager
    def profile_thread(self, name: str = None):
        """Context manager to profile a thread's execution."""
        thread_id = threading.get_ident()

        with self._metrics_lock:
            if thread_id not in self.thread_metrics:
                thread_name = name or threading.current_thread().name
                self.thread_metrics[thread_id] = ThreadMetrics(
                    thread_id=thread_id, name=thread_name
                )

        metrics = self.thread_metrics[thread_id]

        yield metrics

        # Update metrics
        metrics.total_time = time.perf_counter() - metrics.start_time
        metrics.cpu_time = time.process_time() - metrics.start_cpu_time
        metrics.wait_time = metrics.total_time - metrics.cpu_time
        metrics.tasks_completed += 1

    def profile_thread_pool(
        self,
        pool: ThreadPoolExecutor,
        workload: List[Callable],
        workload_args: List[Tuple] = None,
    ) -> Dict[str, Any]:
        """Profile ThreadPoolExecutor performance."""
        if workload_args is None:
            workload_args = [()] * len(workload)

        self.start_time = time.perf_counter()

        # Submit all tasks
        futures = []
        submit_times = []

        for func, args in zip(workload, workload_args):
            submit_time = time.perf_counter()
            future = pool.submit(self._profile_worker, func, *args)
            futures.append(future)
            submit_times.append(submit_time)

        # Collect results
        completion_times = []
        results = []

        for future in as_completed(futures):
            completion_time = time.perf_counter()
            completion_times.append(completion_time)
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)

        self.end_time = time.perf_counter()
        total_duration = self.end_time - self.start_time

        # Calculate pool metrics
        pool_metrics = {
            "total_duration": total_duration,
            "tasks_submitted": len(workload),
            "tasks_completed": len([r for r in results if r is not None]),
            "avg_submit_time": np.mean(
                [completion_times[i] - submit_times[i] for i in range(len(futures))]
            ),
            "throughput": len(workload) / total_duration,
            "thread_efficiency": self._calculate_thread_efficiency(),
        }

        return pool_metrics

    def _profile_worker(self, func: Callable, *args, **kwargs):
        """Worker function wrapper for profiling."""
        with self.profile_thread():
            return func(*args, **kwargs)

    def _calculate_thread_efficiency(self) -> float:
        """Calculate overall thread efficiency."""
        with self._metrics_lock:
            if not self.thread_metrics:
                return 0.0

            total_cpu_time = sum(m.cpu_time for m in self.thread_metrics.values())
            total_wall_time = sum(m.total_time for m in self.thread_metrics.values())

            if total_wall_time == 0:
                return 0.0

            return total_cpu_time / total_wall_time

    def measure_gil_contention(self, duration: float = 1.0,
        thread_count: int = 4) -> float:
        """Measure GIL contention using CPU-bound workload."""
        import math

        def cpu_bound_work():
            """CPU-intensive work to stress GIL."""
            result = 0
            for i in range(1000000):
                result += math.sqrt(i)
            return result

        # Single-threaded baseline
        single_start = time.perf_counter()
        for _ in range(thread_count):
            cpu_bound_work()
        single_duration = time.perf_counter() - single_start

        # Multi-threaded test
        with ThreadPoolExecutor(max_workers=thread_count) as pool:
            multi_start = time.perf_counter()
            futures = [pool.submit(cpu_bound_work) for _ in range(thread_count)]
            for future in futures:
                future.result()
            multi_duration = time.perf_counter() - multi_start

        # GIL contention factor (1.0 = no benefit from threading)
        gil_contention = multi_duration / single_duration

        return gil_contention

    def find_optimal_thread_count(
        self, workload: Callable, max_threads: int = 32
    ) -> Tuple[int, Dict[int, float]]:
        """Find optimal thread count for given workload."""
        thread_counts = [1, 2, 4, 8, 16, min(32, max_threads)]
        throughputs = {}

        for count in thread_counts:
            with ThreadPoolExecutor(max_workers=count) as pool:
                start = time.perf_counter()

                # Run workload
                futures = [pool.submit(workload) for _ in range(count * 10)]
                for future in as_completed(futures):
                    future.result()

                duration = time.perf_counter() - start
                throughputs[count] = (count * 10) / duration

        # Find optimal count (highest throughput)
        optimal = max(throughputs.keys(), key=lambda k: throughputs[k])

        return optimal, throughputs

    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage across threads."""
        process = psutil.Process()

        # Force garbage collection
        gc.collect()

        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
        }

    def identify_bottlenecks(self) -> List[str]:
        """Identify threading bottlenecks."""
        bottlenecks = []

        # Lock contention
        for lock_id, metrics in self.lock_metrics.items():
            if metrics.contentions > metrics.acquisitions * 0.1:  # >10% contention
                bottlenecks.append(
                    f"High lock contention on {lock_id}: "
                    f"{metrics.contentions}/{metrics.acquisitions} acquisitions contended"
                )

            if metrics.max_wait_time > 0.01:  # >10ms wait
                bottlenecks.append(
                    f"Long lock wait on {lock_id}: max {metrics.max_wait_time * 1000:.1f}ms"
                )

        # Thread efficiency
        efficiency = self._calculate_thread_efficiency()
        if efficiency < 0.7:  # <70% CPU utilization
            bottlenecks.append(f"Low thread efficiency: {efficiency:.1%} CPU utilization")

        # Imbalanced workload
        if self.thread_metrics:
            cpu_times = [m.cpu_time for m in self.thread_metrics.values()]
            if cpu_times:
                cv = np.std(cpu_times) / np.mean(cpu_times) if
                    np.mean(cpu_times) > 0 else 0
                if cv > 0.3:  # >30% coefficient of variation
                    bottlenecks.append(f"Imbalanced thread workload: CV={cv:.1%}")

        return bottlenecks

    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Based on lock contention
        high_contention_locks = [
            (lock_id, metrics)
            for lock_id, metrics in self.lock_metrics.items()
            if metrics.contentions > metrics.acquisitions * 0.1
        ]

        if high_contention_locks:
            recommendations.append(
                "Consider using lock-free data structures or finer-grained locking "
                f"for locks: {', '.join(lock[0] for lock in high_contention_locks)}"
            )

        # Based on thread efficiency
        efficiency = self._calculate_thread_efficiency()
        if efficiency < 0.5:
            recommendations.append(
                "Low CPU utilization suggests I/O-bound workload. "
                "Consider using asyncio or reducing thread count."
            )
        elif efficiency > 0.9:
            recommendations.append(
                "High CPU utilization. Consider increasing thread pool size "
                "or using process-based parallelism for CPU-bound tasks."
            )

        # Based on GIL contention
        gil_contention = self.measure_gil_contention()
        if gil_contention > 0.8:
            recommendations.append(
                f"High GIL contention ({gil_contention:.1f}x slowdown). "
                "Consider using multiprocessing for CPU-bound operations."
            )

        # Memory usage
        memory_metrics = self.profile_memory_usage()
        if memory_metrics["rss_mb"] > 1000:  # >1GB RSS
            recommendations.append(
                f"High memory usage ({memory_metrics['rss_mb']:.0f}MB). "
                "Consider agent pooling or memory optimization."
            )

        return recommendations

    def generate_report(self) -> ThreadingProfile:
        """Generate comprehensive threading profile report."""
        bottlenecks = self.identify_bottlenecks()
        recommendations = self.generate_recommendations()
        gil_contention = self.measure_gil_contention()

        # Determine optimal thread count based on current metrics
        efficiency = self._calculate_thread_efficiency()
        current_threads = len(self.thread_metrics)

        if efficiency < 0.5 and current_threads > 4:
            optimal_threads = max(2, current_threads // 2)
        elif efficiency > 0.9 and gil_contention < 0.5:
            optimal_threads = min(32, current_threads * 2)
        else:
            optimal_threads = current_threads

        return ThreadingProfile(
            total_duration=self.end_time - self.start_time if self.end_time else 0,
            thread_metrics=self.thread_metrics,
            lock_metrics=self.lock_metrics,
            pool_metrics=self.profile_memory_usage(),
            memory_metrics=self.profile_memory_usage(),
            gil_contention=gil_contention,
            optimal_thread_count=optimal_threads,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )


def profile_code_section(func: Callable) -> Tuple[Any, pstats.Stats]:
    """Profile a code section using cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func()

    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    return result, stats


def benchmark_threading_implementations():
    """Benchmark different threading implementations."""

    print("=" * 60)
    print("THREADING PERFORMANCE ANALYSIS")
    print("=" * 60)

    profiler = ThreadingProfiler()

    # Test workload
    def simulate_agent_work():
        """Simulate PyMDP agent computation."""
        import time

        import numpy as np

        # Matrix operations (simulate PyMDP)
        matrix = np.random.rand(100, 100)
        result = np.dot(matrix, matrix.T)

        # Simulate I/O
        time.sleep(0.001)

        return result.sum()

    # 1. Profile ThreadPoolExecutor
    print("\n1. Profiling ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool_metrics = profiler.profile_thread_pool(pool, [simulate_agent_work] * 100)
        print(f"   Throughput: {pool_metrics['throughput']:.1f} tasks/sec")
        print(f"   Efficiency: {pool_metrics['thread_efficiency']:.1%}")

    # 2. Find optimal thread count
    print("\n2. Finding optimal thread count...")
    optimal, throughputs = profiler.find_optimal_thread_count(simulate_agent_work)
    print(f"   Optimal threads: {optimal}")
    for count, throughput in sorted(throughputs.items()):
        print(f"   {count} threads: {throughput:.1f} tasks/sec")

    # 3. Measure GIL contention
    print("\n3. Measuring GIL contention...")
    gil_contention = profiler.measure_gil_contention()
    print(f"   GIL contention factor: {gil_contention:.2f}x")
    if gil_contention > 0.8:
        print("   ⚠️ High GIL contention detected")

    # 4. Generate full report
    print("\n4. Generating optimization report...")
    report = profiler.generate_report()

    print("\nBottlenecks identified:")
    for bottleneck in report.bottlenecks:
        print(f"  - {bottleneck}")

    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_threading_implementations()
