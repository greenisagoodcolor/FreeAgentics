"""Test the benchmark framework itself works correctly."""

import pytest

from tests.performance.pymdp_benchmarks import (
    BenchmarkResult,
    BenchmarkTimer,
    MemoryMonitor,
    PyMDPBenchmark,
)


class SimpleBenchmark(PyMDPBenchmark):
    """Simple test benchmark."""

    def __init__(self):
        super().__init__("simple_test")

    def setup(self):
        pass

    def run_iteration(self):
        # Simple operation
        total = sum(range(1000))
        return {"result": total}


def test_benchmark_timer():
    """Test the benchmark timer."""
    timer = BenchmarkTimer()

    timer.start()
    # Do some work
    sum(range(10000))
    lap_time = timer.lap()

    assert lap_time > 0
    assert len(timer.laps) == 1

    # Second lap
    sum(range(10000))
    lap_time2 = timer.lap()

    assert lap_time2 > lap_time  # Second lap includes first lap time
    assert len(timer.laps) == 2


def test_memory_monitor():
    """Test the memory monitor."""
    monitor = MemoryMonitor()

    monitor.start()

    # Allocate some memory
    data = [0] * 1000000

    usage = monitor.get_usage()
    assert usage >= 0  # Could be 0 if GC collected something else


def test_simple_benchmark():
    """Test running a simple benchmark."""
    benchmark = SimpleBenchmark()

    # Run with minimal iterations
    result = benchmark.run(iterations=10, warmup=2)

    assert isinstance(result, BenchmarkResult)
    assert result.name == "simple_test"
    assert result.iterations == 10
    assert result.mean_time_ms > 0
    assert result.std_dev_ms >= 0
    assert len(result.percentiles) == 4
    assert "p50" in result.percentiles
    assert result.memory_usage_mb >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
