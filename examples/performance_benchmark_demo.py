#!/usr/bin/env python3
"""
Performance Benchmark Demo.
=========================

This script demonstrates how to use the performance benchmarking system
to measure and track performance of FreeAgentics components.
"""


# Add project root to path
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from benchmarks.ci_integration import (
    CIIntegration,
    PerformanceBaseline,
    PerformanceResult,
    RegressionDetector,
)
from benchmarks.performance_suite import (
    BenchmarkMetrics,
    MemoryTracker,
    PerformanceReportGenerator,
    track_performance,
)


def demo_memory_tracking():
    """Demonstrate memory tracking capabilities."""
    print("=== Memory Tracking Demo ===")

    tracker = MemoryTracker()
    tracker.start()

    print(f"Initial memory: {tracker.start_memory:.2f} MB")

    # Simulate some work that uses memory
    data = []
    for i in range(5):
        # Allocate some memory
        chunk = list(range(10000))
        data.append(chunk)

        current_mem = tracker.sample()
        print(f"After chunk {i + 1}: {current_mem:.2f} MB")

        time.sleep(0.1)

    start, end, peak = tracker.stop()
    print("\nMemory summary:")
    print(f"  Start: {start:.2f} MB")
    print(f"  End: {end:.2f} MB")
    print(f"  Peak: {peak:.2f} MB")
    print(f"  Growth: {end - start:.2f} MB")


def demo_performance_tracking():
    """Demonstrate performance tracking context manager."""
    print("\n=== Performance Tracking Demo ===")

    with track_performance("example_operation", "demo") as metrics:
        # Simulate some CPU-intensive work
        result = sum(i * i for i in range(100000))

        # Add some metadata
        metrics.metadata["result"] = result
        metrics.metadata["iterations"] = 100000

        # Simulate some I/O
        time.sleep(0.01)

    print(f"Operation: {metrics.name}")
    print(f"Category: {metrics.category}")
    print(f"Duration: {metrics.duration_ms:.2f} ms")
    print(f"Memory growth: {metrics.memory_growth_mb:.2f} MB")
    print(f"CPU usage: {metrics.cpu_percent:.2f}%")
    print(f"Metadata: {metrics.metadata}")


def demo_regression_detection():
    """Demonstrate regression detection."""
    print("\n=== Regression Detection Demo ===")

    # Create temporary baseline
    baseline_path = Path("demo_baseline.json")
    baseline = PerformanceBaseline(baseline_path)

    # Create baseline results
    baseline_results = [
        PerformanceResult(
            benchmark_name="demo_benchmark",
            category="demo",
            duration_ms=100.0,
            throughput_ops_sec=10.0,
            memory_mb=50.0,
            cpu_percent=25.0,
            timestamp=datetime.now().isoformat(),
        )
    ]

    print("Setting baseline...")
    baseline.update_baseline(baseline_results)

    # Create detector
    detector = RegressionDetector(baseline)

    # Test with improved results
    improved_results = [
        PerformanceResult(
            benchmark_name="demo_benchmark",
            category="demo",
            duration_ms=80.0,  # 20% improvement
            throughput_ops_sec=12.5,
            memory_mb=40.0,
            cpu_percent=20.0,
            timestamp=datetime.now().isoformat(),
        )
    ]

    print("Testing with improved results...")
    regressions, improvements = detector.detect_regressions(improved_results)

    print(f"Regressions detected: {len(regressions)}")
    print(f"Improvements detected: {len(improvements)}")

    for improvement in improvements:
        print(
            f"  {improvement.benchmark_name} ({improvement.metric}): "
            f"{improvement.regression_percent:.1f}% improvement"
        )

    # Test with regressed results
    regressed_results = [
        PerformanceResult(
            benchmark_name="demo_benchmark",
            category="demo",
            duration_ms=130.0,  # 30% regression
            throughput_ops_sec=7.7,
            memory_mb=65.0,
            cpu_percent=35.0,
            timestamp=datetime.now().isoformat(),
        )
    ]

    print("\nTesting with regressed results...")
    regressions, improvements = detector.detect_regressions(regressed_results)

    print(f"Regressions detected: {len(regressions)}")
    print(f"Improvements detected: {len(improvements)}")

    for regression in regressions:
        print(
            f"  {regression.benchmark_name} ({regression.metric}): "
            f"{regression.regression_percent:.1f}% regression ({regression.severity})"
        )

    # Cleanup
    baseline_path.unlink(missing_ok=True)


def demo_ci_integration():
    """Demonstrate CI integration workflow."""
    print("\n=== CI Integration Demo ===")

    # Create temporary directories
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        baseline_dir = Path(tmpdir) / "baselines"
        results_dir.mkdir()
        baseline_dir.mkdir()

        # Create CI integration
        ci = CIIntegration(results_dir, baseline_dir)

        # Create some test results
        test_results = [
            PerformanceResult(
                benchmark_name="agent_spawn",
                category="agents",
                duration_ms=45.0,
                throughput_ops_sec=22.2,
                memory_mb=30.0,
                cpu_percent=15.0,
                timestamp=datetime.now().isoformat(),
            ),
            PerformanceResult(
                benchmark_name="message_throughput",
                category="communication",
                duration_ms=0.5,
                throughput_ops_sec=2000.0,
                memory_mb=10.0,
                cpu_percent=5.0,
                timestamp=datetime.now().isoformat(),
            ),
        ]

        print("Running CI integration workflow...")

        # Set baseline
        ci.baseline.update_baseline(test_results)
        print("✓ Baseline updated")

        # Add to history
        ci.history.add_results(test_results)
        print("✓ Added to history")

        # Run regression check (should pass)
        report = ci.run_regression_check(test_results)
        print(f"✓ Regression check: {report['overall_status']}")

        # Generate GitHub comment
        comment = ci.generate_github_comment(report)
        print("✓ GitHub comment generated")

        print("\nExample GitHub comment:")
        print("-" * 50)
        print(comment[:300] + "..." if len(comment) > 300 else comment)


def demo_report_generation():
    """Demonstrate report generation."""
    print("\n=== Report Generation Demo ===")

    # Create sample metrics
    metrics = []

    # Simulate running multiple benchmarks
    benchmark_names = [
        "agent_spawn",
        "message_pass",
        "database_query",
        "websocket_connect",
    ]
    categories = ["agents", "communication", "database", "websocket"]

    for name, category in zip(benchmark_names, categories):
        metric = BenchmarkMetrics(
            name=name,
            category=category,
            duration_ms=50.0 + hash(name) % 100,  # Semi-random duration
            operations_per_second=1000.0 / (50.0 + hash(name) % 100),
            memory_start_mb=100.0,
            memory_end_mb=120.0 + hash(name) % 50,
            memory_peak_mb=150.0 + hash(name) % 50,
            cpu_percent=20.0 + hash(name) % 30,
            metadata={"iterations": 1000, "test_mode": True},
        )
        metrics.append(metric)

    # Generate report
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        report = PerformanceReportGenerator.generate_report(metrics, output_dir)

        print(f"Generated report for {report['summary']['total_benchmarks']} benchmarks")
        print(f"Categories: {', '.join(report['summary']['categories'])}")
        print(f"Total duration: {report['summary']['total_duration_ms']:.2f} ms")
        print(f"Average memory growth: {report['summary']['avg_memory_growth_mb']:.2f} MB")

        # Show category breakdown
        print("\nCategory breakdown:")
        for category, stats in report["benchmarks"].items():
            print(
                f"  {category}: {stats['count']} benchmarks, "
                f"avg {stats['avg_duration_ms']:.2f}ms, "
                f"{stats['avg_ops_per_sec']:.1f} ops/sec"
            )


def main():
    """Run all demos."""
    print("🚀 FreeAgentics Performance Benchmarking Demo")
    print("=" * 60)

    try:
        demo_memory_tracking()
        demo_performance_tracking()
        demo_regression_detection()
        demo_ci_integration()
        demo_report_generation()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nTo run actual benchmarks:")
        print("  pytest benchmarks/performance_suite.py -v --benchmark-only")
        print("  ./benchmarks/run_performance_benchmarks.sh")
        print("\nTo check for regressions:")
        print(
            "  python benchmarks/ci_integration.py --results-file results.json --fail-on-regression"
        )

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
