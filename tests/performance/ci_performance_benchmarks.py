#!/usr/bin/env python3
"""
CI/CD Performance Benchmark Suite

This module provides automated performance benchmarks designed to run in CI/CD
pipelines to detect performance regressions. It includes baseline measurements,
regression detection, and performance reporting.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil

from agents.base_agent import BasicExplorerAgent
from tests.performance.performance_profiler import ComponentProfiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance baselines (updated based on historical data)
PERFORMANCE_BASELINES = {
    "single_agent_inference_ms": 50.0,  # Target: <50ms per inference
    "memory_per_agent_mb": 34.5,  # Current baseline: 34.5MB per agent
    "coordination_efficiency_percent": 70.0,  # Target: >70% at 10 agents
    "cache_hit_rate_percent": 22.0,  # Current baseline: 22% hit rate
    "threading_advantage_factor": 3.0,  # Threading should be 3x+ faster than multiprocessing
}

# Regression thresholds
REGRESSION_THRESHOLDS = {
    "critical": 25.0,  # >25% regression is critical
    "warning": 10.0,  # >10% regression is warning
    "minor": 5.0,  # >5% regression is minor
}


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.profiler = ComponentProfiler()

    def run(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        raise NotImplementedError

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate result against baseline and return status."""
        raise NotImplementedError


class SingleAgentInferenceBenchmark(PerformanceBenchmark):
    """Benchmark single agent inference performance."""

    def __init__(self):
        super().__init__(
            "single_agent_inference",
            "Measures single agent inference latency and memory usage",
        )

    def run(self) -> Dict[str, Any]:
        """Run single agent inference benchmark."""
        logger.info("Running single agent inference benchmark...")

        # Create test agent
        agent = BasicExplorerAgent(
            "benchmark-agent", "Benchmark Agent", grid_size=5
        )
        agent.start()

        # Measure baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Benchmark inference operations
        num_operations = 100
        inference_times = []

        for i in range(num_operations):
            observation = {
                "position": [2, 2],
                "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            }

            start_time = time.time()

            # Full inference cycle
            agent.perceive(observation)
            if agent._should_update_beliefs():
                agent.update_beliefs()
            agent.select_action()

            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            # Brief pause to prevent overwhelming the system
            time.sleep(0.001)

        # Measure peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_per_agent = peak_memory - baseline_memory

        agent.stop()

        # Calculate statistics
        result = {
            "mean_inference_time_ms": np.mean(inference_times),
            "median_inference_time_ms": np.median(inference_times),
            "p95_inference_time_ms": np.percentile(inference_times, 95),
            "p99_inference_time_ms": np.percentile(inference_times, 99),
            "std_inference_time_ms": np.std(inference_times),
            "memory_per_agent_mb": memory_per_agent,
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "operations_completed": num_operations,
            "total_duration_s": sum(inference_times) / 1000,
            "ops_per_second": num_operations / (sum(inference_times) / 1000),
        }

        logger.info(
            f"Single agent inference: {result['mean_inference_time_ms']:.2f}ms avg, "
            f"{result['memory_per_agent_mb']:.2f}MB memory"
        )

        return result

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single agent inference results."""
        validation = {
            "benchmark": self.name,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "metrics": result,
        }

        # Check inference time regression
        baseline_time = PERFORMANCE_BASELINES["single_agent_inference_ms"]
        actual_time = result["mean_inference_time_ms"]
        time_regression = ((actual_time - baseline_time) / baseline_time) * 100

        if time_regression > REGRESSION_THRESHOLDS["critical"]:
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "inference_time",
                    "regression_percent": time_regression,
                    "severity": "critical",
                    "baseline": baseline_time,
                    "actual": actual_time,
                    "threshold": REGRESSION_THRESHOLDS["critical"],
                }
            )
        elif time_regression > REGRESSION_THRESHOLDS["warning"]:
            validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "inference_time",
                    "regression_percent": time_regression,
                    "severity": "warning",
                    "baseline": baseline_time,
                    "actual": actual_time,
                    "threshold": REGRESSION_THRESHOLDS["warning"],
                }
            )
        elif time_regression < -REGRESSION_THRESHOLDS["minor"]:
            validation["improvements"].append(
                {
                    "metric": "inference_time",
                    "improvement_percent": -time_regression,
                    "baseline": baseline_time,
                    "actual": actual_time,
                }
            )

        # Check memory regression
        baseline_memory = PERFORMANCE_BASELINES["memory_per_agent_mb"]
        actual_memory = result["memory_per_agent_mb"]
        memory_regression = (
            (actual_memory - baseline_memory) / baseline_memory
        ) * 100

        if memory_regression > REGRESSION_THRESHOLDS["critical"]:
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "memory_usage",
                    "regression_percent": memory_regression,
                    "severity": "critical",
                    "baseline": baseline_memory,
                    "actual": actual_memory,
                    "threshold": REGRESSION_THRESHOLDS["critical"],
                }
            )
        elif memory_regression > REGRESSION_THRESHOLDS["warning"]:
            if validation["status"] == "pass":
                validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "memory_usage",
                    "regression_percent": memory_regression,
                    "severity": "warning",
                    "baseline": baseline_memory,
                    "actual": actual_memory,
                    "threshold": REGRESSION_THRESHOLDS["warning"],
                }
            )
        elif memory_regression < -REGRESSION_THRESHOLDS["minor"]:
            validation["improvements"].append(
                {
                    "metric": "memory_usage",
                    "improvement_percent": -memory_regression,
                    "baseline": baseline_memory,
                    "actual": actual_memory,
                }
            )

        return validation


class MultiAgentCoordinationBenchmark(PerformanceBenchmark):
    """Benchmark multi-agent coordination efficiency."""

    def __init__(self):
        super().__init__(
            "multi_agent_coordination",
            "Measures coordination efficiency and scaling behavior",
        )

    def run(self) -> Dict[str, Any]:
        """Run multi-agent coordination benchmark."""
        logger.info("Running multi-agent coordination benchmark...")

        # Test with 10 agents (reasonable for CI)
        num_agents = 10
        operations_per_agent = 5

        # Create agents
        agents = [
            BasicExplorerAgent(
                f"coord-agent-{i}", f"Coordination Agent {i}", grid_size=5
            )
            for i in range(num_agents)
        ]

        for agent in agents:
            agent.start()

        # Measure baseline (single agent performance)
        single_agent = agents[0]
        observation = {
            "position": [2, 2],
            "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        }

        single_times = []
        for _ in range(operations_per_agent):
            start_time = time.time()
            single_agent.perceive(observation)
            if single_agent._should_update_beliefs():
                single_agent.update_beliefs()
            single_agent.select_action()
            single_times.append(time.time() - start_time)

        single_agent_avg = np.mean(single_times)

        # Measure multi-agent coordination
        from concurrent.futures import ThreadPoolExecutor

        def agent_worker(agent):
            times = []
            for _ in range(operations_per_agent):
                start_time = time.time()
                agent.perceive(observation)
                if agent._should_update_beliefs():
                    agent.update_beliefs()
                agent.select_action()
                times.append(time.time() - start_time)
            return times

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [
                executor.submit(agent_worker, agent) for agent in agents
            ]
            all_results = [future.result() for future in futures]
        total_time = time.time() - start_time

        # Calculate coordination efficiency
        all_times = [t for agent_times in all_results for t in agent_times]
        total_operations = len(all_times)

        # Theoretical time if perfectly parallel
        theoretical_time = single_agent_avg * operations_per_agent
        coordination_efficiency = (theoretical_time / total_time) * 100

        # Cleanup agents
        for agent in agents:
            agent.stop()

        result = {
            "num_agents": num_agents,
            "operations_per_agent": operations_per_agent,
            "total_operations": total_operations,
            "single_agent_avg_ms": single_agent_avg * 1000,
            "coordination_total_time_s": total_time,
            "coordination_efficiency_percent": coordination_efficiency,
            "avg_operation_time_ms": np.mean(all_times) * 1000,
            "throughput_ops_per_sec": total_operations / total_time,
            "theoretical_max_throughput": num_agents / single_agent_avg,
            "scaling_factor": (total_operations / total_time)
            / (1 / single_agent_avg),
        }

        logger.info(
            f"Multi-agent coordination: {coordination_efficiency:.1f}% efficiency "
            f"with {num_agents} agents"
        )

        return result

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate multi-agent coordination results."""
        validation = {
            "benchmark": self.name,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "metrics": result,
        }

        # Check coordination efficiency regression
        baseline_efficiency = PERFORMANCE_BASELINES[
            "coordination_efficiency_percent"
        ]
        actual_efficiency = result["coordination_efficiency_percent"]
        efficiency_regression = (
            (baseline_efficiency - actual_efficiency) / baseline_efficiency
        ) * 100

        if efficiency_regression > REGRESSION_THRESHOLDS["critical"]:
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "coordination_efficiency",
                    "regression_percent": efficiency_regression,
                    "severity": "critical",
                    "baseline": baseline_efficiency,
                    "actual": actual_efficiency,
                    "threshold": REGRESSION_THRESHOLDS["critical"],
                }
            )
        elif efficiency_regression > REGRESSION_THRESHOLDS["warning"]:
            validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "coordination_efficiency",
                    "regression_percent": efficiency_regression,
                    "severity": "warning",
                    "baseline": baseline_efficiency,
                    "actual": actual_efficiency,
                    "threshold": REGRESSION_THRESHOLDS["warning"],
                }
            )
        elif efficiency_regression < -REGRESSION_THRESHOLDS["minor"]:
            validation["improvements"].append(
                {
                    "metric": "coordination_efficiency",
                    "improvement_percent": -efficiency_regression,
                    "baseline": baseline_efficiency,
                    "actual": actual_efficiency,
                }
            )

        return validation


class CachePerformanceBenchmark(PerformanceBenchmark):
    """Benchmark matrix caching performance."""

    def __init__(self):
        super().__init__(
            "cache_performance",
            "Measures matrix caching effectiveness and memory overhead",
        )

    def run(self) -> Dict[str, Any]:
        """Run cache performance benchmark."""
        logger.info("Running cache performance benchmark...")

        # Create agent with caching enabled
        agent = BasicExplorerAgent("cache-agent", "Cache Agent", grid_size=10)
        agent.start()

        # Measure cache performance over multiple operations
        num_operations = 50
        cache_times = []

        # Create repeated observation to test cache effectiveness
        observation = {
            "position": [3, 3],
            "surroundings": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        }

        for i in range(num_operations):
            start_time = time.time()
            agent.perceive(observation)
            if agent._should_update_beliefs():
                agent.update_beliefs()
            agent.select_action()
            cache_times.append((time.time() - start_time) * 1000)  # ms

        # Test without caching (restart agent)
        agent.stop()

        # For comparison, create agent without caching optimizations
        agent_no_cache = BasicExplorerAgent(
            "no-cache-agent", "No Cache Agent", grid_size=10
        )
        agent_no_cache.start()

        no_cache_times = []
        for i in range(num_operations):
            # Use different observations to prevent any caching
            varied_observation = {
                "position": [i % 5, i % 5],
                "surroundings": np.array(
                    [[i % 2, 0, 1], [0, 1, 0], [1, 0, i % 2]]
                ),
            }

            start_time = time.time()
            agent_no_cache.perceive(varied_observation)
            if agent_no_cache._should_update_beliefs():
                agent_no_cache.update_beliefs()
            agent_no_cache.select_action()
            no_cache_times.append((time.time() - start_time) * 1000)  # ms

        agent_no_cache.stop()

        # Calculate cache effectiveness
        cache_avg = np.mean(cache_times)
        no_cache_avg = np.mean(no_cache_times)

        # Estimate cache hit rate based on performance improvement
        # This is a simplified estimation - actual implementation would need
        # instrumentation in the caching layer
        speedup_factor = no_cache_avg / cache_avg if cache_avg > 0 else 1.0
        estimated_hit_rate = max(0, min(100, (speedup_factor - 1) * 100))

        result = {
            "num_operations": num_operations,
            "cache_avg_time_ms": cache_avg,
            "no_cache_avg_time_ms": no_cache_avg,
            "speedup_factor": speedup_factor,
            "estimated_hit_rate_percent": estimated_hit_rate,
            "cache_times": cache_times,
            "no_cache_times": no_cache_times,
            "cache_std_ms": np.std(cache_times),
            "no_cache_std_ms": np.std(no_cache_times),
        }

        logger.info(
            f"Cache performance: {speedup_factor:.2f}x speedup, "
            f"~{estimated_hit_rate:.1f}% estimated hit rate"
        )

        return result

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cache performance results."""
        validation = {
            "benchmark": self.name,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "metrics": result,
        }

        # Check cache hit rate regression
        baseline_hit_rate = PERFORMANCE_BASELINES["cache_hit_rate_percent"]
        actual_hit_rate = result["estimated_hit_rate_percent"]
        hit_rate_regression = (
            (baseline_hit_rate - actual_hit_rate) / baseline_hit_rate
        ) * 100

        if hit_rate_regression > REGRESSION_THRESHOLDS["critical"]:
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "cache_hit_rate",
                    "regression_percent": hit_rate_regression,
                    "severity": "critical",
                    "baseline": baseline_hit_rate,
                    "actual": actual_hit_rate,
                    "threshold": REGRESSION_THRESHOLDS["critical"],
                }
            )
        elif hit_rate_regression > REGRESSION_THRESHOLDS["warning"]:
            validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "cache_hit_rate",
                    "regression_percent": hit_rate_regression,
                    "severity": "warning",
                    "baseline": baseline_hit_rate,
                    "actual": actual_hit_rate,
                    "threshold": REGRESSION_THRESHOLDS["warning"],
                }
            )
        elif hit_rate_regression < -REGRESSION_THRESHOLDS["minor"]:
            validation["improvements"].append(
                {
                    "metric": "cache_hit_rate",
                    "improvement_percent": -hit_rate_regression,
                    "baseline": baseline_hit_rate,
                    "actual": actual_hit_rate,
                }
            )

        return validation


class MemoryRegressionBenchmark(PerformanceBenchmark):
    """Benchmark memory usage and detect memory leaks."""

    def __init__(self):
        super().__init__(
            "memory_regression",
            "Detects memory leaks and excessive memory usage",
        )

    def run(self) -> Dict[str, Any]:
        """Run memory regression benchmark."""
        logger.info("Running memory regression benchmark...")

        process = psutil.Process()

        # Baseline memory measurement
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Create and destroy agents multiple times to test for leaks
        num_cycles = 5
        agents_per_cycle = 3
        memory_samples = [baseline_memory]

        for cycle in range(num_cycles):
            # Create agents
            agents = [
                BasicExplorerAgent(
                    f"leak-test-{cycle}-{i}",
                    f"Leak Test {cycle}-{i}",
                    grid_size=5,
                )
                for i in range(agents_per_cycle)
            ]

            # Start agents and run operations
            for agent in agents:
                agent.start()

            # Run some operations
            for _ in range(10):
                observation = {
                    "position": [2, 2],
                    "surroundings": np.array(
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    ),
                }

                for agent in agents:
                    agent.perceive(observation)
                    if agent._should_update_beliefs():
                        agent.update_beliefs()
                    agent.select_action()

            # Stop agents
            for agent in agents:
                agent.stop()

            # Measure memory after cleanup
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append(current_memory)

            # Brief pause to allow garbage collection
            time.sleep(0.1)

        # Final memory measurement
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Calculate memory leak indicators
        memory_growth = final_memory - baseline_memory
        max_memory = max(memory_samples)
        memory_variance = np.var(memory_samples)

        # Check for consistent memory growth (leak indicator)
        if len(memory_samples) >= 3:
            memory_slope = np.polyfit(
                range(len(memory_samples)), memory_samples, 1
            )[0]
        else:
            memory_slope = 0

        result = {
            "baseline_memory_mb": baseline_memory,
            "final_memory_mb": final_memory,
            "max_memory_mb": max_memory,
            "memory_growth_mb": memory_growth,
            "memory_slope_mb_per_cycle": memory_slope,
            "memory_variance": memory_variance,
            "memory_samples": memory_samples,
            "num_cycles": num_cycles,
            "agents_per_cycle": agents_per_cycle,
            "total_agents_tested": num_cycles * agents_per_cycle,
        }

        logger.info(
            f"Memory regression: {memory_growth:.2f}MB growth, "
            f"{memory_slope:.4f}MB/cycle slope"
        )

        return result

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory regression results."""
        validation = {
            "benchmark": self.name,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "metrics": result,
        }

        # Check for memory leaks (consistent growth)
        memory_slope = result["memory_slope_mb_per_cycle"]
        if memory_slope > 1.0:  # >1MB growth per cycle indicates leak
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "memory_leak",
                    "regression_percent": memory_slope
                    * 100,  # Convert to percentage-like metric
                    "severity": "critical",
                    "baseline": 0,
                    "actual": memory_slope,
                    "threshold": 1.0,
                }
            )
        elif memory_slope > 0.5:  # >0.5MB growth per cycle is warning
            validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "memory_leak",
                    "regression_percent": memory_slope * 100,
                    "severity": "warning",
                    "baseline": 0,
                    "actual": memory_slope,
                    "threshold": 0.5,
                }
            )

        # Check for excessive memory growth
        memory_growth = result["memory_growth_mb"]
        if memory_growth > 50:  # >50MB growth is critical
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "memory_growth",
                    "regression_percent": memory_growth,
                    "severity": "critical",
                    "baseline": 0,
                    "actual": memory_growth,
                    "threshold": 50,
                }
            )
        elif memory_growth > 20:  # >20MB growth is warning
            if validation["status"] == "pass":
                validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": "memory_growth",
                    "regression_percent": memory_growth,
                    "severity": "warning",
                    "baseline": 0,
                    "actual": memory_growth,
                    "threshold": 20,
                }
            )

        return validation


class CIPerformanceBenchmarkSuite:
    """Main CI performance benchmark suite."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("tests/performance/ci_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benchmarks = [
            SingleAgentInferenceBenchmark(),
            MultiAgentCoordinationBenchmark(),
            CachePerformanceBenchmark(),
            MemoryRegressionBenchmark(),
        ]

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return consolidated results."""
        logger.info("Starting CI performance benchmark suite...")

        suite_start = time.time()
        results = {
            "suite_info": {
                "timestamp": datetime.now().isoformat(),
                "python_version": os.sys.version,
                "platform": os.uname().sysname,
                "cpu_count": os.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            },
            "benchmarks": {},
            "overall_status": "pass",
            "summary": {
                "total_benchmarks": len(self.benchmarks),
                "passed": 0,
                "warnings": 0,
                "failed": 0,
                "total_regressions": 0,
                "total_improvements": 0,
            },
        }

        for benchmark in self.benchmarks:
            logger.info(f"Running {benchmark.name} benchmark...")

            try:
                # Run benchmark
                bench_start = time.time()
                bench_result = benchmark.run()
                bench_duration = time.time() - bench_start

                # Validate results
                validation = benchmark.validate_result(bench_result)

                # Store results
                results["benchmarks"][benchmark.name] = {
                    "description": benchmark.description,
                    "duration_seconds": bench_duration,
                    "result": bench_result,
                    "validation": validation,
                }

                # Update summary
                if validation["status"] == "pass":
                    results["summary"]["passed"] += 1
                elif validation["status"] == "warning":
                    results["summary"]["warnings"] += 1
                    if results["overall_status"] == "pass":
                        results["overall_status"] = "warning"
                else:  # fail
                    results["summary"]["failed"] += 1
                    results["overall_status"] = "fail"

                results["summary"]["total_regressions"] += len(
                    validation["regressions"]
                )
                results["summary"]["total_improvements"] += len(
                    validation["improvements"]
                )

                logger.info(
                    f"Completed {benchmark.name}: {validation['status']}"
                )

            except Exception as e:
                logger.error(f"Error running {benchmark.name}: {e}")
                results["benchmarks"][benchmark.name] = {
                    "description": benchmark.description,
                    "error": str(e),
                    "validation": {
                        "status": "error",
                        "regressions": [],
                        "improvements": [],
                    },
                }
                results["summary"]["failed"] += 1
                results["overall_status"] = "fail"

        suite_duration = time.time() - suite_start
        results["suite_info"]["duration_seconds"] = suite_duration

        # Save results
        self._save_results(results)

        logger.info(
            f"Benchmark suite completed in {suite_duration:.2f}s: "
            f"{results['overall_status']}"
        )

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.output_dir / f"ci_benchmark_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save latest results (for CI parsing)
        latest_path = self.output_dir / "latest_results.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2)

        # Generate summary report
        self._generate_summary_report(results, timestamp)

    def _generate_summary_report(
        self, results: Dict[str, Any], timestamp: str
    ):
        """Generate human-readable summary report."""
        report_path = self.output_dir / f"ci_benchmark_report_{timestamp}.md"

        with open(report_path, "w") as f:
            f.write("# CI Performance Benchmark Report\n\n")
            f.write(f"**Generated**: {results['suite_info']['timestamp']}\n")
            f.write(
                f"**Duration**: {results['suite_info']['duration_seconds']:.2f}s\n"
            )
            f.write(f"**Status**: {results['overall_status'].upper()}\n\n")

            # Summary
            summary = results["summary"]
            f.write("## Summary\n\n")
            f.write(f"- **Total Benchmarks**: {summary['total_benchmarks']}\n")
            f.write(f"- **Passed**: {summary['passed']}\n")
            f.write(f"- **Warnings**: {summary['warnings']}\n")
            f.write(f"- **Failed**: {summary['failed']}\n")
            f.write(
                f"- **Total Regressions**: {summary['total_regressions']}\n"
            )
            f.write(
                f"- **Total Improvements**: {summary['total_improvements']}\n\n"
            )

            # Benchmark details
            f.write("## Benchmark Results\n\n")

            for bench_name, bench_data in results["benchmarks"].items():
                validation = bench_data.get("validation", {})
                status = validation.get("status", "unknown")

                f.write(f"### {bench_name}\n\n")
                f.write(
                    f"**Description**: {bench_data.get('description', 'N/A')}\n"
                )
                f.write(f"**Status**: {status.upper()}\n")
                f.write(
                    f"**Duration**: {bench_data.get('duration_seconds', 0):.2f}s\n\n"
                )

                # Regressions
                regressions = validation.get("regressions", [])
                if regressions:
                    f.write("**Regressions**:\n")
                    for reg in regressions:
                        f.write(
                            f"- {reg['metric']}: {reg['regression_percent']:.1f}% "
                            f"({reg['severity']}) - {reg['actual']:.2f} vs {reg['baseline']:.2f}\n"
                        )
                    f.write("\n")

                # Improvements
                improvements = validation.get("improvements", [])
                if improvements:
                    f.write("**Improvements**:\n")
                    for imp in improvements:
                        f.write(
                            f"- {imp['metric']}: {imp['improvement_percent']:.1f}% improvement "
                            f"- {imp['actual']:.2f} vs {imp['baseline']:.2f}\n"
                        )
                    f.write("\n")

                # Key metrics
                result = bench_data.get("result", {})
                if result:
                    f.write("**Key Metrics**:\n")
                    for key, value in result.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            f.write(f"- {key}: {value:.2f}\n")
                    f.write("\n")

        logger.info(f"Summary report saved to: {report_path}")


def main():
    """Run the CI performance benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CI performance benchmarks"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run benchmark suite
    suite = CIPerformanceBenchmarkSuite(args.output_dir)
    results = suite.run_all_benchmarks()

    # Exit with appropriate code for CI
    if results["overall_status"] == "fail":
        exit(1)
    elif results["overall_status"] == "warning":
        exit(2)  # Warning exit code
    else:
        exit(0)  # Success


if __name__ == "__main__":
    main()
