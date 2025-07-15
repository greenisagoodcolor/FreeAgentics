#!/usr/bin/env python3
"""
Enhanced CI Performance Benchmarks
Comprehensive benchmark suite for continuous integration with detailed performance validation.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Simplified metrics collection for enhanced benchmarks
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

from agents.base_agent import BasicExplorerAgent


class MetricSource(Enum):
    SYSTEM = "system"
    AGENT = "agent"
    DATABASE = "database"


class MetricType(Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"


def record_metric(
    name: str, value: float, source: MetricSource, type: MetricType, tags: Optional[Dict] = None
):
    """Simple metric recording for benchmarks."""
    pass  # For now, just pass - this would integrate with real metrics system


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced performance baselines based on comprehensive analysis
PERFORMANCE_BASELINES = {
    # Single Agent Performance
    "single_agent_inference_ms": 50.0,
    "single_agent_p95_ms": 85.0,
    "single_agent_p99_ms": 120.0,
    "single_agent_memory_mb": 34.5,
    "single_agent_throughput_ops_sec": 20.0,
    # Multi-Agent Coordination
    "coordination_efficiency_10_agents": 70.0,
    "coordination_efficiency_50_agents": 28.4,
    "coordination_overhead_percent": 72.0,
    # Cache Performance
    "cache_hit_rate_percent": 22.1,
    "cache_memory_overhead_percent": 23.0,
    "cache_speedup_factor": 3.2,
    # Memory Management
    "memory_growth_per_cycle_mb": 0.5,
    "memory_leak_threshold_mb": 1.0,
    "gc_frequency_per_minute": 5.0,
    # System Resources
    "cpu_utilization_percent": 40.0,
    "memory_utilization_percent": 60.0,
    "disk_io_mb_per_sec": 10.0,
}

# Regression detection thresholds
REGRESSION_THRESHOLDS = {
    "critical": 25.0,  # >25% regression triggers CI failure
    "warning": 10.0,  # >10% regression triggers warning
    "minor": 5.0,  # >5% regression logged
}

# Performance targets for production readiness
PRODUCTION_TARGETS = {
    "max_agents_realtime": 20,  # Maximum agents for real-time operation
    "max_agents_batch": 50,  # Maximum agents for batch processing
    "target_inference_ms": 10.0,  # Target for real-time inference
    "target_memory_mb": 30.0,  # Target memory per agent
    "target_efficiency_percent": 80.0,  # Target coordination efficiency
}


class EnhancedPerformanceBenchmark:
    """Enhanced base class for performance benchmarks with detailed metrics."""

    def __init__(self, name: str, description: str, category: str = "general"):
        self.name = name
        self.description = description
        self.category = category
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def start_benchmark(self):
        """Start benchmark timing and resource monitoring."""
        self.start_time = time.time()
        self.metrics["start_timestamp"] = datetime.now().isoformat()

        # Record initial system state
        process = psutil.Process()
        self.metrics["initial_cpu_percent"] = process.cpu_percent()
        self.metrics["initial_memory_mb"] = process.memory_info().rss / (1024 * 1024)

    def end_benchmark(self):
        """End benchmark timing and calculate final metrics."""
        self.end_time = time.time()
        self.metrics["end_timestamp"] = datetime.now().isoformat()
        self.metrics["duration_seconds"] = self.end_time - self.start_time

        # Record final system state
        process = psutil.Process()
        self.metrics["final_cpu_percent"] = process.cpu_percent()
        self.metrics["final_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        self.metrics["memory_delta_mb"] = (
            self.metrics["final_memory_mb"] - self.metrics["initial_memory_mb"]
        )

    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric to both internal storage and unified collector."""
        self.metrics[name] = value

        # Record to unified collector for real-time monitoring
        record_metric(
            name=f"{self.name}_{name}",
            value=value,
            source=MetricSource.SYSTEM,
            type=MetricType.GAUGE,
            tags=tags or {"benchmark": self.name, "category": self.category},
        )

    def run(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        raise NotImplementedError

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate results against baselines and return analysis."""
        raise NotImplementedError


class ComprehensiveSingleAgentBenchmark(EnhancedPerformanceBenchmark):
    """Comprehensive single agent performance benchmark."""

    def __init__(self):
        super().__init__(
            "comprehensive_single_agent", "Comprehensive single agent performance analysis", "agent"
        )

    def run(self) -> Dict[str, Any]:
        """Run comprehensive single agent benchmark."""
        logger.info("Running comprehensive single agent benchmark...")
        self.start_benchmark()

        # Create test agent
        agent = BasicExplorerAgent("benchmark-agent", "Benchmark Agent", grid_size=10)
        agent.start()

        # Measure baseline system state
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)

        # Comprehensive inference testing
        results = self._run_inference_tests(agent, baseline_memory)

        # Memory analysis
        memory_results = self._analyze_memory_usage(agent, baseline_memory)
        results.update(memory_results)

        # Throughput analysis
        throughput_results = self._analyze_throughput(agent)
        results.update(throughput_results)

        # Cleanup
        agent.stop()

        self.end_benchmark()
        results.update(self.metrics)

        return results

    def _run_inference_tests(self, agent, baseline_memory: float) -> Dict[str, Any]:
        """Run comprehensive inference performance tests."""
        # Test different observation patterns
        test_scenarios = [
            {
                "name": "simple",
                "observation": {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                },
                "iterations": 100,
            },
            {
                "name": "complex",
                "observation": {
                    "position": [5, 5],
                    "surroundings": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
                },
                "iterations": 50,
            },
            {
                "name": "varied",
                "observation": None,  # Will be varied per iteration
                "iterations": 75,
            },
        ]

        all_times = []
        scenario_results = {}

        for scenario in test_scenarios:
            times = []

            for i in range(scenario["iterations"]):
                if scenario["name"] == "varied":
                    # Create varied observation
                    obs = {
                        "position": [i % 10, (i * 2) % 10],
                        "surroundings": np.random.randint(0, 2, size=(3, 3)),
                    }
                else:
                    obs = scenario["observation"]

                start_time = time.time()

                # Full inference cycle
                agent.perceive(obs)
                if agent._should_update_beliefs():
                    agent.update_beliefs()
                agent.select_action()

                duration = (time.time() - start_time) * 1000  # ms
                times.append(duration)
                all_times.append(duration)

                # Brief pause to prevent system overload
                time.sleep(0.001)

            # Calculate scenario statistics
            scenario_results[f"{scenario['name']}_avg_ms"] = np.mean(times)
            scenario_results[f"{scenario['name']}_median_ms"] = np.median(times)
            scenario_results[f"{scenario['name']}_std_ms"] = np.std(times)
            scenario_results[f"{scenario['name']}_min_ms"] = np.min(times)
            scenario_results[f"{scenario['name']}_max_ms"] = np.max(times)

            # Record metrics
            self.record_metric(f"{scenario['name']}_avg_ms", np.mean(times))

        # Overall statistics
        return {
            "total_operations": len(all_times),
            "avg_inference_time_ms": np.mean(all_times),
            "median_inference_time_ms": np.median(all_times),
            "p95_inference_time_ms": np.percentile(all_times, 95),
            "p99_inference_time_ms": np.percentile(all_times, 99),
            "std_inference_time_ms": np.std(all_times),
            "min_inference_time_ms": np.min(all_times),
            "max_inference_time_ms": np.max(all_times),
            **scenario_results,
        }

    def _analyze_memory_usage(self, agent, baseline_memory: float) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        process = psutil.Process()

        # Memory measurements at different stages
        current_memory = process.memory_info().rss / (1024 * 1024)
        memory_per_agent = current_memory - baseline_memory

        # Memory growth analysis
        memory_samples = []
        for i in range(10):
            # Perform operations that might cause memory growth
            obs = {
                "position": [i % 5, i % 5],
                "surroundings": np.random.randint(0, 2, size=(3, 3)),
            }

            for _ in range(10):
                agent.perceive(obs)
                if agent._should_update_beliefs():
                    agent.update_beliefs()
                agent.select_action()

            current_mem = process.memory_info().rss / (1024 * 1024)
            memory_samples.append(current_mem)
            time.sleep(0.05)  # Allow GC

        # Analyze memory growth trend
        if len(memory_samples) > 2:
            memory_slope = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        else:
            memory_slope = 0

        return {
            "memory_per_agent_mb": memory_per_agent,
            "memory_growth_slope_mb": memory_slope,
            "memory_variance": np.var(memory_samples),
            "peak_memory_mb": max(memory_samples),
            "memory_samples": memory_samples,
        }

    def _analyze_throughput(self, agent) -> Dict[str, Any]:
        """Analyze agent throughput under different conditions."""
        # Sustained throughput test
        duration = 5.0  # seconds
        start_time = time.time()
        operations = 0

        while time.time() - start_time < duration:
            obs = {
                "position": [2, 2],
                "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            }

            agent.perceive(obs)
            if agent._should_update_beliefs():
                agent.update_beliefs()
            agent.select_action()

            operations += 1

        actual_duration = time.time() - start_time
        throughput = operations / actual_duration

        return {
            "sustained_operations": operations,
            "sustained_duration_s": actual_duration,
            "sustained_throughput_ops_sec": throughput,
        }

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single agent results against baselines."""
        validation = {
            "benchmark": self.name,
            "category": self.category,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "warnings": [],
            "metrics": results,
        }

        # Performance validations
        validations = [
            ("avg_inference_time_ms", "single_agent_inference_ms", "lower_is_better"),
            ("p95_inference_time_ms", "single_agent_p95_ms", "lower_is_better"),
            ("p99_inference_time_ms", "single_agent_p99_ms", "lower_is_better"),
            ("memory_per_agent_mb", "single_agent_memory_mb", "lower_is_better"),
            ("sustained_throughput_ops_sec", "single_agent_throughput_ops_sec", "higher_is_better"),
        ]

        for metric_name, baseline_key, direction in validations:
            if metric_name in results:
                self._validate_metric(
                    validation,
                    metric_name,
                    results[metric_name],
                    PERFORMANCE_BASELINES[baseline_key],
                    direction,
                )

        # Memory growth validation
        if "memory_growth_slope_mb" in results:
            slope = results["memory_growth_slope_mb"]
            if slope > PERFORMANCE_BASELINES["memory_leak_threshold_mb"]:
                validation["status"] = "fail"
                validation["regressions"].append(
                    {
                        "metric": "memory_growth",
                        "severity": "critical",
                        "actual": slope,
                        "threshold": PERFORMANCE_BASELINES["memory_leak_threshold_mb"],
                        "message": "Potential memory leak detected",
                    }
                )

        return validation

    def _validate_metric(
        self, validation: Dict, metric_name: str, actual: float, baseline: float, direction: str
    ):
        """Validate a single metric against its baseline."""
        if direction == "lower_is_better":
            regression_percent = ((actual - baseline) / baseline) * 100
        else:  # higher_is_better
            regression_percent = ((baseline - actual) / baseline) * 100

        if regression_percent > REGRESSION_THRESHOLDS["critical"]:
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": metric_name,
                    "regression_percent": regression_percent,
                    "severity": "critical",
                    "baseline": baseline,
                    "actual": actual,
                    "threshold": REGRESSION_THRESHOLDS["critical"],
                }
            )
        elif regression_percent > REGRESSION_THRESHOLDS["warning"]:
            if validation["status"] == "pass":
                validation["status"] = "warning"
            validation["regressions"].append(
                {
                    "metric": metric_name,
                    "regression_percent": regression_percent,
                    "severity": "warning",
                    "baseline": baseline,
                    "actual": actual,
                    "threshold": REGRESSION_THRESHOLDS["warning"],
                }
            )
        elif regression_percent < -REGRESSION_THRESHOLDS["minor"]:
            validation["improvements"].append(
                {
                    "metric": metric_name,
                    "improvement_percent": -regression_percent,
                    "baseline": baseline,
                    "actual": actual,
                }
            )


class ScalabilityAnalysisBenchmark(EnhancedPerformanceBenchmark):
    """Comprehensive scalability analysis benchmark."""

    def __init__(self):
        super().__init__(
            "scalability_analysis", "Comprehensive multi-agent scalability analysis", "coordination"
        )

    def run(self) -> Dict[str, Any]:
        """Run comprehensive scalability analysis."""
        logger.info("Running scalability analysis benchmark...")
        self.start_benchmark()

        # Test different agent counts
        agent_counts = [1, 2, 5, 10, 15, 20]
        if os.environ.get("FULL_SCALABILITY_TEST"):
            agent_counts.extend([30, 50])

        results = {
            "agent_counts_tested": agent_counts,
            "scalability_data": [],
            "efficiency_curve": [],
            "throughput_curve": [],
            "memory_scaling": [],
        }

        for agent_count in agent_counts:
            logger.info(f"Testing {agent_count} agents...")

            # Run coordination test
            coordination_result = self._test_coordination(agent_count)
            results["scalability_data"].append(coordination_result)

            # Extract key metrics
            efficiency = coordination_result["coordination_efficiency_percent"]
            throughput = coordination_result["total_throughput_ops_sec"]
            memory_total = coordination_result["total_memory_mb"]

            results["efficiency_curve"].append(
                {
                    "agents": agent_count,
                    "efficiency": efficiency,
                }
            )
            results["throughput_curve"].append(
                {
                    "agents": agent_count,
                    "throughput": throughput,
                }
            )
            results["memory_scaling"].append(
                {
                    "agents": agent_count,
                    "total_memory": memory_total,
                    "memory_per_agent": memory_total / agent_count,
                }
            )

            # Record metrics
            self.record_metric(f"efficiency_{agent_count}_agents", efficiency)
            self.record_metric(f"throughput_{agent_count}_agents", throughput)
            self.record_metric(f"memory_total_{agent_count}_agents", memory_total)

        # Calculate scalability metrics
        scalability_metrics = self._calculate_scalability_metrics(results)
        results.update(scalability_metrics)

        self.end_benchmark()
        results.update(self.metrics)

        return results

    def _test_coordination(self, num_agents: int) -> Dict[str, Any]:
        """Test coordination performance with specified number of agents."""
        # Create agents
        agents = [
            BasicExplorerAgent(f"scale-agent-{i}", f"Scale Agent {i}", grid_size=5)
            for i in range(num_agents)
        ]

        # Start agents
        for agent in agents:
            agent.start()

        # Measure baseline performance (single agent)
        if num_agents > 0:
            baseline_agent = agents[0]
            baseline_times = []

            for _ in range(10):
                obs = {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                }

                start_time = time.time()
                baseline_agent.perceive(obs)
                if baseline_agent._should_update_beliefs():
                    baseline_agent.update_beliefs()
                baseline_agent.select_action()
                baseline_times.append(time.time() - start_time)

            baseline_avg = np.mean(baseline_times)
        else:
            baseline_avg = 0.05  # Default baseline

        # Test coordination
        operations_per_agent = 10

        def agent_worker(agent):
            times = []
            for i in range(operations_per_agent):
                obs = {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                }

                start_time = time.time()
                agent.perceive(obs)
                if agent._should_update_beliefs():
                    agent.update_beliefs()
                agent.select_action()
                times.append(time.time() - start_time)

            return times

        # Measure coordination performance
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        coordination_start = time.time()

        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [executor.submit(agent_worker, agent) for agent in agents]
            all_results = [future.result() for future in futures]

        coordination_duration = time.time() - coordination_start
        final_memory = process.memory_info().rss / (1024 * 1024)

        # Calculate metrics
        all_times = [t for agent_times in all_results for t in agent_times]
        total_operations = len(all_times)

        # Theoretical optimal time (perfect parallelism)
        theoretical_time = baseline_avg * operations_per_agent
        coordination_efficiency = (theoretical_time / coordination_duration) * 100

        # Throughput calculations
        total_throughput = total_operations / coordination_duration
        theoretical_throughput = num_agents / baseline_avg

        # Cleanup
        for agent in agents:
            agent.stop()

        return {
            "num_agents": num_agents,
            "operations_per_agent": operations_per_agent,
            "total_operations": total_operations,
            "coordination_duration_s": coordination_duration,
            "coordination_efficiency_percent": coordination_efficiency,
            "total_throughput_ops_sec": total_throughput,
            "theoretical_throughput_ops_sec": theoretical_throughput,
            "scaling_factor": total_throughput / (1 / baseline_avg),
            "avg_operation_time_ms": np.mean(all_times) * 1000,
            "baseline_time_ms": baseline_avg * 1000,
            "total_memory_mb": final_memory - initial_memory,
            "overhead_percent": 100 - coordination_efficiency,
        }

    def _calculate_scalability_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall scalability metrics."""
        efficiency_data = results["efficiency_curve"]
        throughput_data = results["throughput_curve"]
        memory_data = results["memory_scaling"]

        # Find efficiency at key points
        efficiency_10 = next((d["efficiency"] for d in efficiency_data if d["agents"] == 10), 0)
        efficiency_20 = next((d["efficiency"] for d in efficiency_data if d["agents"] == 20), 0)

        # Calculate efficiency degradation
        if len(efficiency_data) > 1:
            efficiency_slope = np.polyfit(
                [d["agents"] for d in efficiency_data],
                [d["efficiency"] for d in efficiency_data],
                1,
            )[0]
        else:
            efficiency_slope = 0

        # Memory scaling analysis
        memory_per_agent_avg = np.mean([d["memory_per_agent"] for d in memory_data])
        memory_variance = np.var([d["memory_per_agent"] for d in memory_data])

        # Throughput scaling analysis
        throughput_scaling = []
        for i in range(1, len(throughput_data)):
            prev_throughput = throughput_data[i - 1]["throughput"]
            curr_throughput = throughput_data[i]["throughput"]
            prev_agents = throughput_data[i - 1]["agents"]
            curr_agents = throughput_data[i]["agents"]

            if prev_throughput > 0:
                scaling_factor = (curr_throughput / prev_throughput) / (curr_agents / prev_agents)
                throughput_scaling.append(scaling_factor)

        avg_scaling_factor = np.mean(throughput_scaling) if throughput_scaling else 1.0

        return {
            "efficiency_at_10_agents": efficiency_10,
            "efficiency_at_20_agents": efficiency_20,
            "efficiency_degradation_per_agent": -efficiency_slope,
            "memory_per_agent_avg_mb": memory_per_agent_avg,
            "memory_scaling_variance": memory_variance,
            "avg_throughput_scaling_factor": avg_scaling_factor,
            "max_efficient_agents": self._find_max_efficient_agents(efficiency_data),
            "scalability_rating": self._calculate_scalability_rating(efficiency_data),
        }

    def _find_max_efficient_agents(self, efficiency_data: List[Dict]) -> int:
        """Find maximum number of agents with acceptable efficiency."""
        for data in sorted(efficiency_data, key=lambda x: x["agents"], reverse=True):
            if data["efficiency"] >= 50:  # 50% efficiency threshold
                return data["agents"]
        return 1

    def _calculate_scalability_rating(self, efficiency_data: List[Dict]) -> str:
        """Calculate overall scalability rating."""
        if not efficiency_data:
            return "unknown"

        max_agents = max(d["agents"] for d in efficiency_data)
        max_efficiency = max(d["efficiency"] for d in efficiency_data)

        if max_agents >= 20 and max_efficiency >= 70:
            return "excellent"
        elif max_agents >= 15 and max_efficiency >= 60:
            return "good"
        elif max_agents >= 10 and max_efficiency >= 50:
            return "moderate"
        else:
            return "poor"

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scalability results against targets."""
        validation = {
            "benchmark": self.name,
            "category": self.category,
            "timestamp": datetime.now().isoformat(),
            "status": "pass",
            "regressions": [],
            "improvements": [],
            "warnings": [],
            "metrics": results,
        }

        # Check efficiency at key points
        if "efficiency_at_10_agents" in results:
            efficiency_10 = results["efficiency_at_10_agents"]
            baseline_10 = PERFORMANCE_BASELINES["coordination_efficiency_10_agents"]

            if efficiency_10 < baseline_10 * 0.75:  # 25% below baseline
                validation["status"] = "fail"
                validation["regressions"].append(
                    {
                        "metric": "efficiency_at_10_agents",
                        "severity": "critical",
                        "actual": efficiency_10,
                        "baseline": baseline_10,
                        "threshold": baseline_10 * 0.75,
                    }
                )

        # Check memory scaling
        if "memory_per_agent_avg_mb" in results:
            memory_avg = results["memory_per_agent_avg_mb"]
            baseline_memory = PERFORMANCE_BASELINES["single_agent_memory_mb"]

            if memory_avg > baseline_memory * 1.2:  # 20% above baseline
                if validation["status"] == "pass":
                    validation["status"] = "warning"
                validation["warnings"].append(
                    {
                        "metric": "memory_per_agent",
                        "actual": memory_avg,
                        "baseline": baseline_memory,
                        "message": "Memory usage above expected baseline",
                    }
                )

        # Check scalability rating
        rating = results.get("scalability_rating", "unknown")
        if rating == "poor":
            validation["status"] = "fail"
            validation["regressions"].append(
                {
                    "metric": "scalability_rating",
                    "severity": "critical",
                    "actual": rating,
                    "baseline": "good",
                    "message": "Poor scalability performance",
                }
            )

        return validation


class EnhancedCIBenchmarkSuite:
    """Enhanced CI benchmark suite with comprehensive analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("tests/performance/enhanced_ci_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benchmarks = [
            ComprehensiveSingleAgentBenchmark(),
            ScalabilityAnalysisBenchmark(),
        ]

    def run_all_benchmarks(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run all enhanced benchmarks."""
        logger.info("Starting enhanced CI performance benchmark suite...")

        if quick_mode:
            logger.info("Running in quick mode for CI")
            os.environ["QUICK_BENCHMARK"] = "1"

        suite_start = time.time()
        results = {
            "suite_info": {
                "timestamp": datetime.now().isoformat(),
                "suite_version": "2.0",
                "quick_mode": quick_mode,
                "platform": {
                    "system": os.uname().sysname,
                    "python_version": os.sys.version,
                    "cpu_count": os.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                },
                "baselines": PERFORMANCE_BASELINES,
                "thresholds": REGRESSION_THRESHOLDS,
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
                "critical_regressions": 0,
                "warning_regressions": 0,
            },
            "performance_analysis": {},
        }

        # Run benchmarks
        for benchmark in self.benchmarks:
            logger.info(f"Running {benchmark.name} benchmark...")

            try:
                # Run benchmark
                bench_start = time.time()
                bench_result = benchmark.run()
                bench_duration = time.time() - bench_start

                # Validate results
                validation = benchmark.validate_results(bench_result)

                # Store results
                results["benchmarks"][benchmark.name] = {
                    "description": benchmark.description,
                    "category": benchmark.category,
                    "duration_seconds": bench_duration,
                    "result": bench_result,
                    "validation": validation,
                }

                # Update summary
                self._update_summary(results, validation)

                logger.info(f"Completed {benchmark.name}: {validation['status']}")

            except Exception as e:
                logger.error(f"Error running {benchmark.name}: {e}")
                results["benchmarks"][benchmark.name] = {
                    "description": benchmark.description,
                    "category": benchmark.category,
                    "error": str(e),
                    "validation": {
                        "status": "error",
                        "regressions": [],
                        "improvements": [],
                        "warnings": [],
                    },
                }
                results["summary"]["failed"] += 1
                results["overall_status"] = "fail"

        # Generate performance analysis
        results["performance_analysis"] = self._generate_performance_analysis(results)

        # Finalize results
        suite_duration = time.time() - suite_start
        results["suite_info"]["duration_seconds"] = suite_duration

        # Save results
        self._save_results(results)

        logger.info(
            f"Enhanced benchmark suite completed in {suite_duration:.2f}s: "
            f"{results['overall_status']}"
        )

        return results

    def _update_summary(self, results: Dict, validation: Dict):
        """Update suite summary with validation results."""
        status = validation["status"]

        if status == "pass":
            results["summary"]["passed"] += 1
        elif status == "warning":
            results["summary"]["warnings"] += 1
            if results["overall_status"] == "pass":
                results["overall_status"] = "warning"
        else:  # fail or error
            results["summary"]["failed"] += 1
            results["overall_status"] = "fail"

        # Count regressions by severity
        for regression in validation.get("regressions", []):
            results["summary"]["total_regressions"] += 1
            if regression["severity"] == "critical":
                results["summary"]["critical_regressions"] += 1
            elif regression["severity"] == "warning":
                results["summary"]["warning_regressions"] += 1

        results["summary"]["total_improvements"] += len(validation.get("improvements", []))

    def _generate_performance_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        analysis = {
            "overall_health": "unknown",
            "critical_issues": [],
            "recommendations": [],
            "performance_trend": "stable",
            "production_readiness": "unknown",
        }

        # Analyze overall health
        if results["overall_status"] == "pass":
            analysis["overall_health"] = "excellent"
        elif results["overall_status"] == "warning":
            analysis["overall_health"] = "good"
        else:
            analysis["overall_health"] = "poor"

        # Identify critical issues
        for bench_name, bench_data in results["benchmarks"].items():
            validation = bench_data.get("validation", {})

            for regression in validation.get("regressions", []):
                if regression.get("severity") == "critical":
                    analysis["critical_issues"].append(
                        {
                            "benchmark": bench_name,
                            "metric": regression.get("metric", "unknown"),
                            "regression": regression.get(
                                "regression_percent", regression.get("actual", 0)
                            ),
                            "impact": "high",
                        }
                    )

        # Generate recommendations
        if analysis["critical_issues"]:
            analysis["recommendations"].append(
                "Address critical performance regressions before production deployment"
            )

        # Assess production readiness
        critical_count = results["summary"]["critical_regressions"]
        if critical_count == 0:
            analysis["production_readiness"] = "ready"
        elif critical_count <= 2:
            analysis["production_readiness"] = "needs_attention"
        else:
            analysis["production_readiness"] = "not_ready"

        return analysis

    def _save_results(self, results: Dict[str, Any]):
        """Save enhanced benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive JSON results
        json_path = self.output_dir / f"enhanced_ci_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save latest results for CI parsing
        latest_path = self.output_dir / "latest_enhanced_results.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2)

        # Generate detailed report
        self._generate_detailed_report(results, timestamp)

    def _generate_detailed_report(self, results: Dict[str, Any], timestamp: str):
        """Generate detailed performance report."""
        report_path = self.output_dir / f"enhanced_performance_report_{timestamp}.md"

        with open(report_path, "w") as f:
            f.write("# Enhanced CI Performance Report\n\n")
            f.write(f"**Generated**: {results['suite_info']['timestamp']}\n")
            f.write(f"**Duration**: {results['suite_info']['duration_seconds']:.2f}s\n")
            f.write(f"**Status**: {results['overall_status'].upper()}\n")
            f.write(f"**Suite Version**: {results['suite_info']['suite_version']}\n\n")

            # Performance analysis
            analysis = results["performance_analysis"]
            f.write("## Performance Analysis\n\n")
            f.write(f"- **Overall Health**: {analysis['overall_health']}\n")
            f.write(f"- **Production Readiness**: {analysis['production_readiness']}\n")
            f.write(f"- **Critical Issues**: {len(analysis['critical_issues'])}\n\n")

            if analysis["critical_issues"]:
                f.write("### Critical Issues\n\n")
                for issue in analysis["critical_issues"]:
                    f.write(
                        f"- **{issue['benchmark']}**: {issue['metric']} "
                        f"({issue['regression']:.1f}% regression)\n"
                    )
                f.write("\n")

            if analysis["recommendations"]:
                f.write("### Recommendations\n\n")
                for rec in analysis["recommendations"]:
                    f.write(f"- {rec}\n")
                f.write("\n")

            # Detailed benchmark results
            f.write("## Benchmark Results\n\n")

            for bench_name, bench_data in results["benchmarks"].items():
                f.write(f"### {bench_name}\n\n")
                f.write(f"**Category**: {bench_data.get('category', 'N/A')}\n")
                f.write(f"**Description**: {bench_data.get('description', 'N/A')}\n")
                f.write(f"**Duration**: {bench_data.get('duration_seconds', 0):.2f}s\n")

                validation = bench_data.get("validation", {})
                f.write(f"**Status**: {validation.get('status', 'unknown').upper()}\n\n")

                # Performance metrics
                result = bench_data.get("result", {})
                if result:
                    f.write("**Key Metrics**:\n")
                    # Show only the most important metrics
                    key_metrics = [
                        "avg_inference_time_ms",
                        "memory_per_agent_mb",
                        "coordination_efficiency_percent",
                        "sustained_throughput_ops_sec",
                        "scalability_rating",
                    ]

                    for metric in key_metrics:
                        if metric in result:
                            value = result[metric]
                            if isinstance(value, (int, float)):
                                f.write(f"- {metric}: {value:.2f}\n")
                            else:
                                f.write(f"- {metric}: {value}\n")
                    f.write("\n")

                # Regressions and improvements
                if validation.get("regressions"):
                    f.write("**Regressions**:\n")
                    for reg in validation["regressions"]:
                        metric = reg.get("metric", "unknown")
                        regression = reg.get("regression_percent", reg.get("actual", 0))
                        severity = reg.get("severity", "unknown")
                        f.write(f"- {metric}: {regression:.1f}% ({severity})\n")
                    f.write("\n")

                if validation.get("improvements"):
                    f.write("**Improvements**:\n")
                    for imp in validation["improvements"]:
                        f.write(
                            f"- {imp['metric']}: {imp['improvement_percent']:.1f}% "
                            f"improvement\n"
                        )
                    f.write("\n")

        logger.info(f"Detailed report saved to: {report_path}")


def main():
    """Run the enhanced CI performance benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced CI performance benchmarks")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode for CI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run enhanced benchmark suite
    suite = EnhancedCIBenchmarkSuite(args.output_dir)
    results = suite.run_all_benchmarks(quick_mode=args.quick)

    # Exit with appropriate code for CI
    if results["overall_status"] == "fail":
        exit(1)
    elif results["overall_status"] == "warning":
        exit(2)  # Warning exit code
    else:
        exit(0)  # Success


if __name__ == "__main__":
    main()
