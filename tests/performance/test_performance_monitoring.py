"""Test suite for performance monitoring system components."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.performance.performance_report_generator import (
    PerformanceMetric,
    PerformanceReportGenerator,
    RegressionAlert,
)
from tests.performance.run_performance_monitoring import PerformanceMonitor


class TestPerformanceReportGenerator:
    """Test the performance report generator functionality."""

    def test_load_benchmark_results(self, tmp_path):
        """Test loading benchmark results from JSON files."""
        # Create test result files
        results_dir = tmp_path / "test_results"
        results_dir.mkdir()

        test_result = {
            "name": "test_benchmark",
            "mean_time_ms": 1.5,
            "memory_usage_mb": 10.0,
            "timestamp": "2025-07-04T12:00:00",
            "additional_metrics": {
                "cache_hit_rate": 0.8,
                "speedup_factor": 2.5,
            },
        }

        # Write test result file
        result_file = (
            results_dir / "test_benchmark_results_20250704_120000.json"
        )
        with open(result_file, "w") as f:
            json.dump([test_result], f)

        generator = PerformanceReportGenerator(str(results_dir))
        results = generator.load_benchmark_results()

        assert len(results) == 1
        assert results[0]["name"] == "test_benchmark"
        assert results[0]["mean_time_ms"] == 1.5

    def test_extract_metrics(self, tmp_path):
        """Test extraction of performance metrics from results."""
        test_results = [
            {
                "name": "test_benchmark",
                "mean_time_ms": 2.0,
                "memory_usage_mb": 5.0,
                "timestamp": "2025-07-04T12:00:00",
                "configuration": {"test": True},
                "additional_metrics": {
                    "cache_hit_rate": 0.75,
                    "total_operations": 100,
                },
            }
        ]

        generator = PerformanceReportGenerator(str(tmp_path))
        metrics = generator.extract_metrics(test_results)

        assert (
            len(metrics) == 4
        )  # mean_time, memory_usage, cache_hit_rate, total_operations

        # Check metric extraction
        metric_names = [m.name for m in metrics]
        assert "mean_time" in metric_names
        assert "memory_usage" in metric_names
        assert "cache_hit_rate" in metric_names
        assert "total_operations" in metric_names

    def test_detect_regressions(self, tmp_path):
        """Test regression detection functionality."""
        generator = PerformanceReportGenerator(str(tmp_path))

        # Create metrics showing performance regression
        base_time = datetime.now()
        metrics = [
            PerformanceMetric(
                "mean_time", 1.0, "ms", base_time, "test_benchmark", {}
            ),
            PerformanceMetric(
                "mean_time", 1.5, "ms", base_time, "test_benchmark", {}
            ),  # 50% regression
        ]

        regressions = generator.detect_regressions(
            metrics, threshold_percent=10.0
        )

        assert len(regressions) == 1
        assert regressions[0].benchmark_name == "test_benchmark"
        assert regressions[0].metric_name == "mean_time"
        assert regressions[0].regression_percent == 50.0
        assert regressions[0].severity == "severe"

    def test_infer_unit(self, tmp_path):
        """Test unit inference from metric names."""
        generator = PerformanceReportGenerator(str(tmp_path))

        assert generator._infer_unit("mean_time") == "ms"
        assert generator._infer_unit("memory_usage") == "MB"
        assert generator._infer_unit("cache_hit_rate") == "%"
        assert generator._infer_unit("speedup_factor") == "x"
        assert generator._infer_unit("operation_count") == "count"
        assert generator._infer_unit("unknown_metric") == "unit"


class TestPerformanceMonitor:
    """Test the performance monitoring functionality."""

    def test_check_performance_gates(self, tmp_path):
        """Test performance quality gates checking."""
        # Create test result files
        results_dir = tmp_path / "test_results"
        results_dir.mkdir()

        # Create result with good performance
        good_result = {
            "name": "good_benchmark",
            "additional_metrics": {
                "speedup_factor": 2.0,
                "cache_hit_rate": 0.8,
            },
        }

        # Create result with poor performance
        poor_result = {
            "name": "poor_benchmark",
            "memory_usage_mb": 150.0,  # Over 100MB limit
            "additional_metrics": {
                "speedup_factor": 0.5,  # Poor speedup
                "cache_hit_rate": 0.1,  # Low hit rate
            },
        }

        # Write test results (use proper naming pattern)
        with open(
            results_dir / "test_benchmark_results_20250704_120000.json", "w"
        ) as f:
            json.dump([good_result, poor_result], f)

        monitor = PerformanceMonitor(regression_threshold=10.0)
        monitor.results_dir = results_dir

        gates = monitor.check_performance_gates()

        # Should fail due to poor performance
        assert gates["overall_status"] == "fail"
        assert "severe_regressions" in gates["checks"]
        assert "cache_effectiveness" in gates["checks"]

    def test_create_ci_summary(self):
        """Test CI summary generation."""
        monitor = PerformanceMonitor()

        benchmark_results = {
            "success": True,
            "benchmarks": {"test.py": {"status": "success"}},
            "errors": [],
        }

        report_results = {
            "status": "success",
            "reports": {
                "charts": ["chart1.png", "chart2.png"],
                "regressions": 2,
            },
        }

        gates = {
            "overall_status": "pass",
            "regression_threshold": 10.0,
            "failures": [],
            "checks": {
                "cache_effectiveness": {"status": "pass", "details": []},
                "memory_efficiency": {
                    "status": "warn",
                    "details": ["high memory"],
                },
            },
        }

        summary = monitor.create_ci_summary(
            benchmark_results, report_results, gates
        )

        assert "Performance Monitoring Summary" in summary
        assert "✅ Success" in summary
        assert "Charts**: 2" in summary
        assert "Regressions Detected**: 2" in summary

    @patch("subprocess.run")
    def test_run_all_benchmarks(self, mock_run, tmp_path):
        """Test benchmark execution."""
        # Mock successful benchmark execution
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Benchmark completed successfully", stderr=""
        )

        # Create test benchmark files
        results_dir = tmp_path / "test_performance"
        results_dir.mkdir()

        # Create mock benchmark script
        benchmark_script = results_dir / "test_benchmark.py"
        benchmark_script.write_text("# Test benchmark script")

        monitor = PerformanceMonitor()
        monitor.results_dir = results_dir

        # Test with no benchmark scripts (should handle gracefully)
        results = monitor.run_all_benchmarks()

        assert "start_time" in results
        assert "end_time" in results
        assert "benchmarks" in results
        assert "success" in results


class TestIntegrationBehavior:
    """Test integrated behavior of the performance monitoring system."""

    def test_full_pipeline_with_real_data(self, tmp_path):
        """Test the complete monitoring pipeline with realistic data."""
        # Create realistic benchmark results
        results_dir = tmp_path / "integration_test"
        results_dir.mkdir()

        realistic_results = [
            {
                "name": "matrix_caching_enabled",
                "mean_time_ms": 0.5,
                "memory_usage_mb": 12.0,
                "timestamp": "2025-07-04T12:00:00",
                "configuration": {"cache_enabled": True},
                "additional_metrics": {
                    "cache_hit_rate": 0.85,
                    "speedup_factor": 5.2,
                    "cache_memory_mb": 8.5,
                },
            },
            {
                "name": "matrix_caching_disabled",
                "mean_time_ms": 2.6,
                "memory_usage_mb": 4.0,
                "timestamp": "2025-07-04T12:01:00",
                "configuration": {"cache_enabled": False},
                "additional_metrics": {
                    "cache_hit_rate": 0.0,
                    "speedup_factor": 1.0,
                    "cache_memory_mb": 0.0,
                },
            },
        ]

        # Write realistic results (use proper naming pattern)
        with open(
            results_dir / "realistic_benchmark_results_20250704_120000.json",
            "w",
        ) as f:
            json.dump(realistic_results, f)

        # Test report generation
        generator = PerformanceReportGenerator(str(results_dir))
        results = generator.load_benchmark_results()

        assert len(results) == 2

        metrics = generator.extract_metrics(results)
        assert len(metrics) > 0

        # Should show caching effectiveness
        cache_metrics = [m for m in metrics if m.name == "cache_hit_rate"]
        assert len(cache_metrics) == 2
        assert any(m.value == 0.85 for m in cache_metrics)

        # Test quality gates
        monitor = PerformanceMonitor()
        monitor.results_dir = results_dir

        gates = monitor.check_performance_gates()

        # Should pass with good performance
        assert gates["overall_status"] in [
            "pass",
            "warn",
        ]  # May warn on some thresholds

    def test_regression_detection_scenarios(self, tmp_path):
        """Test various regression detection scenarios."""
        generator = PerformanceReportGenerator(str(tmp_path))

        base_time = datetime.now()

        # Test severe time regression
        time_metrics = [
            PerformanceMetric("mean_time", 1.0, "ms", base_time, "test", {}),
            PerformanceMetric(
                "mean_time", 2.0, "ms", base_time, "test", {}
            ),  # 100% regression
        ]

        regressions = generator.detect_regressions(time_metrics, 10.0)
        assert len(regressions) == 1
        assert regressions[0].severity == "severe"

        # Test cache hit rate regression
        cache_metrics = [
            PerformanceMetric(
                "cache_hit_rate", 0.8, "%", base_time, "test", {}
            ),
            PerformanceMetric(
                "cache_hit_rate", 0.6, "%", base_time, "test", {}
            ),  # 25% drop
        ]

        cache_regressions = generator.detect_regressions(cache_metrics, 10.0)
        assert len(cache_regressions) == 1
        assert cache_regressions[0].regression_percent == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
