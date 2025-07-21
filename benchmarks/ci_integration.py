#!/usr/bin/env python3
"""
CI/CD Integration for Performance Benchmarks.
===========================================

This module provides integration with CI/CD pipelines for automated performance
testing, regression detection, and reporting.

Features:
- Automatic performance regression detection
- Historical data comparison
- Performance trend analysis
- CI/CD pipeline integration (GitHub Actions, GitLab CI, Jenkins)
- Performance dashboards and alerts
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Performance thresholds
THRESHOLDS = {
    "critical": 0.10,  # 10% regression fails CI
    "warning": 0.05,  # 5% regression triggers warning
    "improvement": -0.05,  # 5% improvement noted
}

# Baseline storage
BASELINE_FILE = "performance_baseline.json"
HISTORY_FILE = "performance_history.json"


@dataclass
class PerformanceResult:
    """Container for performance test results."""

    benchmark_name: str
    category: str
    duration_ms: float
    throughput_ops_sec: float
    memory_mb: float
    cpu_percent: float
    timestamp: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RegressionReport:
    """Performance regression report."""

    benchmark_name: str
    metric: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBaseline:
    """Manage performance baselines."""

    def __init__(self, baseline_path: Path):
        """Initialize the performance baseline."""
        self.baseline_path = baseline_path
        self.baseline_data = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline data from file."""
        if self.baseline_path.exists():
            with open(self.baseline_path, "r") as f:
                return json.load(f)
        return {}

    def save_baseline(self):
        """Save baseline data to file."""
        with open(self.baseline_path, "w") as f:
            json.dump(self.baseline_data, f, indent=2)

    def update_baseline(self, results: List[PerformanceResult]):
        """Update baseline with new results."""
        for result in results:
            key = f"{result.category}.{result.benchmark_name}"
            self.baseline_data[key] = {
                "duration_ms": result.duration_ms,
                "throughput_ops_sec": result.throughput_ops_sec,
                "memory_mb": result.memory_mb,
                "cpu_percent": result.cpu_percent,
                "timestamp": result.timestamp,
                "git_commit": result.git_commit,
            }
        self.save_baseline()

    def get_baseline(self, category: str, benchmark: str) -> Optional[Dict[str, Any]]:
        """Get baseline for specific benchmark."""
        key = f"{category}.{benchmark}"
        return self.baseline_data.get(key)


class PerformanceHistory:
    """Track performance history over time."""

    def __init__(self, history_path: Path):
        """Initialize the performance history tracker."""
        self.history_path = history_path
        self.history_data = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history data from file."""
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                return json.load(f)
        return []

    def save_history(self):
        """Save history data to file."""
        with open(self.history_path, "w") as f:
            json.dump(self.history_data, f, indent=2)

    def add_results(self, results: List[PerformanceResult]):
        """Add results to history."""
        for result in results:
            self.history_data.append(result.to_dict())

        # Keep only last 90 days of data
        cutoff = datetime.now() - timedelta(days=90)
        self.history_data = [
            entry
            for entry in self.history_data
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]

        self.save_history()

    def get_trend(
        self, category: str, benchmark: str, metric: str
    ) -> List[Tuple[str, float]]:
        """Get performance trend for specific benchmark."""
        trend = []
        for entry in self.history_data:
            if entry["category"] == category and entry["benchmark_name"] == benchmark:
                timestamp = entry["timestamp"]
                value = entry.get(metric, 0)
                trend.append((timestamp, value))
        return sorted(trend, key=lambda x: x[0])


class RegressionDetector:
    """Detect performance regressions."""

    def __init__(self, baseline: PerformanceBaseline):
        """Initialize the regression detector."""
        self.baseline = baseline

    def detect_regressions(
        self, results: List[PerformanceResult]
    ) -> Tuple[List[RegressionReport], List[RegressionReport]]:
        """
        Detect regressions and improvements.

        Returns:
            Tuple of (regressions, improvements)
        """
        regressions = []
        improvements = []

        for result in results:
            baseline_data = self.baseline.get_baseline(
                result.category, result.benchmark_name
            )
            if not baseline_data:
                continue

            # Check each metric
            metrics = {
                "duration_ms": (
                    result.duration_ms,
                    baseline_data.get("duration_ms", 0),
                ),
                "memory_mb": (
                    result.memory_mb,
                    baseline_data.get("memory_mb", 0),
                ),
                "cpu_percent": (
                    result.cpu_percent,
                    baseline_data.get("cpu_percent", 0),
                ),
            }

            for metric_name, (current, baseline_val) in metrics.items():
                if baseline_val <= 0:
                    continue

                # Calculate regression percentage
                regression_pct = (current - baseline_val) / baseline_val

                # Determine severity
                if regression_pct > THRESHOLDS["critical"]:
                    severity = "critical"
                elif regression_pct > THRESHOLDS["warning"]:
                    severity = "warning"
                elif regression_pct < THRESHOLDS["improvement"]:
                    severity = "improvement"
                else:
                    continue

                report = RegressionReport(
                    benchmark_name=result.benchmark_name,
                    metric=metric_name,
                    current_value=current,
                    baseline_value=baseline_val,
                    regression_percent=regression_pct * 100,
                    severity=severity,
                )

                if severity == "improvement":
                    improvements.append(report)
                else:
                    regressions.append(report)

        return regressions, improvements


class CIIntegration:
    """Main CI integration class."""

    def __init__(self, results_dir: Path, baseline_dir: Path):
        """Initialize the CI integration."""
        self.results_dir = results_dir
        self.baseline_dir = baseline_dir
        self.baseline = PerformanceBaseline(baseline_dir / BASELINE_FILE)
        self.history = PerformanceHistory(baseline_dir / HISTORY_FILE)
        self.detector = RegressionDetector(self.baseline)

    def load_benchmark_results(self, results_file: Path) -> List[PerformanceResult]:
        """Load benchmark results from pytest-benchmark JSON."""
        with open(results_file, "r") as f:
            data = json.load(f)

        results = []

        # Parse pytest-benchmark format
        for benchmark in data.get("benchmarks", []):
            # Extract category from group or name
            parts = benchmark["name"].split(".")
            category = parts[1] if len(parts) > 1 else "general"
            name = parts[-1]

            # Get stats
            stats = benchmark.get("stats", {})

            result = PerformanceResult(
                benchmark_name=name,
                category=category,
                duration_ms=stats.get("mean", 0) * 1000,  # Convert to ms
                throughput_ops_sec=1.0 / stats.get("mean", 1),  # Ops per second
                memory_mb=benchmark.get("extra_info", {}).get("memory_mb", 0),
                cpu_percent=benchmark.get("extra_info", {}).get("cpu_percent", 0),
                timestamp=datetime.now().isoformat(),
                git_commit=os.environ.get("GIT_COMMIT", "unknown"),
                git_branch=os.environ.get("GIT_BRANCH", "unknown"),
            )

            results.append(result)

        return results

    def run_regression_check(self, results: List[PerformanceResult]) -> Dict[str, Any]:
        """Run regression check and generate report."""
        regressions, improvements = self.detector.detect_regressions(results)

        # Determine overall status
        critical_count = sum(1 for r in regressions if r.severity == "critical")
        warning_count = sum(1 for r in regressions if r.severity == "warning")

        if critical_count > 0:
            overall_status = "fail"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "pass"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_benchmarks": len(results),
                "regressions": len(regressions),
                "improvements": len(improvements),
                "critical": critical_count,
                "warnings": warning_count,
            },
            "regressions": [r.to_dict() for r in regressions],
            "improvements": [r.to_dict() for r in improvements],
            "results": [r.to_dict() for r in results],
        }

        return report

    def generate_trend_report(self, output_file: Path):
        """Generate performance trend report."""
        # Get unique benchmarks
        benchmarks = set()
        for entry in self.history.history_data:
            key = (entry["category"], entry["benchmark_name"])
            benchmarks.add(key)

        trends = {}
        for category, benchmark in benchmarks:
            key = f"{category}.{benchmark}"
            trends[key] = {
                "duration_ms": self.history.get_trend(
                    category, benchmark, "duration_ms"
                ),
                "memory_mb": self.history.get_trend(category, benchmark, "memory_mb"),
            }

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "period": "last_90_days",
            "trends": trends,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

    def update_baseline(self, results: List[PerformanceResult], force: bool = False):
        """Update performance baseline."""
        if force:
            self.baseline.update_baseline(results)
            logger.info("Baseline updated (forced)")
        else:
            # Only update if all results are improvements or within threshold
            regressions, _ = self.detector.detect_regressions(results)
            if not regressions:
                self.baseline.update_baseline(results)
                logger.info("Baseline updated (no regressions)")
            else:
                logger.warning(
                    f"Baseline not updated due to {len(regressions)} regressions"
                )

    def generate_github_comment(self, report: Dict[str, Any]) -> str:
        """Generate GitHub PR comment from report."""
        comment = "## 🚀 Performance Benchmark Results\n\n"

        # Summary
        summary = report["summary"]
        comment += f"**Overall Status:** {report['overall_status'].upper()}\n\n"
        comment += f"- Total Benchmarks: {summary['total_benchmarks']}\n"
        comment += f"- Regressions: {summary['regressions']}\n"
        comment += f"- Improvements: {summary['improvements']}\n\n"

        # Regressions
        if report["regressions"]:
            comment += "### ⚠️ Performance Regressions\n\n"
            comment += (
                "| Benchmark | Metric | Current | Baseline | Change | Severity |\n"
            )
            comment += (
                "|-----------|--------|---------|----------|--------|----------|\n"
            )

            for reg in report["regressions"]:
                comment += f"| {reg['benchmark_name']} | {reg['metric']} | "
                comment += (
                    f"{reg['current_value']:.2f} | {reg['baseline_value']:.2f} | "
                )
                comment += f"+{reg['regression_percent']:.1f}% | {reg['severity']} |\n"

            comment += "\n"

        # Improvements
        if report["improvements"]:
            comment += "### ✅ Performance Improvements\n\n"
            comment += "| Benchmark | Metric | Current | Baseline | Change |\n"
            comment += "|-----------|--------|---------|----------|--------|\n"

            for imp in report["improvements"]:
                comment += f"| {imp['benchmark_name']} | {imp['metric']} | "
                comment += (
                    f"{imp['current_value']:.2f} | {imp['baseline_value']:.2f} | "
                )
                comment += f"{imp['regression_percent']:.1f}% |\n"

        return comment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CI/CD integration for performance benchmarks"
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to pytest-benchmark results JSON",
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("benchmarks/baselines"),
        help="Directory for baseline storage",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/ci_results"),
        help="Output directory for reports",
    )

    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current results",
    )

    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Force baseline update even with regressions",
    )

    parser.add_argument(
        "--github-comment",
        action="store_true",
        help="Generate GitHub PR comment",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code on regression",
    )

    args = parser.parse_args()

    # Create directories
    args.baseline_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CI integration
    ci = CIIntegration(args.output_dir, args.baseline_dir)

    # Load benchmark results
    try:
        results = ci.load_benchmark_results(args.results_file)
        logger.info(f"Loaded {len(results)} benchmark results")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        sys.exit(1)

    # Add results to history
    ci.history.add_results(results)

    # Run regression check
    report = ci.run_regression_check(results)

    # Save report
    report_file = (
        args.output_dir
        / f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    latest_file = args.output_dir / "latest_regression_report.json"
    with open(latest_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Regression report saved to {report_file}")

    # Generate trend report
    trend_file = args.output_dir / "performance_trends.json"
    ci.generate_trend_report(trend_file)
    logger.info(f"Trend report saved to {trend_file}")

    # Update baseline if requested
    if args.update_baseline:
        ci.update_baseline(results, force=args.force_baseline)

    # Generate GitHub comment if requested
    if args.github_comment:
        comment = ci.generate_github_comment(report)
        comment_file = args.output_dir / "github_comment.md"
        with open(comment_file, "w") as f:
            f.write(comment)
        logger.info(f"GitHub comment saved to {comment_file}")

    # Print summary
    print(f"\nPerformance Check: {report['overall_status'].upper()}")
    print(f"Regressions: {report['summary']['regressions']}")
    print(f"Improvements: {report['summary']['improvements']}")

    # Exit with appropriate code
    if args.fail_on_regression and report["overall_status"] == "fail":
        logger.error("Critical performance regressions detected!")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
