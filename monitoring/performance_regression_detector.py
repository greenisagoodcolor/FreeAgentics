#!/usr/bin/env python3
"""
FreeAgentics Performance Regression Detection System
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline definition."""

    name: str
    query: str
    baseline_value: float
    critical_threshold: float
    warning_threshold: float
    measurement_window: str
    unit: str
    direction: str  # 'lower' or 'higher' is better


@dataclass
class PerformanceResult:
    """Performance test result."""

    metric_name: str
    current_value: float
    baseline_value: float
    threshold_value: float
    status: str  # 'pass', 'warning', 'critical'
    deviation_percent: float
    measurement_time: str


@dataclass
class RegressionReport:
    """Performance regression report."""

    test_timestamp: str
    deployment_version: str
    overall_status: str
    passed_tests: int
    warning_tests: int
    critical_tests: int
    total_tests: int
    results: List[PerformanceResult]
    recommendations: List[str]


class PerformanceRegressionDetector:
    """Main performance regression detector class."""

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> List[PerformanceBaseline]:
        """Load performance baselines."""
        return [
            # System-level baselines
            PerformanceBaseline(
                name="system_availability",
                query='avg_over_time(up{job="freeagentics-backend"}[5m])',
                baseline_value=0.999,
                critical_threshold=0.995,
                warning_threshold=0.998,
                measurement_window="5m",
                unit="ratio",
                direction="higher",
            ),
            PerformanceBaseline(
                name="system_memory_usage",
                query="freeagentics_system_memory_usage_bytes / (1024*1024*1024)",
                baseline_value=1.5,
                critical_threshold=2.0,
                warning_threshold=1.8,
                measurement_window="5m",
                unit="GB",
                direction="lower",
            ),
            # Agent coordination baselines
            PerformanceBaseline(
                name="active_agents",
                query="freeagentics_system_active_agents_total",
                baseline_value=15,
                critical_threshold=50,
                warning_threshold=40,
                measurement_window="5m",
                unit="count",
                direction="lower",
            ),
            PerformanceBaseline(
                name="coordination_success_rate",
                query='rate(freeagentics_agent_coordination_requests_total{status="success"}[5m]) / rate(freeagentics_agent_coordination_requests_total[5m])',
                baseline_value=0.95,
                critical_threshold=0.90,
                warning_threshold=0.93,
                measurement_window="5m",
                unit="ratio",
                direction="higher",
            ),
            PerformanceBaseline(
                name="coordination_p95_duration",
                query="histogram_quantile(0.95, rate(freeagentics_agent_coordination_duration_seconds_bucket[5m]))",
                baseline_value=1.5,
                critical_threshold=2.0,
                warning_threshold=1.8,
                measurement_window="5m",
                unit="seconds",
                direction="lower",
            ),
            PerformanceBaseline(
                name="coordination_timeout_rate",
                query='rate(freeagentics_agent_coordination_errors_total{error_type="timeout"}[5m])',
                baseline_value=0.02,
                critical_threshold=0.05,
                warning_threshold=0.03,
                measurement_window="5m",
                unit="ratio",
                direction="lower",
            ),
            # Memory baselines
            PerformanceBaseline(
                name="avg_agent_memory",
                query="avg(freeagentics_agent_memory_usage_bytes) / (1024*1024)",
                baseline_value=20,
                critical_threshold=34.5,
                warning_threshold=30,
                measurement_window="5m",
                unit="MB",
                direction="lower",
            ),
            PerformanceBaseline(
                name="max_agent_memory",
                query="max(freeagentics_agent_memory_usage_bytes) / (1024*1024)",
                baseline_value=30,
                critical_threshold=34.5,
                warning_threshold=32,
                measurement_window="5m",
                unit="MB",
                direction="lower",
            ),
            # API performance baselines
            PerformanceBaseline(
                name="api_p95_response_time",
                query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="freeagentics-backend"}[5m]))',
                baseline_value=0.3,
                critical_threshold=0.5,
                warning_threshold=0.4,
                measurement_window="5m",
                unit="seconds",
                direction="lower",
            ),
            PerformanceBaseline(
                name="api_error_rate",
                query='rate(http_requests_total{job="freeagentics-backend",status=~"5.."}[5m]) / rate(http_requests_total{job="freeagentics-backend"}[5m])',
                baseline_value=0.01,
                critical_threshold=0.10,
                warning_threshold=0.05,
                measurement_window="5m",
                unit="ratio",
                direction="lower",
            ),
            # Belief system baselines
            PerformanceBaseline(
                name="belief_free_energy",
                query="freeagentics_belief_free_energy_current",
                baseline_value=2.5,
                critical_threshold=10,
                warning_threshold=8,
                measurement_window="5m",
                unit="value",
                direction="lower",
            ),
            PerformanceBaseline(
                name="belief_accuracy",
                query="freeagentics_belief_accuracy_ratio",
                baseline_value=0.8,
                critical_threshold=0.7,
                warning_threshold=0.75,
                measurement_window="5m",
                unit="ratio",
                direction="higher",
            ),
            # Business metrics baselines
            PerformanceBaseline(
                name="user_interaction_rate",
                query="rate(freeagentics_business_user_interactions_total[1h])",
                baseline_value=0.1,
                critical_threshold=0.01,
                warning_threshold=0.05,
                measurement_window="1h",
                unit="per_hour",
                direction="higher",
            ),
            PerformanceBaseline(
                name="response_quality",
                query="freeagentics_business_response_quality_score",
                baseline_value=0.75,
                critical_threshold=0.6,
                warning_threshold=0.7,
                measurement_window="5m",
                unit="ratio",
                direction="higher",
            ),
        ]

    def query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus and return single value."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            if data["status"] == "success" and data["data"]["result"]:
                return float(data["data"]["result"][0]["value"][1])

            return None

        except Exception as e:
            logger.error(f"Prometheus query failed for '{query}': {e}")
            return None

    def evaluate_baseline(self, baseline: PerformanceBaseline) -> PerformanceResult:
        """Evaluate a single performance baseline."""
        logger.info(f"Evaluating baseline: {baseline.name}")

        current_value = self.query_prometheus(baseline.query)

        if current_value is None:
            return PerformanceResult(
                metric_name=baseline.name,
                current_value=0,
                baseline_value=baseline.baseline_value,
                threshold_value=baseline.critical_threshold,
                status="error",
                deviation_percent=0,
                measurement_time=datetime.now().isoformat(),
            )

        # Calculate deviation
        deviation_percent = (
            (current_value - baseline.baseline_value) / baseline.baseline_value
        ) * 100

        # Determine status based on direction
        if baseline.direction == "lower":
            if current_value > baseline.critical_threshold:
                status = "critical"
                threshold_value = baseline.critical_threshold
            elif current_value > baseline.warning_threshold:
                status = "warning"
                threshold_value = baseline.warning_threshold
            else:
                status = "pass"
                threshold_value = baseline.baseline_value
        else:  # higher is better
            if current_value < baseline.critical_threshold:
                status = "critical"
                threshold_value = baseline.critical_threshold
            elif current_value < baseline.warning_threshold:
                status = "warning"
                threshold_value = baseline.warning_threshold
            else:
                status = "pass"
                threshold_value = baseline.baseline_value

        return PerformanceResult(
            metric_name=baseline.name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            threshold_value=threshold_value,
            status=status,
            deviation_percent=deviation_percent,
            measurement_time=datetime.now().isoformat(),
        )

    def run_regression_tests(self, deployment_version: str = "unknown") -> RegressionReport:
        """Run all performance regression tests."""
        logger.info("Starting performance regression tests...")

        results = []
        passed_tests = 0
        warning_tests = 0
        critical_tests = 0

        for baseline in self.baselines:
            result = self.evaluate_baseline(baseline)
            results.append(result)

            if result.status == "pass":
                passed_tests += 1
            elif result.status == "warning":
                warning_tests += 1
            elif result.status == "critical":
                critical_tests += 1

        # Determine overall status
        overall_status = "pass"
        if critical_tests > 0:
            overall_status = "critical"
        elif warning_tests > 0:
            overall_status = "warning"

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return RegressionReport(
            test_timestamp=datetime.now().isoformat(),
            deployment_version=deployment_version,
            overall_status=overall_status,
            passed_tests=passed_tests,
            warning_tests=warning_tests,
            critical_tests=critical_tests,
            total_tests=len(results),
            results=results,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, results: List[PerformanceResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for critical failures
        critical_results = [r for r in results if r.status == "critical"]
        if critical_results:
            recommendations.append(
                "CRITICAL: Immediate action required - deployment should be blocked"
            )

            for result in critical_results:
                if result.metric_name == "system_availability":
                    recommendations.append(
                        "- System availability is below acceptable threshold - check service health"
                    )
                elif result.metric_name == "active_agents":
                    recommendations.append(
                        "- Agent count exceeds coordination limit - implement agent throttling"
                    )
                elif result.metric_name == "max_agent_memory":
                    recommendations.append(
                        "- Agent memory usage exceeds limit - investigate memory leaks"
                    )
                elif result.metric_name == "api_p95_response_time":
                    recommendations.append(
                        "- API response time is too high - optimize slow endpoints"
                    )
                elif result.metric_name == "api_error_rate":
                    recommendations.append(
                        "- API error rate is too high - investigate service errors"
                    )

        # Check for warning conditions
        warning_results = [r for r in results if r.status == "warning"]
        if warning_results:
            recommendations.append("WARNING: Performance degradation detected - monitor closely")

            for result in warning_results:
                if result.metric_name == "coordination_success_rate":
                    recommendations.append(
                        "- Coordination success rate declining - check agent coordination logic"
                    )
                elif result.metric_name == "belief_accuracy":
                    recommendations.append(
                        "- Belief system accuracy declining - review belief update algorithms"
                    )
                elif result.metric_name == "response_quality":
                    recommendations.append(
                        "- Response quality declining - review response generation logic"
                    )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "All performance tests passed - system is performing within acceptable limits"
            )

        return recommendations

    def save_report(self, report: RegressionReport, filename: str = None):
        """Save regression report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_regression_report_{timestamp}.json"

        # Create reports directory if it doesn't exist
        reports_dir = "monitoring/reports"
        os.makedirs(reports_dir, exist_ok=True)

        filepath = os.path.join(reports_dir, filename)

        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2)

        logger.info(f"Report saved to: {filepath}")
        return filepath

    def print_report(self, report: RegressionReport):
        """Print regression report to console."""
        print("\n" + "=" * 80)
        print("FREEAGENTICS PERFORMANCE REGRESSION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.test_timestamp}")
        print(f"Deployment Version: {report.deployment_version}")
        print(f"Overall Status: {report.overall_status.upper()}")
        print(
            f"Results: {report.passed_tests} passed, {report.warning_tests} warnings, {report.critical_tests} critical"
        )
        print(f"Total Tests: {report.total_tests}")

        print("\n" + "-" * 80)
        print("DETAILED RESULTS")
        print("-" * 80)

        for result in report.results:
            status_icon = (
                "✅" if result.status == "pass" else "⚠️" if result.status == "warning" else "❌"
            )
            print(f"{status_icon} {result.metric_name}")
            print(f"   Current: {result.current_value:.4f}")
            print(f"   Baseline: {result.baseline_value:.4f}")
            print(f"   Threshold: {result.threshold_value:.4f}")
            print(f"   Deviation: {result.deviation_percent:+.2f}%")
            print()

        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 80)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="FreeAgentics Performance Regression Detector")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090", help="Prometheus URL")
    parser.add_argument("--deployment-version", default="unknown", help="Deployment version")
    parser.add_argument("--output-file", help="Output file for report")
    parser.add_argument("--fail-on-warning", action="store_true", help="Fail if warnings detected")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    detector = PerformanceRegressionDetector(args.prometheus_url)

    try:
        report = detector.run_regression_tests(args.deployment_version)

        if not args.quiet:
            detector.print_report(report)

        # Save report
        detector.save_report(report, args.output_file)

        # Determine exit code
        if report.overall_status == "critical":
            print("\n❌ CRITICAL: Performance regression detected - deployment should be blocked")
            sys.exit(1)
        elif report.overall_status == "warning":
            if args.fail_on_warning:
                print(
                    "\n⚠️ WARNING: Performance degradation detected - failing due to --fail-on-warning"
                )
                sys.exit(1)
            else:
                print("\n⚠️ WARNING: Performance degradation detected - monitor closely")
                sys.exit(0)
        else:
            print("\n✅ SUCCESS: All performance tests passed")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Performance regression detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
