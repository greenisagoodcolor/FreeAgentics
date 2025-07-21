"""
Master Performance Testing Runner
=================================

This is the main entry point for running comprehensive performance tests.
It integrates all performance testing components and provides a unified
interface for executing the complete performance validation suite.

Features:
- Orchestrates all performance testing components
- Validates SLA requirements against test results
- Generates comprehensive reports
- Integrates with CI/CD pipelines
- Provides performance baseline tracking
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all performance testing components
from tests.performance.comprehensive_performance_suite import (
    ComprehensivePerformanceSuite,
    LoadTestConfig,
)
from tests.performance.database_load_testing import DatabaseLoadTester
from tests.performance.load_testing_framework import (
    LoadTestingFramework,
)
from tests.performance.performance_monitoring_dashboard import (
    PerformanceMonitoringDashboard,
)
from tests.performance.performance_regression_detector import (
    PerformanceRegressionDetector,
)
from tests.performance.stress_testing_framework import StressTestingFramework
from tests.performance.websocket_performance_tests import (
    WebSocketPerformanceTester,
)

logger = logging.getLogger(__name__)


class MasterPerformanceRunner:
    """Master controller for all performance testing."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        db_config: Optional[Dict[str, Any]] = None,
        websocket_url: str = "ws://localhost:8000/ws",
    ):
        self.base_url = base_url
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5432,
            "database": "freeagentics",
            "user": "postgres",
            "password": "password",
        }
        self.websocket_url = websocket_url

        # Initialize all testing components
        self.performance_suite = ComprehensivePerformanceSuite()
        self.load_framework = LoadTestingFramework(base_url)
        self.websocket_tester = WebSocketPerformanceTester(websocket_url)
        self.database_tester = DatabaseLoadTester(self.db_config)
        self.stress_framework = StressTestingFramework(base_url)
        self.monitoring_dashboard = PerformanceMonitoringDashboard()
        self.regression_detector = PerformanceRegressionDetector()

        # Test configuration
        self.test_config = {
            "concurrent_users": 100,
            "test_duration": 300,  # 5 minutes
            "ramp_up_time": 60,  # 1 minute
            "target_response_time": 3000,  # 3 seconds
            "min_success_rate": 99.0,
            "environment": "testing",
            "version": "1.0.0",
            "branch": "main",
            "commit_hash": "unknown",
        }

        # SLA requirements from documentation
        self.sla_requirements = {
            "response_time": {
                "p50_ms": 500,
                "p95_ms": 2000,
                "p99_ms": 3000,
                "max_ms": 5000,
            },
            "throughput": {
                "min_rps": 100,
                "target_rps": 500,
                "peak_rps": 1000,
            },
            "availability": {
                "uptime_percent": 99.9,
                "max_downtime_minutes": 43.2,
            },
            "error_rate": {"max_percent": 1.0, "target_percent": 0.1},
            "resource_usage": {
                "cpu_max_percent": 85,
                "memory_max_mb": 4096,
                "memory_growth_mb_per_hour": 200,
            },
        }

    async def run_full_performance_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite."""
        logger.info("Starting full performance validation suite")

        # Start monitoring
        self.monitoring_dashboard.start_monitoring()

        try:
            results = {
                "start_time": datetime.now().isoformat(),
                "test_configuration": self.test_config,
                "sla_requirements": self.sla_requirements,
                "test_results": {},
                "sla_validation": {},
                "overall_status": "unknown",
                "recommendations": [],
            }

            # 1. Run comprehensive performance suite
            logger.info("Running comprehensive performance suite...")
            perf_config = LoadTestConfig(
                concurrent_users=self.test_config["concurrent_users"],
                test_duration_seconds=self.test_config["test_duration"],
                target_response_time_ms=self.test_config["target_response_time"],
                min_success_rate=self.test_config["min_success_rate"],
            )

            perf_results = await self.performance_suite.run_all_tests(perf_config)
            results["test_results"]["comprehensive_suite"] = perf_results

            # 2. Run load testing scenarios
            logger.info("Running load testing scenarios...")
            load_results = {}

            # Run baseline, standard, and peak scenarios
            for scenario_name in ["baseline", "standard", "peak"]:
                scenario = self.load_framework.get_scenario(scenario_name)
                if scenario:
                    # Scale down for testing
                    scenario.user_count = min(scenario.user_count, 50)
                    scenario.duration_seconds = min(scenario.duration_seconds, 180)

                    scenario_result = await self.load_framework.run_load_test(scenario)
                    load_results[scenario_name] = (
                        self.load_framework.generate_load_test_report(scenario_result)
                    )

            results["test_results"]["load_testing"] = load_results

            # 3. Run WebSocket performance tests
            logger.info("Running WebSocket performance tests...")
            websocket_results = (
                await self.websocket_tester.run_comprehensive_websocket_test_suite()
            )
            results["test_results"]["websocket_performance"] = websocket_results

            # 4. Run database load tests
            logger.info("Running database load tests...")
            try:
                database_results = (
                    await self.database_tester.run_comprehensive_database_test_suite()
                )
                results["test_results"]["database_load"] = database_results
            except Exception as e:
                logger.warning(
                    f"Database tests failed (expected if no DB running): {e}"
                )
                results["test_results"]["database_load"] = {"error": str(e)}

            # 5. Run stress tests
            logger.info("Running stress tests...")
            stress_results = {}

            # Run progressive load and failure recovery scenarios
            for scenario_name in ["progressive_load", "failure_recovery"]:
                try:
                    stress_result = (
                        await self.stress_framework.run_stress_test_scenario(
                            scenario_name
                        )
                    )
                    stress_report = self.stress_framework.generate_stress_test_report(
                        stress_result
                    )
                    stress_results[scenario_name] = stress_report
                except Exception as e:
                    logger.warning(f"Stress test {scenario_name} failed: {e}")
                    stress_results[scenario_name] = {"error": str(e)}

            results["test_results"]["stress_testing"] = stress_results

            # 6. Validate SLA requirements
            logger.info("Validating SLA requirements...")
            sla_validation = self._validate_sla_requirements(results["test_results"])
            results["sla_validation"] = sla_validation

            # 7. Generate overall assessment
            overall_status = self._determine_overall_status(results)
            results["overall_status"] = overall_status

            # 8. Generate recommendations
            recommendations = self._generate_recommendations(results)
            results["recommendations"] = recommendations

            # 9. Add regression detection if we have historical data
            logger.info("Checking for performance regressions...")
            regression_results = await self._check_regressions(results)
            results["regression_analysis"] = regression_results

            # 10. Save results
            results["end_time"] = datetime.now().isoformat()
            results["total_duration_seconds"] = (
                datetime.fromisoformat(results["end_time"])
                - datetime.fromisoformat(results["start_time"])
            ).total_seconds()

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"performance_validation_results_{timestamp}.json"

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Performance validation results saved to: {results_file}")

            return results

        finally:
            # Stop monitoring
            self.monitoring_dashboard.stop_monitoring()

    def _validate_sla_requirements(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate test results against SLA requirements."""
        validation = {
            "overall_sla_met": True,
            "violations": [],
            "metrics": {},
            "compliance_score": 0.0,
        }

        violations = []
        total_checks = 0
        passed_checks = 0

        # Check comprehensive suite results
        if "comprehensive_suite" in test_results:
            suite_results = test_results["comprehensive_suite"]

            if "performance_metrics" in suite_results:
                metrics = suite_results["performance_metrics"]

                # Response time validation
                if "response_time" in metrics:
                    rt_metrics = metrics["response_time"]

                    total_checks += 4

                    if (
                        rt_metrics.get("p95_ms", 0)
                        > self.sla_requirements["response_time"]["p95_ms"]
                    ):
                        violations.append(
                            {
                                "requirement": "P95 Response Time",
                                "expected": f"≤ {self.sla_requirements['response_time']['p95_ms']}ms",
                                "actual": f"{rt_metrics.get('p95_ms', 0):.1f}ms",
                                "severity": "critical",
                            }
                        )
                    else:
                        passed_checks += 1

                    if (
                        rt_metrics.get("p99_ms", 0)
                        > self.sla_requirements["response_time"]["p99_ms"]
                    ):
                        violations.append(
                            {
                                "requirement": "P99 Response Time",
                                "expected": f"≤ {self.sla_requirements['response_time']['p99_ms']}ms",
                                "actual": f"{rt_metrics.get('p99_ms', 0):.1f}ms",
                                "severity": "high",
                            }
                        )
                    else:
                        passed_checks += 1

                # Throughput validation
                if "throughput" in metrics:
                    tp_metrics = metrics["throughput"]

                    total_checks += 1

                    if (
                        tp_metrics.get("average_ops_per_second", 0)
                        < self.sla_requirements["throughput"]["min_rps"]
                    ):
                        violations.append(
                            {
                                "requirement": "Minimum Throughput",
                                "expected": f"≥ {self.sla_requirements['throughput']['min_rps']} RPS",
                                "actual": f"{tp_metrics.get('average_ops_per_second', 0):.1f} RPS",
                                "severity": "high",
                            }
                        )
                    else:
                        passed_checks += 1

                # Reliability validation
                if "reliability" in metrics:
                    rel_metrics = metrics["reliability"]

                    total_checks += 1

                    error_rate = 100 - rel_metrics.get("average_success_rate", 0)
                    if error_rate > self.sla_requirements["error_rate"]["max_percent"]:
                        violations.append(
                            {
                                "requirement": "Maximum Error Rate",
                                "expected": f"≤ {self.sla_requirements['error_rate']['max_percent']}%",
                                "actual": f"{error_rate:.2f}%",
                                "severity": "critical",
                            }
                        )
                    else:
                        passed_checks += 1

        # Check load testing results
        if "load_testing" in test_results:
            for scenario_name, scenario_results in test_results["load_testing"].items():
                if "error" not in scenario_results:
                    total_checks += 1

                    req_summary = scenario_results.get("request_summary", {})
                    if (
                        req_summary.get("success_rate", 0)
                        < self.sla_requirements["error_rate"]["target_percent"]
                    ):
                        violations.append(
                            {
                                "requirement": f"Load Test Success Rate ({scenario_name})",
                                "expected": f"≥ {100 - self.sla_requirements['error_rate']['max_percent']}%",
                                "actual": f"{req_summary.get('success_rate', 0):.1f}%",
                                "severity": "high",
                            }
                        )
                    else:
                        passed_checks += 1

        # Calculate compliance score
        if total_checks > 0:
            validation["compliance_score"] = (passed_checks / total_checks) * 100

        validation["violations"] = violations
        validation["overall_sla_met"] = len(violations) == 0
        validation["metrics"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": len(violations),
        }

        return validation

    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall performance test status."""
        sla_validation = results.get("sla_validation", {})

        if not sla_validation.get("overall_sla_met", False):
            critical_violations = [
                v
                for v in sla_validation.get("violations", [])
                if v["severity"] == "critical"
            ]
            if critical_violations:
                return "FAIL"
            else:
                return "WARNING"

        compliance_score = sla_validation.get("compliance_score", 0)
        if compliance_score >= 95:
            return "PASS"
        elif compliance_score >= 90:
            return "PASS_WITH_WARNINGS"
        else:
            return "WARNING"

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []

        sla_validation = results.get("sla_validation", {})

        # SLA violation recommendations
        if sla_validation.get("violations"):
            critical_violations = [
                v for v in sla_validation["violations"] if v["severity"] == "critical"
            ]
            if critical_violations:
                recommendations.append(
                    f"CRITICAL: {len(critical_violations)} critical SLA violations detected. "
                    "System is not ready for production deployment."
                )

        # Response time recommendations
        response_time_violations = [
            v
            for v in sla_validation.get("violations", [])
            if "Response Time" in v["requirement"]
        ]
        if response_time_violations:
            recommendations.append(
                "High response times detected. Consider implementing caching, "
                "optimizing database queries, or scaling horizontally."
            )

        # Throughput recommendations
        throughput_violations = [
            v
            for v in sla_validation.get("violations", [])
            if "Throughput" in v["requirement"]
        ]
        if throughput_violations:
            recommendations.append(
                "Low throughput detected. Consider optimizing application performance, "
                "implementing connection pooling, or increasing server resources."
            )

        # Error rate recommendations
        error_rate_violations = [
            v
            for v in sla_validation.get("violations", [])
            if "Error Rate" in v["requirement"]
        ]
        if error_rate_violations:
            recommendations.append(
                "High error rates detected. Implement better error handling, "
                "circuit breakers, and investigate root causes."
            )

        # Stress test recommendations
        stress_results = results.get("test_results", {}).get("stress_testing", {})
        for scenario_name, scenario_results in stress_results.items():
            if "error" not in scenario_results:
                resilience = scenario_results.get("resilience_assessment", {})
                if resilience.get("system_resilience_score", 100) < 70:
                    recommendations.append(
                        f"Low resilience score in {scenario_name}. "
                        "Implement fault tolerance and recovery mechanisms."
                    )

        # General recommendations
        compliance_score = sla_validation.get("compliance_score", 0)
        if compliance_score >= 95:
            recommendations.append(
                "Excellent performance results. System meets all SLA requirements "
                "and is ready for production deployment."
            )
        elif compliance_score >= 90:
            recommendations.append(
                "Good performance results with minor issues. "
                "Address warnings before production deployment."
            )
        else:
            recommendations.append(
                "Performance results need improvement. "
                "Address violations before production deployment."
            )

        return recommendations

    async def _check_regressions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check for performance regressions."""
        try:
            # Extract metrics for regression analysis
            metrics = {}

            # From comprehensive suite
            if "comprehensive_suite" in results["test_results"]:
                suite_results = results["test_results"]["comprehensive_suite"]
                if "performance_metrics" in suite_results:
                    perf_metrics = suite_results["performance_metrics"]

                    if "response_time" in perf_metrics:
                        metrics["api_response_time_ms"] = perf_metrics[
                            "response_time"
                        ].get("average_ms", 0)

                    if "throughput" in perf_metrics:
                        metrics["api_requests_per_second"] = perf_metrics[
                            "throughput"
                        ].get("average_ops_per_second", 0)

                    if "resource_usage" in perf_metrics:
                        metrics["memory_usage_mb"] = perf_metrics["resource_usage"].get(
                            "average_memory_mb", 0
                        )
                        metrics["cpu_usage_percent"] = perf_metrics[
                            "resource_usage"
                        ].get("average_cpu_percent", 0)

            # Add test run to regression detector
            run_id = self.regression_detector.add_performance_test_run(
                version=self.test_config["version"],
                environment=self.test_config["environment"],
                branch=self.test_config["branch"],
                commit_hash=self.test_config["commit_hash"],
                metrics=metrics,
                test_duration_seconds=self.test_config["test_duration"],
            )

            # Generate regression report
            regression_report = self.regression_detector.generate_regression_report(
                run_id
            )

            return regression_report

        except Exception as e:
            logger.warning(f"Regression analysis failed: {e}")
            return {"error": str(e)}

    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "=" * 80)
        print("PERFORMANCE VALIDATION RESULTS SUMMARY")
        print("=" * 80)

        # Overall status
        status = results.get("overall_status", "UNKNOWN")
        status_emoji = {
            "PASS": "✅",
            "PASS_WITH_WARNINGS": "⚠️",
            "WARNING": "⚠️",
            "FAIL": "❌",
        }.get(status, "❓")

        print(f"\nOverall Status: {status_emoji} {status}")

        # Test duration
        duration = results.get("total_duration_seconds", 0)
        print(f"Total Test Duration: {duration:.1f} seconds")

        # SLA validation
        sla_validation = results.get("sla_validation", {})
        compliance_score = sla_validation.get("compliance_score", 0)
        print(f"SLA Compliance Score: {compliance_score:.1f}%")

        violations = sla_validation.get("violations", [])
        if violations:
            print(f"\nSLA Violations ({len(violations)}):")
            for violation in violations:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                }.get(violation["severity"], "⚪")
                print(
                    f"  {severity_emoji} {violation['requirement']}: {violation['actual']} (expected: {violation['expected']})"
                )
        else:
            print("\n✅ All SLA requirements met!")

        # Test results summary
        test_results = results.get("test_results", {})
        print("\nTest Results Summary:")

        for test_name, test_result in test_results.items():
            if isinstance(test_result, dict) and "error" not in test_result:
                print(f"  ✅ {test_name}: Completed successfully")
            else:
                print(f"  ❌ {test_name}: Failed or skipped")

        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        # Regression analysis
        regression_analysis = results.get("regression_analysis", {})
        if "error" not in regression_analysis:
            regression_summary = regression_analysis.get("regression_analysis", {})
            overall_regression_status = regression_summary.get(
                "overall_status", "unknown"
            )
            print(f"\nRegression Analysis: {overall_regression_status}")

            if overall_regression_status == "fail":
                print("  🔴 Performance regressions detected!")
            elif overall_regression_status == "warning":
                print("  🟡 Minor performance changes detected")
            else:
                print("  ✅ No significant regressions detected")

        print("\n" + "=" * 80)


async def main():
    """Main entry point for performance testing."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get configuration from environment variables
    base_url = os.getenv("PERFORMANCE_TEST_URL", "http://localhost:8000")
    environment = os.getenv("ENVIRONMENT", "testing")
    version = os.getenv("VERSION", "1.0.0")
    branch = os.getenv("BRANCH", "main")
    commit_hash = os.getenv("COMMIT_HASH", "unknown")

    # Database configuration
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "freeagentics"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "password"),
    }

    # Create runner
    runner = MasterPerformanceRunner(base_url=base_url, db_config=db_config)

    # Update configuration
    runner.test_config.update(
        {
            "environment": environment,
            "version": version,
            "branch": branch,
            "commit_hash": commit_hash,
        }
    )

    try:
        # Run full validation
        results = await runner.run_full_performance_validation()

        # Print summary
        runner.print_results_summary(results)

        # Exit with appropriate code
        overall_status = results.get("overall_status", "UNKNOWN")
        if overall_status == "PASS":
            sys.exit(0)
        elif overall_status in ["PASS_WITH_WARNINGS", "WARNING"]:
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
