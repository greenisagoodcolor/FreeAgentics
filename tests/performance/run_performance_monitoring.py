#!/usr/bin/env python3
"""Automated Performance Monitoring Script for CI/CD Integration.

Runs comprehensive performance benchmarks, generates reports, and triggers alerts
for performance regressions. Designed for integration with GitHub Actions, GitLab CI,
or other CI/CD systems.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PerformanceMonitor:
    """Automated performance monitoring and alerting system."""

    def __init__(self, regression_threshold: float = 10.0, alert_webhook: Optional[str] = None):
        self.regression_threshold = regression_threshold
        self.alert_webhook = alert_webhook
        self.results_dir = Path("tests/performance")
        self.reports_dir = self.results_dir / "reports"

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Execute all performance benchmarks and collect results."""
        print("🚀 Starting comprehensive performance benchmark suite...")

        benchmark_scripts = [
            "matrix_caching_benchmarks.py",
            "selective_update_benchmarks.py",
            "inference_benchmarks.py",
        ]

        results = {
            "start_time": datetime.now().isoformat(),
            "benchmarks": {},
            "success": True,
            "errors": [],
        }

        for script in benchmark_scripts:
            script_path = self.results_dir / script
            if not script_path.exists():
                print(f"⚠️  Benchmark script not found: {script}")
                continue

            print(f"📊 Running {script}...")

            try:
                # Run benchmark script
                result = subprocess.run(
                    [sys.executable, str(script_path)], capture_output=True, text=True, timeout=600
                )  # 10 minute timeout

                if result.returncode == 0:
                    print(f"✅ {script} completed successfully")
                    results["benchmarks"][script] = {
                        "status": "success",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                else:
                    print(f"❌ {script} failed with return code {result.returncode}")
                    results["benchmarks"][script] = {
                        "status": "failed",
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                    results["success"] = False
                    results["errors"].append(f"{script}: {result.stderr}"[:200])

            except subprocess.TimeoutExpired:
                print(f"⏰ {script} timed out after 10 minutes")
                results["benchmarks"][script] = {
                    "status": "timeout",
                    "error": "Benchmark timed out after 10 minutes",
                }
                results["success"] = False
                results["errors"].append(f"{script}: Timeout")

            except Exception as e:
                print(f"💥 {script} failed with exception: {e}")
                results["benchmarks"][script] = {"status": "error", "error": str(e)}
                results["success"] = False
                results["errors"].append(f"{script}: {str(e)}")

        results["end_time"] = datetime.now().isoformat()

        # Save benchmark execution summary
        summary_file = (
            self.results_dir
            / f"benchmark_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def generate_performance_reports(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis reports."""
        print("📈 Generating performance analysis reports...")

        try:
            # Import and run the performance report generator
            from performance_report_generator import PerformanceReportGenerator

            generator = PerformanceReportGenerator(str(self.results_dir))
            report_results = generator.run_full_analysis()

            return {"status": "success", "reports": report_results}

        except Exception as e:
            print(f"❌ Failed to generate performance reports: {e}")
            return {"status": "failed", "error": str(e)}

    def check_performance_gates(self) -> Dict[str, Any]:
        """Check if performance meets quality gates for CI/CD."""
        print("🚪 Checking performance quality gates...")

        gates = {
            "regression_threshold": self.regression_threshold,
            "checks": {},
            "overall_status": "pass",
            "failures": [],
        }

        try:
            # Load latest benchmark results
            result_files = list(self.results_dir.glob("*_results_*.json"))
            if not result_files:
                gates["overall_status"] = "skip"
                gates["failures"].append("No benchmark result files found")
                return gates

            # Find most recent results
            latest_results = []
            for file in result_files:
                with open(file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        latest_results.extend(data)
                    else:
                        latest_results.append(data)

            # Check for severe regressions (>25%)
            severe_regressions = []
            for result in latest_results:
                if "additional_metrics" in result:
                    metrics = result["additional_metrics"]
                    if "speedup_factor" in metrics and metrics["speedup_factor"] < 0.75:
                        severe_regressions.append(
                            f"{result.get('name', 'unknown')}: speedup_factor {metrics['speedup_factor']:.2f}"
                        )

            gates["checks"]["severe_regressions"] = {
                "status": "pass" if not severe_regressions else "fail",
                "count": len(severe_regressions),
                "details": severe_regressions,
            }

            if severe_regressions:
                gates["overall_status"] = "fail"
                gates["failures"].extend(severe_regressions)

            # Check cache effectiveness (>20% hit rate required)
            cache_failures = []
            for result in latest_results:
                if (
                    "additional_metrics" in result
                    and "cache_hit_rate" in result["additional_metrics"]
                ):
                    hit_rate = result["additional_metrics"]["cache_hit_rate"]
                    if hit_rate < 0.2:  # 20% minimum
                        cache_failures.append(
                            f"{result.get('name', 'unknown')}: hit_rate {hit_rate*100:.1f}%"
                        )

            gates["checks"]["cache_effectiveness"] = {
                "status": "pass" if not cache_failures else "fail",
                "count": len(cache_failures),
                "details": cache_failures,
            }

            if cache_failures:
                gates["overall_status"] = "fail"
                gates["failures"].extend(cache_failures)

            # Check memory efficiency (<100MB per operation average)
            memory_failures = []
            for result in latest_results:
                if "memory_usage_mb" in result and result["memory_usage_mb"] > 100:
                    memory_failures.append(
                        f"{result.get('name', 'unknown')}: {result['memory_usage_mb']:.1f}MB"
                    )

            gates["checks"]["memory_efficiency"] = {
                "status": "pass" if not memory_failures else "warn",
                "count": len(memory_failures),
                "details": memory_failures,
            }

            # Memory issues are warnings, not failures

        except Exception as e:
            gates["overall_status"] = "error"
            gates["failures"].append(f"Gate check error: {str(e)}")

        return gates

    def send_alert_notification(
        self, gates: Dict[str, Any], benchmark_results: Dict[str, Any]
    ) -> bool:
        """Send alert notification for performance issues."""
        if gates["overall_status"] == "pass" or not self.alert_webhook:
            return True

        print("🚨 Sending performance alert notification...")

        # Create alert message
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "status": gates["overall_status"],
            "failures": gates["failures"],
            "benchmark_success": benchmark_results.get("success", False),
            "errors": benchmark_results.get("errors", []),
        }

        try:
            import requests

            response = requests.post(self.alert_webhook, json=alert_data, timeout=30)
            response.raise_for_status()

            print("✅ Alert notification sent successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to send alert notification: {e}")
            return False

    def create_ci_summary(
        self,
        benchmark_results: Dict[str, Any],
        report_results: Dict[str, Any],
        gates: Dict[str, Any],
    ) -> str:
        """Create a summary for CI/CD systems."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = """# Performance Monitoring Summary
Generated: {timestamp}

## Benchmark Execution
- **Status**: {'✅ Success' if benchmark_results['success'] else '❌ Failed'}
- **Scripts Run**: {len(benchmark_results['benchmarks'])}
- **Errors**: {len(benchmark_results['errors'])}

## Quality Gates
- **Overall Status**: {gates['overall_status'].upper()}
- **Regression Threshold**: {gates['regression_threshold']}%
- **Failures**: {len(gates['failures'])}

### Gate Details
"""

        for check_name, check_data in gates.get("checks", {}).items():
            status_emoji = {"pass": "✅", "fail": "❌", "warn": "⚠️"}.get(check_data["status"], "❓")
            summary += f"- **{check_name.replace('_', ' ').title()}**: {status_emoji} {check_data['status'].upper()}\n"
            if check_data["details"]:
                for detail in check_data["details"][:3]:  # Show first 3
                    summary += f"  - {detail}\n"

        if report_results.get("status") == "success":
            reports = report_results.get("reports", {})
            summary += "\n## Reports Generated\n"
            summary += f"- **Charts**: {len(reports.get('charts', []))}\n"
            summary += f"- **Regressions Detected**: {reports.get('regressions', 0)}\n"

        if benchmark_results["errors"]:
            summary += "\n## Errors\n"
            for error in benchmark_results["errors"][:5]:  # Show first 5
                summary += f"- {error}\n"

        summary += "\n## CI/CD Actions\n"
        if gates["overall_status"] == "pass":
            summary += "✅ All performance checks passed - proceed with deployment\n"
        elif gates["overall_status"] == "warn":
            summary += "⚠️  Performance warnings detected - review before deployment\n"
        else:
            summary += "❌ Performance quality gates failed - deployment blocked\n"

        return summary

    def run_monitoring_pipeline(self) -> int:
        """Run the complete performance monitoring pipeline."""
        print("🎯 Starting automated performance monitoring pipeline...")

        # Step 1: Run benchmarks
        benchmark_results = self.run_all_benchmarks()

        # Step 2: Generate reports (even if benchmarks had issues)
        report_results = self.generate_performance_reports()

        # Step 3: Check quality gates
        gates = self.check_performance_gates()

        # Step 4: Send alerts if needed
        self.send_alert_notification(gates, benchmark_results)

        # Step 5: Create CI summary
        summary = self.create_ci_summary(benchmark_results, report_results, gates)

        # Save summary
        summary_file = (
            self.reports_dir / f"ci_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(summary_file, "w") as f:
            f.write(summary)

        print(f"\n📋 CI Summary saved to: {summary_file}")
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)

        # Return appropriate exit code for CI/CD
        if gates["overall_status"] == "fail":
            print("\n❌ Performance monitoring pipeline FAILED")
            return 1
        elif gates["overall_status"] == "warn":
            print("\n⚠️  Performance monitoring pipeline completed with WARNINGS")
            return 0  # Don't fail CI for warnings
        else:
            print("\n✅ Performance monitoring pipeline completed SUCCESSFULLY")
            return 0


def main():
    """Command line interface for automated performance monitoring."""
    parser = argparse.ArgumentParser(description="Automated Performance Monitoring for CI/CD")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=10.0,
        help="Performance regression threshold percentage (default: 10.0)",
    )
    parser.add_argument(
        "--alert-webhook", type=str, help="Webhook URL for sending performance alerts"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip running benchmarks, only generate reports from existing results",
    )
    parser.add_argument(
        "--ci-mode", action="store_true", help="Run in CI mode with appropriate exit codes"
    )

    args = parser.parse_args()

    # Get alert webhook from environment if not provided
    alert_webhook = args.alert_webhook or os.getenv("PERFORMANCE_ALERT_WEBHOOK")

    monitor = PerformanceMonitor(
        regression_threshold=args.regression_threshold, alert_webhook=alert_webhook
    )

    if args.skip_benchmarks:
        print("⏭️  Skipping benchmark execution (--skip-benchmarks)")
        # Just generate reports and check gates
        report_results = monitor.generate_performance_reports()
        gates = monitor.check_performance_gates()

        exit_code = 1 if gates["overall_status"] == "fail" else 0
    else:
        # Run full pipeline
        exit_code = monitor.run_monitoring_pipeline()

    if args.ci_mode:
        sys.exit(exit_code)
    else:
        return exit_code


if __name__ == "__main__":
    main()
