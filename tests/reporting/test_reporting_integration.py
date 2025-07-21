"""Integration script for comprehensive test reporting system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.reporting.coverage_analyzer import CoverageAnalyzer
from tests.reporting.dashboard_generator import DashboardGenerator
from tests.reporting.report_archival_system import ReportArchivalSystem
from tests.reporting.test_metrics_collector import MetricsCollector


class ReportingIntegration:
    """Comprehensive test reporting integration."""

    def __init__(self, output_dir: str = "tests/reporting"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize components
        self.coverage_analyzer = CoverageAnalyzer()
        self.metrics_collector = MetricsCollector()
        self.dashboard_generator = DashboardGenerator()
        self.archival_system = ReportArchivalSystem()

        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / "integration.log"),
                logging.StreamHandler(),
            ],
        )

    def run_comprehensive_reporting(self, test_run_id: str = None) -> Dict[str, Any]:
        """Run comprehensive test reporting workflow."""
        if test_run_id is None:
            test_run_id = f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting comprehensive reporting for run: {test_run_id}")

        results = {
            "test_run_id": test_run_id,
            "start_time": datetime.now().isoformat(),
            "reports_generated": [],
            "errors": [],
        }

        try:
            # Step 1: Analyze coverage
            self.logger.info("Analyzing test coverage...")
            coverage_report = self.coverage_analyzer.analyze_coverage(test_run_id)

            # Generate coverage reports
            html_report = self.coverage_analyzer.generate_coverage_report_html()
            json_report = self.coverage_analyzer.export_coverage_json()

            results["reports_generated"].extend([html_report, json_report])
            results["coverage_summary"] = {
                "total_coverage": coverage_report.total_coverage,
                "total_statements": coverage_report.total_statements,
                "total_missing": coverage_report.total_missing,
            }

            # Step 2: Generate metrics reports
            self.logger.info("Generating test metrics reports...")
            metrics_html = self.metrics_collector.generate_metrics_report()
            metrics_json = self.metrics_collector.export_metrics_json()

            results["reports_generated"].extend([metrics_html, metrics_json])

            # Get test quality insights
            flaky_tests = self.metrics_collector.get_flaky_tests()
            slow_tests = self.metrics_collector.get_slow_tests()

            results["quality_insights"] = {
                "flaky_tests_count": len(flaky_tests),
                "slow_tests_count": len(slow_tests),
                "top_flaky_tests": flaky_tests[:5],
                "top_slow_tests": slow_tests[:5],
            }

            # Step 3: Generate comprehensive dashboard
            self.logger.info("Generating comprehensive dashboard...")
            dashboard_html = self.dashboard_generator.generate_dashboard()
            dashboard_json = self.dashboard_generator.generate_json_export()

            results["reports_generated"].extend([dashboard_html, dashboard_json])

            # Step 4: Run archival process
            self.logger.info("Running report archival process...")
            archival_results = self.archival_system.run_archival_process()

            results["archival_summary"] = {
                "files_archived": archival_results["files_archived"],
                "files_deleted": archival_results["files_deleted"],
                "space_freed": archival_results["space_freed"],
            }

            # Step 5: Generate integration summary
            self.logger.info("Generating integration summary...")
            summary_path = self.generate_integration_summary(results)
            results["reports_generated"].append(summary_path)

            # Step 6: Perform quality checks
            quality_score = self.calculate_overall_quality_score(
                coverage_report, flaky_tests, slow_tests
            )
            results["quality_score"] = quality_score

            # Step 7: Generate recommendations
            recommendations = self.generate_recommendations(
                coverage_report, flaky_tests, slow_tests
            )
            results["recommendations"] = recommendations

        except Exception as e:
            error_msg = f"Error during comprehensive reporting: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        results["end_time"] = datetime.now().isoformat()
        results["duration"] = (
            datetime.fromisoformat(results["end_time"])
            - datetime.fromisoformat(results["start_time"])
        ).total_seconds()

        self.logger.info(
            f"Comprehensive reporting completed in {results['duration']:.2f} seconds"
        )
        return results

    def calculate_overall_quality_score(
        self, coverage_report, flaky_tests, slow_tests
    ) -> Dict[str, float]:
        """Calculate overall test quality score."""
        scores = {}

        # Coverage score (0-100)
        scores["coverage_score"] = coverage_report.total_coverage

        # Reliability score (based on flaky tests)
        total_tests = len(coverage_report.files)  # Approximation
        if total_tests > 0:
            flaky_rate = len(flaky_tests) / total_tests
            scores["reliability_score"] = max(0, 100 - (flaky_rate * 100))
        else:
            scores["reliability_score"] = 100

        # Performance score (based on slow tests)
        if total_tests > 0:
            slow_rate = len(slow_tests) / total_tests
            scores["performance_score"] = max(0, 100 - (slow_rate * 50))
        else:
            scores["performance_score"] = 100

        # Overall score (weighted average)
        weights = {
            "coverage_score": 0.4,
            "reliability_score": 0.4,
            "performance_score": 0.2,
        }

        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores["overall_score"] = overall_score

        return scores

    def generate_recommendations(
        self, coverage_report, flaky_tests, slow_tests
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on test analysis."""
        recommendations = []

        # Coverage recommendations
        if coverage_report.total_coverage < 80:
            recommendations.append(
                {
                    "type": "coverage",
                    "priority": "high",
                    "title": "Improve Test Coverage",
                    "description": f"Current coverage is {coverage_report.total_coverage:.1f}%. Target: 80%+",
                    "action": "Add tests for uncovered code paths",
                }
            )

        # Zero coverage files
        zero_coverage_files = self.coverage_analyzer.get_zero_coverage_files()
        if zero_coverage_files:
            recommendations.append(
                {
                    "type": "coverage",
                    "priority": "high",
                    "title": "Address Zero Coverage Files",
                    "description": f"{len(zero_coverage_files)} files have zero test coverage",
                    "action": "Create tests for these files or remove if unnecessary",
                }
            )

        # Flaky tests recommendations
        if flaky_tests:
            recommendations.append(
                {
                    "type": "reliability",
                    "priority": "medium",
                    "title": "Fix Flaky Tests",
                    "description": f"{len(flaky_tests)} tests are flaky",
                    "action": "Investigate and fix non-deterministic behavior",
                }
            )

        # Slow tests recommendations
        if slow_tests:
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "low",
                    "title": "Optimize Slow Tests",
                    "description": f"{len(slow_tests)} tests are slower than average",
                    "action": "Optimize test execution time or use appropriate markers",
                }
            )

        return recommendations

    def generate_integration_summary(self, results: Dict[str, Any]) -> str:
        """Generate integration summary report."""
        summary_path = self.output_dir / "integration_summary.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Reporting Integration Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e0e0e0; border-radius: 3px; }}
                .good {{ color: #4caf50; }}
                .warning {{ color: #ff9800; }}
                .critical {{ color: #f44336; }}
                .recommendation {{ padding: 10px; margin: 10px 0; border-left: 4px solid #2196f3; background-color: #f3f3f3; }}
                .high {{ border-left-color: #f44336; }}
                .medium {{ border-left-color: #ff9800; }}
                .low {{ border-left-color: #4caf50; }}
                ul {{ margin: 10px 0; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Test Reporting Integration Summary</h1>
                <p>Test Run ID: {results["test_run_id"]}</p>
                <p>Generated: {results["start_time"]}</p>
                <p>Duration: {results.get("duration", 0):.2f} seconds</p>
            </div>

            <div class="section">
                <h2>📊 Quality Score</h2>
        """

        if "quality_score" in results:
            for score_type, score in results["quality_score"].items():
                css_class = (
                    "good" if score >= 80 else "warning" if score >= 60 else "critical"
                )
                html_content += f"""
                <div class="metric {css_class}">
                    <strong>{score_type.replace("_", " ").title()}: {score:.1f}%</strong>
                </div>
                """

        html_content += """
            </div>

            <div class="section">
                <h2>📋 Reports Generated</h2>
                <ul>
        """

        for report in results["reports_generated"]:
            html_content += f"<li>{report}</li>"

        html_content += """
                </ul>
            </div>

            <div class="section">
                <h2>🔍 Quality Insights</h2>
        """

        if "quality_insights" in results:
            insights = results["quality_insights"]
            html_content += f"""
                <div class="metric">
                    <strong>Flaky Tests: {insights["flaky_tests_count"]}</strong>
                </div>
                <div class="metric">
                    <strong>Slow Tests: {insights["slow_tests_count"]}</strong>
                </div>
            """

        html_content += """
            </div>

            <div class="section">
                <h2>💡 Recommendations</h2>
        """

        if "recommendations" in results:
            for rec in results["recommendations"]:
                html_content += f"""
                <div class="recommendation {rec["priority"]}">
                    <h3>{rec["title"]}</h3>
                    <p><strong>Priority:</strong> {rec["priority"].upper()}</p>
                    <p><strong>Description:</strong> {rec["description"]}</p>
                    <p><strong>Action:</strong> {rec["action"]}</p>
                </div>
                """

        html_content += """
            </div>

            <div class="section">
                <h2>🗂️ Archival Summary</h2>
        """

        if "archival_summary" in results:
            archival = results["archival_summary"]
            html_content += f"""
                <div class="metric">
                    <strong>Files Archived: {archival["files_archived"]}</strong>
                </div>
                <div class="metric">
                    <strong>Files Deleted: {archival["files_deleted"]}</strong>
                </div>
                <div class="metric">
                    <strong>Space Freed: {archival["space_freed"]} bytes</strong>
                </div>
            """

        html_content += """
            </div>

            <div class="section">
                <h2>🔗 Quick Links</h2>
                <ul>
                    <li><a href="dashboard.html">Test Dashboard</a></li>
                    <li><a href="coverage_report.html">Coverage Report</a></li>
                    <li><a href="metrics_report.html">Metrics Report</a></li>
                    <li><a href="coverage_html/index.html">Detailed Coverage</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(summary_path, "w") as f:
            f.write(html_content)

        return str(summary_path)

    def run_health_check(self) -> Dict[str, Any]:
        """Run health check on reporting system."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "recommendations": [],
        }

        # Check database connectivity
        try:
            self.coverage_analyzer.analyze_coverage()
            health_status["components"]["coverage_analyzer"] = "healthy"
        except Exception as e:
            health_status["components"]["coverage_analyzer"] = f"error: {e}"
            health_status["overall_status"] = "degraded"

        try:
            self.metrics_collector.get_flaky_tests()
            health_status["components"]["metrics_collector"] = "healthy"
        except Exception as e:
            health_status["components"]["metrics_collector"] = f"error: {e}"
            health_status["overall_status"] = "degraded"

        try:
            self.dashboard_generator.generate_json_export()
            health_status["components"]["dashboard_generator"] = "healthy"
        except Exception as e:
            health_status["components"]["dashboard_generator"] = f"error: {e}"
            health_status["overall_status"] = "degraded"

        try:
            self.archival_system.get_archival_status()
            health_status["components"]["archival_system"] = "healthy"
        except Exception as e:
            health_status["components"]["archival_system"] = f"error: {e}"
            health_status["overall_status"] = "degraded"

        return health_status


def main():
    """Main function for running integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test reporting integration")
    parser.add_argument("--run-id", help="Test run ID")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument(
        "--output-dir", default="tests/reporting", help="Output directory"
    )

    args = parser.parse_args()

    integration = ReportingIntegration(args.output_dir)

    if args.health_check:
        health_status = integration.run_health_check()
        print("Health Check Results:")
        print(f"Overall Status: {health_status['overall_status']}")
        print("Components:")
        for component, status in health_status["components"].items():
            print(f"  {component}: {status}")
        return

    # Run comprehensive reporting
    results = integration.run_comprehensive_reporting(args.run_id)

    print("\\n" + "=" * 60)
    print("TEST REPORTING INTEGRATION RESULTS")
    print("=" * 60)
    print(f"Test Run ID: {results['test_run_id']}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    print(f"Reports Generated: {len(results['reports_generated'])}")

    if "quality_score" in results:
        print("\\nQuality Scores:")
        for score_type, score in results["quality_score"].items():
            print(f"  {score_type.replace('_', ' ').title()}: {score:.1f}%")

    if "coverage_summary" in results:
        print("\\nCoverage Summary:")
        coverage = results["coverage_summary"]
        print(f"  Total Coverage: {coverage['total_coverage']:.1f}%")
        print(f"  Total Statements: {coverage['total_statements']}")
        print(f"  Missing Statements: {coverage['total_missing']}")

    if "quality_insights" in results:
        print("\\nQuality Insights:")
        insights = results["quality_insights"]
        print(f"  Flaky Tests: {insights['flaky_tests_count']}")
        print(f"  Slow Tests: {insights['slow_tests_count']}")

    if "recommendations" in results:
        print(f"\\nRecommendations ({len(results['recommendations'])}):")
        for rec in results["recommendations"]:
            print(f"  [{rec['priority'].upper()}] {rec['title']}")
            print(f"    {rec['description']}")

    if results["errors"]:
        print(f"\\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")

    print(f"\\nReports saved to: {integration.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
