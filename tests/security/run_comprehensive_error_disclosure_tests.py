"""
Comprehensive Error Handling Information Disclosure Test Runner

This script runs all error handling and information disclosure tests
and generates a comprehensive security report for the FreeAgentics platform.

Test Coverage:
1. General error handling information disclosure
2. Authentication-specific error disclosure
3. API security response validation
4. Production hardening validation

Usage:
    python run_comprehensive_error_disclosure_tests.py [--output-format json|html|both]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add the project root to the path
sys.path.insert(0, "/home/green/FreeAgentics")

from fastapi.testclient import TestClient
from test_api_security_responses import APISecurityResponseTester
from test_authentication_error_disclosure import AuthenticationErrorTester

# Import our test classes
from test_error_handling_information_disclosure import ErrorHandlingTester
from test_production_hardening_validation import ProductionHardeningTester

from api.main import app


class ComprehensiveErrorDisclosureTestRunner:
    """Comprehensive test runner for all error disclosure tests."""

    def __init__(self):
        self.client = TestClient(app)
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all error disclosure test suites."""
        print("=" * 80)
        print("COMPREHENSIVE ERROR HANDLING INFORMATION DISCLOSURE TESTING")
        print("=" * 80)
        print(
            f"Starting comprehensive security testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print()

        self.start_time = time.time()

        # Initialize all testers
        testers = {
            "error_handling": ErrorHandlingTester(self.client),
            "authentication_errors": AuthenticationErrorTester(self.client),
            "api_security_responses": APISecurityResponseTester(self.client),
            "production_hardening": ProductionHardeningTester(self.client),
        }

        # Run each test suite
        for test_name, tester in testers.items():
            print(f"Running {test_name.replace('_', ' ').title()} Tests...")
            print("-" * 60)

            try:
                if hasattr(tester, "run_all_tests"):
                    result = tester.run_all_tests()
                elif hasattr(tester, f"run_all_{test_name}_tests"):
                    method = getattr(tester, f"run_all_{test_name}_tests")
                    result = method()
                else:
                    # For testers with different method names
                    if test_name == "authentication_errors":
                        result = tester.run_all_authentication_tests()
                    elif test_name == "api_security_responses":
                        result = tester.run_all_api_security_tests()
                    elif test_name == "production_hardening":
                        result = tester.run_all_production_hardening_tests()
                    else:
                        result = {"error": f"No runner method found for {test_name}"}

                self.test_results[test_name] = result

                # Print summary for this test suite
                if "summary" in result:
                    summary = result["summary"]
                    print(f"  Tests: {summary.get('total_tests', 0)}")
                    print(f"  Passed: {summary.get('passed_tests', 0)}")
                    print(f"  Failed: {summary.get('failed_tests', 0)}")
                    print(f"  Pass Rate: {summary.get('pass_rate', 0):.1f}%")

                    if "critical_failures" in summary:
                        print(f"  Critical Issues: {summary.get('critical_failures', 0)}")
                    if "high_failures" in summary:
                        print(f"  High Issues: {summary.get('high_failures', 0)}")

                print()

            except Exception as e:
                print(f"ERROR running {test_name}: {str(e)}")
                self.test_results[test_name] = {
                    "error": str(e),
                    "summary": {
                        "total_tests": 0,
                        "passed_tests": 0,
                        "failed_tests": 1,
                        "pass_rate": 0,
                    },
                }
                print()

        self.end_time = time.time()

        # Generate comprehensive report
        return self._generate_comprehensive_report()

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        critical_issues = 0
        high_issues = 0
        medium_issues = 0

        all_recommendations = []
        test_suite_summaries = {}

        for test_name, result in self.test_results.items():
            if "summary" in result:
                summary = result["summary"]
                total_tests += summary.get("total_tests", 0)
                total_passed += summary.get("passed_tests", 0)
                total_failed += summary.get("failed_tests", 0)

                # Count issues by severity
                critical_issues += summary.get("critical_failures", 0) + summary.get(
                    "critical_findings", 0
                )
                high_issues += summary.get("high_failures", 0) + summary.get("high_findings", 0)
                medium_issues += summary.get("medium_failures", 0) + summary.get(
                    "medium_findings", 0
                )

                # Collect recommendations
                if "recommendations" in result:
                    all_recommendations.extend(result["recommendations"])

                test_suite_summaries[test_name] = summary

        # Remove duplicate recommendations
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        # Calculate security scores
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Security score based on pass rate and issue severity
        security_score = overall_pass_rate
        if critical_issues > 0:
            security_score = max(0, security_score - critical_issues * 20)
        if high_issues > 0:
            security_score = max(0, security_score - high_issues * 10)
        if medium_issues > 0:
            security_score = max(0, security_score - medium_issues * 5)

        # Determine overall security status
        if critical_issues > 0:
            security_status = "CRITICAL - IMMEDIATE ACTION REQUIRED"
            security_level = "CRITICAL"
        elif high_issues > 5 or security_score < 70:
            security_status = "HIGH RISK - SIGNIFICANT ISSUES DETECTED"
            security_level = "HIGH"
        elif high_issues > 0 or medium_issues > 10 or security_score < 85:
            security_status = "MEDIUM RISK - IMPROVEMENTS NEEDED"
            security_level = "MEDIUM"
        elif security_score >= 95:
            security_status = "EXCELLENT - STRONG SECURITY POSTURE"
            security_level = "EXCELLENT"
        else:
            security_status = "GOOD - MINOR IMPROVEMENTS RECOMMENDED"
            security_level = "GOOD"

        # Check production readiness
        production_ready = critical_issues == 0 and high_issues <= 2 and security_score >= 80

        comprehensive_report = {
            "metadata": {
                "test_execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_duration_seconds": (
                    self.end_time - self.start_time if self.end_time and self.start_time else 0
                ),
                "platform": "FreeAgentics",
                "test_version": "1.0.0",
                "test_categories": list(self.test_results.keys()),
            },
            "executive_summary": {
                "security_status": security_status,
                "security_level": security_level,
                "security_score": round(security_score, 1),
                "production_ready": production_ready,
                "immediate_action_required": critical_issues > 0,
                "total_issues": critical_issues + high_issues + medium_issues,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
                "medium_issues": medium_issues,
            },
            "overall_statistics": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_pass_rate": round(overall_pass_rate, 1),
                "test_suites_run": len(self.test_results),
                "test_suites_with_failures": len(
                    [
                        r
                        for r in self.test_results.values()
                        if r.get("summary", {}).get("failed_tests", 0) > 0
                    ]
                ),
            },
            "test_suite_results": test_suite_summaries,
            "detailed_results": self.test_results,
            "security_recommendations": {
                "immediate_actions": self._get_immediate_actions(critical_issues, high_issues),
                "all_recommendations": unique_recommendations[:20],  # Limit to top 20
                "priority_recommendations": self._prioritize_recommendations(
                    unique_recommendations
                ),
            },
            "compliance_assessment": self._assess_compliance(),
            "risk_assessment": self._assess_risk_levels(
                critical_issues, high_issues, medium_issues
            ),
        }

        return comprehensive_report

    def _get_immediate_actions(self, critical_issues: int, high_issues: int) -> List[str]:
        """Get immediate actions based on issue severity."""
        immediate_actions = []

        if critical_issues > 0:
            immediate_actions.extend(
                [
                    "STOP: Do not deploy to production until critical issues are resolved",
                    "Review and fix all critical security vulnerabilities immediately",
                    "Implement proper error handling to prevent information disclosure",
                    "Ensure debug mode is disabled in production environment",
                ]
            )

        if high_issues > 5:
            immediate_actions.extend(
                [
                    "Review and address high-severity security issues before production deployment",
                    "Implement comprehensive security headers",
                    "Review authentication and authorization mechanisms",
                ]
            )

        if not immediate_actions:
            immediate_actions.append(
                "Continue monitoring and addressing remaining security improvements"
            )

        return immediate_actions

    def _prioritize_recommendations(self, recommendations: List[str]) -> Dict[str, List[str]]:
        """Prioritize recommendations by category."""
        categorized = {"critical": [], "high": [], "medium": [], "low": []}

        for rec in recommendations:
            rec_lower = rec.lower()

            if any(
                critical_term in rec_lower
                for critical_term in [
                    "critical",
                    "urgent",
                    "immediate",
                    "stop",
                    "disable debug",
                ]
            ):
                categorized["critical"].append(rec)
            elif any(
                high_term in rec_lower
                for high_term in [
                    "security header",
                    "authentication",
                    "authorization",
                    "sanitize",
                    "validate",
                ]
            ):
                categorized["high"].append(rec)
            elif any(
                medium_term in rec_lower
                for medium_term in [
                    "configure",
                    "implement",
                    "review",
                    "update",
                ]
            ):
                categorized["medium"].append(rec)
            else:
                categorized["low"].append(rec)

        # Limit each category
        for category in categorized:
            categorized[category] = categorized[category][:10]

        return categorized

    def _assess_compliance(self) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        compliance_results = {}

        # Check OWASP Top 10 compliance
        owasp_issues = []

        for test_name, result in self.test_results.items():
            if "detailed_results" in result:
                for test_result in result["detailed_results"]:
                    if not test_result.get("passed", True):
                        error_details = test_result.get("error_details", [])
                        for error in error_details:
                            category = error.get("category", "")
                            if "injection" in category.lower():
                                owasp_issues.append("A03:2021 – Injection")
                            elif "authentication" in category.lower():
                                owasp_issues.append(
                                    "A07:2021 – Identification and Authentication Failures"
                                )
                            elif "disclosure" in category.lower():
                                owasp_issues.append(
                                    "A09:2021 – Security Logging and Monitoring Failures"
                                )

        compliance_results["owasp_top_10"] = {
            "issues_found": list(set(owasp_issues)),
            "compliance_level": "Non-Compliant" if owasp_issues else "Compliant",
        }

        # Add other compliance frameworks as needed
        compliance_results["general_security"] = {
            "security_headers": "Needs Review",
            "error_handling": "Needs Review",
            "authentication": "Needs Review",
        }

        return compliance_results

    def _assess_risk_levels(self, critical: int, high: int, medium: int) -> Dict[str, Any]:
        """Assess risk levels for different attack vectors."""
        risk_assessment = {
            "information_disclosure": {
                "level": "HIGH" if critical > 0 else "MEDIUM" if high > 2 else "LOW",
                "description": "Risk of sensitive information being disclosed through error messages",
            },
            "authentication_bypass": {
                "level": (
                    "HIGH"
                    if any("authentication" in str(result) for result in self.test_results.values())
                    else "LOW"
                ),
                "description": "Risk of authentication mechanisms being bypassed",
            },
            "injection_attacks": {
                "level": (
                    "MEDIUM"
                    if any("injection" in str(result) for result in self.test_results.values())
                    else "LOW"
                ),
                "description": "Risk of injection attacks through error handling",
            },
            "denial_of_service": {
                "level": "LOW",
                "description": "Risk of DoS through error handling manipulation",
            },
        }

        return risk_assessment

    def save_report(
        self,
        report: Dict[str, Any],
        output_format: str = "both",
        output_dir: str = None,
    ) -> List[str]:
        """Save the comprehensive report in specified format(s)."""
        if output_dir is None:
            output_dir = "/home/green/FreeAgentics/tests/security"

        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_files = []

        # Save JSON report
        if output_format in ["json", "both"]:
            json_file = os.path.join(
                output_dir,
                f"comprehensive_error_disclosure_report_{timestamp}.json",
            )
            try:
                with open(json_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                saved_files.append(json_file)
                print(f"JSON report saved to: {json_file}")
            except Exception as e:
                print(f"Error saving JSON report: {e}")

        # Save HTML report
        if output_format in ["html", "both"]:
            html_file = os.path.join(
                output_dir,
                f"comprehensive_error_disclosure_report_{timestamp}.html",
            )
            try:
                html_content = self._generate_html_report(report)
                with open(html_file, "w") as f:
                    f.write(html_content)
                saved_files.append(html_file)
                print(f"HTML report saved to: {html_file}")
            except Exception as e:
                print(f"Error saving HTML report: {e}")

        return saved_files

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate an HTML version of the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreeAgentics Security Assessment Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .executive-summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
        }}
        .status-critical {{ background: #e74c3c; color: white; }}
        .status-high {{ background: #f39c12; color: white; }}
        .status-medium {{ background: #f1c40f; color: #2c3e50; }}
        .status-good {{ background: #27ae60; color: white; }}
        .status-excellent {{ background: #16a085; color: white; }}
        .status-badge {{
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            background: white;
            padding: 15px;
            margin: 10px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 120px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        .recommendations h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .test-results {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .test-suite {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
        }}
        .test-suite h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FreeAgentics Security Assessment Report</h1>
            <p>Error Handling & Information Disclosure Security Analysis</p>
            <p>Generated on {report['metadata']['test_execution_time']}</p>
        </div>

        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="status-badge status-{report['executive_summary']['security_level'].lower()}">
                {report['executive_summary']['security_status']}
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{report['executive_summary']['security_score']}</div>
                    <div class="metric-label">Security Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['overall_statistics']['total_tests']}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['overall_statistics']['overall_pass_rate']}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['executive_summary']['critical_issues']}</div>
                    <div class="metric-label">Critical Issues</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['executive_summary']['high_issues']}</div>
                    <div class="metric-label">High Issues</div>
                </div>
            </div>
        </div>
        """

        # Add immediate actions if any
        if report["security_recommendations"]["immediate_actions"]:
            html += """
            <div class="recommendations">
                <h3>🚨 Immediate Actions Required</h3>
                <ul>
            """
            for action in report["security_recommendations"]["immediate_actions"]:
                html += f"<li>{action}</li>"
            html += "</ul></div>"

        # Add test suite results
        html += """
        <div class="section">
            <h2>Test Suite Results</h2>
            <div class="test-results">
        """

        for suite_name, suite_data in report["test_suite_results"].items():
            pass_rate = suite_data.get("pass_rate", 0)
            status_class = "pass" if pass_rate >= 90 else "fail"

            html += f"""
                <div class="test-suite">
                    <h3>{suite_name.replace('_', ' ').title()}</h3>
                    <p><strong>Pass Rate:</strong> <span class="{status_class}">{pass_rate:.1f}%</span></p>
                    <p><strong>Tests:</strong> {suite_data.get('total_tests', 0)} total, {suite_data.get('passed_tests', 0)} passed, {suite_data.get('failed_tests', 0)} failed</p>
                </div>
            """

        html += "</div></div>"

        # Add top recommendations
        if report["security_recommendations"]["all_recommendations"]:
            html += """
            <div class="section">
                <h2>Top Security Recommendations</h2>
                <ul>
            """
            for rec in report["security_recommendations"]["all_recommendations"][:10]:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"

        # Add footer
        html += f"""
        <div class="footer">
            <p>Report generated by FreeAgentics Security Testing Suite</p>
            <p>Test duration: {report['metadata']['test_duration_seconds']:.1f} seconds</p>
        </div>
    </div>
</body>
</html>
        """

        return html

    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the test results."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ERROR DISCLOSURE SECURITY ASSESSMENT SUMMARY")
        print("=" * 80)

        exec_summary = report["executive_summary"]
        stats = report["overall_statistics"]

        print(f"Security Status: {exec_summary['security_status']}")
        print(f"Security Score: {exec_summary['security_score']}/100")
        print(f"Production Ready: {'✅ YES' if exec_summary['production_ready'] else '❌ NO'}")
        print()

        print("Test Statistics:")
        print(f"  Total Tests Run: {stats['total_tests']}")
        print(f"  Tests Passed: {stats['total_passed']}")
        print(f"  Tests Failed: {stats['total_failed']}")
        print(f"  Overall Pass Rate: {stats['overall_pass_rate']:.1f}%")
        print()

        print("Issues by Severity:")
        print(f"  🔴 Critical: {exec_summary['critical_issues']}")
        print(f"  🟠 High: {exec_summary['high_issues']}")
        print(f"  🟡 Medium: {exec_summary['medium_issues']}")
        print()

        if exec_summary["immediate_action_required"]:
            print("🚨 IMMEDIATE ACTION REQUIRED:")
            for action in report["security_recommendations"]["immediate_actions"]:
                print(f"  • {action}")
            print()

        if report["security_recommendations"]["all_recommendations"]:
            print("Top Recommendations:")
            for i, rec in enumerate(
                report["security_recommendations"]["all_recommendations"][:5],
                1,
            ):
                print(f"  {i}. {rec}")


def main():
    """Main function to run comprehensive error disclosure tests."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive error handling disclosure tests"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format for the report",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output during testing",
    )

    args = parser.parse_args()

    # Initialize and run tests
    runner = ComprehensiveErrorDisclosureTestRunner()

    if args.quiet:
        # Redirect stdout temporarily
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            report = runner.run_all_tests()
    else:
        report = runner.run_all_tests()

    # Print summary
    runner.print_summary(report)

    # Save reports
    saved_files = runner.save_report(report, args.output_format, args.output_dir)

    if saved_files:
        print("\nReports saved:")
        for file_path in saved_files:
            print(f"  {file_path}")

    # Exit with appropriate code
    exec_summary = report["executive_summary"]
    if exec_summary["critical_issues"] > 0:
        print("\n❌ CRITICAL ISSUES DETECTED - Application not ready for production")
        sys.exit(1)
    elif exec_summary["high_issues"] > 5:
        print("\n⚠️  HIGH RISK ISSUES DETECTED - Significant security improvements needed")
        sys.exit(2)
    elif not exec_summary["production_ready"]:
        print("\n⚠️  APPLICATION NOT PRODUCTION READY - Address security issues before deployment")
        sys.exit(3)
    else:
        print("\n✅ SECURITY ASSESSMENT COMPLETED - Application has acceptable security posture")
        sys.exit(0)


if __name__ == "__main__":
    main()
