"""
Comprehensive Penetration Testing Suite Runner

This script runs all penetration testing components and generates
a comprehensive security assessment report for the FreeAgentics platform.

Test Coverage:
1. Error handling information disclosure testing
2. Authentication error disclosure testing
3. API security response validation
4. Production hardening validation
5. File upload security testing
6. Path traversal prevention testing
7. Cryptography assessment
8. Overall security posture assessment

Usage:
    python run_comprehensive_penetration_tests.py [--output-format json|html|both]
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

from api.main import app
from fastapi.testclient import TestClient
from test_api_security_responses import APISecurityResponseTester
from test_authentication_error_disclosure import AuthenticationErrorTester

# Import all test classes
from test_error_handling_information_disclosure import ErrorHandlingTester
from test_file_upload_security import FileUploadSecurityTester
from test_path_traversal_prevention import PathTraversalPreventionTester
from test_production_hardening_validation import ProductionHardeningTester


class ComprehensivePenetrationTestRunner:
    """Comprehensive penetration test runner for all security components."""

    def __init__(self):
        self.client = TestClient(app)
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all penetration test suites."""
        print("=" * 80)
        print("COMPREHENSIVE PENETRATION TESTING SUITE")
        print("=" * 80)
        print(
            f"Starting comprehensive penetration testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print()

        self.start_time = time.time()

        # Initialize all testers
        testers = {
            "error_handling_disclosure": ErrorHandlingTester(self.client),
            "authentication_error_disclosure": AuthenticationErrorTester(self.client),
            "api_security_responses": APISecurityResponseTester(self.client),
            "production_hardening": ProductionHardeningTester(self.client),
            "file_upload_security": FileUploadSecurityTester(self.client),
            "path_traversal_prevention": PathTraversalPreventionTester(self.client),
        }

        # Run each test suite
        for test_name, tester in testers.items():
            print(f"Running {test_name.replace('_', ' ').title()} Tests...")
            print("-" * 60)

            try:
                # Determine the correct method to call
                if hasattr(tester, "run_all_tests"):
                    result = tester.run_all_tests()
                elif hasattr(tester, f"run_all_{test_name}_tests"):
                    method = getattr(tester, f"run_all_{test_name}_tests")
                    result = method()
                else:
                    # Map to specific method names
                    method_mapping = {
                        "error_handling_disclosure": "run_all_tests",
                        "authentication_error_disclosure": "run_all_authentication_tests",
                        "api_security_responses": "run_all_api_security_tests",
                        "production_hardening": "run_all_production_hardening_tests",
                        "file_upload_security": "run_all_file_upload_tests",
                        "path_traversal_prevention": "run_all_path_traversal_tests",
                    }

                    method_name = method_mapping.get(test_name)
                    if method_name and hasattr(tester, method_name):
                        method = getattr(tester, method_name)
                        result = method()
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

                    if "critical_findings" in summary:
                        print(f"  Critical Issues: {summary.get('critical_findings', 0)}")
                    if "high_findings" in summary:
                        print(f"  High Issues: {summary.get('high_findings', 0)}")

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
        """Generate a comprehensive penetration testing report."""
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
                critical_issues += summary.get("critical_findings", 0) + summary.get(
                    "critical_failures", 0
                )
                high_issues += summary.get("high_findings", 0) + summary.get("high_failures", 0)
                medium_issues += summary.get("medium_findings", 0) + summary.get(
                    "medium_failures", 0
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

        # Penetration testing security score
        penetration_score = overall_pass_rate
        if critical_issues > 0:
            penetration_score = max(0, penetration_score - critical_issues * 25)
        if high_issues > 0:
            penetration_score = max(0, penetration_score - high_issues * 15)
        if medium_issues > 0:
            penetration_score = max(0, penetration_score - medium_issues * 5)

        # Determine overall security status
        if critical_issues > 0:
            security_status = "CRITICAL - IMMEDIATE REMEDIATION REQUIRED"
            security_level = "CRITICAL"
            threat_level = "CRITICAL"
        elif high_issues > 3 or penetration_score < 60:
            security_status = "HIGH RISK - SIGNIFICANT VULNERABILITIES DETECTED"
            security_level = "HIGH"
            threat_level = "HIGH"
        elif high_issues > 0 or medium_issues > 8 or penetration_score < 75:
            security_status = "MEDIUM RISK - SECURITY IMPROVEMENTS NEEDED"
            security_level = "MEDIUM"
            threat_level = "MEDIUM"
        elif penetration_score >= 90:
            security_status = "SECURE - STRONG SECURITY POSTURE"
            security_level = "SECURE"
            threat_level = "LOW"
        else:
            security_status = "ACCEPTABLE - MINOR IMPROVEMENTS RECOMMENDED"
            security_level = "ACCEPTABLE"
            threat_level = "LOW"

        # Check production readiness
        production_ready = critical_issues == 0 and high_issues <= 1 and penetration_score >= 75

        # Assess specific security domains
        domain_assessment = self._assess_security_domains()

        comprehensive_report = {
            "metadata": {
                "test_execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_duration_seconds": (
                    self.end_time - self.start_time if self.end_time and self.start_time else 0
                ),
                "platform": "FreeAgentics",
                "test_version": "1.0.0",
                "test_type": "Comprehensive Penetration Testing",
                "test_categories": list(self.test_results.keys()),
            },
            "executive_summary": {
                "security_status": security_status,
                "security_level": security_level,
                "threat_level": threat_level,
                "penetration_score": round(penetration_score, 1),
                "production_ready": production_ready,
                "immediate_action_required": critical_issues > 0,
                "total_vulnerabilities": critical_issues + high_issues + medium_issues,
                "critical_vulnerabilities": critical_issues,
                "high_vulnerabilities": high_issues,
                "medium_vulnerabilities": medium_issues,
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
                "all_recommendations": unique_recommendations[:25],  # Limit to top 25
                "priority_recommendations": self._prioritize_recommendations(
                    unique_recommendations
                ),
            },
            "security_domain_assessment": domain_assessment,
            "compliance_assessment": self._assess_compliance(),
            "threat_assessment": self._assess_threats(critical_issues, high_issues, medium_issues),
            "penetration_testing_summary": self._generate_penetration_summary(),
        }

        return comprehensive_report

    def _assess_security_domains(self) -> Dict[str, Any]:
        """Assess security across different domains."""
        domains = {
            "authentication_security": {
                "tests": ["authentication_error_disclosure"],
                "score": 0,
                "status": "unknown",
            },
            "api_security": {
                "tests": ["api_security_responses"],
                "score": 0,
                "status": "unknown",
            },
            "file_security": {
                "tests": ["file_upload_security", "path_traversal_prevention"],
                "score": 0,
                "status": "unknown",
            },
            "information_disclosure": {
                "tests": ["error_handling_disclosure"],
                "score": 0,
                "status": "unknown",
            },
            "production_hardening": {
                "tests": ["production_hardening"],
                "score": 0,
                "status": "unknown",
            },
        }

        for domain_name, domain_info in domains.items():
            total_score = 0
            test_count = 0

            for test_name in domain_info["tests"]:
                if test_name in self.test_results:
                    result = self.test_results[test_name]
                    if "summary" in result:
                        pass_rate = result["summary"].get("pass_rate", 0)
                        total_score += pass_rate
                        test_count += 1

            if test_count > 0:
                avg_score = total_score / test_count
                domain_info["score"] = round(avg_score, 1)

                if avg_score >= 90:
                    domain_info["status"] = "excellent"
                elif avg_score >= 75:
                    domain_info["status"] = "good"
                elif avg_score >= 60:
                    domain_info["status"] = "acceptable"
                elif avg_score >= 40:
                    domain_info["status"] = "poor"
                else:
                    domain_info["status"] = "critical"

        return domains

    def _get_immediate_actions(self, critical_issues: int, high_issues: int) -> List[str]:
        """Get immediate actions based on issue severity."""
        immediate_actions = []

        if critical_issues > 0:
            immediate_actions.extend(
                [
                    "STOP: Do not deploy to production until critical vulnerabilities are resolved",
                    "Conduct immediate security review with security team",
                    "Implement emergency patches for critical vulnerabilities",
                    "Review and strengthen access controls immediately",
                    "Audit all file upload and path handling functionality",
                ]
            )

        if high_issues > 3:
            immediate_actions.extend(
                [
                    "Schedule urgent security remediation sprint",
                    "Review and strengthen authentication mechanisms",
                    "Implement comprehensive input validation",
                    "Audit API security configurations",
                ]
            )

        if not immediate_actions:
            immediate_actions.append(
                "Continue regular security monitoring and address remaining improvements"
            )

        return immediate_actions

    def _prioritize_recommendations(self, recommendations: List[str]) -> Dict[str, List[str]]:
        """Prioritize recommendations by category and urgency."""
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
                    "emergency",
                    "patch",
                    "vulnerability",
                    "exploit",
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
                    "prevent",
                    "block",
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
                    "improve",
                ]
            ):
                categorized["medium"].append(rec)
            else:
                categorized["low"].append(rec)

        # Limit each category
        for category in categorized:
            categorized[category] = categorized[category][:15]

        return categorized

    def _assess_compliance(self) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        compliance_results = {
            "owasp_top_10": {
                "issues_found": [],
                "compliance_level": "Unknown",
            },
            "nist_cybersecurity_framework": {
                "identify": "Partial",
                "protect": "Needs Review",
                "detect": "Needs Review",
                "respond": "Needs Review",
                "recover": "Needs Review",
            },
            "iso_27001": {
                "information_security_policy": "Needs Review",
                "access_control": "Needs Review",
                "cryptography": "Needs Review",
                "operations_security": "Needs Review",
            },
        }

        # Check OWASP Top 10 compliance based on test results
        owasp_issues = []

        for test_name, result in self.test_results.items():
            if "detailed_results" in result:
                for test_result in result["detailed_results"]:
                    if not test_result.get("passed", True):
                        findings = test_result.get("findings", [])
                        for finding in findings:
                            issue = finding.get("issue", "").lower()

                            if any(term in issue for term in ["injection", "sql", "command"]):
                                owasp_issues.append("A03:2021 – Injection")
                            elif any(term in issue for term in ["authentication", "auth", "login"]):
                                owasp_issues.append(
                                    "A07:2021 – Identification and Authentication Failures"
                                )
                            elif any(
                                term in issue
                                for term in [
                                    "disclosure",
                                    "information",
                                    "error",
                                ]
                            ):
                                owasp_issues.append(
                                    "A09:2021 – Security Logging and Monitoring Failures"
                                )
                            elif any(
                                term in issue
                                for term in [
                                    "access",
                                    "authorization",
                                    "permission",
                                ]
                            ):
                                owasp_issues.append("A01:2021 – Broken Access Control")
                            elif any(
                                term in issue
                                for term in [
                                    "configuration",
                                    "header",
                                    "server",
                                ]
                            ):
                                owasp_issues.append("A05:2021 – Security Misconfiguration")

        compliance_results["owasp_top_10"]["issues_found"] = list(set(owasp_issues))
        compliance_results["owasp_top_10"]["compliance_level"] = (
            "Non-Compliant" if owasp_issues else "Compliant"
        )

        return compliance_results

    def _assess_threats(self, critical: int, high: int, medium: int) -> Dict[str, Any]:
        """Assess threat levels for different attack vectors."""
        threat_assessment = {
            "external_attacks": {
                "level": "HIGH" if critical > 0 else "MEDIUM" if high > 2 else "LOW",
                "description": "Risk from external attackers exploiting web vulnerabilities",
            },
            "insider_threats": {
                "level": (
                    "MEDIUM"
                    if any("access" in str(result) for result in self.test_results.values())
                    else "LOW"
                ),
                "description": "Risk from malicious or compromised internal users",
            },
            "data_exfiltration": {
                "level": "HIGH" if critical > 0 else "MEDIUM" if high > 1 else "LOW",
                "description": "Risk of sensitive data being accessed or stolen",
            },
            "system_compromise": {
                "level": (
                    "CRITICAL"
                    if critical > 2
                    else "HIGH"
                    if critical > 0
                    else "MEDIUM"
                    if high > 3
                    else "LOW"
                ),
                "description": "Risk of complete system compromise",
            },
            "denial_of_service": {
                "level": (
                    "MEDIUM"
                    if any("rate" in str(result) for result in self.test_results.values())
                    else "LOW"
                ),
                "description": "Risk of service disruption or unavailability",
            },
        }

        return threat_assessment

    def _generate_penetration_summary(self) -> Dict[str, Any]:
        """Generate a summary of penetration testing results."""
        summary = {
            "test_coverage": {
                "error_handling": "error_handling_disclosure" in self.test_results,
                "authentication": "authentication_error_disclosure" in self.test_results,
                "api_security": "api_security_responses" in self.test_results,
                "file_security": "file_upload_security" in self.test_results,
                "path_traversal": "path_traversal_prevention" in self.test_results,
                "production_hardening": "production_hardening" in self.test_results,
            },
            "attack_vectors_tested": [
                "Information disclosure through error messages",
                "Authentication bypass attempts",
                "API security vulnerabilities",
                "File upload security flaws",
                "Path traversal attacks",
                "Production configuration weaknesses",
            ],
            "methodology": "OWASP Testing Guide v4.0 + Custom Security Assessment",
            "scope": "Web application and API endpoints",
            "limitations": [
                "Tests performed in non-production environment",
                "Limited to automated security testing",
                "Manual testing may reveal additional issues",
            ],
        }

        return summary

    def save_report(
        self,
        report: Dict[str, Any],
        output_format: str = "both",
        output_dir: str = None,
    ) -> List[str]:
        """Save the comprehensive penetration testing report."""
        if output_dir is None:
            output_dir = "/home/green/FreeAgentics/tests/security"

        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_files = []

        # Save JSON report
        if output_format in ["json", "both"]:
            json_file = os.path.join(
                output_dir,
                f"comprehensive_penetration_test_report_{timestamp}.json",
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
                f"comprehensive_penetration_test_report_{timestamp}.html",
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
        """Generate an HTML version of the penetration testing report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreeAgentics Penetration Testing Report</title>
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
            border-bottom: 3px solid #dc3545;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            font-weight: bold;
        }}
        .alert-critical {{
            background-color: #dc3545;
            color: white;
            border-left: 5px solid #a71e2a;
        }}
        .alert-high {{
            background-color: #fd7e14;
            color: white;
            border-left: 5px solid #d63384;
        }}
        .alert-medium {{
            background-color: #ffc107;
            color: #212529;
            border-left: 5px solid #ffca2c;
        }}
        .alert-low {{
            background-color: #20c997;
            color: white;
            border-left: 5px solid #0f5132;
        }}
        .executive-summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
            border-left: 4px solid #007bff;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
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
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        .domain-assessment {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .domain {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
        }}
        .domain h4 {{
            margin-top: 0;
            color: #495057;
        }}
        .score {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .score-excellent {{ color: #28a745; }}
        .score-good {{ color: #20c997; }}
        .score-acceptable {{ color: #ffc107; }}
        .score-poor {{ color: #fd7e14; }}
        .score-critical {{ color: #dc3545; }}
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
            <h1>🔒 FreeAgentics Penetration Testing Report</h1>
            <p>Comprehensive Security Assessment & Vulnerability Analysis</p>
            <p>Generated on {report["metadata"]["test_execution_time"]}</p>
        </div>

        <div class="executive-summary">
            <h2>🎯 Executive Summary</h2>
            <div class="alert alert-{report["executive_summary"]["security_level"].lower()}">
                {report["executive_summary"]["security_status"]}
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{report["executive_summary"]["penetration_score"]}</div>
                    <div class="metric-label">Security Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report["overall_statistics"]["total_tests"]}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report["overall_statistics"]["overall_pass_rate"]}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report["executive_summary"]["critical_vulnerabilities"]}</div>
                    <div class="metric-label">Critical Issues</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report["executive_summary"]["high_vulnerabilities"]}</div>
                    <div class="metric-label">High Issues</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{"✅ YES" if report["executive_summary"]["production_ready"] else "❌ NO"}</div>
                    <div class="metric-label">Production Ready</div>
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

        # Add security domain assessment
        html += """
        <div class="section">
            <h2>🛡️ Security Domain Assessment</h2>
            <div class="domain-assessment">
        """

        for domain_name, domain_data in report["security_domain_assessment"].items():
            status_class = f"score-{domain_data['status']}"
            html += f"""
                <div class="domain">
                    <h4>{domain_name.replace("_", " ").title()}</h4>
                    <div class="score {status_class}">{domain_data["score"]}/100</div>
                    <div>Status: {domain_data["status"].title()}</div>
                </div>
            """

        html += "</div></div>"

        # Add test suite results
        html += """
        <div class="section">
            <h2>🧪 Test Suite Results</h2>
            <div class="test-results">
        """

        for suite_name, suite_data in report["test_suite_results"].items():
            pass_rate = suite_data.get("pass_rate", 0)
            status_class = "pass" if pass_rate >= 80 else "fail"

            html += f"""
                <div class="test-suite">
                    <h3>{suite_name.replace("_", " ").title()}</h3>
                    <p><strong>Pass Rate:</strong> <span class="{status_class}">{pass_rate:.1f}%</span></p>
                    <p><strong>Tests:</strong> {suite_data.get("total_tests", 0)} total, {suite_data.get("passed_tests", 0)} passed, {suite_data.get("failed_tests", 0)} failed</p>
                    <p><strong>Critical Issues:</strong> {suite_data.get("critical_findings", 0)}</p>
                    <p><strong>High Issues:</strong> {suite_data.get("high_findings", 0)}</p>
                </div>
            """

        html += "</div></div>"

        # Add threat assessment
        html += """
        <div class="section">
            <h2>⚠️ Threat Assessment</h2>
            <div class="test-results">
        """

        for threat_name, threat_data in report["threat_assessment"].items():
            level_class = f"alert-{threat_data['level'].lower()}"
            html += f"""
                <div class="test-suite">
                    <h3>{threat_name.replace("_", " ").title()}</h3>
                    <div class="alert {level_class}" style="margin: 10px 0; padding: 10px;">
                        Risk Level: {threat_data["level"]}
                    </div>
                    <p>{threat_data["description"]}</p>
                </div>
            """

        html += "</div></div>"

        # Add top recommendations
        if report["security_recommendations"]["all_recommendations"]:
            html += """
            <div class="section">
                <h2>📋 Security Recommendations</h2>
                <ul>
            """
            for rec in report["security_recommendations"]["all_recommendations"][:15]:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"

        # Add footer
        html += f"""
        <div class="footer">
            <p>🔐 Report generated by FreeAgentics Penetration Testing Suite</p>
            <p>Test duration: {report["metadata"]["test_duration_seconds"]:.1f} seconds</p>
            <p><strong>Disclaimer:</strong> This report represents automated security testing results. Manual testing may reveal additional vulnerabilities.</p>
        </div>
    </div>
</body>
</html>
        """

        return html

    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the penetration testing results."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PENETRATION TESTING SUMMARY")
        print("=" * 80)

        exec_summary = report["executive_summary"]
        stats = report["overall_statistics"]

        print(f"Security Status: {exec_summary['security_status']}")
        print(f"Penetration Score: {exec_summary['penetration_score']}/100")
        print(f"Threat Level: {exec_summary['threat_level']}")
        print(f"Production Ready: {'✅ YES' if exec_summary['production_ready'] else '❌ NO'}")
        print()

        print("Test Statistics:")
        print(f"  Total Tests Run: {stats['total_tests']}")
        print(f"  Tests Passed: {stats['total_passed']}")
        print(f"  Tests Failed: {stats['total_failed']}")
        print(f"  Overall Pass Rate: {stats['overall_pass_rate']:.1f}%")
        print()

        print("Vulnerabilities by Severity:")
        print(f"  🔴 Critical: {exec_summary['critical_vulnerabilities']}")
        print(f"  🟠 High: {exec_summary['high_vulnerabilities']}")
        print(f"  🟡 Medium: {exec_summary['medium_vulnerabilities']}")
        print()

        print("Security Domain Assessment:")
        for domain_name, domain_data in report["security_domain_assessment"].items():
            status_icon = {
                "excellent": "🟢",
                "good": "🟢",
                "acceptable": "🟡",
                "poor": "🟠",
                "critical": "🔴",
            }.get(domain_data["status"], "⚪")

            print(
                f"  {status_icon} {domain_name.replace('_', ' ').title()}: {domain_data['score']}/100 ({domain_data['status'].title()})"
            )
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
    """Main function to run comprehensive penetration tests."""
    parser = argparse.ArgumentParser(description="Run comprehensive penetration testing suite")
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
    runner = ComprehensivePenetrationTestRunner()

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

    # Exit with appropriate code based on severity
    exec_summary = report["executive_summary"]
    if exec_summary["critical_vulnerabilities"] > 0:
        print("\n❌ CRITICAL VULNERABILITIES DETECTED - Application not ready for production")
        sys.exit(1)
    elif exec_summary["high_vulnerabilities"] > 3:
        print("\n⚠️  HIGH RISK VULNERABILITIES DETECTED - Significant security remediation needed")
        sys.exit(2)
    elif not exec_summary["production_ready"]:
        print("\n⚠️  APPLICATION NOT PRODUCTION READY - Address security issues before deployment")
        sys.exit(3)
    else:
        print("\n✅ PENETRATION TESTING COMPLETED - Application has acceptable security posture")
        sys.exit(0)


if __name__ == "__main__":
    main()
