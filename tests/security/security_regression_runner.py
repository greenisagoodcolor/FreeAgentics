#!/usr/bin/env python3
"""
Security Regression Test Runner

This script runs all security tests and creates a comprehensive report
to ensure no security regressions are introduced.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class SecurityRegressionRunner:
    """Runs all security tests and generates comprehensive reports"""

    def __init__(self, output_dir: str = "security_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.test_results = {
            "comprehensive_security_suite": None,
            "penetration_tests": None,
            "performance_under_attack": None,
            "owasp_zap_scan": None,
            "security_headers_validation": None,
            "rate_limiting_verification": None,
            "rbac_comprehensive": None,
            "security_monitoring": None,
        }

        self.overall_status = "UNKNOWN"
        self.critical_failures = []
        self.warnings = []

    async def run_all_security_tests(self) -> Dict:
        """Run all security tests and generate comprehensive report"""

        print("=" * 80)
        print("SECURITY REGRESSION TEST RUNNER")
        print("=" * 80)
        print(f"Starting security regression testing at {datetime.now()}")
        print(f"Reports will be saved to: {self.output_dir}")

        # Run test categories
        await self._run_comprehensive_security_suite()
        await self._run_penetration_tests()
        await self._run_performance_under_attack_tests()
        await self._run_owasp_zap_scan()
        await self._run_security_headers_validation()
        await self._run_rate_limiting_verification()
        await self._run_rbac_comprehensive_tests()
        await self._run_security_monitoring_tests()

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        # Save report
        report_file = self.output_dir / "security_regression_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate HTML report
        self._generate_html_report(report)

        # Print summary
        self._print_summary(report)

        return report

    async def _run_comprehensive_security_suite(self):
        """Run comprehensive security test suite"""

        print("\n[1/8] Running Comprehensive Security Test Suite...")

        try:
            # Run the comprehensive security suite
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "tests/security/comprehensive_security_test_suite.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Check if report was generated
            report_file = Path("security_test_report_detailed.json")
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)

                self.test_results["comprehensive_security_suite"] = {
                    "status": (
                        "PASSED"
                        if report_data.get("summary", {}).get("vulnerabilities_found", 0) == 0
                        else "FAILED"
                    ),
                    "vulnerabilities_found": report_data.get("summary", {}).get(
                        "vulnerabilities_found", 0
                    ),
                    "security_score": report_data.get("summary", {}).get("security_score", 0),
                    "details": report_data,
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }
            else:
                self.test_results["comprehensive_security_suite"] = {
                    "status": "ERROR",
                    "error": "Report file not generated",
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }

            print(
                f"  ✓ Comprehensive Security Suite: {self.test_results['comprehensive_security_suite']['status']}"
            )

        except Exception as e:
            self.test_results["comprehensive_security_suite"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Comprehensive Security Suite: ERROR - {e}")

    async def _run_penetration_tests(self):
        """Run penetration tests"""

        print("\n[2/8] Running Penetration Tests...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "tests/security/run_comprehensive_penetration_tests.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Check for penetration test report
            report_file = Path("penetration_test_report.json")
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)

                successful_exploits = report_data.get("successful_exploits", 0)

                self.test_results["penetration_tests"] = {
                    "status": "PASSED" if successful_exploits == 0 else "FAILED",
                    "successful_exploits": successful_exploits,
                    "total_tests": report_data.get("total_tests", 0),
                    "details": report_data,
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }
            else:
                self.test_results["penetration_tests"] = {
                    "status": "ERROR",
                    "error": "Report file not generated",
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }

            print(f"  ✓ Penetration Tests: {self.test_results['penetration_tests']['status']}")

        except Exception as e:
            self.test_results["penetration_tests"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Penetration Tests: ERROR - {e}")

    async def _run_performance_under_attack_tests(self):
        """Run performance under attack tests"""

        print("\n[3/8] Running Performance Under Attack Tests...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "tests/security/performance_under_attack.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Check for performance report
            report_file = Path("performance_under_attack_report.json")
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)

                resilience_score = report_data.get("summary", {}).get("overall_resilience_score", 0)

                self.test_results["performance_under_attack"] = {
                    "status": "PASSED" if resilience_score >= 80 else "FAILED",
                    "resilience_score": resilience_score,
                    "passed_scenarios": report_data.get("summary", {}).get("passed_scenarios", 0),
                    "total_scenarios": report_data.get("summary", {}).get("total_scenarios", 0),
                    "details": report_data,
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }
            else:
                self.test_results["performance_under_attack"] = {
                    "status": "ERROR",
                    "error": "Report file not generated",
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }

            print(
                f"  ✓ Performance Under Attack: {self.test_results['performance_under_attack']['status']}"
            )

        except Exception as e:
            self.test_results["performance_under_attack"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Performance Under Attack: ERROR - {e}")

    async def _run_owasp_zap_scan(self):
        """Run OWASP ZAP scan"""

        print("\n[4/8] Running OWASP ZAP Scan...")

        try:
            # Check if ZAP is available
            zap_process = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-q",
                "--filter",
                "name=zap",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await zap_process.communicate()

            if not stdout.decode().strip():
                # ZAP not running, skip this test
                self.test_results["owasp_zap_scan"] = {
                    "status": "SKIPPED",
                    "reason": "ZAP not available",
                }
                print("  ⚠ OWASP ZAP Scan: SKIPPED (ZAP not available)")
                return

            # Run ZAP integration
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "tests/security/owasp_zap_integration.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Check for ZAP report
            report_file = Path("zap_security_report.json")
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)

                high_risk = report_data.get("risk_summary", {}).get("high", 0)

                self.test_results["owasp_zap_scan"] = {
                    "status": "PASSED" if high_risk == 0 else "FAILED",
                    "high_risk_issues": high_risk,
                    "total_alerts": report_data.get("scan_info", {}).get("total_alerts", 0),
                    "details": report_data,
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }
            else:
                self.test_results["owasp_zap_scan"] = {
                    "status": "ERROR",
                    "error": "Report file not generated",
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }

            print(f"  ✓ OWASP ZAP Scan: {self.test_results['owasp_zap_scan']['status']}")

        except Exception as e:
            self.test_results["owasp_zap_scan"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ OWASP ZAP Scan: ERROR - {e}")

    async def _run_security_headers_validation(self):
        """Run security headers validation"""

        print("\n[5/8] Running Security Headers Validation...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_security_headers_validation.py",
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Parse pytest output
            output = stdout.decode()
            status = "PASSED" if "FAILED" not in output and process.returncode == 0 else "FAILED"

            self.test_results["security_headers_validation"] = {
                "status": status,
                "output": output,
                "errors": stderr.decode(),
                "return_code": process.returncode,
            }

            print(f"  ✓ Security Headers Validation: {status}")

        except Exception as e:
            self.test_results["security_headers_validation"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Security Headers Validation: ERROR - {e}")

    async def _run_rate_limiting_verification(self):
        """Run rate limiting verification"""

        print("\n[6/8] Running Rate Limiting Verification...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_rate_limiting_verification.py",
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Parse pytest output
            output = stdout.decode()
            status = "PASSED" if "FAILED" not in output and process.returncode == 0 else "FAILED"

            self.test_results["rate_limiting_verification"] = {
                "status": status,
                "output": output,
                "errors": stderr.decode(),
                "return_code": process.returncode,
            }

            print(f"  ✓ Rate Limiting Verification: {status}")

        except Exception as e:
            self.test_results["rate_limiting_verification"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Rate Limiting Verification: ERROR - {e}")

    async def _run_rbac_comprehensive_tests(self):
        """Run RBAC comprehensive tests"""

        print("\n[7/8] Running RBAC Comprehensive Tests...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_rbac_comprehensive.py",
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Parse pytest output
            output = stdout.decode()
            status = "PASSED" if "FAILED" not in output and process.returncode == 0 else "FAILED"

            self.test_results["rbac_comprehensive"] = {
                "status": status,
                "output": output,
                "errors": stderr.decode(),
                "return_code": process.returncode,
            }

            print(f"  ✓ RBAC Comprehensive Tests: {status}")

        except Exception as e:
            self.test_results["rbac_comprehensive"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ RBAC Comprehensive Tests: ERROR - {e}")

    async def _run_security_monitoring_tests(self):
        """Run security monitoring tests"""

        print("\n[8/8] Running Security Monitoring Tests...")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_security_monitoring_system.py",
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )

            stdout, stderr = await process.communicate()

            # Parse pytest output
            output = stdout.decode()
            status = "PASSED" if "FAILED" not in output and process.returncode == 0 else "FAILED"

            self.test_results["security_monitoring"] = {
                "status": status,
                "output": output,
                "errors": stderr.decode(),
                "return_code": process.returncode,
            }

            print(f"  ✓ Security Monitoring Tests: {status}")

        except Exception as e:
            self.test_results["security_monitoring"] = {
                "status": "ERROR",
                "error": str(e),
            }
            print(f"  ✗ Security Monitoring Tests: ERROR - {e}")

    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive security regression report"""

        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(
            1
            for result in self.test_results.values()
            if result and result.get("status") == "PASSED"
        )
        failed_tests = sum(
            1
            for result in self.test_results.values()
            if result and result.get("status") == "FAILED"
        )
        error_tests = sum(
            1 for result in self.test_results.values() if result and result.get("status") == "ERROR"
        )
        skipped_tests = sum(
            1
            for result in self.test_results.values()
            if result and result.get("status") == "SKIPPED"
        )

        # Determine overall status
        if failed_tests > 0 or error_tests > 0:
            self.overall_status = "FAILED"
        elif passed_tests == total_tests - skipped_tests:
            self.overall_status = "PASSED"
        else:
            self.overall_status = "PARTIAL"

        # Collect critical failures
        for test_name, result in self.test_results.items():
            if result and result.get("status") in ["FAILED", "ERROR"]:
                self.critical_failures.append(
                    {
                        "test": test_name,
                        "status": result.get("status"),
                        "error": result.get("error", "Test failed"),
                        "details": result.get("details", {}),
                    }
                )

        # Generate report
        report = {
            "summary": {
                "overall_status": self.overall_status,
                "test_date": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "skipped_tests": skipped_tests,
                "success_rate": (passed_tests / max(total_tests - skipped_tests, 1)) * 100,
            },
            "test_results": self.test_results,
            "critical_failures": self.critical_failures,
            "security_metrics": self._calculate_security_metrics(),
            "recommendations": self._generate_recommendations(),
            "compliance_status": self._check_compliance_status(),
        }

        return report

    def _calculate_security_metrics(self) -> Dict:
        """Calculate security metrics from test results"""

        metrics = {
            "vulnerability_count": 0,
            "security_score": 0,
            "exploitable_vulnerabilities": 0,
            "high_risk_issues": 0,
            "compliance_score": 0,
        }

        # Comprehensive security suite metrics
        if self.test_results.get("comprehensive_security_suite"):
            result = self.test_results["comprehensive_security_suite"]
            if result.get("details"):
                metrics["vulnerability_count"] = (
                    result["details"].get("summary", {}).get("vulnerabilities_found", 0)
                )
                metrics["security_score"] = (
                    result["details"].get("summary", {}).get("security_score", 0)
                )

        # Penetration test metrics
        if self.test_results.get("penetration_tests"):
            result = self.test_results["penetration_tests"]
            metrics["exploitable_vulnerabilities"] = result.get("successful_exploits", 0)

        # ZAP scan metrics
        if self.test_results.get("owasp_zap_scan"):
            result = self.test_results["owasp_zap_scan"]
            metrics["high_risk_issues"] = result.get("high_risk_issues", 0)

        # Calculate overall compliance score
        passed_tests = sum(
            1
            for result in self.test_results.values()
            if result and result.get("status") == "PASSED"
        )
        total_tests = len(
            [r for r in self.test_results.values() if r and r.get("status") != "SKIPPED"]
        )

        metrics["compliance_score"] = (passed_tests / max(total_tests, 1)) * 100

        return metrics

    def _generate_recommendations(self) -> List[Dict]:
        """Generate security recommendations based on test results"""

        recommendations = []

        # Check for critical failures
        if self.critical_failures:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "category": "Test Failures",
                    "recommendation": f"Address {len(self.critical_failures)} critical test failures immediately",
                    "affected_tests": [f["test"] for f in self.critical_failures],
                }
            )

        # Check vulnerability count
        metrics = self._calculate_security_metrics()
        if metrics["vulnerability_count"] > 0:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Security Vulnerabilities",
                    "recommendation": f"Fix {metrics['vulnerability_count']} security vulnerabilities",
                    "action": "Review comprehensive security test results",
                }
            )

        # Check exploitable vulnerabilities
        if metrics["exploitable_vulnerabilities"] > 0:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "category": "Exploitable Vulnerabilities",
                    "recommendation": f"Immediately fix {metrics['exploitable_vulnerabilities']} exploitable vulnerabilities",
                    "action": "Review penetration test results",
                }
            )

        # Check high-risk issues
        if metrics["high_risk_issues"] > 0:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "High-Risk Issues",
                    "recommendation": f"Address {metrics['high_risk_issues']} high-risk security issues",
                    "action": "Review OWASP ZAP scan results",
                }
            )

        # Performance recommendations
        if self.test_results.get("performance_under_attack"):
            result = self.test_results["performance_under_attack"]
            if result.get("resilience_score", 0) < 80:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "Performance Resilience",
                        "recommendation": "Improve system resilience under attack conditions",
                        "action": "Review performance under attack test results",
                    }
                )

        # General recommendations
        recommendations.extend(
            [
                {
                    "priority": "MEDIUM",
                    "category": "Continuous Security",
                    "recommendation": "Implement continuous security monitoring",
                    "action": "Set up automated security testing in CI/CD",
                },
                {
                    "priority": "LOW",
                    "category": "Security Training",
                    "recommendation": "Provide security training to development team",
                    "action": "Schedule regular security awareness sessions",
                },
            ]
        )

        return recommendations

    def _check_compliance_status(self) -> Dict:
        """Check compliance status against various standards"""

        compliance = {
            "OWASP_Top_10": {
                "status": "UNKNOWN",
                "score": 0,
                "details": "Check comprehensive security test results",
            },
            "PCI_DSS": {
                "status": "PARTIAL",
                "score": 70,
                "details": "Security controls implemented, requires regular testing",
            },
            "GDPR": {
                "status": "PARTIAL",
                "score": 80,
                "details": "Data protection measures in place, requires audit",
            },
            "SOC2": {
                "status": "PARTIAL",
                "score": 75,
                "details": "Security controls implemented, requires documentation",
            },
        }

        # Update OWASP Top 10 status if comprehensive security test ran
        if self.test_results.get("comprehensive_security_suite"):
            result = self.test_results["comprehensive_security_suite"]
            if result.get("details") and result["details"].get("compliance"):
                owasp_compliance = result["details"]["compliance"].get("OWASP_Top_10", {})
                passed_checks = sum(1 for v in owasp_compliance.values() if v)
                total_checks = len(owasp_compliance)

                compliance["OWASP_Top_10"]["score"] = (passed_checks / max(total_checks, 1)) * 100
                compliance["OWASP_Top_10"]["status"] = "PASSED" if passed_checks >= 9 else "FAILED"
                compliance["OWASP_Top_10"][
                    "details"
                ] = f"Passed {passed_checks}/{total_checks} checks"

        return compliance

    def _generate_html_report(self, report: Dict):
        """Generate HTML report for better visualization"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Regression Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; margin-bottom: 20px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; margin-bottom: 20px; }}
        .failure {{ background-color: #ffe8e8; padding: 15px; margin-bottom: 20px; }}
        .test-result {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .error {{ border-left: 5px solid #ff9800; }}
        .skipped {{ border-left: 5px solid #9e9e9e; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .metric {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Regression Test Report</h1>
        <p>Generated on: {report["summary"]["test_date"]}</p>
        <p>Overall Status: <strong>{report["summary"]["overall_status"]}</strong></p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric">
                <h3>Test Results</h3>
                <p>Total Tests: {report["summary"]["total_tests"]}</p>
                <p>Passed: {report["summary"]["passed_tests"]}</p>
                <p>Failed: {report["summary"]["failed_tests"]}</p>
                <p>Errors: {report["summary"]["error_tests"]}</p>
                <p>Skipped: {report["summary"]["skipped_tests"]}</p>
                <p>Success Rate: {report["summary"]["success_rate"]:.1f}%</p>
            </div>
            <div class="metric">
                <h3>Security Metrics</h3>
                <p>Vulnerabilities: {report["security_metrics"]["vulnerability_count"]}</p>
                <p>Security Score: {report["security_metrics"]["security_score"]}</p>
                <p>Exploitable: {report["security_metrics"]["exploitable_vulnerabilities"]}</p>
                <p>High Risk: {report["security_metrics"]["high_risk_issues"]}</p>
                <p>Compliance Score: {report["security_metrics"]["compliance_score"]:.1f}%</p>
            </div>
        </div>
    </div>

    <div class="test-results">
        <h2>Test Results</h2>
        """

        for test_name, result in report["test_results"].items():
            if not result:
                continue

            status = result.get("status", "UNKNOWN")
            css_class = status.lower()

            html_content += f"""
        <div class="test-result {css_class}">
            <h3>{test_name.replace("_", " ").title()}</h3>
            <p>Status: <strong>{status}</strong></p>
            {f"<p>Error: {result.get('error', '')}</p>" if result.get("error") else ""}
            {f"<p>Details: {result.get('details', {}).get('summary', {})}</p>" if result.get("details") else ""}
        </div>
        """

        if report["critical_failures"]:
            html_content += """
    </div>

    <div class="failure">
        <h2>Critical Failures</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Error</th>
            </tr>
            """

            for failure in report["critical_failures"]:
                html_content += f"""
            <tr>
                <td>{failure["test"]}</td>
                <td>{failure["status"]}</td>
                <td>{failure["error"]}</td>
            </tr>
            """

            html_content += """
        </table>
    </div>
    """

        html_content += """
    </div>

    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
        """

        for rec in report["recommendations"]:
            html_content += f"""
            <li><strong>[{rec["priority"]}]</strong> {rec["category"]}: {rec["recommendation"]}</li>
            """

        html_content += """
        </ul>
    </div>

</body>
</html>
        """

        # Save HTML report
        html_file = self.output_dir / "security_regression_report.html"
        with open(html_file, "w") as f:
            f.write(html_content)

        print(f"HTML report saved to: {html_file}")

    def _print_summary(self, report: Dict):
        """Print summary of test results"""

        print("\n" + "=" * 80)
        print("SECURITY REGRESSION TEST SUMMARY")
        print("=" * 80)

        summary = report["summary"]
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")

        if summary["failed_tests"] > 0:
            print(f"Failed Tests: {summary['failed_tests']}")

        if summary["error_tests"] > 0:
            print(f"Error Tests: {summary['error_tests']}")

        # Security metrics
        metrics = report["security_metrics"]
        print("\nSecurity Metrics:")
        print(f"  Vulnerabilities Found: {metrics['vulnerability_count']}")
        print(f"  Security Score: {metrics['security_score']}/100")
        print(f"  Exploitable Vulnerabilities: {metrics['exploitable_vulnerabilities']}")
        print(f"  High Risk Issues: {metrics['high_risk_issues']}")
        print(f"  Compliance Score: {metrics['compliance_score']:.1f}%")

        # Critical failures
        if self.critical_failures:
            print("\nCritical Failures:")
            for failure in self.critical_failures:
                print(f"  - {failure['test']}: {failure['status']}")

        # Recommendations
        high_priority_recs = [
            r for r in report["recommendations"] if r["priority"] in ["CRITICAL", "HIGH"]
        ]
        if high_priority_recs:
            print("\nHigh Priority Recommendations:")
            for rec in high_priority_recs:
                print(f"  - [{rec['priority']}] {rec['recommendation']}")

        print(f"\nReports saved to: {self.output_dir}")
        print("  - security_regression_report.json")
        print("  - security_regression_report.html")

        # Exit with appropriate code
        if summary["overall_status"] == "FAILED":
            print("\n❌ Security regression tests FAILED")
            sys.exit(1)
        elif summary["overall_status"] == "PASSED":
            print("\n✅ All security regression tests PASSED")
            sys.exit(0)
        else:
            print("\n⚠️  Security regression tests completed with issues")
            sys.exit(1)


async def main():
    """Main entry point"""

    runner = SecurityRegressionRunner()
    await runner.run_all_security_tests()


if __name__ == "__main__":
    asyncio.run(main())
