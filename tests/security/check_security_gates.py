#!/usr/bin/env python3
"""
Security Gate Validation Script

This script validates security test results and determines if the build
should pass or fail based on security criteria.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


class SecurityGateValidator:
    """Validates security test results against defined thresholds"""

    def __init__(self):
        self.thresholds = {
            "high_risk_max": 0,  # No high-risk vulnerabilities allowed
            "medium_risk_max": 5,  # Maximum 5 medium-risk vulnerabilities
            "low_risk_max": 20,  # Maximum 20 low-risk vulnerabilities
            "security_score_min": 85,  # Minimum security score of 85/100
            "test_coverage_min": 90,  # Minimum 90% security test coverage
            "owasp_compliance_min": 9,  # Must pass at least 9/10 OWASP checks
        }

        self.reports = {
            "security_test_report": "security_test_report.json",
            "bandit_report": "bandit-report.json",
            "safety_report": "safety-report.json",
            "zap_report": "zap_security_report.json",
            "penetration_report": "penetration_test_report.json",
        }

        self.validation_results = {
            "passed": True,
            "failures": [],
            "warnings": [],
            "summary": {},
        }

    def validate_all(self) -> Tuple[bool, Dict]:
        """Run all security validations"""

        print("=" * 80)
        print("SECURITY GATE VALIDATION")
        print("=" * 80)
        print(f"Validation started at: {datetime.now().isoformat()}\n")

        # Validate comprehensive security test suite
        self._validate_security_test_suite()

        # Validate SAST results (Bandit)
        self._validate_bandit_results()

        # Validate dependency security (Safety)
        self._validate_safety_results()

        # Validate DAST results (ZAP)
        self._validate_zap_results()

        # Validate penetration test results
        self._validate_penetration_tests()

        # Generate final report
        self._generate_validation_report()

        return self.validation_results["passed"], self.validation_results

    def _validate_security_test_suite(self):
        """Validate comprehensive security test suite results"""

        print("Validating security test suite...")

        try:
            report_path = Path(self.reports["security_test_report"])
            if not report_path.exists():
                self.validation_results["failures"].append(
                    {
                        "test": "Security Test Suite",
                        "reason": "Report file not found",
                        "severity": "CRITICAL",
                    }
                )
                self.validation_results["passed"] = False
                return

            with open(report_path) as f:
                report = json.load(f)

            # Check vulnerability counts
            report.get("summary", {}).get("vulnerabilities_found", 0)
            security_score = report.get("summary", {}).get("security_score", 0)

            # Extract risk levels from detailed results
            high_risk = 0
            medium_risk = 0
            low_risk = 0

            for category, results in report.get("detailed_results", {}).items():
                for result in results:
                    if result.get("vulnerable"):
                        risk = result.get("risk", "low").lower()
                        if risk == "high":
                            high_risk += 1
                        elif risk == "medium":
                            medium_risk += 1
                        else:
                            low_risk += 1

            # Validate against thresholds
            if high_risk > self.thresholds["high_risk_max"]:
                self.validation_results["failures"].append(
                    {
                        "test": "High Risk Vulnerabilities",
                        "found": high_risk,
                        "threshold": self.thresholds["high_risk_max"],
                        "severity": "CRITICAL",
                    }
                )
                self.validation_results["passed"] = False

            if medium_risk > self.thresholds["medium_risk_max"]:
                self.validation_results["failures"].append(
                    {
                        "test": "Medium Risk Vulnerabilities",
                        "found": medium_risk,
                        "threshold": self.thresholds["medium_risk_max"],
                        "severity": "HIGH",
                    }
                )
                self.validation_results["passed"] = False

            if low_risk > self.thresholds["low_risk_max"]:
                self.validation_results["warnings"].append(
                    {
                        "test": "Low Risk Vulnerabilities",
                        "found": low_risk,
                        "threshold": self.thresholds["low_risk_max"],
                        "severity": "MEDIUM",
                    }
                )

            if security_score < self.thresholds["security_score_min"]:
                self.validation_results["failures"].append(
                    {
                        "test": "Security Score",
                        "score": security_score,
                        "threshold": self.thresholds["security_score_min"],
                        "severity": "HIGH",
                    }
                )
                self.validation_results["passed"] = False

            # Check OWASP compliance
            owasp_compliance = report.get("compliance", {}).get("OWASP_Top_10", {})
            passed_checks = sum(1 for v in owasp_compliance.values() if v)

            if passed_checks < self.thresholds["owasp_compliance_min"]:
                self.validation_results["failures"].append(
                    {
                        "test": "OWASP Top 10 Compliance",
                        "passed": passed_checks,
                        "threshold": self.thresholds["owasp_compliance_min"],
                        "severity": "HIGH",
                    }
                )
                self.validation_results["passed"] = False

            self.validation_results["summary"]["security_test_suite"] = {
                "vulnerabilities": {
                    "high": high_risk,
                    "medium": medium_risk,
                    "low": low_risk,
                },
                "security_score": security_score,
                "owasp_compliance": f"{passed_checks}/10",
            }

            print("✓ Security test suite validated")
            print(f"  - High risk: {high_risk}")
            print(f"  - Medium risk: {medium_risk}")
            print(f"  - Low risk: {low_risk}")
            print(f"  - Security score: {security_score}/100")
            print(f"  - OWASP compliance: {passed_checks}/10")

        except Exception as e:
            self.validation_results["failures"].append(
                {
                    "test": "Security Test Suite",
                    "reason": str(e),
                    "severity": "CRITICAL",
                }
            )
            self.validation_results["passed"] = False

    def _validate_bandit_results(self):
        """Validate Bandit SAST results"""

        print("\nValidating Bandit SAST results...")

        try:
            report_path = Path(self.reports["bandit_report"])
            if not report_path.exists():
                print("  - Bandit report not found (may not have run)")
                return

            with open(report_path) as f:
                report = json.load(f)

            # Count issues by severity
            high_severity = len(
                [
                    r
                    for r in report.get("results", [])
                    if r.get("issue_severity") == "HIGH"
                ]
            )
            medium_severity = len(
                [
                    r
                    for r in report.get("results", [])
                    if r.get("issue_severity") == "MEDIUM"
                ]
            )
            low_severity = len(
                [
                    r
                    for r in report.get("results", [])
                    if r.get("issue_severity") == "LOW"
                ]
            )

            # Validate
            if high_severity > 0:
                self.validation_results["failures"].append(
                    {
                        "test": "Bandit High Severity Issues",
                        "found": high_severity,
                        "threshold": 0,
                        "severity": "HIGH",
                    }
                )
                self.validation_results["passed"] = False

            if medium_severity > 5:
                self.validation_results["warnings"].append(
                    {
                        "test": "Bandit Medium Severity Issues",
                        "found": medium_severity,
                        "threshold": 5,
                        "severity": "MEDIUM",
                    }
                )

            self.validation_results["summary"]["bandit"] = {
                "high": high_severity,
                "medium": medium_severity,
                "low": low_severity,
            }

            print("✓ Bandit SAST validated")
            print(f"  - High severity: {high_severity}")
            print(f"  - Medium severity: {medium_severity}")
            print(f"  - Low severity: {low_severity}")

        except Exception as e:
            print(f"  - Error validating Bandit results: {e}")

    def _validate_safety_results(self):
        """Validate Safety dependency check results"""

        print("\nValidating Safety dependency check...")

        try:
            report_path = Path(self.reports["safety_report"])
            if not report_path.exists():
                print("  - Safety report not found (may not have run)")
                return

            with open(report_path) as f:
                report = json.load(f)

            vulnerabilities = report.get("vulnerabilities", [])

            if len(vulnerabilities) > 0:
                # Check severity of vulnerabilities
                critical_vulns = [
                    v
                    for v in vulnerabilities
                    if "critical" in v.get("severity", "").lower()
                ]
                high_vulns = [
                    v
                    for v in vulnerabilities
                    if "high" in v.get("severity", "").lower()
                ]

                if len(critical_vulns) > 0:
                    self.validation_results["failures"].append(
                        {
                            "test": "Critical Dependency Vulnerabilities",
                            "found": len(critical_vulns),
                            "packages": [v.get("package") for v in critical_vulns],
                            "severity": "CRITICAL",
                        }
                    )
                    self.validation_results["passed"] = False

                if len(high_vulns) > 0:
                    self.validation_results["warnings"].append(
                        {
                            "test": "High Severity Dependency Vulnerabilities",
                            "found": len(high_vulns),
                            "packages": [v.get("package") for v in high_vulns],
                            "severity": "HIGH",
                        }
                    )

            self.validation_results["summary"]["safety"] = {
                "total_vulnerabilities": len(vulnerabilities),
                "packages_affected": list(
                    set(v.get("package") for v in vulnerabilities)
                ),
            }

            print("✓ Safety dependency check validated")
            print(f"  - Vulnerabilities found: {len(vulnerabilities)}")

        except Exception as e:
            print(f"  - Error validating Safety results: {e}")

    def _validate_zap_results(self):
        """Validate OWASP ZAP DAST results"""

        print("\nValidating OWASP ZAP results...")

        try:
            report_path = Path(self.reports["zap_report"])
            if not report_path.exists():
                print("  - ZAP report not found (may not have run)")
                return

            with open(report_path) as f:
                report = json.load(f)

            # Extract risk summary
            risk_summary = report.get("risk_summary", {})
            high_risk = risk_summary.get("high", 0)
            medium_risk = risk_summary.get("medium", 0)

            # Validate
            if high_risk > 0:
                self.validation_results["failures"].append(
                    {
                        "test": "ZAP High Risk Findings",
                        "found": high_risk,
                        "threshold": 0,
                        "severity": "HIGH",
                    }
                )
                self.validation_results["passed"] = False

            if medium_risk > 10:
                self.validation_results["warnings"].append(
                    {
                        "test": "ZAP Medium Risk Findings",
                        "found": medium_risk,
                        "threshold": 10,
                        "severity": "MEDIUM",
                    }
                )

            self.validation_results["summary"]["zap"] = risk_summary

            print("✓ OWASP ZAP validated")
            print(f"  - High risk: {high_risk}")
            print(f"  - Medium risk: {medium_risk}")
            print(f"  - Low risk: {risk_summary.get('low', 0)}")

        except Exception as e:
            print(f"  - Error validating ZAP results: {e}")

    def _validate_penetration_tests(self):
        """Validate penetration test results"""

        print("\nValidating penetration test results...")

        try:
            report_path = Path(self.reports["penetration_report"])
            if not report_path.exists():
                print("  - Penetration test report not found (may not have run)")
                return

            with open(report_path) as f:
                report = json.load(f)

            # Check for successful exploits
            successful_exploits = report.get("successful_exploits", 0)
            critical_findings = report.get("critical_findings", 0)

            if successful_exploits > 0:
                self.validation_results["failures"].append(
                    {
                        "test": "Successful Penetration Attempts",
                        "found": successful_exploits,
                        "threshold": 0,
                        "severity": "CRITICAL",
                    }
                )
                self.validation_results["passed"] = False

            if critical_findings > 0:
                self.validation_results["failures"].append(
                    {
                        "test": "Critical Security Findings",
                        "found": critical_findings,
                        "threshold": 0,
                        "severity": "CRITICAL",
                    }
                )
                self.validation_results["passed"] = False

            self.validation_results["summary"]["penetration_tests"] = {
                "successful_exploits": successful_exploits,
                "critical_findings": critical_findings,
                "tests_run": report.get("total_tests", 0),
            }

            print("✓ Penetration tests validated")
            print(f"  - Successful exploits: {successful_exploits}")
            print(f"  - Critical findings: {critical_findings}")

        except Exception as e:
            print(f"  - Error validating penetration test results: {e}")

    def _generate_validation_report(self):
        """Generate final validation report"""

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if self.validation_results["passed"]:
            print("✅ SECURITY GATE: PASSED")
        else:
            print("❌ SECURITY GATE: FAILED")

        if self.validation_results["failures"]:
            print("\nFAILURES:")
            for failure in self.validation_results["failures"]:
                print(f"  - [{failure.get('severity')}] {failure.get('test')}")
                if "found" in failure and "threshold" in failure:
                    print(
                        f"    Found: {failure['found']}, Threshold: {failure['threshold']}"
                    )
                if "reason" in failure:
                    print(f"    Reason: {failure['reason']}")

        if self.validation_results["warnings"]:
            print("\nWARNINGS:")
            for warning in self.validation_results["warnings"]:
                print(f"  - [{warning.get('severity')}] {warning.get('test')}")
                if "found" in warning and "threshold" in warning:
                    print(
                        f"    Found: {warning['found']}, Threshold: {warning['threshold']}"
                    )

        # Save validation report
        report_path = "security_gate_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

        # Generate exit code for CI/CD
        if not self.validation_results["passed"]:
            print("\n❌ Build failed due to security gate violations")
            sys.exit(1)
        else:
            print("\n✅ All security gates passed")
            sys.exit(0)


def main():
    """Main entry point"""

    validator = SecurityGateValidator()

    # Check if running in CI environment
    ci_env = os.getenv("CI", "").lower() == "true"
    if ci_env:
        print("Running in CI environment")

    # Run validation
    passed, results = validator.validate_all()

    # Exit with appropriate code
    if not passed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
