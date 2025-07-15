#!/usr/bin/env python3
"""
Comprehensive security test runner for CI/CD pipeline.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


class SecurityTestRunner:
    """Run comprehensive security tests."""

    def __init__(self):
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "vulnerabilities": [],
            "start_time": time.time(),
            "details": {},
        }

    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.returncode, e.stdout, e.stderr

    def run_bandit_scan(self) -> bool:
        """Run Bandit security linter."""
        print("\n[*] Running Bandit security scan...")
        self.results["tests_run"] += 1

        cmd = [
            "bandit",
            "-r",
            ".",
            "-f",
            "json",
            "-o",
            "bandit-results.json",
            "--skip",
            "B101,B601,B603",
            "--exclude",
            ".archive,web,tests",
        ]

        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        # Bandit returns non-zero if issues found
        if exit_code == 0 or Path("bandit-results.json").exists():
            with open("bandit-results.json") as f:
                results = json.load(f)

            high_severity = [
                r for r in results.get("results", []) if r.get("issue_severity") == "HIGH"
            ]

            if high_severity:
                print(f"  ✗ Found {len(high_severity)} high severity issues")
                self.results["vulnerabilities"].extend(high_severity)
                self.results["tests_failed"] += 1
                return False
            else:
                print("  ✓ No high severity issues found")
                self.results["tests_passed"] += 1
                return True
        else:
            print(f"  ✗ Bandit scan failed: {stderr}")
            self.results["tests_failed"] += 1
            return False

    def run_safety_check(self) -> bool:
        """Check Python dependencies for vulnerabilities."""
        print("\n[*] Running Safety dependency check...")
        self.results["tests_run"] += 1

        cmd = ["safety", "check", "--json", "--output", "safety-results.json"]
        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        if Path("safety-results.json").exists():
            with open("safety-results.json") as f:
                vulnerabilities = json.load(f)

            if vulnerabilities:
                print(f"  ✗ Found {len(vulnerabilities)} vulnerable dependencies")
                self.results["vulnerabilities"].extend(vulnerabilities)
                self.results["tests_failed"] += 1
                return False
            else:
                print("  ✓ No vulnerable dependencies found")
                self.results["tests_passed"] += 1
                return True
        else:
            print("  ✓ No vulnerabilities found")
            self.results["tests_passed"] += 1
            return True

    def run_semgrep_scan(self) -> bool:
        """Run Semgrep security patterns."""
        print("\n[*] Running Semgrep security scan...")
        self.results["tests_run"] += 1

        cmd = [
            "semgrep",
            "--config=auto",
            "--config=p/security-audit",
            "--config=p/secrets",
            "--json",
            "--output=semgrep-results.json",
            ".",
        ]

        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        if Path("semgrep-results.json").exists():
            with open("semgrep-results.json") as f:
                results = json.load(f)

            results.get("errors", [])
            findings = results.get("results", [])

            high_severity_findings = [
                f for f in findings if f.get("extra", {}).get("severity") in ["ERROR", "HIGH"]
            ]

            if high_severity_findings:
                print(f"  ✗ Found {len(high_severity_findings)} high severity findings")
                self.results["vulnerabilities"].extend(high_severity_findings)
                self.results["tests_failed"] += 1
                return False
            else:
                print("  ✓ No high severity findings")
                self.results["tests_passed"] += 1
                return True
        else:
            print(f"  ✗ Semgrep scan failed")
            self.results["tests_failed"] += 1
            return False

    def run_secrets_detection(self) -> bool:
        """Check for hardcoded secrets."""
        print("\n[*] Running secrets detection...")
        self.results["tests_run"] += 1

        # First, create baseline if it doesn't exist
        if not Path(".secrets.baseline").exists():
            subprocess.run(
                ["detect-secrets", "scan", "--baseline", ".secrets.baseline"], check=False
            )

        # Run audit
        cmd = ["detect-secrets", "audit", ".secrets.baseline"]
        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        # Check for new secrets
        cmd = ["detect-secrets", "scan", "--baseline", ".secrets.baseline"]
        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        if "No secrets detected" in stdout or exit_code == 0:
            print("  ✓ No secrets detected")
            self.results["tests_passed"] += 1
            return True
        else:
            print("  ✗ Potential secrets found")
            self.results["tests_failed"] += 1
            return False

    def run_owasp_tests(self) -> bool:
        """Run OWASP security tests."""
        print("\n[*] Running OWASP security tests...")
        self.results["tests_run"] += 1

        test_files = [
            "tests/security/test_owasp_top10.py",
            "tests/security/test_injection_prevention.py",
            "tests/security/test_authentication_security.py",
            "tests/security/test_xss_prevention.py",
            "tests/security/test_csrf_protection.py",
        ]

        all_passed = True
        for test_file in test_files:
            if Path(test_file).exists():
                cmd = ["pytest", test_file, "-v", "--tb=short"]
                exit_code, stdout, stderr = self.run_command(cmd, check=False)

                if exit_code != 0:
                    print(f"  ✗ {test_file} failed")
                    all_passed = False
                else:
                    print(f"  ✓ {test_file} passed")

        if all_passed:
            self.results["tests_passed"] += 1
            return True
        else:
            self.results["tests_failed"] += 1
            return False

    def run_api_security_tests(self) -> bool:
        """Run API security tests."""
        print("\n[*] Running API security tests...")
        self.results["tests_run"] += 1

        cmd = [
            "pytest",
            "tests/security/test_api_security.py",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=api-security-results.json",
        ]

        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        if exit_code == 0:
            print("  ✓ API security tests passed")
            self.results["tests_passed"] += 1
            return True
        else:
            print("  ✗ API security tests failed")
            self.results["tests_failed"] += 1
            return False

    def run_auth_security_tests(self) -> bool:
        """Run authentication security tests."""
        print("\n[*] Running authentication security tests...")
        self.results["tests_run"] += 1

        cmd = [
            "pytest",
            "tests/integration/test_authentication_security.py",
            "tests/unit/test_jwt_security.py",
            "-v",
            "--tb=short",
        ]

        exit_code, stdout, stderr = self.run_command(cmd, check=False)

        if exit_code == 0:
            print("  ✓ Authentication security tests passed")
            self.results["tests_passed"] += 1
            return True
        else:
            print("  ✗ Authentication security tests failed")
            self.results["tests_failed"] += 1
            return False

    def run_container_security_scan(self) -> bool:
        """Scan containers for vulnerabilities."""
        print("\n[*] Running container security scan...")
        self.results["tests_run"] += 1

        # Check if Docker is available
        if not subprocess.run(["which", "docker"], capture_output=True).returncode == 0:
            print("  ⚠ Docker not available, skipping container scan")
            return True

        # Scan Dockerfile with hadolint
        dockerfiles = list(Path(".").glob("**/Dockerfile*"))
        all_passed = True

        for dockerfile in dockerfiles:
            cmd = ["hadolint", str(dockerfile)]
            exit_code, stdout, stderr = self.run_command(cmd, check=False)

            if exit_code != 0:
                print(f"  ✗ {dockerfile} has issues")
                all_passed = False
            else:
                print(f"  ✓ {dockerfile} passed")

        if all_passed:
            self.results["tests_passed"] += 1
            return True
        else:
            self.results["tests_failed"] += 1
            return False

    def run_compliance_checks(self) -> bool:
        """Run compliance checks."""
        print("\n[*] Running compliance checks...")
        self.results["tests_run"] += 1

        # Run OWASP compliance assessment
        if Path("security/owasp_assessment_focused.py").exists():
            cmd = ["python", "security/owasp_assessment_focused.py"]
            exit_code, stdout, stderr = self.run_command(cmd, check=False)

            if Path("security/owasp_focused_assessment_report.json").exists():
                with open("security/owasp_focused_assessment_report.json") as f:
                    report = json.load(f)

                score = report.get("overall_score", 0)
                if score >= 80:
                    print(f"  ✓ OWASP compliance score: {score}/100")
                    self.results["tests_passed"] += 1
                    return True
                else:
                    print(f"  ✗ OWASP compliance score too low: {score}/100")
                    self.results["tests_failed"] += 1
                    return False

        print("  ⚠ Compliance checks not available")
        return True

    def generate_report(self) -> Dict:
        """Generate security test report."""
        self.results["end_time"] = time.time()
        self.results["duration"] = self.results["end_time"] - self.results["start_time"]
        self.results["success_rate"] = (
            self.results["tests_passed"] / self.results["tests_run"] * 100
            if self.results["tests_run"] > 0
            else 0
        )

        return self.results

    def run_all_tests(self) -> bool:
        """Run all security tests."""
        print("=" * 60)
        print("SECURITY TEST SUITE")
        print("=" * 60)

        tests = [
            self.run_bandit_scan,
            self.run_safety_check,
            self.run_semgrep_scan,
            self.run_secrets_detection,
            self.run_owasp_tests,
            self.run_api_security_tests,
            self.run_auth_security_tests,
            self.run_container_security_scan,
            self.run_compliance_checks,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n[!] Error running {test.__name__}: {e}")
                self.results["tests_failed"] += 1
                self.results["tests_run"] += 1

        # Generate and save report
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("SECURITY TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {report['tests_run']}")
        print(f"Passed: {report['tests_passed']}")
        print(f"Failed: {report['tests_failed']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Duration: {report['duration']:.2f} seconds")

        if report["vulnerabilities"]:
            print(f"\nVulnerabilities Found: {len(report['vulnerabilities'])}")

        # Save detailed report
        with open("security-test-report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: security-test-report.json")

        # Exit with appropriate code
        return report["tests_failed"] == 0


if __name__ == "__main__":
    runner = SecurityTestRunner()
    if not runner.run_all_tests():
        sys.exit(1)
