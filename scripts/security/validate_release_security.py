#!/usr/bin/env python3
"""
Validate security requirements before release.
"""

import json
import sys
from pathlib import Path
from typing import Dict


class ReleaseSecurityValidator:
    """Validates security requirements for release."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0

    def validate_security_tests(self) -> bool:
        """Ensure all security tests pass."""
        try:
            # Check for test results
            test_results_path = Path(".pytest_cache/v/cache/lastfailed")
            if test_results_path.exists():
                with open(test_results_path) as f:
                    failed_tests = json.load(f)
                    security_failures = [
                        test for test in failed_tests if "security" in test or "auth" in test
                    ]
                    if security_failures:
                        self.errors.append(f"Security tests failed: {len(security_failures)} tests")
                        return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating security tests: {e}")
            return False

    def validate_vulnerability_scan(self) -> bool:
        """Check vulnerability scan results."""
        try:
            # Check Bandit results
            bandit_report = Path("bandit-report.json")
            if bandit_report.exists():
                with open(bandit_report) as f:
                    results = json.load(f)

                high_severity = sum(
                    1 for r in results.get("results", []) if r.get("issue_severity") == "HIGH"
                )

                if high_severity > 0:
                    self.errors.append(f"High severity vulnerabilities found: {high_severity}")
                    return False

            # Check dependency vulnerabilities
            safety_report = Path("safety-report.json")
            if safety_report.exists():
                with open(safety_report) as f:
                    vulnerabilities = json.load(f)
                    if vulnerabilities:
                        self.errors.append(f"Dependency vulnerabilities: {len(vulnerabilities)}")
                        return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error checking vulnerabilities: {e}")
            return False

    def validate_security_headers(self) -> bool:
        """Validate security headers configuration."""
        try:
            # Check middleware configuration
            middleware_path = Path("api/middleware/security_headers.py")
            if not middleware_path.exists():
                self.errors.append("Security headers middleware not found")
                return False

            with open(middleware_path) as f:
                content = f.read()

            required_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy",
                "Referrer-Policy",
                "Permissions-Policy",
            ]

            missing_headers = []
            for header in required_headers:
                if header not in content:
                    missing_headers.append(header)

            if missing_headers:
                self.errors.append(f"Missing security headers: {', '.join(missing_headers)}")
                return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating headers: {e}")
            return False

    def validate_authentication(self) -> bool:
        """Validate authentication configuration."""
        try:
            # Check JWT configuration
            auth_path = Path("auth/security_implementation.py")
            if not auth_path.exists():
                self.errors.append("Authentication implementation not found")
                return False

            with open(auth_path) as f:
                content = f.read()

            # Check for secure practices
            checks = {
                "JWT expiration": "expires_delta" in content,
                "Password hashing": "bcrypt" in content or "argon2" in content,
                "Secure random": "secrets" in content,
                "Rate limiting": "rate_limit" in content.lower(),
            }

            failed_checks = [check for check, passed in checks.items() if not passed]

            if failed_checks:
                self.errors.append(f"Authentication issues: {', '.join(failed_checks)}")
                return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating authentication: {e}")
            return False

    def validate_encryption(self) -> bool:
        """Validate encryption configuration."""
        try:
            # Check for encryption usage
            encryption_patterns = [
                "from cryptography",
                "encrypt(",
                "decrypt(",
                "Fernet",
                "AES",
                "RSA",
            ]

            encryption_found = False
            for py_file in Path(".").rglob("*.py"):
                if ".archive" in str(py_file) or "test" in str(py_file):
                    continue

                try:
                    with open(py_file) as f:
                        content = f.read()
                        if any(pattern in content for pattern in encryption_patterns):
                            encryption_found = True
                            break
                except:
                    continue

            if not encryption_found:
                self.warnings.append("No encryption implementation found")

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating encryption: {e}")
            return False

    def validate_compliance(self) -> bool:
        """Validate compliance requirements."""
        try:
            # Check OWASP compliance
            owasp_report = Path("security/owasp_focused_assessment_report.json")
            if owasp_report.exists():
                with open(owasp_report) as f:
                    report = json.load(f)

                score = report.get("overall_score", 0)
                if score < 80:
                    self.errors.append(f"OWASP compliance score too low: {score}/100")
                    return False
            else:
                self.warnings.append("OWASP assessment report not found")

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating compliance: {e}")
            return False

    def validate_secrets(self) -> bool:
        """Ensure no secrets in code."""
        try:
            # Check detect-secrets baseline
            baseline_path = Path(".secrets.baseline")
            if baseline_path.exists():
                with open(baseline_path) as f:
                    baseline = json.load(f)

                # Check for unaudited secrets
                total_secrets = 0
                for file_secrets in baseline.get("results", {}).values():
                    total_secrets += len(file_secrets)

                if total_secrets > 0:
                    self.errors.append(f"Potential secrets found: {total_secrets}")
                    return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error checking secrets: {e}")
            return False

    def validate_container_security(self) -> bool:
        """Validate container security."""
        try:
            # Check Dockerfile best practices
            dockerfiles = list(Path(".").glob("**/Dockerfile*"))

            for dockerfile in dockerfiles:
                with open(dockerfile) as f:
                    content = f.read()

                # Check for security issues
                issues = []

                if (
                    "USER root" in content
                    and "USER" not in content[content.find("USER root") + 9 :]
                ):
                    issues.append("Running as root user")

                if "--no-cache" not in content:
                    self.warnings.append(f"{dockerfile}: Consider using --no-cache")

                if "COPY . ." in content or "ADD . ." in content:
                    issues.append("Copying entire context")

                if issues:
                    self.errors.append(f"{dockerfile}: {', '.join(issues)}")
                    return False

            self.checks_passed += 1
            return True

        except Exception as e:
            self.errors.append(f"Error validating containers: {e}")
            return False

    def generate_report(self) -> Dict:
        """Generate validation report."""
        return {
            "passed": self.checks_failed == 0,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "errors": self.errors,
            "warnings": self.warnings,
            "security_score": (
                self.checks_passed / (self.checks_passed + self.checks_failed) * 100
                if (self.checks_passed + self.checks_failed) > 0
                else 0
            ),
        }

    def run_validation(self) -> bool:
        """Run all security validations."""
        validations = [
            ("Security Tests", self.validate_security_tests),
            ("Vulnerability Scan", self.validate_vulnerability_scan),
            ("Security Headers", self.validate_security_headers),
            ("Authentication", self.validate_authentication),
            ("Encryption", self.validate_encryption),
            ("Compliance", self.validate_compliance),
            ("Secrets Detection", self.validate_secrets),
            ("Container Security", self.validate_container_security),
        ]

        print("Running release security validation...")
        print("=" * 50)

        for name, validator in validations:
            print(f"\nValidating {name}...", end=" ")
            try:
                if validator():
                    print("✓ PASSED")
                else:
                    print("✗ FAILED")
                    self.checks_failed += 1
            except Exception as e:
                print(f"✗ ERROR: {e}")
                self.checks_failed += 1

        print("\n" + "=" * 50)
        report = self.generate_report()

        print(f"\nSecurity Score: {report['security_score']:.1f}/100")
        print(f"Checks Passed: {report['checks_passed']}")
        print(f"Checks Failed: {report['checks_failed']}")

        if report["errors"]:
            print("\nErrors:")
            for error in report["errors"]:
                print(f"  ✗ {error}")

        if report["warnings"]:
            print("\nWarnings:")
            for warning in report["warnings"]:
                print(f"  ⚠ {warning}")

        # Save report
        with open("release-security-validation.json", "w") as f:
            json.dump(report, f, indent=2)

        return report["passed"]


if __name__ == "__main__":
    validator = ReleaseSecurityValidator()
    if not validator.run_validation():
        sys.exit(1)
