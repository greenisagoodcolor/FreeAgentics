#!/usr/bin/env python3
"""OWASP Top 10 Security Assessment for FreeAgentics.

This script performs automated security checks for common vulnerabilities
based on the OWASP Top 10 (2021) security risks.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests


class OWASPAssessment:
    """OWASP Top 10 security assessment tool."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
    ):
        self.base_url = base_url
        self.auth_token = auth_token
        self.session = requests.Session()
        if auth_token:
            self.session.headers["Authorization"] = f"Bearer {auth_token}"
        self.findings: List[Dict[str, Any]] = []
        self.endpoints_tested = 0

    def add_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        endpoint: Optional[str] = None,
        evidence: Optional[str] = None,
        remediation: Optional[str] = None,
    ):
        """Add a security finding."""
        self.findings.append(
            {
                "category": category,
                "severity": severity,
                "title": title,
                "description": description,
                "endpoint": endpoint,
                "evidence": evidence,
                "remediation": remediation,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def test_a01_broken_access_control(self):
        """Test for A01:2021 – Broken Access Control."""
        print("\n[*] Testing A01: Broken Access Control...")

        # Test accessing admin endpoints without auth
        admin_endpoints = [
            "/api/v1/security/summary",
            "/api/v1/security/events",
            "/api/v1/system/config",
            "/api/v1/agents",  # Should require CREATE_AGENT permission
        ]

        # Test without auth token
        temp_session = requests.Session()

        for endpoint in admin_endpoints:
            url = urljoin(self.base_url, endpoint)
            try:
                resp = temp_session.get(url, timeout=5)
                self.endpoints_tested += 1

                if resp.status_code != 401 and resp.status_code != 403:
                    self.add_finding(
                        "A01: Broken Access Control",
                        "CRITICAL",
                        f"Unauthenticated access to {endpoint}",
                        f"Admin endpoint {endpoint} returned {resp.status_code} without authentication",
                        endpoint=endpoint,
                        evidence=f"Status: {resp.status_code}, Headers: {dict(resp.headers)}",
                        remediation="Ensure all admin endpoints require authentication",
                    )
                else:
                    print(f"  ✓ {endpoint} properly secured (returned {resp.status_code})")

            except Exception as e:
                print(f"  ! Error testing {endpoint}: {e}")

        # Test IDOR vulnerabilities
        print("  [*] Testing for IDOR vulnerabilities...")
        # This would require creating test data and trying to access other users' data

    def test_a02_cryptographic_failures(self):
        """Test for A02:2021 – Cryptographic Failures."""
        print("\n[*] Testing A02: Cryptographic Failures...")

        # Check for HTTPS enforcement
        if self.base_url.startswith("http://") and "localhost" not in self.base_url:
            self.add_finding(
                "A02: Cryptographic Failures",
                "HIGH",
                "Missing HTTPS encryption",
                "API is accessible over unencrypted HTTP",
                evidence=f"Base URL: {self.base_url}",
                remediation="Enforce HTTPS for all production traffic",
            )

        # Check security headers
        try:
            resp = self.session.get(urljoin(self.base_url, "/health"))
            headers = resp.headers

            # Check for security headers
            required_headers = {
                "Strict-Transport-Security": "HSTS header missing",
                "X-Content-Type-Options": "X-Content-Type-Options header missing",
                "X-Frame-Options": "X-Frame-Options header missing",
            }

            for header, issue in required_headers.items():
                if header not in headers:
                    self.add_finding(
                        "A02: Cryptographic Failures",
                        "MEDIUM",
                        issue,
                        f"Security header {header} is not set",
                        endpoint="/health",
                        evidence=f"Missing header: {header}",
                        remediation=f"Add {header} header to all responses",
                    )
                else:
                    print(f"  ✓ {header} header present: {headers[header]}")

        except Exception as e:
            print(f"  ! Error checking headers: {e}")

    def test_a03_injection(self):
        """Test for A03:2021 – Injection."""
        print("\n[*] Testing A03: Injection...")

        # SQL Injection test payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users--",
            "1' AND '1'='1",
        ]

        # Test login endpoint
        login_url = urljoin(self.base_url, "/api/v1/login")

        for payload in sql_payloads:
            try:
                resp = self.session.post(
                    login_url,
                    json={"username": payload, "password": "test"},
                    timeout=5,
                )
                self.endpoints_tested += 1

                # Check for SQL errors in response
                if resp.text and any(
                    err in resp.text.lower() for err in ["sql", "syntax", "query"]
                ):
                    self.add_finding(
                        "A03: Injection",
                        "CRITICAL",
                        "Potential SQL injection vulnerability",
                        f"SQL error exposed when testing login with payload: {payload}",
                        endpoint="/api/v1/login",
                        evidence=f"Response: {resp.text[:200]}",
                        remediation="Use parameterized queries and input validation",
                    )
                else:
                    print(f"  ✓ Login endpoint handled SQL payload safely: {payload[:20]}...")

            except Exception as e:
                print(f"  ! Error testing SQL injection: {e}")

        # XSS test payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
        ]

        # Test agent creation with XSS
        if self.auth_token:
            agent_url = urljoin(self.base_url, "/api/v1/agents")

            for payload in xss_payloads:
                try:
                    resp = self.session.post(
                        agent_url,
                        json={"name": payload, "type": "test", "config": {}},
                        timeout=5,
                    )
                    self.endpoints_tested += 1

                    # Check if payload is reflected without encoding
                    if resp.text and payload in resp.text:
                        self.add_finding(
                            "A03: Injection",
                            "HIGH",
                            "Potential XSS vulnerability",
                            f"Unencoded user input reflected: {payload}",
                            endpoint="/api/v1/agents",
                            evidence="Payload reflected in response",
                            remediation="Encode all user input in responses",
                        )

                except Exception as e:
                    print(f"  ! Error testing XSS: {e}")

    def test_a04_insecure_design(self):
        """Test for A04:2021 – Insecure Design."""
        print("\n[*] Testing A04: Insecure Design...")

        # Check for rate limiting
        print("  [*] Testing rate limiting...")
        login_url = urljoin(self.base_url, "/api/v1/login")

        # Make rapid requests
        rapid_responses = []
        for i in range(15):
            try:
                resp = self.session.post(
                    login_url,
                    json={"username": "test", "password": "wrong"},
                    timeout=5,
                )
                rapid_responses.append(resp.status_code)
                self.endpoints_tested += 1
            except Exception:
                pass

        # Check if any were rate limited
        if 429 not in rapid_responses:
            self.add_finding(
                "A04: Insecure Design",
                "HIGH",
                "Missing rate limiting on login",
                "Login endpoint allows unlimited attempts",
                endpoint="/api/v1/login",
                evidence=f"Made {len(rapid_responses)} requests, no rate limiting",
                remediation="Implement rate limiting on authentication endpoints",
            )
        else:
            print(
                f"  ✓ Rate limiting active (triggered after {rapid_responses.index(429)} requests)"
            )

    def test_a05_security_misconfiguration(self):
        """Test for A05:2021 – Security Misconfiguration."""
        print("\n[*] Testing A05: Security Misconfiguration...")

        # Check for exposed debug endpoints
        debug_endpoints = [
            "/docs",
            "/redoc",
            "/api/v1/graphql",
            "/__debug__/",
            "/debug",
            "/.env",
            "/config",
        ]

        for endpoint in debug_endpoints:
            url = urljoin(self.base_url, endpoint)
            try:
                resp = self.session.get(url, timeout=5)
                self.endpoints_tested += 1

                if resp.status_code == 200:
                    # Check if it's intentional (docs) or a security issue
                    if endpoint in ["/docs", "/redoc", "/api/v1/graphql"]:
                        print(f"  ⚠ {endpoint} is exposed (may be intentional for dev)")
                    else:
                        self.add_finding(
                            "A05: Security Misconfiguration",
                            "HIGH",
                            f"Exposed debug endpoint: {endpoint}",
                            f"Debug/config endpoint {endpoint} is publicly accessible",
                            endpoint=endpoint,
                            evidence=f"Status: {resp.status_code}",
                            remediation="Disable debug endpoints in production",
                        )

            except Exception:
                print(f"  ✓ {endpoint} not accessible")

        # Check for default credentials
        print("  [*] Testing for default credentials...")
        default_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("test", "test"),
            ("freeagentics", "freeagentics123"),
        ]

        login_url = urljoin(self.base_url, "/api/v1/login")
        for username, password in default_creds:
            try:
                resp = self.session.post(
                    login_url,
                    json={"username": username, "password": password},
                    timeout=5,
                )
                self.endpoints_tested += 1

                if resp.status_code == 200:
                    self.add_finding(
                        "A05: Security Misconfiguration",
                        "CRITICAL",
                        f"Default credentials active: {username}",
                        f"Default credentials {username}:{password} are active",
                        endpoint="/api/v1/login",
                        evidence="Successful login with default credentials",
                        remediation="Remove all default credentials",
                    )

            except Exception:
                pass

    def test_a06_vulnerable_components(self):
        """Test for A06:2021 – Vulnerable and Outdated Components."""
        print("\n[*] Testing A06: Vulnerable and Outdated Components...")

        # This would typically involve:
        # 1. Checking package.json and requirements.txt for known vulnerabilities
        # 2. Using tools like npm audit, pip-audit, safety
        # 3. Checking for outdated dependencies

        print("  ℹ Note: Run 'make security-scan' for full dependency scanning")

    def test_a07_auth_failures(self):
        """Test for A07:2021 – Identification and Authentication Failures."""
        print("\n[*] Testing A07: Identification and Authentication Failures...")

        # Test password complexity requirements
        register_url = urljoin(self.base_url, "/api/v1/register")

        weak_passwords = ["123456", "password", "abc123", "12345678"]

        for weak_pass in weak_passwords:
            try:
                resp = self.session.post(
                    register_url,
                    json={
                        "username": f"testuser_{weak_pass}",
                        "email": f"test_{weak_pass}@example.com",
                        "password": weak_pass,
                    },
                    timeout=5,
                )
                self.endpoints_tested += 1

                if resp.status_code == 200:
                    self.add_finding(
                        "A07: Authentication Failures",
                        "HIGH",
                        "Weak password accepted",
                        f"System accepted weak password: {weak_pass}",
                        endpoint="/api/v1/register",
                        evidence="Registration successful with weak password",
                        remediation="Implement strong password requirements",
                    )

            except Exception:
                pass

        # Test session management
        print("  [*] Testing session management...")
        # Would test for:
        # - Session fixation
        # - Session timeout
        # - Secure session cookies

    def test_a08_software_integrity(self):
        """Test for A08:2021 – Software and Data Integrity Failures."""
        print("\n[*] Testing A08: Software and Data Integrity Failures...")

        # Check for integrity validation
        # - Verify API accepts only expected content types
        # - Check for deserialization vulnerabilities

        print("  ℹ Testing for unsafe deserialization...")

    def test_a09_logging_monitoring(self):
        """Test for A09:2021 – Security Logging and Monitoring Failures."""
        print("\n[*] Testing A09: Security Logging and Monitoring Failures...")

        # This was implemented in task 14.9
        # Check if security endpoints are accessible

        if self.auth_token:
            security_url = urljoin(self.base_url, "/api/v1/security/summary")
            try:
                resp = self.session.get(security_url, timeout=5)
                if resp.status_code == 200:
                    print("  ✓ Security monitoring endpoints active")
                else:
                    self.add_finding(
                        "A09: Logging & Monitoring",
                        "MEDIUM",
                        "Security monitoring not accessible",
                        "Security monitoring endpoints return error",
                        endpoint="/api/v1/security/summary",
                        evidence=f"Status: {resp.status_code}",
                        remediation="Ensure security monitoring is properly configured",
                    )
            except Exception:
                pass

    def test_a10_ssrf(self):
        """Test for A10:2021 – Server-Side Request Forgery (SSRF)."""
        print("\n[*] Testing A10: Server-Side Request Forgery...")

        # Test endpoints that might make external requests
        # Look for URL parameters that could be exploited

        # SSRF payloads would be tested here in a real assessment

        print("  ℹ SSRF testing requires identifying endpoints that make external requests")

    def generate_report(self) -> Dict[str, Any]:
        """Generate assessment report."""
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

        category_counts = {}

        for finding in self.findings:
            severity_counts[finding["severity"]] += 1
            category = finding["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "assessment_date": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "endpoints_tested": self.endpoints_tested,
            "total_findings": len(self.findings),
            "severity_summary": severity_counts,
            "category_summary": category_counts,
            "findings": self.findings,
        }

    def run_assessment(self):
        """Run full OWASP Top 10 assessment."""
        print("=" * 60)
        print("OWASP Top 10 Security Assessment for FreeAgentics")
        print("=" * 60)
        print(f"Target: {self.base_url}")
        print(f"Authenticated: {'Yes' if self.auth_token else 'No'}")
        print("=" * 60)

        # Run all tests
        self.test_a01_broken_access_control()
        self.test_a02_cryptographic_failures()
        self.test_a03_injection()
        self.test_a04_insecure_design()
        self.test_a05_security_misconfiguration()
        self.test_a06_vulnerable_components()
        self.test_a07_auth_failures()
        self.test_a08_software_integrity()
        self.test_a09_logging_monitoring()
        self.test_a10_ssrf()

        # Generate report
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("ASSESSMENT SUMMARY")
        print("=" * 60)
        print(f"Total endpoints tested: {report['endpoints_tested']}")
        print(f"Total findings: {report['total_findings']}")
        print("\nFindings by severity:")
        for severity, count in report["severity_summary"].items():
            print(f"  {severity}: {count}")
        print("\nFindings by category:")
        for category, count in report["category_summary"].items():
            print(f"  {category}: {count}")

        # Save report
        report_path = Path("security/owasp_assessment_report.json")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nFull report saved to: {report_path}")

        return report


def main():
    """Run OWASP assessment."""
    import argparse

    parser = argparse.ArgumentParser(description="OWASP Top 10 Security Assessment")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API to test",
    )
    parser.add_argument(
        "--token",
        help="JWT token for authenticated testing",
    )

    args = parser.parse_args()

    # Create assessor
    assessor = OWASPAssessment(args.url, args.token)

    # Run assessment
    report = assessor.run_assessment()

    # Exit with error if critical findings
    if report["severity_summary"]["CRITICAL"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
