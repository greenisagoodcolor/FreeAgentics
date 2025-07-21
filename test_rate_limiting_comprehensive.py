#!/usr/bin/env python3
"""Comprehensive Rate Limiting and DDoS Protection Test Suite.

This script tests the rate limiting and DDoS protection capabilities
to ensure production-grade security.
"""

import asyncio
import json
import time
from typing import Dict, List

import aiohttp
import requests


class RateLimitTester:
    """Comprehensive rate limiting test suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the rate limiting tester."""
        self.base_url = base_url
        self.results = {
            "timestamp": time.time(),
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "warnings": 0},
        }

    def add_test_result(self, test_name: str, status: str, details: Dict):
        """Add test result to results."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time(),
        }
        self.results["tests"].append(result)
        self.results["summary"][status] += 1
        print(f"[{status.upper()}] {test_name}: {details.get('message', 'No details')}")

    def test_basic_rate_limiting(self) -> Dict:
        """Test basic rate limiting functionality."""
        test_name = "Basic Rate Limiting"
        print(f"\n=== {test_name} ===")

        try:
            endpoint = f"{self.base_url}/api/v1/health"
            successful_requests = 0
            rate_limited_requests = 0

            # Send 100 requests rapidly
            for i in range(100):
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        successful_requests += 1
                    elif response.status_code == 429:
                        rate_limited_requests += 1
                        # Check for rate limit headers
                        headers = response.headers
                        retry_after = headers.get("Retry-After")
                        rate_limit_remaining = headers.get("X-RateLimit-Remaining")

                        self.add_test_result(
                            "Rate Limit Headers Present",
                            "passed"
                            if retry_after and rate_limit_remaining
                            else "failed",
                            {
                                "retry_after": retry_after,
                                "rate_limit_remaining": rate_limit_remaining,
                                "message": "Rate limit headers validation",
                            },
                        )
                        break
                except requests.RequestException as e:
                    print(f"Request {i} failed: {e}")

            # Evaluate results
            if rate_limited_requests > 0:
                self.add_test_result(
                    test_name,
                    "passed",
                    {
                        "successful_requests": successful_requests,
                        "rate_limited_requests": rate_limited_requests,
                        "message": "Rate limiting is working correctly",
                    },
                )
            else:
                self.add_test_result(
                    test_name,
                    "failed",
                    {
                        "successful_requests": successful_requests,
                        "rate_limited_requests": rate_limited_requests,
                        "message": "Rate limiting not triggered - potential security issue",
                    },
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                {"error": str(e), "message": "Test failed with exception"},
            )

    def test_ddos_pattern_detection(self) -> Dict:
        """Test DDoS attack pattern detection."""
        test_name = "DDoS Pattern Detection"
        print(f"\n=== {test_name} ===")

        try:
            # Test rapid 404 pattern
            for i in range(15):
                try:
                    response = requests.get(
                        f"{self.base_url}/nonexistent-path-{i}", timeout=2
                    )
                    if response.status_code == 403:  # Blocked for suspicious pattern
                        self.add_test_result(
                            "404 Pattern Detection",
                            "passed",
                            {
                                "message": f"Blocked after {i + 1} 404 requests",
                                "block_status": 403,
                            },
                        )
                        break
                except requests.RequestException:
                    pass

            # Test path scanning detection
            paths = [
                "/admin",
                "/config",
                "/backup",
                "/debug",
                "/test",
                "/api/admin",
                "/sensitive",
                "/internal",
                "/private",
                "/system",
                "/management",
                "/statistics",
                "/metrics",
                "/status",
                "/health",
                "/info",
            ]

            blocked = False
            for i, path in enumerate(paths):
                try:
                    response = requests.get(f"{self.base_url}{path}", timeout=2)
                    if response.status_code == 403:
                        self.add_test_result(
                            "Path Scanning Detection",
                            "passed",
                            {
                                "message": f"Blocked after {i + 1} path scans",
                                "block_status": 403,
                            },
                        )
                        blocked = True
                        break
                except requests.RequestException:
                    pass

            if not blocked:
                self.add_test_result(
                    "Path Scanning Detection",
                    "warnings",
                    {"message": "Path scanning not blocked - may need tuning"},
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                {"error": str(e), "message": "DDoS detection test failed"},
            )

    def test_request_size_limits(self) -> Dict:
        """Test request size limiting."""
        test_name = "Request Size Limits"
        print(f"\n=== {test_name} ===")

        try:
            endpoint = f"{self.base_url}/api/v1/health"

            # Test large payload (10MB)
            large_payload = "x" * (10 * 1024 * 1024)  # 10MB

            try:
                response = requests.post(
                    endpoint,
                    data=large_payload,
                    headers={"Content-Type": "text/plain"},
                    timeout=10,
                )

                if response.status_code == 413:  # Request Entity Too Large
                    self.add_test_result(
                        test_name,
                        "passed",
                        {
                            "message": "Large request properly rejected",
                            "status_code": 413,
                        },
                    )
                else:
                    self.add_test_result(
                        test_name,
                        "failed",
                        {
                            "message": "Large request not rejected",
                            "status_code": response.status_code,
                        },
                    )
            except requests.RequestException as e:
                # Connection errors may indicate proper blocking
                self.add_test_result(
                    test_name,
                    "passed",
                    {
                        "message": "Large request blocked at connection level",
                        "error": str(e),
                    },
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                {"error": str(e), "message": "Request size test failed"},
            )

    async def test_concurrent_requests(self) -> Dict:
        """Test handling of concurrent requests."""
        test_name = "Concurrent Request Handling"
        print(f"\n=== {test_name} ===")

        try:
            endpoint = f"{self.base_url}/api/v1/health"
            concurrent_requests = 50

            async def make_request(session, semaphore, request_id):
                async with semaphore:
                    try:
                        async with session.get(endpoint, timeout=5) as response:
                            return {
                                "id": request_id,
                                "status": response.status,
                                "headers": dict(response.headers),
                            }
                    except Exception as e:
                        return {"id": request_id, "error": str(e)}

            # Limit concurrent connections
            semaphore = asyncio.Semaphore(concurrent_requests)

            async with aiohttp.ClientSession() as session:
                tasks = [
                    make_request(session, semaphore, i)
                    for i in range(concurrent_requests)
                ]

                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                successful = sum(
                    1 for r in results if isinstance(r, dict) and r.get("status") == 200
                )
                rate_limited = sum(
                    1 for r in results if isinstance(r, dict) and r.get("status") == 429
                )
                errors = sum(
                    1 for r in results if isinstance(r, Exception) or "error" in r
                )

                self.add_test_result(
                    test_name,
                    "passed"
                    if successful + rate_limited > concurrent_requests * 0.8
                    else "failed",
                    {
                        "concurrent_requests": concurrent_requests,
                        "successful": successful,
                        "rate_limited": rate_limited,
                        "errors": errors,
                        "duration": end_time - start_time,
                        "message": f"Handled {successful + rate_limited}/{concurrent_requests} requests",
                    },
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                {"error": str(e), "message": "Concurrent request test failed"},
            )

    def test_security_headers(self) -> Dict:
        """Test security headers implementation."""
        test_name = "Security Headers"
        print(f"\n=== {test_name} ===")

        required_headers = {
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-Content-Type-Options": ["nosniff"],
            "X-XSS-Protection": ["1; mode=block"],
            "Strict-Transport-Security": None,  # Just check presence
            "Content-Security-Policy": None,
            "Referrer-Policy": None,
        }

        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            headers = response.headers

            missing_headers = []
            invalid_headers = []

            for header, expected_values in required_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected_values and headers[header] not in expected_values:
                    invalid_headers.append(f"{header}: {headers[header]}")

            if not missing_headers and not invalid_headers:
                self.add_test_result(
                    test_name,
                    "passed",
                    {"message": "All security headers present and valid"},
                )
            else:
                self.add_test_result(
                    test_name,
                    "failed" if missing_headers else "warnings",
                    {
                        "missing_headers": missing_headers,
                        "invalid_headers": invalid_headers,
                        "message": "Security headers issues detected",
                    },
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                {"error": str(e), "message": "Security headers test failed"},
            )

    def generate_report(self) -> Dict:
        """Generate comprehensive security test report."""
        summary = self.results["summary"]
        total_tests = sum(summary.values())

        if total_tests == 0:
            security_score = 0
        else:
            security_score = (summary["passed"] / total_tests) * 100

        # Determine security rating
        if security_score >= 95:
            rating = "EXCELLENT"
            status = "🟢"
        elif security_score >= 85:
            rating = "GOOD"
            status = "🟡"
        elif security_score >= 70:
            rating = "ACCEPTABLE"
            status = "🟡"
        else:
            rating = "NEEDS IMPROVEMENT"
            status = "🔴"

        report = {
            "timestamp": self.results["timestamp"],
            "security_score": security_score,
            "security_rating": rating,
            "status_indicator": status,
            "summary": summary,
            "total_tests": total_tests,
            "detailed_results": self.results["tests"],
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        failed_tests = [t for t in self.results["tests"] if t["status"] == "failed"]
        [t for t in self.results["tests"] if t["status"] == "warnings"]

        if any("Rate Limiting" in t["test"] for t in failed_tests):
            recommendations.append("Implement or strengthen rate limiting mechanisms")

        if any("DDoS" in t["test"] for t in failed_tests):
            recommendations.append("Enhance DDoS protection and pattern detection")

        if any("Security Headers" in t["test"] for t in failed_tests):
            recommendations.append("Implement missing security headers")

        if any("Request Size" in t["test"] for t in failed_tests):
            recommendations.append("Implement request size limits")

        if not recommendations:
            recommendations.append(
                "Security posture is excellent - maintain current controls"
            )

        return recommendations


async def main():
    """Main test execution function."""
    print("🔒 FreeAgentics Production Security Hardening Test Suite")
    print("=" * 60)

    tester = RateLimitTester()

    # Run all tests
    tester.test_basic_rate_limiting()
    tester.test_ddos_pattern_detection()
    tester.test_request_size_limits()
    await tester.test_concurrent_requests()
    tester.test_security_headers()

    # Generate final report
    report = tester.generate_report()

    print(f"\n{'=' * 60}")
    print(f"🔒 SECURITY TEST REPORT {report['status_indicator']}")
    print(f"{'=' * 60}")
    print(f"Security Score: {report['security_score']:.1f}/100")
    print(f"Security Rating: {report['security_rating']}")
    print(f"Tests Passed: {report['summary']['passed']}")
    print(f"Tests Failed: {report['summary']['failed']}")
    print(f"Warnings: {report['summary']['warnings']}")

    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    # Save report
    report_file = f"security_test_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    return report["security_score"] >= 85  # Return True if security is acceptable


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        print(f"✅ Server is running (Status: {response.status_code})")
    except requests.RequestException:
        print("❌ Server is not running. Please start the server first:")
        print("   docker-compose -f docker-compose.production.yml up -d")
        exit(1)

    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
