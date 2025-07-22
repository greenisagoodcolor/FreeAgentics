"""
API Security Response Validation Testing

Tests for validating secure API responses, proper error handling,
security headers, rate limiting, and response sanitization.

This module focuses on:
1. Security headers validation
2. Rate limiting response testing
3. CORS policy validation
4. Content-Type security
5. Response sanitization
6. Error response consistency
7. Security information leakage in headers
8. Response timing consistency
"""

import json
import re
import time
from typing import Any, Dict
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api.main import app


class APISecurityResponseTester:
    """Comprehensive API security response tester."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []

        # Required security headers for production
        self.required_security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": [
                "strict-origin-when-cross-origin",
                "strict-origin",
                "no-referrer",
            ],
            "Content-Security-Policy": None,  # Should exist but content varies
            "Permissions-Policy": None,  # Should exist but content varies
        }

        # Headers that should NOT be present (information disclosure)
        self.forbidden_headers = [
            "Server",  # Web server version
            "X-Powered-By",  # Technology stack
            "X-AspNet-Version",  # .NET version
            "X-AspNetMvc-Version",  # MVC version
            "X-Debug-Token",  # Debug information
            "X-Debug-Token-Link",  # Debug links
            "X-Generator",  # Generator information
            "X-Drupal-Cache",  # Drupal specific
            "X-Varnish",  # Varnish cache info
            "Via",  # Proxy information (in some contexts)
            "X-Cache",  # Cache information
            "X-Cache-Hits",  # Cache hit information
            "X-Backend-Server",  # Backend server info
            "X-Node",  # Node.js specific
            "X-Runtime",  # Runtime information
            "X-Served-By",  # Server identification
        ]

        # Patterns that should not appear in any response
        self.sensitive_response_patterns = [
            r"password.*=",
            r"secret.*=",
            r"key.*=.*[a-zA-Z0-9]{10,}",
            r"token.*=.*[a-zA-Z0-9]{10,}",
            r"api.*key.*[a-zA-Z0-9]{10,}",
            r"database.*url",
            r"redis.*url",
            r"mongodb.*uri",
            r"connection.*string",
            r"dsn.*=",
            r"\.env",
            r"config\..*",
            r"settings\..*",
            r"/etc/passwd",
            r"/etc/shadow",
            r"C:\\Windows\\",
            r"/var/log/",
            r"/tmp/",
            r"stack.*trace",
            r"traceback",
            r"exception.*at.*line",
            r"error.*at.*line.*\d+",
            r'file.*".*\.py".*line.*\d+',
        ]

    def test_security_headers_presence(self) -> Dict[str, Any]:
        """Test that required security headers are present."""
        results = {
            "test_name": "security_headers_presence",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various endpoints
        test_endpoints = [
            "/",
            "/health",
            "/api/v1/system/status",
            "/api/v1/agents",
            "/api/v1/auth/login",
            "/nonexistent",  # 404 response
        ]

        for endpoint in test_endpoints:
            try:
                response = self.client.get(endpoint)

                # Check required headers
                missing_headers = []
                for (
                    header_name,
                    expected_values,
                ) in self.required_security_headers.items():
                    if header_name not in response.headers:
                        missing_headers.append(header_name)
                    elif expected_values is not None:
                        actual_value = response.headers[header_name]
                        if isinstance(expected_values, list):
                            if actual_value not in expected_values:
                                results["findings"].append(
                                    {
                                        "issue": f"Invalid value for {header_name}",
                                        "endpoint": endpoint,
                                        "expected": expected_values,
                                        "actual": actual_value,
                                    }
                                )
                        elif isinstance(expected_values, str):
                            if actual_value != expected_values:
                                results["findings"].append(
                                    {
                                        "issue": f"Invalid value for {header_name}",
                                        "endpoint": endpoint,
                                        "expected": expected_values,
                                        "actual": actual_value,
                                    }
                                )

                if missing_headers:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Missing required security headers",
                            "endpoint": endpoint,
                            "missing_headers": missing_headers,
                        }
                    )

                # Check for forbidden headers
                forbidden_found = []
                for forbidden_header in self.forbidden_headers:
                    if forbidden_header in response.headers:
                        forbidden_found.append(
                            {
                                "header": forbidden_header,
                                "value": response.headers[forbidden_header],
                            }
                        )

                if forbidden_found:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Information disclosure headers present",
                            "endpoint": endpoint,
                            "forbidden_headers": forbidden_found,
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing endpoint {endpoint}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement comprehensive security headers middleware",
                    "Remove or customize server identification headers",
                    "Ensure consistent security headers across all endpoints",
                ]
            )

        return results

    def test_rate_limiting_responses(self) -> Dict[str, Any]:
        """Test rate limiting implementation and responses."""
        results = {
            "test_name": "rate_limiting_responses",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test rate limiting on various endpoints
        rate_limit_endpoints = [
            "/api/v1/auth/login",
            "/api/v1/agents",
            "/api/v1/system/status",
        ]

        for endpoint in rate_limit_endpoints:
            try:
                # Make rapid requests to trigger rate limiting
                responses = []
                for i in range(15):  # Try to exceed typical rate limits
                    response = self.client.post(endpoint, json={"test": f"request_{i}"})
                    responses.append(response)

                    # Small delay to avoid overwhelming the server
                    time.sleep(0.01)

                # Check if rate limiting kicked in
                rate_limited = False
                rate_limit_headers_found = False

                for response in responses:
                    if response.status_code == 429:  # Too Many Requests
                        rate_limited = True

                        # Check for proper rate limit headers
                        rate_limit_headers = [
                            "X-RateLimit-Limit",
                            "X-RateLimit-Remaining",
                            "X-RateLimit-Reset",
                            "Retry-After",
                        ]

                        for header in rate_limit_headers:
                            if header in response.headers:
                                rate_limit_headers_found = True
                                break

                        # Check response content for information disclosure
                        response_text = response.text
                        for pattern in self.sensitive_response_patterns:
                            if re.search(pattern, response_text, re.IGNORECASE):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Sensitive information in rate limit response",
                                        "endpoint": endpoint,
                                        "pattern": pattern,
                                        "response_preview": response_text[:200],
                                    }
                                )

                # Rate limiting should be implemented
                if not rate_limited:
                    results["findings"].append(
                        {
                            "issue": "Rate limiting not detected",
                            "endpoint": endpoint,
                            "note": "May indicate missing or ineffective rate limiting",
                        }
                    )

                # Rate limit headers should be present when rate limited
                if rate_limited and not rate_limit_headers_found:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Missing rate limit headers",
                            "endpoint": endpoint,
                            "note": "Rate limit responses should include proper headers",
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing rate limiting for {endpoint}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement proper rate limiting with informative headers",
                    "Ensure rate limit responses do not disclose sensitive information",
                    "Use consistent rate limiting across all API endpoints",
                ]
            )

        return results

    def test_cors_policy_validation(self) -> Dict[str, Any]:
        """Test CORS policy configuration and responses."""
        results = {
            "test_name": "cors_policy_validation",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test CORS with various origins
        test_origins = [
            "https://example.com",
            "https://malicious-site.com",
            "http://localhost:3000",
            "https://freeagentics.com",
            "null",
            "*",
        ]

        test_endpoint = "/api/v1/system/status"

        for origin in test_origins:
            try:
                headers = {"Origin": origin}

                # Test preflight request
                preflight_response = self.client.options(
                    test_endpoint,
                    headers={
                        **headers,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type,Authorization",
                    },
                )

                # Test actual request
                actual_response = self.client.get(test_endpoint, headers=headers)

                # Check CORS headers in responses
                for response in [preflight_response, actual_response]:
                    cors_headers = {
                        "Access-Control-Allow-Origin": response.headers.get(
                            "Access-Control-Allow-Origin"
                        ),
                        "Access-Control-Allow-Credentials": response.headers.get(
                            "Access-Control-Allow-Credentials"
                        ),
                        "Access-Control-Allow-Methods": response.headers.get(
                            "Access-Control-Allow-Methods"
                        ),
                        "Access-Control-Allow-Headers": response.headers.get(
                            "Access-Control-Allow-Headers"
                        ),
                    }

                    # Check for overly permissive CORS
                    if (
                        cors_headers["Access-Control-Allow-Origin"] == "*"
                        and cors_headers["Access-Control-Allow-Credentials"] == "true"
                    ):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Dangerous CORS configuration",
                                "detail": "Allow-Origin: * with Allow-Credentials: true",
                                "origin_tested": origin,
                            }
                        )

                    # Check if unknown origins are allowed
                    if (
                        origin in ["https://malicious-site.com"]
                        and cors_headers["Access-Control-Allow-Origin"] == origin
                    ):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Unknown origin allowed by CORS",
                                "origin": origin,
                                "cors_response": cors_headers,
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing CORS with origin {origin}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Review and restrict CORS policy to known origins only",
                    "Avoid using Access-Control-Allow-Origin: * with credentials",
                    "Implement proper CORS validation for all endpoints",
                ]
            )

        return results

    def test_content_type_security(self) -> Dict[str, Any]:
        """Test Content-Type security and validation."""
        results = {
            "test_name": "content_type_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various content types that could be dangerous
        test_scenarios = [
            ("text/html", '<script>alert("xss")</script>'),
            ("application/javascript", 'alert("xss")'),
            ("text/javascript", 'alert("xss")'),
            ("application/xml", '<?xml version="1.0"?><root>test</root>'),
            ("text/xml", '<?xml version="1.0"?><root>test</root>'),
            ("image/svg+xml", '<svg><script>alert("xss")</script></svg>'),
            ("text/plain", "plain text"),
            ("application/octet-stream", b"binary data"),
        ]

        test_endpoint = "/api/v1/agents"

        for content_type, payload in test_scenarios:
            try:
                headers = {"Content-Type": content_type}

                if isinstance(payload, str):
                    response = self.client.post(test_endpoint, data=payload, headers=headers)
                else:
                    response = self.client.post(test_endpoint, content=payload, headers=headers)

                # Check response content type
                response_content_type = response.headers.get("Content-Type", "")

                # Responses should generally be JSON for API endpoints
                if test_endpoint.startswith("/api/") and not response_content_type.startswith(
                    "application/json"
                ):
                    results["findings"].append(
                        {
                            "issue": "Non-JSON response from API endpoint",
                            "endpoint": test_endpoint,
                            "request_content_type": content_type,
                            "response_content_type": response_content_type,
                            "status_code": response.status_code,
                        }
                    )

                # Check if dangerous content types are reflected
                response_text = response.text
                if content_type in [
                    "text/html",
                    "application/javascript",
                    "text/javascript",
                ]:
                    if any(
                        dangerous in response_text
                        for dangerous in ["<script>", "javascript:", "alert("]
                    ):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Potential XSS via content type reflection",
                                "content_type": content_type,
                                "response_preview": response_text[:200],
                            }
                        )

                # Check for information disclosure in responses
                for pattern in self.sensitive_response_patterns:
                    if re.search(pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Sensitive information in response",
                                "content_type": content_type,
                                "pattern": pattern,
                                "response_preview": response_text[:200],
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing content type {content_type}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Validate and restrict acceptable content types",
                    "Always set appropriate Content-Type headers in responses",
                    "Sanitize content to prevent XSS via content type manipulation",
                ]
            )

        return results

    def test_response_sanitization(self) -> Dict[str, Any]:
        """Test response content sanitization."""
        results = {
            "test_name": "response_sanitization",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various injection attempts
        injection_payloads = [
            '<script>alert("xss")</script>',
            '"><script>alert("xss")</script>',
            "'; DROP TABLE users; --",
            "${7*7}",  # Template injection
            "#{7*7}",  # Template injection
            "{{7*7}}",  # Template injection
            "%3Cscript%3Ealert%28%22xss%22%29%3C%2Fscript%3E",  # URL encoded
            'javascript:alert("xss")',
            'data:text/html,<script>alert("xss")</script>',
            'vbscript:msgbox("xss")',
            '<img src=x onerror=alert("xss")>',
            '<svg onload=alert("xss")>',
            "<iframe src=\"javascript:alert('xss')\">",
            "../../etc/passwd",
            "file:///etc/passwd",
            "\\\\server\\share\\file",
        ]

        test_endpoints = [
            "/api/v1/agents",
            "/api/v1/system/search",
            "/api/v1/auth/login",
        ]

        for endpoint in test_endpoints:
            for payload in injection_payloads:
                try:
                    # Test in JSON body
                    response = self.client.post(
                        endpoint,
                        json={
                            "name": payload,
                            "description": payload,
                            "search": payload,
                            "username": payload,
                            "input": payload,
                        },
                    )

                    response_text = response.text

                    # Check if payload appears unsanitized in response
                    if payload in response_text:
                        # Some payloads might be acceptable in certain contexts
                        # Flag for review but not automatic failure
                        results["findings"].append(
                            {
                                "issue": "Potential unsanitized input in response",
                                "endpoint": endpoint,
                                "payload": payload,
                                "severity": "medium",  # Review needed
                                "response_preview": response_text[:200],
                            }
                        )

                    # Check for specific dangerous patterns
                    dangerous_patterns = [
                        r"<script[^>]*>",
                        r"javascript:",
                        r"on\w+\s*=",
                        r"<iframe[^>]*>",
                        r"<object[^>]*>",
                        r"<embed[^>]*>",
                    ]

                    for pattern in dangerous_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Dangerous content in response",
                                    "endpoint": endpoint,
                                    "payload": payload,
                                    "pattern": pattern,
                                    "severity": "high",
                                    "response_preview": response_text[:200],
                                }
                            )

                    # Test in query parameters
                    if endpoint.endswith("/search") or "?" not in endpoint:
                        search_endpoint = f"{endpoint}?q={quote(payload)}"
                        search_response = self.client.get(search_endpoint)

                        search_response_text = search_response.text

                        for pattern in dangerous_patterns:
                            if re.search(pattern, search_response_text, re.IGNORECASE):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Dangerous content in query parameter response",
                                        "endpoint": search_endpoint,
                                        "payload": payload,
                                        "pattern": pattern,
                                        "severity": "high",
                                    }
                                )

                except Exception as e:
                    results["findings"].append(
                        {
                            "issue": f"Error testing sanitization for {endpoint} with payload {payload[:50]}",
                            "error": str(e),
                        }
                    )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement comprehensive input sanitization",
                    "Escape output properly based on context (HTML, JSON, etc.)",
                    "Use parameterized queries to prevent injection attacks",
                    "Validate and sanitize all user inputs before processing",
                ]
            )

        return results

    def test_error_response_consistency(self) -> Dict[str, Any]:
        """Test that error responses are consistent and secure."""
        results = {
            "test_name": "error_response_consistency",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various error scenarios
        error_scenarios = [
            ("404_not_found", "GET", "/api/v1/nonexistent", None),
            ("405_method_not_allowed", "PATCH", "/api/v1/agents", None),
            ("400_bad_request", "POST", "/api/v1/agents", '{"invalid": json}'),
            ("422_validation_error", "POST", "/api/v1/agents", {}),
            ("401_unauthorized", "GET", "/api/v1/agents/protected", None),
            ("403_forbidden", "DELETE", "/api/v1/system/admin", None),
            ("413_payload_too_large", "POST", "/api/v1/agents", "A" * 100000),
        ]

        for scenario_name, method, endpoint, data in error_scenarios:
            try:
                if method == "GET":
                    response = self.client.get(endpoint)
                elif method == "POST":
                    if data and data.startswith("{"):
                        response = self.client.post(
                            endpoint,
                            data=data,
                            headers={"Content-Type": "application/json"},
                        )
                    else:
                        response = self.client.post(endpoint, json={"data": data} if data else {})
                elif method == "DELETE":
                    response = self.client.delete(endpoint)
                elif method == "PATCH":
                    response = self.client.patch(endpoint, json={})
                else:
                    continue

                # Check response structure
                try:
                    response_data = response.json()

                    # Error responses should have consistent structure
                    required_fields = ["detail"]  # FastAPI standard

                    for field in required_fields:
                        if field not in response_data:
                            results["findings"].append(
                                {
                                    "issue": f"Missing required error field: {field}",
                                    "scenario": scenario_name,
                                    "endpoint": endpoint,
                                    "status_code": response.status_code,
                                }
                            )

                    # Check for information disclosure in error message
                    response_text = json.dumps(response_data)
                    for pattern in self.sensitive_response_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Sensitive information in error response",
                                    "scenario": scenario_name,
                                    "pattern": pattern,
                                    "response_data": response_data,
                                }
                            )

                except json.JSONDecodeError:
                    # Non-JSON error responses should be investigated
                    results["findings"].append(
                        {
                            "issue": "Non-JSON error response",
                            "scenario": scenario_name,
                            "endpoint": endpoint,
                            "status_code": response.status_code,
                            "content_type": response.headers.get("Content-Type", "unknown"),
                        }
                    )

                # Check security headers on error responses
                for header_name in self.required_security_headers:
                    if header_name not in response.headers:
                        results["findings"].append(
                            {
                                "issue": f"Missing security header in error response: {header_name}",
                                "scenario": scenario_name,
                                "endpoint": endpoint,
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing scenario {scenario_name}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Ensure consistent error response structure across all endpoints",
                    "Apply security headers to all responses including errors",
                    "Sanitize error messages to prevent information disclosure",
                ]
            )

        return results

    def test_response_timing_consistency(self) -> Dict[str, Any]:
        """Test response timing consistency to prevent timing attacks."""
        results = {
            "test_name": "response_timing_consistency",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test timing consistency for various scenarios
        timing_scenarios = [
            ("valid_request", "/api/v1/system/status", {}),
            ("invalid_request", "/api/v1/system/status", {"invalid": "data"}),
            ("not_found", "/api/v1/nonexistent", {}),
            (
                "auth_failure",
                "/api/v1/auth/login",
                {"username": "invalid", "password": "invalid"},
            ),
            ("validation_error", "/api/v1/agents", {}),
        ]

        timing_results = {}

        for scenario_name, endpoint, data in timing_scenarios:
            times = []

            # Run multiple requests to get average timing
            for _ in range(5):
                start_time = time.time()

                if data:
                    self.client.post(endpoint, json=data)
                else:
                    self.client.get(endpoint)

                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            timing_results[scenario_name] = {
                "avg_time": avg_time,
                "std_dev": std_dev,
                "times": times,
            }

        # Check for significant timing differences
        all_avg_times = [data["avg_time"] for data in timing_results.values()]
        overall_avg = sum(all_avg_times) / len(all_avg_times)

        for scenario_name, data in timing_results.items():
            deviation = abs(data["avg_time"] - overall_avg)

            # Flag significant deviations (> 100ms)
            if deviation > 0.1:
                results["findings"].append(
                    {
                        "issue": "Significant timing difference detected",
                        "scenario": scenario_name,
                        "avg_time": data["avg_time"],
                        "overall_avg": overall_avg,
                        "deviation": deviation,
                        "note": "May indicate timing side-channel vulnerability",
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement consistent response timing across all endpoints",
                    "Consider adding artificial delays to normalize timing",
                    "Review code paths that might cause timing differences",
                ]
            )

        return results

    def run_all_api_security_tests(self) -> Dict[str, Any]:
        """Run all API security response tests."""
        print("Running comprehensive API security response tests...")

        test_methods = [
            self.test_security_headers_presence,
            self.test_rate_limiting_responses,
            self.test_cors_policy_validation,
            self.test_content_type_security,
            self.test_response_sanitization,
            self.test_error_response_consistency,
            self.test_response_timing_consistency,
        ]

        all_results = []

        for test_method in test_methods:
            try:
                result = test_method()
                all_results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  {result['test_name']}: {status}")

                if not result["passed"]:
                    for finding in result["findings"]:
                        print(f"    - {finding['issue']}")

            except Exception as e:
                print(f"  {test_method.__name__}: ERROR - {str(e)}")
                all_results.append(
                    {
                        "test_name": test_method.__name__,
                        "passed": False,
                        "findings": [{"issue": f"Test execution error: {str(e)}"}],
                        "recommendations": ["Fix test execution error"],
                    }
                )

        # Compile overall results
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r["passed"])
        failed_tests = total_tests - passed_tests

        # Categorize findings by severity
        critical_findings = []
        high_findings = []
        medium_findings = []

        for result in all_results:
            for finding in result.get("findings", []):
                severity = finding.get("severity", "medium")
                if severity == "critical":
                    critical_findings.append(finding)
                elif severity == "high":
                    high_findings.append(finding)
                else:
                    medium_findings.append(finding)

        # Collect all recommendations
        all_recommendations = []
        for result in all_results:
            all_recommendations.extend(result.get("recommendations", []))

        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "critical_findings": len(critical_findings),
                "high_findings": len(high_findings),
                "medium_findings": len(medium_findings),
            },
            "test_results": all_results,
            "recommendations": unique_recommendations,
            "overall_status": "PASS" if failed_tests == 0 else "FAIL",
        }

        return summary


class TestAPISecurityResponses:
    """pytest test class for API security response validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def api_tester(self, client):
        """Create API security response tester."""
        return APISecurityResponseTester(client)

    def test_security_headers_present(self, api_tester):
        """Test that security headers are present."""
        result = api_tester.test_security_headers_presence()

        if not result["passed"]:
            failure_msg = "Security headers validation failed:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_rate_limiting_secure(self, api_tester):
        """Test that rate limiting is implemented securely."""
        result = api_tester.test_rate_limiting_responses()

        if not result["passed"]:
            failure_msg = "Rate limiting security issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_cors_policy_secure(self, api_tester):
        """Test that CORS policy is properly configured."""
        result = api_tester.test_cors_policy_validation()

        if not result["passed"]:
            failure_msg = "CORS policy security issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_response_sanitization_effective(self, api_tester):
        """Test that response sanitization is effective."""
        result = api_tester.test_response_sanitization()

        if not result["passed"]:
            failure_msg = "Response sanitization failures:\n"
            for finding in result["findings"]:
                if finding.get("severity") == "high":
                    failure_msg += f"  - HIGH: {finding['issue']}\n"

            if "HIGH:" in failure_msg:
                pytest.fail(failure_msg)

    def test_error_responses_secure(self, api_tester):
        """Test that error responses are secure and consistent."""
        result = api_tester.test_error_response_consistency()

        if not result["passed"]:
            failure_msg = "Error response security issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_comprehensive_api_security(self, api_tester):
        """Run comprehensive API security tests."""
        summary = api_tester.run_all_api_security_tests()

        if summary["overall_status"] == "FAIL":
            failure_msg = f"API security test failures: {summary['summary']['failed_tests']} out of {summary['summary']['total_tests']} tests failed\n"

            # Check for critical/high severity issues
            if summary["summary"]["critical_findings"] > 0:
                failure_msg += f"\nCRITICAL ISSUES: {summary['summary']['critical_findings']}\n"

            if summary["summary"]["high_findings"] > 0:
                failure_msg += f"HIGH SEVERITY ISSUES: {summary['summary']['high_findings']}\n"

            if summary["recommendations"]:
                failure_msg += "\nRecommendations:\n"
                for rec in summary["recommendations"][:10]:  # Limit to first 10
                    failure_msg += f"  - {rec}\n"

            pytest.fail(failure_msg)


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    client = TestClient(app)
    tester = APISecurityResponseTester(client)

    print("Running API security response validation tests...")
    summary = tester.run_all_api_security_tests()

    print(f"\n{'=' * 60}")
    print("API SECURITY RESPONSE VALIDATION REPORT")
    print(f"{'=' * 60}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Pass Rate: {summary['summary']['pass_rate']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")

    print("\nFindings by Severity:")
    print(f"  Critical: {summary['summary']['critical_findings']}")
    print(f"  High: {summary['summary']['high_findings']}")
    print(f"  Medium: {summary['summary']['medium_findings']}")

    if summary["recommendations"]:
        print(f"\n{'=' * 40}")
        print("RECOMMENDATIONS")
        print(f"{'=' * 40}")
        for rec in summary["recommendations"]:
            print(f"• {rec}")

    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = (
        f"/home/green/FreeAgentics/tests/security/api_security_response_report_{timestamp}.json"
    )

    try:
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code
    exit(0 if summary["overall_status"] == "PASS" else 1)
