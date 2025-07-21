"""
API Security Testing Module

This module implements comprehensive API security testing including:
- Parameter pollution attacks
- HTTP method tampering
- API versioning attacks
- Rate limiting bypass
- Content-type confusion
- Input validation bypass
- API endpoint enumeration
- Mass assignment vulnerabilities
"""

import json
import logging
import time
import uuid
from typing import Dict, List

from .penetration_testing_framework import (
    BasePenetrationTest,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
    generate_parameter_pollution_payloads,
)

logger = logging.getLogger(__name__)


class APISecurityTests(BasePenetrationTest):
    """Comprehensive API security vulnerability testing."""

    async def execute(self) -> TestResult:
        """Execute all API security tests."""
        start_time = time.time()

        try:
            # Test parameter pollution attacks
            await self._test_parameter_pollution()

            # Test HTTP method tampering
            await self._test_http_method_tampering()

            # Test API versioning attacks
            await self._test_api_versioning()

            # Test rate limiting bypass
            await self._test_rate_limiting_bypass()

            # Test content-type confusion
            await self._test_content_type_confusion()

            # Test input validation bypass
            await self._test_input_validation_bypass()

            # Test API endpoint enumeration
            await self._test_endpoint_enumeration()

            # Test mass assignment vulnerabilities
            await self._test_mass_assignment()

            # Test API response manipulation
            await self._test_response_manipulation()

            # Test API error handling
            await self._test_error_handling()

            execution_time = time.time() - start_time

            return TestResult(
                test_name="APISecurityTests",
                success=True,
                vulnerabilities=self.vulnerabilities,
                execution_time=execution_time,
                metadata={
                    "api_endpoints_tested": len(self._get_api_endpoints()),
                    "attack_vectors_tested": 10,
                    "payloads_used": self._count_payloads(),
                },
            )

        except Exception as e:
            logger.error(f"API security test failed: {e}")
            return TestResult(
                test_name="APISecurityTests",
                success=False,
                vulnerabilities=self.vulnerabilities,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _test_parameter_pollution(self):
        """Test parameter pollution attacks."""
        logger.info("Testing parameter pollution vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test different parameter pollution techniques
        pollution_payloads = generate_parameter_pollution_payloads()

        # Test endpoints susceptible to parameter pollution
        test_endpoints = [
            ("/api/v1/agents", "GET"),
            ("/api/v1/agents", "POST"),
            ("/api/v1/auth/login", "POST"),
            ("/api/v1/users", "GET"),
        ]

        for endpoint, method in test_endpoints:
            for payload in pollution_payloads:
                # Test query parameter pollution
                if method == "GET":
                    response = self._test_query_pollution(endpoint, payload, token)
                else:
                    response = self._test_body_pollution(endpoint, method, payload, token)

                if self._detect_pollution_success(response, payload):
                    self.add_vulnerability(
                        VulnerabilityFinding(
                            vulnerability_type=VulnerabilityType.PARAMETER_POLLUTION,
                            severity=SeverityLevel.MEDIUM,
                            title=f"Parameter Pollution in {endpoint}",
                            description=f"Parameter pollution attack successful on {endpoint}",
                            affected_endpoint=endpoint,
                            proof_of_concept=f"{method} {endpoint}\nPayload: {json.dumps(payload)}",
                            exploitation_steps=[
                                "1. Identify parameters that accept arrays/multiple values",
                                "2. Send conflicting parameter values",
                                "3. Exploit server-side parameter parsing inconsistencies",
                                "4. Bypass input validation or business logic",
                            ],
                            remediation_steps=[
                                "Implement consistent parameter parsing",
                                "Validate parameter arrays properly",
                                "Use strict input validation",
                                "Reject ambiguous parameter formats",
                                "Implement parameter deduplication",
                            ],
                            cwe_id="CWE-235",
                            cvss_score=5.3,
                            test_method="parameter_pollution",
                        )
                    )

        # Test HTTP header pollution
        await self._test_header_pollution(token)

    async def _test_http_method_tampering(self):
        """Test HTTP method tampering attacks."""
        logger.info("Testing HTTP method tampering")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test method override techniques
        override_headers = [
            "X-HTTP-Method-Override",
            "X-HTTP-Method",
            "X-Method-Override",
            "_method",
        ]

        dangerous_methods = ["DELETE", "PUT", "PATCH", "POST"]
        test_endpoints = [
            "/api/v1/agents/123",
            "/api/v1/users/123",
            "/api/v1/admin/users",
        ]

        for endpoint in test_endpoints:
            for override_header in override_headers:
                for method in dangerous_methods:
                    # Send GET request with method override header
                    headers = self.get_auth_headers(token)
                    headers[override_header] = method

                    response = self.client.get(endpoint, headers=headers)

                    # Check if method override was accepted
                    if self._detect_method_override_success(response, method):
                        severity = (
                            SeverityLevel.HIGH if method == "DELETE" else SeverityLevel.MEDIUM
                        )

                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.HTTP_METHOD_TAMPERING,
                                severity=severity,
                                title=f"HTTP Method Override in {endpoint}",
                                description=f"Server accepts {override_header} header to override HTTP method",
                                affected_endpoint=endpoint,
                                proof_of_concept=f"GET {endpoint}\n{override_header}: {method}",
                                exploitation_steps=[
                                    "1. Send GET request to target endpoint",
                                    f"2. Add header: {override_header}: {method}",
                                    "3. Server processes request as different HTTP method",
                                    "4. Bypass method-based access controls",
                                ],
                                remediation_steps=[
                                    "Disable HTTP method override functionality",
                                    "Use proper HTTP methods for operations",
                                    "Implement method-based authorization consistently",
                                    "Validate HTTP methods server-side",
                                ],
                                cwe_id="CWE-436",
                                cvss_score=7.5 if severity == SeverityLevel.HIGH else 5.4,
                                test_method="http_method_tampering",
                            )
                        )

        # Test verb tunneling through POST
        await self._test_verb_tunneling(token)

    async def _test_api_versioning(self):
        """Test API versioning attacks."""
        logger.info("Testing API versioning vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_headers(username, password)

        # Test different API version access methods
        version_techniques = [
            # URL path versioning
            ("/api/v2/admin/users", "v2 path access"),
            ("/api/v0/admin/users", "v0 path access"),
            ("/api/beta/admin/users", "beta path access"),
            ("/api/internal/admin/users", "internal path access"),
            # Header versioning
            (
                "/api/v1/admin/users",
                "Accept: application/vnd.api+json;version=2",
            ),
            ("/api/v1/admin/users", "API-Version: 2"),
            ("/api/v1/admin/users", "Version: admin"),
        ]

        for endpoint, technique in version_techniques:
            if ":" in technique:  # Header technique
                header_name, header_value = technique.split(": ")
                headers = token.copy()
                headers[header_name] = header_value
                response = self.client.get(endpoint, headers=headers)
            else:  # Path technique
                response = self.client.get(endpoint, headers=token)

            if response.status_code in [200, 201, 202]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.API_VERSIONING_ATTACKS,
                        severity=SeverityLevel.MEDIUM,
                        title="API Version Access Control Bypass",
                        description=f"Access to different API version bypasses access controls: {technique}",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"GET {endpoint}\nTechnique: {technique}",
                        exploitation_steps=[
                            "1. Identify different API versions",
                            "2. Test access to restricted endpoints in different versions",
                            "3. Exploit version-specific access control differences",
                            "4. Access unauthorized functionality",
                        ],
                        remediation_steps=[
                            "Implement consistent access controls across all API versions",
                            "Properly deprecate and secure old API versions",
                            "Use version-aware authorization",
                            "Regular security audits of all API versions",
                        ],
                        cwe_id="CWE-863",
                        cvss_score=6.1,
                        test_method="api_versioning",
                    )
                )

        # Test legacy endpoint access
        await self._test_legacy_endpoints(token)

    async def _test_rate_limiting_bypass(self):
        """Test rate limiting bypass techniques."""
        logger.info("Testing rate limiting bypass")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test rate limiting on login endpoint (should have strict limits)
        baseline_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "nonexistent", "password": "wrong"},
        )

        if baseline_response.status_code != 429:
            # Rate limiting might not be active, test bypass techniques
            bypass_techniques = [
                # IP spoofing
                {"X-Forwarded-For": "1.1.1.1"},
                {"X-Real-IP": "2.2.2.2"},
                {"X-Originating-IP": "3.3.3.3"},
                {"X-Remote-IP": "4.4.4.4"},
                {"X-Client-IP": "5.5.5.5"},
                # User agent rotation
                {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                # Session/token manipulation
                {"Authorization": f"Bearer {self._generate_fake_token()}"},
                {"X-Session-ID": str(uuid.uuid4())},
            ]

            successful_bypasses = 0
            total_attempts = 0

            for bypass_headers in bypass_techniques:
                for attempt in range(20):  # 20 rapid attempts per technique
                    response = self.client.post(
                        "/api/v1/auth/login",
                        json={"username": "test", "password": "wrong"},
                        headers=bypass_headers,
                    )
                    total_attempts += 1

                    if response.status_code != 429:
                        successful_bypasses += 1

                    time.sleep(0.05)  # Small delay between requests

            bypass_ratio = successful_bypasses / total_attempts if total_attempts > 0 else 0

            if bypass_ratio > 0.7:  # More than 70% bypass success
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.RATE_LIMITING_BYPASS,
                        severity=SeverityLevel.MEDIUM,
                        title="Rate Limiting Bypass",
                        description=f"Rate limiting bypassed in {bypass_ratio:.1%} of attempts",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"Header manipulation allows {successful_bypasses}/{total_attempts} requests",
                        exploitation_steps=[
                            "1. Identify rate-limited endpoints",
                            "2. Use header manipulation to spoof client identity",
                            "3. Rotate IP addresses and user agents",
                            "4. Perform attacks beyond rate limits",
                        ],
                        remediation_steps=[
                            "Implement multiple rate limiting factors",
                            "Use account-based rate limiting",
                            "Validate and sanitize proxy headers",
                            "Implement progressive delays and CAPTCHA",
                            "Monitor for distributed attacks",
                        ],
                        cwe_id="CWE-770",
                        cvss_score=5.3,
                        test_method="rate_limiting_bypass",
                    )
                )

        # Test application-level rate limiting bypass
        await self._test_application_rate_limits(token)

    async def _test_content_type_confusion(self):
        """Test content-type confusion attacks."""
        logger.info("Testing content-type confusion")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test different content-type manipulations
        content_type_tests = [
            # JSON to form data confusion
            {
                "content_type": "application/x-www-form-urlencoded",
                "data": "username=admin&password=admin&role=admin",
                "description": "Form data with JSON endpoint",
            },
            # XML injection
            {
                "content_type": "application/xml",
                "data": "<?xml version='1.0'?><user><role>admin</role></user>",
                "description": "XML data injection",
            },
            # Text injection
            {
                "content_type": "text/plain",
                "data": '{"role": "admin", "permissions": ["admin_system"]}',
                "description": "Text content type with JSON data",
            },
            # Charset manipulation
            {
                "content_type": "application/json; charset=utf-7",
                "data": '{"username": "admin"}',
                "description": "Charset confusion attack",
            },
        ]

        test_endpoints = [
            "/api/v1/auth/login",
            "/api/v1/agents",
            "/api/v1/users",
        ]

        for endpoint in test_endpoints:
            for test_case in content_type_tests:
                headers = self.get_auth_headers(token)
                headers["Content-Type"] = test_case["content_type"]

                response = self.client.post(endpoint, data=test_case["data"], headers=headers)

                if self._detect_content_type_success(response):
                    self.add_vulnerability(
                        VulnerabilityFinding(
                            vulnerability_type=VulnerabilityType.CONTENT_TYPE_CONFUSION,
                            severity=SeverityLevel.MEDIUM,
                            title=f"Content-Type Confusion in {endpoint}",
                            description=f"{test_case['description']} accepted by server",
                            affected_endpoint=endpoint,
                            proof_of_concept=f"POST {endpoint}\nContent-Type: {test_case['content_type']}\n"
                            f"Data: {test_case['data'][:100]}...",
                            exploitation_steps=[
                                "1. Identify API endpoints expecting specific content types",
                                "2. Send data with different content-type headers",
                                "3. Exploit parser confusion vulnerabilities",
                                "4. Bypass input validation or authentication",
                            ],
                            remediation_steps=[
                                "Implement strict content-type validation",
                                "Reject unexpected content types",
                                "Use consistent parsers for each content type",
                                "Implement input validation regardless of content type",
                            ],
                            cwe_id="CWE-436",
                            cvss_score=5.4,
                            test_method="content_type_confusion",
                        )
                    )

    async def _test_input_validation_bypass(self):
        """Test input validation bypass techniques."""
        logger.info("Testing input validation bypass")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test various input validation bypass payloads
        bypass_payloads = [
            # Encoding bypasses
            {
                "username": "%61%64%6d%69%6e",
                "description": "URL encoding bypass",
            },
            {
                "username": "\\u0061\\u0064\\u006d\\u0069\\u006e",
                "description": "Unicode bypass",
            },
            {"username": "admin\x00", "description": "Null byte bypass"},
            {"username": "admin\r\n", "description": "CRLF injection"},
            # Case manipulation
            {"username": "ADMIN", "description": "Case bypass"},
            {"username": "Admin", "description": "Mixed case bypass"},
            # Special characters
            {"username": "admin'", "description": "Quote injection"},
            {"username": 'admin"', "description": "Double quote injection"},
            {"username": "admin;", "description": "Semicolon injection"},
            # Length bypasses
            {"username": "a" * 1000, "description": "Length overflow"},
            {"username": "", "description": "Empty string bypass"},
        ]

        for payload in bypass_payloads:
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": payload["username"], "password": "test"},
                headers=self.get_auth_headers(token),
            )

            if self._detect_validation_bypass(response, payload):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.INPUT_VALIDATION_BYPASS,
                        severity=SeverityLevel.MEDIUM,
                        title=f"Input Validation Bypass - {payload['description']}",
                        description=f"Input validation bypassed using {payload['description']}",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"Username: {repr(payload['username'])}",
                        exploitation_steps=[
                            "1. Identify input validation mechanisms",
                            "2. Test encoding and character manipulation",
                            "3. Bypass validation using special characters",
                            "4. Inject malicious payloads",
                        ],
                        remediation_steps=[
                            "Implement comprehensive input validation",
                            "Use whitelist-based validation",
                            "Normalize input before validation",
                            "Validate after decoding/parsing",
                            "Use parameterized queries and prepared statements",
                        ],
                        cwe_id="CWE-20",
                        cvss_score=5.3,
                        test_method="input_validation_bypass",
                    )
                )

        # Test JSON injection
        await self._test_json_injection(token)

    async def _test_endpoint_enumeration(self):
        """Test API endpoint enumeration."""
        logger.info("Testing API endpoint enumeration")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test common API endpoints
        common_endpoints = [
            # Admin endpoints
            "/api/v1/admin",
            "/api/v1/admin/users",
            "/api/v1/admin/config",
            "/api/v1/admin/logs",
            "/api/v1/admin/debug",
            "/api/v1/admin/stats",
            # Debug/test endpoints
            "/api/v1/debug",
            "/api/v1/test",
            "/api/v1/dev",
            "/api/v1/internal",
            # Documentation endpoints
            "/api/v1/docs",
            "/api/v1/swagger",
            "/api/v1/openapi.json",
            "/api/v1/redoc",
            # Health/monitoring
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/metrics",
            "/api/v1/ping",
            # File operations
            "/api/v1/files",
            "/api/v1/upload",
            "/api/v1/download",
            "/api/v1/backup",
        ]

        accessible_endpoints = []

        for endpoint in common_endpoints:
            # Test without authentication
            response = self.client.get(endpoint)
            if response.status_code in [200, 201, 202, 301, 302]:
                accessible_endpoints.append((endpoint, "unauthenticated"))

            # Test with authentication
            response = self.client.get(endpoint, headers=self.get_auth_headers(token))
            if response.status_code in [200, 201, 202]:
                accessible_endpoints.append((endpoint, "authenticated"))

        if accessible_endpoints:
            endpoints_list = "\n".join([f"{ep} ({auth})" for ep, auth in accessible_endpoints])

            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.API_ENDPOINT_ENUMERATION,
                    severity=SeverityLevel.LOW,
                    title="API Endpoint Enumeration",
                    description=f"Discovered {len(accessible_endpoints)} accessible endpoints",
                    affected_endpoint="/api/v1/*",
                    proof_of_concept=f"Accessible endpoints:\n{endpoints_list}",
                    exploitation_steps=[
                        "1. Enumerate API endpoints using common patterns",
                        "2. Identify debug, admin, or internal endpoints",
                        "3. Test endpoint accessibility",
                        "4. Exploit discovered endpoints for information disclosure",
                    ],
                    remediation_steps=[
                        "Remove debug/test endpoints from production",
                        "Implement proper access controls on all endpoints",
                        "Use API gateway for endpoint management",
                        "Regular endpoint security audits",
                    ],
                    cwe_id="CWE-200",
                    cvss_score=3.7,
                    test_method="endpoint_enumeration",
                )
            )

    async def _test_mass_assignment(self):
        """Test mass assignment vulnerabilities."""
        logger.info("Testing mass assignment vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test mass assignment on user creation/update
        mass_assignment_payloads = [
            # Try to set admin role
            {
                "username": "test_mass_1",
                "email": "test@example.com",
                "password": "password",
                "role": "admin",
                "is_admin": True,
                "permissions": ["admin_system", "delete_agent"],
            },
            # Try to set system fields
            {
                "username": "test_mass_2",
                "email": "test2@example.com",
                "password": "password",
                "id": "custom_id",
                "user_id": "admin",
                "created_by": "system",
                "is_active": True,
            },
            # Try to inject additional fields
            {
                "username": "test_mass_3",
                "email": "test3@example.com",
                "password": "password",
                "credits": 999999,
                "premium": True,
                "access_level": "unlimited",
            },
        ]

        for payload in mass_assignment_payloads:
            # Test on registration endpoint
            response = self.client.post("/api/v1/auth/register", json=payload)

            if response.status_code in [200, 201]:
                try:
                    data = response.json()

                    # Check if unauthorized fields were set
                    unauthorized_fields = []
                    for field in [
                        "role",
                        "is_admin",
                        "permissions",
                        "id",
                        "user_id",
                        "created_by",
                        "credits",
                        "premium",
                        "access_level",
                    ]:
                        if field in data.get("user", {}) and payload.get(field):
                            unauthorized_fields.append(field)

                    if unauthorized_fields:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.MASS_ASSIGNMENT,
                                severity=SeverityLevel.HIGH,
                                title="Mass Assignment Vulnerability in User Registration",
                                description=f"Unauthorized fields set via mass assignment: {unauthorized_fields}",
                                affected_endpoint="/api/v1/auth/register",
                                proof_of_concept=f"Payload: {json.dumps(payload, indent=2)}\n"
                                f"Set fields: {unauthorized_fields}",
                                exploitation_steps=[
                                    "1. Identify API endpoints accepting object data",
                                    "2. Include additional fields in request payload",
                                    "3. Exploit automatic object mapping/assignment",
                                    "4. Gain unauthorized privileges or access",
                                ],
                                remediation_steps=[
                                    "Use explicit field whitelisting",
                                    "Implement data transfer objects (DTOs)",
                                    "Validate and filter input fields",
                                    "Use ORM features to prevent mass assignment",
                                    "Implement field-level access controls",
                                ],
                                cwe_id="CWE-915",
                                cvss_score=8.1,
                                test_method="mass_assignment",
                            )
                        )

                except Exception as e:
                    logger.debug(f"Mass assignment test error: {e}")

        # Test mass assignment on existing resources
        await self._test_resource_mass_assignment(token)

    async def _test_response_manipulation(self):
        """Test API response manipulation attacks."""
        logger.info("Testing API response manipulation")

        # Create test user
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test response manipulation techniques
        manipulation_headers = [
            # Content-Type override
            {"Accept": "application/xml"},
            {"Accept": "text/html"},
            {"Accept": "application/javascript"},
            # Response format manipulation
            {"X-Requested-With": "XMLHttpRequest"},
            {"X-Response-Format": "xml"},
            {"Format": "json"},
            # Callback injection (JSONP)
            {"callback": "alert(1)"},
            {"jsonp": "malicious_function"},
        ]

        for headers in manipulation_headers:
            response = self.client.get(
                "/api/v1/auth/me",
                headers={**self.get_auth_headers(token), **headers},
            )

            if self._detect_response_manipulation(response, headers):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.RESPONSE_MANIPULATION,
                        severity=SeverityLevel.MEDIUM,
                        title="API Response Manipulation",
                        description=f"Response format manipulated using headers: {headers}",
                        affected_endpoint="/api/v1/auth/me",
                        proof_of_concept=f"Headers: {json.dumps(headers)}",
                        exploitation_steps=[
                            "1. Identify API endpoints with format flexibility",
                            "2. Manipulate Accept or format headers",
                            "3. Exploit response format differences",
                            "4. Extract additional information or execute code",
                        ],
                        remediation_steps=[
                            "Implement strict response format controls",
                            "Validate Accept headers",
                            "Disable JSONP if not needed",
                            "Use consistent response formats",
                        ],
                        cwe_id="CWE-436",
                        cvss_score=4.3,
                        test_method="response_manipulation",
                    )
                )

    async def _test_error_handling(self):
        """Test API error handling for information disclosure."""
        logger.info("Testing API error handling")

        # Test various error conditions
        error_test_cases = [
            # Malformed JSON
            {
                "endpoint": "/api/v1/auth/login",
                "data": '{"username": "test", "password":}',
                "content_type": "application/json",
                "description": "Malformed JSON",
            },
            # SQL injection attempts
            {
                "endpoint": "/api/v1/auth/login",
                "data": '{"username": "admin\'; DROP TABLE users; --", "password": "test"}',
                "content_type": "application/json",
                "description": "SQL injection",
            },
            # Large payload
            {
                "endpoint": "/api/v1/agents",
                "data": '{"name": "' + "A" * 10000 + '"}',
                "content_type": "application/json",
                "description": "Large payload",
            },
        ]

        for test_case in error_test_cases:
            response = self.client.post(
                test_case["endpoint"],
                data=test_case["data"],
                headers={"Content-Type": test_case["content_type"]},
            )

            if self._detect_information_disclosure(response):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                        severity=SeverityLevel.LOW,
                        title="Information Disclosure in Error Handling",
                        description=f"Error response reveals sensitive information: {test_case['description']}",
                        affected_endpoint=test_case["endpoint"],
                        proof_of_concept=f"Error response: {response.text[:200]}...",
                        exploitation_steps=[
                            "1. Send malformed or malicious requests",
                            "2. Analyze error responses for sensitive information",
                            "3. Extract system details, stack traces, or database info",
                            "4. Use information for further attacks",
                        ],
                        remediation_steps=[
                            "Implement generic error messages",
                            "Log detailed errors server-side only",
                            "Remove stack traces from API responses",
                            "Use custom error pages",
                            "Sanitize error responses",
                        ],
                        cwe_id="CWE-209",
                        cvss_score=3.7,
                        test_method="error_handling",
                    )
                )

    # Helper methods

    def _test_query_pollution(self, endpoint: str, payload: Dict, token: str):
        """Test query parameter pollution."""
        # Convert payload to query string with duplicates
        query_parts = []
        for key, value in payload.items():
            if isinstance(value, list):
                for v in value:
                    query_parts.append(f"{key}={v}")
            else:
                query_parts.append(f"{key}={value}")

        query_string = "&".join(query_parts)
        full_url = f"{endpoint}?{query_string}"

        return self.client.get(full_url, headers=self.get_auth_headers(token))

    def _test_body_pollution(self, endpoint: str, method: str, payload: Dict, token: str):
        """Test request body parameter pollution."""
        headers = self.get_auth_headers(token)

        if method == "POST":
            return self.client.post(endpoint, json=payload, headers=headers)
        elif method == "PUT":
            return self.client.put(endpoint, json=payload, headers=headers)
        else:
            return self.client.patch(endpoint, json=payload, headers=headers)

    async def _test_header_pollution(self, token: str):
        """Test HTTP header pollution."""
        pollution_headers = [
            {"Host": ["example.com", "evil.com"]},
            {"X-Forwarded-Host": ["legitimate.com", "attacker.com"]},
            {"Origin": ["https://trusted.com", "https://evil.com"]},
        ]

        for headers in pollution_headers:
            # Simulate header pollution (some proxies/servers handle this differently)
            response = self.client.get(
                "/api/v1/auth/me",
                headers={**self.get_auth_headers(token), **headers},
            )

            if response.status_code == 200:
                # Check if response indicates header confusion
                pass  # Would need specific detection logic

    async def _test_verb_tunneling(self, token: str):
        """Test HTTP verb tunneling through POST."""
        # Test hiding dangerous verbs in POST body
        tunneling_attempts = [
            {"_method": "DELETE"},
            {"method": "DELETE"},
            {"http_method": "DELETE"},
            {"verb": "DELETE"},
        ]

        for params in tunneling_attempts:
            response = self.client.post(
                "/api/v1/agents/123",
                json=params,
                headers=self.get_auth_headers(token),
            )

            if response.status_code in [200, 202, 204]:
                # Successful verb tunneling detected
                pass

    async def _test_legacy_endpoints(self, headers: Dict[str, str]):
        """Test access to legacy API endpoints."""
        legacy_patterns = [
            "/api/legacy/",
            "/api/old/",
            "/api/deprecated/",
            "/v0/",
            "/beta/",
            "/internal/",
        ]

        for pattern in legacy_patterns:
            test_endpoint = f"{pattern}users"
            response = self.client.get(test_endpoint, headers=headers)

            if response.status_code in [200, 201, 202]:
                # Legacy endpoint accessible
                pass

    async def _test_application_rate_limits(self, token: str):
        """Test application-level rate limiting."""
        # Test API operations that should have rate limits
        rate_limited_operations = [
            ("/api/v1/agents", "POST"),  # Resource creation
            ("/api/v1/auth/refresh", "POST"),  # Token refresh
        ]

        for endpoint, method in rate_limited_operations:
            requests_sent = 0
            rate_limited = False

            for i in range(50):  # Try 50 rapid requests
                if method == "POST":
                    response = self.client.post(
                        endpoint,
                        json={"name": f"test_{i}"},
                        headers=self.get_auth_headers(token),
                    )
                else:
                    response = self.client.get(endpoint, headers=self.get_auth_headers(token))

                requests_sent += 1

                if response.status_code == 429:
                    rate_limited = True
                    break

                time.sleep(0.01)  # Very small delay

            if not rate_limited and requests_sent > 20:
                # No rate limiting detected after many requests
                pass

    async def _test_json_injection(self, token: str):
        """Test JSON injection attacks."""
        json_payloads = [
            '{"username": "admin", "role": "admin"}',
            '{"username": "test"} {"role": "admin"}',  # JSON splitting
            '{"username": "test", "password": "test", "admin": true}',
        ]

        for payload in json_payloads:
            response = self.client.post(
                "/api/v1/auth/login",
                data=payload,
                headers={
                    **self.get_auth_headers(token),
                    "Content-Type": "application/json",
                },
            )

            if response.status_code == 200:
                # Successful JSON injection
                pass

    async def _test_resource_mass_assignment(self, token: str):
        """Test mass assignment on resource updates."""
        # Create a resource first
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "test_agent", "type": "basic"},
            headers=self.get_auth_headers(token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try mass assignment on update
                    mass_update = {
                        "name": "updated_agent",
                        "owner": "admin",
                        "permissions": ["admin"],
                        "is_public": True,
                        "system_agent": True,
                    }

                    update_response = self.client.put(
                        f"/api/v1/agents/{agent_id}",
                        json=mass_update,
                        headers=self.get_auth_headers(token),
                    )

                    if update_response.status_code in [200, 202]:
                        # Check if unauthorized fields were updated
                        pass

            except Exception as e:
                logger.debug(f"Resource mass assignment test error: {e}")

    def _detect_pollution_success(self, response, payload: Dict) -> bool:
        """Detect if parameter pollution was successful."""
        # Look for signs that multiple values were processed
        if response.status_code in [200, 201, 202]:
            try:
                data = response.json() if response.text else {}

                # Check if response contains multiple values from pollution
                for key, value in payload.items():
                    if isinstance(value, list) and key in str(data):
                        return True

                return "error" not in response.text.lower()
            except Exception:
                pass
        return False

    def _detect_method_override_success(self, response, method: str) -> bool:
        """Detect if HTTP method override was successful."""
        # Check response codes that indicate method was processed
        if method == "DELETE" and response.status_code in [200, 202, 204]:
            return True
        elif method in ["PUT", "PATCH"] and response.status_code in [200, 202]:
            return True
        elif method == "POST" and response.status_code in [200, 201, 202]:
            return True

        # Check response body for method indicators
        return method.lower() in response.text.lower()

    def _detect_content_type_success(self, response) -> bool:
        """Detect if content-type confusion was successful."""
        # Check if server processed unexpected content type
        return (
            response.status_code in [200, 201, 202]
            and "error" not in response.text.lower()
            and "invalid" not in response.text.lower()
        )

    def _detect_validation_bypass(self, response, payload: Dict) -> bool:
        """Detect if input validation was bypassed."""
        # Check if malicious input was processed without error
        if response.status_code in [200, 201, 202]:
            return True

        # Check for specific error messages that indicate processing
        processed_indicators = ["syntax error", "database", "query", "parse"]
        return any(indicator in response.text.lower() for indicator in processed_indicators)

    def _detect_response_manipulation(self, response, headers: Dict) -> bool:
        """Detect if response format was manipulated."""
        content_type = response.headers.get("content-type", "").lower()

        # Check if response format changed based on headers
        if "accept" in headers:
            expected_type = headers["accept"].lower()
            if expected_type in content_type and "json" not in content_type:
                return True

        # Check for JSONP callback injection
        if "callback" in headers and headers["callback"] in response.text:
            return True

        return False

    def _detect_information_disclosure(self, response) -> bool:
        """Detect information disclosure in error responses."""
        sensitive_indicators = [
            "stack trace",
            "traceback",
            "exception",
            "sql",
            "database",
            "file not found",
            "path",
            "directory",
            "server error",
            "internal error",
            "debug",
            "line number",
            "function",
            "postgresql",
            "mysql",
            "sqlite",
            "mongodb",
        ]

        response_text = response.text.lower()
        return any(indicator in response_text for indicator in sensitive_indicators)

    def _generate_fake_token(self) -> str:
        """Generate a fake token for testing."""
        return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiZmFrZSJ9.fake_signature"

    def _get_api_endpoints(self) -> List[str]:
        """Get list of API endpoints tested."""
        return [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/me",
            "/api/v1/agents",
            "/api/v1/users",
            "/api/v1/admin/*",
        ]

    def _count_payloads(self) -> int:
        """Count total payloads used in testing."""
        return (
            len(generate_parameter_pollution_payloads())
            + len(self._get_method_override_headers())
            + len(self._get_content_type_tests())
            + len(self._get_validation_bypass_payloads())
            + 50
        )  # Endpoint enumeration attempts

    def _get_method_override_headers(self) -> List[str]:
        """Get method override headers."""
        return [
            "X-HTTP-Method-Override",
            "X-HTTP-Method",
            "X-Method-Override",
            "_method",
        ]

    def _get_content_type_tests(self) -> List[Dict]:
        """Get content type test cases."""
        return [
            {"content_type": "application/x-www-form-urlencoded"},
            {"content_type": "application/xml"},
            {"content_type": "text/plain"},
            {"content_type": "application/json; charset=utf-7"},
        ]

    def _get_validation_bypass_payloads(self) -> List[Dict]:
        """Get validation bypass payloads."""
        return [
            {"username": "%61%64%6d%69%6e"},
            {"username": "\\u0061\\u0064\\u006d\\u0069\\u006e"},
            {"username": "admin\x00"},
            {"username": "ADMIN"},
            {"username": "admin'"},
            {"username": "a" * 1000},
        ]
