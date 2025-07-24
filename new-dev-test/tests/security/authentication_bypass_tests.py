"""
Authentication Bypass Testing Module

This module implements comprehensive authentication bypass testing including:
- SQL injection in authentication endpoints
- NoSQL injection for user lookups
- LDAP injection for directory services
- JWT manipulation attacks
- Session fixation attempts
- Brute force protection bypass
- Password reset vulnerabilities
"""

import json
import logging
import time

from .penetration_testing_framework import (
    BasePenetrationTest,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
    generate_jwt_manipulation_payloads,
    generate_ldap_injection_payloads,
    generate_nosql_injection_payloads,
    generate_sql_injection_payloads,
)

logger = logging.getLogger(__name__)


class AuthenticationBypassTests(BasePenetrationTest):
    """Comprehensive authentication bypass testing."""

    async def execute(self) -> TestResult:
        """Execute all authentication bypass tests."""
        start_time = time.time()

        try:
            # Test SQL injection in login endpoint
            await self._test_sql_injection_login()

            # Test NoSQL injection in authentication
            await self._test_nosql_injection_auth()

            # Test LDAP injection
            await self._test_ldap_injection()

            # Test JWT manipulation
            await self._test_jwt_manipulation()

            # Test session fixation
            await self._test_session_fixation()

            # Test brute force protection bypass
            await self._test_brute_force_bypass()

            # Test password reset vulnerabilities
            await self._test_password_reset_bypass()

            # Test authentication timing attacks
            await self._test_timing_attacks()

            # Test weak credential policies
            await self._test_weak_credentials()

            # Test account enumeration
            await self._test_account_enumeration()

            execution_time = time.time() - start_time

            return TestResult(
                test_name="AuthenticationBypassTests",
                success=True,
                vulnerabilities=self.vulnerabilities,
                execution_time=execution_time,
                metadata={
                    "total_payloads_tested": self._get_payload_count(),
                    "endpoints_tested": [
                        "/api/v1/auth/login",
                        "/api/v1/auth/register",
                        "/api/v1/auth/refresh",
                        "/api/v1/auth/reset-password",
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Authentication bypass test failed: {e}")
            return TestResult(
                test_name="AuthenticationBypassTests",
                success=False,
                vulnerabilities=self.vulnerabilities,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _test_sql_injection_login(self):
        """Test SQL injection vulnerabilities in login endpoint."""
        logger.info("Testing SQL injection in authentication endpoints")

        payloads = generate_sql_injection_payloads()

        for payload in payloads:
            # Test in username field
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": payload, "password": "any_password"},
            )

            if self._detect_sql_injection_success(response, payload):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SQL_INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        title="SQL Injection in Login Username Field",
                        description=f"The login endpoint is vulnerable to SQL injection through the username parameter. "
                        f"Payload '{payload}' resulted in successful authentication bypass.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"POST /api/v1/auth/login\n"
                        f"Content-Type: application/json\n\n"
                        f'{{"username": "{payload}", "password": "test"}}',
                        exploitation_steps=[
                            "1. Send POST request to /api/v1/auth/login",
                            f"2. Use payload in username field: {payload}",
                            "3. Use any password",
                            "4. Authentication bypass achieved",
                        ],
                        remediation_steps=[
                            "Implement parameterized queries for all database operations",
                            "Use ORM with built-in SQL injection protection",
                            "Validate and sanitize all user inputs",
                            "Implement input length limits",
                            "Use prepared statements instead of string concatenation",
                        ],
                        cwe_id="CWE-89",
                        cvss_score=9.8,
                        test_method="sql_injection_login",
                    )
                )

            # Test in password field
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "test_user", "password": payload},
            )

            if self._detect_sql_injection_success(response, payload):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SQL_INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        title="SQL Injection in Login Password Field",
                        description="The login endpoint is vulnerable to SQL injection through the password parameter.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"POST /api/v1/auth/login\n"
                        f"Content-Type: application/json\n\n"
                        f'{{"username": "test", "password": "{payload}"}}',
                        exploitation_steps=[
                            "1. Send POST request to /api/v1/auth/login",
                            "2. Use valid username",
                            f"3. Use payload in password field: {payload}",
                            "4. Authentication bypass achieved",
                        ],
                        remediation_steps=[
                            "Implement parameterized queries for password verification",
                            "Hash passwords using bcrypt or similar strong algorithms",
                            "Never include password in SQL queries directly",
                            "Use constant-time comparison for password verification",
                        ],
                        cwe_id="CWE-89",
                        cvss_score=9.8,
                        test_method="sql_injection_login",
                    )
                )

    async def _test_nosql_injection_auth(self):
        """Test NoSQL injection vulnerabilities in authentication."""
        logger.info("Testing NoSQL injection in authentication")

        payloads = generate_nosql_injection_payloads()

        for payload in payloads:
            # Test MongoDB-style injection
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": payload, "password": "any_password"},
            )

            if self._detect_nosql_injection_success(response):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.NOSQL_INJECTION,
                        severity=SeverityLevel.HIGH,
                        title="NoSQL Injection in Authentication",
                        description="The authentication system is vulnerable to NoSQL injection attacks.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"POST /api/v1/auth/login\n"
                        f"Content-Type: application/json\n\n"
                        f'{{"username": {json.dumps(payload)}, "password": "test"}}',
                        exploitation_steps=[
                            "1. Send POST request to /api/v1/auth/login",
                            f"2. Use NoSQL operator in username: {json.dumps(payload)}",
                            "3. Authentication bypass achieved",
                        ],
                        remediation_steps=[
                            "Validate input types before database queries",
                            "Use MongoDB's $where operator restrictions",
                            "Implement proper input sanitization",
                            "Use parameterized queries for NoSQL databases",
                        ],
                        cwe_id="CWE-943",
                        cvss_score=8.1,
                        test_method="nosql_injection_auth",
                    )
                )

    async def _test_ldap_injection(self):
        """Test LDAP injection vulnerabilities."""
        logger.info("Testing LDAP injection in authentication")

        payloads = generate_ldap_injection_payloads()

        for payload in payloads:
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": payload, "password": "any_password"},
            )

            if self._detect_ldap_injection_success(response):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.LDAP_INJECTION,
                        severity=SeverityLevel.HIGH,
                        title="LDAP Injection in Authentication",
                        description="The authentication system is vulnerable to LDAP injection attacks.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"POST /api/v1/auth/login\n"
                        f"Content-Type: application/json\n\n"
                        f'{{"username": "{payload}", "password": "test"}}',
                        exploitation_steps=[
                            "1. Send POST request to /api/v1/auth/login",
                            f"2. Use LDAP injection payload: {payload}",
                            "3. Authentication bypass achieved",
                        ],
                        remediation_steps=[
                            "Escape special LDAP characters in user input",
                            "Use LDAP libraries with built-in injection protection",
                            "Validate input against allowlist patterns",
                            "Implement proper error handling for LDAP operations",
                        ],
                        cwe_id="CWE-90",
                        cvss_score=7.5,
                        test_method="ldap_injection",
                    )
                )

    async def _test_jwt_manipulation(self):
        """Test JWT manipulation attacks."""
        logger.info("Testing JWT manipulation vulnerabilities")

        # First, get a valid token
        username, password, user_id = self.create_test_user()

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )

        if response.status_code != 200:
            logger.warning("Could not obtain valid JWT token for manipulation testing")
            return

        token_data = response.json()
        original_token = token_data.get("access_token")

        if not original_token:
            logger.warning("No access token in login response")
            return

        # Generate manipulation payloads
        manipulation_payloads = generate_jwt_manipulation_payloads(original_token)

        for manipulated_token in manipulation_payloads:
            # Test if manipulated token is accepted
            headers = {"Authorization": f"Bearer {manipulated_token}"}
            response = self.client.get("/api/v1/auth/me", headers=headers)

            if response.status_code == 200:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.JWT_MANIPULATION,
                        severity=SeverityLevel.CRITICAL,
                        title="JWT Token Manipulation Vulnerability",
                        description="The application accepts manipulated JWT tokens, allowing authentication bypass.",
                        affected_endpoint="/api/v1/auth/me",
                        proof_of_concept=f"GET /api/v1/auth/me\n"
                        f"Authorization: Bearer {manipulated_token[:50]}...",
                        exploitation_steps=[
                            "1. Obtain a valid JWT token",
                            "2. Manipulate the token algorithm or payload",
                            "3. Use manipulated token to access protected resources",
                            "4. Authentication bypass achieved",
                        ],
                        remediation_steps=[
                            "Use strong cryptographic algorithms (RS256 instead of HS256)",
                            "Validate JWT algorithm in token verification",
                            "Use proper key management for token signing",
                            "Implement token blacklisting for revocation",
                            "Add token binding to client fingerprints",
                        ],
                        cwe_id="CWE-345",
                        cvss_score=9.1,
                        test_method="jwt_manipulation",
                    )
                )

        # Test algorithm confusion attack specifically
        await self._test_algorithm_confusion(original_token)

        # Test token signature bypass
        await self._test_signature_bypass(original_token)

    async def _test_algorithm_confusion(self, original_token: str):
        """Test JWT algorithm confusion attack."""
        try:
            import jwt as pyjwt

            # Decode the original token
            payload = pyjwt.decode(original_token, options={"verify_signature": False})

            # Try to sign with "none" algorithm
            none_token = pyjwt.encode(payload, "", algorithm="none")

            headers = {"Authorization": f"Bearer {none_token}"}
            response = self.client.get("/api/v1/auth/me", headers=headers)

            if response.status_code == 200:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.JWT_MANIPULATION,
                        severity=SeverityLevel.CRITICAL,
                        title="JWT Algorithm Confusion - None Algorithm",
                        description="The application accepts JWT tokens with 'none' algorithm, bypassing signature verification.",
                        affected_endpoint="/api/v1/auth/me",
                        proof_of_concept=f"Token with alg: none accepted\nPayload: {json.dumps(payload, indent=2)}",
                        exploitation_steps=[
                            "1. Obtain valid JWT token",
                            "2. Decode token payload",
                            "3. Create new token with algorithm 'none'",
                            "4. Access protected resources",
                        ],
                        remediation_steps=[
                            "Explicitly reject tokens with 'none' algorithm",
                            "Whitelist allowed algorithms in JWT verification",
                            "Use asymmetric algorithms (RS256) instead of symmetric (HS256)",
                        ],
                        cwe_id="CWE-345",
                        cvss_score=9.8,
                        test_method="algorithm_confusion",
                    )
                )

        except Exception as e:
            logger.debug(f"Algorithm confusion test error: {e}")

    async def _test_signature_bypass(self, original_token: str):
        """Test JWT signature bypass techniques."""
        try:
            import jwt as pyjwt

            # Test with modified payload but original signature
            payload = pyjwt.decode(original_token, options={"verify_signature": False})
            pyjwt.get_unverified_header(original_token)

            # Modify role to admin
            if "role" in payload:
                modified_payload = payload.copy()
                modified_payload["role"] = "admin"

                # Create token with modified payload but try to keep original signature structure
                modified_token = pyjwt.encode(modified_payload, "fake_secret", algorithm="HS256")

                headers = {"Authorization": f"Bearer {modified_token}"}
                response = self.client.get("/api/v1/auth/permissions", headers=headers)

                if response.status_code == 200:
                    perms = response.json()
                    if perms.get("can_admin_system", False):
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.JWT_MANIPULATION,
                                severity=SeverityLevel.CRITICAL,
                                title="JWT Role Escalation via Token Manipulation",
                                description="Successfully escalated privileges by modifying JWT token role claim.",
                                affected_endpoint="/api/v1/auth/permissions",
                                proof_of_concept=f"Modified token accepted with admin role\nOriginal role: {payload.get('role')}\nNew role: admin",
                                exploitation_steps=[
                                    "1. Obtain valid JWT token",
                                    "2. Decode and modify role claim to 'admin'",
                                    "3. Re-sign token with weak/guessed secret",
                                    "4. Access admin-only resources",
                                ],
                                remediation_steps=[
                                    "Use strong, random secrets for JWT signing",
                                    "Implement proper role validation on server side",
                                    "Use asymmetric signing algorithms",
                                    "Add additional authorization checks beyond JWT claims",
                                ],
                                cwe_id="CWE-862",
                                cvss_score=9.3,
                                test_method="signature_bypass",
                            )
                        )

        except Exception as e:
            logger.debug(f"Signature bypass test error: {e}")

    async def _test_session_fixation(self):
        """Test session fixation vulnerabilities."""
        logger.info("Testing session fixation vulnerabilities")

        # Create a test user
        username, password, user_id = self.create_test_user()

        # Try to set a specific session ID before login
        # This would depend on how sessions are implemented

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
            headers={"X-Session-ID": "attacker_controlled_session"},
        )

        if response.status_code == 200:
            # Check if the application uses the provided session ID
            if (
                "attacker_controlled_session" in str(response.headers)
                or "attacker_controlled_session" in response.text
            ):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SESSION_FIXATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Session Fixation Vulnerability",
                        description="The application accepts externally provided session identifiers.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept="POST /api/v1/auth/login\nX-Session-ID: attacker_controlled_session",
                        exploitation_steps=[
                            "1. Attacker provides session ID to victim",
                            "2. Victim logs in with attacker's session ID",
                            "3. Attacker uses the known session ID to hijack session",
                        ],
                        remediation_steps=[
                            "Generate new session ID upon successful login",
                            "Ignore externally provided session identifiers",
                            "Implement proper session management",
                            "Use secure session configuration",
                        ],
                        cwe_id="CWE-384",
                        cvss_score=6.1,
                        test_method="session_fixation",
                    )
                )

    async def _test_brute_force_bypass(self):
        """Test brute force protection bypass techniques."""
        logger.info("Testing brute force protection bypass")

        username, password, user_id = self.create_test_user()

        # Test rate limiting bypass techniques
        bypass_techniques = [
            # Different user agents
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            # X-Forwarded-For header manipulation
            {"X-Forwarded-For": "1.1.1.1"},
            {"X-Forwarded-For": "8.8.8.8"},
            # X-Real-IP header manipulation
            {"X-Real-IP": "1.1.1.1"},
            # Different client fingerprints
            {"X-Client-Fingerprint": "fingerprint1"},
            {"X-Client-Fingerprint": "fingerprint2"},
        ]

        successful_attempts = 0
        total_attempts = 0

        for headers in bypass_techniques:
            for i in range(15):  # Try 15 rapid attempts
                response = self.client.post(
                    "/api/v1/auth/login",
                    json={"username": username, "password": "wrong_password"},
                    headers=headers,
                )
                total_attempts += 1

                if response.status_code != 429:  # Not rate limited
                    successful_attempts += 1

        bypass_ratio = successful_attempts / total_attempts if total_attempts > 0 else 0

        if bypass_ratio > 0.5:  # More than 50% of attempts succeeded
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                    severity=SeverityLevel.MEDIUM,
                    title="Brute Force Protection Bypass",
                    description=f"Rate limiting can be bypassed using header manipulation. "
                    f"{successful_attempts}/{total_attempts} attempts succeeded.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept="Multiple rapid login attempts with different headers bypass rate limiting",
                    exploitation_steps=[
                        "1. Identify rate limiting implementation",
                        "2. Use different User-Agent, X-Forwarded-For, or X-Real-IP headers",
                        "3. Perform brute force attack with header rotation",
                        "4. Bypass rate limiting protection",
                    ],
                    remediation_steps=[
                        "Implement rate limiting based on multiple factors",
                        "Use account-based rate limiting instead of IP-based only",
                        "Implement progressive delays for failed attempts",
                        "Add CAPTCHA after multiple failed attempts",
                        "Monitor and alert on brute force patterns",
                    ],
                    cwe_id="CWE-307",
                    cvss_score=5.3,
                    test_method="brute_force_bypass",
                )
            )

    async def _test_password_reset_bypass(self):
        """Test password reset vulnerabilities."""
        logger.info("Testing password reset bypass vulnerabilities")

        # This would require implementation of password reset endpoint
        # For now, test common password reset vulnerabilities conceptually

        # Test if password reset endpoint exists
        response = self.client.post(
            "/api/v1/auth/reset-password", json={"email": "test@example.com"}
        )

        if response.status_code in [200, 201, 202]:
            # Test for information disclosure
            if "not found" not in response.text.lower() and "invalid" not in response.text.lower():
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        severity=SeverityLevel.LOW,
                        title="Password Reset Information Disclosure",
                        description="Password reset endpoint may disclose whether email addresses exist in the system.",
                        affected_endpoint="/api/v1/auth/reset-password",
                        proof_of_concept='POST /api/v1/auth/reset-password\n{"email": "test@example.com"}',
                        exploitation_steps=[
                            "1. Send password reset requests for various email addresses",
                            "2. Observe differences in responses",
                            "3. Enumerate valid email addresses",
                        ],
                        remediation_steps=[
                            "Return identical responses for valid and invalid emails",
                            "Implement rate limiting on password reset requests",
                            "Use generic success messages",
                        ],
                        cwe_id="CWE-203",
                        cvss_score=3.7,
                        test_method="password_reset_info_disclosure",
                    )
                )

    async def _test_timing_attacks(self):
        """Test timing attack vulnerabilities in authentication."""
        logger.info("Testing timing attacks in authentication")

        # Create test user
        username, password, user_id = self.create_test_user()

        # Measure timing for valid vs invalid usernames
        valid_times = []
        invalid_times = []

        for i in range(10):
            # Time valid username with wrong password
            start_time = time.time()
            self.client.post(
                "/api/v1/auth/login",
                json={"username": username, "password": "wrong_password"},
            )
            valid_times.append(time.time() - start_time)

            # Time invalid username
            start_time = time.time()
            self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"nonexistent_user_{i}",
                    "password": "wrong_password",
                },
            )
            invalid_times.append(time.time() - start_time)

        avg_valid_time = sum(valid_times) / len(valid_times)
        avg_invalid_time = sum(invalid_times) / len(invalid_times)

        # If there's a significant timing difference, it may indicate vulnerability
        timing_diff_ratio = abs(avg_valid_time - avg_invalid_time) / min(
            avg_valid_time, avg_invalid_time
        )

        if timing_diff_ratio > 0.1:  # 10% difference threshold
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                    severity=SeverityLevel.LOW,
                    title="Authentication Timing Attack Vulnerability",
                    description=f"Timing differences in authentication responses may allow username enumeration. "
                    f"Valid users: {avg_valid_time:.4f}s, Invalid users: {avg_invalid_time:.4f}s",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"Average response times show {timing_diff_ratio:.2%} difference",
                    exploitation_steps=[
                        "1. Measure authentication response times for various usernames",
                        "2. Identify timing patterns for valid vs invalid usernames",
                        "3. Enumerate valid usernames based on response timing",
                    ],
                    remediation_steps=[
                        "Implement constant-time string comparison",
                        "Add random delays to normalize response times",
                        "Use consistent processing paths for valid and invalid inputs",
                        "Hash passwords even for non-existent users",
                    ],
                    cwe_id="CWE-208",
                    cvss_score=2.7,
                    test_method="timing_attacks",
                )
            )

    async def _test_weak_credentials(self):
        """Test for weak credential acceptance."""
        logger.info("Testing weak credential acceptance")

        weak_passwords = [
            "123456",
            "password",
            "admin",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
            "pass",
            "master",
            "hello",
            "a",
            "aa",
            "123",
            "",
        ]

        for weak_password in weak_passwords:
            try:
                response = self.client.post(
                    "/api/v1/auth/register",
                    json={
                        "username": f"test_weak_{int(time.time())}",
                        "email": f"test_{int(time.time())}@example.com",
                        "password": weak_password,
                        "role": "observer",
                    },
                )

                if response.status_code in [200, 201]:
                    self.add_vulnerability(
                        VulnerabilityFinding(
                            vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                            severity=SeverityLevel.MEDIUM,
                            title="Weak Password Policy",
                            description=f"The system accepts weak passwords. Password '{weak_password}' was accepted.",
                            affected_endpoint="/api/v1/auth/register",
                            proof_of_concept=f"Registration successful with password: '{weak_password}'",
                            exploitation_steps=[
                                "1. Attempt registration with common weak passwords",
                                "2. Gain access to accounts with predictable passwords",
                                "3. Perform brute force attacks using common passwords",
                            ],
                            remediation_steps=[
                                "Implement strong password policy (minimum length, complexity)",
                                "Check passwords against common password lists",
                                "Require mix of uppercase, lowercase, numbers, and symbols",
                                "Implement password strength meter for users",
                            ],
                            cwe_id="CWE-521",
                            cvss_score=4.3,
                            test_method="weak_credentials",
                        )
                    )
                    break  # Only report once

            except Exception as e:
                logger.debug(f"Weak password test error: {e}")

    async def _test_account_enumeration(self):
        """Test for account enumeration vulnerabilities."""
        logger.info("Testing account enumeration vulnerabilities")

        # Create a test user to have a known valid account
        username, password, user_id = self.create_test_user()

        # Test login endpoint for different responses
        valid_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": "wrong_password"},
        )

        invalid_response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": "definitely_nonexistent_user_12345",
                "password": "wrong_password",
            },
        )

        # Compare responses for information leakage
        if (
            valid_response.status_code != invalid_response.status_code
            or valid_response.text != invalid_response.text
        ):
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                    severity=SeverityLevel.LOW,
                    title="Account Enumeration via Login Response Differences",
                    description="Different responses for valid vs invalid usernames allow account enumeration.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"Valid user response: {valid_response.status_code}\n"
                    f"Invalid user response: {invalid_response.status_code}",
                    exploitation_steps=[
                        "1. Send login requests with various usernames",
                        "2. Analyze response codes and messages",
                        "3. Identify valid usernames based on response differences",
                    ],
                    remediation_steps=[
                        "Return identical error messages for invalid username and password",
                        "Use generic error messages like 'Invalid credentials'",
                        "Implement consistent response timing",
                        "Consider implementing account lockout after multiple failures",
                    ],
                    cwe_id="CWE-203",
                    cvss_score=3.7,
                    test_method="account_enumeration",
                )
            )

    def _detect_sql_injection_success(self, response, payload: str) -> bool:
        """Detect if SQL injection was successful."""
        # Check for authentication bypass (status 200 with token)
        if response.status_code == 200:
            try:
                data = response.json()
                if "access_token" in data:
                    return True
            except Exception:
                pass

        # Check for SQL error messages that indicate injection
        error_indicators = [
            "syntax error",
            "mysql",
            "postgresql",
            "sqlite",
            "oracle",
            "sql error",
            "database error",
            "column",
            "table",
            "select",
            "unexpected token",
            "unclosed quotation mark",
            "'",
            '"',
        ]

        response_text = response.text.lower()
        return any(indicator in response_text for indicator in error_indicators)

    def _detect_nosql_injection_success(self, response) -> bool:
        """Detect if NoSQL injection was successful."""
        if response.status_code == 200:
            try:
                data = response.json()
                if "access_token" in data:
                    return True
            except Exception:
                pass

        # Check for MongoDB error messages
        nosql_errors = [
            "mongodb",
            "bson",
            "objectid",
            "gridfs",
            "$where",
            "$regex",
        ]
        response_text = response.text.lower()
        return any(error in response_text for error in nosql_errors)

    def _detect_ldap_injection_success(self, response) -> bool:
        """Detect if LDAP injection was successful."""
        if response.status_code == 200:
            try:
                data = response.json()
                if "access_token" in data:
                    return True
            except Exception:
                pass

        # Check for LDAP error messages
        ldap_errors = [
            "ldap",
            "distinguished name",
            "objectclass",
            "ldapexception",
        ]
        response_text = response.text.lower()
        return any(error in response_text for error in ldap_errors)

    def _get_payload_count(self) -> int:
        """Get total number of payloads tested."""
        return (
            len(generate_sql_injection_payloads()) * 2  # username and password fields
            + len(generate_nosql_injection_payloads())
            + len(generate_ldap_injection_payloads())
            + 10  # JWT manipulation attempts
            + 15 * 7  # Brute force bypass attempts
            + 14  # Weak passwords
            + 10
        )  # Timing attack measurements
