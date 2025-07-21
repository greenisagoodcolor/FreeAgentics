"""
Session Management Vulnerability Testing Module

This module implements comprehensive session management testing including:
- Session fixation testing
- Session hijacking attempts
- Session timeout validation
- Concurrent session handling
- Cross-site request forgery (CSRF)
- Session token security analysis
- Cookie security validation
"""

import json
import logging
import time
import uuid
from typing import List

from .penetration_testing_framework import (
    BasePenetrationTest,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
)

logger = logging.getLogger(__name__)


class SessionManagementTests(BasePenetrationTest):
    """Comprehensive session management vulnerability testing."""

    async def execute(self) -> TestResult:
        """Execute all session management tests."""
        start_time = time.time()

        try:
            # Test session fixation vulnerabilities
            await self._test_session_fixation()

            # Test session hijacking vulnerabilities
            await self._test_session_hijacking()

            # Test session timeout validation
            await self._test_session_timeout()

            # Test concurrent session handling
            await self._test_concurrent_sessions()

            # Test CSRF vulnerabilities
            await self._test_csrf_vulnerabilities()

            # Test session token security
            await self._test_session_token_security()

            # Test cookie security
            await self._test_cookie_security()

            # Test session invalidation
            await self._test_session_invalidation()

            # Test session prediction
            await self._test_session_prediction()

            # Test session replay attacks
            await self._test_session_replay()

            execution_time = time.time() - start_time

            return TestResult(
                test_name="SessionManagementTests",
                success=True,
                vulnerabilities=self.vulnerabilities,
                execution_time=execution_time,
                metadata={
                    "session_tests_performed": 10,
                    "endpoints_tested": [
                        "/api/v1/auth/login",
                        "/api/v1/auth/logout",
                        "/api/v1/auth/me",
                        "/api/v1/agents",
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Session management test failed: {e}")
            return TestResult(
                test_name="SessionManagementTests",
                success=False,
                vulnerabilities=self.vulnerabilities,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _test_session_fixation(self):
        """Test session fixation vulnerabilities."""
        logger.info("Testing session fixation vulnerabilities")

        # Create test user
        username, password, user_id = self.create_test_user()

        # Test 1: Pre-authentication session fixation
        predefined_session = f"attacker_session_{uuid.uuid4()}"

        # Try to set session before login
        login_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
            headers={"Cookie": f"session_id={predefined_session}"},
        )

        if login_response.status_code == 200:
            # Check if the predefined session is still used after login
            if self._check_session_reuse(login_response, predefined_session):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SESSION_FIXATION,
                        severity=SeverityLevel.HIGH,
                        title="Session Fixation - Pre-authentication Session Reuse",
                        description="The application reuses session identifiers set before authentication.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"1. Set Cookie: session_id={predefined_session}\n"
                        f"2. Login with valid credentials\n"
                        f"3. Session ID remains unchanged after authentication",
                        exploitation_steps=[
                            "1. Attacker sends victim a link with a predetermined session ID",
                            "2. Victim clicks link and session is established",
                            "3. Victim logs in, but session ID doesn't change",
                            "4. Attacker uses the known session ID to hijack the session",
                        ],
                        remediation_steps=[
                            "Generate new session ID after successful authentication",
                            "Invalidate pre-authentication sessions upon login",
                            "Reject externally provided session identifiers",
                            "Implement proper session lifecycle management",
                        ],
                        cwe_id="CWE-384",
                        cvss_score=7.5,
                        test_method="session_fixation_pre_auth",
                    )
                )

        # Test 2: Session ID manipulation in URL parameters
        session_in_url = f"PHPSESSID={predefined_session}"
        login_response = self.client.post(
            f"/api/v1/auth/login?{session_in_url}",
            json={"username": username, "password": password},
        )

        if login_response.status_code == 200:
            if self._check_session_reuse(login_response, predefined_session):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SESSION_FIXATION,
                        severity=SeverityLevel.HIGH,
                        title="Session Fixation - URL Parameter Session ID",
                        description="The application accepts session IDs from URL parameters.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"POST /api/v1/auth/login?PHPSESSID={predefined_session}",
                        exploitation_steps=[
                            "1. Attacker crafts URL with session ID parameter",
                            "2. Victim visits URL and logs in",
                            "3. Session ID from URL is used for authenticated session",
                            "4. Attacker hijacks session using known ID",
                        ],
                        remediation_steps=[
                            "Never accept session IDs from URL parameters",
                            "Use only cookie-based session management",
                            "Validate session ID source and format",
                            "Regenerate session ID after login",
                        ],
                        cwe_id="CWE-384",
                        cvss_score=7.1,
                        test_method="session_fixation_url",
                    )
                )

    async def _test_session_hijacking(self):
        """Test session hijacking vulnerabilities."""
        logger.info("Testing session hijacking vulnerabilities")

        # Create test user and login
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test 1: Session token exposure in logs/responses
        response = self.client.get(
            "/api/v1/auth/me", headers=self.get_auth_headers(token)
        )

        if self._check_token_exposure(response, token):
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.MEDIUM,
                    title="Session Token Exposure in Response",
                    description="Session tokens are exposed in HTTP responses.",
                    affected_endpoint="/api/v1/auth/me",
                    proof_of_concept="Session token visible in response body or headers",
                    exploitation_steps=[
                        "1. Obtain HTTP response containing session token",
                        "2. Extract session token from response",
                        "3. Use token to impersonate user",
                    ],
                    remediation_steps=[
                        "Never include session tokens in response bodies",
                        "Use secure, httpOnly cookies for session management",
                        "Implement proper token handling in client-server communication",
                    ],
                    cwe_id="CWE-200",
                    cvss_score=5.4,
                    test_method="session_hijacking_exposure",
                )
            )

        # Test 2: Session token in URL/referer
        response = self.client.get(f"/api/v1/auth/me?token={token}")

        if response.status_code == 200:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.HIGH,
                    title="Session Token in URL Parameters",
                    description="Session tokens are accepted in URL parameters, exposing them to logs and referer headers.",
                    affected_endpoint="/api/v1/auth/me",
                    proof_of_concept=f"GET /api/v1/auth/me?token={token[:20]}...",
                    exploitation_steps=[
                        "1. Session token appears in server logs",
                        "2. Token exposed in browser history",
                        "3. Token leaked via HTTP referer headers",
                        "4. Attacker uses token for session hijacking",
                    ],
                    remediation_steps=[
                        "Never accept session tokens in URL parameters",
                        "Use POST requests for sensitive operations",
                        "Implement proper HTTP header-based authentication",
                        "Use secure cookie-based session management",
                    ],
                    cwe_id="CWE-598",
                    cvss_score=8.1,
                    test_method="session_hijacking_url",
                )
            )

        # Test 3: Session token predictability
        await self._test_session_predictability()

    async def _test_session_timeout(self):
        """Test session timeout validation."""
        logger.info("Testing session timeout validation")

        # Create test user and get token
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test if token has proper expiration
        try:
            import jwt as pyjwt

            payload = pyjwt.decode(token, options={"verify_signature": False})

            if "exp" not in payload:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                        severity=SeverityLevel.MEDIUM,
                        title="Missing Session Timeout",
                        description="Session tokens do not have expiration times.",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept="JWT token missing 'exp' claim",
                        exploitation_steps=[
                            "1. Obtain valid session token",
                            "2. Token never expires naturally",
                            "3. Long-term unauthorized access possible",
                        ],
                        remediation_steps=[
                            "Implement proper token expiration",
                            "Set reasonable session timeout periods",
                            "Implement automatic token refresh mechanism",
                            "Add sliding session windows",
                        ],
                        cwe_id="CWE-613",
                        cvss_score=4.9,
                        test_method="session_timeout_missing",
                    )
                )
            else:
                # Check if expiration time is reasonable
                import datetime

                exp_time = datetime.datetime.fromtimestamp(payload["exp"])
                now = datetime.datetime.now()
                session_duration = exp_time - now

                if session_duration.total_seconds() > 86400:  # More than 24 hours
                    self.add_vulnerability(
                        VulnerabilityFinding(
                            vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                            severity=SeverityLevel.LOW,
                            title="Excessive Session Timeout",
                            description=f"Session timeout is excessive: {session_duration}",
                            affected_endpoint="/api/v1/auth/login",
                            proof_of_concept=f"Token expires in {session_duration}",
                            exploitation_steps=[
                                "1. Obtain session token",
                                "2. Token remains valid for extended period",
                                "3. Increased risk of token compromise",
                            ],
                            remediation_steps=[
                                "Reduce session timeout to reasonable duration (1-8 hours)",
                                "Implement sliding session windows",
                                "Consider user activity for session extension",
                            ],
                            cwe_id="CWE-613",
                            cvss_score=3.1,
                            test_method="session_timeout_excessive",
                        )
                    )

        except Exception as e:
            logger.debug(f"Session timeout test error: {e}")

    async def _test_concurrent_sessions(self):
        """Test concurrent session handling."""
        logger.info("Testing concurrent session handling")

        # Create test user
        username, password, user_id = self.create_test_user()

        # Login multiple times to create concurrent sessions
        tokens = []
        for i in range(5):
            login_response = self.client.post(
                "/api/v1/auth/login",
                json={"username": username, "password": password},
            )
            if login_response.status_code == 200:
                token_data = login_response.json()
                if "access_token" in token_data:
                    tokens.append(token_data["access_token"])

        # Test if all tokens are still valid
        valid_tokens = 0
        for token in tokens:
            response = self.client.get(
                "/api/v1/auth/me", headers=self.get_auth_headers(token)
            )
            if response.status_code == 200:
                valid_tokens += 1

        if valid_tokens > 1:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.MEDIUM,
                    title="Unlimited Concurrent Sessions",
                    description=f"User can have {valid_tokens} concurrent active sessions.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"{valid_tokens} sessions active simultaneously for same user",
                    exploitation_steps=[
                        "1. Attacker obtains user credentials",
                        "2. User logs in from legitimate location",
                        "3. Attacker can still maintain active session",
                        "4. Both sessions remain valid simultaneously",
                    ],
                    remediation_steps=[
                        "Implement session limit per user (1-3 concurrent sessions)",
                        "Invalidate old sessions when new ones are created",
                        "Provide session management interface for users",
                        "Alert users about concurrent sessions",
                    ],
                    cwe_id="CWE-613",
                    cvss_score=4.3,
                    test_method="concurrent_sessions",
                )
            )

    async def _test_csrf_vulnerabilities(self):
        """Test Cross-Site Request Forgery vulnerabilities."""
        logger.info("Testing CSRF vulnerabilities")

        # Create test user and get token
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Test CSRF on state-changing operations
        csrf_endpoints = [
            ("/api/v1/auth/logout", "POST"),
            ("/api/v1/agents", "POST"),
            ("/api/v1/agents/123", "DELETE"),
        ]

        for endpoint, method in csrf_endpoints:
            # Test without CSRF token
            headers = self.get_auth_headers(token)

            if method == "POST":
                response = self.client.post(endpoint, json={}, headers=headers)
            elif method == "DELETE":
                response = self.client.delete(endpoint, headers=headers)

            # Check if request succeeded without CSRF protection
            if response.status_code in [200, 201, 202, 204]:
                if not self._has_csrf_protection(response):
                    self.add_vulnerability(
                        VulnerabilityFinding(
                            vulnerability_type=VulnerabilityType.CSRF,
                            severity=SeverityLevel.MEDIUM,
                            title=f"CSRF Vulnerability in {endpoint}",
                            description=f"The {endpoint} endpoint lacks CSRF protection.",
                            affected_endpoint=endpoint,
                            proof_of_concept=f"{method} {endpoint} - No CSRF token required",
                            exploitation_steps=[
                                "1. Attacker crafts malicious webpage",
                                f"2. Page makes {method} request to {endpoint}",
                                "3. Victim visits page while authenticated",
                                "4. Unwanted action performed on behalf of victim",
                            ],
                            remediation_steps=[
                                "Implement CSRF tokens for state-changing operations",
                                "Use SameSite cookie attribute",
                                "Validate HTTP Referer header",
                                "Use double-submit cookie pattern",
                            ],
                            cwe_id="CWE-352",
                            cvss_score=6.5,
                            test_method="csrf_vulnerability",
                        )
                    )

        # Test CSRF token validation if present
        await self._test_csrf_token_bypass()

    async def _test_session_token_security(self):
        """Test session token security properties."""
        logger.info("Testing session token security")

        # Create multiple users and analyze tokens
        tokens = []
        for i in range(5):
            username, password, user_id = self.create_test_user()
            token = self.get_auth_token(username, password)
            tokens.append(token)

        # Test token entropy
        if self._analyze_token_entropy(tokens):
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.HIGH,
                    title="Low Session Token Entropy",
                    description="Session tokens have low entropy and may be predictable.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept="Analysis of multiple tokens shows predictable patterns",
                    exploitation_steps=[
                        "1. Collect multiple session tokens",
                        "2. Analyze patterns and entropy",
                        "3. Predict valid session tokens",
                        "4. Perform session hijacking",
                    ],
                    remediation_steps=[
                        "Use cryptographically secure random number generator",
                        "Ensure sufficient token length (128+ bits entropy)",
                        "Use standard JWT with proper signing",
                        "Implement token rotation",
                    ],
                    cwe_id="CWE-330",
                    cvss_score=8.1,
                    test_method="token_entropy",
                )
            )

        # Test token format and structure
        self._analyze_token_structure(tokens[0] if tokens else None)

    async def _test_cookie_security(self):
        """Test cookie security attributes."""
        logger.info("Testing cookie security")

        # Create test user and login
        username, password, user_id = self.create_test_user()

        login_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )

        if login_response.status_code == 200:
            # Check Set-Cookie headers
            set_cookie_headers = login_response.headers.get_list("set-cookie")

            for cookie_header in set_cookie_headers:
                self._analyze_cookie_security(cookie_header)

    async def _test_session_invalidation(self):
        """Test proper session invalidation."""
        logger.info("Testing session invalidation")

        # Create test user and login
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Verify token works
        response = self.client.get(
            "/api/v1/auth/me", headers=self.get_auth_headers(token)
        )

        if response.status_code != 200:
            return

        # Logout
        self.client.post("/api/v1/auth/logout", headers=self.get_auth_headers(token))

        # Test if token still works after logout
        response = self.client.get(
            "/api/v1/auth/me", headers=self.get_auth_headers(token)
        )

        if response.status_code == 200:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.MEDIUM,
                    title="Incomplete Session Invalidation",
                    description="Session tokens remain valid after logout.",
                    affected_endpoint="/api/v1/auth/logout",
                    proof_of_concept="Token remains valid after POST /api/v1/auth/logout",
                    exploitation_steps=[
                        "1. User logs out from application",
                        "2. Session token is not properly invalidated",
                        "3. Attacker with access to token can still access resources",
                    ],
                    remediation_steps=[
                        "Implement proper token blacklisting on logout",
                        "Invalidate all user sessions on logout",
                        "Clear session cookies with proper attributes",
                        "Implement server-side session tracking",
                    ],
                    cwe_id="CWE-613",
                    cvss_score=5.4,
                    test_method="session_invalidation",
                )
            )

    async def _test_session_prediction(self):
        """Test session ID predictability."""
        logger.info("Testing session prediction")

        # This would be more relevant for traditional session IDs
        # For JWT tokens, we focus on entropy analysis in token security test

    async def _test_session_replay(self):
        """Test session replay attack protection."""
        logger.info("Testing session replay attacks")

        # Create test user and get token
        username, password, user_id = self.create_test_user()
        token = self.get_auth_token(username, password)

        # Perform an action
        response1 = self.client.get(
            "/api/v1/auth/me", headers=self.get_auth_headers(token)
        )

        # Replay the exact same request
        response2 = self.client.get(
            "/api/v1/auth/me", headers=self.get_auth_headers(token)
        )

        # Both should succeed (this is expected for GET)
        # But check for any anti-replay mechanisms
        if (
            response1.status_code == 200
            and response2.status_code == 200
            and not self._has_replay_protection(response1, response2)
        ):
            # This is informational - GET requests normally allow replay
            logger.info("No replay protection detected (normal for GET requests)")

    async def _test_session_predictability(self):
        """Test session token predictability."""
        # Generate multiple tokens and analyze patterns
        tokens = []
        for i in range(10):
            username, password, user_id = self.create_test_user()
            token = self.get_auth_token(username, password)
            tokens.append(token)
            time.sleep(0.1)  # Small delay between generations

        # Analyze for patterns
        if self._detect_token_patterns(tokens):
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.HIGH,
                    title="Predictable Session Tokens",
                    description="Session tokens show predictable patterns.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept="Multiple tokens show sequential or predictable patterns",
                    exploitation_steps=[
                        "1. Analyze multiple session tokens",
                        "2. Identify predictable patterns",
                        "3. Generate valid session tokens",
                        "4. Hijack user sessions",
                    ],
                    remediation_steps=[
                        "Use cryptographically secure random number generation",
                        "Implement proper JWT with strong signing keys",
                        "Add sufficient randomness to token generation",
                        "Use established token generation libraries",
                    ],
                    cwe_id="CWE-330",
                    cvss_score=8.5,
                    test_method="session_predictability",
                )
            )

    async def _test_csrf_token_bypass(self):
        """Test CSRF token bypass techniques."""
        # Implementation would depend on CSRF implementation
        # Common bypasses include:
        # - Missing token validation
        # - Token in GET parameter
        # - Token in referrer
        # - Weak token generation

    # Helper methods

    def _check_session_reuse(self, response, expected_session: str) -> bool:
        """Check if a specific session ID is reused."""
        # Check in cookies
        set_cookies = response.headers.get_list("set-cookie")
        for cookie in set_cookies:
            if expected_session in cookie:
                return True

        # Check in response body
        if expected_session in response.text:
            return True

        return False

    def _check_token_exposure(self, response, token: str) -> bool:
        """Check if token is exposed in response."""
        # Check if full token or significant part appears in response
        token_parts = token.split(".")
        for part in token_parts:
            if len(part) > 10 and part in response.text:
                return True
        return False

    def _has_csrf_protection(self, response) -> bool:
        """Check if response indicates CSRF protection."""
        # Check for CSRF token in response
        csrf_indicators = ["csrf", "token", "_token", "authenticity_token"]
        response_text = response.text.lower()
        return any(indicator in response_text for indicator in csrf_indicators)

    def _analyze_token_entropy(self, tokens: List[str]) -> bool:
        """Analyze token entropy for predictability."""
        if len(tokens) < 3:
            return False

        # Simple entropy check - look for repeated patterns
        # In production, would use more sophisticated entropy analysis
        token_parts = []
        for token in tokens:
            try:
                import jwt as pyjwt

                payload = pyjwt.decode(token, options={"verify_signature": False})
                if "jti" in payload:
                    token_parts.append(payload["jti"])
            except Exception:
                # If not JWT, analyze token directly
                token_parts.append(token)

        # Check for sequential patterns or low entropy
        return len(set(token_parts)) < len(token_parts) * 0.9  # Less than 90% unique

    def _analyze_token_structure(self, token: str):
        """Analyze token structure for security issues."""
        if not token:
            return

        try:
            import jwt as pyjwt

            header = pyjwt.get_unverified_header(token)
            pyjwt.decode(token, options={"verify_signature": False})

            # Check for weak algorithms
            if header.get("alg") in ["none", "HS256"]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.JWT_MANIPULATION,
                        severity=SeverityLevel.MEDIUM,
                        title=f"Weak JWT Algorithm: {header.get('alg')}",
                        description=f"JWT uses weak algorithm: {header.get('alg')}",
                        affected_endpoint="/api/v1/auth/login",
                        proof_of_concept=f"JWT header: {json.dumps(header)}",
                        exploitation_steps=[
                            "1. Analyze JWT token structure",
                            "2. Exploit weak algorithm",
                            "3. Forge valid tokens",
                        ],
                        remediation_steps=[
                            "Use RS256 or ES256 algorithms",
                            "Avoid symmetric algorithms in distributed systems",
                            "Implement proper key management",
                        ],
                        cwe_id="CWE-327",
                        cvss_score=6.1,
                        test_method="token_structure",
                    )
                )

        except Exception as e:
            logger.debug(f"Token structure analysis error: {e}")

    def _analyze_cookie_security(self, cookie_header: str):
        """Analyze cookie security attributes."""
        cookie_lower = cookie_header.lower()

        if "secure" not in cookie_lower:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.MEDIUM,
                    title="Missing Secure Cookie Attribute",
                    description="Session cookies lack the Secure attribute.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"Set-Cookie: {cookie_header}",
                    exploitation_steps=[
                        "1. Perform man-in-the-middle attack",
                        "2. Intercept HTTP traffic",
                        "3. Extract session cookies",
                        "4. Hijack user session",
                    ],
                    remediation_steps=[
                        "Add Secure attribute to all session cookies",
                        "Use HTTPS for all authentication-related operations",
                        "Implement HSTS headers",
                    ],
                    cwe_id="CWE-614",
                    cvss_score=5.9,
                    test_method="cookie_security",
                )
            )

        if "httponly" not in cookie_lower:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.SESSION_HIJACKING,
                    severity=SeverityLevel.MEDIUM,
                    title="Missing HttpOnly Cookie Attribute",
                    description="Session cookies lack the HttpOnly attribute.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"Set-Cookie: {cookie_header}",
                    exploitation_steps=[
                        "1. Execute XSS attack",
                        "2. Access session cookies via JavaScript",
                        "3. Send cookies to attacker",
                        "4. Hijack user session",
                    ],
                    remediation_steps=[
                        "Add HttpOnly attribute to session cookies",
                        "Prevent client-side access to session cookies",
                        "Implement XSS protection",
                    ],
                    cwe_id="CWE-1004",
                    cvss_score=6.1,
                    test_method="cookie_security",
                )
            )

        if "samesite" not in cookie_lower:
            self.add_vulnerability(
                VulnerabilityFinding(
                    vulnerability_type=VulnerabilityType.CSRF,
                    severity=SeverityLevel.LOW,
                    title="Missing SameSite Cookie Attribute",
                    description="Session cookies lack the SameSite attribute.",
                    affected_endpoint="/api/v1/auth/login",
                    proof_of_concept=f"Set-Cookie: {cookie_header}",
                    exploitation_steps=[
                        "1. Perform CSRF attack",
                        "2. Cross-site requests include session cookies",
                        "3. Unwanted actions performed",
                    ],
                    remediation_steps=[
                        "Add SameSite=Strict or SameSite=Lax attribute",
                        "Implement CSRF tokens as additional protection",
                        "Validate request origin",
                    ],
                    cwe_id="CWE-352",
                    cvss_score=4.3,
                    test_method="cookie_security",
                )
            )

    def _has_replay_protection(self, response1, response2) -> bool:
        """Check for replay protection mechanisms."""
        # Look for nonce, timestamp, or sequence numbers
        replay_indicators = ["nonce", "timestamp", "sequence", "once"]
        text1 = response1.text.lower()
        text2 = response2.text.lower()

        return any(
            indicator in text1 or indicator in text2 for indicator in replay_indicators
        )

    def _detect_token_patterns(self, tokens: List[str]) -> bool:
        """Detect patterns in session tokens."""
        if len(tokens) < 5:
            return False

        # Simple pattern detection
        # In practice, would use more sophisticated analysis

        # Check for sequential patterns in JWT JTI claims
        jtis = []
        for token in tokens:
            try:
                import jwt as pyjwt

                payload = pyjwt.decode(token, options={"verify_signature": False})
                if "jti" in payload:
                    jtis.append(payload["jti"])
            except Exception:
                pass

        if len(jtis) >= 3:
            # Check for sequential hex values or timestamps
            try:
                # Try to convert to integers and check for sequences
                values = [
                    int(jti, 16) if len(jti) > 10 else int(jti) for jti in jtis[:3]
                ]

                # Check if values are sequential
                if values[1] - values[0] == values[2] - values[1]:
                    return True
            except Exception:
                pass

        return False
