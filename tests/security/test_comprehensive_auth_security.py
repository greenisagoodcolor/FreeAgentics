"""
Comprehensive Authentication Security Test Suite

This test suite provides extensive security testing for authentication flows including:
- Complete authentication workflow security
- Invalid credential handling and timing attacks
- Account lockout and brute force protection
- Session management security
- Token security and manipulation prevention
- Rate limiting and resource exhaustion protection
- Input validation and injection attack prevention
"""

import concurrent.futures
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import jwt
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import (
    AuthenticationManager,
    User,
    UserRole,
    rate_limiter,
)


class SecurityTestResults:
    """Track security test results for reporting."""

    def __init__(self):
        self.attack_attempts = []
        self.vulnerabilities = []
        self.successful_defenses = []
        self.timing_results = []

    def record_attack(
        self,
        attack_type: str,
        target: str,
        payload: str,
        result: str,
        severity: str = "medium",
        timing: float = 0.0,
    ):
        """Record an attack attempt."""
        self.attack_attempts.append(
            {
                "attack_type": attack_type,
                "target": target,
                "payload": payload[:100] + "..." if len(payload) > 100 else payload,
                "result": result,
                "severity": severity,
                "timing": timing,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if result == "successful":
            self.vulnerabilities.append(
                {
                    "type": attack_type,
                    "severity": severity,
                    "description": f"{attack_type} vulnerability in {target}",
                    "payload": payload[:100] + "..." if len(payload) > 100 else payload,
                }
            )
        elif result == "blocked":
            self.successful_defenses.append(
                {"type": attack_type, "target": target, "severity": severity}
            )

    def record_timing(self, operation: str, duration: float, context: str = ""):
        """Record timing information for analysis."""
        self.timing_results.append(
            {
                "operation": operation,
                "duration": duration,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def generate_report(self) -> Dict:
        """Generate comprehensive security test report."""
        total_attacks = len(self.attack_attempts)
        successful_attacks = len(self.vulnerabilities)
        blocked_attacks = len(self.successful_defenses)

        return {
            "summary": {
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "blocked_attacks": blocked_attacks,
                "vulnerabilities_found": len(self.vulnerabilities),
                "security_score": (
                    (blocked_attacks / total_attacks * 100) if total_attacks > 0 else 100
                ),
            },
            "vulnerabilities": self.vulnerabilities,
            "attack_breakdown": {
                attack_type: len(
                    [a for a in self.attack_attempts if a["attack_type"] == attack_type]
                )
                for attack_type in set(a["attack_type"] for a in self.attack_attempts)
            },
            "timing_analysis": self._analyze_timing(),
            "recommendations": self._generate_recommendations(),
        }

    def _analyze_timing(self) -> Dict:
        """Analyze timing patterns for potential vulnerabilities."""
        if not self.timing_results:
            return {}

        operations = {}
        for result in self.timing_results:
            op = result["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(result["duration"])

        analysis = {}
        for op, durations in operations.items():
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)

            analysis[op] = {
                "avg_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration,
                "variance": max_duration - min_duration,
                "samples": len(durations),
            }

        return analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        if self.vulnerabilities:
            recommendations.append(
                "CRITICAL: Vulnerabilities found - immediate remediation required"
            )

        if len(self.vulnerabilities) > 0:
            recommendations.append("Implement additional input validation and sanitization")

        # Check timing attack vulnerability
        timing_variances = []
        for result in self.timing_results:
            if "login" in result["operation"]:
                timing_variances.append(result["duration"])

        if timing_variances:
            variance = max(timing_variances) - min(timing_variances)
            if variance > 0.01:  # 10ms variance
                recommendations.append(
                    "Consider implementing constant-time authentication to prevent timing attacks"
                )

        if len(self.attack_attempts) > 0:
            recommendations.append("Continue monitoring for attack patterns")

        return recommendations


class TestAuthenticationSecuritySuite:
    """Comprehensive authentication security test suite."""

    def setup_method(self):
        """Setup for each test method."""
        self.results = SecurityTestResults()
        self.client = TestClient(app)
        self.auth_manager = AuthenticationManager()

        # Clear any existing test data
        self.auth_manager.users.clear()
        self.auth_manager.refresh_tokens.clear()
        self.auth_manager.blacklist.clear()
        rate_limiter.requests.clear()

        # Create test users
        self.test_users = self._create_test_users()

    def _create_test_users(self) -> List[User]:
        """Create test users with different roles."""
        users = []
        roles = [
            UserRole.ADMIN,
            UserRole.RESEARCHER,
            UserRole.AGENT_MANAGER,
            UserRole.OBSERVER,
        ]

        for i, role in enumerate(roles):
            user = User(
                user_id=f"test-user-{i}",
                username=f"testuser{i}",
                email=f"test{i}@security.test",
                role=role,
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)

            # Register with auth manager
            self.auth_manager.users[user.username] = {
                "user": user,
                "password_hash": self.auth_manager.hash_password(f"SecurePass{i}123!"),
            }

        return users

    def test_comprehensive_authentication_flow_security(self):
        """Test complete authentication flow with security focus."""
        # Test 1: Registration security
        self._test_registration_security()

        # Test 2: Login security
        self._test_login_security()

        # Test 3: Token security
        self._test_token_security()

        # Test 4: Logout security
        self._test_logout_security()

        # Test 5: Session management security
        self._test_session_management_security()

        # Verify overall security posture
        report = self.results.generate_report()
        assert report["summary"]["successful_attacks"] == 0, (
            f"Security vulnerabilities found: {report['vulnerabilities']}"
        )

    def _test_registration_security(self):
        """Test registration endpoint security."""
        # Test duplicate registration attempt
        user_data = {
            "username": "securitytest",
            "email": "security@test.com",
            "password": "SecurePass123!",
            "role": "observer",
        }

        # First registration should succeed
        response1 = self.client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == 200

        # Second registration should fail
        response2 = self.client.post("/api/v1/auth/register", json=user_data)
        if response2.status_code == 200:
            self.results.record_attack(
                "Authentication Bypass",
                "registration",
                "duplicate username",
                "successful",
                "high",
            )
        else:
            self.results.record_attack(
                "Authentication Bypass",
                "registration",
                "duplicate username",
                "blocked",
                "medium",
            )

        # Test registration with malicious input
        malicious_inputs = [
            {
                "username": "'; DROP TABLE users; --",
                "email": "test@test.com",
                "password": "Pass123!",
                "role": "admin",
            },
            {
                "username": "<script>alert('xss')</script>",
                "email": "test@test.com",
                "password": "Pass123!",
                "role": "observer",
            },
            {
                "username": "admin",
                "email": "admin@test.com",
                "password": "Pass123!",
                "role": "admin",
            },  # Role escalation
            {
                "username": "test",
                "email": "test@test.com",
                "password": "a" * 10000,
                "role": "observer",
            },  # Large input
        ]

        for malicious_data in malicious_inputs:
            response = self.client.post("/api/v1/auth/register", json=malicious_data)
            if response.status_code == 200:
                self.results.record_attack(
                    "Input Validation Bypass",
                    "registration",
                    str(malicious_data),
                    "successful",
                    "high",
                )
            else:
                self.results.record_attack(
                    "Input Validation Bypass",
                    "registration",
                    str(malicious_data),
                    "blocked",
                    "medium",
                )

    def _test_login_security(self):
        """Test login endpoint security."""
        # Test with valid user
        valid_user = self.test_users[0]

        # Test correct credentials
        start_time = time.time()
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": valid_user.username,
                "password": "SecurePass0123!",
            },
        )
        valid_login_time = time.time() - start_time

        assert response.status_code == 200
        self.results.record_timing("valid_login", valid_login_time, "correct_credentials")

        # Test incorrect password
        start_time = time.time()
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": valid_user.username,
                "password": "WrongPassword123!",
            },
        )
        invalid_password_time = time.time() - start_time

        self.results.record_timing("invalid_password", invalid_password_time, "wrong_password")

        if response.status_code == 200:
            self.results.record_attack(
                "Authentication Bypass",
                "login",
                "wrong password accepted",
                "successful",
                "critical",
            )
        else:
            self.results.record_attack(
                "Authentication Bypass",
                "login",
                "wrong password rejected",
                "blocked",
                "medium",
            )

        # Test nonexistent user
        start_time = time.time()
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": "nonexistent_user",
                "password": "AnyPassword123!",
            },
        )
        nonexistent_user_time = time.time() - start_time

        self.results.record_timing("nonexistent_user", nonexistent_user_time, "user_not_found")

        # Check for timing attack vulnerability
        timing_variance = abs(valid_login_time - nonexistent_user_time)
        if timing_variance > 0.01:  # 10ms threshold
            self.results.record_attack(
                "Timing Attack",
                "login",
                f"timing variance: {timing_variance:.3f}s",
                "potential_vulnerability",
                "medium",
            )

    def _test_token_security(self):
        """Test JWT token security."""
        # Get valid token
        user = self.test_users[0]
        login_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": user.username, "password": "SecurePass0123!"},
        )

        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Test token manipulation attacks
        token_attacks = [
            (
                "Algorithm Confusion",
                self._create_algorithm_confusion_token(token),
            ),
            ("Signature Stripping", self._strip_token_signature(token)),
            ("Expired Token", self._create_expired_token(token)),
            ("Role Escalation", self._create_role_escalation_token(token)),
            ("Malformed Token", "invalid.token.format"),
            ("Empty Token", ""),
            ("Null Token", None),
        ]

        for attack_name, malicious_token in token_attacks:
            if malicious_token is None:
                continue

            headers = {"Authorization": f"Bearer {malicious_token}"}
            response = self.client.get("/api/v1/auth/me", headers=headers)

            if response.status_code == 200:
                self.results.record_attack(
                    "Token Manipulation",
                    "token_verification",
                    attack_name,
                    "successful",
                    "critical",
                )
            else:
                self.results.record_attack(
                    "Token Manipulation",
                    "token_verification",
                    attack_name,
                    "blocked",
                    "high",
                )

    def _test_logout_security(self):
        """Test logout security."""
        # Login first
        user = self.test_users[0]
        login_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": user.username, "password": "SecurePass0123!"},
        )

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Logout
        logout_response = self.client.post("/api/v1/auth/logout", headers=headers)
        assert logout_response.status_code == 200

        # Try to use token after logout
        response = self.client.get("/api/v1/auth/me", headers=headers)
        if response.status_code == 200:
            self.results.record_attack(
                "Session Management",
                "logout",
                "token still valid after logout",
                "successful",
                "high",
            )
        else:
            self.results.record_attack(
                "Session Management",
                "logout",
                "token invalidated after logout",
                "blocked",
                "medium",
            )

    def _test_session_management_security(self):
        """Test session management security."""
        # Test concurrent sessions
        user = self.test_users[0]
        login_data = {"username": user.username, "password": "SecurePass0123!"}

        # Create multiple sessions
        tokens = []
        for i in range(3):
            response = self.client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 200
            tokens.append(response.json()["access_token"])

        # Verify all tokens are valid
        valid_tokens = 0
        for token in tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.client.get("/api/v1/auth/me", headers=headers)
            if response.status_code == 200:
                valid_tokens += 1

        if valid_tokens == len(tokens):
            self.results.record_attack(
                "Session Management",
                "concurrent_sessions",
                f"{valid_tokens} concurrent sessions allowed",
                "acceptable",
                "low",
            )

    def test_brute_force_protection(self):
        """Test brute force attack protection."""
        user = self.test_users[0]
        failed_attempts = 0

        # Attempt multiple failed logins
        for i in range(15):  # Exceed typical rate limits
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": user.username,
                    "password": f"wrong_password_{i}",
                },
            )

            if response.status_code == 401:
                failed_attempts += 1
            elif response.status_code == 429:  # Rate limited
                self.results.record_attack(
                    "Rate Limiting",
                    "brute_force_protection",
                    f"blocked after {failed_attempts} attempts",
                    "blocked",
                    "medium",
                )
                break
            elif response.status_code == 200:
                self.results.record_attack(
                    "Brute Force",
                    "login",
                    f"successful after {i} attempts",
                    "successful",
                    "critical",
                )
                break

        # Verify that rate limiting is working
        if failed_attempts >= 10:  # No rate limiting after 10 attempts
            self.results.record_attack(
                "Rate Limiting",
                "brute_force_protection",
                "no rate limiting detected",
                "potential_vulnerability",
                "medium",
            )

    def test_account_enumeration_protection(self):
        """Test protection against account enumeration."""
        # Test with valid username
        start_time = time.time()
        response1 = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": self.test_users[0].username,
                "password": "wrong_password",
            },
        )
        valid_user_time = time.time() - start_time

        # Test with invalid username
        start_time = time.time()
        response2 = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": "definitely_not_a_user",
                "password": "wrong_password",
            },
        )
        invalid_user_time = time.time() - start_time

        # Check response consistency
        if response1.status_code != response2.status_code:
            self.results.record_attack(
                "Account Enumeration",
                "login_response",
                "different status codes",
                "potential_vulnerability",
                "medium",
            )

        # Check timing consistency
        timing_difference = abs(valid_user_time - invalid_user_time)
        if timing_difference > 0.01:  # 10ms threshold
            self.results.record_attack(
                "Account Enumeration",
                "login_timing",
                f"timing difference: {timing_difference:.3f}s",
                "potential_vulnerability",
                "medium",
            )
        else:
            self.results.record_attack(
                "Account Enumeration",
                "login_timing",
                "consistent timing",
                "blocked",
                "low",
            )

    def test_input_validation_security(self):
        """Test input validation against injection attacks."""
        # SQL Injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "' OR 1=1--",
            '" OR ""="',
            "1' AND '1'='1",
            "'; EXEC xp_cmdshell('dir'); --",
        ]

        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "';alert(String.fromCharCode(88,83,83))//",
            "<script>document.cookie</script>",
            '<<SCRIPT>alert("XSS");//<</SCRIPT>',
        ]

        # Command injection payloads
        cmd_payloads = [
            "; ls -la",
            "| whoami",
            "& net user",
            "`id`",
            "$(whoami)",
            "; cat /etc/passwd",
            "|| sleep 10",
            "& dir",
        ]

        all_payloads = [
            ("SQL Injection", sql_payloads),
            ("XSS", xss_payloads),
            ("Command Injection", cmd_payloads),
        ]

        for attack_type, payloads in all_payloads:
            for payload in payloads:
                # Test in registration
                registration_data = {
                    "username": payload,
                    "email": "test@test.com",
                    "password": "SecurePass123!",
                    "role": "observer",
                }

                response = self.client.post("/api/v1/auth/register", json=registration_data)
                if response.status_code == 200:
                    self.results.record_attack(
                        attack_type,
                        "registration_username",
                        payload,
                        "successful",
                        "high",
                    )
                else:
                    self.results.record_attack(
                        attack_type,
                        "registration_username",
                        payload,
                        "blocked",
                        "medium",
                    )

                # Test in login
                login_data = {"username": payload, "password": "test123"}

                response = self.client.post("/api/v1/auth/login", json=login_data)
                # Login should fail safely
                if response.status_code not in [400, 401, 422]:
                    self.results.record_attack(
                        attack_type,
                        "login_username",
                        payload,
                        "potential_vulnerability",
                        "medium",
                    )
                else:
                    self.results.record_attack(
                        attack_type,
                        "login_username",
                        payload,
                        "blocked",
                        "low",
                    )

    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        # Large payload attack
        large_payload = "A" * 1000000  # 1MB payload

        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "test",
                "email": "test@test.com",
                "password": large_payload,
                "role": "observer",
            },
        )

        if response.status_code == 200:
            self.results.record_attack(
                "Resource Exhaustion",
                "large_payload",
                "1MB password accepted",
                "potential_vulnerability",
                "medium",
            )
        else:
            self.results.record_attack(
                "Resource Exhaustion",
                "large_payload",
                "1MB password rejected",
                "blocked",
                "low",
            )

        # Rapid request attack
        rapid_requests = 0
        for i in range(100):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "test", "password": "test"},
            )

            if response.status_code == 429:  # Rate limited
                break
            rapid_requests += 1

        if rapid_requests >= 50:  # No rate limiting after 50 requests
            self.results.record_attack(
                "Resource Exhaustion",
                "rapid_requests",
                f"{rapid_requests} requests allowed",
                "potential_vulnerability",
                "medium",
            )
        else:
            self.results.record_attack(
                "Resource Exhaustion",
                "rapid_requests",
                f"limited to {rapid_requests} requests",
                "blocked",
                "low",
            )

    def test_concurrent_authentication_security(self):
        """Test authentication security under concurrent load."""
        results = []

        def concurrent_login_attempt(thread_id):
            """Attempt concurrent login."""
            user = self.test_users[thread_id % len(self.test_users)]

            try:
                response = self.client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": user.username,
                        "password": f"SecurePass{thread_id % len(self.test_users)}123!",
                    },
                )

                return {
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                }
            except Exception as e:
                return {
                    "thread_id": thread_id,
                    "error": str(e),
                    "success": False,
                }

        # Launch concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_login_attempt, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze results
        successful_logins = sum(1 for r in results if r.get("success", False))
        len(results) - successful_logins

        if successful_logins > 0:
            self.results.record_attack(
                "Concurrent Authentication",
                "concurrent_login",
                f"{successful_logins}/{len(results)} successful",
                "acceptable",
                "low",
            )

        # Check for race conditions or errors
        errors = [r for r in results if "error" in r]
        if errors:
            self.results.record_attack(
                "Concurrent Authentication",
                "race_conditions",
                f"{len(errors)} errors occurred",
                "potential_vulnerability",
                "medium",
            )

    def test_token_refresh_security(self):
        """Test token refresh security."""
        # Get tokens
        user = self.test_users[0]
        login_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": user.username, "password": "SecurePass0123!"},
        )

        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]

        # Test valid refresh
        refresh_response = self.client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )

        if refresh_response.status_code != 200:
            self.results.record_attack(
                "Token Refresh",
                "valid_refresh",
                "valid refresh token rejected",
                "potential_issue",
                "medium",
            )

        # Test refresh token reuse
        refresh_response2 = self.client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )

        if refresh_response2.status_code == 200:
            self.results.record_attack(
                "Token Refresh",
                "token_reuse",
                "refresh token reused successfully",
                "potential_vulnerability",
                "medium",
            )
        else:
            self.results.record_attack(
                "Token Refresh",
                "token_reuse",
                "refresh token reuse blocked",
                "blocked",
                "low",
            )

        # Test malformed refresh tokens
        malformed_tokens = [
            "invalid.refresh.token",
            "",
            None,
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature",
        ]

        for token in malformed_tokens:
            if token is None:
                continue

            response = self.client.post("/api/v1/auth/refresh", json={"refresh_token": token})

            if response.status_code == 200:
                self.results.record_attack(
                    "Token Refresh",
                    "malformed_token",
                    "malformed token accepted",
                    "successful",
                    "high",
                )
            else:
                self.results.record_attack(
                    "Token Refresh",
                    "malformed_token",
                    "malformed token rejected",
                    "blocked",
                    "low",
                )

    def test_generate_comprehensive_security_report(self):
        """Generate comprehensive security report."""
        # Run all security tests
        self.test_comprehensive_authentication_flow_security()
        self.test_brute_force_protection()
        self.test_account_enumeration_protection()
        self.test_input_validation_security()
        self.test_resource_exhaustion_protection()
        self.test_concurrent_authentication_security()
        self.test_token_refresh_security()

        # Generate report
        report = self.results.generate_report()

        # Print report for visibility
        print("\\n" + "=" * 60)
        print("COMPREHENSIVE AUTHENTICATION SECURITY REPORT")
        print("=" * 60)
        print(f"Security Score: {report['summary']['security_score']:.1f}%")
        print(f"Total Attacks Tested: {report['summary']['total_attacks']}")
        print(f"Successful Attacks: {report['summary']['successful_attacks']}")
        print(f"Blocked Attacks: {report['summary']['blocked_attacks']}")
        print(f"Vulnerabilities Found: {report['summary']['vulnerabilities_found']}")

        if report["vulnerabilities"]:
            print("\\nVULNERABILITIES FOUND:")
            for vuln in report["vulnerabilities"]:
                print(f"  - {vuln['type']} ({vuln['severity']}): {vuln['description']}")

        print("\\nATTACK BREAKDOWN:")
        for attack_type, count in report["attack_breakdown"].items():
            print(f"  - {attack_type}: {count} attempts")

        if report["recommendations"]:
            print("\\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        # Assert no critical vulnerabilities
        critical_vulns = [v for v in report["vulnerabilities"] if v["severity"] == "critical"]
        assert len(critical_vulns) == 0, f"Critical vulnerabilities found: {critical_vulns}"

        # Assert security score is acceptable
        assert report["summary"]["security_score"] >= 95, (
            f"Security score too low: {report['summary']['security_score']:.1f}%"
        )

        return report

    # Helper methods for token manipulation
    def _create_algorithm_confusion_token(self, valid_token: str) -> str:
        """Create token with algorithm confusion attack."""
        try:
            payload = jwt.decode(valid_token, options={"verify_signature": False})
            return jwt.encode(payload, "secret", algorithm="HS256")
        except Exception:
            return "invalid.algorithm.token"

    def _strip_token_signature(self, valid_token: str) -> str:
        """Strip signature from token."""
        parts = valid_token.split(".")
        if len(parts) == 3:
            return f"{parts[0]}.{parts[1]}."
        return valid_token

    def _create_expired_token(self, valid_token: str) -> str:
        """Create expired token."""
        try:
            payload = jwt.decode(valid_token, options={"verify_signature": False})
            payload["exp"] = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
            # Can't properly sign without private key, so return original for testing
            return valid_token
        except Exception:
            return "invalid.expired.token"

    def _create_role_escalation_token(self, valid_token: str) -> str:
        """Create token with role escalation."""
        try:
            payload = jwt.decode(valid_token, options={"verify_signature": False})
            payload["role"] = "admin"
            payload["permissions"] = [
                "admin_system",
                "create_agent",
                "delete_agent",
            ]
            # Can't properly sign without private key, so return original for testing
            return valid_token
        except Exception:
            return "invalid.role.token"

    def teardown_method(self):
        """Cleanup after each test."""
        self.auth_manager.users.clear()
        self.auth_manager.refresh_tokens.clear()
        self.auth_manager.blacklist.clear()
        rate_limiter.requests.clear()


if __name__ == "__main__":
    # Run security tests
    test_suite = TestAuthenticationSecuritySuite()
    test_suite.setup_method()

    try:
        report = test_suite.test_generate_comprehensive_security_report()
        print("\\nSECURITY TESTING COMPLETED SUCCESSFULLY")
        print(f"Final Security Score: {report['summary']['security_score']:.1f}%")
    except Exception as e:
        print(f"\\nSECURITY TESTING FAILED: {e}")
        raise
    finally:
        test_suite.teardown_method()
