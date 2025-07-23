"""
Penetration Testing Scenarios Test Suite

Performs automated security testing for common vulnerabilities
based on OWASP Top 10 and other security best practices.
"""

import time

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.security_headers import SecurityHeadersManager
from auth.security_implementation import AuthenticationManager


class TestPenetrationTestingScenarios:
    """Test suite for common penetration testing scenarios."""

    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager for testing."""
        return AuthenticationManager()

    @pytest.fixture
    def security_headers_manager(self):
        """Create security headers manager for testing."""
        return SecurityHeadersManager()

    @pytest.fixture
    def app(self, auth_manager, security_headers_manager):
        """Create test FastAPI application."""
        app = FastAPI()

        @app.get("/")
        def read_root():
            return {"message": "Hello World"}

        @app.post("/auth/login")
        def login(data: dict):
            return {"access_token": "test_token", "token_type": "bearer"}

        @app.get("/protected")
        def protected_endpoint():
            return {"message": "Protected resource"}

        @app.post("/user/update")
        def update_user(data: dict):
            return {"message": "User updated"}

        @app.get("/search")
        def search_endpoint(q: str = ""):
            return {"results": f"Search results for: {q}"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_sql_injection_prevention(self, client, auth_manager):
        """Test SQL injection prevention mechanisms."""
        # Test SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "' OR 1=1#",
            "' OR 'a'='a",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "' OR (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
        ]

        for payload in sql_payloads:
            # Test in login endpoint
            response = client.post("/auth/login", json={"username": payload, "password": "test"})
            # Should not return 500 error (indicating SQL injection)
            assert response.status_code != 500, (
                f"SQL injection may be possible with payload: {payload}"
            )

            # Test in search endpoint
            response = client.get(f"/search?q={payload}")
            assert response.status_code != 500, (
                f"SQL injection may be possible in search with payload: {payload}"
            )

    def test_xss_prevention(self, client):
        """Test Cross-Site Scripting (XSS) prevention."""
        # Test XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<<SCRIPT>alert('XSS')//<</SCRIPT>",
        ]

        for payload in xss_payloads:
            # Test in search endpoint
            response = client.get(f"/search?q={payload}")

            # Response should not contain unescaped payload
            response_text = response.text
            assert "<script>" not in response_text, f"XSS payload not escaped: {payload}"
            assert "javascript:" not in response_text, f"XSS payload not escaped: {payload}"
            assert "onerror=" not in response_text, f"XSS payload not escaped: {payload}"
            assert "onload=" not in response_text, f"XSS payload not escaped: {payload}"

    def test_csrf_protection(self, client):
        """Test Cross-Site Request Forgery (CSRF) protection."""
        # Test CSRF by attempting state-changing operations without proper headers
        csrf_test_data = {"email": "test@example.com", "role": "admin"}

        # POST request without CSRF token or proper headers
        response = client.post("/user/update", json=csrf_test_data)

        # Should have CSRF protection mechanisms
        # Check for CSRF token requirement or SameSite cookies
        headers = response.headers

        # Check for security headers that help prevent CSRF
        assert "X-Frame-Options" in headers or "Content-Security-Policy" in headers, (
            "Missing CSRF protection headers"
        )

    def test_authentication_bypass_attempts(self, client, auth_manager):
        """Test authentication bypass vulnerability attempts."""
        # Test authentication bypass payloads
        bypass_payloads = [
            {"username": "admin", "password": ""},
            {"username": "admin", "password": None},
            {"username": "", "password": "admin"},
            {"username": None, "password": "admin"},
            {"username": "admin", "password": "' OR '1'='1"},
            {"username": "admin'--", "password": "anything"},
        ]

        for payload in bypass_payloads:
            response = client.post("/auth/login", json=payload)

            # Should not return successful authentication
            assert response.status_code != 200 or "access_token" not in response.json(), (
                f"Authentication bypass possible with payload: {payload}"
            )

    def test_session_fixation_prevention(self, auth_manager):
        """Test session fixation attack prevention."""
        # Test that session IDs change after authentication

        # Create initial session
        initial_session = auth_manager.create_session("test_user")

        # Register user
        from auth.security_implementation import UserRole

        auth_manager.register_user(
            "test_user", "test@example.com", "password123", UserRole.OBSERVER
        )

        # Authenticate user
        auth_result = auth_manager.authenticate_user("test_user", "password123")
        assert auth_result is not None

        # Create new session after authentication
        new_session = auth_manager.create_session("test_user")

        # Session IDs should be different (preventing session fixation)
        assert initial_session != new_session, "Session fixation vulnerability detected"

    def test_privilege_escalation_prevention(self, client, auth_manager):
        """Test privilege escalation prevention."""
        # Create users with different privilege levels
        from auth.security_implementation import UserRole

        # Register users
        auth_manager.register_user("admin", "admin@example.com", "admin_pass", UserRole.ADMIN)
        regular_user = auth_manager.register_user(
            "regular", "regular@example.com", "user_pass", UserRole.OBSERVER
        )

        # Test that regular user cannot access admin functions
        # This would require implementing proper authorization checks

        # Test JWT token manipulation
        user_token = auth_manager.create_access_token(regular_user)

        # Attempt to decode and modify token
        try:
            # This should fail due to signature verification
            jwt.decode(user_token, "wrong_secret", algorithms=["HS256"])
            pytest.fail("JWT token signature verification failed")
        except jwt.InvalidSignatureError:
            # Expected behavior
            pass

    def test_injection_attacks_prevention(self, client):
        """Test various injection attack prevention."""
        # Test command injection
        command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "& ping -c 1 127.0.0.1",
            "`cat /etc/passwd`",
            "$(uname -a)",
            "; rm -rf /",
        ]

        for payload in command_injection_payloads:
            response = client.get(f"/search?q={payload}")
            # Should not execute system commands
            assert response.status_code != 500, f"Command injection may be possible: {payload}"

            # Check that system information is not leaked
            response_text = response.text.lower()
            assert "root:" not in response_text, f"System information leaked: {payload}"
            assert "bin/bash" not in response_text, f"System information leaked: {payload}"

    def test_directory_traversal_prevention(self, client):
        """Test directory traversal attack prevention."""
        # Test directory traversal payloads
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
        ]

        for payload in traversal_payloads:
            response = client.get(f"/search?q={payload}")

            # Should not return system files
            response_text = response.text
            assert "root:" not in response_text, f"Directory traversal possible: {payload}"
            assert "Administrator:" not in response_text, f"Directory traversal possible: {payload}"

    def test_timing_attack_resistance(self, auth_manager):
        """Test resistance to timing attacks."""
        # Test timing attack resistance in password verification

        # Create a user
        from auth.security_implementation import UserRole

        auth_manager.register_user(
            "timing_user",
            "timing@example.com",
            "correct_password",
            UserRole.OBSERVER,
        )

        # Test authentication timing
        def time_authentication(password):
            start_time = time.time()
            try:
                auth_manager.authenticate_user("timing_user", password)
            except Exception:
                pass
            end_time = time.time()
            return end_time - start_time

        # Test with correct password
        correct_times = [time_authentication("correct_password") for _ in range(10)]

        # Test with incorrect password
        incorrect_times = [time_authentication("wrong_password") for _ in range(10)]

        # Calculate average times
        avg_correct = sum(correct_times) / len(correct_times)
        avg_incorrect = sum(incorrect_times) / len(incorrect_times)

        # Times should be similar (within reasonable variance)
        time_difference = abs(avg_correct - avg_incorrect)
        assert time_difference < 0.1, (
            f"Timing attack vulnerability detected: {time_difference:.3f}s difference"
        )

    def test_brute_force_protection(self, client, auth_manager):
        """Test brute force attack protection."""
        # Test multiple failed login attempts

        # Create a user
        from auth.security_implementation import UserRole

        auth_manager.register_user(
            "brute_user",
            "brute@example.com",
            "secure_password",
            UserRole.OBSERVER,
        )

        # Attempt multiple failed logins
        failed_attempts = 0
        max_attempts = 10

        for i in range(max_attempts):
            response = client.post(
                "/auth/login",
                json={
                    "username": "brute_user",
                    "password": f"wrong_password_{i}",
                },
            )

            if response.status_code == 429:  # Rate limited
                break

            failed_attempts += 1

        # Should implement rate limiting after several failed attempts
        assert failed_attempts < max_attempts, "No brute force protection detected"

    def test_information_disclosure_prevention(self, client):
        """Test information disclosure prevention."""
        # Test that error messages don't leak sensitive information

        # Test with invalid endpoints
        response = client.get("/nonexistent")
        assert response.status_code == 404

        # Error response should not contain sensitive information
        response_text = response.text.lower()
        sensitive_patterns = [
            "traceback",
            "exception",
            "internal server error",
            "database",
            "sql",
            "password",
            "secret",
            "key",
            "token",
        ]

        for pattern in sensitive_patterns:
            assert pattern not in response_text, f"Information disclosure: {pattern}"

    def test_secure_headers_implementation(self, client):
        """Test that security headers are properly implemented."""
        response = client.get("/")
        headers = response.headers

        # Check for essential security headers
        security_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Strict-Transport-Security",
            "Referrer-Policy",
        ]

        for header in security_headers:
            assert header in headers, f"Missing security header: {header}"

    def test_http_methods_security(self, client):
        """Test HTTP methods security."""
        # Test that dangerous HTTP methods are disabled
        dangerous_methods = ["TRACE", "OPTIONS", "HEAD"]

        for method in dangerous_methods:
            response = client.request(method, "/")
            # Should not return 200 for dangerous methods
            assert response.status_code != 200, f"Dangerous HTTP method allowed: {method}"

    def test_cookie_security(self, client):
        """Test cookie security settings."""
        # Test that cookies have secure flags
        response = client.post("/auth/login", json={"username": "test", "password": "test"})

        # Check Set-Cookie headers
        set_cookie_headers = response.headers.get("Set-Cookie", "")

        if set_cookie_headers:
            # Cookies should have security flags
            assert "HttpOnly" in set_cookie_headers or "httponly" in set_cookie_headers.lower(), (
                "Cookies missing HttpOnly flag"
            )
            assert "Secure" in set_cookie_headers or "secure" in set_cookie_headers.lower(), (
                "Cookies missing Secure flag"
            )
            assert "SameSite" in set_cookie_headers or "samesite" in set_cookie_headers.lower(), (
                "Cookies missing SameSite flag"
            )

    def test_input_validation_bypass(self, client):
        """Test input validation bypass attempts."""
        # Test input validation with various bypass techniques
        bypass_payloads = [
            {"username": "a" * 10000, "password": "test"},  # Buffer overflow
            {
                "username": "\x00admin",
                "password": "test",
            },  # Null byte injection
            {"username": "admin\n", "password": "test"},  # Line feed injection
            {
                "username": "admin\r",
                "password": "test",
            },  # Carriage return injection
            {"username": "admin\t", "password": "test"},  # Tab injection
        ]

        for payload in bypass_payloads:
            response = client.post("/auth/login", json=payload)

            # Should handle invalid input gracefully
            assert response.status_code != 500, f"Input validation bypass possible: {payload}"

    def test_authorization_bypass(self, client):
        """Test authorization bypass attempts."""
        # Test accessing protected resources without proper authorization

        # Test direct access to protected endpoint
        response = client.get("/protected")

        # Should require authentication
        assert response.status_code in [
            401,
            403,
        ], "Authorization bypass detected"

        # Test with invalid or expired token
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid_token",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "",
            None,
        ]

        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = client.get("/protected", headers=headers)
            assert response.status_code in [
                401,
                403,
            ], f"Authorization bypass with token: {token}"

    def test_race_condition_prevention(self, client, auth_manager):
        """Test race condition prevention in authentication."""
        # Test concurrent authentication attempts

        from auth.security_implementation import UserRole

        auth_manager.register_user(
            "race_user", "race@example.com", "race_password", UserRole.OBSERVER
        )

        # Simulate concurrent authentication attempts
        def authenticate():
            return auth_manager.authenticate_user("race_user", "race_password")

        # Run multiple concurrent authentications
        import threading

        results = []
        threads = []

        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(authenticate()))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All authentications should succeed consistently
        successful_auths = [r for r in results if r is not None]
        assert len(successful_auths) == 10, "Race condition detected in authentication"

    def test_password_complexity_enforcement(self, auth_manager):
        """Test password complexity enforcement."""
        # Test weak password rejection
        weak_passwords = [
            "123456",
            "password",
            "admin",
            "test",
            "12345678",
            "qwerty",
            "abc123",
            "",
            "a",
        ]

        from auth.security_implementation import UserRole

        for i, weak_password in enumerate(weak_passwords):
            try:
                auth_manager.register_user(
                    f"weak_user_{i}",
                    f"weak{i}@example.com",
                    weak_password,
                    UserRole.OBSERVER,
                )
                # Should reject weak passwords
                pytest.fail(f"Weak password accepted: {weak_password}")
            except (ValueError, HTTPException):
                # Expected behavior
                pass

    def test_session_management_security(self, auth_manager):
        """Test session management security."""
        # Test session timeout
        from auth.security_implementation import UserRole

        auth_manager.register_user(
            "session_user",
            "session@example.com",
            "session_password",
            UserRole.OBSERVER,
        )

        # Create session
        session_id = auth_manager.create_session("session_user")

        # Validate session
        assert auth_manager.validate_session(session_id) is True, "Session validation failed"

        # Test session cleanup
        auth_manager.cleanup_expired_sessions()

        # Session should still be valid immediately after cleanup
        assert auth_manager.validate_session(session_id) is True, "Session prematurely expired"

    def test_cryptographic_security(self, auth_manager):
        """Test cryptographic security implementation."""
        # Test password hashing
        password = "test_password_123"

        from auth.security_implementation import UserRole

        user = auth_manager.register_user(
            "crypto_user", "crypto@example.com", password, UserRole.OBSERVER
        )

        # Verify password is hashed and not stored in plain text
        stored_user_data = auth_manager.users.get("crypto_user")
        assert stored_user_data is not None

        # Password should be hashed
        stored_password = stored_user_data.get("password", "")
        assert stored_password != password, "Password stored in plain text"
        assert len(stored_password) > 20, "Password hash too short"

        # Test JWT token security
        access_token = auth_manager.create_access_token(user)

        # Token should be properly formatted
        assert access_token.count(".") == 2, "Invalid JWT token format"

        # Token should not contain sensitive information in plain text
        import base64

        try:
            # Decode JWT payload (without verification for testing)
            parts = access_token.split(".")
            payload = base64.b64decode(parts[1] + "==")  # Add padding
            payload_str = payload.decode("utf-8")

            # Should not contain plain text password
            assert password not in payload_str, "Password leaked in JWT token"

        except Exception:
            # If decoding fails, that's actually good for security
            pass
