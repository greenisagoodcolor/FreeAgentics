"""
Authentication Security Header Validation Tests
Task #6.5 - Implement security header validation

This test suite validates security headers for authentication endpoints:
1. HSTS (HTTP Strict Transport Security) headers
2. CSP (Content Security Policy) headers
3. X-Frame-Options for clickjacking protection
4. X-Content-Type-Options for MIME sniffing protection
5. Referrer-Policy for privacy
6. CORS headers for cross-origin requests
7. Certificate pinning headers
8. Security headers under different scenarios
"""

import re
from typing import Dict, List
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from auth.security_headers import SecurityHeadersManager
from auth.security_implementation import AuthenticationManager


class MockAuthAPI:
    """Mock authentication API for testing."""

    def __init__(self):
        self.app = FastAPI()
        self.auth_manager = AuthenticationManager()
        self.security_headers_manager = SecurityHeadersManager()

        # Setup routes
        @self.app.post("/auth/login")
        async def login(request: Request, response: Response):
            # Apply security headers
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"access_token": "mock_token", "token_type": "bearer"}

        @self.app.post("/auth/register")
        async def register(request: Request, response: Response):
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"message": "User registered successfully"}

        @self.app.post("/auth/logout")
        async def logout(request: Request, response: Response):
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"message": "Logged out successfully"}

        @self.app.post("/auth/refresh")
        async def refresh(request: Request, response: Response):
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"access_token": "new_mock_token", "token_type": "bearer"}

        @self.app.post("/auth/password-reset")
        async def password_reset(request: Request, response: Response):
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"message": "Password reset email sent"}

        @self.app.get("/auth/verify")
        async def verify_token(request: Request, response: Response):
            headers = self.security_headers_manager.get_security_headers(request, response)
            for key, value in headers.items():
                response.headers[key] = value
            return {"valid": True, "user_id": "test-user-123"}

        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            # Additional security headers can be added here
            return response


class TestAuthenticationSecurityHeaders:
    """Test security headers for authentication endpoints."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_api = MockAuthAPI()
        self.client = TestClient(self.mock_api.app)
        self.required_headers = {
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
            "Permissions-Policy",
        }
        self.production_headers = {
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Expect-CT",
        }

    def _validate_common_headers(self, headers: Dict[str, str], endpoint: str = None) -> List[str]:
        """Validate common security headers."""
        issues = []

        # Check required headers
        for header in self.required_headers:
            if header not in headers:
                issues.append(f"Missing required header: {header}")

        # Validate header values
        if "X-Content-Type-Options" in headers:
            if headers["X-Content-Type-Options"] != "nosniff":
                issues.append("X-Content-Type-Options should be 'nosniff'")

        if "X-Frame-Options" in headers:
            if headers["X-Frame-Options"] not in ["DENY", "SAMEORIGIN"]:
                issues.append("X-Frame-Options should be 'DENY' or 'SAMEORIGIN'")

        if "Referrer-Policy" in headers:
            valid_policies = [
                "no-referrer",
                "no-referrer-when-downgrade",
                "strict-origin",
                "strict-origin-when-cross-origin",
            ]
            if headers["Referrer-Policy"] not in valid_policies:
                issues.append(f"Invalid Referrer-Policy: {headers['Referrer-Policy']}")

        return issues

    def test_login_endpoint_headers(self):
        """Test security headers for login endpoint."""
        response = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "testpass"},
        )

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "login")
        assert len(issues) == 0, f"Header validation issues: {issues}"

        # Login endpoint specific checks
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_register_endpoint_headers(self):
        """Test security headers for registration endpoint."""
        response = self.client.post(
            "/auth/register",
            json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "securepass123",
            },
        )

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "register")
        assert len(issues) == 0, f"Header validation issues: {issues}"

    def test_logout_endpoint_headers(self):
        """Test security headers for logout endpoint."""
        response = self.client.post("/auth/logout", headers={"Authorization": "Bearer mock_token"})

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "logout")
        assert len(issues) == 0, f"Header validation issues: {issues}"

    def test_token_refresh_headers(self):
        """Test security headers for token refresh endpoint."""
        response = self.client.post("/auth/refresh", json={"refresh_token": "mock_refresh_token"})

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "refresh")
        assert len(issues) == 0, f"Header validation issues: {issues}"

    def test_password_reset_headers(self):
        """Test security headers for password reset endpoint."""
        response = self.client.post("/auth/password-reset", json={"email": "user@example.com"})

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "password-reset")
        assert len(issues) == 0, f"Header validation issues: {issues}"

    def test_token_verification_headers(self):
        """Test security headers for token verification endpoint."""
        response = self.client.get("/auth/verify", headers={"Authorization": "Bearer mock_token"})

        assert response.status_code == 200

        # Validate headers
        issues = self._validate_common_headers(response.headers, "verify")
        assert len(issues) == 0, f"Header validation issues: {issues}"

    def test_hsts_header_production(self):
        """Test HSTS header in production mode."""
        with patch.dict("os.environ", {"PRODUCTION": "true"}):
            # Recreate client with production settings
            self.mock_api.security_headers_manager = SecurityHeadersManager()
            response = self.client.post(
                "/auth/login",
                json={"username": "testuser", "password": "testpass"},
            )

            assert "Strict-Transport-Security" in response.headers
            hsts_header = response.headers["Strict-Transport-Security"]

            # Validate HSTS directives
            assert "max-age=" in hsts_header
            max_age_match = re.search(r"max-age=(\d+)", hsts_header)
            if max_age_match:
                max_age = int(max_age_match.group(1))
                assert max_age >= 31536000  # At least 1 year

            assert "includeSubDomains" in hsts_header
            # Production should have preload
            assert "preload" in hsts_header

    def test_csp_header_configuration(self):
        """Test Content Security Policy header configuration."""
        with patch.dict("os.environ", {"PRODUCTION": "true"}):
            self.mock_api.security_headers_manager = SecurityHeadersManager()
            response = self.client.post(
                "/auth/login",
                json={"username": "testuser", "password": "testpass"},
            )

            if "Content-Security-Policy" in response.headers:
                csp = response.headers["Content-Security-Policy"]

                # Validate CSP directives
                assert "default-src" in csp
                assert "script-src" in csp
                assert "style-src" in csp
                assert "img-src" in csp
                assert "connect-src" in csp

                # Should not allow unsafe inline scripts
                assert "'unsafe-inline'" not in csp or "nonce-" in csp

    def test_cors_headers_for_auth(self):
        """Test CORS headers for authentication endpoints."""
        # Test preflight request
        response = self.client.options(
            "/auth/login",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Check CORS headers if present
        if "Access-Control-Allow-Origin" in response.headers:
            origin = response.headers["Access-Control-Allow-Origin"]
            # Should not be wildcard for auth endpoints
            assert origin != "*"

            # Should have credentials support
            if "Access-Control-Allow-Credentials" in response.headers:
                assert response.headers["Access-Control-Allow-Credentials"] == "true"

    def test_cache_control_headers(self):
        """Test cache control headers for sensitive endpoints."""
        sensitive_endpoints = [
            ("/auth/login", "POST"),
            ("/auth/verify", "GET"),
            ("/auth/refresh", "POST"),
        ]

        for endpoint, method in sensitive_endpoints:
            if method == "POST":
                response = self.client.post(endpoint, json={})
            else:
                response = self.client.get(endpoint, headers={"Authorization": "Bearer mock_token"})

            # Sensitive endpoints should not be cached
            if "Cache-Control" in response.headers:
                cache_control = response.headers["Cache-Control"]
                assert any(
                    directive in cache_control
                    for directive in [
                        "no-store",
                        "no-cache",
                        "private",
                        "max-age=0",
                    ]
                )

    def test_permissions_policy_header(self):
        """Test Permissions-Policy (formerly Feature-Policy) header."""
        response = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "testpass"},
        )

        if "Permissions-Policy" in response.headers:
            policy = response.headers["Permissions-Policy"]

            # Should restrict dangerous features
            restricted_features = [
                "geolocation",
                "microphone",
                "camera",
                "payment",
                "usb",
                "magnetometer",
            ]

            for feature in restricted_features:
                # Feature should be restricted (either not mentioned or set to 'none' or 'self')
                if feature in policy:
                    assert f"{feature}=()" in policy or f"{feature}=(self)" in policy

    def test_security_headers_consistency(self):
        """Test that security headers are consistent across all auth endpoints."""
        endpoints = [
            ("/auth/login", "POST", {"username": "test", "password": "test"}),
            (
                "/auth/register",
                "POST",
                {
                    "username": "new",
                    "email": "new@test.com",
                    "password": "test",
                },
            ),
            ("/auth/logout", "POST", {}),
            ("/auth/refresh", "POST", {"refresh_token": "token"}),
            ("/auth/password-reset", "POST", {"email": "test@test.com"}),
        ]

        all_headers = []

        for endpoint, method, data in endpoints:
            response = self.client.request(method, endpoint, json=data)
            all_headers.append({"endpoint": endpoint, "headers": dict(response.headers)})

        # Check consistency
        common_headers = None
        for entry in all_headers:
            endpoint_headers = set(
                k
                for k in entry["headers"].keys()
                if k.startswith("X-") or k in ["Referrer-Policy", "Permissions-Policy"]
            )

            if common_headers is None:
                common_headers = endpoint_headers
            else:
                # All endpoints should have the same security headers
                missing = common_headers - endpoint_headers
                endpoint_headers - common_headers

                assert len(missing) == 0, f"{entry['endpoint']} missing headers: {missing}"

    def test_certificate_pinning_headers(self):
        """Test certificate pinning headers in production."""
        with patch.dict("os.environ", {"PRODUCTION": "true", "ENABLE_CERT_PINNING": "true"}):
            self.mock_api.security_headers_manager = SecurityHeadersManager()
            response = self.client.post(
                "/auth/login",
                json={"username": "testuser", "password": "testpass"},
            )

            # Check for Public-Key-Pins or Expect-CT header
            if "Public-Key-Pins" in response.headers:
                pkp = response.headers["Public-Key-Pins"]
                assert "pin-sha256=" in pkp
                assert "max-age=" in pkp

            if "Expect-CT" in response.headers:
                expect_ct = response.headers["Expect-CT"]
                assert "max-age=" in expect_ct
                # In production, should enforce
                if "enforce" in expect_ct:
                    assert "report-uri=" in expect_ct

    def test_security_headers_error_responses(self):
        """Test that security headers are present even in error responses."""
        # Test 404
        response = self.client.post("/auth/nonexistent")
        if response.status_code == 404:
            self._validate_common_headers(response.headers, "error-404")
            # Some headers might be missing in error responses, but critical ones should be there
            critical_headers = {"X-Content-Type-Options", "X-Frame-Options"}
            for header in critical_headers:
                assert header in response.headers

        # Test 401 (unauthorized)
        response = self.client.get("/auth/verify")  # No auth header
        if response.status_code == 401:
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers

    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks."""
        # Try to inject headers through user input
        malicious_inputs = [
            "test\r\nX-Injected: malicious",
            "test\nX-Injected: malicious",
            "test\rX-Injected: malicious",
        ]

        for malicious_input in malicious_inputs:
            response = self.client.post(
                "/auth/login",
                json={"username": malicious_input, "password": "test"},
            )

            # Injected header should not appear
            assert "X-Injected" not in response.headers

    def test_security_headers_performance(self):
        """Test that adding security headers doesn't significantly impact performance."""
        import time

        # Warm up
        self.client.post("/auth/login", json={"username": "test", "password": "test"})

        # Measure with headers
        times_with_headers = []
        for _ in range(100):
            start = time.time()
            _ = self.client.post("/auth/login", json={"username": "test", "password": "test"})
            times_with_headers.append(time.time() - start)

        avg_time = sum(times_with_headers) / len(times_with_headers)

        # Header processing should be fast
        assert avg_time < 0.01  # Less than 10ms average

    @pytest.mark.parametrize("production", [True, False])
    def test_environment_specific_headers(self, production):
        """Test headers change appropriately between development and production."""
        with patch.dict("os.environ", {"PRODUCTION": str(production).lower()}):
            self.mock_api.security_headers_manager = SecurityHeadersManager()
            response = self.client.post(
                "/auth/login",
                json={"username": "testuser", "password": "testpass"},
            )

            if production:
                # Production should have strict security headers
                assert "Strict-Transport-Security" in response.headers
                if "Content-Security-Policy" in response.headers:
                    csp = response.headers["Content-Security-Policy"]
                    assert "upgrade-insecure-requests" in csp
            else:
                # Development might have more relaxed headers
                hsts = response.headers.get("Strict-Transport-Security", "")
                if hsts:
                    # Development HSTS should not have preload
                    assert "preload" not in hsts


class TestSecurityHeadersIntegration:
    """Integration tests for security headers with authentication flow."""

    def setup_method(self):
        """Setup for integration tests."""
        self.mock_api = MockAuthAPI()
        self.client = TestClient(self.mock_api.app)
        self.auth_manager = self.mock_api.auth_manager

    def test_complete_auth_flow_with_headers(self):
        """Test complete authentication flow maintains security headers."""
        # 1. Register
        register_response = self.client.post(
            "/auth/register",
            json={
                "username": "secureuser",
                "email": "secure@example.com",
                "password": "SecurePass123!",
            },
        )
        assert register_response.status_code == 200
        assert "X-Content-Type-Options" in register_response.headers

        # 2. Login
        login_response = self.client.post(
            "/auth/login",
            json={"username": "secureuser", "password": "SecurePass123!"},
        )
        assert login_response.status_code == 200
        assert "X-Frame-Options" in login_response.headers

        # 3. Verify token
        verify_response = self.client.get(
            "/auth/verify",
            headers={"Authorization": f"Bearer {login_response.json()['access_token']}"},
        )
        assert verify_response.status_code == 200
        assert "Referrer-Policy" in verify_response.headers

        # 4. Refresh token
        refresh_response = self.client.post(
            "/auth/refresh", json={"refresh_token": "mock_refresh_token"}
        )
        assert refresh_response.status_code == 200
        assert "X-Content-Type-Options" in refresh_response.headers

        # 5. Logout
        logout_response = self.client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {login_response.json()['access_token']}"},
        )
        assert logout_response.status_code == 200
        assert "X-Frame-Options" in logout_response.headers

    def test_security_headers_with_concurrent_requests(self):
        """Test security headers remain consistent under concurrent load."""
        import concurrent.futures
        import threading

        results = []
        results_lock = threading.Lock()

        def make_request(endpoint, method="POST"):
            """Make a request and check headers."""
            if method == "POST":
                response = self.client.post(endpoint, json={"username": "test", "password": "test"})
            else:
                response = self.client.get(endpoint)

            with results_lock:
                results.append(
                    {
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "has_security_headers": all(
                            header in response.headers
                            for header in [
                                "X-Content-Type-Options",
                                "X-Frame-Options",
                            ]
                        ),
                    }
                )

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(50):
                futures.append(executor.submit(make_request, "/auth/login"))
                futures.append(executor.submit(make_request, "/auth/verify", "GET"))

            concurrent.futures.wait(futures)

        # All requests should have security headers
        assert all(r["has_security_headers"] for r in results)
