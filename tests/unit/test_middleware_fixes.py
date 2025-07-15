"""
Unit tests for middleware fixes - Task #14.5
"""

import os
from unittest.mock import patch

from api.middleware.security_monitoring import SecurityHeadersMiddleware
from auth.security_implementation import SecurityMiddleware


class TestSecurityMiddlewareFixes:
    """Test fixes to security middleware."""

    def test_security_middleware_typo_fixed(self):
        """Test that the typo in SecurityMiddleware is fixed."""
        # Create a mock app
        app = lambda scope, receive, send: None
        SecurityMiddleware(app)

        # Test that when we get the headers, the typo is fixed
        # Check the actual header value in the source code
        import inspect

        source = inspect.getsource(SecurityMiddleware)

        # Verify the correct header value is present
        assert 'b"nosniff"' in source

        # Verify the typo is not in the actual header definition
        # Look for the pattern: (b"X-Content-Type-Options", b"nosnif")
        assert '(b"X-Content-Type-Options", b"nosnif")' not in source

    def test_enhanced_security_headers_middleware_uses_manager(self):
        """Test that SecurityHeadersMiddleware uses the unified manager."""
        app = lambda scope, receive, send: None
        middleware = SecurityHeadersMiddleware(app)

        # Verify it has our security manager
        assert hasattr(middleware, "security_manager")
        assert middleware.security_manager is not None

        # Verify it's using our SecurityHeadersManager
        from auth.security_headers import SecurityHeadersManager

        assert isinstance(middleware.security_manager, SecurityHeadersManager)


class TestUnifiedSecurityApproach:
    """Test that the unified security approach works correctly."""

    def test_comprehensive_headers_coverage(self):
        """Test that our unified approach provides comprehensive headers."""
        from auth.security_headers import SecurityHeadersManager

        manager = SecurityHeadersManager()

        # Mock request and response
        class MockRequest:
            url = type("obj", (object,), {"scheme": "https", "path": "/test"})()
            headers = {"host": "example.com"}

        class MockResponse:
            headers = {"content-type": "text/html"}

        headers = manager.get_security_headers(MockRequest(), MockResponse())

        # Verify we have all the important security headers
        expected_headers = [
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Permissions-Policy",
            "Expect-CT",
        ]

        for header in expected_headers:
            assert header in headers, f"Missing security header: {header}"

    def test_auth_endpoint_enhanced_security(self):
        """Test enhanced security for authentication endpoints."""
        from auth.security_headers import SecurityHeadersManager

        manager = SecurityHeadersManager()

        # Mock auth endpoint request
        class MockRequest:
            url = type("obj", (object,), {"scheme": "https", "path": "/api/auth/login"})()
            headers = {"host": "example.com"}

        class MockResponse:
            headers = {"content-type": "application/json"}

        headers = manager.get_security_headers(MockRequest(), MockResponse())

        # Auth endpoints should have additional security
        assert "Cache-Control" in headers
        assert "no-store" in headers["Cache-Control"]
        assert "Pragma" in headers
        assert "Expires" in headers

    def test_production_vs_development_behavior(self):
        """Test different behavior in production vs development."""
        from auth.security_headers import SecurityHeadersManager

        # Test development mode
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            dev_manager = SecurityHeadersManager()
            dev_config = dev_manager.get_secure_cookie_config()
            assert dev_config["secure"] is False  # HTTP allowed in dev

        # Test production mode
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            prod_manager = SecurityHeadersManager()
            prod_config = prod_manager.get_secure_cookie_config()
            assert prod_config["secure"] is True  # HTTPS required in prod
            assert prod_manager.policy.hsts_preload is True

    def test_environment_customization_support(self):
        """Test that environment variables can customize security headers."""
        from auth.security_headers import SecurityHeadersManager

        with patch.dict(
            os.environ,
            {
                "HSTS_MAX_AGE": "63072000",
                "CSP_SCRIPT_SRC": "'self' 'unsafe-inline' https://cdn.example.com",
                "CSP_REPORT_URI": "/csp-violations",
            },
        ):
            manager = SecurityHeadersManager()

            # Test HSTS customization
            hsts = manager.generate_hsts_header()
            assert "max-age=63072000" in hsts

            # Test CSP customization
            csp = manager.generate_csp_header()
            assert "https://cdn.example.com" in csp
            assert "report-uri /csp-violations" in csp
