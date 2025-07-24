"""
Unit tests for security headers implementation.

Tests for Task #14.5 - Security Headers and SSL/TLS Configuration
"""

import os
from unittest.mock import Mock, patch

from auth.security_headers import (
    PRODUCTION_SECURITY_POLICY,
    CertificatePinner,
    SecurityHeadersManager,
    SecurityPolicy,
)


class TestSecurityPolicy:
    """Test SecurityPolicy configuration."""

    def test_default_policy(self):
        """Test default security policy configuration."""
        policy = SecurityPolicy()

        assert policy.enable_hsts is True
        assert policy.hsts_max_age == 31536000
        assert policy.hsts_include_subdomains is True
        assert policy.hsts_preload is True  # Now enabled by default
        assert policy.x_frame_options == "DENY"
        assert policy.x_content_type_options == "nosniff"
        assert policy.referrer_policy == "strict-origin-when-cross-origin"

    def test_production_policy(self):
        """Test production security policy configuration."""
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            policy = SecurityPolicy()

            assert policy.production_mode is True
            assert policy.hsts_preload is True
            assert policy.enable_certificate_pinning is True
            assert policy.expect_ct_enforce is True

    def test_development_policy(self):
        """Test development security policy configuration."""
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            policy = SecurityPolicy()

            assert policy.production_mode is False
            assert policy.secure_cookies is False  # HTTP allowed in dev


class TestCertificatePinner:
    """Test certificate pinning functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.pinner = CertificatePinner()

    def test_add_pin(self):
        """Test adding certificate pins."""
        self.pinner.add_pin(
            "example.com",
            "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        )

        assert "example.com" in self.pinner.pins
        assert (
            "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" in self.pinner.pins["example.com"]
        )

    def test_add_backup_pin(self):
        """Test adding backup certificate pins."""
        self.pinner.add_backup_pin(
            "example.com",
            "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
        )

        assert "example.com" in self.pinner.backup_pins
        assert (
            "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="
            in self.pinner.backup_pins["example.com"]
        )

    def test_generate_header_with_pins(self):
        """Test generating Public-Key-Pins header."""
        self.pinner.add_pin(
            "example.com",
            "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        )
        self.pinner.add_backup_pin(
            "example.com",
            "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
        )

        header = self.pinner.generate_header("example.com")

        assert header is not None
        assert "pin-sha256=" in header
        assert "max-age=5184000" in header
        assert "includeSubDomains" in header

    def test_generate_header_no_pins(self):
        """Test generating header when no pins are configured."""
        header = self.pinner.generate_header("nonexistent.com")

        assert header is None

    def test_load_pins_from_env(self):
        """Test loading pins from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CERT_PINS": "example.com:sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "CERT_BACKUP_PINS": "example.com:sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
            },
        ):
            pinner = CertificatePinner()
            pinner.load_pins_from_env()

            assert "example.com" in pinner.pins
            assert "example.com" in pinner.backup_pins


class TestSecurityHeadersManager:
    """Test SecurityHeadersManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = SecurityHeadersManager()

    def test_initialization(self):
        """Test SecurityHeadersManager initialization."""
        assert self.manager.policy is not None
        assert self.manager.certificate_pinner is not None

    def test_generate_hsts_header(self):
        """Test HSTS header generation."""
        hsts = self.manager.generate_hsts_header()

        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts

    def test_generate_hsts_header_with_preload(self):
        """Test HSTS header generation with preload."""
        self.manager.policy.hsts_preload = True
        hsts = self.manager.generate_hsts_header()

        assert "preload" in hsts

    def test_generate_csp_header_default(self):
        """Test default CSP header generation."""
        csp = self.manager.generate_csp_header()

        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "img-src 'self' data: https:" in csp
        assert "font-src 'self' data:" in csp
        assert "connect-src 'self' wss: https:" in csp
        assert "frame-ancestors 'none'" in csp  # Updated to stricter 'none'
        assert "base-uri 'self'" in csp
        assert "form-action 'self'" in csp

    def test_generate_csp_header_with_nonce(self):
        """Test CSP header generation with nonce."""
        nonce = "test-nonce-12345"
        csp = self.manager.generate_csp_header(nonce=nonce)

        assert f"'nonce-{nonce}'" in csp

    def test_generate_csp_header_custom_policy(self):
        """Test CSP header generation with custom policy."""
        self.manager.policy.csp_policy = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        csp = self.manager.generate_csp_header()

        assert csp == "default-src 'self'; script-src 'self' 'unsafe-inline'"

    def test_generate_csp_header_with_report_uri(self):
        """Test CSP header generation with report URI."""
        self.manager.policy.csp_report_uri = "/csp-violations"
        csp = self.manager.generate_csp_header()

        assert "report-uri /csp-violations" in csp

    def test_generate_expect_ct_header(self):
        """Test Expect-CT header generation."""
        expect_ct = self.manager.generate_expect_ct_header()

        assert "max-age=86400" in expect_ct
        assert "enforce" in expect_ct

    def test_generate_permissions_policy(self):
        """Test Permissions-Policy header generation."""
        permissions = self.manager.generate_permissions_policy()

        assert "geolocation=()" in permissions
        assert "microphone=()" in permissions
        assert "camera=()" in permissions

    def test_generate_nonce(self):
        """Test nonce generation."""
        nonce = self.manager.generate_nonce()

        assert len(nonce) >= 16
        # Nonce is base64 encoded, so contains '=' and other chars
        import base64

        try:
            decoded = base64.b64decode(nonce)
            assert len(decoded) == 16
        except Exception as e:
            pytest.fail(f"Nonce should be valid base64: {e}")

    def test_get_secure_cookie_config_production(self):
        """Test secure cookie config in production."""
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            manager = SecurityHeadersManager()
            config = manager.get_secure_cookie_config()

            assert config["secure"] is True
            assert config["httponly"] is True
            assert config["samesite"] == "strict"

    def test_get_secure_cookie_config_development(self):
        """Test secure cookie config in development."""
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            manager = SecurityHeadersManager()
            config = manager.get_secure_cookie_config()

            assert config["secure"] is False  # HTTP allowed in dev
            assert config["httponly"] is True
            assert config["samesite"] == "strict"

    def test_is_websocket_path(self):
        """Test WebSocket path detection."""
        assert self.manager.is_websocket_path("/ws/test") is True
        assert self.manager.is_websocket_path("/websocket/connect") is True
        assert self.manager.is_websocket_path("/socket.io/connect") is True
        assert self.manager.is_websocket_path("/api/test") is False

    def test_get_security_headers(self):
        """Test getting all security headers."""
        # Mock request and response
        request = Mock()
        request.url.scheme = "https"
        request.url.path = "/test"
        request.headers = {"host": "example.com"}

        response = Mock()
        response.headers = {"content-type": "text/html"}
        response.context = {}  # Mock response.context properly

        headers = self.manager.get_security_headers(request, response)

        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Referrer-Policy" in headers
        assert "Permissions-Policy" in headers
        assert "Expect-CT" in headers

    def test_get_security_headers_auth_endpoint(self):
        """Test enhanced headers for auth endpoints."""
        request = Mock()
        request.url.scheme = "https"
        request.url.path = "/api/auth/login"
        request.headers = {"host": "example.com"}

        response = Mock()
        response.headers = {"content-type": "application/json"}
        response.context = {}  # Mock response.context properly

        headers = self.manager.get_security_headers(request, response)

        assert "Cache-Control" in headers
        assert "no-store" in headers["Cache-Control"]
        assert "Pragma" in headers
        assert "Expires" in headers

    def test_env_customizations(self):
        """Test environment variable customizations."""
        with patch.dict(
            os.environ,
            {
                "HSTS_MAX_AGE": "63072000",
                "CSP_SCRIPT_SRC": "'self' 'unsafe-inline' https://cdn.example.com",
                "CSP_REPORT_URI": "/csp-violations",
                "EXPECT_CT_MAX_AGE": "604800",
            },
        ):
            manager = SecurityHeadersManager()

            # Test HSTS customization
            assert manager.policy.hsts_max_age == 63072000

            # Test CSP customization
            csp = manager.generate_csp_header()
            assert "https://cdn.example.com" in csp
            assert "report-uri /csp-violations" in csp

            # Test Expect-CT customization
            assert manager.policy.expect_ct_max_age == 604800


class TestProductionSecurityPolicy:
    """Test production security policy."""

    def test_production_policy_config(self):
        """Test production policy configuration."""
        policy = PRODUCTION_SECURITY_POLICY

        assert policy.enable_hsts is True
        assert policy.hsts_max_age == 31536000
        assert policy.hsts_include_subdomains is True
        assert policy.hsts_preload is True
        assert policy.enable_expect_ct is True
        assert policy.expect_ct_enforce is True
        assert policy.enable_certificate_pinning is True
        assert policy.production_mode is True
