"""
Comprehensive tests for security headers and SSL/TLS configuration.

Tests all security headers for OWASP compliance and SSL Labs A+ rating requirements.
"""

import ssl
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from auth.security_headers import (
    SecurityHeadersManager,
    SecurityHeadersMiddleware,
    SecurityPolicy,
    generate_csp_nonce,
    validate_csp_header,
    validate_hsts_header,
)
from auth.ssl_tls_config import (
    SSLContextBuilder,
    TLSConfiguration,
    create_production_ssl_context,
    validate_ssl_configuration,
)


class TestSecurityHeaders:
    """Test security headers implementation."""

    def test_hsts_header_production(self):
        """Test HSTS header generation for production."""
        policy = SecurityPolicy(
            enable_hsts=True,
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            hsts_preload=True,
            production_mode=True,
        )
        manager = SecurityHeadersManager(policy)

        hsts = manager.generate_hsts_header()
        assert hsts == "max-age=31536000; includeSubDomains; preload"

        # Validate header format
        assert validate_hsts_header(hsts) is True

    def test_hsts_header_development(self):
        """Test HSTS header generation for development."""
        policy = SecurityPolicy(
            enable_hsts=True,
            hsts_max_age=300,  # 5 minutes for dev
            hsts_include_subdomains=False,
            hsts_preload=False,
            production_mode=False,
        )
        manager = SecurityHeadersManager(policy)

        hsts = manager.generate_hsts_header()
        assert hsts == "max-age=300"

    def test_csp_header_with_nonce(self):
        """Test CSP header generation with nonce."""
        manager = SecurityHeadersManager()
        nonce = manager.generate_nonce()

        csp = manager.generate_csp_header(nonce)

        # Check required directives
        assert "default-src 'self'" in csp
        assert f"script-src 'self' 'nonce-{nonce}'" in csp
        assert f"style-src 'self' 'nonce-{nonce}'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "object-src 'none'" in csp
        assert "base-uri 'self'" in csp
        assert "form-action 'self'" in csp
        assert "upgrade-insecure-requests" in csp
        assert "block-all-mixed-content" in csp

        # Validate CSP format
        assert validate_csp_header(csp) is True

    def test_csp_header_without_nonce(self):
        """Test CSP header generation without nonce."""
        manager = SecurityHeadersManager()
        csp = manager.generate_csp_header()

        assert "script-src 'self' 'strict-dynamic'" in csp
        assert "'nonce-" not in csp

    def test_permissions_policy_header(self):
        """Test Permissions Policy header generation."""
        manager = SecurityHeadersManager()
        permissions = manager.generate_permissions_policy()

        # Check that dangerous features are disabled
        dangerous_features = [
            "camera=()",
            "microphone=()",
            "geolocation=()",
            "payment=()",
            "usb=()",
            "battery=()",
            "ambient-light-sensor=()",
        ]

        for feature in dangerous_features:
            assert feature in permissions

    def test_expect_ct_header(self):
        """Test Expect-CT header generation."""
        policy = SecurityPolicy(
            enable_expect_ct=True,
            expect_ct_max_age=86400,
            expect_ct_enforce=True,
            expect_ct_report_uri="/api/security/ct-report",
        )
        manager = SecurityHeadersManager(policy)

        expect_ct = manager.generate_expect_ct_header()
        assert "max-age=86400" in expect_ct
        assert "enforce" in expect_ct
        assert 'report-uri="/api/security/ct-report"' in expect_ct

    def test_cache_control_headers(self):
        """Test cache control for different endpoint types."""
        manager = SecurityHeadersManager()

        # Test sensitive endpoint
        request = MagicMock()
        request.url.scheme = "https"
        request.url.path = "/api/auth/login"
        request.headers = {"host": "example.com"}

        response = MagicMock()
        response.headers = {"content-type": "application/json"}

        headers = manager.get_security_headers(request, response)

        assert (
            headers["Cache-Control"]
            == "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0"
        )
        assert headers["Pragma"] == "no-cache"
        assert headers["Expires"] == "0"

    def test_static_asset_cache_control(self):
        """Test cache control for static assets."""
        manager = SecurityHeadersManager()

        request = MagicMock()
        request.url.scheme = "https"
        request.url.path = "/static/css/style.css"
        request.headers = {"host": "example.com"}

        response = MagicMock()
        response.headers = {"content-type": "text/css"}

        headers = manager.get_security_headers(request, response)

        assert headers["Cache-Control"] == "public, max-age=31536000, immutable"

    def test_additional_security_headers(self):
        """Test additional security headers."""
        manager = SecurityHeadersManager()

        request = MagicMock()
        request.url.scheme = "https"
        request.url.path = "/"
        request.headers = {"host": "example.com"}

        response = MagicMock()
        response.headers = {"content-type": "text/html"}

        headers = manager.get_security_headers(request, response)

        # Check all required headers
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert headers["X-Permitted-Cross-Domain-Policies"] == "none"
        assert headers["X-DNS-Prefetch-Control"] == "off"
        assert headers["X-Download-Options"] == "noopen"

    def test_logout_clear_site_data(self):
        """Test Clear-Site-Data header on logout."""
        manager = SecurityHeadersManager()

        request = MagicMock()
        request.url.scheme = "https"
        request.url.path = "/auth/logout"
        request.headers = {"host": "example.com"}

        response = MagicMock()
        response.headers = {"content-type": "application/json"}

        headers = manager.get_security_headers(request, response)

        assert headers["Clear-Site-Data"] == '"cache"'

    def test_nonce_generation(self):
        """Test CSP nonce generation."""
        manager = SecurityHeadersManager()

        nonce1 = manager.generate_nonce()
        nonce2 = manager.generate_nonce()

        # Nonces should be unique
        assert nonce1 != nonce2

        # Nonces should be base64 encoded
        import base64

        try:
            decoded = base64.b64decode(nonce1)
            assert len(decoded) == 16  # 16 bytes
        except Exception:
            pytest.fail("Nonce is not valid base64")

    @pytest.mark.asyncio
    async def test_security_headers_middleware(self):
        """Test security headers middleware."""
        from fastapi import FastAPI

        app = FastAPI()
        manager = SecurityHeadersManager()
        app.add_middleware(SecurityHeadersMiddleware, security_manager=manager)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        with TestClient(app) as client:
            response = client.get("/test")

            # Check that security headers are present
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers
            assert "Content-Security-Policy" in response.headers
            assert "Referrer-Policy" in response.headers

    def test_certificate_pinning_header(self):
        """Test certificate pinning header generation."""
        policy = SecurityPolicy(enable_certificate_pinning=True, production_mode=True)
        manager = SecurityHeadersManager(policy)

        # Mock the certificate pinner
        with patch.object(
            manager.certificate_pinner, "generate_header"
        ) as mock_generate:
            mock_generate.return_value = 'pin-sha256="test123"; max-age=5184000'

            request = MagicMock()
            request.url.scheme = "https"
            request.url.path = "/"
            request.headers = {
                "host": "example.com",
                "user-agent": "FreeAgentics-iOS/1.0",
            }

            response = MagicMock()
            response.headers = {"content-type": "text/html"}

            manager.get_security_headers(request, response)

            # Public-Key-Pins header should be present for mobile apps
            assert mock_generate.called


class TestSSLTLSConfiguration:
    """Test SSL/TLS configuration."""

    def test_tls_configuration_defaults(self):
        """Test TLS configuration default values."""
        config = TLSConfiguration()

        assert config.min_tls_version == ssl.TLSVersion.TLSv1_2
        assert config.preferred_tls_version == ssl.TLSVersion.TLSv1_3
        assert config.enable_ocsp_stapling is True
        assert config.session_timeout == 86400
        assert config.dhparam_size == 4096

    def test_ssl_context_creation(self):
        """Test SSL context creation with secure settings."""
        config = TLSConfiguration()
        builder = SSLContextBuilder(config)

        # Note: Some SSL operations require actual certificates
        # This is a basic test of context creation
        with patch("ssl.create_default_context") as mock_create:
            mock_context = MagicMock()
            mock_create.return_value = mock_context

            builder.create_server_context()

            # Verify secure options are set
            assert mock_context.minimum_version == ssl.TLSVersion.TLSv1_2
            assert mock_context.options.__or__.called

    def test_cipher_string_building(self):
        """Test cipher string building."""
        config = TLSConfiguration()
        builder = SSLContextBuilder(config)

        cipher_string = builder._build_cipher_string()

        # Check that only strong ciphers are included
        strong_ciphers = [
            "ECDHE-ECDSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-ECDSA-CHACHA20-POLY1305",
            "ECDHE-RSA-CHACHA20-POLY1305",
        ]

        for cipher in strong_ciphers:
            assert cipher in cipher_string

    def test_ssl_validation(self):
        """Test SSL configuration validation."""
        # Create a mock SSL context
        mock_context = MagicMock()
        mock_context.options = ssl.OP_NO_COMPRESSION | ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        mock_context.cert_store_stats.return_value = {
            "x509": 1,
            "crl": 0,
            "ca": 1,
        }

        validation_results = validate_ssl_configuration(mock_context)

        assert validation_results["tls_1_2_enabled"] is True
        assert validation_results["compression_disabled"] is True
        assert validation_results["certificate_loaded"] is True

    def test_production_ssl_context(self):
        """Test production SSL context creation."""
        with patch.dict("os.environ", {"PRODUCTION": "true"}):
            with patch("ssl.create_default_context") as mock_create:
                mock_context = MagicMock()
                mock_create.return_value = mock_context

                create_production_ssl_context()

                assert mock_create.called
                assert mock_context.minimum_version == ssl.TLSVersion.TLSv1_2


class TestSecurityHeadersIntegration:
    """Integration tests for security headers."""

    @pytest.mark.asyncio
    async def test_full_security_stack(self):
        """Test full security headers stack with FastAPI."""
        from fastapi import FastAPI

        app = FastAPI()

        # Production security policy
        policy = SecurityPolicy(
            enable_hsts=True,
            hsts_preload=True,
            enable_expect_ct=True,
            enable_certificate_pinning=True,
            production_mode=True,
        )

        manager = SecurityHeadersManager(policy)
        app.add_middleware(SecurityHeadersMiddleware, security_manager=manager)

        @app.get("/")
        async def root():
            return {"message": "secure"}

        @app.get("/api/data")
        async def api_data():
            return {"data": "sensitive"}

        @app.get("/static/style.css")
        async def static_file():
            return "body { margin: 0; }"

        with TestClient(app) as client:
            # Test root endpoint
            response = client.get("/", headers={"X-Forwarded-Proto": "https"})
            assert response.status_code == 200
            assert "Strict-Transport-Security" in response.headers
            assert "preload" in response.headers["Strict-Transport-Security"]

            # Test API endpoint
            response = client.get("/api/data")
            assert (
                response.headers["Cache-Control"]
                == "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0"
            )
            assert response.headers["X-Frame-Options"] == "DENY"

            # Test static file
            response = client.get("/static/style.css")
            assert "max-age=31536000" in response.headers["Cache-Control"]

    def test_security_headers_error_handling(self):
        """Test security headers are applied even during errors."""
        from fastapi import FastAPI

        app = FastAPI()
        manager = SecurityHeadersManager()
        app.add_middleware(SecurityHeadersMiddleware, security_manager=manager)

        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test error")

        with TestClient(app) as client:
            response = client.get("/error")

            # Even error responses should have security headers
            assert response.status_code == 500
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers


class TestSecurityHeadersCompatibility:
    """Test backward compatibility functions."""

    def test_get_security_headers_function(self):
        """Test standalone get_security_headers function."""
        from auth.security_headers import get_security_headers

        headers = get_security_headers()

        assert "Content-Security-Policy" in headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers

    def test_add_security_headers_function(self):
        """Test add_security_headers function."""
        from auth.security_headers import add_security_headers

        response = MagicMock()
        response.headers = {}

        add_security_headers(response)

        assert "X-Frame-Options" in response.headers
        assert "Content-Security-Policy" in response.headers

    def test_generate_csp_nonce_function(self):
        """Test standalone generate_csp_nonce function."""
        nonce = generate_csp_nonce()

        assert isinstance(nonce, str)
        assert len(nonce) > 0
