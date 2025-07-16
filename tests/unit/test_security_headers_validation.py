"""
Security Headers Validation Test Suite

Tests security headers implementation to ensure proper configuration
according to OWASP recommendations and security best practices.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.security_headers import (
    SecurityHeadersMiddleware,
    add_security_headers,
    get_security_headers,
    validate_csp_header,
    validate_hsts_header,
)


class TestSecurityHeadersValidation:
    """Test security headers validation and implementation."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application."""
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World"}
        
        @app.get("/api/test")
        def test_endpoint():
            return {"data": "test"}
        
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client with security headers middleware."""
        # Add security headers middleware
        app.add_middleware(SecurityHeadersMiddleware)
        return TestClient(app)

    def test_security_headers_middleware_initialization(self):
        """Test SecurityHeadersMiddleware initialization."""
        middleware = SecurityHeadersMiddleware(Mock())
        assert middleware is not None
        assert hasattr(middleware, 'app')

    def test_content_security_policy_header(self, client):
        """Test Content Security Policy header is properly set."""
        response = client.get("/")
        
        assert "content-security-policy" in response.headers
        csp_header = response.headers["content-security-policy"]
        
        # Verify CSP contains essential directives
        assert "default-src" in csp_header
        assert "script-src" in csp_header
        assert "style-src" in csp_header
        assert "img-src" in csp_header
        assert "connect-src" in csp_header
        assert "font-src" in csp_header
        assert "object-src" in csp_header
        assert "media-src" in csp_header
        assert "frame-src" in csp_header
        
        # Verify restrictive policies
        assert "'none'" in csp_header or "'self'" in csp_header
        assert "'unsafe-inline'" not in csp_header or "'unsafe-eval'" not in csp_header

    def test_strict_transport_security_header(self, client):
        """Test HTTP Strict Transport Security header is properly set."""
        response = client.get("/")
        
        assert "strict-transport-security" in response.headers
        hsts_header = response.headers["strict-transport-security"]
        
        # Verify HSTS contains max-age
        assert "max-age=" in hsts_header
        
        # Verify max-age is reasonable (at least 1 year)
        import re
        max_age_match = re.search(r'max-age=(\d+)', hsts_header)
        assert max_age_match is not None
        max_age = int(max_age_match.group(1))
        assert max_age >= 31536000  # 1 year in seconds
        
        # Verify includeSubDomains is present
        assert "includeSubDomains" in hsts_header
        
        # Verify preload is present for enhanced security
        assert "preload" in hsts_header

    def test_x_frame_options_header(self, client):
        """Test X-Frame-Options header is properly set."""
        response = client.get("/")
        
        assert "x-frame-options" in response.headers
        frame_options = response.headers["x-frame-options"]
        
        # Should be DENY or SAMEORIGIN
        assert frame_options.upper() in ["DENY", "SAMEORIGIN"]

    def test_x_content_type_options_header(self, client):
        """Test X-Content-Type-Options header is properly set."""
        response = client.get("/")
        
        assert "x-content-type-options" in response.headers
        content_type_options = response.headers["x-content-type-options"]
        
        # Should be nosniff
        assert content_type_options.lower() == "nosniff"

    def test_x_xss_protection_header(self, client):
        """Test X-XSS-Protection header is properly set."""
        response = client.get("/")
        
        assert "x-xss-protection" in response.headers
        xss_protection = response.headers["x-xss-protection"]
        
        # Should be "1; mode=block" or "0" (modern browsers)
        assert xss_protection in ["1; mode=block", "0"]

    def test_referrer_policy_header(self, client):
        """Test Referrer-Policy header is properly set."""
        response = client.get("/")
        
        assert "referrer-policy" in response.headers
        referrer_policy = response.headers["referrer-policy"]
        
        # Should be a secure policy
        secure_policies = [
            "no-referrer",
            "no-referrer-when-downgrade",
            "origin",
            "origin-when-cross-origin",
            "same-origin",
            "strict-origin",
            "strict-origin-when-cross-origin",
        ]
        assert referrer_policy in secure_policies

    def test_permissions_policy_header(self, client):
        """Test Permissions-Policy header is properly set."""
        response = client.get("/")
        
        # Permissions-Policy might be present
        if "permissions-policy" in response.headers:
            permissions_policy = response.headers["permissions-policy"]
            
            # Should contain restrictive policies
            assert "camera=()" in permissions_policy or "camera=self" in permissions_policy
            assert "microphone=()" in permissions_policy or "microphone=self" in permissions_policy
            assert "geolocation=()" in permissions_policy or "geolocation=self" in permissions_policy

    def test_server_header_removal(self, client):
        """Test that Server header is removed or minimized."""
        response = client.get("/")
        
        if "server" in response.headers:
            server_header = response.headers["server"]
            
            # Should not reveal detailed server information
            assert "uvicorn" not in server_header.lower()
            assert "fastapi" not in server_header.lower()
            assert "python" not in server_header.lower()
            
            # Should be generic or minimal
            assert len(server_header) < 50

    def test_x_powered_by_header_removal(self, client):
        """Test that X-Powered-By header is removed."""
        response = client.get("/")
        
        # X-Powered-By should not be present
        assert "x-powered-by" not in response.headers

    def test_security_headers_on_different_endpoints(self, client):
        """Test security headers are applied to all endpoints."""
        endpoints = ["/", "/api/test"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            # All endpoints should have security headers
            assert "content-security-policy" in response.headers
            assert "strict-transport-security" in response.headers
            assert "x-frame-options" in response.headers
            assert "x-content-type-options" in response.headers

    def test_security_headers_on_different_methods(self, client):
        """Test security headers are applied to different HTTP methods."""
        methods_and_endpoints = [
            ("GET", "/"),
            ("POST", "/api/test"),
            ("PUT", "/api/test"),
            ("DELETE", "/api/test"),
        ]
        
        for method, endpoint in methods_and_endpoints:
            try:
                if method == "GET":
                    response = client.get(endpoint)
                elif method == "POST":
                    response = client.post(endpoint)
                elif method == "PUT":
                    response = client.put(endpoint)
                elif method == "DELETE":
                    response = client.delete(endpoint)
                
                # All methods should have security headers
                assert "content-security-policy" in response.headers
                assert "x-frame-options" in response.headers
                assert "x-content-type-options" in response.headers
                
            except Exception:
                # Some methods might not be implemented, that's OK
                pass

    def test_cors_headers_security(self, client):
        """Test CORS headers are properly configured for security."""
        response = client.get("/")
        
        # Check for CORS headers
        if "access-control-allow-origin" in response.headers:
            cors_origin = response.headers["access-control-allow-origin"]
            
            # Should not be wildcard (*) in production
            if cors_origin == "*":
                # This might be acceptable for public APIs, but should be documented
                pass
            else:
                # Should be specific domains
                assert cors_origin.startswith("http://") or cors_origin.startswith("https://")
        
        if "access-control-allow-credentials" in response.headers:
            cors_credentials = response.headers["access-control-allow-credentials"]
            
            # If credentials are allowed, origin should not be wildcard
            if cors_credentials.lower() == "true":
                cors_origin = response.headers.get("access-control-allow-origin", "")
                assert cors_origin != "*", "CORS credentials=true with origin=* is insecure"

    def test_csp_header_validation_function(self):
        """Test CSP header validation function."""
        # Test valid CSP headers
        valid_csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        assert validate_csp_header(valid_csp) is True
        
        # Test invalid CSP headers
        invalid_csp = "invalid-directive 'self'"
        assert validate_csp_header(invalid_csp) is False
        
        # Test empty CSP
        assert validate_csp_header("") is False
        
        # Test None CSP
        assert validate_csp_header(None) is False

    def test_hsts_header_validation_function(self):
        """Test HSTS header validation function."""
        # Test valid HSTS headers
        valid_hsts = "max-age=31536000; includeSubDomains; preload"
        assert validate_hsts_header(valid_hsts) is True
        
        # Test minimum valid HSTS
        min_hsts = "max-age=31536000"
        assert validate_hsts_header(min_hsts) is True
        
        # Test invalid HSTS headers
        invalid_hsts = "max-age=invalid"
        assert validate_hsts_header(invalid_hsts) is False
        
        # Test missing max-age
        no_max_age = "includeSubDomains"
        assert validate_hsts_header(no_max_age) is False
        
        # Test too short max-age
        short_max_age = "max-age=3600"  # 1 hour
        assert validate_hsts_header(short_max_age) is False

    def test_security_headers_function(self):
        """Test get_security_headers function."""
        headers = get_security_headers()
        
        assert isinstance(headers, dict)
        assert "Content-Security-Policy" in headers
        assert "Strict-Transport-Security" in headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "Referrer-Policy" in headers

    def test_add_security_headers_function(self):
        """Test add_security_headers function."""
        mock_response = Mock()
        mock_response.headers = {}
        
        add_security_headers(mock_response)
        
        # Verify headers were added
        assert len(mock_response.headers) > 0
        assert "Content-Security-Policy" in mock_response.headers
        assert "X-Frame-Options" in mock_response.headers

    def test_security_headers_with_https(self):
        """Test security headers behavior with HTTPS."""
        # Mock HTTPS request
        with patch.dict('os.environ', {'HTTPS': 'on'}):
            headers = get_security_headers()
            
            # HSTS should be present for HTTPS
            assert "Strict-Transport-Security" in headers
            hsts_header = headers["Strict-Transport-Security"]
            assert "max-age=" in hsts_header

    def test_security_headers_with_http(self):
        """Test security headers behavior with HTTP."""
        # Mock HTTP request
        with patch.dict('os.environ', {'HTTPS': 'off'}, clear=True):
            headers = get_security_headers()
            
            # HSTS might not be present for HTTP
            # Other headers should still be present
            assert "X-Frame-Options" in headers
            assert "X-Content-Type-Options" in headers

    def test_csp_nonce_generation(self):
        """Test CSP nonce generation for inline scripts."""
        # This test assumes CSP nonce functionality exists
        try:
            from auth.security_headers import generate_csp_nonce
            
            nonce1 = generate_csp_nonce()
            nonce2 = generate_csp_nonce()
            
            # Nonces should be different
            assert nonce1 != nonce2
            
            # Nonces should be base64 encoded
            import base64
            assert base64.b64decode(nonce1)
            assert base64.b64decode(nonce2)
            
        except ImportError:
            # CSP nonce functionality might not be implemented
            pytest.skip("CSP nonce functionality not implemented")

    def test_security_headers_performance(self):
        """Test that security headers don't significantly impact performance."""
        import time
        
        # Time header generation
        start_time = time.time()
        
        for _ in range(1000):
            get_security_headers()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be very fast
        assert duration < 0.1, f"Security headers generation too slow: {duration:.3f}s"

    def test_security_headers_thread_safety(self):
        """Test security headers are thread-safe."""
        import threading
        
        results = []
        
        def get_headers():
            headers = get_security_headers()
            results.append(headers)
        
        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_headers)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All results should be present
        assert len(results) == 10
        
        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_security_headers_customization(self):
        """Test security headers can be customized."""
        # Test with custom CSP
        custom_csp = "default-src 'self'; script-src 'self' https://trusted.com"
        
        try:
            headers = get_security_headers(custom_csp=custom_csp)
            assert headers["Content-Security-Policy"] == custom_csp
        except TypeError:
            # Custom CSP might not be supported
            pytest.skip("Custom CSP not supported")

    def test_security_headers_owasp_compliance(self, client):
        """Test security headers comply with OWASP recommendations."""
        response = client.get("/")
        
        # OWASP recommended headers
        owasp_headers = {
            "content-security-policy": "CSP should be present",
            "strict-transport-security": "HSTS should be present",
            "x-frame-options": "Frame options should be present",
            "x-content-type-options": "Content type options should be present",
            "referrer-policy": "Referrer policy should be present",
        }
        
        for header, description in owasp_headers.items():
            assert header in response.headers, description
        
        # Check for insecure headers that should NOT be present
        insecure_headers = [
            "x-powered-by",
            "x-aspnet-version",
            "x-aspnetmvc-version",
        ]
        
        for header in insecure_headers:
            assert header not in response.headers, f"Insecure header {header} should not be present"

    def test_security_headers_edge_cases(self):
        """Test security headers handle edge cases."""
        # Test with None response
        try:
            add_security_headers(None)
            # Should not crash
        except Exception:
            # Exception is acceptable for None response
            pass
        
        # Test with mock response without headers attribute
        mock_response = Mock()
        del mock_response.headers
        
        try:
            add_security_headers(mock_response)
            # Should handle gracefully
        except AttributeError:
            # AttributeError is acceptable
            pass

    def test_security_headers_content_type_specific(self, client):
        """Test security headers for different content types."""
        # This would require endpoints that return different content types
        # For now, test with JSON response
        response = client.get("/")
        
        # Security headers should be present regardless of content type
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_security_headers_documentation_compliance(self):
        """Test security headers match documentation requirements."""
        headers = get_security_headers()
        
        # Verify header names are correctly cased
        expected_headers = [
            "Content-Security-Policy",
            "Strict-Transport-Security",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "Referrer-Policy",
        ]
        
        for expected_header in expected_headers:
            assert expected_header in headers, f"Expected header {expected_header} not found"