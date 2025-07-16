"""
Comprehensive SSL/TLS Configuration Tests for FreeAgentics

Tests HTTPS enforcement, certificate validation, HSTS, secure cookies,
and SSL/TLS security configurations.
"""

import asyncio
import os
import re
import ssl
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from httpx import AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from auth.https_enforcement import (
    HTTPSEnforcementMiddleware,
    LoadBalancerSSLConfig,
    SSLCertificateManager,
    SSLConfiguration,
    generate_self_signed_cert,
    setup_https_enforcement,
)
from auth.security_headers import SecurityHeadersMiddleware, SecurityPolicy


class TestSSLConfiguration:
    """Test SSL/TLS configuration settings."""
    
    def test_default_configuration(self):
        """Test default SSL configuration values."""
        config = SSLConfiguration()
        
        # Check default values
        assert config.min_tls_version == "TLSv1.2"
        assert config.preferred_tls_version == "TLSv1.3"
        assert config.hsts_enabled is True
        assert config.hsts_max_age == 31536000  # 1 year
        assert config.secure_cookies is True
        assert config.cookie_samesite == "strict"
        assert len(config.cipher_suites) > 0
        
        # Check strong ciphers only
        weak_ciphers = ["RC4", "3DES", "MD5", "NULL", "EXPORT"]
        for cipher in config.cipher_suites:
            for weak in weak_ciphers:
                assert weak not in cipher
    
    def test_production_mode_configuration(self):
        """Test production mode configuration."""
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            config = SSLConfiguration()
            
            assert config.production_mode is True
            assert config.hsts_preload is True
            assert config.secure_cookies is True
    
    def test_environment_loading(self):
        """Test loading configuration from environment."""
        env_vars = {
            "LETSENCRYPT_EMAIL": "admin@example.com",
            "LETSENCRYPT_DOMAINS": "example.com,www.example.com",
            "LETSENCRYPT_STAGING": "true",
            "HSTS_MAX_AGE": "63072000",
            "TRUSTED_PROXIES": "10.0.0.1,10.0.0.2"
        }
        
        with patch.dict(os.environ, env_vars):
            config = SSLConfiguration()
            
            assert config.letsencrypt_email == "admin@example.com"
            assert config.letsencrypt_domains == ["example.com", "www.example.com"]
            assert config.letsencrypt_staging is True
            assert config.hsts_max_age == 63072000
            assert config.trusted_proxies == ["10.0.0.1", "10.0.0.2"]


class TestHTTPSEnforcementMiddleware:
    """Test HTTPS enforcement middleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        @app.get("/secure")
        async def secure():
            response = Response(content="Secure content")
            response.set_cookie("session", "abc123")
            return response
        
        @app.get("/.well-known/acme-challenge/test")
        async def acme_challenge():
            return {"challenge": "test"}
        
        return app
    
    def test_http_to_https_redirect(self, app):
        """Test HTTP to HTTPS redirect in production mode."""
        config = SSLConfiguration(production_mode=True)
        app.add_middleware(HTTPSEnforcementMiddleware, config=config)
        
        client = TestClient(app)
        response = client.get("/", follow_redirects=False)
        
        assert response.status_code == 301
        assert response.headers["location"].startswith("https://")
        assert "HTTPS-Required" in response.headers.get("X-Redirect-Reason", "")
    
    def test_acme_challenge_allowed_over_http(self, app):
        """Test Let's Encrypt challenges allowed over HTTP."""
        config = SSLConfiguration(production_mode=True)
        app.add_middleware(HTTPSEnforcementMiddleware, config=config)
        
        client = TestClient(app)
        response = client.get("/.well-known/acme-challenge/test")
        
        assert response.status_code == 200
        assert response.json() == {"challenge": "test"}
    
    def test_hsts_header_on_https(self, app):
        """Test HSTS header is added on HTTPS requests."""
        config = SSLConfiguration(
            production_mode=False,  # Allow HTTP in tests
            hsts_enabled=True,
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            hsts_preload=True
        )
        app.add_middleware(HTTPSEnforcementMiddleware, config=config)
        
        client = TestClient(app)
        
        # Simulate HTTPS request
        with patch.object(client, "_base_url", "https://testserver"):
            response = client.get("/", headers={"X-Forwarded-Proto": "https"})
        
        hsts = response.headers.get("Strict-Transport-Security")
        assert hsts is not None
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts
        assert "preload" in hsts
    
    def test_secure_cookie_enforcement(self, app):
        """Test secure cookie flags are enforced."""
        config = SSLConfiguration(
            production_mode=False,
            secure_cookies=True,
            cookie_samesite="strict"
        )
        app.add_middleware(HTTPSEnforcementMiddleware, config=config)
        
        client = TestClient(app)
        
        # Test with HTTPS
        with patch.object(client, "_base_url", "https://testserver"):
            response = client.get("/secure", headers={"X-Forwarded-Proto": "https"})
        
        set_cookie = response.headers.get("set-cookie")
        assert set_cookie is not None
        assert "Secure" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie
    
    def test_load_balancer_forwarded_headers(self, app):
        """Test handling of forwarded headers from load balancer."""
        config = SSLConfiguration(
            production_mode=True,
            behind_load_balancer=True,
            trusted_proxies=["127.0.0.1"]
        )
        app.add_middleware(HTTPSEnforcementMiddleware, config=config)
        
        client = TestClient(app)
        
        # Simulate request from load balancer
        response = client.get(
            "/",
            headers={
                "X-Forwarded-Proto": "https",
                "X-Forwarded-For": "192.168.1.100",
                "X-Real-IP": "192.168.1.100"
            }
        )
        
        # Should not redirect when X-Forwarded-Proto is https
        assert response.status_code == 200
        assert "Strict-Transport-Security" in response.headers


class TestSSLCertificateManager:
    """Test SSL certificate management."""
    
    @pytest.fixture
    def config(self):
        """Create test SSL configuration."""
        return SSLConfiguration(
            letsencrypt_email="test@example.com",
            letsencrypt_domains=["example.com", "www.example.com"],
            cert_path="/tmp/test-cert.pem",
            key_path="/tmp/test-key.pem",
            chain_path="/tmp/test-chain.pem"
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create certificate manager."""
        return SSLCertificateManager(config)
    
    @patch("subprocess.run")
    def test_find_certbot(self, mock_run):
        """Test finding certbot executable."""
        mock_run.return_value = MagicMock(
            stdout="/usr/bin/certbot\n",
            returncode=0
        )
        
        config = SSLConfiguration()
        manager = SSLCertificateManager(config)
        
        assert manager.certbot_path == "/usr/bin/certbot"
        mock_run.assert_called_once_with(
            ["which", "certbot"],
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch("subprocess.run")
    def test_setup_letsencrypt_success(self, mock_run, manager):
        """Test successful Let's Encrypt setup."""
        mock_run.return_value = MagicMock(returncode=0)
        manager.certbot_path = "/usr/bin/certbot"
        
        # Mock certificate copying
        with patch.object(manager, "_copy_certificates"):
            result = manager.setup_letsencrypt()
        
        assert result is True
        
        # Verify certbot command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/usr/bin/certbot"
        assert "certonly" in call_args
        assert "--webroot" in call_args
        assert "--email" in call_args
        assert "test@example.com" in call_args
        assert "-d" in call_args
        assert "example.com" in call_args
        assert "www.example.com" in call_args
    
    @patch("subprocess.run")
    @patch("os.chmod")
    def test_copy_certificates(self, mock_chmod, mock_run, manager):
        """Test copying Let's Encrypt certificates."""
        mock_run.return_value = MagicMock(returncode=0)
        
        manager._copy_certificates()
        
        # Verify copy commands
        assert mock_run.call_count == 3  # cert, key, chain
        
        # Verify permissions
        assert mock_chmod.call_count == 3
        mock_chmod.assert_any_call("/tmp/test-key.pem", 0o600)
        mock_chmod.assert_any_call("/tmp/test-cert.pem", 0o644)
        mock_chmod.assert_any_call("/tmp/test-chain.pem", 0o644)
    
    @patch("subprocess.run")
    @patch("builtins.open", create=True)
    @patch("os.chmod")
    def test_setup_auto_renewal(self, mock_chmod, mock_open, mock_run, manager):
        """Test setting up automatic certificate renewal."""
        # Mock crontab listing
        mock_run.return_value = MagicMock(
            stdout="",
            returncode=0
        )
        
        result = manager.setup_auto_renewal()
        
        assert result is True
        
        # Verify script was written
        mock_open.assert_called_once_with("/usr/local/bin/renew-letsencrypt.sh", "w")
        
        # Verify script permissions
        mock_chmod.assert_called_with("/usr/local/bin/renew-letsencrypt.sh", 0o755)
    
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_check_certificate_expiry(self, mock_exists, mock_run, manager):
        """Test checking certificate expiry."""
        mock_exists.return_value = True
        
        # Mock certificate with 45 days validity
        expiry_date = datetime.utcnow() + timedelta(days=45)
        mock_run.return_value = MagicMock(
            stdout=f"notAfter={expiry_date.strftime('%b %d %H:%M:%S %Y GMT')}\n",
            returncode=0
        )
        
        time_until_expiry = manager.check_certificate_expiry()
        
        assert time_until_expiry is not None
        assert 44 <= time_until_expiry.days <= 45
    
    @patch("subprocess.run")
    def test_validate_certificate_chain(self, mock_run, manager):
        """Test certificate chain validation."""
        mock_run.return_value = MagicMock(
            stdout="test-cert.pem: OK\n",
            returncode=0
        )
        
        result = manager.validate_certificate_chain()
        
        assert result is True
        
        # Verify openssl verify command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "openssl"
        assert call_args[1] == "verify"
        assert "-CAfile" in call_args


class TestLoadBalancerSSLConfig:
    """Test load balancer SSL configuration."""
    
    def test_generate_aws_alb_config(self):
        """Test generating AWS ALB SSL configuration."""
        config = SSLConfiguration()
        lb_config = LoadBalancerSSLConfig(config)
        
        alb_config = lb_config.generate_aws_alb_config()
        
        assert alb_config["Protocol"] == "HTTPS"
        assert alb_config["Port"] == 443
        assert "SslPolicy" in alb_config
        assert "TLS-1-2" in alb_config["SslPolicy"]
    
    def test_generate_nginx_upstream_config(self):
        """Test generating nginx upstream configuration."""
        config = SSLConfiguration(
            trusted_proxies=["10.0.0.1", "10.0.0.2"],
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            hsts_preload=True
        )
        lb_config = LoadBalancerSSLConfig(config)
        
        nginx_config = lb_config.generate_nginx_upstream_config()
        
        # Verify configuration elements
        assert "set_real_ip_from 10.0.0.1 10.0.0.2" in nginx_config
        assert "real_ip_header X-Forwarded-For" in nginx_config
        assert 'if ($http_x_forwarded_proto != "https")' in nginx_config
        assert "return 301 https://$host$request_uri" in nginx_config
        assert "Strict-Transport-Security" in nginx_config
        assert "max-age=31536000" in nginx_config
        assert "includeSubDomains" in nginx_config
        assert "preload" in nginx_config


class TestDevelopmentSSL:
    """Test development SSL utilities."""
    
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_generate_self_signed_cert(self, mock_mkdir, mock_run):
        """Test generating self-signed certificate for development."""
        mock_run.return_value = MagicMock(returncode=0)
        
        cert_path, key_path = generate_self_signed_cert("localhost", days=365)
        
        assert cert_path == "./ssl/localhost.crt"
        assert key_path == "./ssl/localhost.key"
        
        # Verify openssl command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "openssl"
        assert call_args[1] == "req"
        assert "-x509" in call_args
        assert "-days" in call_args
        assert "365" in call_args


class TestSSLSecurityValidation:
    """Test SSL/TLS security validation."""
    
    def test_cipher_suite_strength(self):
        """Test cipher suite strength validation."""
        config = SSLConfiguration()
        
        # All configured ciphers should be strong
        for cipher in config.cipher_suites:
            # Should use AEAD ciphers (GCM or POLY1305)
            assert "GCM" in cipher or "POLY1305" in cipher
            
            # Should use ECDHE or DHE for forward secrecy
            assert cipher.startswith("ECDHE") or cipher.startswith("DHE")
            
            # Should use strong encryption (AES128 or AES256)
            assert "AES128" in cipher or "AES256" in cipher or "CHACHA20" in cipher
    
    def test_tls_version_security(self):
        """Test TLS version security."""
        config = SSLConfiguration()
        
        # Should not allow TLS < 1.2
        assert config.min_tls_version >= "TLSv1.2"
        
        # Should prefer TLS 1.3
        assert config.preferred_tls_version == "TLSv1.3"
    
    def test_hsts_configuration(self):
        """Test HSTS configuration security."""
        config = SSLConfiguration(production_mode=True)
        
        # HSTS should be enabled in production
        assert config.hsts_enabled is True
        
        # Max age should be at least 1 year
        assert config.hsts_max_age >= 31536000
        
        # Should include subdomains
        assert config.hsts_include_subdomains is True
        
        # Should be preload ready in production
        assert config.hsts_preload is True
    
    def test_certificate_validation_requirements(self):
        """Test certificate validation requirements."""
        config = SSLConfiguration()
        
        # Certificate expiry warning should be reasonable
        assert config.cert_expiry_warning_days >= 30
        
        # Renewal should happen before expiry warning
        assert config.cert_renewal_days <= config.cert_expiry_warning_days


class TestIntegrationSSLTLS:
    """Integration tests for SSL/TLS functionality."""
    
    @pytest.fixture
    def ssl_app(self):
        """Create app with full SSL/TLS stack."""
        app = FastAPI()
        
        # Add security headers middleware
        security_policy = SecurityPolicy(
            production_mode=True,
            enable_hsts=True,
            secure_cookies=True
        )
        app.add_middleware(SecurityHeadersMiddleware, security_manager=security_policy)
        
        # Add HTTPS enforcement
        ssl_config = SSLConfiguration(
            production_mode=True,
            hsts_enabled=True,
            secure_cookies=True
        )
        app.add_middleware(HTTPSEnforcementMiddleware, config=ssl_config)
        
        @app.get("/api/data")
        async def get_data():
            response = Response(content='{"data": "secure"}')
            response.set_cookie("api_session", "xyz789", max_age=3600)
            return response
        
        return app
    
    def test_full_ssl_security_stack(self, ssl_app):
        """Test full SSL/TLS security stack."""
        client = TestClient(ssl_app)
        
        # Test HTTP request (should redirect)
        response = client.get("/api/data", follow_redirects=False)
        assert response.status_code == 301
        assert response.headers["location"].startswith("https://")
        
        # Test HTTPS request simulation
        with patch.object(client, "_base_url", "https://testserver"):
            response = client.get(
                "/api/data",
                headers={"X-Forwarded-Proto": "https"}
            )
        
        assert response.status_code == 200
        
        # Verify all security headers
        assert "Strict-Transport-Security" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Content-Security-Policy" in response.headers
        
        # Verify secure cookies
        set_cookie = response.headers.get("set-cookie")
        assert "Secure" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite" in set_cookie
    
    @pytest.mark.asyncio
    async def test_websocket_ssl_support(self, ssl_app):
        """Test WebSocket connections over SSL."""
        # Add WebSocket endpoint
        @ssl_app.websocket("/ws")
        async def websocket_endpoint(websocket):
            await websocket.accept()
            await websocket.send_text("Hello SSL WebSocket")
            await websocket.close()
        
        # WebSocket connections should work over SSL
        # (TestClient doesn't support WebSocket SSL testing directly)
        # This would be tested in a real environment with:
        # async with websockets.connect("wss://localhost/ws") as websocket:
        #     message = await websocket.recv()
        #     assert message == "Hello SSL WebSocket"
        pass


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])