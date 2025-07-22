"""
HTTPS Enforcement and SSL/TLS Configuration Module for FreeAgentics.

This module implements comprehensive HTTPS enforcement and SSL/TLS setup
following OWASP security guidelines and Task #14.10 requirements.
"""

import logging
import os
import subprocess  # nosec B404
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class SSLConfiguration:
    """SSL/TLS configuration settings."""

    # Certificate paths
    cert_path: str = "/etc/nginx/ssl/cert.pem"
    key_path: str = "/etc/nginx/ssl/key.pem"
    chain_path: str = "/etc/nginx/ssl/chain.pem"

    # Let's Encrypt settings
    enable_letsencrypt: bool = True
    letsencrypt_email: str = ""
    letsencrypt_domains: Optional[List[str]] = None
    letsencrypt_staging: bool = False

    # SSL/TLS settings
    min_tls_version: str = "TLSv1.2"
    preferred_tls_version: str = "TLSv1.3"
    cipher_suites: Optional[List[str]] = None

    # HSTS settings
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True

    # Cookie security
    secure_cookies: bool = True
    cookie_samesite: str = "strict"

    # Certificate monitoring
    cert_expiry_warning_days: int = 30
    cert_renewal_days: int = 30

    # Load balancer settings
    behind_load_balancer: bool = False
    trusted_proxies: List[str] = None

    # Environment detection
    production_mode: bool = None

    def __post_init__(self) -> None:
        """Initialize configuration with defaults."""
        if self.production_mode is None:
            self.production_mode = os.getenv("PRODUCTION", "false").lower() == "true"

        if self.letsencrypt_domains is None:
            self.letsencrypt_domains = []

        if self.cipher_suites is None:
            # Strong cipher suites for modern compatibility
            self.cipher_suites = [
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-ECDSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-ECDSA-CHACHA20-POLY1305",
                "ECDHE-RSA-CHACHA20-POLY1305",
                "DHE-RSA-AES128-GCM-SHA256",
                "DHE-RSA-AES256-GCM-SHA384",
            ]

        if self.trusted_proxies is None:
            self.trusted_proxies = ["127.0.0.1", "::1"]

        # Load from environment
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if email := os.getenv("LETSENCRYPT_EMAIL"):
            self.letsencrypt_email = email

        if domains := os.getenv("LETSENCRYPT_DOMAINS"):
            self.letsencrypt_domains = [d.strip() for d in domains.split(",")]

        if staging := os.getenv("LETSENCRYPT_STAGING"):
            self.letsencrypt_staging = staging.lower() == "true"

        if max_age := os.getenv("HSTS_MAX_AGE"):
            try:
                self.hsts_max_age = int(max_age)
            except ValueError:
                logger.warning(f"Invalid HSTS_MAX_AGE: {max_age}")

        if proxies := os.getenv("TRUSTED_PROXIES"):
            self.trusted_proxies = [p.strip() for p in proxies.split(",")]


class HTTPSEnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce HTTPS and handle SSL/TLS configuration."""

    def __init__(self, app, config: Optional[SSLConfiguration] = None):
        """Initialize HTTPS enforcement middleware."""
        super().__init__(app)
        self.config = config or SSLConfiguration()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and enforce HTTPS."""
        # Check if request is secure
        is_secure = await self._is_secure_request(request)

        # Allow Let's Encrypt challenges over HTTP
        if request.url.path.startswith("/.well-known/acme-challenge/"):
            return await call_next(request)

        # Redirect HTTP to HTTPS in production
        if not is_secure and self.config.production_mode:
            return self._redirect_to_https(request)

        # Process request
        response = await call_next(request)

        # Add HSTS header for HTTPS requests
        if is_secure and self.config.hsts_enabled:
            response.headers["Strict-Transport-Security"] = self._generate_hsts_header()

        # Ensure secure cookies
        if self.config.secure_cookies:
            self._enforce_secure_cookies(response, is_secure)

        return response

    async def _is_secure_request(self, request: Request) -> bool:
        """Check if request is over HTTPS."""
        # Direct HTTPS check
        if request.url.scheme == "https":
            return True

        # Check X-Forwarded-Proto header (for load balancers)
        if self.config.behind_load_balancer:
            client_host = request.client.host if request.client else None
            if client_host in self.config.trusted_proxies:
                forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
                if forwarded_proto == "https":
                    return True

        return False

    def _redirect_to_https(self, request: Request) -> RedirectResponse:
        """Redirect HTTP request to HTTPS."""
        # Build HTTPS URL
        https_url = request.url.replace(scheme="https")

        # Use 301 permanent redirect
        return RedirectResponse(
            url=str(https_url),
            status_code=301,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Redirect-Reason": "HTTPS-Required",
            },
        )

    def _generate_hsts_header(self) -> str:
        """Generate HSTS header value."""
        parts = [f"max-age={self.config.hsts_max_age}"]

        if self.config.hsts_include_subdomains:
            parts.append("includeSubDomains")

        if self.config.hsts_preload:
            parts.append("preload")

        return "; ".join(parts)

    def _enforce_secure_cookies(self, response: Response, is_secure: bool) -> None:
        """Enforce secure cookie flags."""
        # Parse Set-Cookie headers
        set_cookie_headers = []

        for header_name, header_value in response.raw_headers:
            if header_name.lower() == b"set-cookie":
                cookie_str = header_value.decode("latin-1")

                # Add secure flag if HTTPS
                if is_secure and "secure" not in cookie_str.lower():
                    cookie_str += "; Secure"

                # Add HttpOnly if not present
                if "httponly" not in cookie_str.lower():
                    cookie_str += "; HttpOnly"

                # Add SameSite if not present
                if "samesite" not in cookie_str.lower():
                    cookie_str += f"; SameSite={self.config.cookie_samesite}"

                set_cookie_headers.append((b"set-cookie", cookie_str.encode("latin-1")))
            else:
                set_cookie_headers.append((header_name, header_value))

        # Update headers
        response.raw_headers = set_cookie_headers


class SSLCertificateManager:
    """Manages SSL certificates including Let's Encrypt integration."""

    def __init__(self, config: SSLConfiguration):
        """Initialize SSL certificate manager."""
        self.config = config
        self.certbot_path = self._find_certbot()

    def _find_certbot(self) -> Optional[str]:
        """Find certbot executable."""
        try:
            result = (
                subprocess.run(  # nosec B607 B603 # Safe use of which command for certbot detection
                    ["which", "certbot"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.warning("Certbot not found in PATH")
            return None

    def setup_letsencrypt(self) -> bool:
        """Set up Let's Encrypt certificates."""
        if not self.certbot_path:
            logger.error("Certbot not installed")
            return False

        if not self.config.letsencrypt_email:
            logger.error("Let's Encrypt email not configured")
            return False

        if not self.config.letsencrypt_domains:
            logger.error("No domains configured for Let's Encrypt")
            return False

        # Build certbot command
        cmd = [
            self.certbot_path,
            "certonly",
            "--webroot",
            "--webroot-path",
            "/var/www/certbot",
            "--email",
            self.config.letsencrypt_email,
            "--agree-tos",
            "--no-eff-email",
            "--force-renewal",
        ]

        # Add staging flag if configured
        if self.config.letsencrypt_staging:
            cmd.append("--staging")

        # Add domains
        for domain in self.config.letsencrypt_domains:
            cmd.extend(["-d", domain])

        try:
            # Run certbot
            logger.info(
                f"Obtaining Let's Encrypt certificate for domains: {self.config.letsencrypt_domains}"
            )
            subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )  # nosec B603 # Safe certbot command execution
            logger.info("Let's Encrypt certificate obtained successfully")

            # Copy certificates to configured paths
            self._copy_certificates()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to obtain Let's Encrypt certificate: {e.stderr}")
            return False

    def _copy_certificates(self) -> None:
        """Copy Let's Encrypt certificates to configured paths."""
        primary_domain = self.config.letsencrypt_domains[0]
        le_path = f"/etc/letsencrypt/live/{primary_domain}"

        try:
            # Copy certificate files
            subprocess.run(
                ["cp", f"{le_path}/fullchain.pem", self.config.cert_path],
                check=True,
            )

            subprocess.run(
                ["cp", f"{le_path}/privkey.pem", self.config.key_path],
                check=True,
            )

            subprocess.run(
                ["cp", f"{le_path}/chain.pem", self.config.chain_path],
                check=True,
            )

            # Set proper permissions
            os.chmod(self.config.key_path, 0o600)
            os.chmod(self.config.cert_path, 0o644)
            os.chmod(self.config.chain_path, 0o644)

            logger.info("Certificates copied successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy certificates: {e}")
            raise

    def setup_auto_renewal(self) -> bool:
        """Set up automatic certificate renewal."""
        # Create renewal script
        renewal_script = """#!/bin/bash
# Auto-renewal script for Let's Encrypt certificates

set -e

# Renew certificates
certbot renew --quiet --no-self-upgrade

# Reload nginx if renewal was successful
if [ $? -eq 0 ]; then
    nginx -s reload || systemctl reload nginx
fi
"""

        script_path = "/usr/local/bin/renew-letsencrypt.sh"

        try:
            # Write renewal script
            with open(script_path, "w") as f:
                f.write(renewal_script)

            # Make executable (only owner/root can read, write, execute)
            # Security: 0o700 is restrictive - only owner can access (rwx------)
            os.chmod(script_path, 0o700)  # nosec B103

            # Add to crontab (runs twice daily)
            cron_entry = f"0 0,12 * * * {script_path} >> /var/log/letsencrypt-renewal.log 2>&1\n"

            # Check if cron entry exists
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)

            if script_path not in result.stdout:
                # Add cron entry
                current_crontab = result.stdout if result.returncode == 0 else ""
                new_crontab = current_crontab + cron_entry

                process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
                process.communicate(input=new_crontab)

                logger.info("Auto-renewal cron job added successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to set up auto-renewal: {e}")
            return False

    def check_certificate_expiry(self) -> Optional[timedelta]:
        """Check certificate expiry time."""
        if not Path(self.config.cert_path).exists():
            return None

        try:
            # Use openssl to check certificate expiry
            result = subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-in",
                    self.config.cert_path,
                    "-noout",
                    "-enddate",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse expiry date
            expiry_str = result.stdout.strip().replace("notAfter=", "")
            expiry_date = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")

            # Calculate time until expiry
            time_until_expiry = expiry_date - datetime.utcnow()

            # Log warning if expiring soon
            if time_until_expiry.days <= self.config.cert_expiry_warning_days:
                logger.warning(f"Certificate expiring in {time_until_expiry.days} days")

            return time_until_expiry

        except Exception as e:
            logger.error(f"Failed to check certificate expiry: {e}")
            return None

    def validate_certificate_chain(self) -> bool:
        """Validate certificate chain."""
        try:
            # Verify certificate chain
            result = subprocess.run(
                [
                    "openssl",
                    "verify",
                    "-CAfile",
                    self.config.chain_path,
                    self.config.cert_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return "OK" in result.stdout

        except subprocess.CalledProcessError:
            return False


class LoadBalancerSSLConfig:
    """Configuration for SSL termination at load balancer."""

    def __init__(self, config: SSLConfiguration):
        """Initialize load balancer SSL configuration."""
        self.config = config

    def generate_aws_alb_config(self) -> Dict[str, Any]:
        """Generate AWS Application Load Balancer SSL configuration."""
        return {
            "Protocol": "HTTPS",
            "Port": 443,
            "SslPolicy": "ELBSecurityPolicy-TLS-1-2-2017-01",
            "Certificates": [{"CertificateArn": "arn:aws:acm:region:account:certificate/id"}],
            "DefaultActions": [
                {
                    "Type": "forward",
                    "TargetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/name",
                }
            ],
        }

    def generate_nginx_upstream_config(self) -> str:
        """Generate nginx configuration for SSL termination."""
        return f"""
# SSL termination at load balancer
server {{
    listen 80;
    server_name _;

    # Trust X-Forwarded headers from load balancer
    set_real_ip_from {" ".join(self.config.trusted_proxies)};
    real_ip_header X-Forwarded-For;

    # Enforce HTTPS through X-Forwarded-Proto
    if ($http_x_forwarded_proto != "https") {{
        return 301 https://$host$request_uri;
    }}

    # HSTS header (load balancer should also add this)
    add_header Strict-Transport-Security "{self._generate_hsts_value()}" always;

    # Proxy to application
    location / {{
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
        proxy_set_header X-Forwarded-Port $server_port;
        proxy_set_header X-Forwarded-SSL on;
    }}
}}
"""

    def _generate_hsts_value(self) -> str:
        """Generate HSTS header value."""
        parts = [f"max-age={self.config.hsts_max_age}"]

        if self.config.hsts_include_subdomains:
            parts.append("includeSubDomains")

        if self.config.hsts_preload:
            parts.append("preload")

        return "; ".join(parts)


def setup_https_enforcement(app, config: Optional[SSLConfiguration] = None) -> SSLConfiguration:
    """Set up HTTPS enforcement middleware."""
    config = config or SSLConfiguration()
    app.add_middleware(HTTPSEnforcementMiddleware, config=config)

    logger.info("HTTPS enforcement middleware configured")
    return config


# Development SSL setup helper
def generate_self_signed_cert(domain: str = "localhost", days: int = 365) -> Tuple[str, str]:
    """Generate self-signed certificate for development."""
    cert_dir = Path("./ssl")
    cert_dir.mkdir(exist_ok=True)

    cert_path = cert_dir / f"{domain}.crt"
    key_path = cert_dir / f"{domain}.key"

    try:
        # Generate private key and certificate
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-days",
                str(days),
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-subj",
                f"/C=US/ST=CA/L=San Francisco/O=FreeAgentics/CN={domain}",
            ],
            check=True,
        )

        logger.info(f"Generated self-signed certificate for {domain}")
        return str(cert_path), str(key_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate self-signed certificate: {e}")
        raise


# Export production configuration
PRODUCTION_SSL_CONFIG = SSLConfiguration(
    production_mode=True,
    enable_letsencrypt=True,
    hsts_enabled=True,
    hsts_preload=True,
    secure_cookies=True,
    behind_load_balancer=True,
)
