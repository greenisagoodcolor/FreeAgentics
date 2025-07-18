"""
Unified Security Headers Management for FreeAgentics

Implements comprehensive security headers and SSL/TLS configuration
following OWASP security guidelines and Task #14.5 requirements.
"""

import base64
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Security policy configuration for headers and cookies."""

    # HSTS Configuration
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True  # Enable preload by default for production

    # Content Security Policy
    csp_policy: Optional[str] = None
    csp_report_uri: Optional[str] = None
    csp_report_only: bool = False

    # Frame Options
    x_frame_options: str = "DENY"

    # Content Type Options
    x_content_type_options: str = "nosniff"

    # XSS Protection
    x_xss_protection: str = "1; mode=block"

    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Permissions Policy
    permissions_policy: Optional[str] = None

    # Cache Control
    cache_control_default: str = (
        "no-store, no-cache, must-revalidate, proxy-revalidate"
    )
    cache_control_static: str = "public, max-age=31536000, immutable"
    cache_control_sensitive: str = (
        "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0"
    )

    # Expect-CT
    enable_expect_ct: bool = True
    expect_ct_max_age: int = 86400  # 24 hours
    expect_ct_enforce: bool = True
    expect_ct_report_uri: Optional[str] = None

    # Certificate Pinning
    enable_certificate_pinning: bool = False
    certificate_pins: Dict[str, List[str]] = field(default_factory=dict)

    # Cookie Security
    secure_cookies: bool = True
    httponly_cookies: bool = True
    samesite_cookies: str = "strict"

    # Environment Detection
    production_mode: bool = field(
        default_factory=lambda: os.getenv("PRODUCTION", "false").lower()
        == "true"
    )

    def __post_init__(self):
        """Initialize policy based on environment."""
        if self.production_mode:
            self.hsts_preload = True
            self.enable_certificate_pinning = True
            self.expect_ct_enforce = True
        else:
            # Development mode adjustments
            self.secure_cookies = False  # Allow HTTP in development


# Legacy CertificatePinner for backward compatibility
class CertificatePinner:
    """Legacy certificate pinning implementation for backward compatibility."""

    def __init__(self):
        # Use the enhanced mobile certificate pinner
        from auth.certificate_pinning import mobile_cert_pinner

        self._mobile_pinner = mobile_cert_pinner

        self.pins: Dict[str, List[str]] = {}
        self.backup_pins: Dict[str, List[str]] = {}
        self.max_age = 5184000  # 60 days
        self.include_subdomains = True

    def add_pin(self, domain: str, pin_sha256: str):
        """Add a certificate pin for a domain."""
        if domain not in self.pins:
            self.pins[domain] = []
        self.pins[domain].append(pin_sha256)

        # Also add to mobile pinner
        from auth.certificate_pinning import PinConfiguration

        config = PinConfiguration(primary_pins=[pin_sha256])
        self._mobile_pinner.add_domain_pins(domain, config)

        logger.info(f"Added certificate pin for domain: {domain}")

    def add_backup_pin(self, domain: str, backup_sha256: str):
        """Add a backup certificate pin for a domain."""
        if domain not in self.backup_pins:
            self.backup_pins[domain] = []
        self.backup_pins[domain].append(backup_sha256)
        logger.info(f"Added backup certificate pin for domain: {domain}")

    def generate_header(self, domain: str) -> Optional[str]:
        """Generate Public-Key-Pins header for a domain."""
        # Use enhanced mobile pinner if available
        header = self._mobile_pinner.get_pinning_header(domain)
        if header:
            return header

        # Fallback to legacy implementation
        domain_pins = self.pins.get(domain, [])
        domain_backup_pins = self.backup_pins.get(domain, [])

        if not domain_pins:
            return None

        # Combine primary and backup pins
        all_pins = []
        for pin in domain_pins:
            all_pins.append(f'pin-sha256="{pin}"')
        for backup_pin in domain_backup_pins:
            all_pins.append(f'pin-sha256="{backup_pin}"')

        header_parts = all_pins + [f"max-age={self.max_age}"]

        if self.include_subdomains:
            header_parts.append("includeSubDomains")

        return "; ".join(header_parts)

    def load_pins_from_env(self):
        """Load certificate pins from environment variables."""
        # The mobile pinner loads from environment automatically
        # Keep this for backward compatibility

        # Load primary pins
        primary_pins = os.getenv("CERT_PINS")
        if primary_pins:
            for pin_config in primary_pins.split(","):
                try:
                    domain, pin = pin_config.strip().split(":")
                    self.add_pin(domain, pin)
                except ValueError:
                    logger.warning(f"Invalid pin configuration: {pin_config}")

        # Load backup pins
        backup_pins = os.getenv("CERT_BACKUP_PINS")
        if backup_pins:
            for pin_config in backup_pins.split(","):
                try:
                    domain, pin = pin_config.strip().split(":")
                    self.add_backup_pin(domain, pin)
                except ValueError:
                    logger.warning(
                        f"Invalid backup pin configuration: {pin_config}"
                    )


class SecurityHeadersManager:
    """Centralized security headers management."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.certificate_pinner = CertificatePinner()
        self.certificate_pinner.load_pins_from_env()

        # Load customizations from environment
        self._load_env_customizations()

    def _load_env_customizations(self):
        """Load security policy customizations from environment variables."""
        # HSTS customizations
        if hsts_max_age := os.getenv("HSTS_MAX_AGE"):
            try:
                self.policy.hsts_max_age = int(hsts_max_age)
            except ValueError:
                logger.warning(f"Invalid HSTS_MAX_AGE value: {hsts_max_age}")

        # CSP customizations
        if csp_policy := os.getenv("CSP_POLICY"):
            self.policy.csp_policy = csp_policy

        if csp_report_uri := os.getenv("CSP_REPORT_URI"):
            self.policy.csp_report_uri = csp_report_uri

        # Expect-CT customizations
        if expect_ct_max_age := os.getenv("EXPECT_CT_MAX_AGE"):
            try:
                self.policy.expect_ct_max_age = int(expect_ct_max_age)
            except ValueError:
                logger.warning(
                    f"Invalid EXPECT_CT_MAX_AGE value: {expect_ct_max_age}"
                )

    def generate_hsts_header(self) -> str:
        """Generate Strict-Transport-Security header."""
        if not self.policy.enable_hsts:
            return ""

        header_parts = [f"max-age={self.policy.hsts_max_age}"]

        if self.policy.hsts_include_subdomains:
            header_parts.append("includeSubDomains")

        if self.policy.hsts_preload:
            header_parts.append("preload")

        return "; ".join(header_parts)

    def generate_csp_header(self, nonce: Optional[str] = None) -> str:
        """Generate Content-Security-Policy header."""
        if self.policy.csp_policy:
            csp = self.policy.csp_policy
        else:
            # Default comprehensive CSP with nonce support
            csp_directives = [
                "default-src 'self'",
                "script-src 'self'"
                + (f" 'nonce-{nonce}'" if nonce else " 'strict-dynamic'"),
                "style-src 'self'"
                + (f" 'nonce-{nonce}'" if nonce else " 'unsafe-inline'")
                + " https://fonts.googleapis.com",
                "img-src 'self' data: https:",
                "font-src 'self' data: https://fonts.gstatic.com",
                "connect-src 'self' wss: https:",
                "frame-ancestors 'none'",  # Stricter than 'self'
                "base-uri 'self'",
                "form-action 'self'",
                "object-src 'none'",
                "media-src 'self'",
                "worker-src 'self' blob:",  # Allow blob: for web workers
                "manifest-src 'self'",
                "upgrade-insecure-requests",  # Force HTTPS for all resources
                "block-all-mixed-content",  # Block HTTP content on HTTPS pages
            ]

            # Add custom script sources from environment
            if script_src := os.getenv("CSP_SCRIPT_SRC"):
                csp_directives[1] = f"script-src {script_src}"

            # Add custom style sources from environment
            if style_src := os.getenv("CSP_STYLE_SRC"):
                csp_directives[2] = f"style-src {style_src}"

            csp = "; ".join(csp_directives)

        # Add report URI if configured
        if self.policy.csp_report_uri:
            csp += f"; report-uri {self.policy.csp_report_uri}"

        return csp

    def generate_expect_ct_header(self) -> str:
        """Generate Expect-CT header."""
        if not self.policy.enable_expect_ct:
            return ""

        header_parts = [f"max-age={self.policy.expect_ct_max_age}"]

        if self.policy.expect_ct_enforce:
            header_parts.append("enforce")

        if self.policy.expect_ct_report_uri:
            header_parts.append(
                f'report-uri="{self.policy.expect_ct_report_uri}"'
            )

        return ", ".join(header_parts)

    def generate_permissions_policy(self) -> str:
        """Generate Permissions-Policy header."""
        if self.policy.permissions_policy:
            return self.policy.permissions_policy

        # Default restrictive permissions policy - deny all dangerous features
        return (
            "accelerometer=(), ambient-light-sensor=(), autoplay=(), battery=(), "
            "camera=(), cross-origin-isolated=(), display-capture=(), "
            "document-domain=(), encrypted-media=(), execution-while-not-rendered=(), "
            "execution-while-out-of-viewport=(), fullscreen=(self), geolocation=(), "
            "gyroscope=(), keyboard-map=(), magnetometer=(), microphone=(), "
            "midi=(), navigation-override=(), payment=(), picture-in-picture=(), "
            "publickey-credentials-get=(), screen-wake-lock=(), sync-xhr=(), "
            "usb=(), web-share=(), xr-spatial-tracking=(), clipboard-read=(), "
            "clipboard-write=(), gamepad=(), speaker-selection=(), "
            "conversion-measurement=(), focus-without-user-activation=(), "
            "hid=(), idle-detection=(), interest-cohort=(), serial=(), "
            "sync-script=(), trust-token-redemption=(), window-placement=(), "
            "vertical-scroll=(self)"
        )

    def generate_nonce(self) -> str:
        """Generate cryptographically secure nonce for CSP."""
        # Generate 16 random bytes and base64 encode them
        random_bytes = secrets.token_bytes(16)
        return base64.b64encode(random_bytes).decode("ascii")

    def get_secure_cookie_config(self) -> Dict[str, Any]:
        """Get secure cookie configuration."""
        return {
            "secure": self.policy.secure_cookies
            and self.policy.production_mode,
            "httponly": self.policy.httponly_cookies,
            "samesite": self.policy.samesite_cookies,
        }

    def is_websocket_path(self, path: str) -> bool:
        """Check if path is a WebSocket endpoint."""
        websocket_patterns = ["/ws/", "/websocket/", "/socket.io/"]
        return any(pattern in path for pattern in websocket_patterns)

    def get_security_headers(
        self, request: Request, response: Response
    ) -> Dict[str, str]:
        """Get all security headers for a request/response."""
        headers = {}

        # HSTS (only over HTTPS)
        if request.url.scheme == "https" or self.policy.production_mode:
            if hsts := self.generate_hsts_header():
                headers["Strict-Transport-Security"] = hsts

        # Content Security Policy with nonce
        nonce = None
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            nonce = self.generate_nonce()
            # Store nonce in response for template usage
            if hasattr(response, "context"):
                response.context["csp_nonce"] = nonce

        headers["Content-Security-Policy"] = self.generate_csp_header(nonce)

        # Frame Options
        headers["X-Frame-Options"] = self.policy.x_frame_options

        # Content Type Options
        headers["X-Content-Type-Options"] = self.policy.x_content_type_options

        # XSS Protection (legacy but still useful)
        headers["X-XSS-Protection"] = self.policy.x_xss_protection

        # Referrer Policy
        headers["Referrer-Policy"] = self.policy.referrer_policy

        # Permissions Policy
        headers["Permissions-Policy"] = self.generate_permissions_policy()

        # Expect-CT
        if expect_ct := self.generate_expect_ct_header():
            headers["Expect-CT"] = expect_ct

        # Additional security headers
        headers["X-Permitted-Cross-Domain-Policies"] = "none"
        headers["X-DNS-Prefetch-Control"] = "off"
        headers["X-Download-Options"] = "noopen"

        # Certificate Pinning (for mobile apps)
        if self.policy.enable_certificate_pinning:
            host = request.headers.get("host", "").split(":")[0]  # Remove port
            user_agent = request.headers.get("user-agent", "")

            # Use enhanced mobile certificate pinner
            from auth.certificate_pinning import mobile_cert_pinner

            if pin_header := mobile_cert_pinner.get_pinning_header(
                host, user_agent
            ):
                headers["Public-Key-Pins"] = pin_header

            # Fallback to legacy pinner
            elif pin_header := self.certificate_pinner.generate_header(host):
                headers["Public-Key-Pins"] = pin_header

        # Cache control based on endpoint type
        path = request.url.path
        if any(
            sensitive in path
            for sensitive in ["/auth/", "/api/", "/admin/", "/user/"]
        ):
            # Sensitive endpoints - no caching
            headers.update(
                {
                    "Cache-Control": self.policy.cache_control_sensitive,
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "X-Frame-Options": "DENY",  # Stricter for sensitive endpoints
                    "Clear-Site-Data": (
                        '"cache"' if "/logout" in path else None
                    ),  # Clear cache on logout
                }
            )
            # Remove None values
            headers = {k: v for k, v in headers.items() if v is not None}
        elif any(
            static in path for static in ["/static/", "/assets/", "/public/"]
        ):
            # Static assets - long cache
            headers["Cache-Control"] = self.policy.cache_control_static
        else:
            # Default - no store
            headers["Cache-Control"] = self.policy.cache_control_default

        return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for applying security headers to all responses."""

    def __init__(
        self, app, security_manager: Optional[SecurityHeadersManager] = None
    ):
        super().__init__(app)
        self.security_manager = security_manager or SecurityHeadersManager()

    async def dispatch(self, request: Request, call_next):
        """Apply security headers to response."""
        try:
            response = await call_next(request)

            # Apply security headers
            security_headers = self.security_manager.get_security_headers(
                request, response
            )

            for header_name, header_value in security_headers.items():
                response.headers[header_name] = header_value

            # Log security headers application
            logger.debug(
                f"Applied {len(security_headers)} security headers to {request.url.path}"
            )

            return response

        except Exception as e:
            # Ensure security headers are applied even during errors
            logger.error(f"Error in security headers middleware: {e}")

            # Create error response with security headers
            from fastapi.responses import JSONResponse

            error_response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )

            # Apply security headers to error response
            security_headers = self.security_manager.get_security_headers(
                request, error_response
            )
            for header_name, header_value in security_headers.items():
                error_response.headers[header_name] = header_value

            return error_response


def setup_security_headers(app, policy: Optional[SecurityPolicy] = None):
    """Convenience function to set up security headers middleware."""
    security_manager = SecurityHeadersManager(policy)
    app.add_middleware(
        SecurityHeadersMiddleware, security_manager=security_manager
    )

    logger.info("Security headers middleware configured successfully")
    return security_manager


# Production-ready certificate pins for common domains
PRODUCTION_CERTIFICATE_PINS = {
    "freeagentics.com": [
        "sha256-REPLACE_WITH_ACTUAL_PIN_FOR_PRODUCTION",
        "sha256-REPLACE_WITH_BACKUP_PIN_FOR_PRODUCTION",
    ],
    "api.freeagentics.com": [
        "sha256-REPLACE_WITH_ACTUAL_PIN_FOR_PRODUCTION",
        "sha256-REPLACE_WITH_BACKUP_PIN_FOR_PRODUCTION",
    ],
}

# Default security policy for production
PRODUCTION_SECURITY_POLICY = SecurityPolicy(
    enable_hsts=True,
    hsts_max_age=31536000,  # 1 year
    hsts_include_subdomains=True,
    hsts_preload=True,
    csp_report_uri="/api/security/csp-report",
    enable_expect_ct=True,
    expect_ct_enforce=True,
    expect_ct_report_uri="/api/security/ct-report",
    enable_certificate_pinning=True,
    production_mode=True,
)

# Global security headers manager for compatibility
_default_security_manager = SecurityHeadersManager()


def get_security_headers(custom_csp: Optional[str] = None) -> Dict[str, str]:
    """Get security headers dictionary (for testing compatibility)."""
    # Generate headers directly for better performance
    headers = {}

    # HSTS
    headers[
        "Strict-Transport-Security"
    ] = _default_security_manager.generate_hsts_header()

    # CSP
    if custom_csp:
        headers["Content-Security-Policy"] = custom_csp
    else:
        headers[
            "Content-Security-Policy"
        ] = _default_security_manager.generate_csp_header()

    # Frame Options
    headers[
        "X-Frame-Options"
    ] = _default_security_manager.policy.x_frame_options

    # Content Type Options
    headers[
        "X-Content-Type-Options"
    ] = _default_security_manager.policy.x_content_type_options

    # XSS Protection
    headers[
        "X-XSS-Protection"
    ] = _default_security_manager.policy.x_xss_protection

    # Referrer Policy
    headers[
        "Referrer-Policy"
    ] = _default_security_manager.policy.referrer_policy

    # Permissions Policy
    headers[
        "Permissions-Policy"
    ] = _default_security_manager.generate_permissions_policy()

    # Expect-CT
    if expect_ct := _default_security_manager.generate_expect_ct_header():
        headers["Expect-CT"] = expect_ct

    return headers


def add_security_headers(response) -> None:
    """Add security headers to a response (for testing compatibility)."""
    if not response or not hasattr(response, "headers"):
        return

    headers = get_security_headers()
    for header_name, header_value in headers.items():
        response.headers[header_name] = header_value


def validate_csp_header(csp_header: Optional[str]) -> bool:
    """Validate Content Security Policy header format."""
    if not csp_header:
        return False

    # Check for required directives
    required_directives = ["default-src", "script-src", "style-src"]

    for directive in required_directives:
        if directive not in csp_header:
            return False

    # Check for invalid directives (basic validation)
    invalid_patterns = ["invalid-directive", "unknown-src"]

    for pattern in invalid_patterns:
        if pattern in csp_header:
            return False

    return True


def validate_hsts_header(hsts_header: Optional[str]) -> bool:
    """Validate HTTP Strict Transport Security header format."""
    if not hsts_header:
        return False

    # Must contain max-age
    if "max-age=" not in hsts_header:
        return False

    # Extract max-age value
    import re

    max_age_match = re.search(r"max-age=(\d+)", hsts_header)
    if not max_age_match:
        return False

    try:
        max_age = int(max_age_match.group(1))
        # Must be at least 1 year (31536000 seconds)
        if max_age < 31536000:
            return False
    except ValueError:
        return False

    return True


def generate_csp_nonce() -> str:
    """Generate a CSP nonce for testing compatibility."""
    return _default_security_manager.generate_nonce()
