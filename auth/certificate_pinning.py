"""
Enhanced Certificate Pinning for Mobile Applications.

Provides comprehensive certificate pinning with fallback mechanisms,
mobile app support, and production-ready configuration management.
"""

import hashlib
import json
import logging
import os
import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


@dataclass
class PinConfiguration:
    """Configuration for certificate pinning."""

    # Primary pins (current certificates)
    primary_pins: List[str] = field(default_factory=list)

    # Backup pins (for certificate rotation)
    backup_pins: List[str] = field(default_factory=list)

    # Pin validation settings
    max_age: int = 5184000  # 60 days
    include_subdomains: bool = True
    enforce_pinning: bool = True

    # Fallback settings
    allow_fallback: bool = True
    fallback_timeout: int = 30  # seconds

    # Mobile app specific settings
    mobile_specific: bool = False
    mobile_user_agents: List[str] = field(
        default_factory=lambda: [
            "FreeAgentics-iOS",
            "FreeAgentics-Android",
            "FreeAgentics-Mobile",
        ]
    )

    # Report URI for pin failures
    report_uri: Optional[str] = None

    # Emergency pin bypass (for critical situations)
    emergency_bypass: bool = False
    emergency_bypass_until: Optional[datetime] = None


class CertificateValidator:
    """Certificate validation and pin extraction utilities."""

    @staticmethod
    def extract_spki_pin(certificate_pem: str) -> str:
        """Extract Subject Public Key Info (SPKI) pin from certificate."""
        try:
            cert = x509.load_pem_x509_certificate(certificate_pem.encode())
            public_key = cert.public_key()

            # Serialize the public key in DER format
            der_public_key = public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256(der_public_key).digest()

            # Return base64 encoded hash
            import base64

            return base64.b64encode(sha256_hash).decode("ascii")

        except Exception as e:
            logger.error(f"Failed to extract SPKI pin from certificate: {e}")
            raise

    @staticmethod
    def get_certificate_from_server(hostname: str, port: int = 443) -> str:
        """Retrieve certificate from server for pin generation."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    if cert_der is None:
                        raise ValueError(f"No certificate received from {hostname}:{port}")
                    cert = x509.load_der_x509_certificate(cert_der)

                    return cert.public_bytes(serialization.Encoding.PEM).decode()

        except Exception as e:
            logger.error(f"Failed to retrieve certificate from {hostname}:{port}: {e}")
            raise

    @staticmethod
    def validate_pin_format(pin: str) -> bool:
        """Validate that pin is in correct format."""
        if not pin.startswith("sha256-"):
            return False

        try:
            import base64

            pin_data = pin[7:]  # Remove 'sha256-' prefix
            decoded = base64.b64decode(pin_data)
            return len(decoded) == 32  # SHA256 hash length
        except Exception:
            return False

    @staticmethod
    def generate_backup_pin() -> str:
        """Generate a backup pin for certificate rotation."""
        # This would typically be generated from a backup certificate
        # For now, return a placeholder
        import base64
        import secrets

        backup_key = secrets.token_bytes(32)
        return f"sha256-{base64.b64encode(backup_key).decode('ascii')}"


class MobileCertificatePinner:
    """Enhanced certificate pinning with mobile app support."""

    def __init__(self) -> None:
        """Initialize mobile certificate pinner."""
        self.domain_configs: Dict[str, PinConfiguration] = {}
        self.pin_cache: Dict[str, Tuple[str, datetime]] = {}
        self.failure_count: Dict[str, int] = {}

        # Load configuration from environment and files
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load pinning configuration from environment and files."""
        # Load from environment variables
        self._load_env_configuration()

        # Load from configuration file if present
        config_file = os.getenv("CERT_PIN_CONFIG_FILE", "/app/config/cert_pins.json")
        if os.path.exists(config_file):
            self._load_file_configuration(config_file)

    def _load_env_configuration(self) -> None:
        """Load certificate pins from environment variables."""
        # Production domains
        production_domains = [
            "freeagentics.com",
            "api.freeagentics.com",
            "ws.freeagentics.com",
        ]

        for domain in production_domains:
            env_key = f"CERT_PIN_{domain.replace('.', '_').upper()}"
            backup_key = f"CERT_BACKUP_PIN_{domain.replace('.', '_').upper()}"

            primary_pin = os.getenv(env_key)
            backup_pin = os.getenv(backup_key)

            if primary_pin or backup_pin:
                config = PinConfiguration(
                    primary_pins=[primary_pin] if primary_pin else [],
                    backup_pins=[backup_pin] if backup_pin else [],
                    enforce_pinning=os.getenv("PRODUCTION", "false").lower() == "true",
                )
                self.domain_configs[domain] = config

    def _load_file_configuration(self, config_file: str) -> None:
        """Load certificate pins from JSON configuration file."""
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            for domain, domain_config in config_data.items():
                config = PinConfiguration(
                    primary_pins=domain_config.get("primary_pins", []),
                    backup_pins=domain_config.get("backup_pins", []),
                    max_age=domain_config.get("max_age", 5184000),
                    include_subdomains=domain_config.get("include_subdomains", True),
                    enforce_pinning=domain_config.get("enforce_pinning", True),
                    allow_fallback=domain_config.get("allow_fallback", True),
                    mobile_specific=domain_config.get("mobile_specific", False),
                    report_uri=domain_config.get("report_uri"),
                )
                self.domain_configs[domain] = config

            logger.info(f"Loaded certificate pinning configuration for {len(config_data)} domains")

        except Exception as e:
            logger.error(f"Failed to load certificate pinning configuration: {e}")

    def add_domain_pins(self, domain: str, config: PinConfiguration) -> None:
        """Add certificate pins for a domain."""
        # Validate pins
        for pin in config.primary_pins + config.backup_pins:
            if not CertificateValidator.validate_pin_format(pin):
                raise ValueError(f"Invalid pin format: {pin}")

        self.domain_configs[domain] = config
        logger.info(f"Added certificate pinning for domain: {domain}")

    def get_pinning_header(self, domain: str, user_agent: str = "") -> Optional[str]:
        """Generate Public-Key-Pins header for a domain."""
        config = self.domain_configs.get(domain)
        if not config:
            return None

        # Check if pinning is temporarily bypassed
        if config.emergency_bypass and config.emergency_bypass_until:
            if datetime.now() < config.emergency_bypass_until:
                logger.warning(
                    f"Certificate pinning bypassed for {domain} until "
                    f"{config.emergency_bypass_until}"
                )
                return None

        # Check if this is a mobile request and mobile-specific pinning is configured
        is_mobile = any(ua in user_agent for ua in config.mobile_user_agents)
        if config.mobile_specific and not is_mobile:
            return None

        # Build pin list
        pins = []
        for pin in config.primary_pins:
            pins.append(f'pin-sha256="{pin}"')
        for pin in config.backup_pins:
            pins.append(f'pin-sha256="{pin}"')

        if not pins:
            return None

        # Build header
        header_parts = pins + [f"max-age={config.max_age}"]

        if config.include_subdomains:
            header_parts.append("includeSubDomains")

        if config.report_uri:
            header_parts.append(f'report-uri="{config.report_uri}"')

        return "; ".join(header_parts)

    def validate_certificate_chain(self, domain: str, cert_chain: List[str]) -> bool:
        """Validate certificate chain against pins."""
        config = self.domain_configs.get(domain)
        if not config or not config.enforce_pinning:
            return True

        all_pins = config.primary_pins + config.backup_pins
        if not all_pins:
            return True

        # Extract pins from certificate chain
        chain_pins = []
        for cert_pem in cert_chain:
            try:
                pin = CertificateValidator.extract_spki_pin(cert_pem)
                chain_pins.append(f"sha256-{pin}")
            except Exception as e:
                logger.error(f"Failed to extract pin from certificate: {e}")

        # Check if any pin matches
        for chain_pin in chain_pins:
            if chain_pin in all_pins:
                logger.debug(f"Certificate pin validated for {domain}")
                return True

        # Pin validation failed
        self.failure_count[domain] = self.failure_count.get(domain, 0) + 1
        logger.error(
            f"Certificate pin validation failed for {domain} "
            f"(failure #{self.failure_count[domain]})"
        )

        # Report pin failure if configured
        if config.report_uri:
            self._report_pin_failure(domain, chain_pins, all_pins)

        return False

    def _report_pin_failure(
        self, domain: str, chain_pins: List[str], expected_pins: List[str]
    ) -> None:
        """Report certificate pin validation failure."""
        try:
            import requests

            report_data = {
                "date-time": datetime.now().isoformat(),
                "hostname": domain,
                "port": 443,
                "effective-expiration-date": (datetime.now() + timedelta(days=60)).isoformat(),
                "include-subdomains": self.domain_configs[domain].include_subdomains,
                "served-certificate-chain": chain_pins,
                "validated-certificate-chain": chain_pins,
                "known-pins": expected_pins,
            }

            config = self.domain_configs[domain]
            if config.report_uri:
                requests.post(config.report_uri, json=report_data, timeout=5)
                logger.info(f"Reported pin failure for {domain}")

        except Exception as e:
            logger.error(f"Failed to report pin failure: {e}")

    def emergency_bypass_domain(self, domain: str, duration_hours: int = 24) -> None:
        """Emergency bypass pinning for a domain."""
        config = self.domain_configs.get(domain)
        if config:
            config.emergency_bypass = True
            config.emergency_bypass_until = datetime.now() + timedelta(hours=duration_hours)
            logger.warning(
                f"Emergency bypass activated for {domain} until {config.emergency_bypass_until}"
            )

    def get_mobile_pinning_config(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get mobile app pinning configuration."""
        config = self.domain_configs.get(domain)
        if not config:
            return None

        return {
            "domain": domain,
            "pins": config.primary_pins + config.backup_pins,
            "includeSubdomains": config.include_subdomains,
            "maxAge": config.max_age,
            "enforcePinning": config.enforce_pinning,
            "allowFallback": config.allow_fallback,
            "reportUri": config.report_uri,
        }

    def update_pins_from_server(self, domain: str, port: int = 443) -> bool:
        """Update pins by fetching current certificate from server."""
        try:
            cert_pem = CertificateValidator.get_certificate_from_server(domain, port)
            new_pin = f"sha256-{CertificateValidator.extract_spki_pin(cert_pem)}"

            config = self.domain_configs.get(domain, PinConfiguration())

            # Move current primary to backup
            if config.primary_pins:
                config.backup_pins = config.primary_pins.copy()

            # Set new pin as primary
            config.primary_pins = [new_pin]

            self.domain_configs[domain] = config
            logger.info(f"Updated pins for {domain}: {new_pin}")

            return True

        except Exception as e:
            logger.error(f"Failed to update pins for {domain}: {e}")
            return False


# Global certificate pinner instance
mobile_cert_pinner = MobileCertificatePinner()

# Production pin configurations (replace with actual pins)
PRODUCTION_PIN_CONFIGS = {
    "freeagentics.com": PinConfiguration(
        primary_pins=["sha256-REPLACE_WITH_ACTUAL_PRIMARY_PIN"],
        backup_pins=["sha256-REPLACE_WITH_ACTUAL_BACKUP_PIN"],
        include_subdomains=True,
        enforce_pinning=True,
        mobile_specific=True,
        report_uri="/api/security/pin-report",
    ),
    "api.freeagentics.com": PinConfiguration(
        primary_pins=["sha256-REPLACE_WITH_ACTUAL_API_PRIMARY_PIN"],
        backup_pins=["sha256-REPLACE_WITH_ACTUAL_API_BACKUP_PIN"],
        include_subdomains=False,
        enforce_pinning=True,
        mobile_specific=True,
        report_uri="/api/security/pin-report",
    ),
}


def setup_production_pins() -> None:
    """Set up production certificate pins."""
    if os.getenv("PRODUCTION", "false").lower() == "true":
        for domain, config in PRODUCTION_PIN_CONFIGS.items():
            mobile_cert_pinner.add_domain_pins(domain, config)
        logger.info("Production certificate pinning configured")
