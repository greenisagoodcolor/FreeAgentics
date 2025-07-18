"""
Mutual TLS (mTLS) Certificate Manager for Zero-Trust Architecture.

This module provides comprehensive certificate management including:
- Certificate generation and signing
- Automatic certificate rotation
- Certificate validation and revocation
- Secure storage and retrieval
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtensionOID, NameOID

logger = logging.getLogger(__name__)


class RotationStrategy(str, Enum):
    """Certificate rotation strategies."""

    TIME_BASED = "time_based"
    ON_DEMAND = "on_demand"
    EVENT_TRIGGERED = "event_triggered"


@dataclass
class CertificateInfo:
    """Certificate information container."""

    service_name: str
    certificate: str  # PEM encoded
    private_key: str  # PEM encoded
    fingerprint: str  # SHA256 hex
    serial_number: int
    issued_at: datetime
    expires_at: datetime
    dns_names: List[str] = field(default_factory=list)
    is_ca: bool = False
    revoked: bool = False
    revocation_reason: Optional[str] = None


@dataclass
class CertificateRotationPolicy:
    """Policy for automatic certificate rotation."""

    strategy: RotationStrategy = RotationStrategy.TIME_BASED
    rotation_interval_days: int = 30
    overlap_period_days: int = 7  # Grace period with both certs valid
    auto_rotate: bool = True
    min_validity_days: int = 7  # Rotate if less than this many days remain


@dataclass
class RevocationEntry:
    """Certificate revocation list entry."""

    fingerprint: str
    serial_number: int
    revocation_time: datetime
    reason: str


class MTLSManager:
    """Manages mutual TLS certificates for zero-trust architecture."""

    def __init__(
        self,
        ca_cert_path: str = "./certs/ca-cert.pem",
        ca_key_path: str = "./certs/ca-key.pem",
        cert_store_path: str = "./certs/services",
        crl_path: str = "./certs/crl.json",
        key_size: int = 2048,  # Configurable key size
        enable_key_cache: bool = True,  # Enable key caching for performance
    ):
        self.ca_cert_path = Path(ca_cert_path)
        self.ca_key_path = Path(ca_key_path)
        self.cert_store_path = Path(cert_store_path)
        self.crl_path = Path(crl_path)
        self.key_size = key_size
        self.enable_key_cache = enable_key_cache

        # Thread safety
        self._lock = threading.RLock()

        # Certificate storage
        self._certificates: Dict[str, CertificateInfo] = {}
        self._rotation_policies: Dict[str, CertificateRotationPolicy] = {}
        self._revocation_list: Dict[str, RevocationEntry] = {}

        # Key cache for performance
        self._key_cache: List[rsa.RSAPrivateKey] = []
        self._key_cache_size = 10  # Pre-generate 10 keys

        # Initialize CA
        self.ca_cert, self.ca_key = self._initialize_ca()

        # Pre-generate keys if caching is enabled
        if self.enable_key_cache:
            self._initialize_key_cache()

        # Load existing certificates and CRL
        self._load_certificates()
        self._load_revocation_list()

    def _initialize_key_cache(self) -> None:
        """Pre-generate RSA keys for performance."""
        logger.info(
            f"Pre-generating {self._key_cache_size} RSA keys for cache"
        )
        for _ in range(self._key_cache_size):
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
            )
            self._key_cache.append(key)

    def _get_or_generate_key(self) -> rsa.RSAPrivateKey:
        """Get a key from cache or generate new one."""
        if self.enable_key_cache and self._key_cache:
            with self._lock:
                if self._key_cache:
                    # Refill cache in background if getting low
                    if len(self._key_cache) < 3:
                        threading.Thread(
                            target=self._refill_key_cache, daemon=True
                        ).start()
                    return self._key_cache.pop()

        # Generate new key if cache is empty or disabled
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
        )

    def _refill_key_cache(self) -> None:
        """Refill the key cache in background."""
        try:
            new_keys = []
            for _ in range(self._key_cache_size // 2):
                key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=self.key_size,
                )
                new_keys.append(key)

            with self._lock:
                self._key_cache.extend(new_keys)

        except Exception as e:
            logger.error(f"Failed to refill key cache: {e}")

    def _initialize_ca(self) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Initialize or load CA certificate."""
        if self.ca_cert_path.exists() and self.ca_key_path.exists():
            logger.info("Loading existing CA certificate")
            return self._load_ca()
        else:
            logger.info("Creating new CA certificate")
            return self._create_ca()

    def _load_ca(self) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Load existing CA certificate and key."""
        with open(self.ca_cert_path, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read())

        with open(self.ca_key_path, "rb") as f:
            ca_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

        return ca_cert, ca_key

    def _create_ca(self) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Create new CA certificate."""
        # Generate key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Stronger key for CA
        )

        # Certificate details
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(
                    NameOID.ORGANIZATION_NAME, "Zero Trust Network"
                ),
                x509.NameAttribute(NameOID.COMMON_NAME, "Zero Trust CA"),
            ]
        )

        # Create certificate
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(
                datetime.utcnow() + timedelta(days=3650)
            )  # 10 years
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Save CA certificate and key
        self.ca_cert_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.ca_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(self.ca_key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Set secure permissions
        os.chmod(self.ca_key_path, 0o600)

        return cert, private_key

    def generate_service_certificate(
        self,
        service_name: str,
        dns_names: Optional[List[str]] = None,
        validity_days: int = 30,
    ) -> CertificateInfo:
        """Generate a new service certificate."""
        with self._lock:
            # Get private key from cache or generate new one
            private_key = self._get_or_generate_key()

            # Default DNS names
            if dns_names is None:
                dns_names = [
                    service_name,
                    f"{service_name}.local",
                    f"{service_name}.cluster.local",
                ]

            # Certificate details
            subject = x509.Name(
                [
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                    x509.NameAttribute(
                        NameOID.ORGANIZATION_NAME, "Zero Trust Network"
                    ),
                    x509.NameAttribute(
                        NameOID.ORGANIZATIONAL_UNIT_NAME, "Services"
                    ),
                    x509.NameAttribute(NameOID.COMMON_NAME, service_name),
                ]
            )

            # Build certificate
            builder = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(self.ca_cert.subject)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(
                    datetime.utcnow() + timedelta(days=validity_days)
                )
            )

            # Add SAN extension
            san_list = [x509.DNSName(name) for name in dns_names]
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )

            # Add key usage extensions
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )

            builder = builder.add_extension(
                x509.ExtendedKeyUsage(
                    [
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=True,
            )

            # Sign certificate
            cert = builder.sign(self.ca_key, hashes.SHA256())

            # Convert to PEM
            cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode()

            # Calculate fingerprint
            fingerprint = hashlib.sha256(
                cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            # Create certificate info
            cert_info = CertificateInfo(
                service_name=service_name,
                certificate=cert_pem,
                private_key=key_pem,
                fingerprint=fingerprint,
                serial_number=cert.serial_number,
                issued_at=cert.not_valid_before,
                expires_at=cert.not_valid_after,
                dns_names=dns_names,
            )

            # Store certificate
            self._certificates[service_name] = cert_info
            self.save_certificate(cert_info)

            logger.info(
                f"Generated certificate for {service_name} (fingerprint: {fingerprint})"
            )

            return cert_info

    def validate_certificate(self, certificate_pem: str) -> Tuple[bool, str]:
        """Validate a certificate against the CA."""
        try:
            cert = x509.load_pem_x509_certificate(certificate_pem.encode())

            # Check if issued by our CA
            if cert.issuer != self.ca_cert.subject:
                return False, "Certificate not issued by trusted CA"

            # Check validity period
            now = datetime.utcnow()
            if now < cert.not_valid_before:
                return False, "Certificate not yet valid"
            if now > cert.not_valid_after:
                return False, "Certificate has expired"

            # Check revocation
            fingerprint = hashlib.sha256(
                cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            if fingerprint in self._revocation_list:
                entry = self._revocation_list[fingerprint]
                return False, f"Certificate revoked: {entry.reason}"

            # TODO: Implement proper signature verification
            # For now, basic checks pass

            return True, "Certificate is valid"

        except Exception as e:
            logger.error(f"Certificate validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def rotate_certificate(self, service_name: str) -> CertificateInfo:
        """Rotate a service certificate."""
        with self._lock:
            old_cert = self._certificates.get(service_name)
            if not old_cert:
                raise ValueError(f"No certificate found for {service_name}")

            # Generate new certificate with same DNS names
            new_cert = self.generate_service_certificate(
                service_name=service_name,
                dns_names=old_cert.dns_names,
            )

            # During overlap period, both certificates are valid
            # The old certificate will be naturally invalidated when it expires

            logger.info(f"Rotated certificate for {service_name}")
            logger.info(f"Old fingerprint: {old_cert.fingerprint}")
            logger.info(f"New fingerprint: {new_cert.fingerprint}")

            return new_cert

    def revoke_certificate(
        self, fingerprint: str, reason: str = "Unspecified"
    ) -> bool:
        """Revoke a certificate."""
        with self._lock:
            # Find certificate by fingerprint
            cert_info = None
            for info in self._certificates.values():
                if info.fingerprint == fingerprint:
                    cert_info = info
                    break

            if not cert_info:
                logger.warning(
                    f"Certificate not found for revocation: {fingerprint}"
                )
                return False

            # Add to revocation list
            revocation = RevocationEntry(
                fingerprint=fingerprint,
                serial_number=cert_info.serial_number,
                revocation_time=datetime.utcnow(),
                reason=reason,
            )

            self._revocation_list[fingerprint] = revocation
            cert_info.revoked = True
            cert_info.revocation_reason = reason

            # Save updated CRL
            self._save_revocation_list()

            logger.info(f"Revoked certificate {fingerprint}: {reason}")
            return True

    def set_rotation_policy(
        self, service_name: str, policy: CertificateRotationPolicy
    ) -> None:
        """Set rotation policy for a service."""
        with self._lock:
            self._rotation_policies[service_name] = policy
            logger.info(
                f"Set rotation policy for {service_name}: {policy.strategy.value}"
            )

    def _should_rotate_certificate(self, cert_info: CertificateInfo) -> bool:
        """Check if a certificate should be rotated."""
        policy = self._rotation_policies.get(cert_info.service_name)
        if not policy or not policy.auto_rotate:
            return False

        now = datetime.utcnow()
        days_until_expiry = (cert_info.expires_at - now).days

        if policy.strategy == RotationStrategy.TIME_BASED:
            # Check if certificate is older than rotation interval
            cert_age_days = (now - cert_info.issued_at).days
            if cert_age_days >= policy.rotation_interval_days:
                return True

        # Always rotate if approaching expiry
        if days_until_expiry <= policy.min_validity_days:
            return True

        return False

    def check_and_rotate_certificates(self) -> List[str]:
        """Check all certificates and rotate as needed."""
        rotated = []

        with self._lock:
            for service_name, cert_info in list(self._certificates.items()):
                if self._should_rotate_certificate(cert_info):
                    try:
                        self.rotate_certificate(service_name)
                        rotated.append(service_name)
                    except Exception as e:
                        logger.error(
                            f"Failed to rotate certificate for {service_name}: {e}"
                        )

        return rotated

    def save_certificate(self, cert_info: CertificateInfo) -> None:
        """Save certificate to disk."""
        self.cert_store_path.mkdir(parents=True, exist_ok=True)

        cert_dir = self.cert_store_path / cert_info.service_name
        cert_dir.mkdir(exist_ok=True)

        # Save certificate and key
        cert_file = cert_dir / "cert.pem"
        key_file = cert_dir / "key.pem"
        info_file = cert_dir / "info.json"

        with open(cert_file, "w") as f:
            f.write(cert_info.certificate)

        with open(key_file, "w") as f:
            f.write(cert_info.private_key)

        # Set secure permissions on private key
        os.chmod(key_file, 0o600)

        # Save metadata
        info_data = {
            "service_name": cert_info.service_name,
            "fingerprint": cert_info.fingerprint,
            "serial_number": cert_info.serial_number,
            "issued_at": cert_info.issued_at.isoformat(),
            "expires_at": cert_info.expires_at.isoformat(),
            "dns_names": cert_info.dns_names,
            "revoked": cert_info.revoked,
            "revocation_reason": cert_info.revocation_reason,
        }

        with open(info_file, "w") as f:
            json.dump(info_data, f, indent=2)

    def load_certificate(self, service_name: str) -> Optional[CertificateInfo]:
        """Load certificate from disk."""
        cert_dir = self.cert_store_path / service_name
        if not cert_dir.exists():
            return None

        try:
            # Load files
            with open(cert_dir / "cert.pem") as f:
                cert_pem = f.read()

            with open(cert_dir / "key.pem") as f:
                key_pem = f.read()

            with open(cert_dir / "info.json") as f:
                info_data = json.load(f)

            # Create certificate info
            cert_info = CertificateInfo(
                service_name=info_data["service_name"],
                certificate=cert_pem,
                private_key=key_pem,
                fingerprint=info_data["fingerprint"],
                serial_number=info_data["serial_number"],
                issued_at=datetime.fromisoformat(info_data["issued_at"]),
                expires_at=datetime.fromisoformat(info_data["expires_at"]),
                dns_names=info_data["dns_names"],
                revoked=info_data.get("revoked", False),
                revocation_reason=info_data.get("revocation_reason"),
            )

            return cert_info

        except Exception as e:
            logger.error(f"Failed to load certificate for {service_name}: {e}")
            return None

    def _load_certificates(self) -> None:
        """Load all certificates from disk."""
        if not self.cert_store_path.exists():
            return

        for service_dir in self.cert_store_path.iterdir():
            if service_dir.is_dir():
                cert_info = self.load_certificate(service_dir.name)
                if cert_info:
                    self._certificates[service_dir.name] = cert_info
                    logger.info(f"Loaded certificate for {service_dir.name}")

    def _load_revocation_list(self) -> None:
        """Load certificate revocation list from disk."""
        if not self.crl_path.exists():
            return

        try:
            with open(self.crl_path) as f:
                crl_data = json.load(f)

            for fingerprint, entry_data in crl_data.items():
                self._revocation_list[fingerprint] = RevocationEntry(
                    fingerprint=fingerprint,
                    serial_number=entry_data["serial_number"],
                    revocation_time=datetime.fromisoformat(
                        entry_data["revocation_time"]
                    ),
                    reason=entry_data["reason"],
                )

            logger.info(
                f"Loaded {len(self._revocation_list)} revoked certificates"
            )

        except Exception as e:
            logger.error(f"Failed to load CRL: {e}")

    def _save_revocation_list(self) -> None:
        """Save certificate revocation list to disk."""
        crl_data = {}

        for fingerprint, entry in self._revocation_list.items():
            crl_data[fingerprint] = {
                "serial_number": entry.serial_number,
                "revocation_time": entry.revocation_time.isoformat(),
                "reason": entry.reason,
            }

        self.crl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.crl_path, "w") as f:
            json.dump(crl_data, f, indent=2)

    def get_ca_certificate(self) -> str:
        """Get CA certificate in PEM format."""
        return self.ca_cert.public_bytes(serialization.Encoding.PEM).decode()

    def get_certificate_bundle(
        self, service_name: str
    ) -> Optional[Dict[str, str]]:
        """Get certificate bundle for a service."""
        cert_info = self._certificates.get(service_name)
        if not cert_info:
            return None

        return {
            "certificate": cert_info.certificate,
            "private_key": cert_info.private_key,
            "ca_certificate": self.get_ca_certificate(),
            "fingerprint": cert_info.fingerprint,
        }
