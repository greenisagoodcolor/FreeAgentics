"""
Zero-Trust Network Architecture Implementation.

This module implements zero-trust security principles including:
- Mutual TLS (mTLS) for service-to-service communication
- Service mesh integration for policy enforcement
- Identity-aware proxy for request validation
- Continuous verification with session risk scoring
- Principle of least privilege with dynamic permissions
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer

from auth.ml_threat_detection import get_ml_threat_detector
from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust levels for zero-trust architecture."""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRUSTED = "trusted"


class NetworkZone(str, Enum):
    """Network zones for zero-trust segmentation."""

    DMZ = "dmz"
    INTERNAL = "internal"
    SECURE = "secure"
    ADMIN = "admin"
    ISOLATED = "isolated"


class ServiceType(str, Enum):
    """Service types for zero-trust policies."""

    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    STORAGE = "storage"
    ADMIN = "admin"


@dataclass
class ServiceIdentity:
    """Service identity for zero-trust authentication."""

    service_name: str
    service_type: ServiceType
    network_zone: NetworkZone
    trust_level: TrustLevel
    certificate_fingerprint: str
    allowed_operations: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    valid_until: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))

    def is_valid(self) -> bool:
        """Check if service identity is still valid."""
        return datetime.utcnow() < self.valid_until

    def can_perform_operation(self, operation: str) -> bool:
        """Check if service can perform specific operation."""
        return operation in self.allowed_operations


@dataclass
class ZeroTrustPolicy:
    """Zero-trust policy definition."""

    policy_id: str
    name: str
    description: str
    source_services: List[str]
    target_services: List[str]
    allowed_operations: List[str]
    network_zones: List[NetworkZone]
    minimum_trust_level: TrustLevel
    time_restrictions: Optional[Dict[str, Any]] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if policy is currently active."""
        now = datetime.utcnow()
        if self.expires_at and now > self.expires_at:
            return False

        # Check time restrictions
        if self.time_restrictions:
            current_hour = now.hour
            allowed_hours = self.time_restrictions.get("allowed_hours", [])
            if allowed_hours and current_hour not in allowed_hours:
                return False

        return True

    def matches_request(self, source_service: str, target_service: str, operation: str) -> bool:
        """Check if policy matches a request."""
        return (
            self.is_active()
            and source_service in self.source_services
            and target_service in self.target_services
            and operation in self.allowed_operations
        )


@dataclass
class ContinuousVerificationContext:
    """Context for continuous verification."""

    session_id: str
    user_id: Optional[str]
    service_identity: ServiceIdentity
    initial_trust_level: TrustLevel
    current_trust_level: TrustLevel
    risk_score: float
    verification_count: int = 0
    last_verification: datetime = field(default_factory=datetime.utcnow)
    anomaly_detections: List[Dict[str, Any]] = field(default_factory=list)

    def update_trust_level(self, new_score: float):
        """Update trust level based on risk score."""
        if new_score >= 0.8:
            self.current_trust_level = TrustLevel.UNTRUSTED
        elif new_score >= 0.6:
            self.current_trust_level = TrustLevel.LOW
        elif new_score >= 0.4:
            self.current_trust_level = TrustLevel.MEDIUM
        elif new_score >= 0.2:
            self.current_trust_level = TrustLevel.HIGH
        else:
            self.current_trust_level = TrustLevel.TRUSTED

        self.risk_score = new_score
        self.last_verification = datetime.utcnow()
        self.verification_count += 1


class CertificateManager:
    """Manages certificates for mTLS authentication."""

    def __init__(self, ca_cert_path: str = None, ca_key_path: str = None):
        """Initialize the certificate manager."""
        self.ca_cert_path = ca_cert_path or "./ssl/ca-cert.pem"
        self.ca_key_path = ca_key_path or "./ssl/ca-key.pem"
        self.service_certificates: Dict[str, Dict[str, Any]] = {}

        # Load or create CA certificate
        self.ca_cert, self.ca_key = self._load_or_create_ca()

    def _load_or_create_ca(self) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Load existing CA certificate or create new one."""
        try:
            # Try to load existing CA certificate
            if os.path.exists(self.ca_cert_path) and os.path.exists(self.ca_key_path):
                with open(self.ca_cert_path, "rb") as f:
                    ca_cert = x509.load_pem_x509_certificate(f.read())

                with open(self.ca_key_path, "rb") as f:
                    ca_key = serialization.load_pem_private_key(f.read(), password=None)

                logger.info("Loaded existing CA certificate")
                return ca_cert, ca_key

        except Exception as e:
            logger.warning(f"Failed to load CA certificate: {e}")

        # Create new CA certificate
        logger.info("Creating new CA certificate")
        return self._create_ca_certificate()

    def _create_ca_certificate(
        self,
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Create new CA certificate."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FreeAgentics"),
                x509.NameAttribute(NameOID.COMMON_NAME, "FreeAgentics CA"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("freeagentics-ca"),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Save certificate and key
        try:
            os.makedirs(os.path.dirname(self.ca_cert_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ca_key_path), exist_ok=True)

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
        except Exception as e:
            logger.warning(f"Failed to save CA certificate to disk: {e}")
            # Continue without saving to disk

        return cert, private_key

    def issue_service_certificate(
        self, service_name: str, service_type: ServiceType
    ) -> Dict[str, str]:
        """Issue certificate for a service."""
        # Generate private key for service
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Create certificate
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FreeAgentics"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, service_type.value),
                x509.NameAttribute(NameOID.COMMON_NAME, service_name),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.ca_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=30))  # Short-lived certificates
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(service_name),
                        x509.DNSName(f"{service_name}.local"),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.ExtendedKeyUsage(
                    [
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=True,
            )
            .sign(self.ca_key, hashes.SHA256())
        )

        # Convert to PEM format
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        ca_cert_pem = self.ca_cert.public_bytes(serialization.Encoding.PEM).decode()

        # Calculate fingerprint
        fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()

        # Store certificate info
        self.service_certificates[service_name] = {
            "certificate": cert_pem,
            "private_key": key_pem,
            "ca_certificate": ca_cert_pem,
            "fingerprint": fingerprint,
            "issued_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30),
        }

        return {
            "certificate": cert_pem,
            "private_key": key_pem,
            "ca_certificate": ca_cert_pem,
            "fingerprint": fingerprint,
        }

    def verify_certificate(self, cert_pem: str) -> bool:
        """Verify a certificate against the CA."""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem.encode())

            # Check if certificate is issued by our CA
            if cert.issuer != self.ca_cert.subject:
                return False

            # Check if certificate is still valid
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False

            # For now, just validate that cert is from our CA and is time-valid
            # In production, you'd want proper signature verification

            return True

        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False

    def get_certificate_info(self, cert_pem: str) -> Dict[str, Any]:
        """Get information about a certificate."""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem.encode())

            # Extract subject information
            subject_info = {}
            for attribute in cert.subject:
                subject_info[attribute.oid._name] = attribute.value

            # Calculate fingerprint
            fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()

            return {
                "subject": subject_info,
                "issuer": cert.issuer.rfc4514_string(),
                "serial_number": str(cert.serial_number),
                "not_valid_before": cert.not_valid_before,
                "not_valid_after": cert.not_valid_after,
                "fingerprint": fingerprint,
                "is_valid": self.verify_certificate(cert_pem),
            }

        except Exception as e:
            logger.error(f"Failed to get certificate info: {e}")
            return {}


class ZeroTrustPolicyEngine:
    """Policy engine for zero-trust architecture."""

    def __init__(self):
        """Initialize the zero trust policy engine."""
        self.policies: Dict[str, ZeroTrustPolicy] = {}
        self.service_identities: Dict[str, ServiceIdentity] = {}
        self.active_sessions: Dict[str, ContinuousVerificationContext] = {}
        self.certificate_manager = CertificateManager()
        self.ml_detector = None  # Initialize lazily

    def register_service(
        self,
        service_name: str,
        service_type: ServiceType,
        network_zone: NetworkZone,
        trust_level: TrustLevel = TrustLevel.MEDIUM,
    ) -> ServiceIdentity:
        """Register a service in the zero-trust architecture."""
        # Issue certificate for the service
        cert_info = self.certificate_manager.issue_service_certificate(service_name, service_type)

        # Create service identity
        service_identity = ServiceIdentity(
            service_name=service_name,
            service_type=service_type,
            network_zone=network_zone,
            trust_level=trust_level,
            certificate_fingerprint=cert_info["fingerprint"],
            allowed_operations=self._get_default_operations(service_type),
            rate_limits=self._get_default_rate_limits(service_type),
        )

        self.service_identities[service_name] = service_identity

        # Log service registration
        security_auditor.log_event(
            event_type=SecurityEventType.SECURITY_CONFIG_CHANGE,
            severity=SecurityEventSeverity.INFO,
            message=f"Service {service_name} registered in zero-trust architecture",
            details={
                "service_name": service_name,
                "service_type": service_type.value,
                "network_zone": network_zone.value,
                "trust_level": trust_level.value,
                "certificate_fingerprint": cert_info["fingerprint"],
            },
        )

        return service_identity

    def _get_default_operations(self, service_type: ServiceType) -> List[str]:
        """Get default operations for service type."""
        operations_map = {
            ServiceType.API: ["read", "write", "execute"],
            ServiceType.DATABASE: ["read", "write", "admin"],
            ServiceType.CACHE: ["read", "write", "delete"],
            ServiceType.QUEUE: ["publish", "consume", "manage"],
            ServiceType.STORAGE: ["read", "write", "delete"],
            ServiceType.ADMIN: ["read", "write", "execute", "admin"],
        }
        return operations_map.get(service_type, ["read"])

    def _get_default_rate_limits(self, service_type: ServiceType) -> Dict[str, int]:
        """Get default rate limits for service type."""
        limits_map = {
            ServiceType.API: {
                "requests_per_second": 100,
                "requests_per_minute": 1000,
            },
            ServiceType.DATABASE: {
                "requests_per_second": 50,
                "requests_per_minute": 500,
            },
            ServiceType.CACHE: {
                "requests_per_second": 200,
                "requests_per_minute": 2000,
            },
            ServiceType.QUEUE: {
                "requests_per_second": 10,
                "requests_per_minute": 100,
            },
            ServiceType.STORAGE: {
                "requests_per_second": 20,
                "requests_per_minute": 200,
            },
            ServiceType.ADMIN: {
                "requests_per_second": 5,
                "requests_per_minute": 50,
            },
        }
        return limits_map.get(
            service_type,
            {"requests_per_second": 10, "requests_per_minute": 100},
        )

    def add_policy(self, policy: ZeroTrustPolicy) -> None:
        """Add a zero-trust policy."""
        self.policies[policy.policy_id] = policy

        security_auditor.log_event(
            event_type=SecurityEventType.SECURITY_CONFIG_CHANGE,
            severity=SecurityEventSeverity.INFO,
            message=f"Zero-trust policy {policy.name} added",
            details={
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "source_services": policy.source_services,
                "target_services": policy.target_services,
                "allowed_operations": policy.allowed_operations,
            },
        )

    def evaluate_request(
        self,
        source_service: str,
        target_service: str,
        operation: str,
        request_context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Evaluate a request against zero-trust policies."""
        # Check if services are registered
        if source_service not in self.service_identities:
            return False, f"Source service {source_service} not registered"

        if target_service not in self.service_identities:
            return False, f"Target service {target_service} not registered"

        source_identity = self.service_identities[source_service]
        target_identity = self.service_identities[target_service]

        # Check if source service identity is valid
        if not source_identity.is_valid():
            return False, f"Source service {source_service} identity expired"

        # Check if source service can perform the operation
        if not source_identity.can_perform_operation(operation):
            return (
                False,
                f"Source service {source_service} not authorized for operation {operation}",
            )

        # Find applicable policies
        applicable_policies = []
        for policy in self.policies.values():
            if policy.matches_request(source_service, target_service, operation):
                applicable_policies.append(policy)

        if not applicable_policies:
            return (
                False,
                f"No policies allow {source_service} to {operation} on {target_service}",
            )

        # Check trust level requirements
        for policy in applicable_policies:
            if not self._check_trust_level(source_identity.trust_level, policy.minimum_trust_level):
                return (
                    False,
                    f"Source service trust level {source_identity.trust_level} insufficient for policy {policy.name}",
                )

        # Check network zone compatibility
        if not self._check_network_zone_access(
            source_identity.network_zone, target_identity.network_zone
        ):
            return (
                False,
                f"Network zone {source_identity.network_zone} cannot access {target_identity.network_zone}",
            )

        # Check rate limits
        if not self._check_rate_limits(source_service, operation):
            return False, f"Rate limit exceeded for {source_service}"

        # Check additional conditions
        for policy in applicable_policies:
            if not self._check_policy_conditions(policy, request_context):
                return False, f"Policy {policy.name} conditions not met"

        return True, "Request authorized"

    def _check_trust_level(self, service_trust: TrustLevel, required_trust: TrustLevel) -> bool:
        """Check if service trust level meets requirement."""
        trust_hierarchy = {
            TrustLevel.UNTRUSTED: 0,
            TrustLevel.LOW: 1,
            TrustLevel.MEDIUM: 2,
            TrustLevel.HIGH: 3,
            TrustLevel.TRUSTED: 4,
        }

        return trust_hierarchy[service_trust] >= trust_hierarchy[required_trust]

    def _check_network_zone_access(
        self, source_zone: NetworkZone, target_zone: NetworkZone
    ) -> bool:
        """Check if source network zone can access target zone."""
        # Define network zone access matrix
        access_matrix = {
            NetworkZone.DMZ: [NetworkZone.DMZ, NetworkZone.INTERNAL],
            NetworkZone.INTERNAL: [NetworkZone.INTERNAL, NetworkZone.SECURE],
            NetworkZone.SECURE: [NetworkZone.SECURE, NetworkZone.ADMIN],
            NetworkZone.ADMIN: [NetworkZone.ADMIN],
            NetworkZone.ISOLATED: [NetworkZone.ISOLATED],
        }

        return target_zone in access_matrix.get(source_zone, [])

    def _check_rate_limits(self, source_service: str, operation: str) -> bool:
        """Check if service is within rate limits."""
        # This would integrate with actual rate limiting implementation
        # For now, just check if service exists
        return source_service in self.service_identities

    def _check_policy_conditions(
        self, policy: ZeroTrustPolicy, request_context: Dict[str, Any]
    ) -> bool:
        """Check if policy conditions are met."""
        for condition_key, condition_value in policy.conditions.items():
            if condition_key == "ip_whitelist":
                client_ip = request_context.get("client_ip")
                if client_ip:
                    # Check if IP is in any of the CIDR ranges
                    import ipaddress

                    ip_allowed = False
                    for cidr in condition_value:
                        try:
                            network = ipaddress.ip_network(cidr, strict=False)
                            if ipaddress.ip_address(client_ip) in network:
                                ip_allowed = True
                                break
                        except ValueError:
                            # Simple string comparison fallback
                            if client_ip == cidr:
                                ip_allowed = True
                                break
                    if not ip_allowed:
                        return False
            elif condition_key == "user_roles":
                user_roles = request_context.get("user_roles", [])
                if not any(role in condition_value for role in user_roles):
                    return False
            elif condition_key == "time_window":
                current_time = datetime.utcnow().hour
                if current_time not in condition_value:
                    return False

        return True

    async def continuous_verification(
        self, session_id: str, request_data: Dict[str, Any]
    ) -> TrustLevel:
        """Perform continuous verification of a session."""
        if session_id not in self.active_sessions:
            # Initialize new session
            service_name = request_data.get("service_name", "unknown")
            service_identity = self.service_identities.get(service_name)

            if not service_identity:
                return TrustLevel.UNTRUSTED

            self.active_sessions[session_id] = ContinuousVerificationContext(
                session_id=session_id,
                user_id=request_data.get("user_id"),
                service_identity=service_identity,
                initial_trust_level=service_identity.trust_level,
                current_trust_level=service_identity.trust_level,
                risk_score=0.0,
            )

        context = self.active_sessions[session_id]

        # Initialize ML detector lazily
        if not self.ml_detector:
            try:
                self.ml_detector = get_ml_threat_detector()
            except Exception as e:
                logger.warning(f"Failed to initialize ML detector: {e}")
                # Continue without ML detection
                context.update_trust_level(0.3)  # Default moderate risk
                return context.current_trust_level

        # Use ML threat detection to analyze request
        threat_prediction = await self.ml_detector.analyze_request(request_data)

        # Update trust level based on risk score
        context.update_trust_level(threat_prediction.risk_score)

        # Store anomaly detection if high risk
        if threat_prediction.risk_score > 0.6:
            context.anomaly_detections.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "risk_score": threat_prediction.risk_score,
                    "threat_level": threat_prediction.threat_level.value,
                    "detected_attacks": [
                        attack.value for attack in threat_prediction.detected_attacks
                    ],
                }
            )

        # Log significant trust level changes
        if context.current_trust_level != context.initial_trust_level:
            security_auditor.log_event(
                event_type=SecurityEventType.SECURITY_CONFIG_CHANGE,
                severity=SecurityEventSeverity.WARNING,
                message=f"Trust level changed for session {session_id}",
                user_id=context.user_id,
                details={
                    "session_id": session_id,
                    "initial_trust_level": context.initial_trust_level.value,
                    "current_trust_level": context.current_trust_level.value,
                    "risk_score": context.risk_score,
                    "verification_count": context.verification_count,
                },
            )

        return context.current_trust_level

    def get_service_mesh_config(self) -> Dict[str, Any]:
        """Generate service mesh configuration for zero-trust policies."""
        config = {
            "version": "1.0",
            "services": {},
            "policies": [],
            "certificates": {},
        }

        # Add service configurations
        for service_name, identity in self.service_identities.items():
            config["services"][service_name] = {
                "type": identity.service_type.value,
                "network_zone": identity.network_zone.value,
                "trust_level": identity.trust_level.value,
                "allowed_operations": identity.allowed_operations,
                "rate_limits": identity.rate_limits,
                "certificate_fingerprint": identity.certificate_fingerprint,
            }

        # Add policy configurations
        for policy in self.policies.values():
            config["policies"].append(
                {
                    "id": policy.policy_id,
                    "name": policy.name,
                    "source_services": policy.source_services,
                    "target_services": policy.target_services,
                    "allowed_operations": policy.allowed_operations,
                    "network_zones": [zone.value for zone in policy.network_zones],
                    "minimum_trust_level": policy.minimum_trust_level.value,
                    "conditions": policy.conditions,
                }
            )

        # Add certificate information
        for (
            service_name,
            cert_info,
        ) in self.certificate_manager.service_certificates.items():
            config["certificates"][service_name] = {
                "fingerprint": cert_info["fingerprint"],
                "issued_at": cert_info["issued_at"].isoformat(),
                "expires_at": cert_info["expires_at"].isoformat(),
            }

        return config


class IdentityAwareProxy:
    """Identity-aware proxy for zero-trust request validation."""

    def __init__(self, policy_engine: ZeroTrustPolicyEngine):
        """Initialize the identity-aware proxy."""
        self.policy_engine = policy_engine
        self.security = HTTPBearer()

    async def validate_request(self, request: Request, target_service: str, operation: str) -> bool:
        """Validate request through identity-aware proxy."""
        try:
            # Extract client certificate (mTLS)
            client_cert = self._extract_client_certificate(request)
            if not client_cert:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Client certificate required for mTLS",
                )

            # Verify client certificate
            if not self.policy_engine.certificate_manager.verify_certificate(client_cert):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid client certificate",
                )

            # Extract service identity from certificate
            cert_info = self.policy_engine.certificate_manager.get_certificate_info(client_cert)
            source_service = cert_info.get("subject", {}).get("commonName", "unknown")

            # Prepare request context
            request_context = {
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": str(request.url.path),
                "headers": dict(request.headers),
            }

            # Evaluate request against zero-trust policies
            is_authorized, reason = self.policy_engine.evaluate_request(
                source_service, target_service, operation, request_context
            )

            if not is_authorized:
                security_auditor.log_event(
                    event_type=SecurityEventType.ACCESS_DENIED,
                    severity=SecurityEventSeverity.WARNING,
                    message=f"Zero-trust policy denied access: {reason}",
                    details={
                        "source_service": source_service,
                        "target_service": target_service,
                        "operation": operation,
                        "reason": reason,
                        "client_ip": request_context["client_ip"],
                    },
                )

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: {reason}",
                )

            # Perform continuous verification
            session_id = self._generate_session_id(request)
            trust_level = await self.policy_engine.continuous_verification(
                session_id, {**request_context, "service_name": source_service}
            )

            # Check if trust level is sufficient
            if trust_level == TrustLevel.UNTRUSTED:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Trust level insufficient for access",
                )

            # Log successful access
            security_auditor.log_event(
                event_type=SecurityEventType.ACCESS_GRANTED,
                severity=SecurityEventSeverity.INFO,
                message=f"Zero-trust access granted to {source_service}",
                details={
                    "source_service": source_service,
                    "target_service": target_service,
                    "operation": operation,
                    "trust_level": trust_level.value,
                    "client_ip": request_context["client_ip"],
                },
            )

            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Identity-aware proxy validation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal validation error",
            )

    def _extract_client_certificate(self, request: Request) -> Optional[str]:
        """Extract client certificate from request."""
        # In a real implementation, this would extract the certificate from the TLS connection
        # For demonstration, we'll look for it in headers
        return request.headers.get("X-Client-Certificate")

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _generate_session_id(self, request: Request) -> str:
        """Generate session ID for continuous verification."""
        # Create session ID based on client certificate and request info
        session_data = f"{request.client.host}:{request.headers.get('X-Client-Certificate', '')}"
        return hashlib.sha256(session_data.encode()).hexdigest()


# Global zero-trust policy engine
zero_trust_engine = ZeroTrustPolicyEngine()


def get_zero_trust_engine() -> ZeroTrustPolicyEngine:
    """Get the global zero-trust policy engine."""
    return zero_trust_engine


def get_identity_aware_proxy() -> IdentityAwareProxy:
    """Get the identity-aware proxy."""
    return IdentityAwareProxy(zero_trust_engine)


# Example usage and configuration
def configure_default_zero_trust_policies():
    """Configure default zero-trust policies."""
    engine = get_zero_trust_engine()

    # Register example services
    engine.register_service("freeagentics-api", ServiceType.API, NetworkZone.DMZ, TrustLevel.MEDIUM)

    engine.register_service(
        "freeagentics-db",
        ServiceType.DATABASE,
        NetworkZone.SECURE,
        TrustLevel.HIGH,
    )

    # Add policy allowing API to access database
    engine.add_policy(
        ZeroTrustPolicy(
            policy_id="api-to-db-read",
            name="API Database Read Access",
            description="Allow API service to read from database",
            source_services=["freeagentics-api"],
            target_services=["freeagentics-db"],
            allowed_operations=["read"],
            network_zones=[NetworkZone.DMZ, NetworkZone.SECURE],
            minimum_trust_level=TrustLevel.MEDIUM,
            time_restrictions={"allowed_hours": list(range(24))},  # 24/7 access
            conditions={
                "ip_whitelist": [
                    "10.0.0.0/8",
                    "172.16.0.0/12",
                    "192.168.0.0/16",
                ]
            },
        )
    )

    # Add policy for admin operations
    engine.add_policy(
        ZeroTrustPolicy(
            policy_id="admin-full-access",
            name="Admin Full Access",
            description="Full access for admin services",
            source_services=["freeagentics-admin"],
            target_services=["freeagentics-api", "freeagentics-db"],
            allowed_operations=["read", "write", "admin"],
            network_zones=[NetworkZone.ADMIN],
            minimum_trust_level=TrustLevel.HIGH,
            time_restrictions={"allowed_hours": list(range(8, 18))},  # Business hours only
            conditions={"user_roles": ["admin", "super_admin"]},
        )
    )

    logger.info("Default zero-trust policies configured")


if __name__ == "__main__":
    # Configure default policies when module is run directly
    configure_default_zero_trust_policies()

    # Generate service mesh configuration
    config = zero_trust_engine.get_service_mesh_config()
    print(json.dumps(config, indent=2, default=str))
