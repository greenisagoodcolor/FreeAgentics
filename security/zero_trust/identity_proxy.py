"""
Identity-Aware Proxy for Zero-Trust Architecture.

This module implements an identity-aware proxy that:
- Validates requests at every hop with mTLS
- Performs dynamic permission evaluation
- Implements session risk scoring
- Provides continuous verification
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, Request, status

from .mtls_manager import MTLSManager

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Configuration for identity-aware proxy."""

    enable_mtls: bool = True
    enable_risk_scoring: bool = True
    max_risk_score: float = 0.7
    session_timeout_minutes: int = 30
    verification_interval_seconds: int = 60
    max_failed_attempts: int = 5
    mtls_manager: Optional[MTLSManager] = None


@dataclass
class ServicePolicy:
    """Service-to-service access policy."""

    source_service: str
    target_service: str
    allowed_operations: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    mtls_required: bool = True
    min_risk_score: float = 0.0
    max_risk_score: float = 0.7


@dataclass
class RequestContext:
    """Context for validated request."""

    source_service: str
    target_service: str
    operation: str
    client_ip: str
    user_agent: str
    timestamp: datetime
    certificate_fingerprint: Optional[str] = None
    is_valid: bool = False
    mtls_verified: bool = False
    risk_score: float = 0.0
    session_id: Optional[str] = None


@dataclass
class SessionRiskScore:
    """Risk score for a session."""

    session_id: str
    score: float
    factors: Dict[str, float]
    timestamp: datetime
    requires_reauthentication: bool = False


@dataclass
class VerificationSession:
    """Continuous verification session."""

    session_id: str
    service_name: str
    start_time: datetime
    last_verification: datetime
    verification_count: int = 0
    risk_scores: List[SessionRiskScore] = field(default_factory=list)
    active: bool = True
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


class IdentityAwareProxy:
    """Identity-aware proxy for zero-trust request validation."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.mtls_manager = config.mtls_manager or MTLSManager()

        # Policy storage
        self._policies: Dict[str, ServicePolicy] = {}
        self._policy_index: Dict[Tuple[str, str], List[str]] = {}

        # Session management
        self._sessions: Dict[str, VerificationSession] = {}
        self._session_lock = asyncio.Lock()

        # Risk scoring cache
        self._risk_cache: Dict[str, SessionRiskScore] = {}
        self._risk_cache_ttl = 300  # 5 minutes

        # Failed attempt tracking
        self._failed_attempts: Dict[str, int] = {}

        # Start background tasks
        self._verification_task = None

    async def validate_request(
        self,
        request: Request,
        source_service: str,
        target_service: str,
        operation: str,
    ) -> RequestContext:
        """Validate request through identity-aware proxy."""
        start_time = time.time()

        try:
            # Extract request metadata
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")

            # Create initial context
            context = RequestContext(
                source_service=source_service,
                target_service=target_service,
                operation=operation,
                client_ip=client_ip,
                user_agent=user_agent,
                timestamp=datetime.utcnow(),
            )

            # Validate mTLS if enabled
            if self.config.enable_mtls:
                cert_valid, cert_fingerprint = await self._validate_mtls(
                    request, source_service
                )
                context.mtls_verified = cert_valid
                context.certificate_fingerprint = cert_fingerprint

                if not cert_valid:
                    context.is_valid = False
                    self._track_failed_attempt(client_ip)
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="mTLS authentication failed",
                    )

            # Check service policies
            if not await self._check_service_policy(context):
                context.is_valid = False
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: {source_service} -> {target_service}:{operation}",
                )

            # Calculate risk score if enabled
            if self.config.enable_risk_scoring:
                session_id = self._generate_session_id(request, source_service)
                context.session_id = session_id

                risk_score = await self.calculate_session_risk(
                    session_id=session_id,
                    factors=self._extract_risk_factors(request, context),
                )
                context.risk_score = risk_score.score

                if risk_score.score > self.config.max_risk_score:
                    context.is_valid = False
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Risk score too high: {risk_score.score:.2f}",
                    )

            # All checks passed
            context.is_valid = True

            # Track successful request
            if client_ip in self._failed_attempts:
                del self._failed_attempts[client_ip]

            # Log validation time
            validation_time_ms = (time.time() - start_time) * 1000
            if validation_time_ms > 10:
                logger.warning(
                    f"Request validation took {validation_time_ms:.2f}ms"
                )

            return context

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal validation error",
            )

    async def _validate_mtls(
        self, request: Request, expected_service: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate mTLS certificate from request."""
        # Extract certificate from header (in production, this would come from TLS layer)
        cert_pem = request.headers.get("X-Client-Certificate")
        if not cert_pem:
            return False, None

        # Validate certificate
        is_valid, message = self.mtls_manager.validate_certificate(cert_pem)
        if not is_valid:
            logger.warning(f"Certificate validation failed: {message}")
            return False, None

        # Extract certificate info
        try:
            from cryptography import x509

            cert = x509.load_pem_x509_certificate(cert_pem.encode())

            # Verify service name matches
            common_name = None
            for attribute in cert.subject:
                if attribute.oid == x509.oid.NameOID.COMMON_NAME:
                    common_name = attribute.value
                    break

            if common_name != expected_service:
                logger.warning(
                    f"Certificate CN mismatch: expected {expected_service}, got {common_name}"
                )
                return False, None

            # Calculate fingerprint
            from cryptography.hazmat.primitives import serialization

            fingerprint = hashlib.sha256(
                cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            return True, fingerprint

        except Exception as e:
            logger.error(f"Certificate parsing error: {e}")
            return False, None

    async def _check_service_policy(self, context: RequestContext) -> bool:
        """Check if request is allowed by service policies."""
        policy_key = (context.source_service, context.target_service)
        policy_ids = self._policy_index.get(policy_key, [])

        if not policy_ids:
            # No policies found - default deny
            return False

        # Check each applicable policy
        for policy_id in policy_ids:
            policy = self._policies.get(policy_id)
            if not policy:
                continue

            # Check operation
            if context.operation not in policy.allowed_operations:
                continue

            # Check conditions
            if not self._evaluate_policy_conditions(policy, context):
                continue

            # Check mTLS requirement
            if policy.mtls_required and not context.mtls_verified:
                continue

            # Policy matches and allows access
            return True

        return False

    def _evaluate_policy_conditions(
        self, policy: ServicePolicy, context: RequestContext
    ) -> bool:
        """Evaluate policy conditions."""
        for condition_key, condition_value in policy.conditions.items():
            if condition_key == "time_window":
                if not self._check_time_window(condition_value):
                    return False

            elif condition_key == "ip_whitelist":
                if context.client_ip not in condition_value:
                    return False

            elif condition_key == "max_requests_per_minute":
                # This would integrate with rate limiting
                pass

        return True

    def _check_time_window(self, time_window: Dict[str, str]) -> bool:
        """Check if current time is within allowed window."""
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")

        start_time = time_window.get("start", "00:00")
        end_time = time_window.get("end", "23:59")

        # Simple time comparison (doesn't handle midnight crossover)
        return start_time <= current_time <= end_time

    async def calculate_session_risk(
        self, session_id: str, factors: Dict[str, Any]
    ) -> SessionRiskScore:
        """Calculate risk score for a session."""
        # Check cache first
        cached_score = self._risk_cache.get(session_id)
        if (
            cached_score
            and (datetime.utcnow() - cached_score.timestamp).total_seconds()
            < self._risk_cache_ttl
        ):
            return cached_score

        # Calculate risk score based on factors
        risk_components = {}
        total_risk = 0.0

        # Location change factor
        if factors.get("location_change", False):
            risk_components["location_change"] = 0.3
            total_risk += 0.3

        # Unusual access time
        if factors.get("unusual_time", False):
            risk_components["unusual_time"] = 0.2
            total_risk += 0.2

        # Failed authentication attempts
        failed_attempts = factors.get("failed_attempts", 0)
        if failed_attempts > 0:
            attempt_risk = min(failed_attempts * 0.1, 0.5)
            risk_components["failed_attempts"] = attempt_risk
            total_risk += attempt_risk

        # Anomaly detection score
        anomaly_score = factors.get("anomaly_score", 0.0)
        if anomaly_score > 0:
            risk_components["anomaly"] = anomaly_score * 0.5
            total_risk += anomaly_score * 0.5

        # Normalize to 0-1 range
        total_risk = min(total_risk, 1.0)

        # Create risk score
        risk_score = SessionRiskScore(
            session_id=session_id,
            score=total_risk,
            factors=risk_components,
            timestamp=datetime.utcnow(),
            requires_reauthentication=total_risk > self.config.max_risk_score,
        )

        # Cache the score
        self._risk_cache[session_id] = risk_score

        return risk_score

    def _extract_risk_factors(
        self, request: Request, context: RequestContext
    ) -> Dict[str, Any]:
        """Extract risk factors from request."""
        factors = {}

        # Check for location change (simplified - would use GeoIP in production)
        session_id = context.session_id
        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Check if IP changed
            if session.anomalies:
                last_ip = session.anomalies[-1].get("client_ip")
                if last_ip and last_ip != context.client_ip:
                    factors["location_change"] = True

        # Check for unusual access time
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            factors["unusual_time"] = True

        # Check failed attempts
        factors["failed_attempts"] = self._failed_attempts.get(
            context.client_ip, 0
        )

        # Placeholder for ML-based anomaly detection
        factors["anomaly_score"] = 0.1  # Would be calculated by ML model

        return factors

    async def start_continuous_verification(
        self, session_id: str, request: Request
    ) -> None:
        """Start continuous verification for a session."""
        async with self._session_lock:
            if session_id in self._sessions:
                return  # Already running

            # Extract service name from certificate
            cert_pem = request.headers.get("X-Client-Certificate", "")
            service_name = self._extract_service_name(cert_pem)

            # Create session
            session = VerificationSession(
                session_id=session_id,
                service_name=service_name,
                start_time=datetime.utcnow(),
                last_verification=datetime.utcnow(),
            )

            self._sessions[session_id] = session

        # Run verification loop
        try:
            while session.active:
                await asyncio.sleep(self.config.verification_interval_seconds)

                # Perform verification
                await self._perform_verification(session)

                # Check if session should continue
                if not session.active:
                    break

                # Check session timeout
                if (
                    datetime.utcnow() - session.start_time
                ).total_seconds() > self.config.session_timeout_minutes * 60:
                    session.active = False
                    break

        finally:
            # Clean up session
            async with self._session_lock:
                if session_id in self._sessions:
                    del self._sessions[session_id]

    async def _perform_verification(
        self, session: VerificationSession
    ) -> None:
        """Perform a verification check on a session."""
        session.last_verification = datetime.utcnow()
        session.verification_count += 1

        # Calculate current risk score
        risk_factors = {
            "verification_count": session.verification_count,
            "session_duration": (
                datetime.utcnow() - session.start_time
            ).total_seconds()
            / 3600,  # hours
            "anomaly_count": len(session.anomalies),
        }

        risk_score = await self.calculate_session_risk(
            session.session_id, risk_factors
        )
        session.risk_scores.append(risk_score)

        # Check if session should be terminated
        if risk_score.requires_reauthentication:
            session.active = False
            logger.warning(
                f"Session {session.session_id} terminated due to high risk: {risk_score.score}"
            )

    def stop_continuous_verification(self, session_id: str) -> None:
        """Stop continuous verification for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].active = False

    def get_verification_status(self, session_id: str) -> Dict[str, Any]:
        """Get verification status for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {"active": False, "session_id": session_id}

        latest_risk = session.risk_scores[-1] if session.risk_scores else None

        return {
            "active": session.active,
            "session_id": session_id,
            "service_name": session.service_name,
            "start_time": session.start_time.isoformat(),
            "verification_count": session.verification_count,
            "latest_risk_score": latest_risk.score if latest_risk else None,
            "anomaly_count": len(session.anomalies),
        }

    def add_service_policy(self, policy: ServicePolicy) -> None:
        """Add a service access policy."""
        policy_id = f"{policy.source_service}-{policy.target_service}-{len(self._policies)}"
        self._policies[policy_id] = policy

        # Update index
        key = (policy.source_service, policy.target_service)
        if key not in self._policy_index:
            self._policy_index[key] = []
        self._policy_index[key].append(policy_id)

        logger.info(
            f"Added policy: {policy.source_service} -> {policy.target_service} "
            f"(operations: {policy.allowed_operations})"
        )

    async def evaluate_permission(
        self,
        source: str,
        target: str,
        operation: str,
        context: Dict[str, Any],
    ) -> bool:
        """Evaluate if permission should be granted."""
        # Create minimal request context
        req_context = RequestContext(
            source_service=source,
            target_service=target,
            operation=operation,
            client_ip=context.get("client_ip", "unknown"),
            user_agent=context.get("user_agent", ""),
            timestamp=datetime.utcnow(),
        )

        # Find applicable policies
        policy_key = (source, target)
        policy_ids = self._policy_index.get(policy_key, [])

        for policy_id in policy_ids:
            policy = self._policies.get(policy_id)
            if not policy:
                continue

            if operation not in policy.allowed_operations:
                continue

            # Check time window if specified
            if "time_window" in policy.conditions:
                current_time = context.get(
                    "time", datetime.utcnow().strftime("%H:%M")
                )
                time_window = policy.conditions["time_window"]
                if not (
                    time_window["start"] <= current_time <= time_window["end"]
                ):
                    continue

            # Check request rate if specified
            if "max_requests_per_minute" in policy.conditions:
                request_count = context.get("request_count", 0)
                if (
                    request_count
                    > policy.conditions["max_requests_per_minute"]
                ):
                    continue

            # All conditions met
            return True

        return False

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _generate_session_id(self, request: Request, service_name: str) -> str:
        """Generate session ID for request."""
        # Combine stable identifiers
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        cert_fingerprint = request.headers.get("X-Certificate-Fingerprint", "")

        session_data = (
            f"{service_name}:{client_ip}:{user_agent}:{cert_fingerprint}"
        )
        return hashlib.sha256(session_data.encode()).hexdigest()

    def _extract_service_name(self, cert_pem: str) -> str:
        """Extract service name from certificate."""
        if not cert_pem:
            return "unknown"

        try:
            from cryptography import x509

            cert = x509.load_pem_x509_certificate(cert_pem.encode())

            for attribute in cert.subject:
                if attribute.oid == x509.oid.NameOID.COMMON_NAME:
                    return attribute.value

            return "unknown"
        except Exception:
            return "unknown"

    def _track_failed_attempt(self, client_ip: str) -> None:
        """Track failed authentication attempt."""
        if client_ip not in self._failed_attempts:
            self._failed_attempts[client_ip] = 0

        self._failed_attempts[client_ip] += 1

        # Check if should block
        if self._failed_attempts[client_ip] >= self.config.max_failed_attempts:
            logger.warning(f"Max failed attempts reached for {client_ip}")
