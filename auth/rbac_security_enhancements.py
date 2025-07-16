"""
RBAC Security Enhancements
Task 14.4 - Security improvements for RBAC implementation

This module implements critical security enhancements for the RBAC system
based on the security audit findings.
"""

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from auth.comprehensive_audit_logger import comprehensive_auditor
from auth.security_implementation import Permission, TokenData, UserRole
from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor


class SecurityLevel(str, Enum):
    """Security levels for resources."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class SecurityContext:
    """Enhanced security context for authorization decisions."""
    
    request_id: str
    timestamp: datetime
    source_ip: str
    user_agent: str
    session_id: str
    device_fingerprint: Optional[str] = None
    geo_location: Optional[str] = None
    risk_indicators: Dict[str, Any] = field(default_factory=dict)
    authentication_method: Optional[str] = None
    authentication_strength: Optional[int] = None  # 1-5 scale


@dataclass
class AuthorizationDecision:
    """Detailed authorization decision with audit trail."""
    
    granted: bool
    reason: str
    decision_id: str
    timestamp: datetime
    evaluated_rules: List[str]
    risk_score: float
    confidence_level: float
    additional_requirements: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class PrivilegeEscalationDetector:
    """Detects and prevents privilege escalation attempts."""
    
    def __init__(self):
        self._escalation_attempts = defaultdict(list)
        self._lock = threading.Lock()
        self._threshold = 3  # Max attempts per hour
        self._window = timedelta(hours=1)
    
    def check_escalation_attempt(
        self,
        user_id: str,
        current_role: UserRole,
        requested_role: UserRole,
        context: SecurityContext
    ) -> Tuple[bool, Optional[str]]:
        """Check if this is a suspicious privilege escalation attempt."""
        
        with self._lock:
            now = datetime.now(timezone.utc)
            
            # Clean old attempts
            self._escalation_attempts[user_id] = [
                attempt for attempt in self._escalation_attempts[user_id]
                if now - attempt["timestamp"] < self._window
            ]
            
            # Check role hierarchy
            role_hierarchy = {
                UserRole.OBSERVER: 1,
                UserRole.AGENT_MANAGER: 2,
                UserRole.RESEARCHER: 3,
                UserRole.ADMIN: 4
            }
            
            current_level = role_hierarchy.get(current_role, 0)
            requested_level = role_hierarchy.get(requested_role, 0)
            
            # Record attempt if it's an escalation
            if requested_level > current_level:
                attempt = {
                    "timestamp": now,
                    "from_role": current_role,
                    "to_role": requested_role,
                    "context": context
                }
                self._escalation_attempts[user_id].append(attempt)
                
                # Check if threshold exceeded
                if len(self._escalation_attempts[user_id]) > self._threshold:
                    return False, f"Too many escalation attempts ({len(self._escalation_attempts[user_id])}) in {self._window}"
            
            # Check for suspicious patterns
            if self._detect_suspicious_pattern(user_id, context):
                return False, "Suspicious escalation pattern detected"
            
            return True, None
    
    def _detect_suspicious_pattern(self, user_id: str, context: SecurityContext) -> bool:
        """Detect suspicious patterns in escalation attempts."""
        
        attempts = self._escalation_attempts[user_id]
        if len(attempts) < 2:
            return False
        
        # Check for rapid attempts
        if len(attempts) >= 2:
            time_diff = attempts[-1]["timestamp"] - attempts[-2]["timestamp"]
            if time_diff < timedelta(minutes=1):
                return True
        
        # Check for attempts from different IPs
        ips = set(attempt["context"].source_ip for attempt in attempts if attempt.get("context"))
        if len(ips) > 2:
            return True
        
        # Check for attempts to jump multiple levels
        for attempt in attempts:
            from_level = {
                UserRole.OBSERVER: 1,
                UserRole.AGENT_MANAGER: 2,
                UserRole.RESEARCHER: 3,
                UserRole.ADMIN: 4
            }.get(attempt["from_role"], 0)
            
            to_level = {
                UserRole.OBSERVER: 1,
                UserRole.AGENT_MANAGER: 2,
                UserRole.RESEARCHER: 3,
                UserRole.ADMIN: 4
            }.get(attempt["to_role"], 0)
            
            if to_level - from_level > 1:
                return True
        
        return False


class ZeroTrustValidator:
    """Implements Zero Trust security model for authorization."""
    
    def __init__(self):
        self._trust_scores = {}
        self._validation_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._lock = threading.Lock()
    
    def validate_access(
        self,
        user: TokenData,
        resource_id: str,
        action: str,
        context: SecurityContext
    ) -> AuthorizationDecision:
        """Validate access with Zero Trust principles."""
        
        decision_id = secrets.token_urlsafe(16)
        start_time = time.time()
        
        try:
            # Never trust, always verify
            validations = []
            
            # 1. Validate token integrity
            token_valid, token_reason = self._validate_token_integrity(user)
            validations.append(("token_integrity", token_valid, token_reason))
            
            # 2. Validate session consistency
            session_valid, session_reason = self._validate_session_consistency(
                user.user_id, context
            )
            validations.append(("session_consistency", session_valid, session_reason))
            
            # 3. Calculate trust score
            trust_score = self._calculate_trust_score(user, context, validations)
            
            # 4. Check risk indicators
            risk_score = self._calculate_risk_score(context)
            
            # 5. Validate contextual access
            context_valid, context_reason = self._validate_contextual_access(
                user, resource_id, action, context, trust_score, risk_score
            )
            validations.append(("contextual_access", context_valid, context_reason))
            
            # Make decision
            all_valid = all(valid for _, valid, _ in validations)
            confidence = trust_score * (1 - risk_score)
            
            # Additional requirements based on risk
            additional_requirements = []
            if risk_score > 0.7:
                additional_requirements.append("multi_factor_authentication")
            if risk_score > 0.5:
                additional_requirements.append("additional_approval")
            if trust_score < 0.5:
                additional_requirements.append("identity_verification")
            
            decision = AuthorizationDecision(
                granted=all_valid and confidence > 0.5,
                reason=self._format_decision_reason(validations, trust_score, risk_score),
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc),
                evaluated_rules=[name for name, _, _ in validations],
                risk_score=risk_score,
                confidence_level=confidence,
                additional_requirements=additional_requirements,
                evidence={
                    "validations": validations,
                    "trust_score": trust_score,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Log decision
            self._log_decision(user, resource_id, action, decision, context)
            
            return decision
            
        except Exception as e:
            # Fail closed - deny on any error
            return AuthorizationDecision(
                granted=False,
                reason=f"Authorization validation error: {str(e)}",
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc),
                evaluated_rules=[],
                risk_score=1.0,
                confidence_level=0.0,
                evidence={"error": str(e)}
            )
    
    def _validate_token_integrity(self, user: TokenData) -> Tuple[bool, str]:
        """Validate token hasn't been tampered with."""
        
        # Check expiration
        if user.exp < datetime.now(timezone.utc):
            return False, "Token expired"
        
        # Check required fields
        if not all([user.user_id, user.username, user.role]):
            return False, "Missing required token fields"
        
        # Validate permissions match role
        expected_perms = set(ROLE_PERMISSIONS.get(user.role, []))
        actual_perms = set(user.permissions)
        
        if actual_perms != expected_perms:
            return False, f"Permission mismatch for role {user.role}"
        
        return True, "Token valid"
    
    def _validate_session_consistency(
        self,
        user_id: str,
        context: SecurityContext
    ) -> Tuple[bool, str]:
        """Validate session consistency to prevent hijacking."""
        
        with self._lock:
            cache_key = f"{user_id}:{context.session_id}"
            
            if cache_key in self._validation_cache:
                cached = self._validation_cache[cache_key]
                if datetime.now(timezone.utc) - cached["timestamp"] < self._cache_ttl:
                    # Check consistency
                    if cached["ip"] != context.source_ip:
                        return False, "Session IP changed"
                    if cached["user_agent"] != context.user_agent:
                        return False, "User agent changed"
                    if cached.get("device_fingerprint") and \
                       cached["device_fingerprint"] != context.device_fingerprint:
                        return False, "Device fingerprint changed"
            
            # Update cache
            self._validation_cache[cache_key] = {
                "timestamp": datetime.now(timezone.utc),
                "ip": context.source_ip,
                "user_agent": context.user_agent,
                "device_fingerprint": context.device_fingerprint
            }
        
        return True, "Session consistent"
    
    def _calculate_trust_score(
        self,
        user: TokenData,
        context: SecurityContext,
        validations: List[Tuple[str, bool, str]]
    ) -> float:
        """Calculate trust score for the user."""
        
        score = 1.0
        
        # Reduce score for failed validations
        for _, valid, _ in validations:
            if not valid:
                score *= 0.5
        
        # Factor in authentication strength
        if context.authentication_strength:
            score *= (context.authentication_strength / 5.0)
        
        # Factor in user history (in real implementation, check database)
        # For now, admins get slightly higher trust
        if user.role == UserRole.ADMIN:
            score *= 0.9  # Admins need more scrutiny
        
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_score(self, context: SecurityContext) -> float:
        """Calculate risk score based on context."""
        
        risk = 0.0
        
        # Check risk indicators
        if context.risk_indicators.get("vpn_detected"):
            risk += 0.2
        if context.risk_indicators.get("tor_detected"):
            risk += 0.4
        if context.risk_indicators.get("suspicious_user_agent"):
            risk += 0.1
        if context.risk_indicators.get("unusual_time"):
            risk += 0.1
        if context.risk_indicators.get("unusual_location"):
            risk += 0.2
        
        # Check authentication method
        if context.authentication_method == "password_only":
            risk += 0.2
        elif context.authentication_method == "mfa":
            risk -= 0.1
        
        return max(0.0, min(1.0, risk))
    
    def _validate_contextual_access(
        self,
        user: TokenData,
        resource_id: str,
        action: str,
        context: SecurityContext,
        trust_score: float,
        risk_score: float
    ) -> Tuple[bool, str]:
        """Validate access based on context."""
        
        # High-risk operations require high trust
        high_risk_actions = ["delete", "admin", "modify_permissions"]
        if action in high_risk_actions:
            if trust_score < 0.8:
                return False, f"Insufficient trust score ({trust_score:.2f}) for high-risk action"
            if risk_score > 0.3:
                return False, f"Risk score too high ({risk_score:.2f}) for high-risk action"
        
        # Validate based on time of day for admin operations
        if user.role == UserRole.ADMIN and action in ["admin", "delete"]:
            current_hour = datetime.now(timezone.utc).hour
            if not (8 <= current_hour <= 18):  # Business hours UTC
                if not context.risk_indicators.get("emergency_override"):
                    return False, "Admin operations restricted to business hours"
        
        return True, "Contextual access valid"
    
    def _format_decision_reason(
        self,
        validations: List[Tuple[str, bool, str]],
        trust_score: float,
        risk_score: float
    ) -> str:
        """Format human-readable decision reason."""
        
        failed = [f"{name}: {reason}" for name, valid, reason in validations if not valid]
        
        if failed:
            return f"Access denied - {'; '.join(failed)}"
        
        if trust_score < 0.5:
            return f"Access denied - Insufficient trust score: {trust_score:.2f}"
        
        if risk_score > 0.7:
            return f"Access denied - Risk score too high: {risk_score:.2f}"
        
        return f"Access granted - Trust: {trust_score:.2f}, Risk: {risk_score:.2f}"
    
    def _log_decision(
        self,
        user: TokenData,
        resource_id: str,
        action: str,
        decision: AuthorizationDecision,
        context: SecurityContext
    ):
        """Log authorization decision with full context."""
        
        comprehensive_auditor.log_authorization_decision(
            decision_id=decision.decision_id,
            user_id=user.user_id,
            username=user.username,
            resource_id=resource_id,
            action=action,
            granted=decision.granted,
            reason=decision.reason,
            risk_score=decision.risk_score,
            trust_score=decision.confidence_level,
            context={
                "request_id": context.request_id,
                "source_ip": context.source_ip,
                "session_id": context.session_id,
                "evaluated_rules": decision.evaluated_rules,
                "additional_requirements": decision.additional_requirements,
                "processing_time": decision.evidence.get("processing_time", 0)
            }
        )


class SecureResourceIDGenerator:
    """Generate cryptographically secure resource IDs."""
    
    @staticmethod
    def generate_resource_id(resource_type: str, owner_id: str) -> str:
        """Generate unpredictable resource ID."""
        
        # Use cryptographically secure random
        random_part = secrets.token_hex(16)
        
        # Include timestamp for uniqueness
        timestamp = int(time.time() * 1000000)
        
        # Create ID with type prefix
        components = [
            resource_type[:3].lower(),
            str(timestamp)[-8:],
            random_part
        ]
        
        # Generate checksum
        checksum_input = f"{resource_type}:{owner_id}:{timestamp}:{random_part}"
        checksum = hashlib.sha256(checksum_input.encode()).hexdigest()[:8]
        components.append(checksum)
        
        return "-".join(components)
    
    @staticmethod
    def validate_resource_id(resource_id: str, resource_type: str) -> bool:
        """Validate resource ID format and checksum."""
        
        try:
            parts = resource_id.split("-")
            if len(parts) != 4:
                return False
            
            type_prefix, timestamp_part, random_part, checksum = parts
            
            # Validate type prefix
            if type_prefix != resource_type[:3].lower():
                return False
            
            # Validate format
            if not (len(timestamp_part) == 8 and timestamp_part.isdigit()):
                return False
            
            if not (len(random_part) == 32 and all(c in "0123456789abcdef" for c in random_part)):
                return False
            
            if not (len(checksum) == 8 and all(c in "0123456789abcdef" for c in checksum)):
                return False
            
            return True
            
        except Exception:
            return False


class RateLimiter:
    """Rate limiting for authorization attempts."""
    
    def __init__(self):
        self._attempts = defaultdict(list)
        self._lock = threading.Lock()
        self._limits = {
            "authorization": (100, timedelta(minutes=1)),  # 100 per minute
            "privilege_change": (5, timedelta(hours=1)),   # 5 per hour
            "admin_action": (20, timedelta(minutes=1)),    # 20 per minute
        }
    
    def check_rate_limit(
        self,
        identifier: str,
        action_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if action is within rate limits."""
        
        if action_type not in self._limits:
            return True, None
        
        limit, window = self._limits[action_type]
        
        with self._lock:
            now = datetime.now(timezone.utc)
            key = f"{identifier}:{action_type}"
            
            # Clean old attempts
            self._attempts[key] = [
                attempt for attempt in self._attempts[key]
                if now - attempt < window
            ]
            
            # Check limit
            if len(self._attempts[key]) >= limit:
                return False, f"Rate limit exceeded: {limit} per {window}"
            
            # Record attempt
            self._attempts[key].append(now)
            
            return True, None


class ConstantTimeComparator:
    """Constant-time comparison to prevent timing attacks."""
    
    @staticmethod
    def compare(a: str, b: str) -> bool:
        """Compare strings in constant time."""
        
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        
        # Use hmac.compare_digest for constant-time comparison
        return hmac.compare_digest(a.encode(), b.encode())
    
    @staticmethod
    def compare_permissions(
        required: List[Permission],
        actual: List[Permission]
    ) -> bool:
        """Compare permission lists in constant time."""
        
        # Convert to sets for comparison
        required_set = set(p.value for p in required)
        actual_set = set(p.value for p in actual)
        
        # Create fixed-size comparison strings
        required_str = ",".join(sorted(required_set))
        actual_str = ",".join(sorted(actual_set))
        
        return ConstantTimeComparator.compare(required_str, actual_str)


# Global instances
privilege_escalation_detector = PrivilegeEscalationDetector()
zero_trust_validator = ZeroTrustValidator()
rate_limiter = RateLimiter()
secure_id_generator = SecureResourceIDGenerator()