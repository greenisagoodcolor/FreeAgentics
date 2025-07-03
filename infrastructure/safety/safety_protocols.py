"""
Core Safety Protocols and Definitions for FreeAgentics Multi-Agent Systems.

This module defines the fundamental safety concepts, levels, and protocols
used throughout the safety monitoring and verification systems.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety level classifications"""

    CRITICAL = "critical"  # System-threatening violations
    HIGH = "high"  # Agent boundary violations
    MEDIUM = "medium"  # Performance degradation
    LOW = "low"  # Minor deviations
    INFO = "info"  # Informational alerts


class ViolationType(Enum):
    """Types of safety violations"""

    BOUNDARY_VIOLATION = "boundary_violation"
    INDEPENDENCE_FAILURE = "independence_failure"
    MATHEMATICAL_ERROR = "mathematical_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PROTOCOL_BREACH = "protocol_breach"
    DATA_INTEGRITY = "data_integrity"


@dataclass
class SafetyViolation:
    """Generic safety violation record"""

    violation_id: str
    violation_type: ViolationType
    severity: SafetyLevel
    description: str
    agent_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    resolution_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "agent_id": self.agent_id,
            "evidence": self.evidence,
            "mitigation_actions": self.mitigation_actions,
            "resolved": self.resolved,
            "timestamp": self.timestamp.isoformat(),
            "resolution_timestamp": (
                self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
            ),
        }


@dataclass
class SafetyMetrics:
    """Safety metrics for monitoring and reporting"""

    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    resolved_violations: int = 0
    mean_resolution_time: float = 0.0
    boundary_integrity_average: float = 1.0
    system_safety_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_safety_score(self) -> float:
        """Calculate overall system safety score"""
        if self.total_violations == 0:
            return 1.0

        # Weight violations by severity
        weighted_violations = (
            self.critical_violations * 4
            + self.high_violations * 3
            + self.medium_violations * 2
            + self.low_violations * 1
        )

        max_weighted = self.total_violations * 4

        # Higher resolution rate improves score
        resolution_factor = self.resolved_violations / max(self.total_violations, 1)

        # Combine violation severity and resolution effectiveness
        safety_score = (
            1 - weighted_violations / max(max_weighted, 1)
        ) * 0.7 + resolution_factor * 0.3

        self.system_safety_score = max(0.0, min(1.0, safety_score))
        return self.system_safety_score


class SafetyProtocol:
    """Base class for safety protocols"""

    def __init__(
        self, protocol_name: str, severity_threshold: SafetyLevel = SafetyLevel.MEDIUM
    ) -> None:
        self.protocol_name = protocol_name
        self.severity_threshold = severity_threshold
        self.violations: List[SafetyViolation] = []
        self.enabled = True

        logger.info(f"Initialized safety protocol: {protocol_name}")

    def check_violation(self, **kwargs) -> Optional[SafetyViolation]:
        """Check for safety violations - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement check_violation")

    def handle_violation(self, violation: SafetyViolation) -> List[str]:
        """Handle detected violation - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement handle_violation")

    def get_metrics(self) -> SafetyMetrics:
        """Get safety metrics for this protocol"""
        metrics = SafetyMetrics()
        metrics.total_violations = len(self.violations)

        for violation in self.violations:
            if violation.severity == SafetyLevel.CRITICAL:
                metrics.critical_violations += 1
            elif violation.severity == SafetyLevel.HIGH:
                metrics.high_violations += 1
            elif violation.severity == SafetyLevel.MEDIUM:
                metrics.medium_violations += 1
            elif violation.severity == SafetyLevel.LOW:
                metrics.low_violations += 1

            if violation.resolved:
                metrics.resolved_violations += 1

        return metrics

    def enable(self):
        """Enable the safety protocol"""
        self.enabled = True
        logger.info(f"Enabled safety protocol: {self.protocol_name}")

    def disable(self):
        """Disable the safety protocol"""
        self.enabled = False
        logger.warning(f"Disabled safety protocol: {self.protocol_name}")


class MarkovBlanketSafetyProtocol(SafetyProtocol):
    """Safety protocol specifically for Markov blanket boundary monitoring"""

    def __init__(self, independence_threshold: float = 0.05) -> None:
        super().__init__("Markov Blanket Safety", SafetyLevel.HIGH)
        self.independence_threshold = independence_threshold

    def check_violation(self, **kwargs) -> Optional[SafetyViolation]:
        """Check for Markov blanket boundary violations"""
        if not self.enabled:
            return None

        # Extract required parameters from kwargs
        agent_id = kwargs.get("agent_id")
        independence_measure = kwargs.get("independence_measure")
        mathematical_proof = kwargs.get("mathematical_proof", "")

        # Validate required parameters
        if agent_id is None or independence_measure is None:
            return None

        if independence_measure > self.independence_threshold:
            violation = SafetyViolation(
                violation_id=f"mb_{agent_id}_{
                    datetime.now().timestamp()}",
                violation_type=ViolationType.BOUNDARY_VIOLATION,
                severity=SafetyLevel.HIGH if independence_measure > 0.1 else SafetyLevel.MEDIUM,
                description=f"Markov blanket boundary violation detected for agent {agent_id}",
                agent_id=agent_id,
                evidence={
                    "independence_measure": independence_measure,
                    "threshold": self.independence_threshold,
                    "mathematical_proof": mathematical_proof,
                    **kwargs,
                },
            )

            self.violations.append(violation)
            return violation

        return None

    def handle_violation(self, violation: SafetyViolation) -> List[str]:
        """Handle Markov blanket violation"""
        actions = []

        if violation.severity in [SafetyLevel.CRITICAL, SafetyLevel.HIGH]:
            actions.extend(
                [
                    "Trigger immediate boundary recalculation",
                    "Isolate agent from sensitive operations",
                    "Generate detailed mathematical audit trail",
                    "Notify safety monitoring dashboard",
                ]
            )
        elif violation.severity == SafetyLevel.MEDIUM:
            actions.extend(
                [
                    "Schedule boundary verification",
                    "Log violation for trend analysis",
                    "Update agent safety metrics",
                ]
            )

        violation.mitigation_actions = actions
        logger.warning(
            f"Handling Markov blanket violation for agent {
                violation.agent_id}: {actions}"
        )

        return actions
