"""
Coalition Contract Module for FreeAgentics.

This module implements contracts that govern coalition behavior, resource sharing,
and goal alignment. Contracts ensure fair and transparent collaboration between agents.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ContractStatus(Enum):
    """Status of a coalition contract"""

    DRAFT = "draft"
    PROPOSED = "proposed"
    NEGOTIATING = "negotiating"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class ContractType(Enum):
    """Types of coalition contracts"""

    RESOURCE_SHARING = "resource_sharing"
    GOAL_ACHIEVEMENT = "goal_achievement"
    MUTUAL_DEFENSE = "mutual_defense"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    TRADE_AGREEMENT = "trade_agreement"
    EXPLORATION_PACT = "exploration_pact"


class ViolationType(Enum):
    """Types of contract violations"""

    RESOURCE_MISUSE = "resource_misuse"
    GOAL_ABANDONMENT = "goal_abandonment"
    INFORMATION_WITHHOLDING = "information_withholding"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    DEADLINE_MISSED = "deadline_missed"
    TRUST_BREACH = "trust_breach"


@dataclass
class ContractTerm:
    """Represents a term or condition in a contract"""

    term_id: str = field(default_factory=lambda: f"term_{uuid.uuid4().hex[:8]}")
    description: str = ""
    category: str = "general"
    is_mandatory: bool = True
    penalty_for_violation: float = 0.0
    verification_method: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.penalty_for_violation < 0:
            raise ValueError("Penalty cannot be negative")


@dataclass
class ResourceCommitment:
    """Represents a resource commitment in a contract"""

    agent_id: str
    resource_type: str
    amount: float
    minimum_contribution: float = 0.0
    maximum_contribution: Optional[float] = None
    contribution_schedule: str = "immediate"  # immediate, periodic, on_demand

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Resource amount cannot be negative")
        if self.minimum_contribution < 0:
            raise ValueError("Minimum contribution cannot be negative")
        if (
            self.maximum_contribution is not None
            and self.maximum_contribution < self.minimum_contribution
        ):
            raise ValueError("Maximum contribution must be >= minimum contribution")
        if self.amount < self.minimum_contribution:
            raise ValueError("Amount must be >= minimum contribution")
        if self.maximum_contribution is not None and self.amount > self.maximum_contribution:
            raise ValueError("Amount must be <= maximum contribution")


@dataclass
class ContractViolation:
    """Records a contract violation"""

    violation_id: str = field(default_factory=lambda: f"viol_{uuid.uuid4().hex[:8]}")
    contract_id: str = ""
    violator_id: str = ""
    violation_type: ViolationType = ViolationType.RESOURCE_MISUSE
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: float = 0.5  # 0.0 (minor) to 1.0 (severe)
    evidence: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_details: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("Severity must be between 0.0 and 1.0")


@dataclass
class CoalitionContract:
    """
    Represents a formal contract between coalition members.

    Contracts define the terms of collaboration, resource sharing,
    responsibilities, and penalties for violations.
    """

    contract_id: str = field(default_factory=lambda: f"contract_{uuid.uuid4().hex[:8]}")
    coalition_id: str = ""
    contract_type: ContractType = ContractType.RESOURCE_SHARING
    title: str = ""
    description: str = ""

    # Parties
    initiator_id: str = ""
    member_ids: Set[str] = field(default_factory=set)
    required_signatures: Set[str] = field(default_factory=set)
    current_signatures: Set[str] = field(default_factory=set)

    # Terms and conditions
    terms: List[ContractTerm] = field(default_factory=list)
    resource_commitments: List[ResourceCommitment] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    auto_renew: bool = False
    renewal_period: Optional[timedelta] = None

    # Status tracking
    status: ContractStatus = ContractStatus.DRAFT
    violations: List[ContractViolation] = field(default_factory=list)
    amendment_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    compliance_score: float = 1.0  # 0.0 to 1.0
    value_generated: float = 0.0
    disputes_resolved: int = 0
    successful_completions: int = 0

    def __post_init__(self):
        """Validate contract initialization"""
        if not self.coalition_id:
            raise ValueError("Coalition ID is required")
        if not self.initiator_id:
            raise ValueError("Initiator ID is required")
        if not self.title:
            raise ValueError("Contract title is required")
        if not 0.0 <= self.compliance_score <= 1.0:
            raise ValueError("Compliance score must be between 0.0 and 1.0")

        # Initiator should be in members
        if self.initiator_id and self.initiator_id not in self.member_ids:
            self.member_ids.add(self.initiator_id)

        # Required signatures should be subset of members
        if self.required_signatures and not self.required_signatures.issubset(self.member_ids):
            raise ValueError("Required signatures must be subset of members")

    def add_term(self, term: ContractTerm) -> None:
        """Add a term to the contract"""
        if self.status not in [ContractStatus.DRAFT, ContractStatus.NEGOTIATING]:
            raise ValueError(f"Cannot add terms to contract in {self.status} status")
        self.terms.append(term)

    def add_resource_commitment(self, commitment: ResourceCommitment) -> None:
        """Add a resource commitment"""
        if commitment.agent_id not in self.member_ids:
            raise ValueError(f"Agent {commitment.agent_id} is not a member of this contract")
        if self.status not in [ContractStatus.DRAFT, ContractStatus.NEGOTIATING]:
            raise ValueError(f"Cannot add commitments to contract in {self.status} status")
        self.resource_commitments.append(commitment)

    def sign(self, agent_id: str) -> bool:
        """Sign the contract"""
        if agent_id not in self.member_ids:
            raise ValueError(f"Agent {agent_id} is not a member of this contract")
        if self.status != ContractStatus.PROPOSED:
            raise ValueError(f"Contract must be in PROPOSED status to sign, current: {self.status}")

        self.current_signatures.add(agent_id)

        # Check if all required signatures are collected
        if self.required_signatures.issubset(self.current_signatures):
            self.activate()
            return True
        return False

    def propose(self) -> None:
        """Move contract from draft to proposed status"""
        if self.status != ContractStatus.DRAFT:
            raise ValueError(f"Can only propose contracts in DRAFT status, current: {self.status}")
        if not self.terms:
            raise ValueError("Contract must have at least one term")
        if not self.required_signatures:
            self.required_signatures = self.member_ids.copy()

        self.status = ContractStatus.PROPOSED

    def activate(self) -> None:
        """Activate the contract"""
        if self.status != ContractStatus.PROPOSED:
            raise ValueError(f"Can only activate PROPOSED contracts, current: {self.status}")
        if not self.required_signatures.issubset(self.current_signatures):
            raise ValueError("Not all required signatures collected")

        self.status = ContractStatus.ACTIVE
        if not self.effective_date:
            self.effective_date = datetime.utcnow()

    def record_violation(self, violation: ContractViolation) -> None:
        """Record a contract violation"""
        if self.status != ContractStatus.ACTIVE:
            raise ValueError(
                f"Can only record violations for ACTIVE contracts, current: {self.status}"
            )

        violation.contract_id = self.contract_id
        self.violations.append(violation)

        # Update compliance score
        severity_impact = violation.severity * 0.1  # Each violation reduces score
        self.compliance_score = max(0.0, self.compliance_score - severity_impact)

    def is_expired(self) -> bool:
        """Check if contract has expired"""
        if self.expiration_date:
            return datetime.utcnow() > self.expiration_date
        return False

    def is_active(self) -> bool:
        """Check if contract is currently active"""
        return (
            self.status == ContractStatus.ACTIVE
            and not self.is_expired()
            and self.compliance_score > 0.0
        )

    def terminate(self, reason: str) -> None:
        """Terminate the contract"""
        if self.status not in [ContractStatus.ACTIVE, ContractStatus.SUSPENDED]:
            raise ValueError(
                f"Can only terminate ACTIVE or SUSPENDED contracts, current: {self.status}"
            )

        self.status = ContractStatus.TERMINATED
        self.amendment_history.append(
            {"type": "termination", "timestamp": datetime.utcnow(), "reason": reason}
        )

    def complete(self) -> None:
        """Mark contract as completed"""
        if self.status != ContractStatus.ACTIVE:
            raise ValueError(f"Can only complete ACTIVE contracts, current: {self.status}")

        self.status = ContractStatus.COMPLETED
        self.successful_completions += 1

    def get_agent_commitments(self, agent_id: str) -> List[ResourceCommitment]:
        """Get all resource commitments for a specific agent"""
        return [c for c in self.resource_commitments if c.agent_id == agent_id]

    def get_mandatory_terms(self) -> List[ContractTerm]:
        """Get all mandatory terms"""
        return [t for t in self.terms if t.is_mandatory]

    def calculate_total_penalties(self) -> float:
        """Calculate total penalties from violations"""
        return sum(
            term.penalty_for_violation
            for violation in self.violations
            if not violation.resolved
            for term in self.terms
            if term.term_id in violation.evidence.get("violated_terms", [])
        )

    def amend(self, amendment: Dict[str, Any]) -> None:
        """Amend the contract"""
        if self.status != ContractStatus.ACTIVE:
            raise ValueError(f"Can only amend ACTIVE contracts, current: {self.status}")

        amendment["timestamp"] = datetime.utcnow()
        amendment["amendment_id"] = f"amend_{uuid.uuid4().hex[:8]}"
        self.amendment_history.append(amendment)

        # Reset signatures if amendment requires re-approval
        if amendment.get("requires_reapproval", False):
            self.current_signatures.clear()
            self.status = ContractStatus.NEGOTIATING
