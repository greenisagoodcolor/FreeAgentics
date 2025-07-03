"""
Comprehensive tests for Coalition Contract module
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from coalitions.contracts.coalition_contract import (
    CoalitionContract,
    ContractStatus,
    ContractTerm,
    ContractType,
    ContractViolation,
    ResourceCommitment,
    ViolationType,
)


class TestEnums:
    """Test enumeration types"""

    def test_contract_status_enum(self):
        """Test ContractStatus enum values"""
        assert ContractStatus.DRAFT.value == "draft"
        assert ContractStatus.PROPOSED.value == "proposed"
        assert ContractStatus.NEGOTIATING.value == "negotiating"
        assert ContractStatus.ACTIVE.value == "active"
        assert ContractStatus.SUSPENDED.value == "suspended"
        assert ContractStatus.COMPLETED.value == "completed"
        assert ContractStatus.TERMINATED.value == "terminated"
        assert len(ContractStatus) == 7

    def test_contract_type_enum(self):
        """Test ContractType enum values"""
        assert ContractType.RESOURCE_SHARING.value == "resource_sharing"
        assert ContractType.GOAL_ACHIEVEMENT.value == "goal_achievement"
        assert ContractType.MUTUAL_DEFENSE.value == "mutual_defense"
        assert ContractType.KNOWLEDGE_EXCHANGE.value == "knowledge_exchange"
        assert ContractType.TRADE_AGREEMENT.value == "trade_agreement"
        assert ContractType.EXPLORATION_PACT.value == "exploration_pact"
        assert len(ContractType) == 6

    def test_violation_type_enum(self):
        """Test ViolationType enum values"""
        assert ViolationType.RESOURCE_MISUSE.value == "resource_misuse"
        assert ViolationType.GOAL_ABANDONMENT.value == "goal_abandonment"
        assert ViolationType.INFORMATION_WITHHOLDING.value == "information_withholding"
        assert ViolationType.UNAUTHORIZED_ACTION.value == "unauthorized_action"
        assert ViolationType.DEADLINE_MISSED.value == "deadline_missed"
        assert ViolationType.TRUST_BREACH.value == "trust_breach"
        assert len(ViolationType) == 6


class TestContractTerm:
    """Test ContractTerm dataclass"""

    def test_default_initialization(self):
        """Test default term initialization"""
        term = ContractTerm()

        assert term.term_id.startswith("term_")
        assert term.description == ""
        assert term.category == "general"
        assert term.is_mandatory is True
        assert term.penalty_for_violation == 0.0
        assert term.verification_method == "manual"
        assert term.metadata == {}

    def test_custom_initialization(self):
        """Test custom term initialization"""
        metadata = {"threshold": 0.8, "frequency": "daily"}
        term = ContractTerm(
            description="Share resources equally",
            category="resource_sharing",
            is_mandatory=False,
            penalty_for_violation=100.0,
            verification_method="automated",
            metadata=metadata,
        )

        assert term.description == "Share resources equally"
        assert term.category == "resource_sharing"
        assert term.is_mandatory is False
        assert term.penalty_for_violation == 100.0
        assert term.verification_method == "automated"
        assert term.metadata == metadata

    def test_negative_penalty_validation(self):
        """Test that negative penalties are rejected"""
        with pytest.raises(ValueError, match="Penalty cannot be negative"):
            ContractTerm(penalty_for_violation=-10.0)

    def test_term_id_uniqueness(self):
        """Test that term IDs are unique"""
        terms = [ContractTerm() for _ in range(100)]
        term_ids = [t.term_id for t in terms]
        assert len(set(term_ids)) == 100  # All unique


class TestResourceCommitment:
    """Test ResourceCommitment dataclass"""

    def test_valid_initialization(self):
        """Test valid resource commitment"""
        commitment = ResourceCommitment(
            agent_id="agent_123",
            resource_type="energy",
            amount=100.0,
            minimum_contribution=10.0,
            maximum_contribution=200.0,
            contribution_schedule="periodic",
        )

        assert commitment.agent_id == "agent_123"
        assert commitment.resource_type == "energy"
        assert commitment.amount == 100.0
        assert commitment.minimum_contribution == 10.0
        assert commitment.maximum_contribution == 200.0
        assert commitment.contribution_schedule == "periodic"

    def test_negative_amount_validation(self):
        """Test negative amount validation"""
        with pytest.raises(ValueError, match="Resource amount cannot be negative"):
            ResourceCommitment(agent_id="agent_123", resource_type="energy", amount=-50.0)

    def test_negative_minimum_validation(self):
        """Test negative minimum contribution validation"""
        with pytest.raises(ValueError, match="Minimum contribution cannot be negative"):
            ResourceCommitment(
                agent_id="agent_123",
                resource_type="energy",
                amount=100.0,
                minimum_contribution=-10.0,
            )

    def test_max_less_than_min_validation(self):
        """Test maximum < minimum validation"""
        with pytest.raises(
            ValueError, match="Maximum contribution must be >= minimum contribution"
        ):
            ResourceCommitment(
                agent_id="agent_123",
                resource_type="energy",
                amount=100.0,
                minimum_contribution=50.0,
                maximum_contribution=30.0,
            )

    def test_amount_less_than_minimum_validation(self):
        """Test amount < minimum validation"""
        with pytest.raises(ValueError, match="Amount must be >= minimum contribution"):
            ResourceCommitment(
                agent_id="agent_123", resource_type="energy", amount=30.0, minimum_contribution=50.0
            )

    def test_amount_greater_than_maximum_validation(self):
        """Test amount > maximum validation"""
        with pytest.raises(ValueError, match="Amount must be <= maximum contribution"):
            ResourceCommitment(
                agent_id="agent_123",
                resource_type="energy",
                amount=300.0,
                maximum_contribution=200.0,
            )

    def test_no_maximum_constraint(self):
        """Test resource commitment without maximum constraint"""
        commitment = ResourceCommitment(
            agent_id="agent_123",
            resource_type="energy",
            amount=1000.0,
            minimum_contribution=10.0,
            maximum_contribution=None,
        )

        assert commitment.amount == 1000.0
        assert commitment.maximum_contribution is None


class TestContractViolation:
    """Test ContractViolation dataclass"""

    def test_default_initialization(self):
        """Test default violation initialization"""
        violation = ContractViolation()

        assert violation.violation_id.startswith("viol_")
        assert violation.contract_id == ""
        assert violation.violator_id == ""
        assert violation.violation_type == ViolationType.RESOURCE_MISUSE
        assert violation.description == ""
        assert isinstance(violation.timestamp, datetime)
        assert violation.severity == 0.5
        assert violation.evidence == {}
        assert violation.resolved is False
        assert violation.resolution_details is None

    def test_custom_initialization(self):
        """Test custom violation initialization"""
        evidence = {"witness": "agent_456", "amount": 50}
        violation = ContractViolation(
            contract_id="contract_123",
            violator_id="agent_789",
            violation_type=ViolationType.DEADLINE_MISSED,
            description="Failed to deliver resources on time",
            severity=0.8,
            evidence=evidence,
            resolved=True,
            resolution_details="Penalty applied",
        )

        assert violation.contract_id == "contract_123"
        assert violation.violator_id == "agent_789"
        assert violation.violation_type == ViolationType.DEADLINE_MISSED
        assert violation.severity == 0.8
        assert violation.evidence == evidence
        assert violation.resolved is True
        assert violation.resolution_details == "Penalty applied"

    def test_severity_validation(self):
        """Test severity validation"""
        # Test below range
        with pytest.raises(ValueError, match="Severity must be between 0.0 and 1.0"):
            ContractViolation(severity=-0.1)

        # Test above range
        with pytest.raises(ValueError, match="Severity must be between 0.0 and 1.0"):
            ContractViolation(severity=1.1)

        # Test boundaries
        violation1 = ContractViolation(severity=0.0)
        assert violation1.severity == 0.0

        violation2 = ContractViolation(severity=1.0)
        assert violation2.severity == 1.0


class TestCoalitionContract:
    """Test CoalitionContract class"""

    def test_minimal_valid_initialization(self):
        """Test minimal valid contract initialization"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        assert contract.contract_id.startswith("contract_")
        assert contract.coalition_id == "coalition_123"
        assert contract.initiator_id == "agent_001"
        assert contract.title == "Test Contract"
        assert contract.status == ContractStatus.DRAFT
        assert contract.initiator_id in contract.member_ids
        assert contract.compliance_score == 1.0

    def test_validation_errors(self):
        """Test initialization validation"""
        # Missing coalition_id
        with pytest.raises(ValueError, match="Coalition ID is required"):
            CoalitionContract(initiator_id="agent_001", title="Test")

        # Missing initiator_id
        with pytest.raises(ValueError, match="Initiator ID is required"):
            CoalitionContract(coalition_id="coalition_123", title="Test")

        # Missing title
        with pytest.raises(ValueError, match="Contract title is required"):
            CoalitionContract(coalition_id="coalition_123", initiator_id="agent_001")

        # Invalid compliance score
        with pytest.raises(ValueError, match="Compliance score must be between 0.0 and 1.0"):
            CoalitionContract(
                coalition_id="coalition_123",
                initiator_id="agent_001",
                title="Test",
                compliance_score=1.5,
            )

    def test_required_signatures_validation(self):
        """Test required signatures validation"""
        # Required signatures not subset of members
        with pytest.raises(ValueError, match="Required signatures must be subset of members"):
            CoalitionContract(
                coalition_id="coalition_123",
                initiator_id="agent_001",
                title="Test Contract",
                member_ids={"agent_001", "agent_002"},
                required_signatures={"agent_001", "agent_003"},  # agent_003 not a member
            )

    def test_add_term(self):
        """Test adding terms to contract"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        term = ContractTerm(description="Test term")
        contract.add_term(term)

        assert len(contract.terms) == 1
        assert contract.terms[0] == term

    def test_add_term_invalid_status(self):
        """Test adding terms in invalid status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.status = ContractStatus.ACTIVE

        with pytest.raises(
            ValueError, match="Cannot add terms to contract in ContractStatus.ACTIVE status"
        ):
            contract.add_term(ContractTerm())

    def test_add_resource_commitment(self):
        """Test adding resource commitments"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Test Contract",
            member_ids={"agent_001", "agent_002"},
        )

        commitment = ResourceCommitment(agent_id="agent_002", resource_type="energy", amount=100.0)
        contract.add_resource_commitment(commitment)

        assert len(contract.resource_commitments) == 1
        assert contract.resource_commitments[0] == commitment

    def test_add_resource_commitment_non_member(self):
        """Test adding commitment for non-member"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        commitment = ResourceCommitment(
            agent_id="agent_999", resource_type="energy", amount=100.0  # Not a member
        )

        with pytest.raises(ValueError, match="Agent agent_999 is not a member of this contract"):
            contract.add_resource_commitment(commitment)

    def test_propose_contract(self):
        """Test proposing a contract"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Test Contract",
            member_ids={"agent_001", "agent_002"},
        )

        # Add required term
        contract.add_term(ContractTerm(description="Test term"))

        contract.propose()

        assert contract.status == ContractStatus.PROPOSED
        assert contract.required_signatures == {"agent_001", "agent_002"}

    def test_propose_without_terms(self):
        """Test proposing contract without terms"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        with pytest.raises(ValueError, match="Contract must have at least one term"):
            contract.propose()

    def test_sign_contract(self):
        """Test signing a contract"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Test Contract",
            member_ids={"agent_001", "agent_002"},
            required_signatures={"agent_001", "agent_002"},
        )

        contract.add_term(ContractTerm())
        contract.propose()

        # First signature
        result = contract.sign("agent_001")
        assert result is False  # Not all signatures collected
        assert "agent_001" in contract.current_signatures

        # Second signature - should activate
        result = contract.sign("agent_002")
        assert result is True
        assert contract.status == ContractStatus.ACTIVE
        assert contract.effective_date is not None

    def test_sign_non_member(self):
        """Test non-member trying to sign"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()

        with pytest.raises(ValueError, match="Agent agent_999 is not a member"):
            contract.sign("agent_999")

    def test_sign_wrong_status(self):
        """Test signing in wrong status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        with pytest.raises(ValueError, match="Contract must be in PROPOSED status"):
            contract.sign("agent_001")

    def test_record_violation(self):
        """Test recording contract violations"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")  # Auto-activates since only one member

        violation = ContractViolation(
            violator_id="agent_001", violation_type=ViolationType.RESOURCE_MISUSE, severity=0.3
        )

        contract.record_violation(violation)

        assert len(contract.violations) == 1
        assert contract.violations[0].contract_id == contract.contract_id
        assert contract.compliance_score == 0.97  # 1.0 - (0.3 * 0.1)

    def test_record_violation_wrong_status(self):
        """Test recording violation in wrong status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        violation = ContractViolation()

        with pytest.raises(ValueError, match="Can only record violations for ACTIVE contracts"):
            contract.record_violation(violation)

    def test_compliance_score_floor(self):
        """Test compliance score doesn't go below 0"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")

        # Add many severe violations
        for _ in range(15):
            violation = ContractViolation(severity=1.0)
            contract.record_violation(violation)

        assert contract.compliance_score == 0.0  # Should not go negative

    def test_is_expired(self):
        """Test contract expiration check"""
        # Not expired - no expiration date
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        assert contract.is_expired() is False

        # Not expired - future date
        contract.expiration_date = datetime.utcnow() + timedelta(days=1)
        assert contract.is_expired() is False

        # Expired - past date
        contract.expiration_date = datetime.utcnow() - timedelta(days=1)
        assert contract.is_expired() is True

    def test_is_active(self):
        """Test is_active check"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        # Not active - wrong status
        assert contract.is_active() is False

        # Active
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")
        assert contract.is_active() is True

        # Not active - expired
        contract.expiration_date = datetime.utcnow() - timedelta(days=1)
        assert contract.is_active() is False

        # Not active - zero compliance
        contract.expiration_date = None
        contract.compliance_score = 0.0
        assert contract.is_active() is False

    def test_terminate_contract(self):
        """Test contract termination"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")

        contract.terminate("Breach of trust")

        assert contract.status == ContractStatus.TERMINATED
        assert len(contract.amendment_history) == 1
        assert contract.amendment_history[0]["type"] == "termination"
        assert contract.amendment_history[0]["reason"] == "Breach of trust"

    def test_terminate_wrong_status(self):
        """Test terminating in wrong status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        with pytest.raises(ValueError, match="Can only terminate ACTIVE or SUSPENDED contracts"):
            contract.terminate("Test reason")

    def test_complete_contract(self):
        """Test contract completion"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")

        contract.complete()

        assert contract.status == ContractStatus.COMPLETED
        assert contract.successful_completions == 1

    def test_complete_wrong_status(self):
        """Test completing in wrong status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        with pytest.raises(ValueError, match="Can only complete ACTIVE contracts"):
            contract.complete()

    def test_get_agent_commitments(self):
        """Test getting agent-specific commitments"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Test Contract",
            member_ids={"agent_001", "agent_002", "agent_003"},
        )

        # Add commitments
        contract.add_resource_commitment(
            ResourceCommitment(agent_id="agent_001", resource_type="energy", amount=100)
        )
        contract.add_resource_commitment(
            ResourceCommitment(agent_id="agent_002", resource_type="materials", amount=50)
        )
        contract.add_resource_commitment(
            ResourceCommitment(agent_id="agent_001", resource_type="time", amount=10)
        )

        agent1_commitments = contract.get_agent_commitments("agent_001")
        assert len(agent1_commitments) == 2
        assert all(c.agent_id == "agent_001" for c in agent1_commitments)

        agent2_commitments = contract.get_agent_commitments("agent_002")
        assert len(agent2_commitments) == 1
        assert agent2_commitments[0].resource_type == "materials"

        agent3_commitments = contract.get_agent_commitments("agent_003")
        assert len(agent3_commitments) == 0

    def test_get_mandatory_terms(self):
        """Test getting mandatory terms"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        # Add mixed terms
        contract.add_term(ContractTerm(description="Mandatory 1", is_mandatory=True))
        contract.add_term(ContractTerm(description="Optional 1", is_mandatory=False))
        contract.add_term(ContractTerm(description="Mandatory 2", is_mandatory=True))
        contract.add_term(ContractTerm(description="Optional 2", is_mandatory=False))

        mandatory_terms = contract.get_mandatory_terms()
        assert len(mandatory_terms) == 2
        assert all(t.is_mandatory for t in mandatory_terms)
        assert {t.description for t in mandatory_terms} == {"Mandatory 1", "Mandatory 2"}

    def test_calculate_total_penalties(self):
        """Test penalty calculation"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        # Add terms with penalties
        term1 = ContractTerm(penalty_for_violation=100.0)
        term2 = ContractTerm(penalty_for_violation=50.0)
        contract.add_term(term1)
        contract.add_term(term2)

        contract.propose()
        contract.sign("agent_001")

        # Add violations
        violation1 = ContractViolation(evidence={"violated_terms": [term1.term_id]})
        violation2 = ContractViolation(
            evidence={"violated_terms": [term2.term_id]}, resolved=True  # Resolved, shouldn't count
        )
        violation3 = ContractViolation(evidence={"violated_terms": [term1.term_id, term2.term_id]})

        contract.record_violation(violation1)
        contract.record_violation(violation2)
        contract.record_violation(violation3)

        total_penalties = contract.calculate_total_penalties()
        assert total_penalties == 250.0  # 100 + 0 + 150

    def test_amend_contract(self):
        """Test contract amendment"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")

        amendment = {
            "type": "term_modification",
            "description": "Updated penalty structure",
            "requires_reapproval": False,
        }

        contract.amend(amendment)

        assert len(contract.amendment_history) == 1
        assert contract.amendment_history[0]["type"] == "term_modification"
        assert "amendment_id" in contract.amendment_history[0]
        assert "timestamp" in contract.amendment_history[0]
        assert contract.status == ContractStatus.ACTIVE  # No reapproval needed

    def test_amend_with_reapproval(self):
        """Test amendment requiring reapproval"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Test Contract",
            member_ids={"agent_001", "agent_002"},
        )
        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")
        contract.sign("agent_002")

        amendment = {
            "type": "major_change",
            "description": "Complete restructuring",
            "requires_reapproval": True,
        }

        contract.amend(amendment)

        assert contract.status == ContractStatus.NEGOTIATING
        assert len(contract.current_signatures) == 0  # Signatures cleared

    def test_amend_wrong_status(self):
        """Test amending in wrong status"""
        contract = CoalitionContract(
            coalition_id="coalition_123", initiator_id="agent_001", title="Test Contract"
        )

        with pytest.raises(ValueError, match="Can only amend ACTIVE contracts"):
            contract.amend({"type": "test"})


class TestIntegration:
    """Integration tests for contract workflows"""

    def test_complete_contract_lifecycle(self):
        """Test complete contract lifecycle"""
        # Create contract
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Resource Sharing Agreement",
            description="Agreement to share energy and materials",
            contract_type=ContractType.RESOURCE_SHARING,
            member_ids={"agent_001", "agent_002", "agent_003"},
        )

        # Add terms
        contract.add_term(
            ContractTerm(description="Equal resource sharing", penalty_for_violation=100.0)
        )
        contract.add_term(
            ContractTerm(
                description="Weekly reporting", is_mandatory=False, penalty_for_violation=10.0
            )
        )

        # Add commitments
        for agent_id in contract.member_ids:
            contract.add_resource_commitment(
                ResourceCommitment(
                    agent_id=agent_id,
                    resource_type="energy",
                    amount=100.0,
                    minimum_contribution=50.0,
                    contribution_schedule="periodic",
                )
            )

        # Propose and sign
        contract.propose()
        assert contract.status == ContractStatus.PROPOSED

        for agent_id in ["agent_001", "agent_002", "agent_003"]:
            contract.sign(agent_id)

        assert contract.status == ContractStatus.ACTIVE
        assert contract.is_active()

        # Record some violations
        contract.record_violation(
            ContractViolation(
                violator_id="agent_002", violation_type=ViolationType.RESOURCE_MISUSE, severity=0.2
            )
        )

        assert contract.compliance_score < 1.0

        # Complete contract
        contract.complete()
        assert contract.status == ContractStatus.COMPLETED
        assert not contract.is_active()

    def test_contract_negotiation_workflow(self):
        """Test contract negotiation workflow"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Complex Agreement",
            member_ids={"agent_001", "agent_002"},
        )

        # Draft phase - add initial terms
        contract.add_term(ContractTerm(description="Initial term"))
        contract.propose()

        # One party signs
        contract.sign("agent_001")

        # Before other party signs, need to negotiate
        contract.status = ContractStatus.NEGOTIATING

        # Add more terms during negotiation
        contract.add_term(ContractTerm(description="Negotiated term"))

        # Re-propose
        contract.status = ContractStatus.DRAFT
        contract.propose()

        # Both parties sign
        contract.sign("agent_001")
        contract.sign("agent_002")

        assert contract.status == ContractStatus.ACTIVE

    @patch("coalitions.contracts.coalition_contract.datetime")
    def test_contract_expiration_and_renewal(self, mock_datetime):
        """Test contract expiration and renewal"""
        # Set current time
        current_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = current_time

        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Time-limited Contract",
            expiration_date=current_time + timedelta(days=30),
            auto_renew=True,
            renewal_period=timedelta(days=30),
        )

        contract.add_term(ContractTerm())
        contract.propose()
        contract.sign("agent_001")

        # Check not expired
        assert not contract.is_expired()
        assert contract.is_active()

        # Move time forward
        mock_datetime.utcnow.return_value = current_time + timedelta(days=31)

        # Now expired
        assert contract.is_expired()
        assert not contract.is_active()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_member_ids(self):
        """Test contract with only initiator"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Solo Contract",
            member_ids=set(),  # Empty initially
        )

        # Initiator should be added automatically
        assert "agent_001" in contract.member_ids
        assert len(contract.member_ids) == 1

    def test_unicode_in_strings(self):
        """Test Unicode handling in strings"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="合同 📄",  # Unicode title
            description="Multi-language: español, 中文, العربية",
        )

        term = ContractTerm(description="Special chars: ñ, ü, ß, ø")
        contract.add_term(term)

        assert contract.title == "合同 📄"
        assert "español" in contract.description
        assert "ñ" in contract.terms[0].description

    def test_concurrent_signatures(self):
        """Test handling of concurrent signatures"""
        contract = CoalitionContract(
            coalition_id="coalition_123",
            initiator_id="agent_001",
            title="Concurrent Test",
            member_ids={"agent_001", "agent_002"},
            required_signatures={"agent_001"},  # Only one required
        )

        contract.add_term(ContractTerm())
        contract.propose()

        # Both agents sign, but only one required
        result1 = contract.sign("agent_001")
        assert result1 is True
        assert contract.status == ContractStatus.ACTIVE

        # Second signature on already active contract
        with pytest.raises(ValueError, match="Contract must be in PROPOSED status"):
            contract.sign("agent_002")

    def test_large_scale_data(self):
        """Test with large amounts of data"""
        # Create contract with many members
        member_ids = {f"agent_{i:04d}" for i in range(1000)}
        contract = CoalitionContract(
            coalition_id="coalition_large",
            initiator_id="agent_0001",
            title="Large Scale Contract",
            member_ids=member_ids,
        )

        # Add many terms
        for i in range(100):
            contract.add_term(ContractTerm(description=f"Term {i}", penalty_for_violation=float(i)))

        # Add many commitments
        for i, agent_id in enumerate(list(member_ids)[:100]):
            contract.add_resource_commitment(
                ResourceCommitment(
                    agent_id=agent_id, resource_type=f"resource_{i % 10}", amount=float(i * 10)
                )
            )

        assert len(contract.terms) == 100
        assert len(contract.resource_commitments) == 100
        assert len(contract.member_ids) == 1000
