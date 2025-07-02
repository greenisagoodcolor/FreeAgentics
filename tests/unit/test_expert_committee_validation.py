"""
Comprehensive tests for coalitions.formation.expert_committee_validation module.

Tests the expert committee business intelligence and safety validation framework
that implements mandatory review and validation by Multi-Agent and Architecture experts.
"""

import pytest

from coalitions.formation.expert_committee_validation import (
    ExpertCommitteeValidation,
    ExpertDomain,
    ExpertProfile,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture
def expert_committee():
    """Create an ExpertCommitteeValidation instance for testing."""
    return ExpertCommitteeValidation()


@pytest.fixture
def sample_business_model():
    """Create a sample coalition business model for testing."""
    return {
        "executive_summary": {
            "total_business_value": 1500000.0,
            "confidence_level": 0.87,
            "investment_readiness": True,
        },
        "business_metrics": {
            "synergy_analysis": {
                "efficiency_gain": 0.25,
                "cost_reduction": 0.15,
            },
        },
        "member_analysis": {
            "optimal_size": 3,
            "capability_matrix": "detailed_analysis",
        },
        "strategic_analysis": {
            "market_position": "strong",
            "competitive_advantage": "significant",
        },
        "risk_assessment": {
            "overall_risk": "low",
            "mitigation_strategies": ["strategy1", "strategy2"],
        },
        "financial_projections": {
            "revenue_forecast": [100000, 200000, 300000],
            "growth_rate": 0.25,
        },
        "appendix": {
            "technical_specifications": {
                "architecture": "microservices",
                "scalability": "horizontal",
            },
        },
    }


@pytest.fixture
def sample_system_artifacts():
    """Create sample system artifacts for testing."""
    return {
        "agent_protocols": "detailed_protocol_spec",
        "coalition_algorithms": "mathematical_model",
        "safety_constraints": "comprehensive_safety_spec",
        "performance_benchmarks": "benchmark_results",
    }


class TestExpertProfile:
    """Test the ExpertProfile dataclass."""

    def test_expert_profile_creation(self):
        """Test creating an expert profile with all fields."""
        expert = ExpertProfile(
            expert_id="test_expert",
            name="Test Expert",
            domain=ExpertDomain.MULTI_AGENT_SYSTEMS,
            credentials=["PhD", "Industry Expert"],
            validation_authority=["agent_design", "safety_protocols"],
            contact_info={"email": "test@example.com"},
            active=True,
        )

        assert expert.expert_id == "test_expert"
        assert expert.name == "Test Expert"
        assert expert.domain == ExpertDomain.MULTI_AGENT_SYSTEMS
        assert expert.credentials == ["PhD", "Industry Expert"]
        assert expert.validation_authority == [
            "agent_design", "safety_protocols"]
        assert expert.contact_info == {"email": "test@example.com"}
        assert expert.active is True

    def test_expert_profile_defaults(self):
        """Test expert profile with default values."""
        expert = ExpertProfile(
            expert_id="minimal_expert",
            name="Minimal Expert",
            domain=ExpertDomain.SOFTWARE_ARCHITECTURE,
        )

        assert expert.credentials == []
        assert expert.validation_authority == []
        assert expert.contact_info == {}
        assert expert.active is True


class TestExpertCommitteeInitialization:
    """Test ExpertCommitteeValidation initialization."""

    def test_committee_initialization(self, expert_committee):
        """Test that expert committee initializes properly."""
        assert len(expert_committee.experts) == 6  # 6 predefined experts
        assert len(expert_committee.validation_criteria) == 4  # 4 domains
        assert expert_committee.review_history == []
        assert expert_committee.active_reviews == {}

    def test_expert_committee_members(self, expert_committee):
        """Test that all required experts are initialized."""
        expert_ids = set(expert_committee.experts.keys())
        expected_experts = {
            "harrison_chase",
            "joao_moura",
            "jerry_liu",
            "robert_martin",
            "rich_hickey",
            "kent_beck",
        }
        assert expert_ids == expected_experts

    def test_expert_domains_coverage(self, expert_committee):
        """Test that experts cover all required domains."""
        expert_domains = [
            expert.domain for expert in expert_committee.experts.values()]

        # Should have multiple experts in key domains
        mas_experts = [d for d in expert_domains if d ==
                       ExpertDomain.MULTI_AGENT_SYSTEMS]
        arch_experts = [d for d in expert_domains if d ==
                        ExpertDomain.SOFTWARE_ARCHITECTURE]

        assert len(mas_experts) == 3  # Harrison, João, Jerry
        assert len(arch_experts) == 3  # Robert, Rich, Kent


class TestReviewSubmission:
    """Test review submission functionality."""

    def test_submit_for_review_basic(
            self,
            expert_committee,
            sample_business_model):
        """Test basic review submission."""
        review_id = expert_committee.submit_for_review(
            coalition_id="test_coalition",
            review_type="business_intelligence",
            artifacts=sample_business_model,
        )

        assert review_id.startswith("review_test_coalition_")
        assert review_id in expert_committee.active_reviews

        review = expert_committee.active_reviews[review_id]
        assert review.coalition_id == "test_coalition"
        assert review.review_type == "business_intelligence"
        assert review.overall_status == ValidationStatus.PENDING

    def test_submit_multiple_reviews(
            self,
            expert_committee,
            sample_business_model):
        """Test submitting multiple reviews."""
        review_id1 = expert_committee.submit_for_review(
            coalition_id="coalition1",
            review_type="safety",
            artifacts=sample_business_model,
        )

        review_id2 = expert_committee.submit_for_review(
            coalition_id="coalition2",
            review_type="architecture",
            artifacts=sample_business_model,
        )

        assert len(expert_committee.active_reviews) == 2
        assert review_id1 != review_id2
        assert expert_committee.active_reviews[review_id1].coalition_id == "coalition1"
        assert expert_committee.active_reviews[review_id2].coalition_id == "coalition2"


class TestExpertValidationSimulation:
    """Test expert validation simulation functionality."""

    def test_simulate_expert_validation_basic(
        self, expert_committee, sample_business_model, sample_system_artifacts
    ):
        """Test basic expert validation simulation."""
        review_id = expert_committee.submit_for_review(
            coalition_id="test_coalition",
            review_type="full",
            artifacts=sample_business_model,
        )

        completed_review = expert_committee.simulate_expert_validation(
            review_id, sample_business_model, sample_system_artifacts
        )

        assert completed_review.review_id == review_id
        assert completed_review.completion_timestamp is not None
        assert len(
            completed_review.expert_results) == len(
            expert_committee.experts)
        assert completed_review.consensus_score > 0
        assert completed_review.overall_status in [
            ValidationStatus.APPROVED,
            ValidationStatus.CONDITIONAL_APPROVAL,
            ValidationStatus.REQUIRES_REVISION,
        ]

    def test_validation_with_high_quality_artifacts(
        self, expert_committee, sample_business_model, sample_system_artifacts
    ):
        """Test validation with high-quality artifacts should get approval."""
        review_id = expert_committee.submit_for_review(
            coalition_id="high_quality",
            review_type="business_intelligence",
            artifacts=sample_business_model,
        )

        completed_review = expert_committee.simulate_expert_validation(
            review_id, sample_business_model, sample_system_artifacts
        )

        # High-quality artifacts should get good scores
        assert completed_review.consensus_score >= 0.7
        assert completed_review.overall_status in [
            ValidationStatus.APPROVED,
            ValidationStatus.CONDITIONAL_APPROVAL,
        ]

    def test_simulate_validation_nonexistent_review(self, expert_committee):
        """Test simulating validation for non-existent review."""
        with pytest.raises(ValueError, match="Review nonexistent not found"):
            expert_committee.simulate_expert_validation("nonexistent", {}, {})


class TestExpertEvaluationMethods:
    """Test individual expert evaluation methods."""

    def test_evaluate_multi_agent_aspects(
            self, expert_committee, sample_business_model):
        """Test multi-agent system evaluation."""
        score = expert_committee._evaluate_multi_agent_aspects(
            sample_business_model, {})

        # Should score well with complete business model
        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # Should be high with comprehensive model

    def test_evaluate_business_intelligence(
            self, expert_committee, sample_business_model):
        """Test business intelligence evaluation."""
        score = expert_committee._evaluate_business_intelligence(
            sample_business_model)

        assert 0.0 <= score <= 1.0
        # Should score well with complete executive summary and metrics
        assert score >= 0.8


class TestReviewStatusAndQueries:
    """Test review status and query functionality."""

    def test_get_review_status_active(
            self, expert_committee, sample_business_model):
        """Test getting status of active review."""
        review_id = expert_committee.submit_for_review(
            coalition_id="status_test",
            review_type="safety",
            artifacts=sample_business_model,
        )

        status = expert_committee.get_review_status(review_id)
        assert status is not None
        assert status.review_id == review_id
        assert status.overall_status == ValidationStatus.PENDING

    def test_get_review_status_nonexistent(self, expert_committee):
        """Test getting status of non-existent review."""
        status = expert_committee.get_review_status("nonexistent")
        assert status is None


class TestTaskMathematicalValidation:
    """Test Task 37 mathematical rigor validation."""

    def test_validate_task_37_basic(self, expert_committee):
        """Test basic Task 37 mathematical validation."""
        implementation_details = {
            "coalition_formation_algorithm": "Shapley value based",
            "business_value_calculation": "NPV with Monte Carlo",
            "active_inference_implementation": "Sophisticated prediction",
            "member_scoring_system": "Multi-criteria analysis",
            "strategic_analysis_engine": "Game theory based",
            "risk_assessment_framework": "Comprehensive evaluation",
        }

        result = expert_committee.validate_task_37_mathematical_rigor(
            implementation_details)

        assert isinstance(result, ValidationResult)
        assert result.expert_id == "committee_mathematical_review"
        assert result.domain == ExpertDomain.MULTI_AGENT_SYSTEMS
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.feedback, str)
        assert isinstance(result.recommendations, list)


class TestIntegrationScenarios:
    """Test integrated scenarios combining multiple features."""

    def test_complete_validation_workflow(
        self, expert_committee, sample_business_model, sample_system_artifacts
    ):
        """Test complete end-to-end validation workflow."""
        # Submit for review
        review_id = expert_committee.submit_for_review(
            coalition_id="integration_test",
            review_type="full",
            artifacts=sample_business_model,
        )

        # Verify submission
        assert review_id in expert_committee.active_reviews

        # Simulate validation
        completed_review = expert_committee.simulate_expert_validation(
            review_id, sample_business_model, sample_system_artifacts
        )

        # Verify completion
        assert completed_review.completion_timestamp is not None
        assert review_id not in expert_committee.active_reviews
        assert len(expert_committee.review_history) == 1

        # Check status
        status = expert_committee.get_review_status(review_id)
        assert status is not None
        assert status.review_id == review_id
