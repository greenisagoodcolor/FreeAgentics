"""
Comprehensive tests for coalitions.readiness.comprehensive_readiness_integrator module.

Tests the unified system that integrates technical, business, and safety assessments
into a comprehensive deployment readiness report with prioritized recommendations.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from coalitions.readiness.comprehensive_readiness_integrator import (
    ActionPriority,
    ComprehensiveReadinessIntegrator,
    ComprehensiveReadinessReport,
    OverallReadinessLevel,
    ReadinessScore,
    RecommendedAction,
    RiskLevel,
)

# Mock the problematic hardware dependencies before importing
mock_hardware = MagicMock()
mock_offline_capabilities = MagicMock()
mock_resource_manager = MagicMock()

sys.modules["infrastructure.hardware.offline_capabilities"] = mock_offline_capabilities
sys.modules["infrastructure.hardware.resource_manager"] = mock_resource_manager


@pytest.fixture
def readiness_integrator():
    """Create a ComprehensiveReadinessIntegrator instance for testing."""
    return ComprehensiveReadinessIntegrator()


@pytest.fixture
def sample_coalition_config():
    """Create sample coalition configuration for testing."""
    return {
        "coalition_id": "test_coalition_001",
        "members": ["agent_001", "agent_002", "agent_003"],
        "capabilities": ["data_processing", "machine_learning", "optimization"],
        "resource_requirements": {
            "cpu_cores": 8,
            "memory_gb": 16,
            "storage_gb": 100,
        },
        "performance_targets": {
            "throughput_rps": 1000,
            "latency_ms": 50,
            "availability_percent": 99.9,
        },
    }


@pytest.fixture
def sample_business_context():
    """Create sample business context for testing."""
    return {
        "target_market": "enterprise_edge_computing",
        "market_size": 50000000,
        "competitive_advantage": "multi_agent_optimization",
        "revenue_model": "subscription_based",
        "pricing_strategy": "value_based",
        "customer_segments": ["manufacturing", "logistics", "smart_cities"],
        "expected_revenue": 2000000,
        "investment_required": 500000,
    }


@pytest.fixture
def sample_deployment_context():
    """Create sample deployment context for testing."""
    return {
        "target_environments": ["aws_outposts", "azure_stack_edge"],
        "security_requirements": ["encryption_at_rest", "network_isolation"],
        "compliance_requirements": ["gdpr", "hipaa"],
        "performance_requirements": {
            "max_latency_ms": 100,
            "min_throughput_rps": 500,
        },
        "availability_requirements": {
            "uptime_percent": 99.5,
            "maintenance_windows": "weekends_only",
        },
    }


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        "market_size": {
            "total_addressable_market": 100000000,
            "serviceable_addressable_market_percent": 15,
        },
        "competitive_landscape": {
            "intensity_score": 75,
            "differentiation_score": 80,
        },
        "customer_readiness": {
            "demand_score": 85,
            "adoption_score": 70,
        },
        "pricing": {
            "willingness_to_pay_score": 80,
        },
        "infrastructure": {
            "edge_readiness_score": 75,
        },
    }


class TestReadinessScore:
    """Test the ReadinessScore dataclass."""

    def test_readiness_score_creation(self):
        """Test creating a readiness score with all parameters."""
        score = ReadinessScore(
            overall_score=85.5,
            technical_score=90.0,
            business_score=80.0,
            safety_score=87.0,
            technical_weight=0.4,
            business_weight=0.3,
            safety_weight=0.3,
            confidence_level=0.85,
            assessment_reliability=0.90,
        )

        assert score.overall_score == 85.5
        assert score.technical_score == 90.0
        assert score.business_score == 80.0
        assert score.safety_score == 87.0
        assert score.technical_weight == 0.4
        assert score.business_weight == 0.3
        assert score.safety_weight == 0.3
        assert score.confidence_level == 0.85
        assert score.assessment_reliability == 0.90

    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        score = ReadinessScore(
            overall_score=0.0,  # This is overwritten by calculation
            technical_score=90.0,
            business_score=80.0,
            safety_score=85.0,
            technical_weight=0.35,
            business_weight=0.35,
            safety_weight=0.30,
        )

        weighted = score.calculate_weighted_score()
        expected = 90.0 * 0.35 + 80.0 * 0.35 + 85.0 * 0.30
        assert abs(weighted - expected) < 0.01

    def test_readiness_score_defaults(self):
        """Test readiness score with default weights."""
        score = ReadinessScore(
            overall_score=85.0,
            technical_score=90.0,
            business_score=80.0,
            safety_score=85.0,
        )

        # Check default weights
        assert score.technical_weight == 0.35
        assert score.business_weight == 0.35
        assert score.safety_weight == 0.30
        assert score.confidence_level == 0.0
        assert score.assessment_reliability == 0.0


class TestRecommendedAction:
    """Test the RecommendedAction dataclass."""

    def test_recommended_action_creation(self):
        """Test creating a recommended action."""
        action = RecommendedAction(
            action_id="action_001",
            category="technical",
            priority=ActionPriority.HIGH,
            title="Optimize Database Performance",
            description="Implement database indexing and query optimization",
            expected_impact="20% improvement in query response time",
            estimated_effort="medium",
            estimated_timeline="2 weeks",
            dependencies=["database_analysis"],
            success_criteria=["Query time < 100ms", "Index usage > 80%"],
            risk_if_ignored=RiskLevel.HIGH,
        )

        assert action.action_id == "action_001"
        assert action.category == "technical"
        assert action.priority == ActionPriority.HIGH
        assert action.title == "Optimize Database Performance"
        assert action.description == "Implement database indexing and query optimization"
        assert action.expected_impact == "20% improvement in query response time"
        assert action.estimated_effort == "medium"
        assert action.estimated_timeline == "2 weeks"
        assert action.dependencies == ["database_analysis"]
        assert action.success_criteria == ["Query time < 100ms", "Index usage > 80%"]
        assert action.risk_if_ignored == RiskLevel.HIGH

    def test_recommended_action_to_dict(self):
        """Test converting recommended action to dictionary."""
        action = RecommendedAction(
            action_id="action_001",
            category="business",
            priority=ActionPriority.URGENT,
            title="Market Analysis",
            description="Conduct market research",
            expected_impact="Better market positioning",
            estimated_effort="high",
            estimated_timeline="4 weeks",
            risk_if_ignored=RiskLevel.MEDIUM,
        )

        action_dict = action.to_dict()

        assert action_dict["action_id"] == "action_001"
        assert action_dict["priority"] == "urgent"
        assert action_dict["risk_if_ignored"] == "medium"
        assert isinstance(action_dict, dict)

    def test_recommended_action_defaults(self):
        """Test recommended action with default values."""
        action = RecommendedAction(
            action_id="minimal_action",
            category="safety",
            priority=ActionPriority.LOW,
            title="Basic Safety Check",
            description="Perform basic safety validation",
            expected_impact="Improved safety compliance",
            estimated_effort="low",
            estimated_timeline="1 week",
        )

        assert action.dependencies == []
        assert action.success_criteria == []
        assert action.risk_if_ignored == RiskLevel.MEDIUM


class TestComprehensiveReadinessIntegrator:
    """Test ComprehensiveReadinessIntegrator functionality."""

    def test_integrator_initialization(self, readiness_integrator):
        """Test that integrator initializes properly."""
        assert readiness_integrator.technical_validator is not None
        assert readiness_integrator.business_assessor is not None
        assert readiness_integrator.safety_verifier is not None

    @pytest.mark.asyncio
    async def test_assess_comprehensive_readiness_basic(
        self,
        readiness_integrator,
        sample_coalition_config,
        sample_business_context,
        sample_deployment_context,
    ):
        """Test basic comprehensive readiness assessment."""
        # Mock the individual assessment results
        mock_technical_report = Mock()
        mock_technical_report.overall_score = 85.0
        mock_technical_report.deployment_ready = True
        mock_technical_report.issues = []
        mock_technical_report.recommendations = ["Optimize performance"]

        mock_business_report = Mock()
        mock_business_report.overall_score = 80.0
        mock_business_report.business_readiness_level = Mock()
        mock_business_report.business_readiness_level.value = "market_ready"
        mock_business_report.timeline_recommendations = {"Q1": "Launch MVP"}

        mock_safety_report = Mock()
        mock_safety_report.overall_safety_score = 90.0
        mock_safety_report.deployment_approval = True
        mock_safety_report.critical_issues = []
        mock_safety_report.recommendations = ["Enhance monitoring"]

        # Mock the individual assessment methods
        readiness_integrator.technical_validator.assess_technical_readiness = AsyncMock(
            return_value=mock_technical_report
        )
        readiness_integrator.business_assessor.assess_business_readiness = AsyncMock(
            return_value=mock_business_report
        )
        readiness_integrator.safety_verifier.verify_safety_compliance = AsyncMock(
            return_value=mock_safety_report
        )

        # Perform assessment
        result = await readiness_integrator.assess_comprehensive_readiness(
            coalition_id="test_coalition",
            coalition_config=sample_coalition_config,
            business_context=sample_business_context,
            deployment_context=sample_deployment_context,
        )

        # Verify result
        assert isinstance(result, ComprehensiveReadinessReport)
        assert result.coalition_id == "test_coalition"
        assert isinstance(result.assessment_timestamp, datetime)
        assert isinstance(result.overall_readiness_level, OverallReadinessLevel)
        assert result.overall_score > 0
        assert result.assessment_duration >= 0

    def test_generate_default_market_data(self, readiness_integrator, sample_business_context):
        """Test default market data generation."""
        market_data = readiness_integrator._generate_default_market_data(sample_business_context)

        assert "market_size" in market_data
        assert "competitive_landscape" in market_data
        assert "customer_readiness" in market_data
        assert "pricing" in market_data
        assert "infrastructure" in market_data

        # Check specific values
        assert market_data["market_size"]["total_addressable_market"] == 50000000
        assert market_data["competitive_landscape"]["intensity_score"] == 60


class TestIntegrationMethods:
    """Test integration and analysis methods."""

    def test_determine_overall_readiness_enterprise_ready(self, readiness_integrator):
        """Test determination of enterprise-ready status."""
        # Mock high-quality reports
        mock_technical = Mock()
        mock_technical.deployment_ready = True
        mock_technical.readiness_level = Mock()
        mock_technical.readiness_level.value = "enterprise_ready"

        mock_business = Mock()
        mock_business.business_readiness_level = Mock()
        mock_business.business_readiness_level.value = "investment_ready"
        mock_business.overall_score = 90

        mock_safety = Mock()
        mock_safety.deployment_approval = True
        mock_safety.compliance_level = Mock()
        mock_safety.compliance_level.value = "fully_compliant"

        # Mock the enum comparisons by setting up proper enum instances
        from coalitions.readiness.business_readiness_assessor import BusinessReadinessLevel as BRL
        from coalitions.readiness.safety_compliance_verifier import SafetyComplianceLevel as SCL
        from coalitions.readiness.technical_readiness_validator import ReadinessLevel as TRL

        mock_technical.readiness_level = TRL.ENTERPRISE_READY
        mock_business.business_readiness_level = BRL.INVESTMENT_READY
        mock_safety.compliance_level = SCL.FULLY_COMPLIANT

        readiness = readiness_integrator._determine_overall_readiness(
            90.0, mock_technical, mock_business, mock_safety
        )

        assert readiness == OverallReadinessLevel.ENTERPRISE_READY

    def test_determine_overall_readiness_not_ready(self, readiness_integrator):
        """Test determination of not-ready status due to critical issues."""
        mock_technical = Mock()
        mock_technical.deployment_ready = False

        mock_business = Mock()
        from coalitions.readiness.business_readiness_assessor import BusinessReadinessLevel as BRL

        mock_business.business_readiness_level = BRL.NOT_READY

        mock_safety = Mock()
        mock_safety.deployment_approval = False

        readiness = readiness_integrator._determine_overall_readiness(
            85.0, mock_technical, mock_business, mock_safety
        )

        assert readiness == OverallReadinessLevel.NOT_READY

    def test_assess_integrated_risk_critical(self, readiness_integrator):
        """Test integrated risk assessment with critical safety issues."""
        mock_technical = Mock()
        mock_business = Mock()
        mock_safety = Mock()
        mock_safety.deployment_approval = False

        risk = readiness_integrator._assess_integrated_risk(
            mock_technical, mock_business, mock_safety
        )

        assert risk == "critical"

    def test_assess_integrated_risk_high(self, readiness_integrator):
        """Test integrated risk assessment with technical issues."""
        mock_technical = Mock()
        mock_technical.deployment_ready = False

        mock_business = Mock()
        mock_business.overall_score = 80

        mock_safety = Mock()
        mock_safety.deployment_approval = True

        risk = readiness_integrator._assess_integrated_risk(
            mock_technical, mock_business, mock_safety
        )

        assert risk == "high"

    def test_assess_integrated_risk_low(self, readiness_integrator):
        """Test integrated risk assessment with good scores."""
        mock_technical = Mock()
        mock_technical.deployment_ready = True
        mock_technical.overall_score = 90

        mock_business = Mock()
        mock_business.overall_score = 85

        mock_safety = Mock()
        mock_safety.deployment_approval = True
        mock_safety.overall_safety_score = 90

        risk = readiness_integrator._assess_integrated_risk(
            mock_technical, mock_business, mock_safety
        )

        assert risk == "low"

    def test_generate_integrated_recommendations_critical_safety(self, readiness_integrator):
        """Test recommendation generation with critical safety issues."""
        mock_technical = Mock()
        mock_technical.deployment_ready = True
        mock_technical.recommendations = ["Tech rec 1", "Tech rec 2"]

        mock_business = Mock()
        mock_business.overall_score = 80
        mock_business.timeline_recommendations = {"Q1": "Business rec 1"}

        mock_safety = Mock()
        mock_safety.deployment_approval = False
        mock_safety.recommendations = ["Safety rec 1", "Safety rec 2", "Safety rec 3"]

        recommendations = readiness_integrator._generate_integrated_recommendations(
            mock_technical, mock_business, mock_safety
        )

        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert len(recommendations) <= 10

    def test_generate_integrated_recommendations_no_issues(self, readiness_integrator):
        """Test recommendation generation with no critical issues."""
        mock_technical = Mock()
        mock_technical.deployment_ready = True
        mock_technical.recommendations = []

        mock_business = Mock()
        mock_business.overall_score = 85
        mock_business.timeline_recommendations = {}

        mock_safety = Mock()
        mock_safety.deployment_approval = True
        mock_safety.recommendations = []

        recommendations = readiness_integrator._generate_integrated_recommendations(
            mock_technical, mock_business, mock_safety
        )

        assert len(recommendations) > 0
        assert any("Optimize deployment" in rec for rec in recommendations)


class TestReportingAndSerialization:
    """Test report generation and serialization functionality."""

    def test_comprehensive_readiness_report_to_dict(self):
        """Test converting comprehensive report to dictionary."""
        # Create mock component reports
        mock_technical = Mock()
        mock_technical.to_dict.return_value = {"technical": "data"}

        mock_business = Mock()
        mock_business.to_dict.return_value = {"business": "data"}

        mock_safety = Mock()
        mock_safety.to_dict.return_value = {"safety": "data"}

        # Create comprehensive report
        report = ComprehensiveReadinessReport(
            coalition_id="test_coalition",
            assessment_timestamp=datetime(2024, 1, 1, 12, 0, 0),
            overall_readiness_level=OverallReadinessLevel.DEPLOYMENT_READY,
            overall_score=85.5,
            technical_score=90.0,
            business_score=80.0,
            safety_score=87.0,
            technical_assessment=mock_technical,
            business_assessment=mock_business,
            safety_assessment=mock_safety,
            deployment_ready=True,
            critical_issues=["Issue 1"],
            recommendations=["Rec 1", "Rec 2"],
            risk_level="medium",
            assessment_duration=12.5,
        )

        report_dict = report.to_dict()

        assert report_dict["coalition_id"] == "test_coalition"
        assert report_dict["overall_readiness_level"] == "deployment_ready"
        assert report_dict["overall_score"] == 85.5
        assert report_dict["deployment_ready"] is True
        assert report_dict["risk_level"] == "medium"
        assert report_dict["assessment_duration"] == 12.5

    def test_save_report(self, readiness_integrator):
        """Test saving report to file."""
        # Create mock report
        mock_report = Mock()
        mock_report.to_dict.return_value = {
            "coalition_id": "test",
            "overall_score": 85.0,
        }

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Save report
            readiness_integrator.save_report(mock_report, temp_path)

            # Verify file was created and contains correct data
            assert temp_path.exists()
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert data["coalition_id"] == "test"
            assert data["overall_score"] == 85.0

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_generate_executive_summary(self, readiness_integrator):
        """Test generating executive summary."""
        # Create mock report with proper types for formatting
        report = Mock()
        report.coalition_id = "test_coalition"
        report.assessment_timestamp = datetime(2024, 1, 15, 10, 30, 0)
        report.overall_readiness_level = OverallReadinessLevel.DEPLOYMENT_READY
        report.overall_score = 85.5
        report.deployment_ready = True
        report.critical_issues = ["Issue 1", "Issue 2"]
        report.recommendations = ["Rec 1", "Rec 2", "Rec 3", "Rec 4", "Rec 5", "Rec 6"]
        report.risk_level = "medium"
        report.technical_score = 90.0
        report.business_score = 80.0
        report.safety_score = 87.0
        report.assessment_duration = 12.5  # Add missing duration as float

        summary = readiness_integrator.generate_executive_summary(report)

        assert summary["coalition_id"] == "test_coalition"
        assert summary["assessment_date"] == "2024-01-15"
        assert summary["overall_readiness"] == "deployment_ready"
        assert summary["readiness_score"] == "85.5/100"
        assert summary["deployment_ready"] is True
        assert summary["critical_issues_count"] == 2
        assert summary["recommendations_count"] == 6
        assert summary["risk_level"] == "medium"
        assert len(summary["key_recommendations"]) == 5

        # Check component scores
        assert summary["component_scores"]["technical"] == "90.0/100"
        assert summary["component_scores"]["business"] == "80.0/100"
        assert summary["component_scores"]["safety"] == "87.0/100"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_assess_with_none_market_data(
        self,
        readiness_integrator,
        sample_coalition_config,
        sample_business_context,
        sample_deployment_context,
    ):
        """Test assessment with None market data (should generate defaults)."""
        # Mock individual assessments
        readiness_integrator.technical_validator.assess_technical_readiness = AsyncMock(
            return_value=Mock(
                overall_score=85, deployment_ready=True, issues=[], recommendations=[]
            )
        )
        readiness_integrator.business_assessor.assess_business_readiness = AsyncMock(
            return_value=Mock(
                overall_score=80, business_readiness_level=Mock(), timeline_recommendations={}
            )
        )
        readiness_integrator.safety_verifier.verify_safety_compliance = AsyncMock(
            return_value=Mock(
                overall_safety_score=90,
                deployment_approval=True,
                critical_issues=[],
                recommendations=[],
            )
        )

        # Should not raise exception and should generate default market data
        result = await readiness_integrator.assess_comprehensive_readiness(
            coalition_id="test_coalition",
            coalition_config=sample_coalition_config,
            business_context=sample_business_context,
            deployment_context=sample_deployment_context,
            market_data=None,  # Explicitly None
        )

        assert isinstance(result, ComprehensiveReadinessReport)

    def test_integration_with_missing_business_context(self, readiness_integrator):
        """Test default market data generation with minimal business context."""
        minimal_context = {}

        market_data = readiness_integrator._generate_default_market_data(minimal_context)

        # Should handle missing keys gracefully
        assert "market_size" in market_data
        # Default
        assert market_data["market_size"]["total_addressable_market"] == 10000000

    def test_determine_overall_readiness_edge_scores(self, readiness_integrator):
        """Test overall readiness determination with edge case scores."""
        # Create mocks for boundary testing
        mock_technical = Mock()
        mock_technical.deployment_ready = True

        mock_business = Mock()
        mock_business.overall_score = 70  # Exactly at boundary

        mock_safety = Mock()
        mock_safety.deployment_approval = True

        # Test with score exactly at boundary
        readiness = readiness_integrator._determine_overall_readiness(
            75.0, mock_technical, mock_business, mock_safety
        )

        assert readiness == OverallReadinessLevel.DEPLOYMENT_READY

    def test_recommendations_with_empty_reports(self, readiness_integrator):
        """Test recommendation generation with empty report data."""
        mock_technical = Mock()
        mock_technical.deployment_ready = True
        mock_technical.recommendations = []

        mock_business = Mock()
        mock_business.overall_score = 85
        mock_business.timeline_recommendations = {}

        mock_safety = Mock()
        mock_safety.deployment_approval = True
        mock_safety.recommendations = []

        recommendations = readiness_integrator._generate_integrated_recommendations(
            mock_technical, mock_business, mock_safety
        )

        # Should still generate optimization recommendations
        assert len(recommendations) > 0
        assert any("Optimize deployment" in rec for rec in recommendations)


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_integration_workflow(
        self,
        readiness_integrator,
        sample_coalition_config,
        sample_business_context,
        sample_deployment_context,
        sample_market_data,
    ):
        """Test complete end-to-end integration workflow."""
        # Mock comprehensive assessment results with proper serialization
        mock_technical = Mock()
        mock_technical.overall_score = 88.0
        mock_technical.deployment_ready = True
        mock_technical.issues = []
        mock_technical.recommendations = ["Optimize caching", "Improve monitoring"]
        mock_technical.to_dict.return_value = {"technical_data": "mock_technical"}

        mock_business = Mock()
        mock_business.overall_score = 82.0
        from coalitions.readiness.business_readiness_assessor import BusinessReadinessLevel

        mock_business.business_readiness_level = BusinessReadinessLevel.MARKET_READY
        mock_business.timeline_recommendations = {"Q1": "Beta launch", "Q2": "Full launch"}
        mock_business.to_dict.return_value = {"business_data": "mock_business"}

        mock_safety = Mock()
        mock_safety.overall_safety_score = 91.0
        mock_safety.deployment_approval = True
        mock_safety.critical_issues = []
        mock_safety.recommendations = ["Enhance logging", "Add audit trail"]
        mock_safety.to_dict.return_value = {"safety_data": "mock_safety"}
        from coalitions.readiness.safety_compliance_verifier import SafetyComplianceLevel

        mock_safety.compliance_level = SafetyComplianceLevel.FULLY_COMPLIANT

        # Mock individual assessment methods
        readiness_integrator.technical_validator.assess_technical_readiness = AsyncMock(
            return_value=mock_technical
        )
        readiness_integrator.business_assessor.assess_business_readiness = AsyncMock(
            return_value=mock_business
        )
        readiness_integrator.safety_verifier.verify_safety_compliance = AsyncMock(
            return_value=mock_safety
        )

        # Perform assessment
        result = await readiness_integrator.assess_comprehensive_readiness(
            coalition_id="integration_test_coalition",
            coalition_config=sample_coalition_config,
            business_context=sample_business_context,
            deployment_context=sample_deployment_context,
            market_data=sample_market_data,
        )

        # Verify comprehensive results
        assert result.coalition_id == "integration_test_coalition"
        assert result.overall_score > 80.0  # Should be high with good component scores
        assert result.deployment_ready is True
        assert result.overall_readiness_level in [
            OverallReadinessLevel.DEPLOYMENT_READY,
            OverallReadinessLevel.ENTERPRISE_READY,
        ]
        assert len(result.critical_issues) == 0
        assert len(result.recommendations) > 0
        assert result.risk_level == "low"

        # Test executive summary generation
        summary = readiness_integrator.generate_executive_summary(result)
        assert summary["deployment_ready"] is True
        assert summary["critical_issues_count"] == 0

        # Test report saving
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            readiness_integrator.save_report(result, temp_path)
            assert temp_path.exists()

            # Verify saved data
            with open(temp_path, "r") as f:
                saved_data = json.load(f)
            assert saved_data["coalition_id"] == "integration_test_coalition"
            assert saved_data["deployment_ready"] is True

        finally:
            if temp_path.exists():
                temp_path.unlink()
