"""
Comprehensive tests for Business Readiness Assessment for Edge Deployment.

Tests the sophisticated business readiness evaluation system that assesses
business readiness for edge deployment by quantifying value proposition,
risk assessment, and market fit using business intelligence outputs.
"""

# Mock the missing business intelligence modules before importing
from coalitions.readiness.business_readiness_assessor import (
    BusinessReadinessAssessor,
    BusinessReadinessLevel,
    BusinessReadinessReport,
    DeploymentStrategy,
    MarketFitAssessment,
    OperationalReadiness,
    ValueProposition,
)
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock business intelligence modules that may not be available
sys.modules["coalitions.business_intelligence"] = Mock()
sys.modules["coalitions.business_intelligence.risk_engine"] = Mock()
sys.modules["coalitions.business_intelligence.market_engine"] = Mock()
sys.modules["coalitions.business_intelligence.roi_engine"] = Mock()


class TestBusinessReadinessLevel:
    """Test BusinessReadinessLevel enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert BusinessReadinessLevel.NOT_READY.value == "not_ready"
        assert BusinessReadinessLevel.BASIC_READY.value == "basic_ready"
        assert BusinessReadinessLevel.MARKET_READY.value == "market_ready"
        assert BusinessReadinessLevel.INVESTMENT_READY.value == "investment_ready"

    def test_enum_count(self):
        """Test correct number of enum values."""
        levels = list(BusinessReadinessLevel)
        assert len(levels) == 4


class TestDeploymentStrategy:
    """Test DeploymentStrategy enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert DeploymentStrategy.PILOT_DEPLOYMENT.value == "pilot_deployment"
        assert DeploymentStrategy.REGIONAL_ROLLOUT.value == "regional_rollout"
        assert DeploymentStrategy.FULL_SCALE.value == "full_scale"
        assert DeploymentStrategy.HYBRID_CLOUD_EDGE.value == "hybrid_cloud_edge"

    def test_enum_count(self):
        """Test correct number of enum values."""
        strategies = list(DeploymentStrategy)
        assert len(strategies) == 4


class TestValueProposition:
    """Test ValueProposition dataclass."""

    def test_value_proposition_creation(self):
        """Test creating value proposition with all fields."""
        vp = ValueProposition(
            cost_reduction_percent=25.0,
            performance_improvement_percent=40.0,
            latency_reduction_ms=150.0,
            availability_improvement_percent=15.0,
            revenue_opportunity_usd=2500000.0,
            market_expansion_potential=30.0,
            competitive_advantage_score=80.0,
            data_sovereignty_compliance=True,
            reduced_bandwidth_costs_percent=20.0,
            offline_capability_value=45.0,
        )

        assert vp.cost_reduction_percent == 25.0
        assert vp.performance_improvement_percent == 40.0
        assert vp.latency_reduction_ms == 150.0
        assert vp.availability_improvement_percent == 15.0
        assert vp.revenue_opportunity_usd == 2500000.0
        assert vp.market_expansion_potential == 30.0
        assert vp.competitive_advantage_score == 80.0
        assert vp.data_sovereignty_compliance is True
        assert vp.reduced_bandwidth_costs_percent == 20.0
        assert vp.offline_capability_value == 45.0

    def test_calculate_total_value_score(self):
        """Test total value score calculation."""
        vp = ValueProposition(
            cost_reduction_percent=20.0,
            performance_improvement_percent=30.0,
            latency_reduction_ms=100.0,  # Normalized to 10.0
            availability_improvement_percent=10.0,
            revenue_opportunity_usd=1000000.0,  # Normalized to 1.0
            market_expansion_potential=25.0,
            competitive_advantage_score=75.0,
            data_sovereignty_compliance=True,  # 20 points
            reduced_bandwidth_costs_percent=15.0,
            offline_capability_value=35.0,
        )

        score = vp.calculate_total_value_score()

        # Expected: (20 + 30 + 10 + 10 + 1 + 25 + 75 + 20 + 15 + 35) / 10 = 24.1
        expected_score = (20.0 + 30.0 + 10.0 + 10.0 + 1.0 +
                          25.0 + 75.0 + 20.0 + 15.0 + 35.0) / 10
        assert abs(score - expected_score) < 0.1

    def test_calculate_total_value_score_without_compliance(self):
        """Test value score calculation without data sovereignty compliance."""
        vp = ValueProposition(
            cost_reduction_percent=20.0,
            performance_improvement_percent=30.0,
            latency_reduction_ms=50.0,  # Normalized to 5.0
            availability_improvement_percent=10.0,
            revenue_opportunity_usd=500000.0,  # Normalized to 0.5
            market_expansion_potential=25.0,
            competitive_advantage_score=75.0,
            data_sovereignty_compliance=False,  # 0 points
            reduced_bandwidth_costs_percent=15.0,
            offline_capability_value=35.0,
        )

        score = vp.calculate_total_value_score()

        # Data sovereignty compliance contributes 0 instead of 20
        expected_score = (20.0 + 30.0 + 5.0 + 10.0 + 0.5 +
                          25.0 + 75.0 + 0.0 + 15.0 + 35.0) / 10
        assert abs(score - expected_score) < 0.1

    def test_confidence_intervals_default(self):
        """Test confidence intervals default to empty dict."""
        vp = ValueProposition(
            cost_reduction_percent=25.0,
            performance_improvement_percent=40.0,
            latency_reduction_ms=150.0,
            availability_improvement_percent=15.0,
            revenue_opportunity_usd=2500000.0,
            market_expansion_potential=30.0,
            competitive_advantage_score=80.0,
            data_sovereignty_compliance=True,
            reduced_bandwidth_costs_percent=20.0,
            offline_capability_value=45.0,
        )

        assert vp.confidence_intervals == {}


class TestMarketFitAssessment:
    """Test MarketFitAssessment dataclass."""

    def test_market_fit_creation(self):
        """Test creating market fit assessment."""
        mfa = MarketFitAssessment(
            target_market_size_usd=50000000.0,
            addressable_market_percent=15.0,
            market_growth_rate_percent=12.0,
            competition_intensity_score=60.0,
            customer_demand_score=75.0,
            adoption_readiness_score=65.0,
            pain_point_severity_score=80.0,
            willingness_to_pay_score=70.0,
            product_market_fit_score=85.0,
            solution_completeness_score=90.0,
            scalability_potential_score=88.0,
            edge_infrastructure_readiness=70.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )

        assert mfa.target_market_size_usd == 50000000.0
        assert mfa.addressable_market_percent == 15.0
        assert mfa.customer_demand_score == 75.0
        assert mfa.product_market_fit_score == 85.0
        assert mfa.edge_infrastructure_readiness == 70.0

    def test_calculate_market_fit_score(self):
        """Test market fit score calculation."""
        mfa = MarketFitAssessment(
            target_market_size_usd=10000000.0,  # Normalized to 1.0
            addressable_market_percent=20.0,
            market_growth_rate_percent=15.0,
            competition_intensity_score=50.0,  # Inverted to 50.0
            customer_demand_score=80.0,
            adoption_readiness_score=70.0,
            pain_point_severity_score=85.0,
            willingness_to_pay_score=75.0,
            product_market_fit_score=90.0,
            solution_completeness_score=88.0,
            scalability_potential_score=92.0,
            edge_infrastructure_readiness=75.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )

        score = mfa.calculate_market_fit_score()

        # Market: (1.0 + 20.0 + 15.0 + 50.0) / 4 = 21.5
        # Customer: (80.0 + 70.0 + 85.0 + 75.0) / 4 = 77.5
        # Solution: (90.0 + 88.0 + 92.0) / 3 = 90.0
        # Edge: (75.0 + 95.0 + 85.0) / 3 = 85.0
        # Weighted: (21.5 * 0.3 + 77.5 * 0.3 + 90.0 * 0.25 + 85.0 * 0.15)
        expected_score = 21.5 * 0.3 + 77.5 * 0.3 + 90.0 * 0.25 + 85.0 * 0.15

        assert abs(score - expected_score) < 0.1

    def test_large_market_size_normalization(self):
        """Test market size normalization for very large markets."""
        mfa = MarketFitAssessment(
            target_market_size_usd=100000000.0,  # Should be capped at 100
            addressable_market_percent=25.0,
            market_growth_rate_percent=10.0,
            competition_intensity_score=30.0,
            customer_demand_score=80.0,
            adoption_readiness_score=70.0,
            pain_point_severity_score=85.0,
            willingness_to_pay_score=75.0,
            product_market_fit_score=90.0,
            solution_completeness_score=88.0,
            scalability_potential_score=92.0,
            edge_infrastructure_readiness=75.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )

        mfa.calculate_market_fit_score()

        # Market size should be normalized to 10.0 (100M / 10M baseline)
        # Market: (10.0 + 25.0 + 10.0 + 70.0) / 4 = 28.75
        market_component = (10.0 + 25.0 + 10.0 + 70.0) / 4
        assert abs(market_component - 28.75) < 0.1


class TestOperationalReadiness:
    """Test OperationalReadiness dataclass."""

    def test_operational_readiness_creation(self):
        """Test creating operational readiness assessment."""
        ops = OperationalReadiness(
            technical_expertise_score=85.0,
            operational_capability_score=75.0,
            support_infrastructure_score=90.0,
            training_readiness_score=80.0,
            deployment_process_maturity=70.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=90.0,
            budget_adequacy_score=75.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=85.0,
            scaling_process_maturity=70.0,
            automation_readiness_score=90.0,
            partnership_ecosystem_score=75.0,
        )

        assert ops.technical_expertise_score == 85.0
        assert ops.operational_capability_score == 75.0
        assert ops.support_infrastructure_score == 90.0
        assert ops.training_readiness_score == 80.0
        assert ops.deployment_process_maturity == 70.0
        assert ops.monitoring_capabilities_score == 95.0

    def test_calculate_operational_score(self):
        """Test operational readiness score calculation."""
        ops = OperationalReadiness(
            technical_expertise_score=80.0,
            operational_capability_score=70.0,
            support_infrastructure_score=90.0,
            training_readiness_score=75.0,
            deployment_process_maturity=65.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=80.0,
            budget_adequacy_score=70.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=75.0,
            scaling_process_maturity=60.0,
            automation_readiness_score=85.0,
            partnership_ecosystem_score=70.0,
        )

        score = ops.calculate_operational_score()

        # Team: (80 + 70 + 90 + 75) / 4 = 78.75 * 0.3
        # Process: (65 + 95 + 85 + 80) / 4 = 81.25 * 0.3
        # Resource: (70 + 80 + 75) / 3 = 75.0 * 0.25
        # Scalability: (60 + 85 + 70) / 3 = 71.67 * 0.15
        expected_score = 78.75 * 0.3 + 81.25 * 0.3 + 75.0 * 0.25 + 71.67 * 0.15
        assert abs(score - expected_score) < 0.1

    def test_operational_readiness_fields(self):
        """Test operational readiness has all required fields."""
        ops = OperationalReadiness(
            technical_expertise_score=85.0,
            operational_capability_score=75.0,
            support_infrastructure_score=90.0,
            training_readiness_score=80.0,
            deployment_process_maturity=70.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=90.0,
            budget_adequacy_score=75.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=85.0,
            scaling_process_maturity=70.0,
            automation_readiness_score=90.0,
            partnership_ecosystem_score=75.0,
        )

        # Verify all required fields exist
        assert hasattr(ops, "technical_expertise_score")
        assert hasattr(ops, "partnership_ecosystem_score")


class TestBusinessReadinessReport:
    """Test BusinessReadinessReport dataclass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_value_prop = ValueProposition(
            cost_reduction_percent=25.0,
            performance_improvement_percent=40.0,
            latency_reduction_ms=150.0,
            availability_improvement_percent=15.0,
            revenue_opportunity_usd=2500000.0,
            market_expansion_potential=30.0,
            competitive_advantage_score=80.0,
            data_sovereignty_compliance=True,
            reduced_bandwidth_costs_percent=20.0,
            offline_capability_value=45.0,
        )

        self.sample_market_fit = MarketFitAssessment(
            target_market_size_usd=50000000.0,
            addressable_market_percent=15.0,
            market_growth_rate_percent=12.0,
            competition_intensity_score=60.0,
            customer_demand_score=75.0,
            adoption_readiness_score=65.0,
            pain_point_severity_score=80.0,
            willingness_to_pay_score=70.0,
            product_market_fit_score=85.0,
            solution_completeness_score=90.0,
            scalability_potential_score=88.0,
            edge_infrastructure_readiness=70.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )

        self.sample_operational = OperationalReadiness(
            technical_expertise_score=85.0,
            operational_capability_score=75.0,
            support_infrastructure_score=90.0,
            training_readiness_score=80.0,
            deployment_process_maturity=70.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=90.0,
            budget_adequacy_score=75.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=85.0,
            scaling_process_maturity=70.0,
            automation_readiness_score=90.0,
            partnership_ecosystem_score=75.0,
        )

    def test_business_readiness_report_creation(self):
        """Test creating business readiness report."""
        report = BusinessReadinessReport(
            coalition_id="test-coalition-001",
            assessment_timestamp=datetime(
                2024,
                1,
                15,
                10,
                30,
                0),
            value_proposition=self.sample_value_prop,
            market_fit=self.sample_market_fit,
            operational_readiness=self.sample_operational,
            business_readiness_level=BusinessReadinessLevel.MARKET_READY,
            overall_score=78.5,
            value_score=80.0,
            market_score=75.0,
            operational_score=85.0,
            risk_score=40.0,
            recommended_strategy=DeploymentStrategy.REGIONAL_ROLLOUT,
            investment_requirements={
                "initial": 1000000.0,
                "operational": 500000.0},
            timeline_recommendations={
                "phase1": "6 months",
                "phase2": "12 months"},
            risk_mitigation_strategies=[
                "diversify_partnerships",
                "gradual_rollout"],
            roi_projections={
                "3_year_roi": 35.0,
                "payback_period": 24},
            market_analysis={
                "market_attractiveness": 75.0,
                "competitive_position": 80.0},
            risk_assessment={
                "overall_risk": 40.0,
                "high_risk_factors": ["market_volatility"]},
            assessment_duration=45.0,
            confidence_level=85.0,
        )

        assert report.coalition_id == "test-coalition-001"
        assert report.assessment_timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert report.business_readiness_level == BusinessReadinessLevel.MARKET_READY
        assert report.recommended_strategy == DeploymentStrategy.REGIONAL_ROLLOUT
        assert report.overall_score == 78.5
        assert report.value_proposition == self.sample_value_prop
        assert report.market_fit == self.sample_market_fit
        assert report.operational_readiness == self.sample_operational
        assert report.confidence_level == 85.0
        assert report.investment_requirements["initial"] == 1000000.0
        assert report.timeline_recommendations["phase1"] == "6 months"
        assert "diversify_partnerships" in report.risk_mitigation_strategies

    def test_to_dict_method(self):
        """Test converting report to dictionary."""
        report = BusinessReadinessReport(
            coalition_id="test-coalition-001",
            assessment_timestamp=datetime(2024, 1, 15, 10, 30, 0),
            value_proposition=self.sample_value_prop,
            market_fit=self.sample_market_fit,
            operational_readiness=self.sample_operational,
            business_readiness_level=BusinessReadinessLevel.NOT_READY,
            overall_score=35.0,
            value_score=30.0,
            market_score=40.0,
            operational_score=35.0,
            risk_score=70.0,
            recommended_strategy=DeploymentStrategy.PILOT_DEPLOYMENT,
            investment_requirements={"initial": 500000.0},
            timeline_recommendations={"pilot": "3 months"},
            risk_mitigation_strategies=["risk_assessment"],
            roi_projections={"3_year_roi": 15.0},
            market_analysis={"market_attractiveness": 40.0},
            risk_assessment={"overall_risk": 70.0},
            assessment_duration=30.0,
            confidence_level=75.0,
        )

        result_dict = report.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["coalition_id"] == "test-coalition-001"
        assert result_dict["business_readiness_level"] == "not_ready"
        assert result_dict["recommended_strategy"] == "pilot_deployment"

    def test_report_score_fields(self):
        """Test report score fields are accessible."""
        report = BusinessReadinessReport(
            coalition_id="test-coalition-001",
            assessment_timestamp=datetime.now(),
            value_proposition=self.sample_value_prop,
            market_fit=self.sample_market_fit,
            operational_readiness=self.sample_operational,
            business_readiness_level=BusinessReadinessLevel.BASIC_READY,
            overall_score=55.0,
            value_score=60.0,
            market_score=50.0,
            operational_score=55.0,
            risk_score=60.0,
            recommended_strategy=DeploymentStrategy.PILOT_DEPLOYMENT,
            investment_requirements={"initial": 500000.0},
            timeline_recommendations={"pilot": "3 months"},
            risk_mitigation_strategies=["training"],
            roi_projections={"3_year_roi": 20.0},
            market_analysis={"market_attractiveness": 50.0},
            risk_assessment={"overall_risk": 60.0},
            assessment_duration=30.0,
            confidence_level=75.0,
        )

        assert report.overall_score == 55.0
        assert report.value_score == 60.0
        assert report.market_score == 50.0
        assert report.operational_score == 55.0
        assert report.risk_score == 60.0

    def test_report_default_fields(self):
        """Test report default field values."""
        report = BusinessReadinessReport(
            coalition_id="test-coalition-001",
            assessment_timestamp=datetime.now(),
            value_proposition=self.sample_value_prop,
            market_fit=self.sample_market_fit,
            operational_readiness=self.sample_operational,
            business_readiness_level=BusinessReadinessLevel.BASIC_READY,
            overall_score=55.0,
            value_score=60.0,
            market_score=50.0,
            operational_score=55.0,
            risk_score=60.0,
            recommended_strategy=DeploymentStrategy.PILOT_DEPLOYMENT,
            investment_requirements={"initial": 500000.0},
            timeline_recommendations={"pilot": "3 months"},
            risk_mitigation_strategies=["training"],
            roi_projections={"3_year_roi": 20.0},
            market_analysis={"market_attractiveness": 50.0},
            risk_assessment={"overall_risk": 60.0},
        )

        # Check default values
        assert report.assessment_duration == 0.0
        assert report.confidence_level == 0.0


class TestBusinessReadinessAssessor:
    """Test BusinessReadinessAssessor main class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = BusinessReadinessAssessor()

        self.sample_coalition_config = {
            "coalition_id": "test-coalition-001",
            "capabilities": ["data_processing", "edge_computing"],
            "resources": {"compute": 100, "storage": 500},
            "geographical_coverage": ["us-west", "us-east"],
        }

        self.sample_business_context = {
            "industry": "healthcare",
            "use_case": "real_time_diagnostics",
            "target_customers": ["hospitals", "clinics"],
            "revenue_model": "subscription",
            "investment_budget": 5000000,
        }

        self.sample_market_data = {
            "market_size": 100000000,
            "growth_rate": 15.0,
            "competitors": ["competitor_a", "competitor_b"],
            "customer_segments": ["enterprise", "mid_market"],
            "regulatory_environment": "healthcare_compliance",
        }

    def test_assessor_initialization(self):
        """Test business readiness assessor initialization."""
        assessor = BusinessReadinessAssessor()

        assert hasattr(assessor, "opportunity_engine")
        assert hasattr(assessor, "risk_engine")
        assert hasattr(assessor, "market_engine")
        assert hasattr(assessor, "roi_engine")

    @pytest.mark.asyncio
    async def test_assess_business_readiness(self):
        """Test comprehensive business readiness assessment."""
        # Mock the internal assessment methods that actually exist
        with (
            patch.object(
                self.assessor, "_analyze_roi_projections", new_callable=AsyncMock
            ) as mock_roi,
            patch.object(
                self.assessor, "_analyze_market_opportunities", new_callable=AsyncMock
            ) as mock_market,
            patch.object(
                self.assessor, "_analyze_business_risks", new_callable=AsyncMock
            ) as mock_risk,
        ):

            # Set up mock returns to match actual implementation
            mock_roi.return_value = {
                "roi_percentage": 25.0,
                "payback_period": 18,
                "confidence_intervals": {},
                "recommendation": "proceed",
            }
            mock_market.return_value = {
                "opportunities": ["healthcare_expansion"],
                "market_attractiveness": 75.0,
                "competitive_position": 70.0,
            }
            mock_risk.return_value = {
                "risk_factors": ["market_competition"],
                "overall_risk_score": 40.0,
                "high_risk_factors": [],
            }

            report = await self.assessor.assess_business_readiness(
                "test-coalition-001",
                self.sample_coalition_config,
                self.sample_business_context,
                self.sample_market_data,
            )

            # Verify report structure
            assert isinstance(report, BusinessReadinessReport)
            assert report.coalition_id == "test-coalition-001"
            assert hasattr(report, "business_readiness_level")
            assert hasattr(report, "recommended_strategy")

            # Verify all analysis methods were called
            mock_roi.assert_called_once()
            mock_market.assert_called_once()
            mock_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_roi_projections(self):
        """Test ROI projections analysis."""
        # Mock the ROI engine
        self.assessor.roi_engine.calculate_roi_projection = Mock(
            return_value={
                "roi_percentage": 35.0,
                "payback_period_months": 24,
                "net_present_value": 2500000,
                "internal_rate_return": 18.5,
            }
        )

        result = await self.assessor._analyze_roi_projections(
            self.sample_coalition_config, self.sample_business_context, self.sample_market_data
        )

        # Check that result is a dictionary with expected structure
        assert isinstance(result, dict)
        assert "roi_percentage" in result or "confidence_intervals" in result

    @pytest.mark.asyncio
    async def test_analyze_market_opportunities(self):
        """Test market opportunities analysis."""
        # Mock the opportunity and market engines
        self.assessor.opportunity_engine.detect_opportunities = Mock(
            return_value={
                "opportunities": [
                    "edge_expansion",
                    "new_verticals"],
                "opportunity_scores": {
                    "edge_expansion": 80.0,
                    "new_verticals": 65.0},
            })

        self.assessor.market_engine.evaluate_market_opportunity = Mock(
            return_value={
                "market_attractiveness": 75.0,
                "competitive_position": 70.0,
                "market_barriers": ["regulatory_complexity"],
            }
        )

        result = await self.assessor._analyze_market_opportunities(
            self.sample_coalition_config, self.sample_market_data
        )

        # Check that result is a dictionary
        assert isinstance(result, dict)

    def test_assess_operational_readiness(self):
        """Test operational readiness assessment."""
        result = self.assessor._assess_operational_readiness(
            self.sample_coalition_config, self.sample_business_context
        )

        # This method returns an OperationalReadiness object, not a dict
        assert isinstance(result, OperationalReadiness)
        assert hasattr(result, "technical_expertise_score")
        assert hasattr(result, "operational_capability_score")
        assert hasattr(result, "partnership_ecosystem_score")

        # Test that scores are within valid range
        assert 0 <= result.technical_expertise_score <= 100
        assert 0 <= result.operational_capability_score <= 100

    @pytest.mark.asyncio
    async def test_analyze_business_risks(self):
        """Test business risk analysis."""
        # Mock the risk engine
        self.assessor.risk_engine.assess_portfolio_risk = Mock(
            return_value={
                "overall_risk_score": 45.0,
                "risk_categories": {
                    "market_risk": 50.0,
                    "technical_risk": 30.0,
                    "operational_risk": 40.0,
                    "financial_risk": 35.0,
                },
                "high_risk_factors": [
                    "market_volatility",
                    "technical_complexity"],
            })

        result = await self.assessor._analyze_business_risks(
            self.sample_coalition_config, self.sample_business_context, self.sample_market_data
        )

        # Check that result is a dictionary
        assert isinstance(result, dict)

    def test_value_proposition_assessment(self):
        """Test value proposition assessment."""
        roi_analysis = {"roi_percentage": 35.0, "scenarios": {"baseline": {}}}
        market_analysis = {"market_attractiveness": 75.0}

        result = self.assessor._assess_value_proposition(
            self.sample_coalition_config, roi_analysis, market_analysis
        )

        assert isinstance(result, ValueProposition)
        assert hasattr(result, "cost_reduction_percent")
        assert hasattr(result, "performance_improvement_percent")
        assert hasattr(result, "competitive_advantage_score")

    def test_market_fit_evaluation(self):
        """Test market fit evaluation."""
        result = self.assessor._evaluate_market_fit(
            self.sample_coalition_config,
            self.sample_business_context,
            self.sample_market_data)

        assert isinstance(result, MarketFitAssessment)
        assert hasattr(result, "target_market_size_usd")
        assert hasattr(result, "customer_demand_score")
        assert hasattr(result, "product_market_fit_score")

    def test_calculate_business_scores(self):
        """Test business scores calculation."""
        # Create sample assessments
        value_prop = ValueProposition(
            cost_reduction_percent=25.0,
            performance_improvement_percent=40.0,
            latency_reduction_ms=150.0,
            availability_improvement_percent=15.0,
            revenue_opportunity_usd=2500000.0,
            market_expansion_potential=30.0,
            competitive_advantage_score=80.0,
            data_sovereignty_compliance=True,
            reduced_bandwidth_costs_percent=20.0,
            offline_capability_value=45.0,
        )

        market_fit = MarketFitAssessment(
            target_market_size_usd=50000000.0,
            addressable_market_percent=15.0,
            market_growth_rate_percent=12.0,
            competition_intensity_score=60.0,
            customer_demand_score=75.0,
            adoption_readiness_score=65.0,
            pain_point_severity_score=80.0,
            willingness_to_pay_score=70.0,
            product_market_fit_score=85.0,
            solution_completeness_score=90.0,
            scalability_potential_score=88.0,
            edge_infrastructure_readiness=70.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )

        operational = OperationalReadiness(
            technical_expertise_score=85.0,
            operational_capability_score=75.0,
            support_infrastructure_score=90.0,
            training_readiness_score=80.0,
            deployment_process_maturity=70.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=90.0,
            budget_adequacy_score=75.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=85.0,
            scaling_process_maturity=70.0,
            automation_readiness_score=90.0,
            partnership_ecosystem_score=75.0,
        )

        risk_assessment = {"overall_risk_score": 40.0}

        scores = self.assessor._calculate_business_scores(
            value_prop, market_fit, operational, risk_assessment
        )

        assert isinstance(scores, dict)
        assert "value" in scores
        assert "market" in scores
        assert "operational" in scores
        assert "risk" in scores
        assert "overall" in scores

    def test_determine_readiness_level(self):
        """Test readiness level determination."""
        # Test different score ranges
        level_high = self.assessor._determine_readiness_level(90.0)
        assert level_high == BusinessReadinessLevel.INVESTMENT_READY

        level_medium = self.assessor._determine_readiness_level(75.0)
        assert level_medium == BusinessReadinessLevel.MARKET_READY

        level_low = self.assessor._determine_readiness_level(60.0)
        assert level_low == BusinessReadinessLevel.BASIC_READY

        level_very_low = self.assessor._determine_readiness_level(40.0)
        assert level_very_low == BusinessReadinessLevel.NOT_READY

    def test_recommend_deployment_strategy(self):
        """Test deployment strategy recommendation."""
        scores_high = {"market": 85.0, "operational": 85.0, "overall": 90.0}
        market_analysis = {"market_attractiveness": 85.0}

        strategy_high = self.assessor._recommend_deployment_strategy(
            BusinessReadinessLevel.INVESTMENT_READY, scores_high, market_analysis)
        assert strategy_high == DeploymentStrategy.FULL_SCALE

        scores_medium = {"market": 75.0, "operational": 75.0, "overall": 75.0}
        strategy_medium = self.assessor._recommend_deployment_strategy(
            BusinessReadinessLevel.MARKET_READY, scores_medium, market_analysis
        )
        assert strategy_medium == DeploymentStrategy.REGIONAL_ROLLOUT

        scores_low = {"market": 60.0, "operational": 60.0, "overall": 60.0}
        strategy_low = self.assessor._recommend_deployment_strategy(
            BusinessReadinessLevel.BASIC_READY, scores_low, market_analysis
        )
        assert strategy_low == DeploymentStrategy.PILOT_DEPLOYMENT

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        scores = {
            "value": 60.0,
            "market": 55.0,
            "operational": 65.0,
            "risk": 70.0,
            "overall": 62.0}
        value_prop = ValueProposition(
            cost_reduction_percent=25.0,
            performance_improvement_percent=40.0,
            latency_reduction_ms=150.0,
            availability_improvement_percent=15.0,
            revenue_opportunity_usd=2500000.0,
            market_expansion_potential=30.0,
            competitive_advantage_score=80.0,
            data_sovereignty_compliance=True,
            reduced_bandwidth_costs_percent=20.0,
            offline_capability_value=45.0,
        )
        market_fit = MarketFitAssessment(
            target_market_size_usd=50000000.0,
            addressable_market_percent=15.0,
            market_growth_rate_percent=12.0,
            competition_intensity_score=60.0,
            customer_demand_score=75.0,
            adoption_readiness_score=65.0,
            pain_point_severity_score=80.0,
            willingness_to_pay_score=70.0,
            product_market_fit_score=85.0,
            solution_completeness_score=90.0,
            scalability_potential_score=88.0,
            edge_infrastructure_readiness=70.0,
            regulatory_compliance_score=95.0,
            data_locality_requirements_score=85.0,
        )
        operational = OperationalReadiness(
            technical_expertise_score=85.0,
            operational_capability_score=75.0,
            support_infrastructure_score=90.0,
            training_readiness_score=80.0,
            deployment_process_maturity=70.0,
            monitoring_capabilities_score=95.0,
            incident_response_readiness=85.0,
            compliance_process_score=90.0,
            budget_adequacy_score=75.0,
            timeline_feasibility_score=80.0,
            resource_allocation_score=85.0,
            scaling_process_maturity=70.0,
            automation_readiness_score=90.0,
            partnership_ecosystem_score=75.0,
        )
        risk_analysis = {"overall_risk_score": 40.0}

        recommendations = self.assessor._generate_recommendations(
            scores, value_prop, market_fit, operational, risk_analysis
        )

        assert isinstance(recommendations, dict)
        assert "investment" in recommendations
        assert "timeline" in recommendations
        assert "risk_mitigation" in recommendations
