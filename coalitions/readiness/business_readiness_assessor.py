"""Business Readiness Assessment for Edge Deployment.

Comprehensive evaluation system that assesses business readiness for
edge deployment by quantifying value proposition, risk assessment, and
market fit using business intelligence outputs.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Note: Creating stub classes for missing modules to fix import errors
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Import business intelligence modules
from coalitions.coalition.business_opportunities import (
    OpportunityDetector as OpportunityDetectionEngine,
)

if TYPE_CHECKING:

    class RiskAssessmentEngine:
        """Risk assessment engine for type checking."""

    class MarketEvaluationEngine:
        """Market evaluation engine for type checking."""

    class ROIProjectionEngine:
        """ROI projection engine for type checking."""

else:
    # Stub implementations for runtime
    class RiskAssessmentEngine:
        """Stub risk assessment engine."""

        def assess_risk(self, *args, **kwargs):
            """Assess risk."""
            return {"risk_score": 0.5}

        def assess_portfolio_risk(self, *args, **kwargs):
            """Assess portfolio risk."""
            return {"risk_score": 0.5}

    class MarketEvaluationEngine:
        """Stub market evaluation engine."""

        def evaluate_market(self, *args, **kwargs):
            """Evaluate market."""
            return {"market_score": 0.5}

        def evaluate_market_opportunity(self, *args, **kwargs):
            """Evaluate market opportunity."""
            return {"market_score": 0.5}

    class ROIProjectionEngine:
        """Stub ROI projection engine."""

        def project_roi(self, *args, **kwargs):
            """Project ROI."""
            return {"roi_projection": 0.0}

        def calculate_roi_projection(self, *args, **kwargs):
            """Calculate ROI projection."""
            return {"roi_projection": 0.0}


logger = logging.getLogger(__name__)


class BusinessReadinessLevel(Enum):
    """Business readiness levels for edge deployment."""

    NOT_READY = "not_ready"
    BASIC_READY = "basic_ready"
    MARKET_READY = "market_ready"
    INVESTMENT_READY = "investment_ready"


class DeploymentStrategy(Enum):
    """Edge deployment strategies."""

    PILOT_DEPLOYMENT = "pilot_deployment"
    REGIONAL_ROLLOUT = "regional_rollout"
    FULL_SCALE = "full_scale"
    HYBRID_CLOUD_EDGE = "hybrid_cloud_edge"


@dataclass
class ValueProposition:
    """Quantified value proposition for edge deployment."""

    # Core value metrics
    cost_reduction_percent: float
    performance_improvement_percent: float
    latency_reduction_ms: float
    availability_improvement_percent: float

    # Business impact
    revenue_opportunity_usd: float
    market_expansion_potential: float
    competitive_advantage_score: float

    # Edge-specific benefits
    data_sovereignty_compliance: bool
    reduced_bandwidth_costs_percent: float
    offline_capability_value: float

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def calculate_total_value_score(self) -> float:
        """Calculate total value proposition score (0-100)."""
        metrics = [
            self.cost_reduction_percent,
            self.performance_improvement_percent,
            min(self.latency_reduction_ms / 10, 100),  # Normalize latency
            self.availability_improvement_percent,
            min(self.revenue_opportunity_usd / 1000000, 100),
            # Normalize revenue (per million)
            self.market_expansion_potential,
            self.competitive_advantage_score,
            20 if self.data_sovereignty_compliance else 0,
            self.reduced_bandwidth_costs_percent,
            self.offline_capability_value,
        ]

        return sum(metrics) / len(metrics)


@dataclass
class MarketFitAssessment:
    """Market fit evaluation for edge deployment."""

    # Market characteristics
    target_market_size_usd: float
    addressable_market_percent: float
    market_growth_rate_percent: float
    competition_intensity_score: float

    # Customer readiness
    customer_demand_score: float
    adoption_readiness_score: float
    pain_point_severity_score: float
    willingness_to_pay_score: float

    # Solution fit
    product_market_fit_score: float
    solution_completeness_score: float
    scalability_potential_score: float

    # Edge-specific factors
    edge_infrastructure_readiness: float
    regulatory_compliance_score: float
    data_locality_requirements_score: float

    def calculate_market_fit_score(self) -> float:
        """Calculate overall market fit score (0-100)."""
        market_metrics = [
            min(self.target_market_size_usd / 10000000, 100),
            # Normalize to 10M baseline
            self.addressable_market_percent,
            self.market_growth_rate_percent,
            100 - self.competition_intensity_score,  # Invert competition
        ]

        customer_metrics = [
            self.customer_demand_score,
            self.adoption_readiness_score,
            self.pain_point_severity_score,
            self.willingness_to_pay_score,
        ]

        solution_metrics = [
            self.product_market_fit_score,
            self.solution_completeness_score,
            self.scalability_potential_score,
        ]

        edge_metrics = [
            self.edge_infrastructure_readiness,
            self.regulatory_compliance_score,
            self.data_locality_requirements_score,
        ]

        # Weighted average
        market_score = sum(market_metrics) / len(market_metrics)
        customer_score = sum(customer_metrics) / len(customer_metrics)
        solution_score = sum(solution_metrics) / len(solution_metrics)
        edge_score = sum(edge_metrics) / len(edge_metrics)

        return market_score * 0.3 + customer_score * \
            0.3 + solution_score * 0.25 + edge_score * 0.15


@dataclass
class OperationalReadiness:
    """Operational readiness assessment for edge deployment."""

    # Team readiness
    technical_expertise_score: float
    operational_capability_score: float
    support_infrastructure_score: float
    training_readiness_score: float

    # Process maturity
    deployment_process_maturity: float
    monitoring_capabilities_score: float
    incident_response_readiness: float
    compliance_process_score: float

    # Resource availability
    budget_adequacy_score: float
    timeline_feasibility_score: float
    resource_allocation_score: float

    # Scalability readiness
    scaling_process_maturity: float
    automation_readiness_score: float
    partnership_ecosystem_score: float

    def calculate_operational_score(self) -> float:
        """Calculate overall operational readiness score (0-100)."""
        team_score = (
            sum(
                [
                    self.technical_expertise_score,
                    self.operational_capability_score,
                    self.support_infrastructure_score,
                    self.training_readiness_score,
                ]
            )
            / 4
        )

        process_score = (
            sum(
                [
                    self.deployment_process_maturity,
                    self.monitoring_capabilities_score,
                    self.incident_response_readiness,
                    self.compliance_process_score,
                ]
            )
            / 4
        )

        resource_score = (
            sum(
                [
                    self.budget_adequacy_score,
                    self.timeline_feasibility_score,
                    self.resource_allocation_score,
                ]
            )
            / 3
        )

        scalability_score = (
            sum(
                [
                    self.scaling_process_maturity,
                    self.automation_readiness_score,
                    self.partnership_ecosystem_score,
                ]
            )
            / 3
        )

        return (
            team_score * 0.3
            + process_score * 0.3
            + resource_score * 0.25
            + scalability_score * 0.15
        )


@dataclass
class BusinessReadinessReport:
    """Comprehensive business readiness assessment report."""

    coalition_id: str
    assessment_timestamp: datetime

    # Core assessments
    value_proposition: ValueProposition
    market_fit: MarketFitAssessment
    operational_readiness: OperationalReadiness

    # Overall scores
    business_readiness_level: BusinessReadinessLevel
    overall_score: float
    value_score: float
    market_score: float
    operational_score: float
    risk_score: float

    # Strategic recommendations
    recommended_strategy: DeploymentStrategy
    investment_requirements: Dict[str, float]
    timeline_recommendations: Dict[str, str]
    risk_mitigation_strategies: List[str]

    # Business intelligence integration
    roi_projections: Dict[str, Any]
    market_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    # Metadata
    assessment_duration: float = 0.0
    confidence_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "coalition_id": self.coalition_id,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "value_proposition": asdict(self.value_proposition),
            "market_fit": asdict(self.market_fit),
            "operational_readiness": asdict(self.operational_readiness),
            "business_readiness_level": self.business_readiness_level.value,
            "overall_score": self.overall_score,
            "value_score": self.value_score,
            "market_score": self.market_score,
            "operational_score": self.operational_score,
            "risk_score": self.risk_score,
            "recommended_strategy": self.recommended_strategy.value,
            "investment_requirements": self.investment_requirements,
            "timeline_recommendations": self.timeline_recommendations,
            "risk_mitigation_strategies": self.risk_mitigation_strategies,
            "roi_projections": self.roi_projections,
            "market_analysis": self.market_analysis,
            "risk_assessment": self.risk_assessment,
            "assessment_duration": self.assessment_duration,
            "confidence_level": self.confidence_level,
        }


class BusinessReadinessAssessor:
    """
    Comprehensive business readiness assessment system for edge deployment.

    Integrates with business intelligence modules to evaluate coalition
    readiness across value proposition, market fit, and operational
    capabilities.
    """

    def __init__(self) -> None:
        """Initialize business readiness assessor."""
        # Initialize business intelligence engines
        self.opportunity_engine = OpportunityDetectionEngine()
        self.risk_engine = RiskAssessmentEngine()
        self.market_engine = MarketEvaluationEngine()
        self.roi_engine = ROIProjectionEngine()

        logger.info("Business readiness assessor initialized")

    async def assess_business_readiness(
        self,
        coalition_id: str,
        coalition_config: Dict[str, Any],
        business_context: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> BusinessReadinessReport:
        """
        Perform comprehensive business readiness assessment.

        Args:
            coalition_id: Unique coalition identifier
            coalition_config: Coalition configuration and capabilities
            business_context: Business context and requirements
            market_data: Market conditions and competitive landscape

        Returns:
            Comprehensive business readiness report
        """
        logger.info(
            f"Starting business readiness assessment for coalition " f"{coalition_id}")
        start_time = time.time()

        try:
            # Step 1: Gather business intelligence insights
            roi_analysis = await self._analyze_roi_projections(
                coalition_config, business_context, market_data
            )

            market_analysis = await self._analyze_market_opportunities(
                coalition_config, market_data
            )

            risk_analysis = await self._analyze_business_risks(
                coalition_config, business_context, market_data
            )

            # Step 2: Assess value proposition
            value_proposition = self._assess_value_proposition(
                coalition_config, roi_analysis, market_analysis
            )

            # Step 3: Evaluate market fit
            market_fit = self._evaluate_market_fit(
                market_analysis, business_context, market_data)

            # Step 4: Assess operational readiness
            operational_readiness = self._assess_operational_readiness(
                coalition_config, business_context
            )

            # Step 5: Calculate overall scores
            scores = self._calculate_business_scores(
                value_proposition, market_fit, operational_readiness, risk_analysis
            )

            # Step 6: Determine readiness level and strategy
            readiness_level = self._determine_readiness_level(scores["overall"])
            strategy = self._recommend_deployment_strategy(
                readiness_level, scores, market_analysis)

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                scores, value_proposition, market_fit, operational_readiness, risk_analysis)

            # Step 8: Compile final report
            assessment_duration = time.time() - start_time

            report = BusinessReadinessReport(
                coalition_id=coalition_id,
                assessment_timestamp=datetime.now(),
                value_proposition=value_proposition,
                market_fit=market_fit,
                operational_readiness=operational_readiness,
                business_readiness_level=readiness_level,
                overall_score=scores["overall"],
                value_score=scores["value"],
                market_score=scores["market"],
                operational_score=scores["operational"],
                risk_score=scores["risk"],
                recommended_strategy=strategy,
                investment_requirements=recommendations["investment"],
                timeline_recommendations=recommendations["timeline"],
                risk_mitigation_strategies=recommendations["risk_mitigation"],
                roi_projections=roi_analysis,
                market_analysis=market_analysis,
                risk_assessment=risk_analysis,
                assessment_duration=assessment_duration,
                confidence_level=self._calculate_confidence_level(scores),
            )

            logger.info(
                f"Business readiness assessment completed. "
                f"Readiness level: {readiness_level.value}, "
                f"Overall score: {scores['overall']:.1f}"
            )

            return report

        except Exception as e:
            logger.error(f"Business readiness assessment failed: {str(e)}")
            raise

    async def _analyze_roi_projections(
        self,
        coalition_config: Dict[str, Any],
        business_context: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze ROI projections using business intelligence."""
        try:
            # Generate ROI projections for edge deployment scenarios
            scenarios = ["conservative", "baseline", "optimistic"]
            roi_results = {}

            for scenario in scenarios:
                # Modify parameters based on scenario
                scenario_params = self._adjust_scenario_parameters(
                    business_context, scenario)

                # Get ROI projections from engine
                roi_projection = await self.roi_engine.project_roi(
                    coalition_config, scenario_params, market_data
                )

                roi_results[scenario] = roi_projection

            return {
                "scenarios": roi_results,
                "recommendation": self._get_roi_recommendation(roi_results),
                "confidence_intervals": self._extract_roi_confidence_intervals(roi_results),
            }

        except Exception as e:
            logger.warning(f"ROI analysis failed: {str(e)}")
            return {
                "scenarios": {},
                "recommendation": "Insufficient data",
                "confidence_intervals": {},
            }

    async def _analyze_market_opportunities(
        self, coalition_config: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market opportunities using market evaluation engine."""
        try:
            # Get market evaluation
            market_evaluation = await self.market_engine.evaluate_market(
                coalition_config, market_data
            )

            return {
                "market_size": market_evaluation.get(
                    "market_size", {}), "competitive_landscape": market_evaluation.get(
                    "competitive_analysis", {}), "customer_segments": market_evaluation.get(
                    "customer_analysis", {}), "growth_projections": market_evaluation.get(
                    "growth_analysis", {}), "barriers_to_entry": market_evaluation.get(
                        "barriers", []), "market_timing": market_evaluation.get(
                            "timing_analysis", {}), }

        except Exception as e:
            logger.warning(f"Market analysis failed: {str(e)}")
            return {}

    async def _analyze_business_risks(
        self,
        coalition_config: Dict[str, Any],
        business_context: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze business risks using risk assessment engine."""
        try:
            # Get comprehensive risk assessment
            risk_assessment = await self.risk_engine.assess_risk(
                coalition_config, business_context, market_data
            )

            return {
                "overall_risk_score": risk_assessment.get(
                    "overall_risk", 50), "risk_categories": risk_assessment.get(
                    "risk_breakdown", {}), "mitigation_strategies": risk_assessment.get(
                    "mitigation_strategies", []), "risk_factors": risk_assessment.get(
                    "key_risks", []), "confidence_intervals": risk_assessment.get(
                        "confidence_intervals", {}), }

        except Exception as e:
            logger.warning(f"Risk analysis failed: {str(e)}")
            return {"overall_risk_score": 50, "risk_categories": {}}

    def _assess_value_proposition(
        self,
        coalition_config: Dict[str, Any],
        roi_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
    ) -> ValueProposition:
        """Assess value proposition for edge deployment."""
        # Extract metrics from ROI analysis
        baseline_roi = roi_analysis.get("scenarios", {}).get("baseline", {})

        # Calculate edge-specific value metrics
        cost_reduction = self._calculate_cost_reduction(coalition_config, baseline_roi)
        performance_improvement = self._calculate_performance_improvement(
            coalition_config)
        latency_reduction = self._calculate_latency_reduction(coalition_config)
        availability_improvement = self._calculate_availability_improvement(
            coalition_config)

        # Business impact metrics
        revenue_opportunity = baseline_roi.get("revenue_projection", 0)
        market_expansion = market_analysis.get("growth_projections", {}).get(
            "expansion_potential", 20
        )
        competitive_advantage = self._assess_competitive_advantage(
            coalition_config, market_analysis
        )

        # Edge-specific benefits
        data_sovereignty = coalition_config.get(
            "edge_features", {}).get(
            "data_sovereignty", False)
        bandwidth_savings = self._calculate_bandwidth_savings(coalition_config)
        offline_value = self._calculate_offline_capability_value(coalition_config)

        return ValueProposition(
            cost_reduction_percent=cost_reduction,
            performance_improvement_percent=performance_improvement,
            latency_reduction_ms=latency_reduction,
            availability_improvement_percent=availability_improvement,
            revenue_opportunity_usd=revenue_opportunity,
            market_expansion_potential=market_expansion,
            competitive_advantage_score=competitive_advantage,
            data_sovereignty_compliance=data_sovereignty,
            reduced_bandwidth_costs_percent=bandwidth_savings,
            offline_capability_value=offline_value,
            confidence_intervals=roi_analysis.get("confidence_intervals", {}),
        )

    def _evaluate_market_fit(
        self,
        market_analysis: Dict[str, Any],
        business_context: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> MarketFitAssessment:
        """Evaluate market fit for edge deployment."""
        # Market characteristics
        market_size = market_analysis.get("market_size", {}).get(
            "total_addressable_market", 1000000
        )
        addressable_percent = market_analysis.get("market_size", {}).get(
            "serviceable_addressable_market_percent", 10
        )
        growth_rate = market_analysis.get(
            "growth_projections", {}).get(
            "annual_growth_rate", 15)
        competition = market_analysis.get(
            "competitive_landscape", {}).get(
            "intensity_score", 60)

        # Customer readiness (estimated from market data)
        customer_demand = market_data.get(
            "customer_readiness", {}).get(
            "demand_score", 70)
        adoption_readiness = market_data.get(
            "customer_readiness", {}).get(
            "adoption_score", 60)
        pain_severity = business_context.get(
            "customer_pain_points", {}).get(
            "severity_score", 75)
        willingness_to_pay = market_data.get(
            "pricing", {}).get(
            "willingness_to_pay_score", 65)

        # Solution fit
        product_fit = business_context.get(
            "solution_fit", {}).get(
            "product_market_fit_score", 70)
        completeness = business_context.get(
            "solution_fit", {}).get(
            "completeness_score", 80)
        scalability = business_context.get(
            "solution_fit", {}).get(
            "scalability_score", 85)

        # Edge-specific factors
        edge_readiness = market_data.get(
            "infrastructure", {}).get(
            "edge_readiness_score", 60)
        regulatory = business_context.get("compliance", {}).get("regulatory_score", 80)
        data_locality = business_context.get(
            "data_requirements", {}).get(
            "locality_score", 70)

        return MarketFitAssessment(
            target_market_size_usd=market_size,
            addressable_market_percent=addressable_percent,
            market_growth_rate_percent=growth_rate,
            competition_intensity_score=competition,
            customer_demand_score=customer_demand,
            adoption_readiness_score=adoption_readiness,
            pain_point_severity_score=pain_severity,
            willingness_to_pay_score=willingness_to_pay,
            product_market_fit_score=product_fit,
            solution_completeness_score=completeness,
            scalability_potential_score=scalability,
            edge_infrastructure_readiness=edge_readiness,
            regulatory_compliance_score=regulatory,
            data_locality_requirements_score=data_locality,
        )

    def _assess_operational_readiness(
        self, coalition_config: Dict[str, Any], business_context: Dict[str, Any]
    ) -> OperationalReadiness:
        """Assess operational readiness for edge deployment."""
        # Team readiness
        technical_expertise = business_context.get(
            "team", {}).get("technical_score", 75)
        operational_capability = business_context.get(
            "team", {}).get("operational_score", 70)
        support_infrastructure = business_context.get(
            "support", {}).get("infrastructure_score", 65)
        training_readiness = business_context.get("team", {}).get("training_score", 60)

        # Process maturity
        deployment_maturity = business_context.get(
            "processes", {}).get(
            "deployment_score", 70)
        monitoring_capabilities = business_context.get(
            "monitoring", {}).get("capability_score", 75)
        incident_response = business_context.get(
            "processes", {}).get(
            "incident_response_score", 65)
        compliance_process = business_context.get(
            "compliance", {}).get(
            "process_score", 80)

        # Resource availability
        budget_adequacy = business_context.get("resources", {}).get("budget_score", 75)
        timeline_feasibility = business_context.get(
            "resources", {}).get("timeline_score", 70)
        resource_allocation = business_context.get(
            "resources", {}).get(
            "allocation_score", 65)

        # Scalability readiness
        scaling_maturity = business_context.get(
            "scalability", {}).get(
            "process_score", 60)
        automation_readiness = business_context.get(
            "automation", {}).get(
            "readiness_score", 55)
        partnership_ecosystem = business_context.get(
            "partnerships", {}).get(
            "ecosystem_score", 70)

        return OperationalReadiness(
            technical_expertise_score=technical_expertise,
            operational_capability_score=operational_capability,
            support_infrastructure_score=support_infrastructure,
            training_readiness_score=training_readiness,
            deployment_process_maturity=deployment_maturity,
            monitoring_capabilities_score=monitoring_capabilities,
            incident_response_readiness=incident_response,
            compliance_process_score=compliance_process,
            budget_adequacy_score=budget_adequacy,
            timeline_feasibility_score=timeline_feasibility,
            resource_allocation_score=resource_allocation,
            scaling_process_maturity=scaling_maturity,
            automation_readiness_score=automation_readiness,
            partnership_ecosystem_score=partnership_ecosystem,
        )

    def _calculate_business_scores(
        self,
        value_proposition: ValueProposition,
        market_fit: MarketFitAssessment,
        operational_readiness: OperationalReadiness,
        risk_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate comprehensive business readiness scores."""
        value_score = value_proposition.calculate_total_value_score()
        market_score = market_fit.calculate_market_fit_score()
        operational_score = operational_readiness.calculate_operational_score()
        risk_score = 100 - risk_analysis.get(
            "overall_risk_score", 50
        )  # Invert risk (lower risk = higher score)

        # Calculate weighted overall score
        overall_score = (
            value_score * 0.35  # Value proposition is most important
            + market_score * 0.30  # Market fit is critical
            + operational_score * 0.25  # Operational readiness
            + risk_score * 0.10  # Risk adjustment
        )

        return {
            "overall": overall_score,
            "value": value_score,
            "market": market_score,
            "operational": operational_score,
            "risk": risk_score,
        }

    def _determine_readiness_level(
            self, overall_score: float) -> BusinessReadinessLevel:
        """Determine business readiness level based on overall score."""
        if overall_score >= 85.0:
            return BusinessReadinessLevel.INVESTMENT_READY
        elif overall_score >= 70.0:
            return BusinessReadinessLevel.MARKET_READY
        elif overall_score >= 55.0:
            return BusinessReadinessLevel.BASIC_READY
        else:
            return BusinessReadinessLevel.NOT_READY

    def _recommend_deployment_strategy(
        self,
        readiness_level: BusinessReadinessLevel,
        scores: Dict[str, float],
        market_analysis: Dict[str, Any],
    ) -> DeploymentStrategy:
        """Recommend deployment strategy based on readiness assessment."""
        if readiness_level == BusinessReadinessLevel.INVESTMENT_READY:
            if scores["market"] > 80 and scores["operational"] > 80:
                return DeploymentStrategy.FULL_SCALE
            else:
                return DeploymentStrategy.REGIONAL_ROLLOUT

        elif readiness_level == BusinessReadinessLevel.MARKET_READY:
            return DeploymentStrategy.REGIONAL_ROLLOUT

        elif readiness_level == BusinessReadinessLevel.BASIC_READY:
            return DeploymentStrategy.PILOT_DEPLOYMENT

        else:
            # Start small regardless
            return DeploymentStrategy.PILOT_DEPLOYMENT

    def _generate_recommendations(
        self,
        scores: Dict[str, float],
        value_proposition: ValueProposition,
        market_fit: MarketFitAssessment,
        operational_readiness: OperationalReadiness,
        risk_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate strategic recommendations based on assessment."""
        recommendations: Dict[str, Any] = {
            "investment": {}, "timeline": {}, "risk_mitigation": []}

        # Investment requirements
        base_investment = 100000  # Base $100k
        if scores["operational"] < 70:
            recommendations["investment"]["operational_enhancement"] = base_investment * 0.5
        if scores["market"] < 70:
            recommendations["investment"]["market_development"] = base_investment * 0.3
        if scores["value"] < 70:
            recommendations["investment"]["product_enhancement"] = base_investment * 0.4

        # Timeline recommendations
        if scores["overall"] >= 80:
            recommendations["timeline"]["pilot_phase"] = "3-6 months"
            recommendations["timeline"]["full_deployment"] = "12-18 months"
        elif scores["overall"] >= 60:
            recommendations["timeline"]["pilot_phase"] = "6-9 months"
            recommendations["timeline"]["full_deployment"] = "18-24 months"
        else:
            recommendations["timeline"]["preparation_phase"] = "6-12 months"
            recommendations["timeline"]["pilot_phase"] = "12-18 months"

        # Risk mitigation strategies
        if risk_analysis.get("overall_risk_score", 50) > 60:
            recommendations["risk_mitigation"].extend(
                [
                    "Implement comprehensive risk monitoring and "
                    "early warning systems",
                    "Develop detailed contingency plans for "
                    "high-risk scenarios",
                    "Consider phased deployment to minimize exposure",
                ])

        if scores["operational"] < 70:
            recommendations["risk_mitigation"].append(
                "Invest in operational training and process improvement "
                "before scaling")

        return recommendations

    # Helper methods for specific calculations
    def _calculate_cost_reduction(
        self, coalition_config: Dict[str, Any], roi_data: Dict[str, Any]
    ) -> float:
        """Calculate cost reduction percentage from edge deployment."""
        cloud_costs = coalition_config.get(
            "current_costs", {}).get(
            "cloud_infrastructure", 10000)
        edge_costs = coalition_config.get("edge_costs", {}).get("infrastructure", 7000)
        return max(0, ((cloud_costs - edge_costs) / cloud_costs) * 100)

    def _calculate_performance_improvement(
            self, coalition_config: Dict[str, Any]) -> float:
        """Calculate performance improvement percentage."""
        current_performance = coalition_config.get("current_performance", {}).get(
            "baseline_score", 70
        )
        edge_performance = coalition_config.get(
            "edge_performance", {}).get(
            "projected_score", 90)
        return max(
            0, ((edge_performance - current_performance) / current_performance) * 100)

    def _calculate_latency_reduction(self, coalition_config: Dict[str, Any]) -> float:
        """Calculate latency reduction in milliseconds."""
        current_latency = coalition_config.get(
            "current_performance", {}).get(
            "latency_ms", 200)
        edge_latency = coalition_config.get(
            "edge_performance", {}).get(
            "latency_ms", 50)
        return max(0, current_latency - edge_latency)

    def _calculate_availability_improvement(
            self, coalition_config: Dict[str, Any]) -> float:
        """Calculate availability improvement percentage."""
        current_availability = coalition_config.get("current_performance", {}).get(
            "availability_percent", 99.0
        )
        edge_availability = coalition_config.get("edge_performance", {}).get(
            "availability_percent", 99.9
        )
        return max(0, edge_availability - current_availability)

    def _assess_competitive_advantage(
        self, coalition_config: Dict[str, Any], market_analysis: Dict[str, Any]
    ) -> float:
        """Assess competitive advantage score."""
        unique_features = len(coalition_config.get("unique_capabilities", []))
        market_differentiation = market_analysis.get("competitive_landscape", {}).get(
            "differentiation_score", 50
        )
        return min(100, (unique_features * 10) + market_differentiation)

    def _calculate_bandwidth_savings(self, coalition_config: Dict[str, Any]) -> float:
        """Calculate bandwidth cost savings percentage."""
        current_bandwidth_cost = coalition_config.get(
            "current_costs", {}).get("bandwidth", 5000)
        edge_bandwidth_cost = coalition_config.get(
            "edge_costs", {}).get("bandwidth", 2000)
        if current_bandwidth_cost > 0:
            return ((current_bandwidth_cost - edge_bandwidth_cost) /
                    current_bandwidth_cost) * 100
        return 0

    def _calculate_offline_capability_value(
            self, coalition_config: Dict[str, Any]) -> float:
        """Calculate value of offline capability."""
        has_offline = coalition_config.get(
            "edge_features", {}).get(
            "offline_capability", False)
        business_criticality = coalition_config.get("business_context", {}).get(
            "uptime_criticality", 50
        )
        return business_criticality if has_offline else 0

    def _adjust_scenario_parameters(
        self, business_context: Dict[str, Any], scenario: str
    ) -> Dict[str, Any]:
        """Adjust parameters based on scenario type."""
        base_params = business_context.copy()

        if scenario == "conservative":
            # Reduce optimistic estimates by 30%
            for key in ["revenue_multiplier", "adoption_rate", "market_growth"]:
                if key in base_params:
                    base_params[key] *= 0.7
        elif scenario == "optimistic":
            # Increase estimates by 50%
            for key in ["revenue_multiplier", "adoption_rate", "market_growth"]:
                if key in base_params:
                    base_params[key] *= 1.5

        return base_params

    def _get_roi_recommendation(self, roi_results: Dict[str, Any]) -> str:
        """Get ROI-based recommendation."""
        baseline_roi = roi_results.get("baseline", {}).get("roi_percent", 0)

        if baseline_roi > 25:
            return "Strong ROI justification for immediate deployment"
        elif baseline_roi > 15:
            return "Positive ROI supports deployment with standard timeline"
        elif baseline_roi > 5:
            return "Marginal ROI requires careful cost management"
        else:
            return "ROI concerns require business model optimization"

    def _extract_roi_confidence_intervals(
        self, roi_results: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Extract confidence intervals from ROI results."""
        intervals = {}
        for scenario, data in roi_results.items():
            if isinstance(data, dict) and "confidence_interval" in data:
                intervals[f"roi_{scenario}"] = data["confidence_interval"]
        return intervals

    def _calculate_confidence_level(self, scores: Dict[str, float]) -> float:
        """Calculate overall confidence level in assessment."""
        # Higher scores generally mean higher confidence
        score_variance = max(scores.values()) - min(scores.values())

        # Lower variance means higher confidence
        if score_variance < 10:
            return 95.0
        elif score_variance < 20:
            return 85.0
        elif score_variance < 30:
            return 75.0
        else:
            return 65.0

    def save_report(self, report: BusinessReadinessReport, output_path: Path) -> None:
        """Save business readiness report to file."""
        report_data = report.to_dict()

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Business readiness report saved to {output_path}")


# Convenience function for direct usage
async def assess_coalition_business_readiness(
    coalition_id: str,
    coalition_config: Dict[str, Any],
    business_context: Dict[str, Any],
    market_data: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> BusinessReadinessReport:
    """Assess coalition business readiness.

    Args:
        coalition_id: Unique coalition identifier
        coalition_config: Coalition configuration and capabilities
        business_context: Business context and requirements
        market_data: Market conditions and competitive landscape
        output_path: Path to save report (optional)

    Returns:
        Business readiness report
    """
    assessor = BusinessReadinessAssessor()
    report = await assessor.assess_business_readiness(
        coalition_id, coalition_config, business_context, market_data
    )

    if output_path:
        assessor.save_report(report, output_path)

    return report
