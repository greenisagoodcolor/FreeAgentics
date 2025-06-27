."""
Comprehensive Readiness Report Integration for Edge Deployment

Unified system that integrates technical, business, and safety assessments into a
comprehensive deployment readiness report with prioritized recommendations and
risk mitigation strategies.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from coalitions.readiness.business_readiness_assessor import (
    BusinessReadinessAssessor,
    BusinessReadinessLevel,
    BusinessReadinessReport,
    DeploymentStrategy,
)
from coalitions.readiness.safety_compliance_verifier import (
    ComplianceFramework,
    SafetyComplianceLevel,
    SafetyComplianceReport,
    SafetyComplianceVerifier,
)

# Import assessment modules
from coalitions.readiness.technical_readiness_validator import EdgePlatform
from coalitions.readiness.technical_readiness_validator import (
    ReadinessLevel as TechnicalReadinessLevel,
)
from coalitions.readiness.technical_readiness_validator import (
    TechnicalReadinessReport,
    TechnicalReadinessValidator,
)

logger = logging.getLogger(__name__)


class OverallReadinessLevel(Enum):
    """Overall deployment readiness levels."""

    NOT_READY = "not_ready"
    CONDITIONALLY_READY = "conditionally_ready"
    DEPLOYMENT_READY = "deployment_ready"
    ENTERPRISE_READY = "enterprise_ready"


class RiskLevel(Enum):
    """Risk levels for deployment decisions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionPriority(Enum):
    """Priority levels for recommended actions."""

    IMMEDIATE = "immediate"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ReadinessScore:
    """Comprehensive readiness scoring breakdown."""

    overall_score: float
    technical_score: float
    business_score: float
    safety_score: float

    # Weighted contributions
    technical_weight: float = 0.35
    business_weight: float = 0.35
    safety_weight: float = 0.30

    # Confidence metrics
    confidence_level: float = 0.0
    assessment_reliability: float = 0.0

    def calculate_weighted_score(self) -> float:
        """Calculate the weighted overall score."""
        return (
            self.technical_score * self.technical_weight
            + self.business_score * self.business_weight
            + self.safety_score * self.safety_weight
        )


@dataclass
class RecommendedAction:
    """Individual recommended action for deployment readiness."""

    action_id: str
    category: str  # 'technical', 'business', 'safety', 'integration'
    priority: ActionPriority
    title: str
    description: str
    expected_impact: str
    estimated_effort: str  # 'low', 'medium', 'high'
    estimated_timeline: str
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_if_ignored: RiskLevel = RiskLevel.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "priority": self.priority.value,
            "risk_if_ignored": self.risk_if_ignored.value,
        }


@dataclass
class ComprehensiveReadinessReport:
    """Unified comprehensive readiness assessment report."""

    coalition_id: str
    assessment_timestamp: datetime
    overall_readiness_level: OverallReadinessLevel
    overall_score: float

    # Component scores
    technical_score: float
    business_score: float
    safety_score: float

    # Assessment results
    technical_assessment: TechnicalReadinessReport
    business_assessment: BusinessReadinessReport
    safety_assessment: SafetyComplianceReport

    # Integrated analysis
    deployment_ready: bool
    critical_issues: List[str]
    recommendations: List[str]
    risk_level: str

    # Metadata
    assessment_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "coalition_id": self.coalition_id,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "overall_readiness_level": self.overall_readiness_level.value,
            "overall_score": self.overall_score,
            "technical_score": self.technical_score,
            "business_score": self.business_score,
            "safety_score": self.safety_score,
            "technical_assessment": self.technical_assessment.to_dict(),
            "business_assessment": self.business_assessment.to_dict(),
            "safety_assessment": self.safety_assessment.to_dict(),
            "deployment_ready": self.deployment_ready,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "risk_level": self.risk_level,
            "assessment_duration": self.assessment_duration,
        }


class ComprehensiveReadinessIntegrator:
    """
    Comprehensive readiness report integration system.

    Integrates technical, business, and safety assessments into a unified
    deployment readiness report with prioritized recommendations and
    risk mitigation strategies.
    """

    def __init__(self) -> None:
        # Initialize assessment components
        self.technical_validator = TechnicalReadinessValidator()
        self.business_assessor = BusinessReadinessAssessor()
        self.safety_verifier = SafetyComplianceVerifier()

        logger.info("Comprehensive readiness integrator initialized")

    async def assess_comprehensive_readiness(
        self,
        coalition_id: str,
        coalition_config: Dict[str, Any],
        business_context: Dict[str, Any],
        deployment_context: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        target_platform: Optional[EdgePlatform] = None,
        required_frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> ComprehensiveReadinessReport:
        """
        Perform comprehensive readiness assessment integrating all components.

        Args:
            coalition_id: Unique coalition identifier
            coalition_config: Coalition configuration and capabilities
            business_context: Business context and requirements
            deployment_context: Edge deployment context and requirements
            market_data: Market conditions and competitive landscape
            target_platform: Target edge platform (auto-detect if None)
            required_frameworks: Specific compliance frameworks to check

        Returns:
            Comprehensive readiness report
        """
        logger.info(f"Starting comprehensive readiness assessment for coalition {coalition_id}")
        start_time = time.time()

        try:
            # Default market data if not provided
            if market_data is None:
                market_data = self._generate_default_market_data(business_context)

            # Step 1: Perform individual assessments in parallel
            logger.info("Performing individual readiness assessments")

            # Technical assessment
            technical_report = await self.technical_validator.assess_technical_readiness(
                coalition_id, coalition_config, target_platform
            )

            # Business assessment
            business_report = await self.business_assessor.assess_business_readiness(
                coalition_id, coalition_config, business_context, market_data
            )

            # Safety and compliance assessment
            safety_report = await self.safety_verifier.verify_safety_compliance(
                coalition_id, coalition_config, deployment_context,
                    required_frameworks
            )

            # Step 2: Integrate assessments
            integrated_report = self._integrate_assessments(
                coalition_id, technical_report, business_report, safety_report
            )

            # Step 3: Add timing metadata
            assessment_duration = time.time() - start_time
            integrated_report.assessment_duration = assessment_duration

            logger.info(
                f"Comprehensive readiness assessment completed. "
                f"Overall level: {integrated_report.overall_readiness_level.value}, "
                f"Score: {integrated_report.overall_score:.1f}, "
                f"Ready: {integrated_report.deployment_ready}"
            )

            return integrated_report

        except Exception as e:
            logger.error(f"Comprehensive readiness assessment failed: {str(e)}")
            raise

    def _integrate_assessments(
        self,
        coalition_id: str,
        technical_report: TechnicalReadinessReport,
        business_report: BusinessReadinessReport,
        safety_report: SafetyComplianceReport,
    ) -> ComprehensiveReadinessReport:
        """Integrate all assessment results into unified report."""

        # Extract scores
        technical_score = technical_report.overall_score
        business_score = business_report.overall_score
        safety_score = safety_report.overall_safety_score

        # Calculate weighted overall score
        overall_score = (
            technical_score * 0.35  # Technical infrastructure weight
            + business_score * 0.35  # Business viability weight
            + safety_score * 0.30  # Safety and compliance weight
        )

        # Determine overall readiness level
        overall_readiness = self._determine_overall_readiness(
            overall_score, technical_report, business_report, safety_report
        )

        # Determine deployment readiness
        deployment_ready = overall_readiness in [
            OverallReadinessLevel.DEPLOYMENT_READY,
            OverallReadinessLevel.ENTERPRISE_READY,
        ]

        # Compile critical issues
        critical_issues = []

        if not safety_report.deployment_approval:
            critical_issues.extend(safety_report.critical_issues)

        if not technical_report.deployment_ready:
            critical_issues.extend(technical_report.issues)

        if business_report.business_readiness_level == BusinessReadinessLevel.NOT_READY:
            critical_issues.append("Business model not ready for deployment")

        # Generate integrated recommendations
        recommendations = self._generate_integrated_recommendations(
            technical_report, business_report, safety_report
        )

        # Assess integrated risk level
        risk_level = self._assess_integrated_risk(technical_report, business_report, safety_report)

        # Create comprehensive report
        return ComprehensiveReadinessReport(
            coalition_id=coalition_id,
            assessment_timestamp=datetime.now(),
            overall_readiness_level=overall_readiness,
            overall_score=overall_score,
            technical_score=technical_score,
            business_score=business_score,
            safety_score=safety_score,
            technical_assessment=technical_report,
            business_assessment=business_report,
            safety_assessment=safety_report,
            deployment_ready=deployment_ready,
            critical_issues=critical_issues,
            recommendations=recommendations,
            risk_level=risk_level,
        )

    def _determine_overall_readiness(
        self,
        overall_score: float,
        technical_report: TechnicalReadinessReport,
        business_report: BusinessReadinessReport,
        safety_report: SafetyComplianceReport,
    ) -> OverallReadinessLevel:
        """Determine overall readiness level based on all assessments."""

        # Check for critical blockers first
        has_critical_safety_issues = not safety_report.deployment_approval
        has_critical_technical_issues = not technical_report.deployment_ready
        has_critical_business_issues = (
            business_report.business_readiness_level == BusinessReadinessLevel.NOT_READY
        )

        if (
            has_critical_safety_issues
            or has_critical_technical_issues
            or has_critical_business_issues
        ):
            return OverallReadinessLevel.NOT_READY

        # Assess based on scores and individual readiness levels
        if overall_score >= 85.0 and all(
            [
                technical_report.readiness_level
                in [
                    TechnicalReadinessLevel.PRODUCTION_READY,
                    TechnicalReadinessLevel.ENTERPRISE_READY,
                ],
                business_report.business_readiness_level
                in [BusinessReadinessLevel.MARKET_READY, BusinessReadinessLevel.INVESTMENT_READY],
                safety_report.compliance_level
                in [
                    SafetyComplianceLevel.FULLY_COMPLIANT,
                    SafetyComplianceLevel.ENTERPRISE_COMPLIANT,
                ],
            ]
        ):
            return OverallReadinessLevel.ENTERPRISE_READY

        elif overall_score >= 75.0 and all(
            [
                technical_report.deployment_ready,
                business_report.overall_score >= 70,
                safety_report.deployment_approval,
            ]
        ):
            return OverallReadinessLevel.DEPLOYMENT_READY

        elif overall_score >= 60.0:
            return OverallReadinessLevel.CONDITIONALLY_READY

        else:
            return OverallReadinessLevel.NOT_READY

    def _generate_integrated_recommendations(
        self,
        technical_report: TechnicalReadinessReport,
        business_report: BusinessReadinessReport,
        safety_report: SafetyComplianceReport,
    ) -> List[str]:
        """Generate integrated recommendations across all assessment areas."""

        recommendations = []

        # High-priority safety recommendations
        if not safety_report.deployment_approval:
            recommendations.append(
                "CRITICAL: Resolve all safety and compliance violations before deployment"
            )
            recommendations.extend(
                safety_report.recommendations[:3]
            )  # Top 3 safety recommendations

        # Technical recommendations
        if not technical_report.deployment_ready:
            recommendations.append("Address technical readiness gaps for production deployment")
            recommendations.extend(
                technical_report.recommendations[:2]
            )  # Top 2 technical recommendations

        # Business recommendations
        if business_report.overall_score < 75:
            recommendations.append("Strengthen business model and market validation")
            recommendations.extend(business_report.timeline_recommendations.values())

        # Integration recommendations
        if len(recommendations) == 0:  # If no critical issues
            recommendations.append("Optimize deployment configuration for maximum performance")
            recommendations.append("Implement comprehensive monitoring and
                alerting systems")
            recommendations.append("Develop detailed operational procedures and
                runbooks")

        # Prioritize and limit recommendations
        return recommendations[:10]  # Top 10 recommendations

    def _assess_integrated_risk(
        self,
        technical_report: TechnicalReadinessReport,
        business_report: BusinessReadinessReport,
        safety_report: SafetyComplianceReport,
    ) -> str:
        """Assess integrated risk level across all dimensions."""

        # Safety risks (highest priority)
        if not safety_report.deployment_approval:
            return "critical"

        # Technical risks
        if not technical_report.deployment_ready:
            return "high"

        # Business risks
        if business_report.overall_score < 60:
            return "high"

        # Medium risk conditions
        if any(
            [
                technical_report.overall_score < 80,
                business_report.overall_score < 75,
                safety_report.overall_safety_score < 85,
            ]
        ):
            return "medium"

        # Low risk
        return "low"

    def _generate_default_market_data(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default market data if not provided."""
        return {
            "market_size": {
                "total_addressable_market": business_context.get("market_size", 10000000),
                "serviceable_addressable_market_percent": 10,
            },
            "competitive_landscape": {"intensity_score": 60, "differentiation_score": 70},
            "customer_readiness": {"demand_score": 70, "adoption_score": 65},
            "pricing": {"willingness_to_pay_score": 70},
            "infrastructure": {"edge_readiness_score": 65},
        }

    def save_report(self, report: ComprehensiveReadinessReport, output_path: Path) -> None:
        """Save comprehensive readiness report to file."""
        report_data = report.to_dict()

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Comprehensive readiness report saved to {output_path}")

    def generate_executive_summary(self, report: ComprehensiveReadinessReport) -> Dict[str, Any]:
        """Generate executive summary of readiness assessment."""

        return {
            "coalition_id": report.coalition_id,
            "assessment_date": report.assessment_timestamp.strftime("%Y-%m-%d"),
            "overall_readiness": report.overall_readiness_level.value,
            "readiness_score": f"{report.overall_score:.1f}/100",
            "deployment_ready": report.deployment_ready,
            "critical_issues_count": len(report.critical_issues),
            "recommendations_count": len(report.recommendations),
            "risk_level": report.risk_level,
            "component_scores": {
                "technical": f"{report.technical_score:.1f}/100",
                "business": f"{report.business_score:.1f}/100",
                "safety": f"{report.safety_score:.1f}/100",
            },
            "key_recommendations": report.recommendations[:5],
            "assessment_duration": f"{report.assessment_duration:.1f} seconds",
        }


# Convenience function for direct usage
async def assess_comprehensive_coalition_readiness(
    coalition_id: str,
    coalition_config: Dict[str, Any],
    business_context: Dict[str, Any],
    deployment_context: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None,
    target_platform: Optional[EdgePlatform] = None,
    required_frameworks: Optional[List[ComplianceFramework]] = None,
    output_path: Optional[Path] = None,
) -> ComprehensiveReadinessReport:
    """
    Convenience function to assess comprehensive coalition readiness.

    Args:
        coalition_id: Unique coalition identifier
        coalition_config: Coalition configuration and capabilities
        business_context: Business context and requirements
        deployment_context: Edge deployment context and requirements
        market_data: Market conditions and competitive landscape
        target_platform: Target edge platform (auto-detect if None)
        required_frameworks: Specific compliance frameworks to check
        output_path: Path to save report (optional)

    Returns:
        Comprehensive readiness report
    """
    integrator = ComprehensiveReadinessIntegrator()
    report = await integrator.assess_comprehensive_readiness(
        coalition_id,
        coalition_config,
        business_context,
        deployment_context,
        market_data,
        target_platform,
        required_frameworks,
    )

    if output_path:
        integrator.save_report(report, output_path)

    return report
