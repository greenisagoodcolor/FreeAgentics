"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

"""
Business Opportunity Detection and Management
Implements opportunity detection algorithms, metrics, and validation systems
for multi-agent business discovery.
"""
logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of business opportunities"""

    MARKET_GAP = "market_gap"
    RESOURCE_ARBITRAGE = "resource_arbitrage"
    CAPABILITY_SYNERGY = "capability_synergy"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    NEW_MARKET = "new_market"
    PARTNERSHIP = "partnership"
    INNOVATION = "innovation"


class OpportunityStatus(Enum):
    """Status of a business opportunity"""

    DETECTED = "detected"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class OpportunityMetrics:
    """Quantifiable metrics for evaluating opportunities"""

    potential_value: float  # Estimated monetary value
    success_probability: float  # 0-1 probability of success
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    time_to_realize: timedelta = field(default_factory=lambda: timedelta(days=30))
    risk_score: float = 0.5  # 0 (low risk) to 1 (high risk)
    market_size: float = 0.0  # Total addressable market
    competition_level: float = 0.5  # 0 (no competition) to 1 (saturated)
    innovation_score: float = 0.0  # 0 (standard) to 1 (highly innovative)

    @property
    def expected_value(self) -> float:
        """Calculate risk-adjusted expected value"""
        return self.potential_value * self.success_probability * (1 - self.risk_score)

    @property
    def roi_estimate(self) -> float:
        """Estimate return on investment"""
        total_resource_cost = sum(self.resource_requirements.values())
        if total_resource_cost > 0:
            return (self.expected_value - total_resource_cost) / total_resource_cost
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "potential_value": self.potential_value,
            "success_probability": self.success_probability,
            "resource_requirements": self.resource_requirements,
            "time_to_realize_days": self.time_to_realize.days,
            "risk_score": self.risk_score,
            "market_size": self.market_size,
            "competition_level": self.competition_level,
            "innovation_score": self.innovation_score,
            "expected_value": self.expected_value,
            "roi_estimate": self.roi_estimate,
        }


@dataclass
class BusinessOpportunity:
    """Represents a detected business opportunity"""

    id: str
    type: OpportunityType
    name: str
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    # Key characteristics
    required_capabilities: set[str] = field(default_factory=set)
    target_market: str = ""
    value_proposition: str = ""
    # Metrics and validation
    metrics: OpportunityMetrics = field(default_factory=OpportunityMetrics)
    status: OpportunityStatus = OpportunityStatus.DETECTED
    validation_results: Dict[str, Any] = field(default_factory=dict)
    # Coalition requirements
    min_agents_required: int = 1
    max_agents_allowed: int = 10
    ideal_coalition_size: int = 3
    # Tracking
    interested_agents: set[str] = field(default_factory=set)
    committed_agents: set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if opportunity has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def is_viable(self) -> bool:
        """Check if opportunity is still viable"""
        return (
            self.status in [OpportunityStatus.VALIDATED, OpportunityStatus.IN_PROGRESS]
            and not self.is_expired()
            and self.metrics.success_probability > 0.3
        )

    def add_interested_agent(self, agent_id: str) -> None:
        """Register agent interest"""
        self.interested_agents.add(agent_id)
        logger.debug(f"Agent {agent_id} interested in opportunity {self.id}")

    def commit_agent(self, agent_id: str) -> bool:
        """Commit an agent to the opportunity"""
        if len(self.committed_agents) >= self.max_agents_allowed:
            return False
        self.committed_agents.add(agent_id)
        self.interested_agents.discard(agent_id)
        # Update status if we have minimum agents
        if len(self.committed_agents) >= self.min_agents_required:
            if self.status == OpportunityStatus.VALIDATED:
                self.status = OpportunityStatus.IN_PROGRESS
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "required_capabilities": list(self.required_capabilities),
            "target_market": self.target_market,
            "value_proposition": self.value_proposition,
            "metrics": self.metrics.to_dict(),
            "status": self.status.value,
            "validation_results": self.validation_results,
            "min_agents_required": self.min_agents_required,
            "max_agents_allowed": self.max_agents_allowed,
            "ideal_coalition_size": self.ideal_coalition_size,
            "interested_agents": list(self.interested_agents),
            "committed_agents": list(self.committed_agents),
        }


class OpportunityDetector:
    """
    Detects business opportunities from various data sources and patterns.
    """

    def __init__(self) -> None:
        """Initialize opportunity detector"""
        self.detection_patterns = {
            OpportunityType.MARKET_GAP: self._detect_market_gap,
            OpportunityType.RESOURCE_ARBITRAGE: self._detect_resource_arbitrage,
            OpportunityType.CAPABILITY_SYNERGY: self._detect_capability_synergy,
            OpportunityType.EFFICIENCY_IMPROVEMENT: self._detect_efficiency_improvement,
        }
        # Thresholds for detection
        self.market_gap_threshold = 0.3
        self.arbitrage_margin_threshold = 0.2
        self.synergy_score_threshold = 0.6
        logger.info("Initialized opportunity detector")

    def detect_opportunities(
        self,
        market_data: Dict[str, Any],
        agent_profiles: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> List[BusinessOpportunity]:
        """
        Detect business opportunities from available data.
        Args:
            market_data: Market information including demand, supply, prices
            agent_profiles: Profiles of available agents
            resource_data: Current resource availability and prices
        Returns:
            List of detected opportunities
        """
        opportunities = []
        # Run each detection pattern
        for opp_type, detector_func in self.detection_patterns.items():
            try:
                detected = detector_func(market_data, agent_profiles, resource_data)
                opportunities.extend(detected)
            except Exception as e:
                logger.error(f"Error in {opp_type.value} detection: {str(e)}")
        # Rank opportunities by expected value
        opportunities.sort(key=lambda o: o.metrics.expected_value, reverse=True)
        logger.info(f"Detected {len(opportunities)} business opportunities")
        return opportunities

    def _detect_market_gap(
        self,
        market_data: Dict[str, Any],
        agent_profiles: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> List[BusinessOpportunity]:
        """Detect market gap opportunities"""
        opportunities = []
        # Analyze supply-demand imbalances
        demands = market_data.get("demands", {})
        supplies = market_data.get("supplies", {})
        for product, demand in demands.items():
            supply = supplies.get(product, 0)
            if supply < demand * (1 - self.market_gap_threshold):
                # Significant gap detected
                gap_size = demand - supply
                market_price = market_data.get("prices", {}).get(product, 100)
                opp = BusinessOpportunity(
                    id=f"gap_{product}_{datetime.utcnow().timestamp()}",
                    type=OpportunityType.MARKET_GAP,
                    name=f"Market Gap: {product}",
                    description=f"Unmet demand for {product}: {gap_size} units",
                    target_market=product,
                    value_proposition=f"Fill market gap of {gap_size} units",
                )
                # Calculate metrics
                opp.metrics.potential_value = gap_size * market_price * 0.8  # Conservative estimate
                opp.metrics.success_probability = min(0.9, 1 - supply / demand)
                opp.metrics.market_size = demand * market_price
                opp.metrics.competition_level = supply / demand if demand > 0 else 0
                # Determine required capabilities
                production_requirements = market_data.get("production_requirements", {})
                opp.required_capabilities.update(
                    production_requirements.get(product, ["production", "logistics"])
                )
                # Resource requirements
                unit_cost = market_data.get("production_costs", {}).get(product, 50)
                opp.metrics.resource_requirements = {
                    "capital": gap_size * unit_cost * 0.3,  # Initial investment
                    "materials": gap_size * unit_cost * 0.7,
                }
                opportunities.append(opp)
        return opportunities

    def _detect_resource_arbitrage(
        self,
        market_data: Dict[str, Any],
        agent_profiles: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> List[BusinessOpportunity]:
        """Detect resource arbitrage opportunities"""
        opportunities = []
        # Analyze price differences across locations/markets
        price_by_location = resource_data.get("price_by_location", {})
        transport_costs = resource_data.get("transport_costs", {})
        for resource in price_by_location:
            locations = list(price_by_location[resource].keys())
            for i, loc1 in enumerate(locations):
                for loc2 in locations[i + 1 :]:
                    price1 = price_by_location[resource][loc1]
                    price2 = price_by_location[resource][loc2]
                    price_diff = abs(price1 - price2)
                    transport_cost = transport_costs.get((loc1, loc2), price_diff * 0.1)
                    margin = (price_diff - transport_cost) / max(price1, price2)
                    if margin > self.arbitrage_margin_threshold:
                        # Profitable arbitrage opportunity
                        buy_loc = loc1 if price1 < price2 else loc2
                        sell_loc = loc2 if price1 < price2 else loc1
                        opp = BusinessOpportunity(
                            id=f"arb_{resource}_{buy_loc}_{sell_loc}_{datetime.utcnow().timestamp()}",
                            type=OpportunityType.RESOURCE_ARBITRAGE,
                            name=f"Arbitrage: {resource}",
                            description=f"Buy {resource} at {buy_loc} for {min(price1, price2)}, "
                            f"sell at {sell_loc} for {max(price1, price2)}",
                            value_proposition=f"Exploit price differential of {margin:.1%}",
                        )
                        # Calculate metrics
                        volume = resource_data.get("available_volume", {}).get(resource, 1000)
                        opp.metrics.potential_value = (
                            volume * price_diff * 0.7
                        )  # Account for market impact
                        opp.metrics.success_probability = (
                            0.8 - margin
                        )  # Higher margins might indicate hidden costs
                        opp.metrics.risk_score = 0.3 + transport_cost / price_diff
                        opp.metrics.time_to_realize = timedelta(days=7)  # Quick turnaround
                        # Requirements
                        opp.required_capabilities.update(["transport", "trading", "market_access"])
                        opp.metrics.resource_requirements = {
                            "capital": volume * min(price1, price2),
                            "transport": transport_cost * volume,
                        }
                        opportunities.append(opp)
        return opportunities

    def _detect_capability_synergy(
        self,
        market_data: Dict[str, Any],
        agent_profiles: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> List[BusinessOpportunity]:
        """Detect opportunities from synergistic agent capabilities"""
        opportunities = []
        # Analyze agent capability combinations
        for i, agent1 in enumerate(agent_profiles):
            for agent2 in agent_profiles[i + 1 :]:
                caps1 = set(agent1.get("capabilities", []))
                caps2 = set(agent2.get("capabilities", []))
                # Look for complementary capabilities
                combined_caps = caps1.union(caps2)
                unique_combination = caps1.symmetric_difference(caps2)
                # Check if combined capabilities unlock new opportunities
                market_requirements = market_data.get("opportunity_requirements", {})
                for opp_name, required_caps in market_requirements.items():
                    required_set = set(required_caps)
                    # Can fulfill requirements together but not separately?
                    if (
                        required_set.issubset(combined_caps)
                        and not required_set.issubset(caps1)
                        and not required_set.issubset(caps2)
                    ):
                        synergy_score = len(unique_combination) / len(combined_caps)
                        if synergy_score > self.synergy_score_threshold:
                            opp = BusinessOpportunity(
                                id=f"syn_{opp_name}_{agent1['agent_id']}_{agent2['agent_id']}",
                                type=OpportunityType.CAPABILITY_SYNERGY,
                                name=f"Synergy: {opp_name}",
                                description=f"Combine capabilities of {agent1['agent_id']} and "
                                f"{agent2['agent_id']} for {opp_name}",
                                value_proposition="Unlock opportunity through capability combination",
                            )
                            # Set metrics based on market data
                            market_value = market_data.get("opportunity_values", {}).get(
                                opp_name, 10000
                            )
                            opp.metrics.potential_value = market_value
                            opp.metrics.success_probability = 0.7 * synergy_score
                            opp.metrics.innovation_score = synergy_score
                            opp.required_capabilities = required_set
                            opp.min_agents_required = 2
                            opp.ideal_coalition_size = len(required_caps) // 2 + 1
                            opportunities.append(opp)
        return opportunities

    def _detect_efficiency_improvement(
        self,
        market_data: Dict[str, Any],
        agent_profiles: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> List[BusinessOpportunity]:
        """Detect efficiency improvement opportunities"""
        opportunities = []
        # Analyze current inefficiencies
        inefficiencies = market_data.get("inefficiencies", {})
        for process, inefficiency_data in inefficiencies.items():
            current_cost = inefficiency_data.get("current_cost", 0)
            optimal_cost = inefficiency_data.get("optimal_cost", 0)
            volume = inefficiency_data.get("volume", 0)
            if current_cost > optimal_cost * 1.2:  # At least 20% improvement possible
                savings = (current_cost - optimal_cost) * volume
                opp = BusinessOpportunity(
                    id=f"eff_{process}_{datetime.utcnow().timestamp()}",
                    type=OpportunityType.EFFICIENCY_IMPROVEMENT,
                    name=f"Optimize: {process}",
                    description=f"Reduce {process} cost from {current_cost} to {optimal_cost}",
                    value_proposition=f"Save {savings:.2f} through process optimization",
                )
                # Calculate metrics
                opp.metrics.potential_value = savings * 0.8  # Share savings
                opp.metrics.success_probability = 0.6  # Moderate success rate
                opp.metrics.risk_score = 0.3  # Low risk
                opp.metrics.time_to_realize = timedelta(days=60)  # Longer implementation
                # Requirements from inefficiency data
                opp.required_capabilities.update(
                    inefficiency_data.get("required_capabilities", ["optimization", "analysis"])
                )
                implementation_cost = inefficiency_data.get("implementation_cost", savings * 0.3)
                opp.metrics.resource_requirements = {
                    "capital": implementation_cost,
                    "expertise": implementation_cost * 0.5,  # Consulting/expertise costs
                }
                opportunities.append(opp)
        return opportunities


class OpportunityValidator:
    """
    Validates detected business opportunities before commitment.
    """

    def __init__(self) -> None:
        """Initialize opportunity validator"""
        self.validation_checks = [
            self._validate_market_demand,
            self._validate_resource_availability,
            self._validate_capability_match,
            self._validate_financial_viability,
            self._validate_risk_assessment,
        ]
        logger.info("Initialized opportunity validator")

    def validate_opportunity(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate a business opportunity.
        Returns:
            Tuple of (is_valid, validation_results)
        """
        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_score": 0.0,
            "recommendations": [],
        }
        passed_checks = 0
        total_weight = 0
        # Run each validation check
        for check_func in self.validation_checks:
            check_name = check_func.__name__.replace("_validate_", "")
            try:
                passed, score, details = check_func(
                    opportunity, market_data, available_agents, resource_data
                )
                validation_results["checks"][check_name] = {
                    "passed": passed,
                    "score": score,
                    "details": details,
                }
                if passed:
                    passed_checks += 1
                    validation_results["overall_score"] += score
                    total_weight += 1
                else:
                    validation_results["recommendations"].append(
                        f"Failed {check_name}: {details.get('reason', 'Unknown')}"
                    )
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {str(e)}")
                validation_results["checks"][check_name] = {
                    "passed": False,
                    "score": 0.0,
                    "details": {"error": str(e)},
                }
        # Calculate overall validation
        if total_weight > 0:
            validation_results["overall_score"] /= total_weight
        is_valid = passed_checks >= 3 and validation_results["overall_score"] >= 0.5
        # Update opportunity
        opportunity.validation_results = validation_results
        opportunity.status = OpportunityStatus.VALIDATED if is_valid else OpportunityStatus.REJECTED
        return is_valid, validation_results

    def _validate_market_demand(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate market demand exists"""
        details = {}
        # Check market size
        if opportunity.metrics.market_size < 1000:
            details["reason"] = "Market size too small"
            return False, 0.0, details
        # Check demand trends
        demand_trend = market_data.get("demand_trends", {}).get(opportunity.target_market, 0)
        details["demand_trend"] = demand_trend
        if demand_trend < -0.2:  # Declining market
            details["reason"] = "Declining market demand"
            return False, 0.2, details
        # Score based on market attractiveness
        score = min(1.0, (opportunity.metrics.market_size / 100000) * (1 + demand_trend))
        return True, score, details

    def _validate_resource_availability(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate required resources are available"""
        details = {"missing_resources": []}
        required = opportunity.metrics.resource_requirements
        available = resource_data.get("total_available", {})
        resource_score = 1.0
        for resource, amount in required.items():
            available_amount = available.get(resource, 0)
            if available_amount < amount:
                details["missing_resources"].append(
                    {
                        "resource": resource,
                        "required": amount,
                        "available": available_amount,
                    }
                )
                resource_score *= available_amount / amount
        if details["missing_resources"]:
            details["reason"] = "Insufficient resources"
            return False, resource_score, details
        return True, resource_score, details

    def _validate_capability_match(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate agent capabilities match requirements"""
        details = {"missing_capabilities": []}
        required_caps = opportunity.required_capabilities
        # Aggregate available capabilities
        available_caps = set()
        for agent in available_agents:
            available_caps.update(agent.get("capabilities", []))
        missing = required_caps - available_caps
        if missing:
            details["missing_capabilities"] = list(missing)
            details["reason"] = f"Missing capabilities: {', '.join(missing)}"
            score = len(required_caps.intersection(available_caps)) / len(required_caps)
            return False, score, details
        # Score based on capability coverage
        capability_redundancy = sum(
            1
            for agent in available_agents
            if any(cap in agent.get("capabilities", []) for cap in required_caps)
        )
        score = min(1.0, capability_redundancy / opportunity.ideal_coalition_size)
        return True, score, details

    def _validate_financial_viability(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate financial viability"""
        details = {}
        # Check ROI
        roi = opportunity.metrics.roi_estimate
        details["roi"] = roi
        if roi < 0.15:  # Minimum 15% ROI
            details["reason"] = f"ROI too low: {roi:.1%}"
            return False, roi, details
        # Check payback period
        if opportunity.metrics.time_to_realize.days > 365:
            details["reason"] = "Payback period too long"
            return False, 0.3, details
        # Score based on financial attractiveness
        score = min(1.0, roi * 2)  # 50% ROI = perfect score
        return True, score, details

    def _validate_risk_assessment(
        self,
        opportunity: BusinessOpportunity,
        market_data: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        resource_data: Dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate risk levels are acceptable"""
        details = {}
        risk_score = opportunity.metrics.risk_score
        details["risk_score"] = risk_score
        if risk_score > 0.8:
            details["reason"] = "Risk level too high"
            return False, 1 - risk_score, details
        # Analyze specific risks
        risks = []
        # Competition risk
        if opportunity.metrics.competition_level > 0.7:
            risks.append("High competition")
        # Innovation risk
        if opportunity.metrics.innovation_score > 0.8:
            risks.append("Unproven innovation")
        # Time risk
        if opportunity.expires_at:
            time_remaining = opportunity.expires_at - datetime.utcnow()
            if time_remaining < timedelta(days=7):
                risks.append("Short time window")
        details["identified_risks"] = risks
        # Score based on risk profile
        score = (1 - risk_score) * (1 - len(risks) * 0.2)
        return True, max(0, score), details
