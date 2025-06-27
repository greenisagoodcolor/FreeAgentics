."""
Merchant Agent for FreeAgentics

This module provides the Merchant agent type, specialized in trading,
resource management, economic activities, and market analysis. Merchants
excel at negotiation, value assessment, and profit optimization.
"""

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Import base agent system
from agents.base import (
    Agent,
    AgentCapability,
    BaseAgent,
    Position,
    get_default_factory)
from agents.base.behaviors import BaseBehavior, BehaviorPriority


class TradeStatus(Enum):
    """Status of trading activities"""

    IDLE = "idle"
    SEEKING_TRADES = "seeking_trades"
    NEGOTIATING = "negotiating"
    EXECUTING_TRADE = "executing_trade"
    EVALUATING_MARKET = "evaluating_market"


class ResourceType(Enum):
    """Types of resources that can be traded"""

    FOOD = "food"
    MATERIALS = "materials"
    TOOLS = "tools"
    INFORMATION = "information"
    SERVICES = "services"
    ENERGY = "energy"
    RARE_ITEMS = "rare_items"
    KNOWLEDGE = "knowledge"


class TradeOffer:
    """Represents a trade offer"""

    def __init__(
        self,
        offering_agent_id: str,
        offered_resources: Dict[ResourceType, float],
        wanted_resources: Dict[ResourceType, float],
        offer_value: float,
        expiration: datetime,
    ) -> None:
        self.offering_agent_id = offering_agent_id
        self.offered_resources = offered_resources
        self.wanted_resources = wanted_resources
        self.offer_value = offer_value
        self.expiration = expiration
        self.created_at = datetime.now()
        self.offer_id = (
            f"trade_{offering_agent_id}_{datetime.now().timestamp()}")
        self.accepted = False
        self.negotiated_value = offer_value


class Market:
    """Represents market conditions and trading opportunities"""

    def __init__(self) -> None:
        self.resource_prices: Dict[ResourceType, float] = {}
        self.active_offers: Dict[str, TradeOffer] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.market_trends: Dict[ResourceType, float] = {}  # -1 to 1,
            negative means declining
        self.last_updated = datetime.now()

        # Initialize default prices
        self._initialize_market_prices()

    def _initialize_market_prices(self):
        """Initialize default market prices for resources"""
        base_prices = {
            ResourceType.FOOD: 1.0,
            ResourceType.MATERIALS: 1.5,
            ResourceType.TOOLS: 3.0,
            ResourceType.INFORMATION: 2.0,
            ResourceType.SERVICES: 2.5,
            ResourceType.ENERGY: 1.2,
            ResourceType.RARE_ITEMS: 10.0,
            ResourceType.KNOWLEDGE: 5.0,
        }

        # Add some random variation
        for resource_type, base_price in base_prices.items():
            variation = random.uniform(0.8, 1.2)
            self.resource_prices[resource_type] = base_price * variation
            self.market_trends[resource_type] = random.uniform(-0.3, 0.3)

    def get_resource_value(self, resource_type: ResourceType,
        quantity: float) -> float:
        ."""Calculate the market value of a resource quantity."""
        base_price = self.resource_prices.get(resource_type, 1.0)

        # Apply market trends
        trend = self.market_trends.get(resource_type, 0.0)
        trend_modifier = 1.0 + (trend * 0.2)  # ±20% based on trend

        return base_price * quantity * trend_modifier

    def add_offer(self, offer: TradeOffer) -> None:
        ."""Add a trade offer to the market."""
        self.active_offers[offer.offer_id] = offer
        self.last_updated = datetime.now()

    def remove_offer(self, offer_id: str) -> bool:
        ."""Remove a trade offer from the market."""
        if offer_id in self.active_offers:
            del self.active_offers[offer_id]
            self.last_updated = datetime.now()
            return True
        return False

    def find_matching_offers(self, wanted_resources: Dict[ResourceType,
        float]) -> List[TradeOffer]:
        """Find offers that match wanted resources"""
        matching_offers = []

        for offer in self.active_offers.values():
            # Check if offer has resources we want
            for resource_type, wanted_quantity in wanted_resources.items():
                offered_quantity = (
                    offer.offered_resources.get(resource_type, 0.0))
                if offered_quantity >= wanted_quantity:
                    matching_offers.append(offer)
                    break

        return matching_offers

    def update_market_trends(self) -> None:
        """Update market trends based on trade activity"""
        # Simulate market dynamics
        for resource_type in self.market_trends:
            # Random market fluctuation
            change = random.uniform(-0.1, 0.1)
            self.market_trends[resource_type] = max(
                -1.0, min(1.0, self.market_trends[resource_type] + change)
            )

            # Update prices based on trends
            trend = self.market_trends[resource_type]
            price_change = trend * 0.05  # 5% max change per update
            self.resource_prices[resource_type] *= 1.0 + price_change

        self.last_updated = datetime.now()


class TradingBehavior(BaseBehavior):
    """Trading and negotiation behavior for merchant agents"""

    def __init__(self) -> None:
        super().__init__(
            "trading",
            BehaviorPriority.HIGH,
            {AgentCapability.COMMUNICATION, AgentCapability.RESOURCE_MANAGEMENT},
            timedelta(seconds=5),
        )
        self.negotiation_rounds = 3

    def _can_execute_custom(self, agent: Agent, context: Dict[str,
        Any]) -> bool:
        # Can trade if agent has resources or needs resources
        market = agent.metadata.get("market")
        if not market:
            return False

        # Check if there are active offers
        return len(market.active_offers) > 0 or agent.resources.get_total_value() > 10.0

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str,
        Any]:
        market = agent.metadata.get("market")
        if not market:
            return {"success": False, "reason": "no_market_access"}

        personality_profile = agent.metadata.get("personality_profile")

        # Determine trading strategy based on personality
        trading_result = (
            self._execute_trading_strategy(agent, market, personality_profile))

        # Update agent status
        agent.metadata["trade_status"] = TradeStatus.SEEKING_TRADES

        return trading_result

    def _execute_trading_strategy(
        self, agent: Agent, market: Market, personality_profile
    ) -> Dict[str, Any]:
        """Execute trading strategy based on agent's personality and market
        conditions."""

        # Assess agent's needs and surpluses
        needs, surpluses = self._assess_resource_situation(agent)

        if surpluses and random.random() < 0.7:  # Try to sell surplus
            return self._create_sell_offer(agent, market, surpluses,
                personality_profile)
        elif needs and random.random() < 0.8:  # Try to buy needed resources
            return self._find_buy_opportunities(agent, market, needs,
                personality_profile)
        else:
            # Market evaluation
            return self._evaluate_market_conditions(agent, market)

    def _assess_resource_situation(
        self, agent: Agent
    ) -> tuple[dict[ResourceType, float], dict[ResourceType, float]]:
        """Assess what resources the agent needs and has in surplus"""
        # This is a simplified assessment - in a real implementation,
        # this would analyze the agent's resource inventory

        needs = {}
        surpluses = {}

        # Mock resource assessment based on agent's current situation
        energy_level = agent.resources.energy / agent.resources.max_energy

        if energy_level < 0.5:
            needs[ResourceType.ENERGY] = (0.8 - energy_level) * 50.0

        if energy_level > 0.8:
            surpluses[ResourceType.ENERGY] = (energy_level - 0.6) * 30.0

        # Add some random resource needs/surpluses
        if random.random() < 0.3:
            needs[random.choice(list(ResourceType))] = random.uniform(1.0,
                10.0)

        if random.random() < 0.4:
            surpluses[random.choice(list(ResourceType))] = random.uniform(2.0,
                15.0)

        return needs, surpluses

    def _create_sell_offer(
        self,
        agent: Agent,
        market: Market,
        surpluses: Dict[ResourceType, float],
        personality_profile,
    ) -> Dict[str, Any]:
        """Create an offer to sell surplus resources"""

        # Select resources to offer
        offered_resources = {}
        total_offer_value = 0.0

        for resource_type, surplus_amount in surpluses.items():
            offer_amount = (
                surplus_amount * random.uniform(0.5, 0.9)  # Don't offer everything)
            offered_resources[resource_type] = offer_amount
            total_offer_value += (
                market.get_resource_value(resource_type, offer_amount))

        # Determine what we want in return
        wanted_resources = self._determine_wanted_resources(
            market, total_offer_value * 0.9
        )  # Accept 90% value

        # Adjust offer value based on personality
        if personality_profile:
            conscientiousness = (
                personality_profile.get_trait_value("conscientiousness"))
            extraversion = personality_profile.get_trait_value("extraversion")

            # More conscientious agents ask for better deals
            value_modifier = 0.9 + (conscientiousness * 0.2)
            # More extraverted agents are better negotiators
            negotiation_bonus = extraversion * 0.1

            total_offer_value *= value_modifier + negotiation_bonus

        # Create trade offer
        offer = TradeOffer(
            offering_agent_id=agent.agent_id,
            offered_resources=offered_resources,
            wanted_resources=wanted_resources,
            offer_value=total_offer_value,
            expiration=datetime.now() + timedelta(hours=6),
        )

        market.add_offer(offer)

        return {
            "success": True,
            "action": "sell_offer_created",
            "offer_id": offer.offer_id,
            "offered_resources": offered_resources,
            "wanted_resources": wanted_resources,
            "offer_value": total_offer_value,
        }

    def _find_buy_opportunities(
        self,
        agent: Agent,
        market: Market,
        needs: Dict[ResourceType, float],
        personality_profile,
    ) -> Dict[str, Any]:
        """Find and evaluate buy opportunities"""

        matching_offers = market.find_matching_offers(needs)

        if not matching_offers:
            return {"success": False, "reason": "no_matching_offers"}

        # Evaluate offers based on value and personality
        best_offer = None
        best_score = 0.0

        for offer in matching_offers:
            score = (
                self._evaluate_trade_offer(offer, needs, market, personality_profile))
            if score > best_score:
                best_score = score
                best_offer = offer

        if best_offer and best_score > 0.5:  # Accept if good enough
            # Execute the trade
            return self._execute_trade(agent, best_offer, market)

        return {"success": False, "reason": "no_acceptable_offers"}

    def _determine_wanted_resources(
        self, market: Market, target_value: float
    ) -> Dict[ResourceType, float]:
        """Determine what resources to request for a given value"""
        wanted_resources = {}
        remaining_value = target_value

        # Prefer resources with good market trends
        sorted_resources = sorted(
            market.market_trends.items(),
            key=lambda x: x[1],  # Sort by trend value
            reverse=True,
        )

        for resource_type, trend in sorted_resources[:3]:  # Top 3 trending resources
            if remaining_value <= 0:
                break

            resource_price = market.resource_prices[resource_type]
            quantity = (
                min(remaining_value / resource_price, 10.0)  # Max 10 units)

            if quantity > 0.1:  # Minimum meaningful quantity
                wanted_resources[resource_type] = quantity
                remaining_value -= quantity * resource_price

        return wanted_resources

    def _evaluate_trade_offer(
        self,
        offer: TradeOffer,
        needs: Dict[ResourceType, float],
        market: Market,
        personality_profile,
    ) -> float:
        """Evaluate how good a trade offer is"""

        # Calculate value we're getting vs giving
        getting_value = sum(
            market.get_resource_value(resource_type, quantity)
            for resource_type, quantity in offer.offered_resources.items()
        )

        giving_value = sum(
            market.get_resource_value(resource_type, quantity)
            for resource_type, quantity in offer.wanted_resources.items()
        )

        if giving_value == 0:
            return 0.0

        value_ratio = getting_value / giving_value

        # Bonus for getting resources we actually need
        need_satisfaction = 0.0
        for resource_type, quantity in offer.offered_resources.items():
            if resource_type in needs:
                satisfaction = min(quantity / needs[resource_type], 1.0)
                need_satisfaction += satisfaction

        # Personality adjustments
        score = value_ratio * 0.7 + need_satisfaction * 0.3

        if personality_profile:
            # Risk tolerance affects willingness to trade
            risk_tolerance = (
                personality_profile.get_trait_value("risk_tolerance"))
            score *= 0.8 + risk_tolerance * 0.4

        return score

    def _execute_trade(self, agent: Agent, offer: TradeOffer, market: Market) -> Dict[str,
        Any]:
        """Execute a trade with another agent"""

        # In a real implementation, this would:
        # 1. Verify both agents have the required resources
        # 2. Transfer resources between agents
        # 3. Update agent inventories
        # 4. Record the trade in market history

        # For now, simulate the trade
        trade_record = {
            "timestamp": datetime.now(),
            "buyer_id": agent.agent_id,
            "seller_id": offer.offering_agent_id,
            "resources_traded": offer.offered_resources,
            "payment": offer.wanted_resources,
            "trade_value": offer.offer_value,
        }

        market.trade_history.append(trade_record)
        market.remove_offer(offer.offer_id)

        # Add to agent memory
        agent.add_to_memory(
            {
                "event": "trade_completed",
                "trade_partner": offer.offering_agent_id,
                "resources_received": offer.offered_resources,
                "resources_given": offer.wanted_resources,
                "trade_value": offer.offer_value,
            },
            is_important=True,
        )

        return {
            "success": True,
            "action": "trade_completed",
            "trade_partner": offer.offering_agent_id,
            "trade_value": offer.offer_value,
            "resources_received": offer.offered_resources,
        }

    def _evaluate_market_conditions(self, agent: Agent, market: Market) -> Dict[str,
        Any]:
        """Evaluate current market conditions"""

        market.update_market_trends()

        # Analyze market trends
        best_trend = max(market.market_trends.items(), key=lambda x: x[1])
        worst_trend = min(market.market_trends.items(), key=lambda x: x[1])

        agent.add_to_memory(
            {
                "event": "market_analysis",
                "best_trend": {"resource": best_trend[0].value,
                    "trend": best_trend[1]},
                "worst_trend": {
                    "resource": worst_trend[0].value,
                    "trend": worst_trend[1],
                },
                "total_offers": len(market.active_offers),
                "analysis_time": datetime.now().isoformat(),
            }
        )

        return {
            "success": True,
            "action": "market_evaluated",
            "best_trend": best_trend,
            "worst_trend": worst_trend,
            "total_offers": len(market.active_offers),
        }


class MarketAnalysisBehavior(BaseBehavior):
    """Market analysis and price prediction behavior"""

    def __init__(self) -> None:
        super().__init__(
            "market_analysis",
            BehaviorPriority.MEDIUM,
            {AgentCapability.LEARNING, AgentCapability.MEMORY},
        )

    def _can_execute_custom(self, agent: Agent, context: Dict[str,
        Any]) -> bool:
        # Can analyze if agent has trading experience
        trade_memories = [
            m
            for m in agent.short_term_memory + agent.long_term_memory
            if m.get("experience", {}).get("event") in ["trade_completed",
                "market_analysis"]
        ]
        return len(trade_memories) > 0

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str,
        Any]:
        market = agent.metadata.get("market")
        if not market:
            return {"success": False, "reason": "no_market_access"}

        # Analyze historical trade data
        analysis = self._analyze_market_patterns(market, agent)

        # Store analysis results
        agent.metadata["market_analysis"] = analysis
        agent.metadata["last_market_analysis"] = datetime.now()

        return {"success": True, "action": "market_analyzed",
            "insights": analysis}

    def _analyze_market_patterns(self, market: Market, agent: Agent) -> Dict[str,
        Any]:
        """Analyze market patterns and predict trends"""

        analysis = {
            "price_volatility": {},
            "trade_volume": len(market.trade_history),
            "profit_opportunities": [],
            "risk_assessment": "moderate",
        }

        # Analyze price volatility for each resource
        for resource_type in ResourceType:
            current_price = market.resource_prices.get(resource_type, 1.0)
            trend = market.market_trends.get(resource_type, 0.0)

            volatility = abs(trend)
            analysis["price_volatility"][resource_type.value] = volatility

            # Identify profit opportunities
            if trend > 0.2:  # Strong upward trend
                analysis["profit_opportunities"].append(
                    {
                        "resource": resource_type.value,
                        "action": "buy",
                        "confidence": min(trend, 1.0),
                        "expected_return": trend * 0.3,
                    }
                )
            elif trend < -0.2:  # Strong downward trend
                analysis["profit_opportunities"].append(
                    {
                        "resource": resource_type.value,
                        "action": "sell",
                        "confidence": min(abs(trend), 1.0),
                        "expected_return": abs(trend) * 0.2,
                    }
                )

        return analysis


class MerchantAgent(BaseAgent):
    """Specialized Merchant Agent with enhanced trading capabilities"""

    def __init__(self, **kwargs) -> None:
        # Set default merchant capabilities
        default_capabilities = {
            AgentCapability.COMMUNICATION,
            AgentCapability.RESOURCE_MANAGEMENT,
            AgentCapability.SOCIAL_INTERACTION,
            AgentCapability.MEMORY,
            AgentCapability.LEARNING,
        }

        # Merge with provided capabilities
        if "capabilities" in kwargs:
            kwargs["capabilities"] = (
                kwargs["capabilities"].union(default_capabilities))
        else:
            kwargs["capabilities"] = default_capabilities

        # Set merchant agent type
        kwargs["agent_type"] = "merchant"

        # Initialize base agent
        super().__init__(**kwargs)

        # Initialize merchant-specific components
        self._setup_merchant_components()

    def _setup_merchant_components(self):
        """Setup merchant-specific components"""
        # Initialize market access
        self.data.metadata["market"] = Market()
        self.data.metadata["trade_status"] = TradeStatus.IDLE
        self.data.metadata["total_trades"] = 0
        self.data.metadata["total_profit"] = 0.0
        self.data.metadata["trade_success_rate"] = 0.0

        # Add merchant-specific behaviors
        behavior_tree = self.get_component("behavior_tree")
        if behavior_tree:
            behavior_tree.add_behavior(TradingBehavior())
            behavior_tree.add_behavior(MarketAnalysisBehavior())

        # Set merchant goals
        self._setup_default_goals()

    def _setup_default_goals(self):
        ."""Setup default trading goals."""
        from agents.base.data_model import AgentGoal

        # Profit goal
        profit_goal = AgentGoal(
            goal_id="generate_profit",
            description="Generate profit through strategic trading",
            priority=0.9,
            target_position=None,
            deadline=datetime.now() + timedelta(hours=48),
        )
        self.data.add_goal(profit_goal)

        # Market presence goal
        market_goal = AgentGoal(
            goal_id="maintain_market_presence",
            description="Maintain active presence in trading markets",
            priority=0.7,
            target_position=None,
            deadline=datetime.now() + timedelta(hours=24),
        )
        self.data.add_goal(market_goal)

    def get_market(self) -> Market:
        """Get the agent's market interface"""
        return self.data.metadata.get("market")

    def get_trade_status(self) -> TradeStatus:
        """Get current trade status"""
        return self.data.metadata.get("trade_status", TradeStatus.IDLE)

    def get_trading_stats(self) -> Dict[str, Any]:
        """Get trading performance statistics"""
        return {
            "total_trades": self.data.metadata.get("total_trades", 0),
            "total_profit": self.data.metadata.get("total_profit", 0.0),
            "success_rate": self.data.metadata.get("trade_success_rate", 0.0),
            "market_analysis": self.data.metadata.get("market_analysis", {}),
            "last_analysis": self.data.metadata.get("last_market_analysis"),
        }

    def create_trade_offer(
        self,
        offered_resources: Dict[ResourceType, float],
        wanted_resources: Dict[ResourceType, float],
        expiration_hours: int = 6,
    ) -> str:
        """Create a new trade offer"""
        market = self.get_market()
        if not market:
            return None

        # Calculate offer value
        total_value = sum(
            market.get_resource_value(resource_type, quantity)
            for resource_type, quantity in offered_resources.items()
        )

        offer = TradeOffer(
            offering_agent_id=self.data.agent_id,
            offered_resources=offered_resources,
            wanted_resources=wanted_resources,
            offer_value=total_value,
            expiration=datetime.now() + timedelta(hours=expiration_hours),
        )

        market.add_offer(offer)
        return offer.offer_id

    def evaluate_trade_opportunity(self, offer_id: str) -> Dict[str, Any]:
        """Evaluate a specific trade opportunity"""
        market = self.get_market()
        if not market or offer_id not in market.active_offers:
            return {"error": "Offer not found"}

        offer = market.active_offers[offer_id]
        personality_profile = self.data.metadata.get("personality_profile")

        # Calculate value analysis
        getting_value = sum(
            market.get_resource_value(resource_type, quantity)
            for resource_type, quantity in offer.offered_resources.items()
        )

        giving_value = sum(
            market.get_resource_value(resource_type, quantity)
            for resource_type, quantity in offer.wanted_resources.items()
        )

        value_ratio = getting_value / giving_value if giving_value > 0 else 0

        # Personality-based risk assessment
        risk_score = 0.5  # Default moderate risk
        if personality_profile:
            risk_tolerance = (
                personality_profile.get_trait_value("risk_tolerance"))
            conscientiousness = (
                personality_profile.get_trait_value("conscientiousness"))

            # Higher risk tolerance and conscientiousness = better evaluation
            risk_score = (risk_tolerance + conscientiousness) / 2.0

        recommendation = "neutral"
        if value_ratio > 1.1 and risk_score > 0.6:
            recommendation = "accept"
        elif value_ratio < 0.9 or risk_score < 0.4:
            recommendation = "reject"

        return {
            "offer_id": offer_id,
            "value_ratio": value_ratio,
            "getting_value": getting_value,
            "giving_value": giving_value,
            "risk_score": risk_score,
            "recommendation": recommendation,
            "expires_at": offer.expiration.isoformat(),
        }


def create_merchant_agent(
    name: str = "Merchant", position: Optional[Position] = None, **kwargs
) -> MerchantAgent:
    """Factory function to create a merchant agent"""

    # Set default position if not provided
    if position is None:
        position = Position(0.0, 0.0, 0.0)

    # Create merchant with default configuration
    merchant = MerchantAgent(name=name, position=position, **kwargs)

    return merchant


def register_merchant_type() -> None:
    """Register the merchant type with the default factory"""
    factory = get_default_factory()

    def _create_merchant(**kwargs):
        return create_merchant_agent(**kwargs)

    factory.register_type("merchant", _create_merchant)

    # Set merchant-specific default config
    factory.set_default_config(
        "merchant",
        {
            "agent_type": "merchant",
            "capabilities": {
                AgentCapability.COMMUNICATION,
                AgentCapability.RESOURCE_MANAGEMENT,
                AgentCapability.SOCIAL_INTERACTION,
                AgentCapability.MEMORY,
                AgentCapability.LEARNING,
            },
        },
    )


# Auto-register when module is imported
register_merchant_type()
