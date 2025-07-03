"""
Comprehensive tests for Merchant Agent Trading System.

Tests the sophisticated trading agent system that includes market analysis,
trade negotiations, resource management, and behavioral economics for
economic simulation and agent-based trading ecosystems.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Mock the missing base agent modules before importing
from agents.merchant.merchant import (
    Market,
    MarketAnalysisBehavior,
    MerchantAgent,
    ResourceType,
    TradeOffer,
    TradeStatus,
    TradingBehavior,
    create_merchant_agent,
    register_merchant_type,
)

# Mock base agent modules that may not be available
sys.modules["agents.base"] = Mock()
sys.modules["agents.base.agent"] = Mock()
sys.modules["agents.base.behaviors"] = Mock()
sys.modules["agents.base.data_model"] = Mock()


# Create mock classes for the imports
class MockAgent:
    def __init__(self):
        self.agent_id = "test_agent"
        self.resources = Mock()
        self.resources.energy = 100
        self.resources.max_energy = 100
        self.resources.get_total_value = Mock(return_value=50.0)
        self.metadata = {}
        self.short_term_memory = []
        self.long_term_memory = []
        self.data = Mock()
        self.data.agent_id = "test_agent"
        self.data.metadata = {}
        self.data.add_goal = Mock()

    def add_to_memory(self, memory, is_important=False):
        if is_important:
            self.long_term_memory.append({"experience": memory})
        else:
            self.short_term_memory.append({"experience": memory})

    def get_component(self, component_name):
        if component_name == "behavior_tree":
            mock_tree = Mock()
            mock_tree.add_behavior = Mock()
            return mock_tree
        return None


class MockBaseAgent(MockAgent):
    def __init__(self, **kwargs):
        super().__init__()
        # Store kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAgentCapability:
    COMMUNICATION = "communication"
    RESOURCE_MANAGEMENT = "resource_management"
    SOCIAL_INTERACTION = "social_interaction"
    MEMORY = "memory"
    LEARNING = "learning"
    MOVEMENT = "movement"
    PERCEPTION = "perception"
    PLANNING = "planning"


class MockBaseBehavior:
    def __init__(self, name, priority, capabilities, interval=None):
        self.name = name
        self.priority = priority
        self.capabilities = capabilities
        self.interval = interval

    def _can_execute_custom(self, agent, context):
        return True

    def _execute_custom(self, agent, context):
        return {"success": True}


class MockBehaviorPriority:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MockPosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockAgentGoal:
    def __init__(self, goal_id, description, priority, target_position, deadline):
        self.goal_id = goal_id
        self.description = description
        self.priority = priority
        self.target_position = target_position
        self.deadline = deadline


class MockPersonalityProfile:
    def get_trait_value(self, trait_name):
        return 0.7  # Default value


# Set up the mocks
sys.modules["agents.base"].Agent = MockAgent
sys.modules["agents.base"].BaseAgent = MockBaseAgent
sys.modules["agents.base"].AgentCapability = MockAgentCapability
sys.modules["agents.base"].Position = MockPosition
sys.modules["agents.base"].get_default_factory = Mock(return_value=Mock())
sys.modules["agents.base.behaviors"].BaseBehavior = MockBaseBehavior
sys.modules["agents.base.behaviors"].BehaviorPriority = MockBehaviorPriority
sys.modules["agents.base.data_model"].AgentGoal = MockAgentGoal


class TestTradeStatus:
    """Test TradeStatus enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert TradeStatus.IDLE.value == "idle"
        assert TradeStatus.SEEKING_TRADES.value == "seeking_trades"
        assert TradeStatus.NEGOTIATING.value == "negotiating"
        assert TradeStatus.EXECUTING_TRADE.value == "executing_trade"
        assert TradeStatus.EVALUATING_MARKET.value == "evaluating_market"

    def test_enum_count(self):
        """Test correct number of enum values."""
        statuses = list(TradeStatus)
        assert len(statuses) == 5


class TestResourceType:
    """Test ResourceType enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert ResourceType.FOOD.value == "food"
        assert ResourceType.MATERIALS.value == "materials"
        assert ResourceType.TOOLS.value == "tools"
        assert ResourceType.INFORMATION.value == "information"
        assert ResourceType.SERVICES.value == "services"
        assert ResourceType.ENERGY.value == "energy"
        assert ResourceType.RARE_ITEMS.value == "rare_items"
        assert ResourceType.KNOWLEDGE.value == "knowledge"

    def test_enum_count(self):
        """Test correct number of resource types."""
        resources = list(ResourceType)
        assert len(resources) == 8


class TestTradeOffer:
    """Test TradeOffer class."""

    def test_trade_offer_creation(self):
        """Test creating trade offer with all fields."""
        expiration = datetime.now() + timedelta(hours=6)
        offered_resources = {ResourceType.FOOD: 10.0, ResourceType.ENERGY: 5.0}
        wanted_resources = {ResourceType.TOOLS: 2.0}

        offer = TradeOffer(
            offering_agent_id="agent123",
            offered_resources=offered_resources,
            wanted_resources=wanted_resources,
            offer_value=25.0,
            expiration=expiration,
        )

        assert offer.offering_agent_id == "agent123"
        assert offer.offered_resources == offered_resources
        assert offer.wanted_resources == wanted_resources
        assert offer.offer_value == 25.0
        assert offer.expiration == expiration
        assert offer.accepted is False
        assert offer.negotiated_value == 25.0
        assert offer.offer_id.startswith("trade_agent123_")
        assert isinstance(offer.created_at, datetime)

    def test_trade_offer_id_uniqueness(self):
        """Test that trade offer IDs are unique."""
        offer1 = TradeOffer("agent1", {}, {}, 10.0, datetime.now())
        offer2 = TradeOffer("agent2", {}, {}, 20.0, datetime.now())

        assert offer1.offer_id != offer2.offer_id


class TestMarket:
    """Test Market class."""

    def setup_method(self):
        """Set up test market."""
        self.market = Market()

    def test_market_initialization(self):
        """Test market initialization."""
        assert isinstance(self.market.resource_prices, dict)
        assert isinstance(self.market.active_offers, dict)
        assert isinstance(self.market.trade_history, list)
        assert isinstance(self.market.market_trends, dict)
        assert isinstance(self.market.last_updated, datetime)

        # Check that all resource types have prices and trends
        for resource_type in ResourceType:
            assert resource_type in self.market.resource_prices
            assert resource_type in self.market.market_trends
            assert self.market.resource_prices[resource_type] > 0
            assert -1.0 <= self.market.market_trends[resource_type] <= 1.0

    def test_get_resource_value(self):
        """Test resource value calculation."""
        # Test basic value calculation
        value = self.market.get_resource_value(ResourceType.FOOD, 10.0)
        assert value > 0

        # Test with zero quantity
        value = self.market.get_resource_value(ResourceType.FOOD, 0.0)
        assert value == 0

        # Test trend modifier effect
        original_trend = self.market.market_trends[ResourceType.FOOD]
        self.market.market_trends[ResourceType.FOOD] = 0.5  # Positive trend
        positive_value = self.market.get_resource_value(ResourceType.FOOD, 10.0)

        self.market.market_trends[ResourceType.FOOD] = -0.5  # Negative trend
        negative_value = self.market.get_resource_value(ResourceType.FOOD, 10.0)

        assert positive_value > negative_value

        # Restore original trend
        self.market.market_trends[ResourceType.FOOD] = original_trend

    def test_add_offer(self):
        """Test adding trade offer to market."""
        offer = TradeOffer(
            "agent1",
            {ResourceType.FOOD: 5.0},
            {ResourceType.TOOLS: 1.0},
            15.0,
            datetime.now() + timedelta(hours=6),
        )

        initial_count = len(self.market.active_offers)
        self.market.add_offer(offer)

        assert len(self.market.active_offers) == initial_count + 1
        assert offer.offer_id in self.market.active_offers
        assert self.market.active_offers[offer.offer_id] == offer

    def test_remove_offer(self):
        """Test removing trade offer from market."""
        offer = TradeOffer(
            "agent1",
            {ResourceType.FOOD: 5.0},
            {ResourceType.TOOLS: 1.0},
            15.0,
            datetime.now() + timedelta(hours=6),
        )

        # Add then remove
        self.market.add_offer(offer)
        assert offer.offer_id in self.market.active_offers

        result = self.market.remove_offer(offer.offer_id)
        assert result is True
        assert offer.offer_id not in self.market.active_offers

        # Try to remove non-existent offer
        result = self.market.remove_offer("non_existent_id")
        assert result is False

    def test_find_matching_offers(self):
        """Test finding matching trade offers."""
        # Create offers
        offer1 = TradeOffer(
            "agent1",
            {ResourceType.FOOD: 10.0},
            {ResourceType.TOOLS: 1.0},
            15.0,
            datetime.now() + timedelta(hours=6),
        )
        offer2 = TradeOffer(
            "agent2",
            {ResourceType.TOOLS: 2.0},
            {ResourceType.FOOD: 8.0},
            25.0,
            datetime.now() + timedelta(hours=6),
        )
        offer3 = TradeOffer(
            "agent3",
            {ResourceType.ENERGY: 5.0},
            {ResourceType.MATERIALS: 3.0},
            20.0,
            datetime.now() + timedelta(hours=6),
        )

        self.market.add_offer(offer1)
        self.market.add_offer(offer2)
        self.market.add_offer(offer3)

        # Look for food
        wanted = {ResourceType.FOOD: 5.0}
        matches = self.market.find_matching_offers(wanted)
        assert len(matches) == 1
        assert matches[0].offer_id == offer1.offer_id

        # Look for tools
        wanted = {ResourceType.TOOLS: 1.0}
        matches = self.market.find_matching_offers(wanted)
        assert len(matches) == 1
        assert matches[0].offer_id == offer2.offer_id

        # Look for non-existent resource amount
        wanted = {ResourceType.RARE_ITEMS: 1.0}
        matches = self.market.find_matching_offers(wanted)
        assert len(matches) == 0

    def test_update_market_trends(self):
        """Test market trend updates."""
        self.market.market_trends.copy()
        original_prices = self.market.resource_prices.copy()

        # Update multiple times to see changes
        for _ in range(10):
            self.market.update_market_trends()

        # Trends should stay within bounds
        for resource_type in ResourceType:
            assert -1.0 <= self.market.market_trends[resource_type] <= 1.0
            assert self.market.resource_prices[resource_type] > 0

        # Prices should have changed
        price_changed = any(
            self.market.resource_prices[rt] != original_prices[rt] for rt in ResourceType
        )
        assert price_changed


class TestTradingBehavior:
    """Test TradingBehavior class."""

    def setup_method(self):
        """Set up test trading behavior."""
        self.behavior = TradingBehavior()
        self.agent = MockAgent()
        self.market = Market()
        self.agent.metadata["market"] = self.market
        self.agent.metadata["personality_profile"] = MockPersonalityProfile()

    def test_trading_behavior_initialization(self):
        """Test trading behavior initialization."""
        assert self.behavior.name == "trading"
        assert self.behavior.priority == MockBehaviorPriority.HIGH
        assert self.behavior.negotiation_rounds == 3
        assert MockAgentCapability.COMMUNICATION in self.behavior.capabilities
        assert MockAgentCapability.RESOURCE_MANAGEMENT in self.behavior.capabilities

    def test_can_execute_custom(self):
        """Test can execute custom logic."""
        context = {}

        # Should not execute without market
        agent_no_market = MockAgent()
        result = self.behavior._can_execute_custom(agent_no_market, context)
        assert result is False

        # Should execute with market and offers
        offer = TradeOffer(
            "other_agent",
            {ResourceType.FOOD: 5.0},
            {ResourceType.TOOLS: 1.0},
            15.0,
            datetime.now() + timedelta(hours=6),
        )
        self.market.add_offer(offer)

        result = self.behavior._can_execute_custom(self.agent, context)
        assert result is True

        # Should execute with high resource value
        self.market.active_offers.clear()
        self.agent.resources.get_total_value.return_value = 50.0
        result = self.behavior._can_execute_custom(self.agent, context)
        assert result is True

    def test_execute_custom(self):
        """Test execute custom trading logic."""
        context = {}

        # Test without market
        agent_no_market = MockAgent()
        result = self.behavior._execute_custom(agent_no_market, context)
        assert result["success"] is False
        assert result["reason"] == "no_market_access"

        # Test with market
        result = self.behavior._execute_custom(self.agent, context)
        assert isinstance(result, dict)
        assert "success" in result
        assert self.agent.metadata["trade_status"] == TradeStatus.SEEKING_TRADES

    def test_assess_resource_situation(self):
        """Test resource situation assessment."""
        # Test with low energy
        self.agent.resources.energy = 25
        self.agent.resources.max_energy = 100

        needs, surpluses = self.behavior._assess_resource_situation(self.agent)

        assert isinstance(needs, dict)
        assert isinstance(surpluses, dict)
        assert ResourceType.ENERGY in needs

        # Test with high energy
        self.agent.resources.energy = 85
        self.agent.resources.max_energy = 100

        needs, surpluses = self.behavior._assess_resource_situation(self.agent)
        assert ResourceType.ENERGY in surpluses

    def test_create_sell_offer(self):
        """Test creating sell offer."""
        surpluses = {ResourceType.FOOD: 10.0, ResourceType.ENERGY: 8.0}

        result = self.behavior._create_sell_offer(
            self.agent, self.market, surpluses, MockPersonalityProfile()
        )

        assert result["success"] is True
        assert result["action"] == "sell_offer_created"
        assert "offer_id" in result
        assert "offered_resources" in result
        assert "wanted_resources" in result
        assert "offer_value" in result

        # Check that offer was added to market
        assert result["offer_id"] in self.market.active_offers

    def test_find_buy_opportunities(self):
        """Test finding buy opportunities."""
        needs = {ResourceType.TOOLS: 2.0}

        # Test with no matching offers
        result = self.behavior._find_buy_opportunities(
            self.agent, self.market, needs, MockPersonalityProfile()
        )
        assert result["success"] is False
        assert result["reason"] == "no_matching_offers"

        # Add matching offer
        offer = TradeOffer(
            "seller_agent",
            {ResourceType.TOOLS: 3.0},
            {ResourceType.FOOD: 5.0},
            20.0,
            datetime.now() + timedelta(hours=6),
        )
        self.market.add_offer(offer)

        result = self.behavior._find_buy_opportunities(
            self.agent, self.market, needs, MockPersonalityProfile()
        )

        # Result depends on offer evaluation, but should at least process
        assert isinstance(result, dict)
        assert "success" in result

    def test_evaluate_trade_offer(self):
        """Test trade offer evaluation."""
        offer = TradeOffer(
            "seller_agent",
            {ResourceType.TOOLS: 2.0},
            {ResourceType.FOOD: 3.0},
            20.0,
            datetime.now() + timedelta(hours=6),
        )
        needs = {ResourceType.TOOLS: 1.0}

        score = self.behavior._evaluate_trade_offer(
            offer, needs, self.market, MockPersonalityProfile()
        )

        assert isinstance(score, float)
        assert score >= 0

    def test_execute_trade(self):
        """Test trade execution."""
        offer = TradeOffer(
            "seller_agent",
            {ResourceType.TOOLS: 2.0},
            {ResourceType.FOOD: 3.0},
            20.0,
            datetime.now() + timedelta(hours=6),
        )
        self.market.add_offer(offer)

        result = self.behavior._execute_trade(self.agent, offer, self.market)

        assert result["success"] is True
        assert result["action"] == "trade_completed"
        assert result["trade_partner"] == "seller_agent"
        assert result["trade_value"] == 20.0

        # Check trade was recorded
        assert len(self.market.trade_history) == 1
        assert offer.offer_id not in self.market.active_offers

        # Check memory was updated
        assert len(self.agent.long_term_memory) == 1

    def test_evaluate_market_conditions(self):
        """Test market conditions evaluation."""
        result = self.behavior._evaluate_market_conditions(self.agent, self.market)

        assert result["success"] is True
        assert result["action"] == "market_evaluated"
        assert "best_trend" in result
        assert "worst_trend" in result
        assert "total_offers" in result

        # Check memory was updated
        assert len(self.agent.short_term_memory) == 1


class TestMarketAnalysisBehavior:
    """Test MarketAnalysisBehavior class."""

    def setup_method(self):
        """Set up test market analysis behavior."""
        self.behavior = MarketAnalysisBehavior()
        self.agent = MockAgent()
        self.market = Market()
        self.agent.metadata["market"] = self.market

    def test_market_analysis_behavior_initialization(self):
        """Test market analysis behavior initialization."""
        assert self.behavior.name == "market_analysis"
        assert self.behavior.priority == MockBehaviorPriority.MEDIUM
        assert MockAgentCapability.LEARNING in self.behavior.capabilities
        assert MockAgentCapability.MEMORY in self.behavior.capabilities

    def test_can_execute_custom(self):
        """Test can execute custom logic."""
        context = {}

        # Should not execute without trading experience
        result = self.behavior._can_execute_custom(self.agent, context)
        assert result is False

        # Add trading experience
        self.agent.short_term_memory.append({"experience": {"event": "trade_completed"}})

        result = self.behavior._can_execute_custom(self.agent, context)
        assert result is True

        # Test with market analysis experience
        self.agent.short_term_memory.clear()
        self.agent.long_term_memory.append({"experience": {"event": "market_analysis"}})

        result = self.behavior._can_execute_custom(self.agent, context)
        assert result is True

    def test_execute_custom(self):
        """Test execute custom market analysis."""
        context = {}

        # Test without market
        agent_no_market = MockAgent()
        result = self.behavior._execute_custom(agent_no_market, context)
        assert result["success"] is False
        assert result["reason"] == "no_market_access"

        # Test with market
        result = self.behavior._execute_custom(self.agent, context)
        assert result["success"] is True
        assert result["action"] == "market_analyzed"
        assert "insights" in result

        # Check metadata was updated
        assert "market_analysis" in self.agent.metadata
        assert "last_market_analysis" in self.agent.metadata

    def test_analyze_market_patterns(self):
        """Test market pattern analysis."""
        analysis = self.behavior._analyze_market_patterns(self.market, self.agent)

        assert "price_volatility" in analysis
        assert "trade_volume" in analysis
        assert "profit_opportunities" in analysis
        assert "risk_assessment" in analysis

        assert isinstance(analysis["price_volatility"], dict)
        assert isinstance(analysis["profit_opportunities"], list)
        assert analysis["trade_volume"] == 0  # No trades yet

        # Check volatility data for all resources
        for resource_type in ResourceType:
            assert resource_type.value in analysis["price_volatility"]


class TestMerchantAgent:
    """Test MerchantAgent class."""

    def setup_method(self):
        """Set up test merchant agent."""
        self.merchant = MerchantAgent(name="TestMerchant")

    def test_merchant_agent_initialization(self):
        """Test merchant agent initialization."""
        assert self.merchant.agent_type == "merchant"
        assert MockAgentCapability.COMMUNICATION in self.merchant.capabilities
        assert MockAgentCapability.RESOURCE_MANAGEMENT in self.merchant.capabilities
        assert MockAgentCapability.SOCIAL_INTERACTION in self.merchant.capabilities
        assert MockAgentCapability.MEMORY in self.merchant.capabilities
        assert MockAgentCapability.LEARNING in self.merchant.capabilities

        # Check merchant-specific metadata
        assert "market" in self.merchant.data.metadata
        assert "trade_status" in self.merchant.data.metadata
        assert "total_trades" in self.merchant.data.metadata
        assert "total_profit" in self.merchant.data.metadata
        assert "trade_success_rate" in self.merchant.data.metadata

        # Check default values
        assert self.merchant.data.metadata["trade_status"] == TradeStatus.IDLE
        assert self.merchant.data.metadata["total_trades"] == 0
        assert self.merchant.data.metadata["total_profit"] == 0.0
        assert self.merchant.data.metadata["trade_success_rate"] == 0.0

    def test_get_market(self):
        """Test get market method."""
        market = self.merchant.get_market()
        assert isinstance(market, Market)

    def test_get_trade_status(self):
        """Test get trade status method."""
        status = self.merchant.get_trade_status()
        assert status == TradeStatus.IDLE

        # Update status
        self.merchant.data.metadata["trade_status"] = TradeStatus.NEGOTIATING
        status = self.merchant.get_trade_status()
        assert status == TradeStatus.NEGOTIATING

    def test_get_trading_stats(self):
        """Test get trading statistics."""
        stats = self.merchant.get_trading_stats()

        assert "total_trades" in stats
        assert "total_profit" in stats
        assert "success_rate" in stats
        assert "market_analysis" in stats
        assert "last_analysis" in stats

        assert stats["total_trades"] == 0
        assert stats["total_profit"] == 0.0
        assert stats["success_rate"] == 0.0

    def test_create_trade_offer(self):
        """Test creating trade offer."""
        offered_resources = {ResourceType.FOOD: 10.0}
        wanted_resources = {ResourceType.TOOLS: 2.0}

        offer_id = self.merchant.create_trade_offer(
            offered_resources, wanted_resources, expiration_hours=12
        )

        assert offer_id is not None
        assert isinstance(offer_id, str)

        # Check offer was added to market
        market = self.merchant.get_market()
        assert offer_id in market.active_offers

        offer = market.active_offers[offer_id]
        assert offer.offering_agent_id == self.merchant.data.agent_id
        assert offer.offered_resources == offered_resources
        assert offer.wanted_resources == wanted_resources

    def test_create_trade_offer_no_market(self):
        """Test creating trade offer without market."""
        # Remove market
        self.merchant.data.metadata["market"] = None

        offered_resources = {ResourceType.FOOD: 10.0}
        wanted_resources = {ResourceType.TOOLS: 2.0}

        offer_id = self.merchant.create_trade_offer(offered_resources, wanted_resources)

        assert offer_id is None

    def test_evaluate_trade_opportunity(self):
        """Test evaluating trade opportunity."""
        # Create an offer first
        offered_resources = {ResourceType.FOOD: 10.0}
        wanted_resources = {ResourceType.TOOLS: 2.0}
        offer_id = self.merchant.create_trade_offer(offered_resources, wanted_resources)

        # Evaluate the offer
        evaluation = self.merchant.evaluate_trade_opportunity(offer_id)

        assert "offer_id" in evaluation
        assert "value_ratio" in evaluation
        assert "getting_value" in evaluation
        assert "giving_value" in evaluation
        assert "risk_score" in evaluation
        assert "recommendation" in evaluation
        assert "expires_at" in evaluation

        assert evaluation["offer_id"] == offer_id
        assert isinstance(evaluation["value_ratio"], float)
        assert evaluation["recommendation"] in ["accept", "reject", "neutral"]

    def test_evaluate_trade_opportunity_not_found(self):
        """Test evaluating non-existent trade opportunity."""
        evaluation = self.merchant.evaluate_trade_opportunity("non_existent_id")

        assert "error" in evaluation
        assert evaluation["error"] == "Offer not found"

    def test_evaluate_trade_opportunity_with_personality(self):
        """Test evaluating trade opportunity with personality profile."""
        # Add personality profile
        self.merchant.data.metadata["personality_profile"] = MockPersonalityProfile()

        # Create and evaluate offer
        offered_resources = {ResourceType.FOOD: 10.0}
        wanted_resources = {ResourceType.TOOLS: 2.0}
        offer_id = self.merchant.create_trade_offer(offered_resources, wanted_resources)

        evaluation = self.merchant.evaluate_trade_opportunity(offer_id)

        assert "risk_score" in evaluation
        assert isinstance(evaluation["risk_score"], float)
        assert 0 <= evaluation["risk_score"] <= 1


class TestCreateMerchantAgent:
    """Test create_merchant_agent factory function."""

    def test_create_merchant_agent_defaults(self):
        """Test creating merchant agent with defaults."""
        merchant = create_merchant_agent()

        assert merchant.name == "Merchant"
        assert hasattr(merchant, "position")
        assert isinstance(merchant, MerchantAgent)

    def test_create_merchant_agent_custom(self):
        """Test creating merchant agent with custom parameters."""
        position = MockPosition(10.0, 20.0, 30.0)
        merchant = create_merchant_agent(
            name="CustomMerchant", position=position, custom_param="test_value"
        )

        assert merchant.name == "CustomMerchant"
        assert merchant.position == position
        assert merchant.custom_param == "test_value"


class TestRegisterMerchantType:
    """Test register_merchant_type function."""

    def test_register_merchant_type(self):
        """Test registering merchant type."""
        # Mock the factory
        mock_factory = Mock()

        with patch("agents.merchant.merchant.get_default_factory", return_value=mock_factory):
            register_merchant_type()

            # Check that register_type was called
            mock_factory.register_type.assert_called_once()
            args = mock_factory.register_type.call_args
            assert args[0][0] == "merchant"

            # Check that set_default_config was called
            mock_factory.set_default_config.assert_called_once()
            config_args = mock_factory.set_default_config.call_args
            assert config_args[0][0] == "merchant"
            assert config_args[0][1]["agent_type"] == "merchant"


class TestIntegrationScenarios:
    """Test integrated merchant trading scenarios."""

    def setup_method(self):
        """Set up integration test scenario."""
        self.merchant1 = MerchantAgent(name="Merchant1")
        self.merchant2 = MerchantAgent(name="Merchant2")

        # Share the same market
        shared_market = Market()
        self.merchant1.data.metadata["market"] = shared_market
        self.merchant2.data.metadata["market"] = shared_market

    def test_basic_trading_scenario(self):
        """Test basic trading scenario between merchants."""
        market = self.merchant1.get_market()

        # Merchant1 creates a sell offer
        offered_resources = {ResourceType.FOOD: 20.0}
        wanted_resources = {ResourceType.TOOLS: 3.0}
        offer_id = self.merchant1.create_trade_offer(offered_resources, wanted_resources)

        assert offer_id in market.active_offers
        offer = market.active_offers[offer_id]

        # Merchant2 evaluates the offer
        evaluation = self.merchant2.evaluate_trade_opportunity(offer_id)
        assert evaluation["offer_id"] == offer_id

        # Simulate trade execution
        behavior = TradingBehavior()
        trade_result = behavior._execute_trade(self.merchant2, offer, market)

        assert trade_result["success"] is True
        assert trade_result["trade_partner"] == self.merchant1.data.agent_id
        assert offer_id not in market.active_offers
        assert len(market.trade_history) == 1

    def test_market_dynamics_scenario(self):
        """Test market dynamics with multiple trades."""
        market = self.merchant1.get_market()

        # Record initial prices
        market.resource_prices[ResourceType.FOOD]

        # Create multiple offers of same resource type
        for i in range(5):
            _ = self.merchant1.create_trade_offer(
                {ResourceType.FOOD: 10.0}, {ResourceType.TOOLS: 1.0}
            )

            # Simulate market activity
            market.update_market_trends()

        # Check that market has evolved
        assert len(market.active_offers) == 5

        # Price may have changed due to market dynamics
        current_food_price = market.resource_prices[ResourceType.FOOD]
        # Prices should still be positive
        assert current_food_price > 0

    def test_complex_negotiation_scenario(self):
        """Test complex negotiation scenario."""
        market = self.merchant1.get_market()

        # Create multiple competing offers
        offer1_id = self.merchant1.create_trade_offer(
            {ResourceType.FOOD: 15.0}, {ResourceType.TOOLS: 2.0}
        )

        # Different merchant creates competing offer
        offer2_id = self.merchant2.create_trade_offer(
            {ResourceType.FOOD: 12.0}, {ResourceType.TOOLS: 1.5}
        )

        # Find matching offers
        wanted = {ResourceType.FOOD: 10.0}
        matches = market.find_matching_offers(wanted)

        assert len(matches) == 2
        assert any(match.offer_id == offer1_id for match in matches)
        assert any(match.offer_id == offer2_id for match in matches)

        # Test offer evaluation comparison
        behavior = TradingBehavior()
        personality = MockPersonalityProfile()

        score1 = behavior._evaluate_trade_offer(
            market.active_offers[offer1_id], wanted, market, personality
        )
        score2 = behavior._evaluate_trade_offer(
            market.active_offers[offer2_id], wanted, market, personality
        )

        # Both should be valid scores
        assert isinstance(score1, float) and score1 >= 0
        assert isinstance(score2, float) and score2 >= 0
