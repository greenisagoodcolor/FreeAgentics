"""
Comprehensive tests for Resource and Business Model Mechanics.

Tests the resource management system, economic transactions, market mechanics,
and business logic for agent interactions including inventory management,
trade protocols, and price dynamics.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from agents.base.resource_business_model import (
    ResourceType,
    TransactionType,
    MarketCondition,
    ResourceUnit,
    MarketPrice,
    TradeOffer,
    Transaction,
    IResourceProvider,
    IResourceConsumer,
    ResourceInventory,
    Market,
    ResourceBusinessManager,
)


class TestResourceType:
    """Test ResourceType enum."""
    
    def test_resource_type_enum_values(self):
        """Test resource type enum has expected values."""
        # Basic resources
        assert ResourceType.ENERGY.value == "energy"
        assert ResourceType.FOOD.value == "food"
        assert ResourceType.WATER.value == "water"
        assert ResourceType.MATERIALS.value == "materials"
        
        # Crafted resources
        assert ResourceType.TOOLS.value == "tools"
        assert ResourceType.SHELTER.value == "shelter"
        assert ResourceType.WEAPONS.value == "weapons"
        assert ResourceType.TECHNOLOGY.value == "technology"
        
        # Social resources
        assert ResourceType.INFORMATION.value == "information"
        assert ResourceType.INFLUENCE.value == "influence"
        assert ResourceType.REPUTATION.value == "reputation"
        assert ResourceType.RELATIONSHIPS.value == "relationships"
        
        # Economic resources
        assert ResourceType.CURRENCY.value == "currency"
        assert ResourceType.CREDIT.value == "credit"
        assert ResourceType.CONTRACTS.value == "contracts"
        assert ResourceType.SERVICES.value == "services"
    
    def test_resource_type_enum_count(self):
        """Test correct number of resource types."""
        resource_types = list(ResourceType)
        assert len(resource_types) == 16


class TestTransactionType:
    """Test TransactionType enum."""
    
    def test_transaction_type_enum_values(self):
        """Test transaction type enum values."""
        assert TransactionType.TRADE.value == "trade"
        assert TransactionType.PURCHASE.value == "purchase"
        assert TransactionType.SALE.value == "sale"
        assert TransactionType.BARTER.value == "barter"
        assert TransactionType.LOAN.value == "loan"
        assert TransactionType.INVESTMENT.value == "investment"
        assert TransactionType.GIFT.value == "gift"
        assert TransactionType.THEFT.value == "theft"
        assert TransactionType.TRIBUTE.value == "tribute"
        assert TransactionType.TAX.value == "tax"
    
    def test_transaction_type_enum_count(self):
        """Test correct number of transaction types."""
        transaction_types = list(TransactionType)
        assert len(transaction_types) == 10


class TestMarketCondition:
    """Test MarketCondition enum."""
    
    def test_market_condition_enum_values(self):
        """Test market condition enum values."""
        assert MarketCondition.BULL.value == "bull"
        assert MarketCondition.BEAR.value == "bear"
        assert MarketCondition.STABLE.value == "stable"
        assert MarketCondition.VOLATILE.value == "volatile"
        assert MarketCondition.CRASHED.value == "crashed"


class TestResourceUnit:
    """Test ResourceUnit dataclass."""
    
    def test_resource_unit_creation(self):
        """Test creating resource unit with all fields."""
        metadata = {"source": "factory", "batch": "A123"}
        
        unit = ResourceUnit(
            resource_type=ResourceType.TOOLS,
            quantity=10.0,
            quality=0.9,
            durability=0.8,
            origin="agent_123",
            age=5.0,
            metadata=metadata
        )
        
        assert unit.resource_type == ResourceType.TOOLS
        assert unit.quantity == 10.0
        assert unit.quality == 0.9
        assert unit.durability == 0.8
        assert unit.origin == "agent_123"
        assert unit.age == 5.0
        assert unit.metadata == metadata
    
    def test_resource_unit_defaults(self):
        """Test default values for optional fields."""
        unit = ResourceUnit(
            resource_type=ResourceType.FOOD,
            quantity=5.0
        )
        
        assert unit.quality == 1.0
        assert unit.durability == 1.0
        assert unit.origin is None
        assert unit.age == 0.0
        assert unit.metadata == {}
    
    def test_get_effective_value(self):
        """Test effective value calculation."""
        unit = ResourceUnit(
            resource_type=ResourceType.TOOLS,
            quantity=10.0,
            quality=0.8,
            durability=0.5
        )
        
        effective_value = unit.get_effective_value()
        assert effective_value == 10.0 * 0.8 * 0.5
        assert effective_value == 4.0
    
    def test_degrade_tools(self):
        """Test degradation for tools/weapons/shelter."""
        unit = ResourceUnit(
            resource_type=ResourceType.TOOLS,
            quantity=10.0,
            quality=1.0,
            durability=1.0
        )
        
        # Test degradation
        unit.degrade(rate=0.1)
        assert unit.durability == 0.9
        assert unit.quality == 1.0  # Quality unchanged for tools
        assert unit.age == 1.0
        
        # Test multiple degradations
        for _ in range(9):
            unit.degrade(rate=0.1)
        assert unit.durability == pytest.approx(0.0, abs=1e-10)
        assert unit.age == 10.0
    
    def test_degrade_other_resources(self):
        """Test degradation for non-durable resources."""
        unit = ResourceUnit(
            resource_type=ResourceType.FOOD,
            quantity=10.0,
            quality=1.0,
            durability=1.0
        )
        
        # Test degradation
        unit.degrade(rate=0.1)
        assert unit.quality == 0.99  # Quality degrades at 10% of rate
        assert unit.durability == 1.0  # Durability unchanged for food
        assert unit.age == 1.0
    
    def test_degrade_minimum_values(self):
        """Test degradation doesn't go below zero."""
        unit = ResourceUnit(
            resource_type=ResourceType.TOOLS,
            quantity=10.0,
            quality=0.05,
            durability=0.05
        )
        
        # Degrade past zero
        unit.degrade(rate=0.1)
        assert unit.durability == 0.0
        assert unit.quality == 0.05


class TestMarketPrice:
    """Test MarketPrice dataclass."""
    
    def setup_method(self):
        """Set up test market price."""
        self.price = MarketPrice(
            resource_type=ResourceType.FOOD,
            base_price=10.0,
            current_price=10.0
        )
    
    def test_market_price_creation(self):
        """Test creating market price with all fields."""
        price = MarketPrice(
            resource_type=ResourceType.ENERGY,
            base_price=5.0,
            current_price=6.0,
            supply=100.0,
            demand=120.0,
            volatility=0.2
        )
        
        assert price.resource_type == ResourceType.ENERGY
        assert price.base_price == 5.0
        assert price.current_price == 6.0
        assert price.supply == 100.0
        assert price.demand == 120.0
        assert price.volatility == 0.2
        assert isinstance(price.last_updated, datetime)
        assert price.price_history == []
    
    @patch('agents.base.resource_business_model.datetime')
    def test_update_price_basic(self, mock_datetime):
        """Test basic price update based on supply and demand."""
        # Mock datetime
        now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        mock_datetime.timezone = timezone
        
        # Test balanced supply and demand
        self.price.update_price(supply=100.0, demand=100.0)
        assert self.price.current_price == 10.0  # Should remain stable
        assert self.price.supply == 100.0
        assert self.price.demand == 100.0
        assert len(self.price.price_history) == 1
        assert self.price.price_history[0] == (now, 10.0)
    
    def test_update_price_high_demand(self):
        """Test price increase with high demand."""
        self.price.update_price(supply=50.0, demand=150.0)
        # Price should increase (demand 3x supply)
        # New price = base_price * (demand/supply) = 10 * 3 = 30
        # With inertia: current * 0.7 + new * 0.3 = 10 * 0.7 + 30 * 0.3 = 16
        assert self.price.current_price == pytest.approx(16.0)
    
    def test_update_price_no_supply(self):
        """Test price update when there's no supply."""
        self.price.update_price(supply=0.0, demand=100.0)
        # Price factor = 2.0 when no supply
        # New price = base_price * 2 = 20
        # With inertia: 10 * 0.7 + 20 * 0.3 = 13
        assert self.price.current_price == pytest.approx(13.0)
    
    def test_update_price_volatility(self):
        """Test volatility calculation."""
        initial_volatility = self.price.volatility
        
        # First update (establishes price history)
        self.price.update_price(supply=100.0, demand=100.0)
        # Volatility not updated on first update
        assert self.price.volatility == initial_volatility
        
        # Second update with no price change
        self.price.update_price(supply=100.0, demand=100.0)
        # No price change, volatility should decrease
        assert self.price.volatility < initial_volatility
        
        # Third update with large change
        self.price.update_price(supply=10.0, demand=100.0)
        # Large price change should increase volatility
        assert self.price.volatility > 0.0
    
    def test_update_price_history_limit(self):
        """Test price history is limited to 100 entries."""
        for i in range(105):
            self.price.update_price(supply=100.0, demand=100.0 + i)
        
        assert len(self.price.price_history) == 100
    
    def test_get_trend_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        assert self.price.get_trend() == "insufficient_data"
        
        # Add some data but not enough
        for i in range(3):
            self.price.update_price(supply=100.0, demand=100.0)
        
        assert self.price.get_trend() == "insufficient_data"
    
    def test_get_trend_stable(self):
        """Test stable trend detection."""
        # Add stable price history
        for i in range(5):
            self.price.update_price(supply=100.0, demand=100.0)
        
        trend = self.price.get_trend()
        assert trend == "stable"
    
    def test_get_trend_rising(self):
        """Test rising trend detection."""
        # Start with some baseline
        self.price.update_price(supply=100.0, demand=100.0)
        
        # Add rising prices
        demands = [100.0, 110.0, 120.0, 130.0, 140.0]
        for demand in demands:
            self.price.update_price(supply=100.0, demand=demand)
        
        trend = self.price.get_trend()
        assert trend == "rising"
    
    def test_get_trend_falling(self):
        """Test falling trend detection."""
        # Start with high demand
        self.price.update_price(supply=100.0, demand=200.0)
        
        # Add falling prices
        demands = [180.0, 160.0, 140.0, 120.0, 100.0]
        for demand in demands:
            self.price.update_price(supply=100.0, demand=demand)
        
        trend = self.price.get_trend()
        assert trend == "falling"


class TestTradeOffer:
    """Test TradeOffer dataclass."""
    
    def setup_method(self):
        """Set up test trade offer."""
        self.expiration = datetime.now(timezone.utc) + timedelta(hours=1)
        self.offer = TradeOffer(
            offer_id="offer_123",
            from_agent="agent_1",
            to_agent="agent_2",
            offered_resources={ResourceType.FOOD: 10.0, ResourceType.WATER: 5.0},
            requested_resources={ResourceType.ENERGY: 15.0},
            expiration=self.expiration
        )
    
    def test_trade_offer_creation(self):
        """Test creating trade offer with all fields."""
        terms = {"delivery": "immediate", "location": "market_square"}
        
        offer = TradeOffer(
            offer_id="offer_456",
            from_agent="agent_a",
            to_agent=None,  # Public offer
            offered_resources={ResourceType.TOOLS: 2.0},
            requested_resources={ResourceType.MATERIALS: 20.0},
            expiration=self.expiration,
            is_public=True,
            minimum_reputation=0.5,
            max_distance=100.0,
            additional_terms=terms
        )
        
        assert offer.offer_id == "offer_456"
        assert offer.from_agent == "agent_a"
        assert offer.to_agent is None
        assert offer.offered_resources == {ResourceType.TOOLS: 2.0}
        assert offer.requested_resources == {ResourceType.MATERIALS: 20.0}
        assert offer.expiration == self.expiration
        assert offer.is_public is True
        assert offer.minimum_reputation == 0.5
        assert offer.max_distance == 100.0
        assert offer.additional_terms == terms
        assert isinstance(offer.created_at, datetime)
    
    def test_trade_offer_defaults(self):
        """Test default values for optional fields."""
        assert self.offer.is_public is True
        assert self.offer.minimum_reputation == 0.0
        assert self.offer.max_distance is None
        assert self.offer.additional_terms == {}
    
    def test_is_valid_active(self):
        """Test validity check for active offer."""
        assert self.offer.is_valid() is True
    
    def test_is_valid_expired(self):
        """Test validity check for expired offer."""
        expired_offer = TradeOffer(
            offer_id="expired",
            from_agent="agent_1",
            to_agent="agent_2",
            offered_resources={ResourceType.FOOD: 10.0},
            requested_resources={ResourceType.ENERGY: 15.0},
            expiration=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        
        assert expired_offer.is_valid() is False
    
    def test_calculate_value_ratio_normal(self):
        """Test value ratio calculation with normal prices."""
        market_prices = {
            ResourceType.FOOD: MarketPrice(ResourceType.FOOD, 2.0, 2.0),
            ResourceType.WATER: MarketPrice(ResourceType.WATER, 1.0, 1.0),
            ResourceType.ENERGY: MarketPrice(ResourceType.ENERGY, 3.0, 3.0),
        }
        
        ratio = self.offer.calculate_value_ratio(market_prices)
        # Offered: 10*2 + 5*1 = 25
        # Requested: 15*3 = 45
        # Ratio: 25/45 = 0.556
        assert ratio == pytest.approx(25.0 / 45.0)
    
    def test_calculate_value_ratio_missing_prices(self):
        """Test value ratio with missing market prices."""
        market_prices = {
            ResourceType.FOOD: MarketPrice(ResourceType.FOOD, 2.0, 2.0),
            # WATER and ENERGY prices missing
        }
        
        ratio = self.offer.calculate_value_ratio(market_prices)
        # Missing prices default to 1.0
        # Offered: 10*2 + 5*1 = 25
        # Requested: 15*1 = 15
        # Ratio: 25/15 = 1.667
        assert ratio == pytest.approx(25.0 / 15.0)
    
    def test_calculate_value_ratio_no_requested(self):
        """Test value ratio when no resources requested (gift)."""
        gift_offer = TradeOffer(
            offer_id="gift",
            from_agent="agent_1",
            to_agent="agent_2",
            offered_resources={ResourceType.FOOD: 10.0},
            requested_resources={},
            expiration=self.expiration
        )
        
        ratio = gift_offer.calculate_value_ratio({})
        assert ratio == float("inf")


class TestTransaction:
    """Test Transaction dataclass."""
    
    def test_transaction_creation(self):
        """Test creating transaction with all fields."""
        metadata = {"contract_id": "contract_123", "witness": "agent_w"}
        resources = {
            "agent_1": {ResourceType.FOOD: 10.0},
            "agent_2": {ResourceType.ENERGY: 15.0}
        }
        
        transaction = Transaction(
            transaction_id="trans_123",
            transaction_type=TransactionType.TRADE,
            parties=["agent_1", "agent_2"],
            resources_exchanged=resources,
            total_value=100.0,
            fee=5.0,
            location="market_1",
            metadata=metadata
        )
        
        assert transaction.transaction_id == "trans_123"
        assert transaction.transaction_type == TransactionType.TRADE
        assert transaction.parties == ["agent_1", "agent_2"]
        assert transaction.resources_exchanged == resources
        assert transaction.total_value == 100.0
        assert transaction.fee == 5.0
        assert transaction.location == "market_1"
        assert transaction.metadata == metadata
        assert isinstance(transaction.timestamp, datetime)
    
    def test_transaction_defaults(self):
        """Test default values for optional fields."""
        transaction = Transaction(
            transaction_id="trans_456",
            transaction_type=TransactionType.PURCHASE,
            parties=["agent_a"],
            resources_exchanged={},
            total_value=50.0
        )
        
        assert transaction.fee == 0.0
        assert transaction.location is None
        assert transaction.metadata == {}


class TestResourceInventory:
    """Test ResourceInventory class."""
    
    def setup_method(self):
        """Set up test inventory."""
        self.inventory = ResourceInventory("agent_123", initial_capacity=100.0)
    
    def test_inventory_initialization(self):
        """Test inventory initialization."""
        assert self.inventory.agent_id == "agent_123"
        assert self.inventory.capacity == 100.0
        assert self.inventory.resources == {}
        assert self.inventory.reserved == {}
    
    def test_add_resource_success(self):
        """Test successfully adding resource to inventory."""
        unit = ResourceUnit(ResourceType.FOOD, quantity=10.0)
        
        result = self.inventory.add_resource(unit)
        
        assert result is True
        assert ResourceType.FOOD in self.inventory.resources
        assert len(self.inventory.resources[ResourceType.FOOD]) == 1
        assert self.inventory.resources[ResourceType.FOOD][0] == unit
    
    def test_add_resource_over_capacity(self):
        """Test adding resource that exceeds capacity."""
        unit = ResourceUnit(ResourceType.FOOD, quantity=150.0)
        
        result = self.inventory.add_resource(unit)
        
        assert result is False
        assert ResourceType.FOOD not in self.inventory.resources
    
    def test_add_multiple_resources(self):
        """Test adding multiple resources of same type."""
        unit1 = ResourceUnit(ResourceType.FOOD, quantity=10.0, quality=0.9)
        unit2 = ResourceUnit(ResourceType.FOOD, quantity=20.0, quality=0.8)
        
        assert self.inventory.add_resource(unit1) is True
        assert self.inventory.add_resource(unit2) is True
        
        assert len(self.inventory.resources[ResourceType.FOOD]) == 2
        assert self.inventory.get_total_amount(ResourceType.FOOD) == 30.0
    
    def test_remove_resource_exact_amount(self):
        """Test removing exact amount of resource."""
        unit = ResourceUnit(ResourceType.WATER, quantity=20.0)
        self.inventory.add_resource(unit)
        
        removed = self.inventory.remove_resource(ResourceType.WATER, 20.0)
        
        assert len(removed) == 1
        assert removed[0].quantity == 20.0
        assert self.inventory.get_total_amount(ResourceType.WATER) == 0.0
    
    def test_remove_resource_partial_amount(self):
        """Test removing partial amount from unit."""
        unit = ResourceUnit(ResourceType.ENERGY, quantity=50.0, quality=0.9)
        self.inventory.add_resource(unit)
        
        removed = self.inventory.remove_resource(ResourceType.ENERGY, 30.0)
        
        assert len(removed) == 1
        assert removed[0].quantity == 30.0
        assert removed[0].quality == 0.9  # Quality preserved
        assert self.inventory.get_total_amount(ResourceType.ENERGY) == 20.0
    
    def test_remove_resource_multiple_units(self):
        """Test removing from multiple units sorted by quality."""
        # Add units with different qualities
        unit1 = ResourceUnit(ResourceType.TOOLS, quantity=10.0, quality=0.5)
        unit2 = ResourceUnit(ResourceType.TOOLS, quantity=10.0, quality=0.9)
        unit3 = ResourceUnit(ResourceType.TOOLS, quantity=10.0, quality=0.7)
        
        self.inventory.add_resource(unit1)
        self.inventory.add_resource(unit2)
        self.inventory.add_resource(unit3)
        
        # Remove 15 units - should take from highest quality first
        removed = self.inventory.remove_resource(ResourceType.TOOLS, 15.0)
        
        # Should have removed all of unit2 (quality 0.9) and 5 from unit3 (quality 0.7)
        assert len(removed) == 2
        assert removed[0].quality == 0.9
        assert removed[0].quantity == 10.0
        assert removed[1].quality == 0.7
        assert removed[1].quantity == 5.0
        
        # Should have 15 units left (5 from unit3 and 10 from unit1)
        assert self.inventory.get_total_amount(ResourceType.TOOLS) == 15.0
    
    def test_remove_resource_not_found(self):
        """Test removing resource type not in inventory."""
        removed = self.inventory.remove_resource(ResourceType.CURRENCY, 10.0)
        
        assert removed == []
    
    def test_get_total_amount_empty(self):
        """Test getting amount of resource not in inventory."""
        amount = self.inventory.get_total_amount(ResourceType.REPUTATION)
        assert amount == 0.0
    
    def test_get_total_volume(self):
        """Test calculating total volume across all resources."""
        self.inventory.add_resource(ResourceUnit(ResourceType.FOOD, 20.0))
        self.inventory.add_resource(ResourceUnit(ResourceType.WATER, 30.0))
        self.inventory.add_resource(ResourceUnit(ResourceType.ENERGY, 15.0))
        
        total_volume = self.inventory.get_total_volume()
        assert total_volume == 65.0
    
    def test_get_available_amount_with_reserved(self):
        """Test getting available amount considering reserved resources."""
        self.inventory.add_resource(ResourceUnit(ResourceType.MATERIALS, 50.0))
        self.inventory.reserved[ResourceType.MATERIALS] = 20.0
        
        available = self.inventory.get_available_amount(ResourceType.MATERIALS)
        assert available == 30.0
    
    def test_reserve_resource_success(self):
        """Test successfully reserving resources."""
        self.inventory.add_resource(ResourceUnit(ResourceType.TOOLS, 20.0))
        
        result = self.inventory.reserve_resource(ResourceType.TOOLS, 15.0)
        
        assert result is True
        assert self.inventory.reserved[ResourceType.TOOLS] == 15.0
        assert self.inventory.get_available_amount(ResourceType.TOOLS) == 5.0
    
    def test_reserve_resource_insufficient(self):
        """Test reserving more than available."""
        self.inventory.add_resource(ResourceUnit(ResourceType.TOOLS, 10.0))
        
        result = self.inventory.reserve_resource(ResourceType.TOOLS, 15.0)
        
        assert result is False
        assert ResourceType.TOOLS not in self.inventory.reserved
    
    def test_release_reservation(self):
        """Test releasing reserved resources."""
        self.inventory.add_resource(ResourceUnit(ResourceType.WEAPONS, 30.0))
        self.inventory.reserve_resource(ResourceType.WEAPONS, 20.0)
        
        self.inventory.release_reservation(ResourceType.WEAPONS, 10.0)
        
        assert self.inventory.reserved[ResourceType.WEAPONS] == 10.0
        assert self.inventory.get_available_amount(ResourceType.WEAPONS) == 20.0
    
    def test_release_all_reservation(self):
        """Test releasing all reserved resources."""
        self.inventory.add_resource(ResourceUnit(ResourceType.SHELTER, 10.0))
        self.inventory.reserve_resource(ResourceType.SHELTER, 10.0)
        
        self.inventory.release_reservation(ResourceType.SHELTER, 10.0)
        
        assert self.inventory.reserved[ResourceType.SHELTER] == 0.0
        assert self.inventory.get_available_amount(ResourceType.SHELTER) == 10.0
    
    def test_degrade_resources(self):
        """Test degrading all resources in inventory."""
        # Add various resources
        tool_unit = ResourceUnit(ResourceType.TOOLS, 10.0, durability=1.0)
        food_unit = ResourceUnit(ResourceType.FOOD, 20.0, quality=1.0)
        
        self.inventory.add_resource(tool_unit)
        self.inventory.add_resource(food_unit)
        
        # Degrade inventory
        self.inventory.degrade_resources()
        
        # Check degradation applied
        assert self.inventory.resources[ResourceType.TOOLS][0].durability == 0.99
        assert self.inventory.resources[ResourceType.FOOD][0].quality == 0.999
    
    def test_get_inventory_summary(self):
        """Test getting inventory summary."""
        self.inventory.add_resource(ResourceUnit(ResourceType.FOOD, 10.0, quality=0.9))
        self.inventory.add_resource(ResourceUnit(ResourceType.FOOD, 5.0, quality=0.8))
        self.inventory.add_resource(ResourceUnit(ResourceType.WATER, 20.0))
        
        summary = self.inventory.get_inventory_summary()
        
        assert summary[ResourceType.FOOD]["quantity"] == 15.0
        assert summary[ResourceType.FOOD]["unit_count"] == 2
        assert summary[ResourceType.FOOD]["avg_quality"] == pytest.approx(0.867, rel=0.01)
        assert summary[ResourceType.WATER]["quantity"] == 20.0
        assert summary[ResourceType.WATER]["unit_count"] == 1
        assert summary[ResourceType.WATER]["avg_quality"] == 1.0


class TestResourceProvider:
    """Test IResourceProvider interface implementation."""
    
    def test_interface_methods(self):
        """Test that interface methods must be implemented."""
        # Create a mock implementation
        class MockProvider(IResourceProvider):
            def get_available_resources(self):
                return {ResourceType.FOOD: 100.0}
            
            def can_provide(self, resource_type, amount):
                return resource_type == ResourceType.FOOD and amount <= 100.0
            
            def provide_resource(self, resource_type, amount):
                return self.can_provide(resource_type, amount)
        
        provider = MockProvider()
        
        # Test interface methods
        resources = provider.get_available_resources()
        assert resources == {ResourceType.FOOD: 100.0}
        
        assert provider.can_provide(ResourceType.FOOD, 50.0) is True
        assert provider.can_provide(ResourceType.FOOD, 150.0) is False
        assert provider.can_provide(ResourceType.WATER, 10.0) is False
        
        assert provider.provide_resource(ResourceType.FOOD, 50.0) is True


class TestResourceConsumer:
    """Test IResourceConsumer interface implementation."""
    
    def test_interface_methods(self):
        """Test that interface methods must be implemented."""
        # Create a mock implementation
        class MockConsumer(IResourceConsumer):
            def get_resource_needs(self):
                return {ResourceType.ENERGY: 50.0, ResourceType.WATER: 30.0}
            
            def can_consume(self, resource_type, amount):
                needs = self.get_resource_needs()
                return resource_type in needs and amount <= needs[resource_type]
            
            def consume_resource(self, resource_type, amount):
                return self.can_consume(resource_type, amount)
        
        consumer = MockConsumer()
        
        # Test interface methods
        needs = consumer.get_resource_needs()
        assert needs == {ResourceType.ENERGY: 50.0, ResourceType.WATER: 30.0}
        
        assert consumer.can_consume(ResourceType.ENERGY, 30.0) is True
        assert consumer.can_consume(ResourceType.ENERGY, 60.0) is False
        assert consumer.can_consume(ResourceType.FOOD, 10.0) is False
        
        assert consumer.consume_resource(ResourceType.WATER, 20.0) is True


