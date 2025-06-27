"""
Module for FreeAgentics Active Inference implementation.
"""

import importlib.util
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

"""
Simple test for Resource Business Model
Tests the resource business model by importing the module directly to avoid
PyTorch and other dependency issues.
"""
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_root)
# Import the resource business model module directly
spec = importlib.util.spec_from_file_location(
    "resource_business_model",
    os.path.join(project_root, "agents", "base", "resource_business_model.py"),
)
resource_business_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_business_model)
# Extract the classes we need
ResourceType = resource_business_model.ResourceType
TransactionType = resource_business_model.TransactionType
ResourceUnit = resource_business_model.ResourceUnit
MarketPrice = resource_business_model.MarketPrice
TradeOffer = resource_business_model.TradeOffer
ResourceInventory = resource_business_model.ResourceInventory
Market = resource_business_model.Market
ResourceBusinessManager = resource_business_model.ResourceBusinessManager


class TestResourceBusinessModel(unittest.TestCase):
    ."""Test the resource and business model mechanics."""

    def test_resource_unit_creation(self) -> None:
        """Test creating resource units"""
        unit = ResourceUnit(
            resource_type=ResourceType.WATER,
            quantity=100.0,
            quality=0.9,
            durability=1.0,
            origin="agent1",
        )
        self.assertEqual(unit.resource_type, ResourceType.WATER)
        self.assertEqual(unit.quantity, 100.0)
        self.assertEqual(unit.quality, 0.9)
        self.assertEqual(unit.get_effective_value(), 90.0)  # 100 * 0.9 * 1.0

    def test_resource_inventory(self) -> None:
        """Test resource inventory management"""
        inventory = ResourceInventory("agent1", initial_capacity=500.0)
        # Add resources
        water = ResourceUnit(ResourceType.WATER, quantity=50.0)
        food = ResourceUnit(ResourceType.FOOD, quantity=30.0)
        self.assertTrue(inventory.add_resource(water))
        self.assertTrue(inventory.add_resource(food))
        # Check amounts
        self.assertEqual(inventory.get_total_amount(ResourceType.WATER), 50.0)
        self.assertEqual(inventory.get_total_amount(ResourceType.FOOD), 30.0)
        self.assertEqual(inventory.get_total_volume(), 80.0)
        # Remove resources
        removed = inventory.remove_resource(ResourceType.WATER, 20.0)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].quantity, 20.0)
        self.assertEqual(inventory.get_total_amount(ResourceType.WATER), 30.0)

    def test_market_price_dynamics(self) -> None:
        """Test market price updates based on supply and demand"""
        price = MarketPrice(resource_type=ResourceType.FOOD, base_price=2.0, current_price=2.0)
        # High demand, low supply should increase price
        price.update_price(supply=10.0, demand=50.0)
        self.assertGreater(price.current_price, 2.0)
        # Save new price for comparison
        high_price = price.current_price
        # Low demand, high supply should decrease price
        price.update_price(supply=100.0, demand=10.0)
        self.assertLess(price.current_price, high_price)

    def test_trade_offer_validation(self) -> None:
        """Test trade offer creation and validation"""
        offer = TradeOffer(
            offer_id="offer1",
            from_agent="agent1",
            to_agent=None,  # Public offer
            offered_resources={ResourceType.WATER: 50.0},
            requested_resources={ResourceType.FOOD: 25.0},
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        self.assertTrue(offer.is_valid())
        # Test expired offer
        expired_offer = TradeOffer(
            offer_id="offer2",
            from_agent="agent1",
            to_agent=None,
            offered_resources={ResourceType.WATER: 50.0},
            requested_resources={ResourceType.FOOD: 25.0},
            expiration=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        self.assertFalse(expired_offer.is_valid())

    def test_market_operations(self) -> None:
        """Test basic market operations"""
        market = Market("market1", location="city_center")
        # Test initial prices
        self.assertIn(ResourceType.WATER, market.prices)
        self.assertEqual(market.prices[ResourceType.WATER].base_price, 1.5)
        # Submit an offer
        offer = TradeOffer(
            offer_id="offer1",
            from_agent="agent1",
            to_agent=None,
            offered_resources={ResourceType.WATER: 50.0},
            requested_resources={ResourceType.FOOD: 25.0},
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        self.assertTrue(market.submit_offer(offer))
        self.assertIn("offer1", market.active_offers)
        # Test finding matching offers
        matches = market.find_matching_offers(
            agent_id="agent2", wanted={ResourceType.WATER: 40.0}, offered={ResourceType.FOOD: 30.0}
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].offer_id, "offer1")

    def test_resource_business_manager(self) -> None:
        """Test the resource business manager"""
        manager = ResourceBusinessManager()
        # Create market and inventories
        market = manager.create_market("main_market")
        self.assertIsNotNone(market)
        inv1 = manager.create_inventory("agent1", capacity=1000.0)
        inv2 = manager.create_inventory("agent2", capacity=1000.0)
        self.assertIsNotNone(inv1)
        self.assertIsNotNone(inv2)
        # Test trust relationships (default trust is 0.5)
        trust = manager.get_trust_level("agent1", "agent2")
        self.assertEqual(trust, 0.5)  # Default trust level
        # Test updating trust
        manager.update_trust_relationship("agent1", "agent2", 0.3)
        trust = manager.get_trust_level("agent1", "agent2")
        self.assertEqual(trust, 0.8)  # 0.5 + 0.3
        # Test negative trust change
        manager.update_trust_relationship("agent1", "agent2", -0.2)
        trust = manager.get_trust_level("agent1", "agent2")
        self.assertAlmostEqual(
            trust, 0.6, places=5
        )  # 0.8 - 0.2 (using assertAlmostEqual for floating point)

    def test_resource_degradation(self) -> None:
        """Test resource degradation over time"""
        unit = ResourceUnit(
            resource_type=ResourceType.TOOLS, quantity=1.0, quality=1.0, durability=1.0
        )
        # Tools should degrade in durability
        unit.degrade(rate=0.1)
        self.assertEqual(unit.durability, 0.9)
        self.assertEqual(unit.quality, 1.0)  # Quality unchanged for tools
        # Food should degrade in quality
        food = ResourceUnit(
            resource_type=ResourceType.FOOD, quantity=10.0, quality=1.0, durability=1.0
        )
        food.degrade(rate=0.1)
        self.assertEqual(food.quality, 0.99)  # Quality degrades slower
        self.assertEqual(food.durability, 1.0)  # Durability unchanged for food


if __name__ == "__main__":
    unittest.main()
