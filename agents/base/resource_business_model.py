"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

"""
Resource and Business Model Mechanics
This module provides comprehensive resource management, economic transactions,
and business logic for agent interactions including markets, trade protocols,
and resource production/consumption systems.
"""
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources in the system"""

    # Basic resources
    ENERGY = "energy"
    FOOD = "food"
    WATER = "water"
    MATERIALS = "materials"
    # Crafted resources
    TOOLS = "tools"
    SHELTER = "shelter"
    WEAPONS = "weapons"
    TECHNOLOGY = "technology"
    # Social resources
    INFORMATION = "information"
    INFLUENCE = "influence"
    REPUTATION = "reputation"
    RELATIONSHIPS = "relationships"
    # Economic resources
    CURRENCY = "currency"
    CREDIT = "credit"
    CONTRACTS = "contracts"
    SERVICES = "services"


class TransactionType(Enum):
    """Types of economic transactions"""

    TRADE = "trade"
    PURCHASE = "purchase"
    SALE = "sale"
    BARTER = "barter"
    LOAN = "loan"
    INVESTMENT = "investment"
    GIFT = "gift"
    THEFT = "theft"
    TRIBUTE = "tribute"
    TAX = "tax"


class MarketCondition(Enum):
    """Market condition states"""

    BULL = "bull"  # Rising prices
    BEAR = "bear"  # Falling prices
    STABLE = "stable"  # Stable prices
    VOLATILE = "volatile"  # Highly fluctuating
    CRASHED = "crashed"  # Severely depressed


@dataclass
class ResourceUnit:
    """Represents a unit of resource with quality and metadata"""

    resource_type: ResourceType
    quantity: float
    quality: float = 1.0  # 0.0 to 1.0
    durability: float = 1.0  # 0.0 to 1.0 (for tools, weapons, etc.)
    origin: Optional[str] = None  # Agent or location that produced it
    age: float = 0.0  # Age in simulation time units
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_value(self) -> float:
        """Calculate effective value considering quality and durability"""
        return self.quantity * self.quality * self.durability

    def degrade(self, rate: float = 0.01) -> None:
        """Degrade resource over time"""
        if self.resource_type in [ResourceType.TOOLS, ResourceType.WEAPONS, ResourceType.SHELTER]:
            self.durability = max(0.0, self.durability - rate)
        else:
            self.quality = max(0.0, self.quality - rate * 0.1)
        self.age += 1.0


@dataclass
class MarketPrice:
    """Market price information for a resource"""

    resource_type: ResourceType
    base_price: float
    current_price: float
    price_history: List[tuple[datetime, float]] = field(default_factory=list)
    supply: float = 0.0
    demand: float = 0.0
    volatility: float = 0.1
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_price(self, supply: float, demand: float) -> None:
        """Update price based on supply and demand"""
        old_price = self.current_price
        # Basic supply and demand pricing
        if supply > 0:
            price_factor = demand / supply
        else:
            price_factor = 2.0  # High price when no supply
        # Apply elasticity and bounds
        new_price = self.base_price * price_factor
        # Add some inertia to prevent wild swings
        self.current_price = (self.current_price * 0.7) + (new_price * 0.3)
        # Update history
        self.price_history.append((datetime.now(timezone.utc), self.current_price))
        if len(self.price_history) > 100:  # Keep last 100 entries
            self.price_history.pop(0)
        # Update volatility
        if len(self.price_history) > 1:
            price_change = abs(self.current_price - old_price) / old_price
            self.volatility = (self.volatility * 0.9) + (price_change * 0.1)
        self.supply = supply
        self.demand = demand
        self.last_updated = datetime.now(timezone.utc)

    def get_trend(self) -> str:
        """Get price trend over recent history"""
        if len(self.price_history) < 5:
            return "insufficient_data"
        recent_prices = [p[1] for p in self.price_history[-5:]]
        trend = recent_prices[-1] - recent_prices[0]
        if abs(trend) < self.current_price * 0.05:
            return "stable"
        elif trend > 0:
            return "rising"
        else:
            return "falling"


@dataclass
class TradeOffer:
    """Represents a trade offer between agents"""

    offer_id: str
    from_agent: str
    to_agent: Optional[str]  # None for public offers
    offered_resources: Dict[ResourceType, float]
    requested_resources: Dict[ResourceType, float]
    expiration: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_public: bool = True
    minimum_reputation: float = 0.0
    max_distance: Optional[float] = None
    additional_terms: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if offer is still valid"""
        return datetime.now(timezone.utc) < self.expiration

    def calculate_value_ratio(self, market_prices: Dict[ResourceType, MarketPrice]) -> float:
        """Calculate the value ratio of offered vs requested resources"""
        offered_value = sum(
            market_prices.get(rt, MarketPrice(rt, 1.0, 1.0)).current_price * amount
            for rt, amount in self.offered_resources.items()
        )
        requested_value = sum(
            market_prices.get(rt, MarketPrice(rt, 1.0, 1.0)).current_price * amount
            for rt, amount in self.requested_resources.items()
        )
        if requested_value == 0:
            return float("inf")
        return offered_value / requested_value


@dataclass
class Transaction:
    """Completed transaction record"""

    transaction_id: str
    transaction_type: TransactionType
    parties: List[str]  # Agent IDs involved
    resources_exchanged: Dict[str, Dict[ResourceType, float]]  # agent_id -> resources
    total_value: float
    fee: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IResourceProvider(ABC):
    """Interface for entities that can provide resources"""

    @abstractmethod
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get currently available resources"""
        pass

    @abstractmethod
    def can_provide(self, resource_type: ResourceType, amount: float) -> bool:
        """Check if can provide specified amount of resource"""
        pass

    @abstractmethod
    def provide_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Provide specified amount of resource"""
        pass


class IResourceConsumer(ABC):
    """Interface for entities that can consume resources"""

    @abstractmethod
    def get_resource_needs(self) -> Dict[ResourceType, float]:
        """Get current resource needs"""
        pass

    @abstractmethod
    def can_consume(self, resource_type: ResourceType, amount: float) -> bool:
        """Check if can consume specified amount of resource"""
        pass

    @abstractmethod
    def consume_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Consume specified amount of resource"""
        pass


class ResourceInventory:
    """Manages an agent's resource inventory"""

    def __init__(self, agent_id: str, initial_capacity: float = 1000.0) -> None:
        self.agent_id = agent_id
        self.capacity = initial_capacity
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {}
        self.reserved: Dict[ResourceType, float] = {}  # Resources reserved for pending trades

    def add_resource(self, resource_unit: ResourceUnit) -> bool:
        """Add a resource unit to inventory"""
        if self.get_total_volume() + resource_unit.quantity > self.capacity:
            return False
        if resource_unit.resource_type not in self.resources:
            self.resources[resource_unit.resource_type] = []
        self.resources[resource_unit.resource_type].append(resource_unit)
        logger.debug(
            f"Added {resource_unit.quantity} {resource_unit.resource_type.value} to {self.agent_id}"
        )
        return True

    def remove_resource(self, resource_type: ResourceType, amount: float) -> List[ResourceUnit]:
        """Remove specified amount of resource, returns removed units"""
        if resource_type not in self.resources:
            return []
        removed_units = []
        remaining_amount = amount
        units_to_remove = []
        # Sort by quality (use best quality first)
        sorted_units = sorted(
            self.resources[resource_type], key=lambda x: x.get_effective_value(), reverse=True
        )
        for unit in sorted_units:
            if remaining_amount <= 0:
                break
            if unit.quantity <= remaining_amount:
                # Take the whole unit
                removed_units.append(unit)
                units_to_remove.append(unit)
                remaining_amount -= unit.quantity
            else:
                # Take partial unit
                new_unit = ResourceUnit(
                    resource_type=unit.resource_type,
                    quantity=remaining_amount,
                    quality=unit.quality,
                    durability=unit.durability,
                    origin=unit.origin,
                    age=unit.age,
                    metadata=unit.metadata.copy(),
                )
                removed_units.append(new_unit)
                unit.quantity -= remaining_amount
                remaining_amount = 0
        # Remove used units
        for unit in units_to_remove:
            self.resources[resource_type].remove(unit)
        return removed_units

    def get_total_amount(self, resource_type: ResourceType) -> float:
        """Get total amount of a resource type"""
        if resource_type not in self.resources:
            return 0.0
        return sum(unit.quantity for unit in self.resources[resource_type])

    def get_available_amount(self, resource_type: ResourceType) -> float:
        """Get available amount (total minus reserved)"""
        total = self.get_total_amount(resource_type)
        reserved = self.reserved.get(resource_type, 0.0)
        return max(0.0, total - reserved)

    def reserve_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Reserve resources for pending transactions"""
        if self.get_available_amount(resource_type) < amount:
            return False
        self.reserved[resource_type] = self.reserved.get(resource_type, 0.0) + amount
        return True

    def release_reservation(self, resource_type: ResourceType, amount: float) -> None:
        """Release reserved resources"""
        current_reserved = self.reserved.get(resource_type, 0.0)
        self.reserved[resource_type] = max(0.0, current_reserved - amount)

    def get_total_volume(self) -> float:
        """Get total volume of all resources"""
        total = 0.0
        for units in self.resources.values():
            total += sum(unit.quantity for unit in units)
        return total

    def degrade_resources(self) -> None:
        """Degrade all resources over time"""
        for resource_type, units in self.resources.items():
            for unit in units:
                unit.degrade()
            # Remove completely degraded items
            self.resources[resource_type] = [
                unit for unit in units if unit.durability > 0.0 and unit.quality > 0.0
            ]

    def get_inventory_summary(self) -> Dict[ResourceType, Dict[str, float]]:
        """Get summary of inventory"""
        summary = {}
        for resource_type, units in self.resources.items():
            total_quantity = sum(unit.quantity for unit in units)
            avg_quality = (
                sum(unit.quality * unit.quantity for unit in units) / total_quantity
                if total_quantity > 0
                else 0
            )
            avg_durability = (
                sum(unit.durability * unit.quantity for unit in units) / total_quantity
                if total_quantity > 0
                else 0
            )
            summary[resource_type] = {
                "quantity": total_quantity,
                "available": self.get_available_amount(resource_type),
                "reserved": self.reserved.get(resource_type, 0.0),
                "avg_quality": avg_quality,
                "avg_durability": avg_durability,
                "unit_count": len(units),
            }
        return summary


class Market:
    """Central market for resource trading"""

    def __init__(self, market_id: str, location: Optional[str] = None) -> None:
        self.market_id = market_id
        self.location = location
        self.prices: Dict[ResourceType, MarketPrice] = {}
        self.active_offers: Dict[str, TradeOffer] = {}
        self.transaction_history: List[Transaction] = []
        self.market_makers: set[str] = set()  # Agents that provide liquidity
        self.transaction_fee_rate = 0.05  # 5% transaction fee
        # Initialize base prices
        self._initialize_base_prices()

    def _initialize_base_prices(self) -> None:
        """Initialize base market prices for all resource types"""
        base_prices = {
            ResourceType.ENERGY: 1.0,
            ResourceType.FOOD: 2.0,
            ResourceType.WATER: 1.5,
            ResourceType.MATERIALS: 3.0,
            ResourceType.TOOLS: 10.0,
            ResourceType.SHELTER: 50.0,
            ResourceType.WEAPONS: 15.0,
            ResourceType.TECHNOLOGY: 100.0,
            ResourceType.INFORMATION: 5.0,
            ResourceType.INFLUENCE: 20.0,
            ResourceType.REPUTATION: 25.0,
            ResourceType.RELATIONSHIPS: 30.0,
            ResourceType.CURRENCY: 1.0,
            ResourceType.CREDIT: 0.8,
            ResourceType.CONTRACTS: 40.0,
            ResourceType.SERVICES: 8.0,
        }
        for resource_type, base_price in base_prices.items():
            self.prices[resource_type] = MarketPrice(
                resource_type=resource_type, base_price=base_price, current_price=base_price
            )

    def submit_offer(self, offer: TradeOffer) -> bool:
        """Submit a trade offer to the market"""
        if not offer.is_valid():
            return False
        self.active_offers[offer.offer_id] = offer
        logger.debug(f"Trade offer {offer.offer_id} submitted to market {self.market_id}")
        return True

    def cancel_offer(self, offer_id: str, agent_id: str) -> bool:
        """Cancel a trade offer"""
        if offer_id not in self.active_offers:
            return False
        offer = self.active_offers[offer_id]
        if offer.from_agent != agent_id:
            return False
        del self.active_offers[offer_id]
        return True

    def find_matching_offers(
        self, agent_id: str, wanted: Dict[ResourceType, float], offered: Dict[ResourceType, float]
    ) -> List[TradeOffer]:
        """Find offers that match agent's requirements"""
        matching_offers = []
        for offer in self.active_offers.values():
            if not offer.is_valid() or offer.from_agent == agent_id:
                continue
            if offer.to_agent and offer.to_agent != agent_id:
                continue
            # Check if this offer provides what we want
            can_satisfy = True
            for resource_type, amount in wanted.items():
                if offer.offered_resources.get(resource_type, 0) < amount:
                    can_satisfy = False
                    break
            if not can_satisfy:
                continue
            # Check if we can provide what they want
            can_provide = True
            for resource_type, amount in offer.requested_resources.items():
                if offered.get(resource_type, 0) < amount:
                    can_provide = False
                    break
            if can_provide:
                matching_offers.append(offer)
        # Sort by value ratio (best deals first)
        matching_offers.sort(key=lambda x: x.calculate_value_ratio(self.prices), reverse=True)
        return matching_offers

    def execute_trade(self, offer_id: str, buyer_id: str) -> Optional[Transaction]:
        """Execute a trade between agents"""
        if offer_id not in self.active_offers:
            return None
        offer = self.active_offers[offer_id]
        if not offer.is_valid():
            del self.active_offers[offer_id]
            return None
        # Calculate transaction value
        total_value = sum(
            self.prices.get(rt, MarketPrice(rt, 1.0, 1.0)).current_price * amount
            for rt, amount in offer.offered_resources.items()
        )
        # Calculate fee
        fee = total_value * self.transaction_fee_rate
        # Create transaction record
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type=TransactionType.TRADE,
            parties=[offer.from_agent, buyer_id],
            resources_exchanged={
                offer.from_agent: {rt: -amount for rt, amount in offer.offered_resources.items()},
                buyer_id: {rt: amount for rt, amount in offer.offered_resources.items()},
            },
            total_value=total_value,
            fee=fee,
            location=self.location,
        )
        self.transaction_history.append(transaction)
        del self.active_offers[offer_id]
        # Update market prices based on transaction
        self._update_prices_from_transaction(transaction)
        logger.info(f"Trade executed: {transaction.transaction_id}")
        return transaction

    def _update_prices_from_transaction(self, transaction: Transaction) -> None:
        """Update market prices based on completed transaction"""
        for agent_id, resources in transaction.resources_exchanged.items():
            for resource_type, amount in resources.items():
                if resource_type not in self.prices:
                    continue
                price_info = self.prices[resource_type]
                if amount > 0:  # Buying (increasing demand)
                    price_info.demand += amount
                else:  # Selling (increasing supply)
                    price_info.supply += abs(amount)
                # Update price
                price_info.update_price(price_info.supply, price_info.demand)

    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of market state"""
        return {
            "market_id": self.market_id,
            "location": self.location,
            "active_offers": len(self.active_offers),
            "recent_transactions": len(
                [
                    t
                    for t in self.transaction_history
                    if (datetime.now(timezone.utc) - t.timestamp).days < 1
                ]
            ),
            "price_summary": {
                rt.value: {
                    "current_price": price.current_price,
                    "trend": price.get_trend(),
                    "volatility": price.volatility,
                }
                for rt, price in self.prices.items()
            },
        }

    def clean_expired_offers(self) -> None:
        """Remove expired offers"""
        expired_offers = [
            offer_id for offer_id, offer in self.active_offers.items() if not offer.is_valid()
        ]
        for offer_id in expired_offers:
            del self.active_offers[offer_id]


class ResourceBusinessManager:
    """Main manager for resource and business operations"""

    def __init__(self) -> None:
        self.markets: Dict[str, Market] = {}
        self.inventories: Dict[str, ResourceInventory] = {}
        self.production_facilities: Dict[str, Any] = {}  # Agent-owned facilities
        self.trade_relationships: Dict[str, Dict[str, float]] = {}  # Trust levels between agents

    def create_market(self, market_id: str, location: Optional[str] = None) -> Market:
        """Create a new market"""
        market = Market(market_id, location)
        self.markets[market_id] = market
        return market

    def get_market(self, market_id: str) -> Optional[Market]:
        """Get market by ID"""
        return self.markets.get(market_id)

    def create_inventory(self, agent_id: str, capacity: float = 1000.0) -> ResourceInventory:
        """Create inventory for an agent"""
        inventory = ResourceInventory(agent_id, capacity)
        self.inventories[agent_id] = inventory
        return inventory

    def get_inventory(self, agent_id: str) -> Optional[ResourceInventory]:
        """Get agent's inventory"""
        return self.inventories.get(agent_id)

    def create_trade_offer(
        self,
        agent_id: str,
        offered: Dict[ResourceType, float],
        requested: Dict[ResourceType, float],
        duration_hours: float = 24.0,
        target_agent: Optional[str] = None,
    ) -> Optional[TradeOffer]:
        """Create a trade offer"""
        inventory = self.get_inventory(agent_id)
        if not inventory:
            return None
        # Check if agent has resources to offer
        for resource_type, amount in offered.items():
            if not inventory.reserve_resource(resource_type, amount):
                # Release any reservations made so far
                for rt, amt in offered.items():
                    if rt == resource_type:
                        break
                    inventory.release_reservation(rt, amt)
                return None
        offer = TradeOffer(
            offer_id=str(uuid.uuid4()),
            from_agent=agent_id,
            to_agent=target_agent,
            offered_resources=offered.copy(),
            requested_resources=requested.copy(),
            expiration=datetime.now(timezone.utc) + timedelta(hours=duration_hours),
            is_public=target_agent is None,
        )
        return offer

    def execute_transaction(
        self, market_id: str, offer_id: str, buyer_id: str
    ) -> Optional[Transaction]:
        """Execute a transaction in a specific market"""
        market = self.get_market(market_id)
        if not market:
            return None
        transaction = market.execute_trade(offer_id, buyer_id)
        if not transaction:
            return None
        # Transfer resources between inventories
        for agent_id, resources in transaction.resources_exchanged.items():
            inventory = self.get_inventory(agent_id)
            if not inventory:
                continue
            for resource_type, amount in resources.items():
                if amount > 0:  # Receiving resources
                    resource_unit = ResourceUnit(
                        resource_type=resource_type,
                        quantity=amount,
                        quality=1.0,
                        origin=f"trade_{transaction.transaction_id}",
                    )
                    inventory.add_resource(resource_unit)
                else:  # Giving resources
                    inventory.remove_resource(resource_type, abs(amount))
        return transaction

    def update_trust_relationship(self, agent1: str, agent2: str, change: float) -> None:
        """Update trust level between two agents"""
        if agent1 not in self.trade_relationships:
            self.trade_relationships[agent1] = {}
        if agent2 not in self.trade_relationships:
            self.trade_relationships[agent2] = {}
        current_trust = self.trade_relationships[agent1].get(agent2, 0.5)
        new_trust = max(0.0, min(1.0, current_trust + change))
        self.trade_relationships[agent1][agent2] = new_trust
        self.trade_relationships[agent2][agent1] = new_trust

    def get_trust_level(self, agent1: str, agent2: str) -> float:
        """Get trust level between two agents"""
        if agent1 in self.trade_relationships:
            return self.trade_relationships[agent1].get(agent2, 0.5)
        return 0.5

    def run_market_cycle(self) -> None:
        """Run one cycle of market operations"""
        for market in self.markets.values():
            market.clean_expired_offers()
            # Degrade resources in all inventories
            for inventory in self.inventories.values():
                inventory.degrade_resources()

    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the entire resource/business system"""
        return {
            "markets": {
                market_id: market.get_market_summary() for market_id, market in self.markets.items()
            },
            "total_agents": len(self.inventories),
            "total_trade_relationships": sum(
                len(relationships) for relationships in self.trade_relationships.values()
            ),
            "active_offers": sum(len(market.active_offers) for market in self.markets.values()),
            "recent_transactions": sum(
                len(
                    [
                        t
                        for t in market.transaction_history
                        if (datetime.now(timezone.utc) - t.timestamp).days < 1
                    ]
                )
                for market in self.markets.values()
            ),
        }


# Factory functions
def create_resource_business_manager() -> ResourceBusinessManager:
    """Create a new resource business manager"""
    return ResourceBusinessManager()


def create_basic_inventory(
    agent_id: str, starting_resources: Optional[Dict[ResourceType, float]] = None
) -> ResourceInventory:
    """Create a basic inventory with starting resources"""
    inventory = ResourceInventory(agent_id)
    if starting_resources:
        for resource_type, amount in starting_resources.items():
            resource_unit = ResourceUnit(
                resource_type=resource_type, quantity=amount, quality=1.0, origin="initial"
            )
            inventory.add_resource(resource_unit)
    return inventory
