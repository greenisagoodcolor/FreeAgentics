"""
Merchant Agent Module for FreeAgentics

This module provides the Merchant agent type and related trading functionality,
including market systems, trade offers, and specialized trading behaviors.
"""

from .merchant import (  # Main agent class, enums, behaviors, and factory functions
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

__all__ = [
    # Main classes
    "MerchantAgent",
    "Market",
    "TradeOffer",
    # Enums
    "TradeStatus",
    "ResourceType",
    # Behaviors
    "TradingBehavior",
    "MarketAnalysisBehavior",
    # Factory functions
    "create_merchant_agent",
    "register_merchant_type",
]

# Version information
__version__ = "0.1.0"
