"""WebSocket Load Testing Framework for FreeAgentics.

This module provides comprehensive tools for load testing WebSocket connections,
including client managers, message generators, metrics collection, and load scenarios.
"""

from .client_manager import WebSocketClient, WebSocketClientManager
from .connection_lifecycle import ConnectionLifecycleManager
from .load_scenarios import (
    BurstLoadScenario,
    LoadScenario,
    RampUpScenario,
    SteadyLoadScenario,
)
from .message_generators import (
    CommandMessageGenerator,
    EventMessageGenerator,
    MessageGenerator,
)
from .metrics_collector import MetricsCollector, WebSocketMetrics

__all__ = [
    "WebSocketClientManager",
    "WebSocketClient",
    "MessageGenerator",
    "EventMessageGenerator",
    "CommandMessageGenerator",
    "MetricsCollector",
    "WebSocketMetrics",
    "ConnectionLifecycleManager",
    "LoadScenario",
    "SteadyLoadScenario",
    "BurstLoadScenario",
    "RampUpScenario",
]
