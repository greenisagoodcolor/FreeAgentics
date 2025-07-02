"""
Coalition Formation Monitoring Integration

This module provides the integration layer between coalition formation algorithms
and the real-time monitoring system, ensuring ADR-002 and ADR-006 compliance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .coalition_formation_algorithms import (
    CoalitionFormationEngine,
    FormationStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class CoalitionMonitoringEvent:
    """Standardized event for coalition formation monitoring"""

    event_type: str
    coalition_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_used: Optional[FormationStrategy] = None
    participants: List[str] = field(default_factory=list)
    business_value: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoalitionFormationMonitor:
    """Monitors coalition formation processes and provides real-time events"""

    def __init__(
            self,
            formation_engine: Optional[CoalitionFormationEngine] = None) -> None:
        self.formation_engine = formation_engine or CoalitionFormationEngine()
        self.event_handlers: List[Callable[[
            CoalitionMonitoringEvent], None]] = []
        self.active_formations: Dict[str, Dict[str, Any]] = {}

    def register_event_handler(
            self, handler: Callable[[CoalitionMonitoringEvent], None]) -> None:
        """Register a handler for coalition formation events"""
        self.event_handlers.append(handler)
        logger.info("Registered coalition formation event handler")

    def _emit_event(self, event: CoalitionMonitoringEvent):
        """Emit an event to all registered handlers"""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")


def create_coalition_monitoring_system():
    """Factory function to create complete coalition monitoring system"""
    monitor = CoalitionFormationMonitor()
    return monitor
