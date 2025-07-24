"""Concurrent user simulation framework for FreeAgentics.

This package provides tools for simulating realistic concurrent user
behaviors to stress test the system with database and WebSocket operations.
"""

from .concurrent_simulator import ConcurrentSimulator, SimulationConfig, SimulationMetrics
from .scenarios import ScenarioScheduler, SimulationScenarios
from .user_personas import (
    ActivityLevel,
    AdminBehavior,
    CoordinatorBehavior,
    InteractionPattern,
    ObserverBehavior,
    PersonaProfile,
    PersonaType,
    ResearcherBehavior,
    UserBehavior,
    create_behavior,
    create_persona,
)

__all__ = [
    # Core simulator
    "ConcurrentSimulator",
    "SimulationConfig",
    "SimulationMetrics",
    # Scenarios
    "SimulationScenarios",
    "ScenarioScheduler",
    # Personas
    "PersonaType",
    "ActivityLevel",
    "InteractionPattern",
    "PersonaProfile",
    "UserBehavior",
    "ResearcherBehavior",
    "CoordinatorBehavior",
    "ObserverBehavior",
    "AdminBehavior",
    "create_persona",
    "create_behavior",
]

__version__ = "1.0.0"
