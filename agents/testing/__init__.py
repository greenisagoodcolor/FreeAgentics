"""
Agent Testing Framework

This package provides comprehensive testing utilities for agent behaviors.
"""

from .agent_test_framework import (
    AgentFactory,
    AgentTestMetrics,
    AgentTestOrchestrator,
    AgentTestScenario,
    BehaviorValidator,
    PerformanceBenchmark,
    SimulationEnvironment,
    create_basic_test_scenarios,
)

# Maintain backwards compatibility with old names
TestScenario = AgentTestScenario
TestMetrics = AgentTestMetrics
TestOrchestrator = AgentTestOrchestrator

__all__ = [
    "AgentTestScenario",
    "AgentTestMetrics",
    "AgentFactory",
    "SimulationEnvironment",
    "BehaviorValidator",
    "PerformanceBenchmark",
    "AgentTestOrchestrator",
    "create_basic_test_scenarios",
    "TestScenario",
    "TestMetrics",
    "TestOrchestrator",
]
