"""
FreeAgentics Agents Package

This package contains the agent system implementation including
basic agents, active inference integration, and testing framework.
"""

from typing import List

# Re-export all public APIs from submodules
from agents.base import (
    ActiveInferenceConfig,
    Agent,
    AgentCapability,
    AgentStatus,
    AgentTemplateFactory,
    BaseAgent,
    BaseAgentTemplate,
    ExplorerTemplate,
    GuardianTemplate,
    IAgentTemplate,
    MerchantTemplate,
    Position,
    ScholarTemplate,
    TemplateMetadata,
    TemplateType,
    create_agent,
    create_explorer_agent,
    create_guardian_agent,
    create_merchant_agent,
    create_scholar_agent,
    get_default_factory,
)
from agents.testing import (
    AgentFactory,
    AgentTestMetrics,
    AgentTestOrchestrator,
    AgentTestScenario,
    BehaviorValidator,
    PerformanceBenchmark,
    SimulationEnvironment,
    TestMetrics,
    TestOrchestrator,
    TestScenario,
    create_basic_test_scenarios,
)

__all__: List[str] = [
    # Core Agent Components
    "BaseAgent",
    "create_agent",
    "Agent",
    "AgentCapability",
    "Position",
    "AgentStatus",
    "get_default_factory",
    # Agent Template System
    "AgentTemplateFactory",
    "TemplateType",
    "IAgentTemplate",
    "BaseAgentTemplate",
    "ActiveInferenceConfig",
    "TemplateMetadata",
    "ExplorerTemplate",
    "GuardianTemplate",
    "MerchantTemplate",
    "ScholarTemplate",
    "create_explorer_agent",
    "create_guardian_agent",
    "create_merchant_agent",
    "create_scholar_agent",
    # Testing Framework
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
