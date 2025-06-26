"""
FreeAgentics Agents Package

This package contains the agent system implementation including
basic agents, active inference integration, and testing framework.
"""

from typing import List

# Re-export all public APIs from submodules
from agents.base import (
    BaseAgent,
    create_agent,
    Agent,
    AgentCapability,
    Position,
    AgentStatus,
    get_default_factory,
    AgentTemplateFactory,
    TemplateType,
    IAgentTemplate,
    BaseAgentTemplate,
    ActiveInferenceConfig,
    TemplateMetadata,
    ExplorerTemplate,
    GuardianTemplate,
    MerchantTemplate,
    ScholarTemplate,
    create_explorer_agent,
    create_guardian_agent,
    create_merchant_agent,
    create_scholar_agent,
)
from agents.testing import (
    AgentTestScenario,
    AgentTestMetrics,
    AgentFactory,
    SimulationEnvironment,
    BehaviorValidator,
    PerformanceBenchmark,
    AgentTestOrchestrator,
    create_basic_test_scenarios,
    TestScenario,
    TestMetrics,
    TestOrchestrator,
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
