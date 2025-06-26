"""Basic Agent System for FreeAgentics"""

from typing import List

from .agent import BaseAgent, create_agent
from .agent_factory import get_default_factory
from .agent_template import (
    ActiveInferenceConfig,
    AgentTemplateFactory,
    BaseAgentTemplate,
    ExplorerTemplate,
    GuardianTemplate,
    IAgentTemplate,
    MerchantTemplate,
    ScholarTemplate,
    TemplateMetadata,
    TemplateType,
    create_explorer_agent,
    create_guardian_agent,
    create_merchant_agent,
    create_scholar_agent,
)
from .data_model import Agent, AgentCapability, AgentStatus, Position

__version__ = "0.1.0"
__all__: List[str] = [
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
]
