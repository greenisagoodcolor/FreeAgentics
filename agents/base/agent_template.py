"""
Agent Template System for FreeAgentics

This module provides a template system for rapid agent creation, enabling users to
instantiate agents from predefined configurations with Active Inference support.
Follows ADR-002, ADR-003, and ADR-005 architectural patterns.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type, cast

import numpy as np
from numpy.typing import NDArray

from .agent import BaseAgent
from .data_model import Agent as AgentData
from .data_model import Personality, Position, Resources


class TemplateType(Enum):
    """Predefined agent template types following Active Inference principles"""

    EXPLORER = "explorer"
    GUARDIAN = "guardian"
    MERCHANT = "merchant"
    SCHOLAR = "scholar"


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference parameters"""

    # Generative model dimensions
    num_states: int = 4
    num_observations: int = 4
    num_actions: int = 4

    # State labels for interpretability
    state_labels: List[str] = field(
        default_factory=lambda: ["idle", "exploring", "interacting", "planning"]
    )

    # Prior preferences (C matrix) - what the agent prefers
    prior_preferences: Optional[NDArray[np.float64]] = None

    # Precision parameters
    precision_sensory: float = 2.0  # γ - sensory precision
    precision_policy: float = 16.0  # β - policy precision
    precision_state: float = 1.0  # α - state precision

    # Planning horizon
    planning_horizon: int = 3

    def __post_init__(self):
        """Initialize default preferences if not provided"""
        if self.prior_preferences is None:
            # Default uniform preferences
            self.prior_preferences = np.ones(self.num_states) / self.num_states


@dataclass
class TemplateMetadata:
    """Metadata for agent templates"""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "FreeAgentics"
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)


class IAgentTemplate(ABC):
    """Abstract interface for agent templates"""

    @abstractmethod
    def get_template_type(self) -> TemplateType:
        """Get the template type"""
        pass

    @abstractmethod
    def get_metadata(self) -> TemplateMetadata:
        """Get template metadata"""
        pass

    @abstractmethod
    def create_agent_data(self, **kwargs) -> AgentData:
        """Create agent data from template"""
        pass

    @abstractmethod
    def get_active_inference_config(self) -> ActiveInferenceConfig:
        """Get Active Inference configuration"""
        pass


class BaseAgentTemplate(IAgentTemplate):
    """Base implementation of agent template"""

    def __init__(
        self,
        template_type: TemplateType,
        metadata: TemplateMetadata,
        active_inference_config: ActiveInferenceConfig,
        default_personality: Optional[Personality] = None,
        default_resources: Optional[Resources] = None,
    ) -> None:
        self.template_type = template_type
        self.metadata = metadata
        self.active_inference_config = active_inference_config
        self.default_personality = default_personality or self._create_default_personality()
        self.default_resources = default_resources or self._create_default_resources()

    def get_template_type(self) -> TemplateType:
        return self.template_type

    def get_metadata(self) -> TemplateMetadata:
        return self.metadata

    def get_active_inference_config(self) -> ActiveInferenceConfig:
        return self.active_inference_config

    def create_agent_data(self, **kwargs) -> AgentData:
        """Create agent data from template with optional overrides"""
        name = kwargs.get("name", f"{self.template_type.value.title()}Agent")
        position = kwargs.get("position", Position(0.0, 0.0, 0.0))
        agent_id = kwargs.get("agent_id", str(uuid.uuid4()))

        # Merge personalities and resources if provided
        personality = kwargs.get("personality", self.default_personality)
        resources = kwargs.get("resources", self.default_resources)

        agent_data = AgentData(
            agent_id=agent_id,
            name=name,
            agent_type=self.template_type.value,
            position=position,
            personality=personality,
            resources=resources,
            goals=[],
        )

        return agent_data

    def _create_default_personality(self) -> Personality:
        """Create default personality for template type"""
        personality_configs = {
            TemplateType.EXPLORER: Personality(
                openness=0.8,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.6,
                neuroticism=0.3,
            ),
            TemplateType.GUARDIAN: Personality(
                openness=0.4,
                conscientiousness=0.9,
                extraversion=0.5,
                agreeableness=0.8,
                neuroticism=0.2,
            ),
            TemplateType.MERCHANT: Personality(
                openness=0.7,
                conscientiousness=0.8,
                extraversion=0.8,
                agreeableness=0.7,
                neuroticism=0.3,
            ),
            TemplateType.SCHOLAR: Personality(
                openness=0.9,
                conscientiousness=0.8,
                extraversion=0.4,
                agreeableness=0.6,
                neuroticism=0.4,
            ),
        }
        return personality_configs.get(self.template_type, Personality())

    def _create_default_resources(self) -> Resources:
        """Create default resources for template type"""
        resource_configs = {
            TemplateType.EXPLORER: Resources(
                energy=90, health=80, memory_used=30, memory_capacity=100
            ),
            TemplateType.GUARDIAN: Resources(
                energy=70, health=100, memory_used=40, memory_capacity=120
            ),
            TemplateType.MERCHANT: Resources(
                energy=80, health=70, memory_used=50, memory_capacity=150
            ),
            TemplateType.SCHOLAR: Resources(
                energy=60, health=70, memory_used=70, memory_capacity=200
            ),
        }
        return resource_configs.get(self.template_type, Resources())


class ExplorerTemplate(BaseAgentTemplate):
    """Template for exploration-focused agents"""

    def __init__(self) -> None:
        metadata = TemplateMetadata(
            name="Explorer Agent",
            description="Agent optimized for exploration and discovery with high curiosity and mobility",
            tags=["exploration", "discovery", "mobile", "curious"],
            use_cases=["Environment mapping", "Resource discovery", "Pathfinding"],
        )

        ai_config = ActiveInferenceConfig(
            num_states=5,
            state_labels=["idle", "exploring", "investigating", "mapping", "returning"],
            precision_sensory=3.0,
            precision_policy=20.0,
            planning_horizon=4,
        )

        super().__init__(TemplateType.EXPLORER, metadata, ai_config)


class GuardianTemplate(BaseAgentTemplate):
    """Template for protection-focused agents"""

    def __init__(self) -> None:
        metadata = TemplateMetadata(
            name="Guardian Agent",
            description="Agent optimized for protection and monitoring with high reliability",
            tags=["protection", "monitoring", "security", "reliable"],
            use_cases=["Area monitoring", "Threat detection", "Resource protection"],
        )

        ai_config = ActiveInferenceConfig(
            num_states=4,
            state_labels=["patrolling", "monitoring", "alert", "responding"],
            precision_sensory=4.0,
            precision_policy=12.0,
            planning_horizon=2,
        )

        super().__init__(TemplateType.GUARDIAN, metadata, ai_config)


class MerchantTemplate(BaseAgentTemplate):
    """Template for trade-focused agents"""

    def __init__(self) -> None:
        metadata = TemplateMetadata(
            name="Merchant Agent",
            description="Agent optimized for trade, negotiation, and resource management",
            tags=["trade", "commerce", "negotiation", "social"],
            use_cases=["Resource trading", "Market analysis", "Negotiation"],
        )

        ai_config = ActiveInferenceConfig(
            num_states=6,
            state_labels=[
                "idle",
                "seeking_trades",
                "negotiating",
                "trading",
                "analyzing",
                "networking",
            ],
            precision_sensory=2.0,
            precision_policy=18.0,
            planning_horizon=5,
        )

        super().__init__(TemplateType.MERCHANT, metadata, ai_config)


class ScholarTemplate(BaseAgentTemplate):
    """Template for knowledge-focused agents"""

    def __init__(self) -> None:
        metadata = TemplateMetadata(
            name="Scholar Agent",
            description="Agent optimized for learning, research, and knowledge synthesis",
            tags=["research", "learning", "knowledge", "analysis"],
            use_cases=["Information gathering", "Pattern analysis", "Research coordination"],
        )

        ai_config = ActiveInferenceConfig(
            num_states=5,
            state_labels=["idle", "researching", "analyzing", "synthesizing", "sharing"],
            precision_sensory=2.5,
            precision_policy=14.0,
            planning_horizon=6,
        )

        super().__init__(TemplateType.SCHOLAR, metadata, ai_config)


class AgentTemplateFactory:
    """Factory for creating agent templates"""

    _templates: Dict[TemplateType, Type[BaseAgentTemplate]] = {
        TemplateType.EXPLORER: ExplorerTemplate,
        TemplateType.GUARDIAN: GuardianTemplate,
        TemplateType.MERCHANT: MerchantTemplate,
        TemplateType.SCHOLAR: ScholarTemplate,
    }

    @classmethod
    def create_template(cls, template_type: TemplateType) -> IAgentTemplate:
        """Create a template instance"""
        template_class = cls._templates.get(template_type)
        if not template_class:
            raise ValueError(f"Unknown template type: {template_type}")
        # Template classes have their own __init__ methods that call super() with args
        return cast(IAgentTemplate, template_class())  # type: ignore[call-arg]

    @classmethod
    def get_available_templates(cls) -> List[TemplateType]:
        """Get list of available template types"""
        return list(cls._templates.keys())

    @classmethod
    def create_agent_from_template(cls, template_type: TemplateType, **kwargs) -> BaseAgent:
        """Create a complete agent instance from template"""
        template = cls.create_template(template_type)
        agent_data = template.create_agent_data(**kwargs)
        return BaseAgent(agent_data)

    @classmethod
    def get_template_metadata(cls, template_type: TemplateType) -> TemplateMetadata:
        """Get metadata for a template type"""
        template = cls.create_template(template_type)
        return template.get_metadata()


# Convenience functions
def create_explorer_agent(**kwargs) -> BaseAgent:
    """Create an explorer agent"""
    return AgentTemplateFactory.create_agent_from_template(TemplateType.EXPLORER, **kwargs)


def create_guardian_agent(**kwargs) -> BaseAgent:
    """Create a guardian agent"""
    return AgentTemplateFactory.create_agent_from_template(TemplateType.GUARDIAN, **kwargs)


def create_merchant_agent(**kwargs) -> BaseAgent:
    """Create a merchant agent"""
    return AgentTemplateFactory.create_agent_from_template(TemplateType.MERCHANT, **kwargs)


def create_scholar_agent(**kwargs) -> BaseAgent:
    """Create a scholar agent"""
    return AgentTemplateFactory.create_agent_from_template(TemplateType.SCHOLAR, **kwargs)
