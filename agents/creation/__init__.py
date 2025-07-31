"""Agent creation from natural language prompts.

This module provides AI-powered agent creation capabilities that analyze
user prompts and generate specialized agents with unique roles and personalities.
"""

from .factory import AgentFactory
from .interfaces import (
    IAgentBuilder,
    IAgentFactory,
    IPersonalityGenerator,
    IPromptAnalyzer,
    ISystemPromptBuilder,
)
from .models import (
    AgentCreationRequest,
    AgentCreationResult,
    AgentSpecification,
    PersonalityProfile,
    PromptAnalysisResult,
)
from .services import AgentBuilder, LLMPromptAnalyzer, PersonalityGenerator, SystemPromptBuilder

__all__ = [
    # Interfaces
    "IAgentFactory",
    "IPromptAnalyzer",
    "IAgentBuilder",
    "IPersonalityGenerator",
    "ISystemPromptBuilder",
    # Models
    "AgentSpecification",
    "PersonalityProfile",
    "PromptAnalysisResult",
    "AgentCreationRequest",
    "AgentCreationResult",
    # Implementation classes
    "AgentFactory",
    "LLMPromptAnalyzer",
    "PersonalityGenerator",
    "SystemPromptBuilder",
    "AgentBuilder",
]
