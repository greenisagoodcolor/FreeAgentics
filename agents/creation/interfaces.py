"""Interfaces for agent creation system.

Defines clear contracts for all components following single responsibility
principle and enabling dependency injection for testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from database.models import Agent, AgentType

from .models import (
    AgentCreationRequest,
    AgentCreationResult,
    AgentSpecification,
    PersonalityProfile,
    PromptAnalysisResult,
)


class IPromptAnalyzer(ABC):
    """Interface for analyzing natural language prompts to extract agent requirements."""

    @abstractmethod
    async def analyze_prompt(self, prompt: str) -> PromptAnalysisResult:
        """Analyze a natural language prompt and extract agent requirements.

        Args:
            prompt: The user's natural language description of what they need

        Returns:
            PromptAnalysisResult containing identified agent type, capabilities, and context

        Raises:
            PromptAnalysisError: If the prompt cannot be analyzed or is invalid
        """
        pass

    @abstractmethod
    async def validate_prompt(self, prompt: str) -> bool:
        """Validate that a prompt is suitable for agent creation.

        Args:
            prompt: The user's natural language prompt

        Returns:
            True if the prompt is valid and can be analyzed
        """
        pass


class IPersonalityGenerator(ABC):
    """Interface for generating personality profiles for agents."""

    @abstractmethod
    async def generate_personality(
        self,
        agent_type: AgentType,
        context: Optional[str] = None,
        traits_hint: Optional[Dict[str, Any]] = None,
    ) -> PersonalityProfile:
        """Generate a personality profile for an agent.

        Args:
            agent_type: The type of agent (Advocate, Analyst, etc.)
            context: Additional context from prompt analysis
            traits_hint: Optional hints for specific personality traits

        Returns:
            PersonalityProfile with generated traits and characteristics
        """
        pass

    @abstractmethod
    def get_default_personality(self, agent_type: AgentType) -> PersonalityProfile:
        """Get a default personality profile for an agent type.

        Args:
            agent_type: The type of agent

        Returns:
            Default PersonalityProfile for the agent type
        """
        pass


class ISystemPromptBuilder(ABC):
    """Interface for building system prompts for agents."""

    @abstractmethod
    async def build_system_prompt(
        self,
        agent_type: AgentType,
        personality: PersonalityProfile,
        context: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> str:
        """Build a system prompt for an agent.

        Args:
            agent_type: The type of agent
            personality: The agent's personality profile
            context: Additional context for the prompt
            capabilities: Specific capabilities the agent should have

        Returns:
            Complete system prompt string
        """
        pass

    @abstractmethod
    def get_template_prompt(self, agent_type: AgentType) -> str:
        """Get a template system prompt for an agent type.

        Args:
            agent_type: The type of agent

        Returns:
            Template system prompt string
        """
        pass


class IAgentBuilder(ABC):
    """Interface for building and persisting agent instances."""

    @abstractmethod
    async def build_agent(self, specification: AgentSpecification) -> Agent:
        """Build and persist an agent from a specification.

        Args:
            specification: Complete agent specification

        Returns:
            Persisted Agent instance

        Raises:
            AgentBuildError: If the agent cannot be built or persisted
        """
        pass

    @abstractmethod
    async def update_agent(self, agent_id: str, specification: AgentSpecification) -> Agent:
        """Update an existing agent with new specification.

        Args:
            agent_id: ID of the agent to update
            specification: Updated agent specification

        Returns:
            Updated Agent instance
        """
        pass


class IAgentFactory(ABC):
    """Main interface for creating agents from natural language prompts."""

    @abstractmethod
    async def create_agent(self, request: AgentCreationRequest) -> AgentCreationResult:
        """Create an agent from a natural language prompt.

        This is the main entry point that orchestrates the entire agent creation process.

        Args:
            request: Agent creation request with prompt and optional parameters

        Returns:
            AgentCreationResult with the created agent and metadata

        Raises:
            AgentCreationError: If agent creation fails at any stage
        """
        pass

    @abstractmethod
    async def preview_agent(self, prompt: str) -> AgentSpecification:
        """Preview what an agent would look like without creating it.

        Args:
            prompt: Natural language prompt

        Returns:
            AgentSpecification showing what would be created
        """
        pass

    @abstractmethod
    async def get_supported_agent_types(self) -> List[AgentType]:
        """Get list of supported agent types.

        Returns:
            List of AgentType values that can be created
        """
        pass
