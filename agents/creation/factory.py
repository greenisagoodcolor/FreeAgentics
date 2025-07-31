"""Agent factory for creating agents from natural language prompts.

Main orchestrator that coordinates all agent creation services with proper
error handling, observability, and fallback mechanisms.
"""

import logging
import time
from typing import List, Optional

from api.v1.services.llm_service import LLMService
from database.models import AgentType

from .interfaces import (
    IAgentBuilder,
    IAgentFactory,
    IPersonalityGenerator,
    IPromptAnalyzer,
    ISystemPromptBuilder,
)
from .models import (
    AgentCreationError,
    AgentCreationRequest,
    AgentCreationResult,
    AgentSpecification,
)
from .services import AgentBuilder, LLMPromptAnalyzer, PersonalityGenerator, SystemPromptBuilder

logger = logging.getLogger(__name__)


class AgentFactory(IAgentFactory):
    """Main factory for creating agents from natural language prompts.

    Orchestrates the entire agent creation pipeline with proper error handling,
    observability, and fallback mechanisms as recommended by the Nemesis Committee.
    """

    def __init__(
        self,
        prompt_analyzer: Optional[IPromptAnalyzer] = None,
        personality_generator: Optional[IPersonalityGenerator] = None,
        system_prompt_builder: Optional[ISystemPromptBuilder] = None,
        agent_builder: Optional[IAgentBuilder] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """Initialize agent factory with dependency injection for testing."""

        self.llm_service = llm_service or LLMService()

        # Initialize services with dependency injection
        self.prompt_analyzer = prompt_analyzer or LLMPromptAnalyzer(self.llm_service)
        self.personality_generator = personality_generator or PersonalityGenerator(self.llm_service)
        self.system_prompt_builder = system_prompt_builder or SystemPromptBuilder(self.llm_service)
        self.agent_builder = agent_builder or AgentBuilder()

        # Metrics tracking
        self._metrics = {
            "agents_created": 0,
            "creation_failures": 0,
            "avg_creation_time_ms": 0.0,
            "llm_calls_total": 0,
            "fallback_used": 0,
        }

    async def create_agent(self, request: AgentCreationRequest) -> AgentCreationResult:
        """Create an agent from a natural language prompt.

        This is the main entry point that orchestrates the entire agent creation process.
        """

        start_time = time.time()
        result = AgentCreationResult()

        try:
            logger.info(f"Starting agent creation from prompt: {request.prompt[:100]}...")

            # Step 1: Analyze the prompt to determine agent requirements
            logger.debug("Analyzing prompt...")
            analysis_result = await self._safe_analyze_prompt(request.prompt)
            result.analysis_result = analysis_result

            # Override agent type if user specified a preference
            agent_type = request.preferred_type or analysis_result.agent_type

            # Step 2: Generate personality profile
            logger.debug(f"Generating personality for {agent_type.value} agent...")
            personality = await self._safe_generate_personality(agent_type, analysis_result.context)

            # Step 3: Build system prompt
            logger.debug("Building system prompt...")
            system_prompt = await self._safe_build_system_prompt(
                agent_type,
                personality,
                analysis_result.context,
                analysis_result.capabilities if request.enable_custom_capabilities else None,
            )

            # Step 4: Create agent specification
            agent_name = request.agent_name or self._generate_agent_name(
                agent_type, analysis_result.domain
            )

            specification = AgentSpecification(
                name=agent_name,
                agent_type=agent_type,
                system_prompt=system_prompt,
                personality=personality,
                source_prompt=request.prompt,
                creation_source="ai_generated",
                capabilities=analysis_result.capabilities
                if request.enable_custom_capabilities
                else [],
            )
            result.specification = specification

            # Step 5: Create and persist the agent (unless preview only)
            if not request.preview_only:
                logger.debug("Creating and persisting agent...")
                agent = await self._safe_build_agent(specification)
                result.agent = agent
                self._metrics["agents_created"] += 1

            # Calculate metrics
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            result.success = True

            # Update average creation time
            self._update_avg_creation_time(processing_time)

            logger.info(f"Agent creation completed successfully in {processing_time}ms")
            return result

        except Exception as e:
            # Record failure metrics
            self._metrics["creation_failures"] += 1
            processing_time = int((time.time() - start_time) * 1000)

            logger.error(f"Agent creation failed after {processing_time}ms: {e}")

            result.success = False
            result.error_message = str(e)
            result.processing_time_ms = processing_time

            return result

    async def preview_agent(self, prompt: str) -> AgentSpecification:
        """Preview what an agent would look like without creating it."""

        request = AgentCreationRequest(prompt=prompt, preview_only=True)
        result = await self.create_agent(request)

        if not result.success:
            raise AgentCreationError(f"Preview failed: {result.error_message}")

        if not result.specification:
            raise AgentCreationError("Preview failed: No specification generated")

        return result.specification

    async def get_supported_agent_types(self) -> List[AgentType]:
        """Get list of supported agent types."""
        return list(AgentType)

    async def _safe_analyze_prompt(self, prompt: str):
        """Safely analyze prompt with error handling and fallback."""

        try:
            self._metrics["llm_calls_total"] += 1
            return await self.prompt_analyzer.analyze_prompt(prompt)
        except Exception as e:
            logger.warning(f"Prompt analysis failed, using fallback: {e}")
            self._metrics["fallback_used"] += 1

            # Create a basic analysis result as fallback
            from .models import AnalysisConfidence, PromptAnalysisResult

            return PromptAnalysisResult(
                agent_type=AgentType.ANALYST,  # Safe default
                confidence=AnalysisConfidence.LOW,
                original_prompt=prompt,
                processed_prompt=prompt,
                reasoning="Fallback analysis due to service failure",
            )

    async def _safe_generate_personality(self, agent_type: AgentType, context: Optional[str]):
        """Safely generate personality with error handling and fallback."""

        try:
            if context:
                self._metrics["llm_calls_total"] += 1
                return await self.personality_generator.generate_personality(agent_type, context)
            else:
                # Use default personality if no context
                return self.personality_generator.get_default_personality(agent_type)
        except Exception as e:
            logger.warning(f"Personality generation failed, using default: {e}")
            self._metrics["fallback_used"] += 1
            return self.personality_generator.get_default_personality(agent_type)

    async def _safe_build_system_prompt(
        self,
        agent_type: AgentType,
        personality,
        context: Optional[str],
        capabilities: Optional[List[str]],
    ):
        """Safely build system prompt with error handling and fallback."""

        try:
            self._metrics["llm_calls_total"] += 1
            return await self.system_prompt_builder.build_system_prompt(
                agent_type, personality, context, capabilities
            )
        except Exception as e:
            logger.warning(f"System prompt building failed, using template: {e}")
            self._metrics["fallback_used"] += 1
            return self.system_prompt_builder.get_template_prompt(agent_type)

    async def _safe_build_agent(self, specification: AgentSpecification):
        """Safely build and persist agent with error handling."""

        try:
            return await self.agent_builder.build_agent(specification)
        except Exception as e:
            logger.error(f"Agent building failed: {e}")
            raise AgentCreationError(f"Failed to create agent: {str(e)}")

    def _generate_agent_name(self, agent_type: AgentType, domain: Optional[str]) -> str:
        """Generate a descriptive name for the agent."""

        base_names = {
            AgentType.ADVOCATE: "Advocate",
            AgentType.ANALYST: "Analyst",
            AgentType.CRITIC: "Critic",
            AgentType.CREATIVE: "Creative",
            AgentType.MODERATOR: "Moderator",
        }

        base_name = base_names[agent_type]

        if domain:
            # Clean up domain name
            domain_clean = domain.replace("_", " ").title()
            return f"{domain_clean} {base_name}"
        else:
            return f"{base_name} Agent"

    def _update_avg_creation_time(self, new_time_ms: int):
        """Update rolling average creation time."""

        current_avg = self._metrics["avg_creation_time_ms"]
        total_created = self._metrics["agents_created"]

        if total_created <= 1:
            self._metrics["avg_creation_time_ms"] = new_time_ms
        else:
            # Simple exponential moving average
            alpha = 0.1
            self._metrics["avg_creation_time_ms"] = (alpha * new_time_ms) + (
                (1 - alpha) * current_avg
            )

    def get_metrics(self) -> dict:
        """Get factory metrics for monitoring."""

        metrics = self._metrics.copy()

        # Add computed metrics
        total_requests = metrics["agents_created"] + metrics["creation_failures"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["agents_created"] / total_requests
            metrics["failure_rate"] = metrics["creation_failures"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0

        if metrics["llm_calls_total"] > 0:
            metrics["fallback_rate"] = metrics["fallback_used"] / metrics["llm_calls_total"]
        else:
            metrics["fallback_rate"] = 0.0

        return metrics

    def clear_metrics(self):
        """Clear metrics (useful for testing)."""
        self._metrics = {
            "agents_created": 0,
            "creation_failures": 0,
            "avg_creation_time_ms": 0.0,
            "llm_calls_total": 0,
            "fallback_used": 0,
        }
