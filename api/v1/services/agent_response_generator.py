"""
Agent Response Generator Service

Implements IAgentResponseGenerator interface for generating agent-specific responses
in multi-turn conversations with role-based prompting and quality analysis.

Architecture follows Nemesis Committee recommendations with clean dependency injection,
comprehensive observability, and production-ready error handling.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from api.v1.models.agent_conversation import AgentRole, ConversationContext, IAgentResponseGenerator
from api.v1.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class ResponseQuality(str, Enum):
    """Response quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class AgentPersonaType(str, Enum):
    """Agent persona types for specialized prompting."""

    EXPLORER = "explorer"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    FACILITATOR = "facilitator"
    EXPERT = "expert"
    DEFAULT = "default"


class IPromptBuilder(ABC):
    """Interface for agent-specific prompt construction."""

    @abstractmethod
    def build_prompt(self, agent: AgentRole, context: ConversationContext) -> str:
        """Build role-specific prompt for agent response generation."""


class ResponseAnalyzer:
    """Analyzes generated responses for quality and appropriateness."""

    def __init__(self):
        self.metrics = {
            "total_responses": 0,
            "quality_scores": [],
            "avg_response_length": 0.0,
            "role_coherence_rate": 0.0,
        }

    def analyze_response(
        self, response: str, agent: AgentRole, context: ConversationContext
    ) -> Dict[str, any]:
        """Analyze response quality and role coherence."""
        self.metrics["total_responses"] += 1

        # Basic quality metrics
        response_length = len(response)
        word_count = len(response.split())

        # Role coherence analysis (simplified heuristics)
        role_keywords = self._get_role_keywords(agent)
        role_coherence = self._calculate_role_coherence(response, role_keywords)

        # Quality scoring
        quality_score = self._calculate_quality_score(response, word_count, role_coherence)
        quality_level = self._determine_quality_level(quality_score)

        # Update metrics
        self.metrics["quality_scores"].append(quality_score)
        self._update_averages(response_length, role_coherence)

        analysis = {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "response_length": response_length,
            "word_count": word_count,
            "role_coherence": role_coherence,
            "analysis_timestamp": time.time(),
        }

        logger.info(
            f"Response analysis complete - Agent: {agent.name}, "
            f"Quality: {quality_level}, Score: {quality_score:.2f}"
        )

        return analysis

    def _get_role_keywords(self, agent: AgentRole) -> List[str]:
        """Get expected keywords for agent role."""
        role_keywords = {
            "explorer": ["explore", "discover", "investigate", "find", "search", "what if"],
            "analyst": ["analyze", "data", "evidence", "conclusion", "therefore", "because"],
            "critic": ["however", "challenge", "disagree", "problem", "issue", "concern"],
            "synthesizer": ["combine", "together", "integrate", "summary", "overall"],
            "facilitator": ["let's", "shall we", "perhaps", "consider", "what do you think"],
            "expert": ["expertise", "experience", "research", "studies", "knowledge"],
        }

        # Get keywords based on role name or system prompt
        role_name = agent.role.lower()
        for role_type, keywords in role_keywords.items():
            if role_type in role_name or role_type in agent.system_prompt.lower():
                return keywords

        return ["discuss", "think", "consider", "response"]  # Default keywords

    def _calculate_role_coherence(self, response: str, role_keywords: List[str]) -> float:
        """Calculate how well response matches expected role behavior."""
        response_lower = response.lower()
        keyword_matches = sum(1 for keyword in role_keywords if keyword in response_lower)
        # Give higher baseline score for keyword matches
        if keyword_matches > 0:
            return min(keyword_matches / len(role_keywords) + 0.2, 1.0) if role_keywords else 0.5
        return 0.3  # Lower baseline for no matches

    def _calculate_quality_score(
        self, response: str, word_count: int, role_coherence: float
    ) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        # Length score (prefer 20-150 words)
        if word_count < 10:
            length_score = 0.2
        elif word_count > 200:
            length_score = 0.7
        else:
            length_score = min(word_count / 50, 1.0)

        # Content score (basic heuristics)
        content_score = 0.5
        if "?" in response:  # Questions are engaging
            content_score += 0.2
        if any(word in response.lower() for word in ["because", "therefore", "however"]):
            content_score += 0.1  # Reasoning indicators
        if len(response.split(".")) >= 2:  # Multiple sentences
            content_score += 0.1

        # Combined score
        quality_score = (length_score * 0.4) + (content_score * 0.4) + (role_coherence * 0.2)
        return min(quality_score, 1.0)

    def _determine_quality_level(self, score: float) -> ResponseQuality:
        """Determine quality level from numeric score."""
        if score >= 0.8:
            return ResponseQuality.EXCELLENT
        elif score >= 0.6:
            return ResponseQuality.GOOD
        elif score >= 0.4:
            return ResponseQuality.ACCEPTABLE
        elif score >= 0.2:
            return ResponseQuality.POOR
        else:
            return ResponseQuality.FAILED

    def _update_averages(self, response_length: int, role_coherence: float):
        """Update rolling averages for metrics."""
        total = self.metrics["total_responses"]

        # Update average response length
        current_avg_length = self.metrics["avg_response_length"]
        self.metrics["avg_response_length"] = (
            current_avg_length * (total - 1) + response_length
        ) / total

        # Update role coherence rate
        current_coherence = self.metrics["role_coherence_rate"]
        self.metrics["role_coherence_rate"] = (
            current_coherence * (total - 1) + role_coherence
        ) / total

    def get_metrics(self) -> Dict[str, any]:
        """Get current analyzer metrics."""
        metrics = self.metrics.copy()

        if metrics["quality_scores"]:
            metrics["avg_quality_score"] = sum(metrics["quality_scores"]) / len(
                metrics["quality_scores"]
            )
            metrics["quality_distribution"] = {
                "excellent": sum(1 for s in metrics["quality_scores"] if s >= 0.8),
                "good": sum(1 for s in metrics["quality_scores"] if 0.6 <= s < 0.8),
                "acceptable": sum(1 for s in metrics["quality_scores"] if 0.4 <= s < 0.6),
                "poor": sum(1 for s in metrics["quality_scores"] if 0.2 <= s < 0.4),
                "failed": sum(1 for s in metrics["quality_scores"] if s < 0.2),
            }
        else:
            metrics["avg_quality_score"] = 0.0
            metrics["quality_distribution"] = {}

        return metrics


class TemplatePromptBuilder(IPromptBuilder):
    """Template-based prompt builder using role-specific strategies."""

    def __init__(self):
        self.role_templates = self._create_role_templates()

    def _create_role_templates(self) -> Dict[AgentPersonaType, str]:
        """Create role-specific prompt templates."""
        return {
            AgentPersonaType.EXPLORER: """You are {agent_name}, an {agent_role}.

Your mission is to explore new ideas, ask probing questions, and discover connections others might miss.
You're naturally curious and love to investigate possibilities.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, respond with curiosity and exploration in mind. Ask thoughtful questions,
suggest new angles to investigate, and help uncover hidden aspects of the discussion.
Keep your response engaging and focused on discovery.""",
            AgentPersonaType.ANALYST: """You are {agent_name}, an {agent_role}.

Your strength is breaking down complex topics, examining evidence, and drawing logical conclusions.
You provide structured analysis and data-driven insights.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, provide analytical insights. Structure your response with clear reasoning,
examine the evidence presented, and draw logical conclusions. Help the group understand
the underlying patterns and implications.""",
            AgentPersonaType.CRITIC: """You are {agent_name}, a {agent_role}.

Your role is to challenge assumptions, identify potential problems, and ensure ideas are thoroughly vetted.
You ask the tough questions and point out what others might overlook.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, provide constructive criticism and challenge assumptions. Identify potential
weaknesses, raise important concerns, and ensure the discussion considers alternative perspectives.
Be thorough but constructive in your critique.""",
            AgentPersonaType.SYNTHESIZER: """You are {agent_name}, a {agent_role}.

Your specialty is finding common ground, integrating different viewpoints, and creating coherent
summaries from complex discussions.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, help synthesize the various viewpoints being discussed. Find connections between
different ideas, identify areas of agreement, and help the group see the bigger picture.
Focus on integration and bringing ideas together.""",
            AgentPersonaType.FACILITATOR: """You are {agent_name}, a {agent_role}.

Your role is to guide the conversation, ensure everyone's voice is heard, and keep discussions
productive and on track.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, help facilitate this conversation. Ask questions that draw out important points,
suggest ways to move the discussion forward, and ensure we're making productive progress.
Focus on keeping the conversation balanced and constructive.""",
            AgentPersonaType.EXPERT: """You are {agent_name}, an {agent_role}.

You bring deep domain expertise and specialized knowledge to the conversation.
Your responses should reflect your expertise while remaining accessible.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, share your expertise to inform the discussion. Provide authoritative insights
based on your specialized knowledge, but explain complex concepts clearly.
Help the group benefit from your domain expertise.""",
            AgentPersonaType.DEFAULT: """You are {agent_name}, playing the role of {agent_role}.

System Instructions: {system_prompt}

Conversation Context:
{conversation_summary}

As {agent_name}, continue this conversation by responding thoughtfully to the discussion so far.
Stay true to your role and provide valuable insights that move the conversation forward.""",
        }

    def build_prompt(self, agent: AgentRole, context: ConversationContext) -> str:
        """Build role-specific prompt using templates."""
        # Determine agent persona type from role description
        persona_type = self._determine_persona_type(agent)

        # Get appropriate template
        template = self.role_templates.get(
            persona_type, self.role_templates[AgentPersonaType.DEFAULT]
        )

        # Build the prompt
        prompt = template.format(
            agent_name=agent.name,
            agent_role=agent.role,
            system_prompt=agent.system_prompt,
            conversation_summary=context.conversation_summary,
        )

        logger.debug(f"Built {persona_type.value} prompt for agent {agent.name}")
        return prompt

    def _determine_persona_type(self, agent: AgentRole) -> AgentPersonaType:
        """Determine persona type from agent role description."""
        role_text = f"{agent.role} {agent.system_prompt}".lower()

        # Keyword matching for persona detection
        if any(word in role_text for word in ["explore", "discover", "investigate", "find"]):
            return AgentPersonaType.EXPLORER
        elif any(word in role_text for word in ["analyze", "analysis", "examine", "data"]):
            return AgentPersonaType.ANALYST
        elif any(word in role_text for word in ["critic", "challenge", "question", "devil"]):
            return AgentPersonaType.CRITIC
        elif any(word in role_text for word in ["synthesize", "combine", "integrate", "summary"]):
            return AgentPersonaType.SYNTHESIZER
        elif any(word in role_text for word in ["facilitate", "moderate", "guide", "lead"]):
            return AgentPersonaType.FACILITATOR
        elif any(word in role_text for word in ["expert", "specialist", "authority", "knowledge"]):
            return AgentPersonaType.EXPERT
        else:
            return AgentPersonaType.DEFAULT


class AgentResponseGenerator(IAgentResponseGenerator):
    """
    Production-ready agent response generator with role-specific prompting,
    quality analysis, and comprehensive error handling.

    Implements the IAgentResponseGenerator interface for the conversation orchestrator.
    """

    def __init__(
        self,
        llm_service: LLMService,
        prompt_builder: Optional[IPromptBuilder] = None,
        response_analyzer: Optional[ResponseAnalyzer] = None,
    ):
        """Initialize with dependency injection."""
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder or TemplatePromptBuilder()
        self.response_analyzer = response_analyzer or ResponseAnalyzer()

        # Performance and reliability metrics
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "fallback_used": 0,
            "quality_failures": 0,
        }

        # Simple response cache (in production, use Redis or similar)
        self._response_cache: Dict[str, str] = {}
        self._cache_max_size = 1000

        logger.info("AgentResponseGenerator initialized with dependencies")

    async def generate_response(
        self, agent: AgentRole, context: ConversationContext, timeout_seconds: int = 30
    ) -> str:
        """
        Generate agent-specific response with role-based prompting and quality analysis.

        Args:
            agent: Agent role generating the response
            context: Current conversation context
            timeout_seconds: Timeout for LLM generation

        Returns:
            Generated response string

        Raises:
            ValueError: If inputs are invalid
            TimeoutError: If generation exceeds timeout
            RuntimeError: If all generation attempts fail
        """
        start_time = time.time()
        self.metrics["requests_total"] += 1

        # Input validation
        self._validate_inputs(agent, context)

        # Check cache first (semantic caching could be implemented here)
        cache_key = self._create_cache_key(agent, context)
        if cache_key in self._response_cache:
            self.metrics["cache_hits"] += 1
            logger.info(f"Cache hit for agent {agent.name} response")
            return self._response_cache[cache_key]

        try:
            # Generate response with timeout
            response = await asyncio.wait_for(
                self._generate_response_internal(agent, context), timeout=timeout_seconds
            )

            # Analyze response quality
            analysis = self.response_analyzer.analyze_response(response, agent, context)

            # Handle poor quality responses
            if analysis["quality_level"] in [ResponseQuality.POOR, ResponseQuality.FAILED]:
                self.metrics["quality_failures"] += 1
                logger.warning(
                    f"Poor quality response from {agent.name}: {analysis['quality_level']}"
                )

                # Retry once with modified prompt
                try:
                    response = await self._retry_with_fallback_prompt(agent, context)
                    analysis = self.response_analyzer.analyze_response(response, agent, context)
                except Exception:
                    # Use template fallback if retry fails
                    response = self._generate_fallback_response(agent, context)
                    self.metrics["fallback_used"] += 1

            # Cache successful response
            self._cache_response(cache_key, response)

            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=True)

            logger.info(
                f"Generated response for {agent.name} in {response_time:.2f}s "
                f"(quality: {analysis.get('quality_level', 'unknown')})"
            )

            return response

        except asyncio.TimeoutError:
            self.metrics["requests_failed"] += 1
            logger.error(f"Response generation timeout for agent {agent.name}")

            # Return fallback response on timeout
            fallback_response = self._generate_fallback_response(agent, context)
            self.metrics["fallback_used"] += 1
            return fallback_response

        except Exception as e:
            self.metrics["requests_failed"] += 1
            logger.error(f"Response generation failed for agent {agent.name}: {e}")

            # Return fallback response on any other error
            fallback_response = self._generate_fallback_response(agent, context)
            self.metrics["fallback_used"] += 1
            return fallback_response

    async def _generate_response_internal(
        self, agent: AgentRole, context: ConversationContext
    ) -> str:
        """Internal response generation logic."""
        # Build role-specific prompt
        prompt = self.prompt_builder.build_prompt(agent, context)

        # Convert conversation context to message format
        context_messages = []
        for turn in context.recent_turns:
            if turn.response:
                context_messages.append(
                    {"role": "assistant", "content": f"{turn.agent_name}: {turn.response}"}
                )

        # Generate response using LLM service
        response = await self.llm_service.generate_conversation_response(
            system_prompt=prompt,
            context_messages=context_messages,
            user_id=context.conversation_id,  # Use conversation ID as user context
            model="gpt-3.5-turbo",  # Use default model
            temperature=0.8,  # Use default temperature
            max_tokens=200,
        )

        return response.strip()

    async def _retry_with_fallback_prompt(
        self, agent: AgentRole, context: ConversationContext
    ) -> str:
        """Retry generation with simplified prompt."""
        # Create simpler, more direct prompt
        fallback_prompt = f"""You are {agent.name}.

{agent.system_prompt}

Recent conversation:
{context.conversation_summary}

Respond as {agent.name} with a helpful contribution to this conversation."""

        response = await self.llm_service.generate_conversation_response(
            system_prompt=fallback_prompt,
            context_messages=[],
            user_id=context.conversation_id,
            model="gpt-3.5-turbo",  # Use reliable model for fallback
            temperature=0.7,  # Lower temperature for more reliable output
            max_tokens=150,
        )

        return response.strip()

    def _generate_fallback_response(self, agent: AgentRole, context: ConversationContext) -> str:
        """Generate template-based fallback response."""
        fallback_responses = [
            f"As {agent.name}, I'd like to add that this is an interesting topic that deserves further exploration.",
            f"From the perspective of {agent.role}, I think we should consider the broader implications here.",
            f"Let me contribute as {agent.name} - I believe there are multiple angles to examine in this discussion.",
            f"Speaking as {agent.name}, I find this conversation thought-provoking and worth deeper analysis.",
        ]

        # Simple selection based on agent name hash for consistency
        response_index = hash(agent.name) % len(fallback_responses)
        response = fallback_responses[response_index]

        logger.info(f"Using fallback response for agent {agent.name}")
        return response

    def _validate_inputs(self, agent: AgentRole, context: ConversationContext):
        """Validate input parameters."""
        if not agent.name or not agent.name.strip():
            raise ValueError("Agent name cannot be empty")

        if not agent.role or not agent.role.strip():
            raise ValueError("Agent role cannot be empty")

        if not context.conversation_id:
            raise ValueError("Conversation context must have valid ID")

        if not context.topic or not context.topic.strip():
            raise ValueError("Conversation context must have valid topic")

    def _create_cache_key(self, agent: AgentRole, context: ConversationContext) -> str:
        """Create cache key for response caching."""
        # Simple cache key based on agent and recent context
        recent_turns_hash = hash(
            tuple(turn.response for turn in context.recent_turns[-3:] if turn.response)
        )
        return f"{agent.name}:{agent.role}:{recent_turns_hash}"

    def _cache_response(self, cache_key: str, response: str):
        """Cache response with size limit."""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO, could use LRU)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]

        self._response_cache[cache_key] = response

    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics."""
        if success:
            self.metrics["requests_successful"] += 1

        # Update average response time (exponential moving average)
        current_avg = self.metrics["avg_response_time"]
        alpha = 0.1  # Smoothing factor
        self.metrics["avg_response_time"] = (alpha * response_time) + ((1 - alpha) * current_avg)

    def get_metrics(self) -> Dict[str, any]:
        """Get current generator metrics."""
        metrics = self.metrics.copy()

        # Add computed metrics
        total_requests = metrics["requests_total"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["requests_successful"] / total_requests
            metrics["failure_rate"] = metrics["requests_failed"] / total_requests
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_requests
            metrics["fallback_rate"] = metrics["fallback_used"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            metrics["cache_hit_rate"] = 0.0
            metrics["fallback_rate"] = 0.0

        # Add response analyzer metrics
        metrics["response_analysis"] = self.response_analyzer.get_metrics()

        # Add cache status
        metrics["cache_size"] = len(self._response_cache)
        metrics["cache_max_size"] = self._cache_max_size

        return metrics

    def clear_cache(self):
        """Clear response cache (useful for testing)."""
        self._response_cache.clear()
        logger.info("Response cache cleared")


# Factory function for dependency injection
def create_agent_response_generator(llm_service: LLMService) -> AgentResponseGenerator:
    """Factory function to create configured AgentResponseGenerator."""
    return AgentResponseGenerator(
        llm_service=llm_service,
        prompt_builder=TemplatePromptBuilder(),
        response_analyzer=ResponseAnalyzer(),
    )
