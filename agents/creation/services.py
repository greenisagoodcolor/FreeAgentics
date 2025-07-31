"""Implementation services for agent creation system.

Concrete implementations of the agent creation interfaces using LLM services
and database persistence.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from api.v1.services.llm_service import LLMService
from core.providers import get_db
from database.models import Agent, AgentStatus, AgentType
from inference.llm.provider_interface import GenerationRequest

from .interfaces import IAgentBuilder, IPersonalityGenerator, IPromptAnalyzer, ISystemPromptBuilder
from .models import (
    AgentBuildError,
    AgentSpecification,
    AnalysisConfidence,
    PersonalityGenerationError,
    PersonalityProfile,
    PromptAnalysisError,
    PromptAnalysisResult,
    SystemPromptBuildError,
)

logger = logging.getLogger(__name__)


class LLMPromptAnalyzer(IPromptAnalyzer):
    """LLM-powered prompt analyzer that identifies agent requirements."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._agent_type_descriptions = {
            AgentType.ADVOCATE: "argues for specific positions, builds compelling cases, persuades others",
            AgentType.ANALYST: "breaks down complex problems, provides data-driven insights, systematic thinking",
            AgentType.CRITIC: "identifies flaws and weaknesses, challenges assumptions, points out problems",
            AgentType.CREATIVE: "generates novel ideas, thinks outside the box, innovative solutions",
            AgentType.MODERATOR: "facilitates discussions, maintains balance, ensures fair participation",
        }

    async def analyze_prompt(self, prompt: str) -> PromptAnalysisResult:
        """Analyze a natural language prompt to extract agent requirements."""

        if not await self.validate_prompt(prompt):
            raise PromptAnalysisError(f"Invalid prompt: {prompt}")

        try:
            # Clean and preprocess the prompt
            processed_prompt = self._preprocess_prompt(prompt)

            # Use LLM to analyze the prompt
            analysis_json = await self._llm_analyze_prompt(processed_prompt)

            # Parse the analysis result
            result = self._parse_analysis_result(analysis_json, prompt, processed_prompt)

            logger.info(
                f"Analyzed prompt: {prompt[:50]}... -> {result.agent_type.value} ({result.confidence.value})"
            )
            return result

        except Exception as e:
            logger.error(f"Prompt analysis failed: {e}")
            # Fall back to rule-based analysis
            return self._fallback_analyze_prompt(prompt)

    async def validate_prompt(self, prompt: str) -> bool:
        """Validate that a prompt is suitable for agent creation."""

        if not prompt or len(prompt.strip()) < 10:
            return False

        # Check for obviously invalid content
        invalid_patterns = [
            r"^\s*$",  # Empty or whitespace only
            r"^[\W\d_]+$",  # Only special characters and numbers
            r"test\s*test\s*test",  # Obvious test patterns
        ]

        prompt_lower = prompt.lower().strip()
        for pattern in invalid_patterns:
            if re.match(pattern, prompt_lower):
                return False

        return True

    def _preprocess_prompt(self, prompt: str) -> str:
        """Clean and normalize the prompt for analysis."""

        # Basic cleaning
        cleaned = prompt.strip()

        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Ensure it ends with punctuation for better LLM parsing
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."

        return cleaned

    async def _llm_analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to analyze the prompt and determine agent type."""

        system_prompt = f"""You are an expert at analyzing requests and determining what type of AI agent would be most helpful.

Based on the user's request, determine which agent type would be most suitable:

{chr(10).join(f"- {agent_type.value.upper()}: {desc}" for agent_type, desc in self._agent_type_descriptions.items())}

Analyze the request and return your analysis as JSON in this exact format:
{{
    "agent_type": "advocate|analyst|critic|creative|moderator",
    "confidence": "high|medium|low",
    "domain": "the subject domain (e.g., finance, health, education)",
    "capabilities": ["list", "of", "required", "capabilities"],
    "context": "additional context for agent creation",
    "reasoning": "why you chose this agent type",
    "alternative_types": ["other", "possible", "types"]
}}

Be precise and only return valid JSON."""

        user_prompt = f"Analyze this request and determine the best agent type: {prompt}"

        try:
            # Get a provider manager for system-level operations
            provider_manager = await self.llm_service.get_provider_manager("system")

            generation_request = GenerationRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="gpt-3.5-turbo",
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500,
            )

            response = provider_manager.generate_with_fallback(generation_request)

            # Extract text content from response
            if hasattr(response, "content"):
                content = response.content
            elif hasattr(response, "text"):
                content = response.text
            else:
                content = str(response)

            # Parse JSON from response
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # Try to find JSON-like content
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))

                raise PromptAnalysisError(f"Could not parse JSON from LLM response: {content}")

        except Exception as e:
            logger.error(f"LLM prompt analysis failed: {e}")
            raise PromptAnalysisError(f"LLM analysis failed: {str(e)}")

    def _parse_analysis_result(
        self, analysis_json: Dict[str, Any], original_prompt: str, processed_prompt: str
    ) -> PromptAnalysisResult:
        """Parse LLM analysis result into structured format."""

        try:
            # Parse agent type
            agent_type_str = analysis_json.get("agent_type", "analyst").lower()
            agent_type = AgentType(agent_type_str)
        except ValueError:
            logger.warning(f"Unknown agent type: {agent_type_str}, defaulting to ANALYST")
            agent_type = AgentType.ANALYST

        # Parse confidence
        confidence_str = analysis_json.get("confidence", "medium").lower()
        try:
            confidence = AnalysisConfidence(confidence_str)
        except ValueError:
            confidence = AnalysisConfidence.MEDIUM

        # Parse alternative types
        alternative_types = []
        for alt_type_str in analysis_json.get("alternative_types", []):
            try:
                alternative_types.append(AgentType(alt_type_str.lower()))
            except ValueError:
                continue  # Skip invalid types

        return PromptAnalysisResult(
            agent_type=agent_type,
            confidence=confidence,
            domain=analysis_json.get("domain"),
            capabilities=analysis_json.get("capabilities", []),
            context=analysis_json.get("context"),
            reasoning=analysis_json.get("reasoning"),
            alternative_types=alternative_types,
            original_prompt=original_prompt,
            processed_prompt=processed_prompt,
        )

    def _fallback_analyze_prompt(self, prompt: str) -> PromptAnalysisResult:
        """Rule-based fallback analysis when LLM fails."""

        prompt_lower = prompt.lower()

        # Simple keyword-based classification
        if any(
            word in prompt_lower for word in ["analyze", "data", "study", "research", "examine"]
        ):
            agent_type = AgentType.ANALYST
            confidence = AnalysisConfidence.MEDIUM
        elif any(
            word in prompt_lower for word in ["argue", "convince", "persuade", "support", "defend"]
        ):
            agent_type = AgentType.ADVOCATE
            confidence = AnalysisConfidence.MEDIUM
        elif any(
            word in prompt_lower for word in ["critique", "criticize", "problems", "flaws", "wrong"]
        ):
            agent_type = AgentType.CRITIC
            confidence = AnalysisConfidence.MEDIUM
        elif any(
            word in prompt_lower
            for word in ["create", "invent", "brainstorm", "innovative", "novel"]
        ):
            agent_type = AgentType.CREATIVE
            confidence = AnalysisConfidence.MEDIUM
        elif any(
            word in prompt_lower
            for word in ["moderate", "facilitate", "balance", "manage", "coordinate"]
        ):
            agent_type = AgentType.MODERATOR
            confidence = AnalysisConfidence.MEDIUM
        else:
            # Default to analyst
            agent_type = AgentType.ANALYST
            confidence = AnalysisConfidence.LOW

        return PromptAnalysisResult(
            agent_type=agent_type,
            confidence=confidence,
            original_prompt=prompt,
            processed_prompt=prompt,
            reasoning="Fallback rule-based analysis",
        )


class PersonalityGenerator(IPersonalityGenerator):
    """Generates personality profiles for agents using LLM and templates."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._default_personalities = self._create_default_personalities()

    async def generate_personality(
        self,
        agent_type: AgentType,
        context: Optional[str] = None,
        traits_hint: Optional[Dict[str, Any]] = None,
    ) -> PersonalityProfile:
        """Generate a personality profile for an agent."""

        try:
            # Use LLM to generate nuanced personality
            personality_json = await self._llm_generate_personality(
                agent_type, context, traits_hint
            )
            return self._parse_personality_result(personality_json)

        except Exception as e:
            logger.error(f"LLM personality generation failed: {e}")
            # Fall back to default personality with slight randomization
            return self._enhance_default_personality(agent_type, context)

    def get_default_personality(self, agent_type: AgentType) -> PersonalityProfile:
        """Get a default personality profile for an agent type."""
        return self._default_personalities.get(
            agent_type, self._default_personalities[AgentType.ANALYST]
        )

    def _create_default_personalities(self) -> Dict[AgentType, PersonalityProfile]:
        """Create default personality profiles for each agent type."""

        return {
            AgentType.ADVOCATE: PersonalityProfile(
                assertiveness=0.8,
                analytical_depth=0.6,
                creativity=0.5,
                empathy=0.7,
                skepticism=0.3,
                formality=0.6,
                verbosity=0.7,
                collaboration=0.6,
                speed=0.7,
                custom_traits={"persuasiveness": 0.9, "conviction": 0.8},
            ),
            AgentType.ANALYST: PersonalityProfile(
                assertiveness=0.5,
                analytical_depth=0.9,
                creativity=0.4,
                empathy=0.5,
                skepticism=0.7,
                formality=0.8,
                verbosity=0.8,
                collaboration=0.5,
                speed=0.4,
                custom_traits={"methodicalness": 0.9, "precision": 0.8},
            ),
            AgentType.CRITIC: PersonalityProfile(
                assertiveness=0.7,
                analytical_depth=0.8,
                creativity=0.3,
                empathy=0.3,
                skepticism=0.9,
                formality=0.7,
                verbosity=0.6,
                collaboration=0.3,
                speed=0.6,
                custom_traits={"scrutiny": 0.9, "directness": 0.8},
            ),
            AgentType.CREATIVE: PersonalityProfile(
                assertiveness=0.4,
                analytical_depth=0.4,
                creativity=0.9,
                empathy=0.6,
                skepticism=0.2,
                formality=0.3,
                verbosity=0.6,
                collaboration=0.7,
                speed=0.8,
                custom_traits={"imagination": 0.9, "openness": 0.9},
            ),
            AgentType.MODERATOR: PersonalityProfile(
                assertiveness=0.5,
                analytical_depth=0.6,
                creativity=0.5,
                empathy=0.9,
                skepticism=0.4,
                formality=0.7,
                verbosity=0.5,
                collaboration=0.9,
                speed=0.6,
                custom_traits={"diplomacy": 0.9, "fairness": 0.9},
            ),
        }

    async def _llm_generate_personality(
        self, agent_type: AgentType, context: Optional[str], traits_hint: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to generate personality traits."""

        system_prompt = """You are an expert in personality psychology and AI agent design.

Generate a personality profile for an AI agent with the following traits (all values 0.0 to 1.0):

- assertiveness: How direct and forceful (0.0 = passive, 1.0 = very assertive)
- analytical_depth: How deep the analysis goes (0.0 = surface, 1.0 = very deep)
- creativity: How creative and novel the thinking (0.0 = conventional, 1.0 = highly creative)
- empathy: How much they consider others' perspectives (0.0 = cold, 1.0 = very empathetic)
- skepticism: How much they question and doubt (0.0 = trusting, 1.0 = very skeptical)
- formality: Communication style (0.0 = casual, 1.0 = very formal)
- verbosity: How much detail they provide (0.0 = concise, 1.0 = very detailed)
- collaboration: How much they like working with others (0.0 = independent, 1.0 = team-focused)
- speed: How quickly they work (0.0 = deliberate, 1.0 = very fast)

Return only valid JSON in this exact format:
{
    "assertiveness": 0.7,
    "analytical_depth": 0.8,
    "creativity": 0.5,
    "empathy": 0.6,
    "skepticism": 0.4,
    "formality": 0.6,
    "verbosity": 0.7,
    "collaboration": 0.5,
    "speed": 0.6,
    "custom_traits": {"trait_name": 0.8}
}"""

        user_prompt = f"Generate personality traits for a {agent_type.value.upper()} agent."
        if context:
            user_prompt += f" Context: {context}"
        if traits_hint:
            user_prompt += f" Personality hints: {traits_hint}"

        try:
            provider_manager = await self.llm_service.get_provider_manager("system")

            generation_request = GenerationRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="gpt-3.5-turbo",
                temperature=0.6,  # Some creativity but still consistent
                max_tokens=300,
            )

            response = provider_manager.generate_with_fallback(generation_request)

            # Extract and parse JSON
            if hasattr(response, "content"):
                content = response.content
            elif hasattr(response, "text"):
                content = response.text
            else:
                content = str(response)

            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                raise PersonalityGenerationError(f"Could not parse JSON: {content}")

        except Exception as e:
            logger.error(f"LLM personality generation failed: {e}")
            raise PersonalityGenerationError(f"LLM generation failed: {str(e)}")

    def _parse_personality_result(self, personality_json: Dict[str, Any]) -> PersonalityProfile:
        """Parse LLM personality result into PersonalityProfile."""

        try:
            return PersonalityProfile(
                assertiveness=max(0.0, min(1.0, personality_json.get("assertiveness", 0.5))),
                analytical_depth=max(0.0, min(1.0, personality_json.get("analytical_depth", 0.5))),
                creativity=max(0.0, min(1.0, personality_json.get("creativity", 0.5))),
                empathy=max(0.0, min(1.0, personality_json.get("empathy", 0.5))),
                skepticism=max(0.0, min(1.0, personality_json.get("skepticism", 0.5))),
                formality=max(0.0, min(1.0, personality_json.get("formality", 0.5))),
                verbosity=max(0.0, min(1.0, personality_json.get("verbosity", 0.5))),
                collaboration=max(0.0, min(1.0, personality_json.get("collaboration", 0.5))),
                speed=max(0.0, min(1.0, personality_json.get("speed", 0.5))),
                custom_traits=personality_json.get("custom_traits", {}),
            )
        except Exception as e:
            logger.error(f"Failed to parse personality result: {e}")
            raise PersonalityGenerationError(f"Invalid personality data: {str(e)}")

    def _enhance_default_personality(
        self, agent_type: AgentType, context: Optional[str]
    ) -> PersonalityProfile:
        """Enhance default personality with slight variations based on context."""

        base_personality = self.get_default_personality(agent_type)

        if not context:
            return base_personality

        # Apply small variations based on context keywords
        context_lower = context.lower()
        modifications = {}

        if any(word in context_lower for word in ["urgent", "quick", "fast"]):
            modifications["speed"] = min(1.0, base_personality.speed + 0.2)

        if any(word in context_lower for word in ["detailed", "thorough", "comprehensive"]):
            modifications["verbosity"] = min(1.0, base_personality.verbosity + 0.2)
            modifications["analytical_depth"] = min(1.0, base_personality.analytical_depth + 0.1)

        if any(word in context_lower for word in ["collaborative", "team", "group"]):
            modifications["collaboration"] = min(1.0, base_personality.collaboration + 0.2)
            modifications["empathy"] = min(1.0, base_personality.empathy + 0.1)

        # Create new personality with modifications
        personality_dict = base_personality.to_dict()
        personality_dict.update(modifications)

        return PersonalityProfile.from_dict(personality_dict)


class SystemPromptBuilder(ISystemPromptBuilder):
    """Builds system prompts for agents using templates and LLM enhancement."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._prompt_templates = self._create_prompt_templates()

    async def build_system_prompt(
        self,
        agent_type: AgentType,
        personality: PersonalityProfile,
        context: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> str:
        """Build a system prompt for an agent."""

        try:
            # Use LLM to create a customized system prompt
            system_prompt = await self._llm_build_prompt(
                agent_type, personality, context, capabilities
            )
            return system_prompt

        except Exception as e:
            logger.error(f"LLM system prompt building failed: {e}")
            # Fall back to template-based approach
            return self._build_template_prompt(agent_type, personality, context, capabilities)

    def get_template_prompt(self, agent_type: AgentType) -> str:
        """Get a template system prompt for an agent type."""
        return self._prompt_templates.get(agent_type, self._prompt_templates[AgentType.ANALYST])

    def _create_prompt_templates(self) -> Dict[AgentType, str]:
        """Create base prompt templates for each agent type."""

        return {
            AgentType.ADVOCATE: """You are an Advocate agent - your role is to build compelling cases and argue for specific positions.

Core responsibilities:
- Present strong arguments supporting your assigned position
- Gather and present evidence that supports your viewpoint
- Address counterarguments proactively and persuasively
- Build consensus around your recommendations
- Communicate with conviction and clarity

Remember to be respectful while being assertive, and always ground your arguments in evidence and logic.""",
            AgentType.ANALYST: """You are an Analyst agent - your role is to break down complex problems and provide data-driven insights.

Core responsibilities:
- Systematically analyze problems and situations
- Gather relevant data and evidence
- Identify patterns, trends, and key insights
- Present findings in a clear, structured manner
- Provide objective, evidence-based recommendations

Approach every task methodically, consider multiple perspectives, and ensure your analysis is thorough and well-reasoned.""",
            AgentType.CRITIC: """You are a Critic agent - your role is to identify flaws, challenge assumptions, and ensure quality.

Core responsibilities:
- Scrutinize ideas, proposals, and solutions for weaknesses
- Challenge assumptions and question underlying premises
- Identify potential risks, problems, and failure points
- Provide constructive criticism that leads to improvement
- Maintain high standards and push for excellence

Be direct and honest in your critiques while remaining constructive. Your goal is to strengthen ideas through rigorous examination.""",
            AgentType.CREATIVE: """You are a Creative agent - your role is to generate novel ideas and innovative solutions.

Core responsibilities:
- Think outside conventional boundaries and explore new possibilities
- Generate multiple creative alternatives and approaches
- Make unexpected connections between different concepts
- Propose innovative solutions to complex problems
- Encourage imagination and open-minded exploration

Embrace unconventional thinking, be willing to take intellectual risks, and help others see problems from fresh perspectives.""",
            AgentType.MODERATOR: """You are a Moderator agent - your role is to facilitate discussions and maintain balanced perspectives.

Core responsibilities:
- Facilitate productive discussions and ensure all voices are heard
- Maintain neutrality and balance competing viewpoints
- Guide conversations toward constructive outcomes
- Resolve conflicts and find common ground
- Ensure fair participation and respectful dialogue

Stay diplomatic and fair-minded, help others communicate effectively, and work toward consensus and mutual understanding.""",
        }

    async def _llm_build_prompt(
        self,
        agent_type: AgentType,
        personality: PersonalityProfile,
        context: Optional[str],
        capabilities: Optional[List[str]],
    ) -> str:
        """Use LLM to build a customized system prompt."""

        system_prompt = """You are an expert at creating system prompts for AI agents.

Create a system prompt that defines the agent's role, personality, and behavior. The prompt should:
1. Clearly define the agent's primary role and responsibilities
2. Reflect the personality traits provided
3. Include specific behavioral guidelines
4. Be engaging and clear
5. Be 2-4 paragraphs long

Make the prompt professional but personalized to the specific agent type and personality."""

        # Build personality description
        personality_desc = self._describe_personality(personality)

        user_prompt = f"""Create a system prompt for a {agent_type.value.upper()} agent with this personality:

{personality_desc}"""

        if context:
            user_prompt += f"\n\nContext: {context}"

        if capabilities:
            user_prompt += f"\n\nRequired capabilities: {', '.join(capabilities)}"

        try:
            provider_manager = await self.llm_service.get_provider_manager("system")

            generation_request = GenerationRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="gpt-3.5-turbo",
                temperature=0.7,  # Some creativity for engaging prompts
                max_tokens=500,
            )

            response = provider_manager.generate_with_fallback(generation_request)

            if hasattr(response, "content"):
                content = response.content
            elif hasattr(response, "text"):
                content = response.text
            else:
                content = str(response)

            return content.strip()

        except Exception as e:
            logger.error(f"LLM system prompt building failed: {e}")
            raise SystemPromptBuildError(f"LLM prompt building failed: {str(e)}")

    def _build_template_prompt(
        self,
        agent_type: AgentType,
        personality: PersonalityProfile,
        context: Optional[str],
        capabilities: Optional[List[str]],
    ) -> str:
        """Build system prompt using templates with personality integration."""

        base_template = self.get_template_prompt(agent_type)

        # Add personality-based modifications
        personality_additions = []

        if personality.assertiveness > 0.7:
            personality_additions.append("Be direct and confident in your communications.")
        elif personality.assertiveness < 0.3:
            personality_additions.append("Take a gentle and collaborative approach.")

        if personality.formality > 0.7:
            personality_additions.append("Maintain a professional and formal tone.")
        elif personality.formality < 0.3:
            personality_additions.append("Use a conversational and approachable tone.")

        if personality.verbosity > 0.7:
            personality_additions.append(
                "Provide detailed explanations and comprehensive coverage."
            )
        elif personality.verbosity < 0.3:
            personality_additions.append("Be concise and get straight to the point.")

        # Build the complete prompt
        complete_prompt = base_template

        if personality_additions:
            complete_prompt += "\n\nPersonality guidelines:\n- " + "\n- ".join(
                personality_additions
            )

        if context:
            complete_prompt += f"\n\nSpecial context: {context}"

        if capabilities:
            complete_prompt += f"\n\nKey capabilities to demonstrate: {', '.join(capabilities)}"

        return complete_prompt

    def _describe_personality(self, personality: PersonalityProfile) -> str:
        """Create a human-readable description of personality traits."""

        traits = []

        if personality.assertiveness > 0.7:
            traits.append("highly assertive and direct")
        elif personality.assertiveness < 0.3:
            traits.append("gentle and collaborative")
        else:
            traits.append("balanced in assertiveness")

        if personality.analytical_depth > 0.7:
            traits.append("very thorough and analytical")
        elif personality.analytical_depth < 0.3:
            traits.append("focused on high-level concepts")
        else:
            traits.append("appropriately detailed")

        if personality.creativity > 0.7:
            traits.append("highly creative and innovative")
        elif personality.creativity < 0.3:
            traits.append("practical and conventional")
        else:
            traits.append("moderately creative")

        if personality.empathy > 0.7:
            traits.append("very empathetic and considerate")
        elif personality.empathy < 0.3:
            traits.append("focused on objective analysis")
        else:
            traits.append("balanced in empathy")

        if personality.formality > 0.7:
            traits.append("formal and professional")
        elif personality.formality < 0.3:
            traits.append("casual and approachable")
        else:
            traits.append("appropriately formal")

        return f"This agent is {', '.join(traits)}."


class AgentBuilder(IAgentBuilder):
    """Builds and persists agent instances to the database."""

    async def build_agent(self, specification: AgentSpecification) -> Agent:
        """Build and persist an agent from a specification."""

        try:
            for session in get_db():
                # Create new agent instance
                agent = Agent(
                    name=specification.name,
                    template=specification.template,
                    status=AgentStatus.PENDING,
                    agent_type=specification.agent_type,
                    system_prompt=specification.system_prompt,
                    personality_traits=specification.personality.to_dict(),
                    creation_source=specification.creation_source,
                    source_prompt=specification.source_prompt,
                    parameters=specification.parameters,
                )

                # Add to session and commit
                session.add(agent)
                session.commit()
                session.refresh(agent)

                logger.info(
                    f"Created agent {agent.id} ({agent.name}) of type {agent.agent_type.value}"
                )
                return agent

        except Exception as e:
            logger.error(f"Failed to build agent: {e}")
            raise AgentBuildError(f"Failed to create agent: {str(e)}")

    async def update_agent(self, agent_id: str, specification: AgentSpecification) -> Agent:
        """Update an existing agent with new specification."""

        try:
            for session in get_db():
                # Find existing agent
                agent = session.get(Agent, agent_id)
                if not agent:
                    raise AgentBuildError(f"Agent {agent_id} not found")

                # Update fields
                agent.name = specification.name
                agent.agent_type = specification.agent_type
                agent.system_prompt = specification.system_prompt
                agent.personality_traits = specification.personality.to_dict()
                agent.parameters = specification.parameters

                session.commit()
                session.refresh(agent)

                logger.info(f"Updated agent {agent.id} ({agent.name})")
                return agent

        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            raise AgentBuildError(f"Failed to update agent: {str(e)}")
