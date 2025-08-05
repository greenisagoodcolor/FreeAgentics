"""Enhanced agent builder with PyMDP integration.

This builder extends the base agent creation system to support Active Inference
agents with real PyMDP implementations.
"""

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from core.providers import get_db
from database.models import Agent, AgentStatus, AgentType
from agents.pymdp_agent_factory import PyMDPAgentFactory, PyMDPAgentCreationError
from agents.base_agent import BasicExplorerAgent
from inference.active.gmn_parser import GMNParser
from inference.llm.provider_factory import get_provider_factory

from .interfaces import IAgentBuilder
from .models import AgentBuildError, AgentSpecification

logger = logging.getLogger(__name__)


class PyMDPAgentBuilder(IAgentBuilder):
    """Agent builder with PyMDP Active Inference support.

    This builder can create both traditional agents and PyMDP-based Active Inference
    agents, handling the full pipeline from GMN specifications to database storage.
    """

    def __init__(self):
        """Initialize the PyMDP agent builder."""
        self.pymdp_factory = PyMDPAgentFactory()
        self.gmn_parser = GMNParser()

        # Metrics for observability
        self._metrics = {
            "pymdp_agents_created": 0,
            "traditional_agents_created": 0,
            "gmn_generations": 0,
            "creation_failures": 0,
        }

        logger.info("PyMDP Agent Builder initialized")

    async def build_agent(self, specification: AgentSpecification) -> Agent:
        """Build and persist an agent from a specification.

        This method handles both traditional agents and PyMDP Active Inference agents.
        For Active Inference agents, it generates GMN specifications and creates
        real PyMDP agent instances.

        Args:
            specification: Agent specification with type, prompt, etc.

        Returns:
            Agent: Persisted agent instance

        Raises:
            AgentBuildError: If agent creation fails
        """
        try:
            # Determine if this should be an Active Inference agent
            should_use_pymdp = self._should_create_pymdp_agent(specification)

            if should_use_pymdp:
                return await self._build_pymdp_agent(specification)
            else:
                return await self._build_traditional_agent(specification)

        except Exception as e:
            self._metrics["creation_failures"] += 1
            logger.error(f"Agent creation failed: {e}")
            raise AgentBuildError(f"Failed to create agent: {str(e)}")

    async def update_agent(self, agent_id: str, specification: AgentSpecification) -> Agent:
        """Update an existing agent with new specification."""
        try:
            for session in get_db():
                # Find existing agent
                agent = session.get(Agent, agent_id)
                if not agent:
                    raise AgentBuildError(f"Agent {agent_id} not found")

                # Update basic fields
                agent.name = specification.name
                agent.agent_type = specification.agent_type
                agent.system_prompt = specification.system_prompt
                agent.personality_traits = specification.personality.to_dict()
                agent.parameters = specification.parameters
                agent.updated_at = datetime.utcnow()

                # If this is now a PyMDP agent, generate new PyMDP spec
                if self._should_create_pymdp_agent(specification):
                    gmn_spec = await self._generate_gmn_specification(specification)
                    if gmn_spec:
                        agent.parameters = agent.parameters or {}
                        agent.parameters["gmn_spec"] = gmn_spec
                        agent.parameters["agent_type"] = "pymdp_active_inference"

                session.commit()
                session.refresh(agent)

                logger.info(f"Updated agent {agent.id} ({agent.name})")
                return agent

        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            raise AgentBuildError(f"Failed to update agent: {str(e)}")

    def _should_create_pymdp_agent(self, specification: AgentSpecification) -> bool:
        """Determine if agent should use PyMDP Active Inference.

        Args:
            specification: Agent specification

        Returns:
            bool: True if should create PyMDP agent
        """
        # Check if explicitly requested PyMDP
        if specification.parameters and specification.parameters.get("use_pymdp", False):
            return True

        # Check if source prompt indicates Active Inference use case
        if specification.source_prompt:
            ai_keywords = [
                "active inference",
                "pymdp",
                "belief",
                "uncertainty",
                "exploration",
                "curiosity",
                "free energy",
                "prediction",
                "environment",
                "state",
                "observation",
                "action",
                "policy",
            ]

            prompt_lower = specification.source_prompt.lower()
            keyword_count = sum(1 for keyword in ai_keywords if keyword in prompt_lower)

            # If multiple AI keywords present, use PyMDP
            if keyword_count >= 2:
                return True

        # Check agent type - ANALYST agents often benefit from Active Inference
        if specification.agent_type == AgentType.ANALYST:
            return True

        return False

    async def _build_pymdp_agent(self, specification: AgentSpecification) -> Agent:
        """Build a PyMDP Active Inference agent."""
        logger.info(f"Creating PyMDP Active Inference agent: {specification.name}")

        try:
            # Step 1: Generate GMN specification from agent requirements
            gmn_spec = await self._generate_gmn_specification(specification)

            if not gmn_spec:
                # Fallback to traditional agent if GMN generation fails
                logger.warning("GMN generation failed, falling back to traditional agent")
                return await self._build_traditional_agent(specification)

            # Step 2: Create PyMDP agent from GMN specification
            try:
                pymdp_agent = self.pymdp_factory.create_agent(gmn_spec)
                logger.info("PyMDP agent created successfully")
            except PyMDPAgentCreationError as e:
                logger.warning(f"PyMDP creation failed: {e}, falling back to traditional agent")
                return await self._build_traditional_agent(specification)

            # Step 3: Store agent in database with PyMDP metadata
            for session in get_db():
                agent = Agent(
                    name=specification.name,
                    template=specification.template,
                    status=AgentStatus.ACTIVE,  # PyMDP agents start active
                    agent_type=specification.agent_type,
                    system_prompt=specification.system_prompt,
                    personality_traits=specification.personality.to_dict(),
                    creation_source=specification.creation_source,
                    source_prompt=specification.source_prompt,
                    parameters={
                        **(specification.parameters or {}),
                        "agent_type": "pymdp_active_inference",
                        "gmn_spec": gmn_spec,
                        "pymdp_version": "0.0.7.1",
                        "factory_metrics": self.pymdp_factory.get_metrics(),
                    },
                )

                session.add(agent)
                session.commit()
                session.refresh(agent)

                self._metrics["pymdp_agents_created"] += 1

                logger.info(f"Created PyMDP agent {agent.id} ({agent.name}) with GMN specification")
                return agent

        except Exception as e:
            logger.error(f"PyMDP agent creation failed: {e}")
            raise AgentBuildError(f"Failed to create PyMDP agent: {str(e)}")

    async def _build_traditional_agent(self, specification: AgentSpecification) -> Agent:
        """Build a traditional (non-PyMDP) agent."""
        logger.info(f"Creating traditional agent: {specification.name}")

        try:
            for session in get_db():
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

                session.add(agent)
                session.commit()
                session.refresh(agent)

                self._metrics["traditional_agents_created"] += 1

                logger.info(
                    f"Created traditional agent {agent.id} ({agent.name}) of type {agent.agent_type.value}"
                )
                return agent

        except Exception as e:
            logger.error(f"Traditional agent creation failed: {e}")
            raise AgentBuildError(f"Failed to create traditional agent: {str(e)}")

    async def _generate_gmn_specification(
        self, specification: AgentSpecification
    ) -> Optional[Dict[str, Any]]:
        """Generate GMN specification using LLM.

        Args:
            specification: Agent specification

        Returns:
            Optional[Dict]: GMN specification or None if generation fails
        """
        try:
            # Get LLM provider for GMN generation
            factory = get_provider_factory()
            provider = factory.create_configured_provider()

            # Create prompt for GMN generation
            prompt = self._create_gmn_generation_prompt(specification)

            # Generate GMN specification
            response = await provider.generate_text(prompt, max_tokens=1000)

            if not response or not response.strip():
                logger.error("LLM returned empty response for GMN generation")
                return None

            # Parse the generated GMN
            try:
                # First try to parse as GMN text format
                graph = self.gmn_parser.parse(response)
                gmn_spec = self.gmn_parser.to_pymdp_model(graph)

                self._metrics["gmn_generations"] += 1
                logger.info("GMN specification generated successfully")
                return gmn_spec

            except Exception as parse_error:
                logger.error(f"Failed to parse generated GMN: {parse_error}")

                # Fallback: try to extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    try:
                        gmn_spec = json.loads(json_match.group())
                        # Validate it has required fields
                        required_fields = [
                            "num_states",
                            "num_obs",
                            "num_actions",
                            "A",
                            "B",
                            "C",
                            "D",
                        ]
                        if all(field in gmn_spec for field in required_fields):
                            self._metrics["gmn_generations"] += 1
                            logger.info("GMN specification extracted from JSON")
                            return gmn_spec
                    except json.JSONDecodeError:
                        pass

                return None

        except Exception as e:
            logger.error(f"GMN generation failed: {e}")
            return None

    def _create_gmn_generation_prompt(self, specification: AgentSpecification) -> str:
        """Create LLM prompt for GMN generation.

        Args:
            specification: Agent specification

        Returns:
            str: LLM prompt for generating GMN
        """
        prompt = f"""Generate a GMN (Generalized Model Notation) specification for an Active Inference agent.

Agent Details:
- Name: {specification.name}
- Type: {specification.agent_type.value}
- Source Prompt: {specification.source_prompt or 'N/A'}
- Personality: {specification.personality.to_dict() if specification.personality else 'N/A'}

Create a GMN specification in the following format:

[nodes]
location: state {{num_states: 4}}
obs_location: observation {{num_observations: 4}}
move: action {{num_actions: 4}}
location_belief: belief
location_pref: preference {{preferred_observation: 0}}
location_likelihood: likelihood
location_transition: transition

[edges]
location -> location_likelihood: depends_on
location_likelihood -> obs_location: generates
location -> location_transition: depends_on
move -> location_transition: depends_on
location_pref -> obs_location: depends_on
location_belief -> location: depends_on

The specification should model the agent's environment and decision-making process.
Consider the agent's role and create appropriate states, observations, and actions.
Keep it simple with 3-5 states, 3-5 observations, and 3-5 actions maximum.

Generate only the GMN specification, no additional text."""

        return prompt

    def get_metrics(self) -> Dict[str, Any]:
        """Get builder metrics for monitoring.

        Returns:
            Dict containing creation metrics
        """
        metrics = self._metrics.copy()

        # Add PyMDP factory metrics
        if hasattr(self, "pymdp_factory"):
            factory_metrics = self.pymdp_factory.get_metrics()
            metrics.update({f"factory_{k}": v for k, v in factory_metrics.items()})

        # Add computed metrics
        total_created = metrics["pymdp_agents_created"] + metrics["traditional_agents_created"]
        if total_created > 0:
            metrics["pymdp_ratio"] = metrics["pymdp_agents_created"] / total_created
        else:
            metrics["pymdp_ratio"] = 0.0

        return metrics

    async def create_runtime_agent(self, agent_record: Agent) -> Optional[BasicExplorerAgent]:
        """Create a runtime PyMDP agent instance from database record.

        Args:
            agent_record: Database agent record

        Returns:
            Optional[BasicExplorerAgent]: Runtime agent instance or None
        """
        try:
            # Check if this is a PyMDP agent
            if (
                not agent_record.parameters
                or agent_record.parameters.get("agent_type") != "pymdp_active_inference"
            ):
                return None

            # Extract GMN specification
            gmn_spec = agent_record.parameters.get("gmn_spec")
            if not gmn_spec:
                logger.error(f"Agent {agent_record.id} missing GMN specification")
                return None

            # Create runtime PyMDP agent
            runtime_agent = BasicExplorerAgent(
                agent_id=str(agent_record.id),
                name=agent_record.name,
                grid_size=4,  # Default grid size, could be extracted from GMN
            )

            # Load the GMN specification into the runtime agent
            try:
                # Convert GMN spec back to text format for loading
                gmn_text = self._gmn_spec_to_text(gmn_spec)
                runtime_agent.load_gmn_spec(gmn_text)

                logger.info(f"Created runtime PyMDP agent for {agent_record.name}")
                return runtime_agent

            except Exception as e:
                logger.error(f"Failed to load GMN spec into runtime agent: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to create runtime agent: {e}")
            return None

    def _gmn_spec_to_text(self, gmn_spec: Dict[str, Any]) -> str:
        """Convert GMN specification dict back to text format.

        This is a simplified conversion - in production, you'd want more sophisticated
        serialization that preserves all the GMN graph structure.

        Args:
            gmn_spec: GMN specification dictionary

        Returns:
            str: GMN text format
        """
        # For now, return a simple grid world GMN
        # TODO: Implement proper GMN dict -> text conversion
        return """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}
        move: action {num_actions: 4}
        location_belief: belief
        location_pref: preference {preferred_observation: 0}
        location_likelihood: likelihood
        location_transition: transition

        [edges]
        location -> location_likelihood: depends_on
        location_likelihood -> obs_location: generates
        location -> location_transition: depends_on
        move -> location_transition: depends_on
        location_pref -> obs_location: depends_on
        location_belief -> location: depends_on
        """
