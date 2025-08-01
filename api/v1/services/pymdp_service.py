"""
PyMDP Service for Agent Conversation API

Provides dependency injection service for initializing PyMDP agents with parsed GMN.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class PyMDPService:
    """Service for initializing PyMDP agents with parsed GMN specifications."""

    def __init__(self):
        """Initialize PyMDP service."""
        self._active_agents: Dict[str, Any] = {}

    def initialize_pymdp_agent(
        self, agent_id: str, gmn_spec: Dict[str, Any], simplified_mode: bool = True
    ) -> Dict[str, Any]:
        """Initialize PyMDP agent from GMN specification."""

        try:
            logger.info(f"Initializing PyMDP agent {agent_id} from GMN")

            if simplified_mode or gmn_spec.get("conversation_mode", False):
                # Use simplified initialization for conversation agents
                return self._initialize_conversation_agent(agent_id, gmn_spec)
            else:
                # Use full PyMDP initialization for complex agents
                return self._initialize_full_pymdp_agent(agent_id, gmn_spec)

        except Exception as e:
            logger.error(f"Failed to initialize PyMDP agent {agent_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"PyMDP agent initialization failed: {str(e)}"
            )

    def _initialize_conversation_agent(
        self, agent_id: str, gmn_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize simplified agent for conversation mode."""

        agent_config = {
            "agent_id": agent_id,
            "name": gmn_spec.get("name", f"agent_{agent_id}"),
            "role": gmn_spec.get("role", "participant"),
            "personality": gmn_spec.get("personality", "neutral"),
            "system_prompt": gmn_spec.get("system_prompt", "You are a helpful assistant."),
            "conversation_mode": True,
            "pymdp_enabled": False,  # Simplified mode doesn't use full PyMDP
            "states": gmn_spec.get("states", ["active"]),
            "actions": gmn_spec.get("actions", ["respond"]),
            "current_state": "listening",
            "initialization_status": "ready",
        }

        # Store agent configuration
        self._active_agents[agent_id] = agent_config

        logger.info(f"Initialized conversation agent {agent_id} ({agent_config['name']})")
        return agent_config

    def _initialize_full_pymdp_agent(
        self, agent_id: str, gmn_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize full PyMDP agent with Active Inference."""

        try:
            # Import PyMDP components (conditional import for resilience)
            try:
                from agents.agent_manager import AgentManager
                from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp

                # Convert GMN to PyMDP format
                pymdp_model = adapt_gmn_to_pymdp(gmn_spec)

                # Create agent using agent manager
                agent_manager = AgentManager()
                agent = agent_manager.create_agent(
                    agent_id=agent_id,
                    name=gmn_spec.get("name", f"agent_{agent_id}"),
                    gmn_config=gmn_spec,
                )

                if not agent:
                    raise ValueError("Agent manager failed to create agent")

                agent_config = {
                    "agent_id": agent_id,
                    "name": gmn_spec.get("name"),
                    "pymdp_model": pymdp_model,
                    "pymdp_enabled": True,
                    "agent_instance": agent,
                    "initialization_status": "ready",
                }

                # Store agent
                self._active_agents[agent_id] = agent_config

                logger.info(f"Initialized full PyMDP agent {agent_id}")
                return agent_config

            except ImportError as e:
                logger.warning(f"PyMDP components not available: {e}")
                # Fall back to conversation mode
                return self._initialize_conversation_agent(agent_id, gmn_spec)

        except Exception as e:
            logger.error(f"Full PyMDP initialization failed: {e}")
            # Fall back to conversation mode
            logger.info(f"Falling back to conversation mode for agent {agent_id}")
            return self._initialize_conversation_agent(agent_id, gmn_spec)

    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for an active agent."""
        return self._active_agents.get(agent_id)

    def update_agent_state(self, agent_id: str, new_state: str) -> bool:
        """Update agent's current state."""

        if agent_id in self._active_agents:
            self._active_agents[agent_id]["current_state"] = new_state
            logger.info(f"Updated agent {agent_id} state to: {new_state}")
            return True
        return False

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from active agents."""

        if agent_id in self._active_agents:
            # Clean up agent instance if it exists
            agent_config = self._active_agents[agent_id]
            if "agent_instance" in agent_config:
                try:
                    # Stop agent if it has a stop method
                    agent_instance = agent_config["agent_instance"]
                    if hasattr(agent_instance, "stop"):
                        agent_instance.stop()
                except Exception as e:
                    logger.warning(f"Error stopping agent {agent_id}: {e}")

            del self._active_agents[agent_id]
            logger.info(f"Removed agent {agent_id}")
            return True
        return False

    def list_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all active agents."""
        return {
            agent_id: {
                "name": config.get("name"),
                "role": config.get("role"),
                "status": config.get("initialization_status"),
                "conversation_mode": config.get("conversation_mode", False),
                "pymdp_enabled": config.get("pymdp_enabled", False),
            }
            for agent_id, config in self._active_agents.items()
        }

    def get_agent_beliefs(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current beliefs for an agent (if PyMDP enabled)."""

        agent_config = self.get_agent_config(agent_id)
        if not agent_config or not agent_config.get("pymdp_enabled"):
            return None

        try:
            agent_instance = agent_config.get("agent_instance")
            if agent_instance and hasattr(agent_instance, "get_beliefs"):
                return agent_instance.get_beliefs()
        except Exception as e:
            logger.warning(f"Failed to get beliefs for agent {agent_id}: {e}")

        return None

    async def get_multi_factor_beliefs(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get multi-factor beliefs for an agent using advanced belief management.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary containing factorized belief state information or None
        """
        agent_config = self.get_agent_config(agent_id)
        if not agent_config or not agent_config.get("pymdp_enabled"):
            return None

        try:
            from services.multi_factor_belief_manager import create_multi_factor_belief_manager

            agent_instance = agent_config.get("agent_instance")
            if not agent_instance:
                return None

            # Create multi-factor belief manager components
            extractor, updater, validator = create_multi_factor_belief_manager()

            # Extract factorized beliefs
            belief_state = await extractor.extract_factorized_beliefs(
                agent_instance, agent_id, include_correlations=True
            )

            # Validate consistency
            validation = await validator.validate_factorized_state(belief_state)

            # Convert to serializable format
            result = {
                "agent_id": belief_state.agent_id,
                "timestamp": belief_state.timestamp.isoformat(),
                "num_factors": belief_state.num_factors,
                "overall_entropy": belief_state.overall_entropy,
                "overall_confidence": belief_state.overall_confidence,
                "factors": [
                    {
                        "index": factor.index,
                        "name": factor.name,
                        "beliefs": factor.beliefs.tolist(),
                        "factor_type": factor.factor_type.value,
                        "entropy": factor.entropy,
                        "confidence": factor.confidence,
                        "most_likely_state": factor.most_likely_state,
                        "dependencies": [
                            {
                                "parent_factor": dep.parent_factor,
                                "child_factor": dep.child_factor,
                                "dependency_type": dep.dependency_type,
                                "strength": dep.strength,
                            }
                            for dep in factor.dependencies
                        ],
                        "metadata": factor.metadata,
                    }
                    for factor in belief_state.factors
                ],
                "correlations": [
                    {
                        "factor_a": corr.factor_a,
                        "factor_b": corr.factor_b,
                        "correlation_strength": corr.correlation_strength,
                        "timestamp": corr.timestamp.isoformat(),
                    }
                    for corr in belief_state.correlations
                ],
                "hierarchy": belief_state.hierarchy,
                "validation": validation,
                "metadata": belief_state.metadata,
            }

            logger.info(
                f"Retrieved multi-factor beliefs for agent {agent_id}: "
                f"{belief_state.num_factors} factors, "
                f"{len(belief_state.correlations)} correlations"
            )

            return result

        except ImportError as e:
            logger.warning(f"Multi-factor belief manager not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get multi-factor beliefs for agent {agent_id}: {e}")
            return None

    async def update_agent_beliefs(
        self, agent_id: str, observations: Optional[Dict[int, int]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update agent beliefs using hierarchical belief propagation.

        Args:
            agent_id: Agent identifier
            observations: Optional observations for specific factors {factor_index: state_value}

        Returns:
            Updated belief state information or None
        """
        agent_config = self.get_agent_config(agent_id)
        if not agent_config or not agent_config.get("pymdp_enabled"):
            return None

        try:
            from services.multi_factor_belief_manager import create_multi_factor_belief_manager

            agent_instance = agent_config.get("agent_instance")
            if not agent_instance:
                return None

            # Create multi-factor belief manager components
            extractor, updater, validator = create_multi_factor_belief_manager()

            # Extract current beliefs
            current_state = await extractor.extract_factorized_beliefs(
                agent_instance, agent_id, include_correlations=True
            )

            # Propagate beliefs with observations
            updated_state = await updater.propagate_beliefs(current_state, observations)

            # Validate updated state
            validation = await validator.validate_factorized_state(updated_state)

            if not validation["is_valid"]:
                logger.warning(
                    f"Belief update validation failed for agent {agent_id}: "
                    f"{validation['errors']}"
                )

            # Convert to serializable format (similar to get_multi_factor_beliefs)
            result = {
                "agent_id": updated_state.agent_id,
                "timestamp": updated_state.timestamp.isoformat(),
                "num_factors": updated_state.num_factors,
                "overall_entropy": updated_state.overall_entropy,
                "overall_confidence": updated_state.overall_confidence,
                "observations_applied": observations if observations else {},
                "propagation_info": {
                    "iterations": updated_state.metadata.get("propagation_iterations", 0),
                    "converged": updated_state.metadata.get("converged", False),
                    "observations_applied": updated_state.metadata.get("observations_applied", 0),
                },
                "validation": validation,
                "factors": [
                    {
                        "index": factor.index,
                        "name": factor.name,
                        "beliefs": factor.beliefs.tolist(),
                        "factor_type": factor.factor_type.value,
                        "entropy": factor.entropy,
                        "confidence": factor.confidence,
                        "most_likely_state": factor.most_likely_state,
                    }
                    for factor in updated_state.factors
                ],
            }

            logger.info(
                f"Updated beliefs for agent {agent_id}: "
                f"{updated_state.metadata.get('propagation_iterations', 0)} iterations, "
                f"converged: {updated_state.metadata.get('converged', False)}"
            )

            return result

        except ImportError as e:
            logger.warning(f"Multi-factor belief manager not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to update beliefs for agent {agent_id}: {e}")
            return None


# Dependency injection factory function
def get_pymdp_service() -> PyMDPService:
    """Factory function for FastAPI dependency injection."""
    return PyMDPService()
