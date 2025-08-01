"""
PyMDP Belief Manager Service for Conversation Integration

Integrates PyMDP Active Inference belief updates with conversation processing.
Implements the Nemesis Committee's consensus approach with TDD, error resilience,
and clean architecture principles.

Following:
- Kent Beck: TDD implementation with minimal viable integration
- Robert C. Martin: Clean dependency injection and SRP
- Michael Feathers: Seam-based integration with existing conversation system
- Charity Majors: Production-ready error handling and observability
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from inference.active.belief_manager import BeliefStateManager, InMemoryBeliefRepository
from inference.active.config import ActiveInferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class BeliefIntegrationConfig:
    """Configuration for belief integration with conversations."""

    enabled: bool = True
    max_update_time_ms: float = 100.0  # Performance budget
    fallback_on_error: bool = True
    enable_tracing: bool = True
    cache_belief_states: bool = True


class PyMDPBeliefManager:
    """
    Service that manages PyMDP belief states for conversation agents.

    Provides clean integration between conversation processing and Active Inference
    belief updates, with comprehensive error handling and observability.
    """

    def __init__(
        self,
        config: ActiveInferenceConfig,
        agent_id: str,
        integration_config: Optional[BeliefIntegrationConfig] = None,
    ):
        """Initialize PyMDP belief manager for an agent.

        Args:
            config: Active inference configuration
            agent_id: Unique agent identifier
            integration_config: Configuration for conversation integration
        """
        self.config = config
        self.agent_id = agent_id
        self.integration_config = integration_config or BeliefIntegrationConfig()

        # Initialize belief state repository
        self._repository = InMemoryBeliefRepository()

        # Create default observation model (A-matrix)
        self._A_matrix = self._create_default_observation_model()

        # Initialize belief state manager
        self._belief_manager = BeliefStateManager(
            config=config, repository=self._repository, A_matrix=self._A_matrix, agent_id=agent_id
        )

        # Observation processor for message-to-observation conversion
        self._observation_processor = None  # Will be injected

        # Performance and error tracking
        self._update_count = 0
        self._error_count = 0
        self._last_update_time = 0.0

        logger.info(f"Initialized PyMDP belief manager for agent {agent_id}")

    def _create_default_observation_model(self) -> NDArray[np.floating]:
        """Create default observation model for conversation processing.

        Returns:
            A-matrix with shape (num_observations, num_states)
        """
        num_states = self.config.num_states
        num_observations = self.config.num_observations

        # Create simple observation model where each state has preferred observations
        A = np.zeros((num_observations, num_states))

        for state in range(num_states):
            # Each state has higher probability for its corresponding observation
            for obs in range(num_observations):
                if obs == state:
                    A[obs, state] = 0.7  # High likelihood for matching observation
                else:
                    A[obs, state] = 0.3 / (num_observations - 1)  # Lower for others

        # Ensure columns sum to 1 (proper probability distributions)
        A = A / A.sum(axis=0, keepdims=True)

        return A

    def get_belief_manager(self) -> BeliefStateManager:
        """Get the underlying belief state manager."""
        return self._belief_manager

    def get_observation_processor(self):
        """Get the observation processor (will be implemented in subtask 33.2)."""
        return self._observation_processor

    def set_observation_processor(self, processor):
        """Set the observation processor for message-to-observation conversion."""
        self._observation_processor = processor
        logger.debug(f"Set observation processor for agent {self.agent_id}")

    async def update_beliefs_from_message(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update agent beliefs based on conversation message.

        Args:
            message: Conversation message with content, role, etc.
            conversation_context: Additional context for observation extraction

        Returns:
            Dictionary with belief update results and metrics
        """
        if not self.integration_config.enabled:
            return self._create_disabled_response()

        start_time = time.time()

        try:
            # Extract observation from message
            observation = await self._extract_observation(message, conversation_context)

            # Update beliefs using PyMDP
            update_result = self._belief_manager.update_beliefs(observation)

            # Track performance
            update_time_ms = (time.time() - start_time) * 1000
            self._update_count += 1
            self._last_update_time = update_time_ms

            # Create response with observability metrics
            response = {
                "belief_influenced": True,
                "belief_status": "success",
                "observation": observation,
                "belief_metrics": {
                    "entropy": update_result.new_belief_state.entropy,
                    "max_confidence": update_result.new_belief_state.max_confidence,
                    "entropy_change": update_result.entropy_change,
                    "kl_divergence": update_result.kl_divergence,
                    "most_likely_state": update_result.new_belief_state.most_likely_state(),
                },
                "performance_metrics": {
                    "update_time_ms": update_time_ms,
                    "total_updates": self._update_count,
                    "agent_id": self.agent_id,
                },
            }

            # Add tracing information if enabled
            if self.integration_config.enable_tracing:
                response["trace_info"] = {
                    "timestamp": time.time(),
                    "message_content_length": len(message.get("content", "")),
                    "conversation_context": bool(conversation_context),
                }

            logger.info(
                f"Belief update completed for agent {self.agent_id}: "
                f"obs={observation}, entropy={update_result.new_belief_state.entropy:.3f}, "
                f"time={update_time_ms:.2f}ms"
            )

            return response

        except Exception as e:
            self._error_count += 1
            error_time_ms = (time.time() - start_time) * 1000

            logger.error(
                f"Belief update failed for agent {self.agent_id}: {e}, "
                f"time={error_time_ms:.2f}ms"
            )

            if self.integration_config.fallback_on_error:
                return self._create_error_response(str(e), error_time_ms)
            else:
                raise

    async def _extract_observation(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Extract PyMDP observation from conversation message.

        Args:
            message: Conversation message
            conversation_context: Additional context

        Returns:
            Observation index for PyMDP processing
        """
        if self._observation_processor:
            return self._observation_processor.extract_observation(message, conversation_context)

        # Simple fallback observation extraction
        content = message.get("content", "").lower()

        # Basic sentiment/uncertainty detection
        if any(word in content for word in ["uncertain", "unsure", "confused", "don't know"]):
            return 0  # Uncertain observation
        elif any(word in content for word in ["confident", "sure", "certain", "absolutely"]):
            return 2  # Confident observation
        else:
            return 1  # Neutral observation

    def get_current_belief_context(self) -> Dict[str, Any]:
        """Get current belief state context for response generation.

        Returns:
            Dictionary with belief context for response influence
        """
        current_beliefs = self._belief_manager.get_current_beliefs()

        confidence_level = "low"
        if current_beliefs.max_confidence > 0.7:
            confidence_level = "high"
        elif current_beliefs.max_confidence > 0.4:
            confidence_level = "medium"

        belief_context = ""
        if confidence_level == "high":
            belief_context = "I'm quite confident in my understanding"
        elif confidence_level == "low":
            belief_context = "I'm still exploring different possibilities"
        else:
            belief_context = "I have some insights but remain open to other perspectives"

        return {
            "confidence_level": confidence_level,
            "belief_context": belief_context,
            "entropy": current_beliefs.entropy,
            "max_confidence": current_beliefs.max_confidence,
            "most_likely_state": current_beliefs.most_likely_state(),
            "belief_distribution": current_beliefs.beliefs.tolist(),
        }

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health and performance metrics for monitoring.

        Returns:
            Dictionary with health metrics for observability
        """
        success_rate = 1.0
        if self._update_count + self._error_count > 0:
            success_rate = self._update_count / (self._update_count + self._error_count)

        return {
            "agent_id": self.agent_id,
            "enabled": self.integration_config.enabled,
            "total_updates": self._update_count,
            "total_errors": self._error_count,
            "success_rate": success_rate,
            "last_update_time_ms": self._last_update_time,
            "current_entropy": self._belief_manager.get_current_beliefs().entropy,
            "current_confidence": self._belief_manager.get_current_beliefs().max_confidence,
        }

    def _create_disabled_response(self) -> Dict[str, Any]:
        """Create response when belief integration is disabled."""
        return {
            "belief_influenced": False,
            "belief_status": "disabled",
            "message": "Belief integration is disabled for this agent",
        }

    def _create_error_response(self, error_message: str, error_time_ms: float) -> Dict[str, Any]:
        """Create response when belief update fails but fallback is enabled."""
        return {
            "belief_influenced": False,
            "belief_status": "failed",
            "error_message": error_message,
            "error_time_ms": error_time_ms,
            "fallback_active": True,
        }


class PyMDPBeliefManagerFactory:
    """Factory for creating PyMDP belief managers with dependency injection."""

    @staticmethod
    def create_manager(
        agent_id: str,
        config: Optional[ActiveInferenceConfig] = None,
        integration_config: Optional[BeliefIntegrationConfig] = None,
    ) -> PyMDPBeliefManager:
        """Create a PyMDP belief manager with default configuration.

        Args:
            agent_id: Unique agent identifier
            config: Active inference configuration (creates default if None)
            integration_config: Integration configuration (creates default if None)

        Returns:
            Configured PyMDP belief manager
        """
        if config is None:
            config = ActiveInferenceConfig(num_states=3, num_observations=3, planning_horizon=5)

        return PyMDPBeliefManager(
            config=config, agent_id=agent_id, integration_config=integration_config
        )


# Global manager registry for conversation integration
_belief_managers: Dict[str, PyMDPBeliefManager] = {}


def get_belief_manager(agent_id: str) -> Optional[PyMDPBeliefManager]:
    """Get belief manager for an agent."""
    return _belief_managers.get(agent_id)


def register_belief_manager(manager: PyMDPBeliefManager) -> None:
    """Register belief manager for an agent."""
    _belief_managers[manager.agent_id] = manager
    logger.info(f"Registered belief manager for agent {manager.agent_id}")


def remove_belief_manager(agent_id: str) -> bool:
    """Remove belief manager for an agent."""
    if agent_id in _belief_managers:
        del _belief_managers[agent_id]
        logger.info(f"Removed belief manager for agent {agent_id}")
        return True
    return False
