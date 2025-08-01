"""Belief State Management System for Active Inference (Task 44.2).

This module implements the core belief state management system following the
Nemesis Committee's architectural recommendations for clean abstractions,
comprehensive observability, and robust error handling.

Based on:
- Kent Beck: TDD with mathematical correctness verification
- Robert C. Martin: Clean domain interfaces with PyMDP as implementation detail
- Martin Fowler: Repository pattern separating computation from persistence
- Jessica Kerr: Comprehensive observability with belief metrics
- Michael Feathers: Proper seams and characterization of failure modes
- Charity Majors: Production-ready health monitoring and recovery
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .config import ActiveInferenceConfig

logger = logging.getLogger(__name__)


class BeliefState:
    """Represents a probabilistic belief state over hidden states.

    This class encapsulates the agent's beliefs about the current state of the
    world, providing both computational access and observability features.
    """

    def __init__(
        self,
        beliefs: NDArray[np.floating],
        timestamp: Optional[float] = None,
        observation_history: Optional[List[int]] = None,
    ):
        """Initialize belief state.

        Args:
            beliefs: Probability distribution over states (must sum to 1)
            timestamp: Creation timestamp (defaults to current time)
            observation_history: Sequence of observations that led to this state

        Raises:
            ValueError: If beliefs are not properly normalized or contain invalid values
        """
        self._validate_beliefs(beliefs)
        self._beliefs = beliefs.copy()
        self._timestamp = timestamp or time.time()
        self._observation_history = observation_history or []

        # Compute belief metrics for observability
        self._entropy = self._compute_entropy()
        self._max_confidence = float(np.max(beliefs))
        self._effective_states = self._compute_effective_states()

        logger.debug(
            f"Created belief state: entropy={self._entropy:.4f}, "
            f"max_confidence={self._max_confidence:.4f}, "
            f"effective_states={self._effective_states}"
        )

    @property
    def beliefs(self) -> NDArray[np.floating]:
        """Get belief distribution (read-only copy)."""
        return self._beliefs.copy()

    @property
    def entropy(self) -> float:
        """Return belief entropy (higher = more uncertain)."""
        return self._entropy

    @property
    def max_confidence(self) -> float:
        """Get maximum confidence in any single state."""
        return self._max_confidence

    @property
    def effective_states(self) -> int:
        """Get number of states with significant probability mass."""
        return self._effective_states

    @property
    def timestamp(self) -> float:
        """Get creation timestamp."""
        return self._timestamp

    @property
    def observation_history(self) -> List[int]:
        """Get observation history (read-only copy)."""
        return self._observation_history.copy()

    def _validate_beliefs(self, beliefs: NDArray[np.floating]) -> None:
        """Validate belief distribution properties.

        Args:
            beliefs: Belief distribution to validate

        Raises:
            ValueError: If beliefs are invalid
        """
        if not isinstance(beliefs, np.ndarray):
            raise ValueError("Beliefs must be numpy array")

        if beliefs.ndim != 1:
            raise ValueError(f"Beliefs must be 1-dimensional, got shape {beliefs.shape}")

        if len(beliefs) == 0:
            raise ValueError("Beliefs cannot be empty")

        if not np.all(beliefs >= 0):
            raise ValueError("All belief values must be non-negative")

        if not np.isfinite(beliefs).all():
            raise ValueError("All belief values must be finite")

        belief_sum = np.sum(beliefs)
        if not np.isclose(belief_sum, 1.0, rtol=1e-5):
            raise ValueError(f"Beliefs must sum to 1.0, got sum={belief_sum}")

    def _compute_entropy(self) -> float:
        """Compute Shannon entropy of belief distribution."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-12
        safe_beliefs = self._beliefs + epsilon
        return float(-np.sum(safe_beliefs * np.log(safe_beliefs)))

    def _compute_effective_states(self) -> int:
        """Compute number of states with significant probability mass."""
        threshold = 0.01  # States with >1% probability
        return int(np.sum(self._beliefs > threshold))

    def kl_divergence_from(self, other: "BeliefState") -> float:
        """Compute KL divergence from another belief state.

        Args:
            other: Reference belief state

        Returns:
            KL divergence D(self || other)
        """
        if len(self._beliefs) != len(other._beliefs):
            raise ValueError("Belief states must have same dimensionality")

        # Avoid division by zero
        epsilon = 1e-12
        p = self._beliefs + epsilon
        q = other._beliefs + epsilon

        return float(np.sum(p * np.log(p / q)))

    def most_likely_state(self) -> int:
        """Get index of most likely state."""
        return int(np.argmax(self._beliefs))

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"BeliefState(entropy={self._entropy:.3f}, "
            f"max_conf={self._max_confidence:.3f}, "
            f"most_likely={self.most_likely_state()})"
        )


class BeliefUpdateResult:
    """Result of a belief update operation with observability metrics."""

    def __init__(
        self,
        new_belief_state: BeliefState,
        previous_belief_state: BeliefState,
        observation: int,
        update_time_ms: float,
    ):
        """Initialize belief update result.

        Args:
            new_belief_state: Updated belief state
            previous_belief_state: Previous belief state
            observation: Observation that triggered the update
            update_time_ms: Time taken for update in milliseconds
        """
        self.new_belief_state = new_belief_state
        self.previous_belief_state = previous_belief_state
        self.observation = observation
        self.update_time_ms = update_time_ms

        # Compute change metrics
        self.entropy_change = new_belief_state.entropy - previous_belief_state.entropy
        self.kl_divergence = new_belief_state.kl_divergence_from(previous_belief_state)
        self.confidence_change = (
            new_belief_state.max_confidence - previous_belief_state.max_confidence
        )

        logger.info(
            f"Belief update completed: obs={observation}, "
            f"time={update_time_ms:.2f}ms, "
            f"entropy_change={self.entropy_change:.4f}, "
            f"kl_divergence={self.kl_divergence:.4f}, "
            f"confidence_change={self.confidence_change:.4f}"
        )


class BeliefStateRepository(ABC):
    """Abstract repository for belief state persistence and retrieval."""

    @abstractmethod
    def save_belief_state(self, belief_state: BeliefState, agent_id: str) -> None:
        """Save belief state for an agent."""
        pass

    @abstractmethod
    def get_current_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get current belief state for an agent."""
        pass

    @abstractmethod
    def get_belief_history(self, agent_id: str, limit: Optional[int] = None) -> List[BeliefState]:
        """Get belief history for an agent."""
        pass

    @abstractmethod
    def clear_history(self, agent_id: str) -> None:
        """Clear belief history for an agent."""
        pass


class InMemoryBeliefRepository(BeliefStateRepository):
    """In-memory implementation of belief state repository for testing."""

    def __init__(self, max_history_per_agent: int = 1000):
        """Initialize in-memory repository.

        Args:
            max_history_per_agent: Maximum belief states to store per agent
        """
        self._current_beliefs: Dict[str, BeliefState] = {}
        self._belief_history: Dict[str, List[BeliefState]] = {}
        self._max_history = max_history_per_agent

    def save_belief_state(self, belief_state: BeliefState, agent_id: str) -> None:
        """Save belief state for an agent."""
        self._current_beliefs[agent_id] = belief_state

        if agent_id not in self._belief_history:
            self._belief_history[agent_id] = []

        self._belief_history[agent_id].append(belief_state)

        # Trim history if needed
        if len(self._belief_history[agent_id]) > self._max_history:
            self._belief_history[agent_id].pop(0)

        logger.debug(f"Saved belief state for agent {agent_id}")

    def get_current_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get current belief state for an agent."""
        return self._current_beliefs.get(agent_id)

    def get_belief_history(self, agent_id: str, limit: Optional[int] = None) -> List[BeliefState]:
        """Get belief history for an agent."""
        history = self._belief_history.get(agent_id, [])
        if limit is not None:
            return history[-limit:]
        return history.copy()

    def clear_history(self, agent_id: str) -> None:
        """Clear belief history for an agent."""
        self._current_beliefs.pop(agent_id, None)
        self._belief_history.pop(agent_id, None)
        logger.debug(f"Cleared belief history for agent {agent_id}")


class BeliefStateManager:
    """Core belief state management system.

    This class orchestrates belief state updates using PyMDP while providing
    clean domain interfaces and comprehensive observability.
    """

    def __init__(
        self,
        config: ActiveInferenceConfig,
        repository: BeliefStateRepository,
        A_matrix: NDArray[np.floating],
        agent_id: str = "default",
    ):
        """Initialize belief state manager.

        Args:
            config: Active inference configuration
            repository: Belief state repository for persistence
            A_matrix: Observation model matrix
            agent_id: Unique identifier for this agent

        Raises:
            ValueError: If configuration or matrices are invalid
        """
        self._config = config
        self._repository = repository
        self._agent_id = agent_id
        self._A_matrix = A_matrix.copy()

        # Validate observation model
        self._validate_observation_model(A_matrix)

        # Initialize with uniform beliefs
        num_states = A_matrix.shape[1]
        initial_beliefs = np.ones(num_states) / num_states
        self._current_belief_state = BeliefState(beliefs=initial_beliefs, observation_history=[])

        # Save initial state
        self._repository.save_belief_state(self._current_belief_state, agent_id)

        logger.info(
            f"Initialized belief state manager for agent {agent_id} " f"with {num_states} states"
        )

    def _validate_observation_model(self, A_matrix: NDArray[np.floating]) -> None:
        """Validate observation model matrix.

        Args:
            A_matrix: Observation model to validate

        Raises:
            ValueError: If observation model is invalid
        """
        if not isinstance(A_matrix, np.ndarray):
            raise ValueError("A matrix must be numpy array")

        if A_matrix.ndim != 2:
            raise ValueError(f"A matrix must be 2-dimensional, got shape {A_matrix.shape}")

        if not np.all(A_matrix >= 0):
            raise ValueError("A matrix values must be non-negative")

        if not np.isfinite(A_matrix).all():
            raise ValueError("A matrix values must be finite")

        # Check column normalization (each column should sum to 1)
        col_sums = A_matrix.sum(axis=0)
        if not np.allclose(col_sums, 1.0, rtol=1e-5):
            raise ValueError("A matrix columns must sum to 1 (likelihood distributions)")

    def update_beliefs(self, observation: int) -> BeliefUpdateResult:
        """Update beliefs based on new observation using Bayesian inference.

        Args:
            observation: New observation index

        Returns:
            BeliefUpdateResult with metrics and new belief state

        Raises:
            ValueError: If observation is invalid
        """
        start_time = time.time()

        # Validate observation
        if not isinstance(observation, int):
            raise ValueError(f"Observation must be integer, got {type(observation)}")

        num_observations = self._A_matrix.shape[0]
        if not (0 <= observation < num_observations):
            raise ValueError(f"Observation {observation} out of range [0, {num_observations})")

        # Get previous beliefs
        previous_beliefs = self._current_belief_state.beliefs

        # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
        # P(o|s) is the observation column in A matrix
        likelihood = self._A_matrix[observation, :]  # P(observation | state)

        # Element-wise multiplication and normalization
        posterior_unnormalized = likelihood * previous_beliefs

        # Normalize to get proper probability distribution
        posterior_sum = np.sum(posterior_unnormalized)
        if posterior_sum == 0:
            logger.warning(
                f"Zero posterior probability for observation {observation}, "
                f"keeping previous beliefs"
            )
            new_beliefs = previous_beliefs.copy()
        else:
            new_beliefs = posterior_unnormalized / posterior_sum

        # Create new belief state
        new_observation_history = self._current_belief_state.observation_history + [observation]
        new_belief_state = BeliefState(
            beliefs=new_beliefs, observation_history=new_observation_history
        )

        # Update current state and save
        previous_belief_state = self._current_belief_state
        self._current_belief_state = new_belief_state
        self._repository.save_belief_state(new_belief_state, self._agent_id)

        # Compute timing
        end_time = time.time()
        update_time_ms = (end_time - start_time) * 1000

        # Create result with observability metrics
        result = BeliefUpdateResult(
            new_belief_state=new_belief_state,
            previous_belief_state=previous_belief_state,
            observation=observation,
            update_time_ms=update_time_ms,
        )

        return result

    def get_current_beliefs(self) -> BeliefState:
        """Get current belief state (read-only)."""
        return self._current_belief_state

    def get_belief_history(self, limit: Optional[int] = None) -> List[BeliefState]:
        """Get belief history for this agent."""
        return self._repository.get_belief_history(self._agent_id, limit)

    def reset_beliefs(self, initial_beliefs: Optional[NDArray[np.floating]] = None) -> None:
        """Reset beliefs to initial state.

        Args:
            initial_beliefs: Custom initial beliefs (defaults to uniform)
        """
        if initial_beliefs is None:
            num_states = self._A_matrix.shape[1]
            initial_beliefs = np.ones(num_states) / num_states

        reset_belief_state = BeliefState(beliefs=initial_beliefs, observation_history=[])

        self._current_belief_state = reset_belief_state
        self._repository.save_belief_state(reset_belief_state, self._agent_id)

        logger.info(f"Reset beliefs for agent {self._agent_id}")

    def get_belief_summary(self) -> Dict[str, Union[float, int, List[float]]]:
        """Get summary of current beliefs for monitoring/debugging.

        Returns:
            Dictionary with belief metrics and state probabilities
        """
        current = self._current_belief_state
        return {
            "entropy": current.entropy,
            "max_confidence": current.max_confidence,
            "effective_states": current.effective_states,
            "most_likely_state": current.most_likely_state(),
            "num_observations": len(current.observation_history),
            "belief_distribution": current.beliefs.tolist(),
            "timestamp": current.timestamp,
        }
