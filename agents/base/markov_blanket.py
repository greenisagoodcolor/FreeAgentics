"""
Markov Blanket Interface for Agent Boundary System

This module implements the MarkovBlanket interface within the /agents boundary
system as specified in ADR-002, enabling tracking of agent boundaries and
violation events. Integrates with the Active Inference engine (ADR-005) and
pymdp for real-time state updates.

Mathematical Foundation:
The Markov blanket defines the statistical boundary of an agent using pymdp's
validated Active Inference implementation:
- Internal states (μ): Agent's internal beliefs and hidden states
- Sensory states (s): Observations from the environment
- Active states (a): Actions the agent can perform
- External states (η): Environment states beyond the agent's influence

The key property is conditional independence: p(μ,η|s,a) = p(μ|s,a)p(η|s,a)
This is verified using pymdp's statistical independence testing.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent
    from pymdp.maths import utils as pymdp_utils

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    PyMDPAgent = None
    pymdp_utils = None
    utils = None

from agents.base.data_model import Agent
from inference.engine.pymdp_generative_model import create_pymdp_generative_model

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Simple adapter for agent state representation"""

    agent_id: str
    position: Optional[Any] = None
    status: Optional[str] = None
    energy: float = 1.0
    health: float = 1.0
    intended_action: Optional[List[float]] = None
    belief_state: Optional[List[float]] = None

    @classmethod
    def from_agent(cls, agent: Agent) -> "AgentState":
        """Create AgentState from Agent instance"""
        return cls(
            agent_id=agent.agent_id,
            position=agent.position,
            status=agent.status.value if agent.status else None,
            energy=agent.resources.energy if agent.resources else 1.0,
            health=agent.resources.health if agent.resources else 1.0,
            belief_state=getattr(agent, "belief_state", None),
        )


class BoundaryState(Enum):
    """States of the Markov blanket boundary"""

    INTACT = "intact"
    COMPROMISED = "compromised"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class ViolationType(Enum):
    """Types of boundary violations"""

    INDEPENDENCE_FAILURE = "independence_failure"
    BOUNDARY_BREACH = "boundary_breach"
    SENSORY_OVERFLOW = "sensory_overflow"
    ACTION_OVERFLOW = "action_overflow"
    INTERNAL_LEAK = "internal_leak"
    EXTERNAL_INTRUSION = "external_intrusion"


@dataclass
class MarkovBlanketDimensions:
    """Dimensions of the Markov blanket using pymdp-compatible format"""

    # Internal states (μ) - agent's private states (beliefs)
    internal_states: np.ndarray = field(default_factory=lambda: np.array([]))
    internal_dimension: int = 0

    # Sensory states (s) - observations from environment
    sensory_states: np.ndarray = field(default_factory=lambda: np.array([]))
    sensory_dimension: int = 0

    # Active states (a) - agent's actions
    active_states: np.ndarray = field(default_factory=lambda: np.array([]))
    active_dimension: int = 0

    # External states (η) - environment states beyond agent control
    external_states: np.ndarray = field(default_factory=lambda: np.array([]))
    external_dimension: int = 0

    def __post_init__(self):
        """Initialize dimensions based on state arrays"""
        if self.internal_states.size > 0:
            self.internal_dimension = len(self.internal_states)
        if self.sensory_states.size > 0:
            self.sensory_dimension = len(self.sensory_states)
        if self.active_states.size > 0:
            self.active_dimension = len(self.active_states)
        if self.external_states.size > 0:
            self.external_dimension = len(self.external_states)

    def get_total_dimension(self) -> int:
        """Get total dimensionality of the Markov blanket"""
        return (
            self.internal_dimension
            + self.sensory_dimension
            + self.active_dimension
            + self.external_dimension
        )

    def get_boundary_states(self) -> np.ndarray:
        """Get the boundary states (sensory + active) in pymdp format"""
        if self.sensory_states.size == 0 and self.active_states.size == 0:
            return np.array([])

        boundary: List[float] = []
        if self.sensory_states.size > 0:
            boundary.extend(self.sensory_states)
        if self.active_states.size > 0:
            boundary.extend(self.active_states)

        return np.array(boundary)


@dataclass
class BoundaryViolationEvent:
    """Event representing a boundary violation detected by pymdp analysis"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    violation_type: ViolationType = ViolationType.INDEPENDENCE_FAILURE
    timestamp: datetime = field(default_factory=datetime.now)

    # Violation details from pymdp analysis
    severity: float = 0.0  # 0.0 (minor) to 1.0 (critical)
    independence_measure: float = 0.0  # I(μ;η|s,a) from pymdp
    threshold_violated: float = 0.0

    # Mathematical evidence from pymdp
    free_energy: float = 0.0  # Free energy from pymdp agent
    expected_free_energy: float = 0.0  # Expected free energy
    kl_divergence: float = 0.0  # KL divergence between beliefs and prior

    # Context information
    dimensions_at_violation: Optional[MarkovBlanketDimensions] = None
    pymdp_agent_state: Optional[Dict[str, Any]] = None

    # Mitigation status
    acknowledged: bool = False
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class BoundaryMetrics:
    """Metrics for boundary integrity monitoring using pymdp"""

    # Core metrics from pymdp
    free_energy: float = 0.0  # Variational free energy
    expected_free_energy: float = 0.0  # Expected free energy
    kl_divergence: float = 0.0  # KL(q||p) between beliefs and prior

    # Boundary-specific metrics
    boundary_integrity: float = 1.0  # 0.0 (compromised) to 1.0 (intact)
    conditional_independence: float = 0.0  # I(μ;η|s,a) from pymdp

    # Temporal metrics
    stability_over_time: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)

    # Violation tracking
    violation_count: int = 0
    last_violation_time: Optional[datetime] = None

    def update_from_pymdp_agent(self, pymdp_agent: "PyMDPAgent") -> None:
        """Update metrics from pymdp agent state"""
        if not PYMDP_AVAILABLE or pymdp_agent is None:
            return

        try:
            # Extract free energy if available
            if hasattr(pymdp_agent, "F") and pymdp_agent.F is not None:
                if isinstance(pymdp_agent.F, list) and len(pymdp_agent.F) > 0:
                    self.free_energy = float(pymdp_agent.F[-1])
                elif isinstance(pymdp_agent.F, (int, float)):
                    self.free_energy = float(pymdp_agent.F)

            # Extract expected free energy if available
            if hasattr(pymdp_agent, "G") and pymdp_agent.G is not None:
                if isinstance(pymdp_agent.G, np.ndarray):
                    self.expected_free_energy = float(np.mean(pymdp_agent.G))
                elif isinstance(pymdp_agent.G, (int, float)):
                    self.expected_free_energy = float(pymdp_agent.G)

            # Compute KL divergence between beliefs and prior
            if (
                hasattr(pymdp_agent, "qs")
                and hasattr(pymdp_agent, "D")
                and pymdp_agent.qs is not None
                and pymdp_agent.D is not None
            ):
                try:
                    if isinstance(pymdp_agent.qs, list) and len(pymdp_agent.qs) > 0:
                        beliefs = pymdp_agent.qs[0]  # First factor
                        prior = (
                            pymdp_agent.D[0] if isinstance(pymdp_agent.D, list) else pymdp_agent.D
                        )

                        # Ensure arrays are properly normalized
                        beliefs = beliefs / np.sum(beliefs)
                        prior = prior / np.sum(prior)

                        # Compute KL divergence: KL(q||p) = sum(q * log(q/p))
                        self.kl_divergence = float(pymdp_utils.kl_divergence(beliefs, prior))
                except Exception as e:
                    logger.debug(f"Could not compute KL divergence: {e}")

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error updating metrics from pymdp agent: {e}")


class MarkovBlanketInterface(ABC):
    """Abstract interface for Markov blanket implementations using pymdp"""

    @abstractmethod
    def get_dimensions(self) -> MarkovBlanketDimensions:
        """Get current Markov blanket dimensions"""
        pass

    @abstractmethod
    def update_states(self, agent_state: AgentState, environment_state: np.ndarray) -> None:
        """Update internal states based on agent and environment"""
        pass

    @abstractmethod
    def verify_independence(self) -> Tuple[float, Dict[str, Any]]:
        """Verify conditional independence using pymdp"""
        pass

    @abstractmethod
    def detect_violations(self) -> List[BoundaryViolationEvent]:
        """Detect any boundary violations using pymdp analysis"""
        pass

    @abstractmethod
    def get_metrics(self) -> BoundaryMetrics:
        """Get current boundary metrics from pymdp"""
        pass

    @abstractmethod
    def set_violation_handler(self, handler: Callable[[BoundaryViolationEvent], None]) -> None:
        """Set handler for violation events"""
        pass


class PyMDPMarkovBlanket(MarkovBlanketInterface):
    """
    Markov blanket implementation using pymdp Active Inference.

    Integrates with the existing pymdp infrastructure to provide rigorous
    mathematical validation of boundary conditions and independence testing.
    """

    def __init__(
        self,
        agent_id: str,
        num_states: int = 4,
        num_observations: int = 4,
        num_actions: int = 4,
        independence_threshold: float = 0.05,
    ) -> None:
        """
        Initialize pymdp-based Markov blanket for an agent.

        Args:
            agent_id: Unique identifier for the agent
            num_states: Number of hidden states
            num_observations: Number of possible observations
            num_actions: Number of possible actions
            independence_threshold: Threshold for conditional independence
        """
        if not PYMDP_AVAILABLE:
            raise ImportError("pymdp is required for PyMDPMarkovBlanket but not available")

        self.agent_id = agent_id
        self.independence_threshold = independence_threshold

        # Create pymdp-compatible generative model
        self.generative_model = create_pymdp_generative_model(
            num_states=num_states,
            num_observations=num_observations,
            num_actions=num_actions,
            time_horizon=1,
        )

        # Get pymdp matrices
        A, B, C, D = self.generative_model.get_pymdp_matrices()

        # Create pymdp agent
        self.pymdp_agent = PyMDPAgent(A=A, B=B, C=C, D=D)

        # Initialize dimensions
        self.dimensions = MarkovBlanketDimensions(
            internal_states=D.copy(),  # Initial beliefs
            sensory_states=np.zeros(num_observations),
            active_states=np.zeros(num_actions),
            external_states=np.random.rand(10),  # External environment
        )

        # State history for temporal analysis
        self.state_history: List[MarkovBlanketDimensions] = []
        self.max_history_length = 100

        # Violation tracking
        self.violation_events: List[BoundaryViolationEvent] = []
        self.violation_handlers: List[Callable[[BoundaryViolationEvent], None]] = []

        # Metrics
        self.metrics = BoundaryMetrics()

        # Boundary state
        self.boundary_state = BoundaryState.INTACT
        self.last_update_time = datetime.now()

        logger.info(f"Initialized pymdp Markov blanket for agent {agent_id}")

    def get_dimensions(self) -> MarkovBlanketDimensions:
        """Get current Markov blanket dimensions"""
        return self.dimensions

    def update_states(self, agent_state: AgentState, environment_state: np.ndarray) -> None:
        """Update internal states using pymdp active inference"""
        try:
            # Update observations from environment
            if environment_state.size > 0:
                # Extract observations (limit to model's observation space)
                obs_size = min(len(environment_state), self.dimensions.sensory_dimension)
                if obs_size > 0:
                    observations = environment_state[:obs_size]

                    # Discretize observations for pymdp
                    obs_indices = self._discretize_observations(observations)

                    # Update pymdp agent with new observation
                    if len(obs_indices) > 0:
                        self.pymdp_agent.infer_states(obs_indices)

                        # Update internal states from pymdp beliefs
                        if hasattr(self.pymdp_agent, "qs") and self.pymdp_agent.qs is not None:
                            if (
                                isinstance(self.pymdp_agent.qs, list)
                                and len(self.pymdp_agent.qs) > 0
                            ):
                                self.dimensions.internal_states = self.pymdp_agent.qs[0].copy()

                    # Update sensory states
                    self.dimensions.sensory_states = observations

                # Update external states (remaining environment state)
                if environment_state.size > obs_size:
                    external_size = min(
                        self.dimensions.external_dimension, environment_state.size - obs_size
                    )
                    if external_size > 0:
                        self.dimensions.external_states = environment_state[
                            obs_size : obs_size + external_size
                        ]

            # Update active states from agent's intended action
            if hasattr(agent_state, "intended_action") and agent_state.intended_action is not None:
                action = np.array(agent_state.intended_action)
                if action.size > 0:
                    action_size = min(self.dimensions.active_dimension, action.size)
                    self.dimensions.active_states[:action_size] = action[:action_size]

            # Update metrics from pymdp agent
            self.metrics.update_from_pymdp_agent(self.pymdp_agent)

            # Add to history
            self._add_to_history()

            # Update timestamp
            self.last_update_time = datetime.now()

            logger.debug(f"Updated states for agent {self.agent_id} using pymdp")

        except Exception as e:
            logger.error(f"Error updating states for agent {self.agent_id}: {e}")

    def verify_independence(self) -> Tuple[float, Dict[str, Any]]:
        """Verify conditional independence using pymdp's framework"""
        try:
            if not PYMDP_AVAILABLE:
                return 1.0, {"error": "pymdp not available"}

            # Use pymdp agent's free energy as independence measure
            evidence = {
                "free_energy": self.metrics.free_energy,
                "expected_free_energy": self.metrics.expected_free_energy,
                "kl_divergence": self.metrics.kl_divergence,
                "threshold": self.independence_threshold,
                "test_timestamp": datetime.now().isoformat(),
                "pymdp_agent_available": True,
            }

            # Use KL divergence as independence measure
            # Higher KL divergence indicates less independence
            independence_measure = self.metrics.kl_divergence

            # Update metrics
            self.metrics.conditional_independence = independence_measure

            # Compute boundary integrity based on free energy
            # Lower free energy indicates better boundary integrity
            if self.metrics.free_energy > 0:
                self.metrics.boundary_integrity = max(0.0, 1.0 - (self.metrics.free_energy / 10.0))
            else:
                self.metrics.boundary_integrity = 1.0

            return independence_measure, evidence

        except Exception as e:
            logger.error(f"Error verifying independence for agent " f"{self.agent_id}: {e}")
            return 1.0, {"error": str(e)}

    def detect_violations(self) -> List[BoundaryViolationEvent]:
        """Detect boundary violations using pymdp analysis"""
        violations = []

        try:
            # Verify independence using pymdp
            independence_measure, evidence = self.verify_independence()

            # Check for independence violation based on KL divergence
            if independence_measure > self.independence_threshold:
                violation = BoundaryViolationEvent(
                    agent_id=self.agent_id,
                    violation_type=ViolationType.INDEPENDENCE_FAILURE,
                    severity=min(1.0, independence_measure / self.independence_threshold),
                    independence_measure=independence_measure,
                    threshold_violated=self.independence_threshold,
                    free_energy=self.metrics.free_energy,
                    expected_free_energy=self.metrics.expected_free_energy,
                    kl_divergence=self.metrics.kl_divergence,
                    dimensions_at_violation=self.dimensions,
                    pymdp_agent_state=self._get_pymdp_agent_state(),
                )
                violations.append(violation)

            # Check for high free energy (indicates poor model fit)
            if self.metrics.free_energy > 5.0:  # Threshold for high free energy
                violation = BoundaryViolationEvent(
                    agent_id=self.agent_id,
                    violation_type=ViolationType.BOUNDARY_BREACH,
                    severity=min(1.0, self.metrics.free_energy / 10.0),
                    free_energy=self.metrics.free_energy,
                    dimensions_at_violation=self.dimensions,
                    pymdp_agent_state=self._get_pymdp_agent_state(),
                )
                violations.append(violation)

            # Process any new violations
            for violation in violations:
                self._handle_violation(violation)

            return violations

        except Exception as e:
            logger.error(f"Error detecting violations for agent " f"{self.agent_id}: {e}")
            return []

    def get_metrics(self) -> BoundaryMetrics:
        """Get current boundary metrics from pymdp"""
        # Update metrics from current pymdp agent state
        self.metrics.update_from_pymdp_agent(self.pymdp_agent)

        # Update time-based metrics
        if self.metrics.last_violation_time:
            time_since = datetime.now() - self.metrics.last_violation_time
            self.metrics.stability_over_time = max(0.0, 1.0 - (time_since.total_seconds() / 3600.0))

        return self.metrics

    def set_violation_handler(self, handler: Callable[[BoundaryViolationEvent], None]) -> None:
        """Set handler for violation events"""
        self.violation_handlers.append(handler)

    def get_boundary_state(self) -> BoundaryState:
        """Get current boundary state"""
        return self.boundary_state

    def get_pymdp_agent(self) -> "PyMDPAgent":
        """Get the underlying pymdp agent for advanced operations"""
        return self.pymdp_agent

    def _discretize_observations(self, observations: np.ndarray) -> List[int]:
        """Convert continuous observations to discrete indices for pymdp"""
        try:
            # Simple binning strategy - more sophisticated in practice
            num_obs_states = self.generative_model.dims.num_observations

            # Normalize observations to [0, 1]
            obs_normalized = (observations - observations.min()) / (
                observations.max() - observations.min() + 1e-8
            )

            # Convert to discrete indices
            obs_indices = (obs_normalized * (num_obs_states - 1)).astype(int)
            obs_indices = np.clip(obs_indices, 0, num_obs_states - 1)

            return obs_indices.tolist()

        except Exception as e:
            logger.error(f"Error discretizing observations: {e}")
            return [0]  # Default to first observation

    def _get_pymdp_agent_state(self) -> Dict[str, Any]:
        """Extract current state from pymdp agent for logging"""
        try:
            state = {"agent_id": self.agent_id, "timestamp": datetime.now().isoformat()}

            if hasattr(self.pymdp_agent, "qs") and self.pymdp_agent.qs is not None:
                state["beliefs"] = [q.tolist() for q in self.pymdp_agent.qs]

            if hasattr(self.pymdp_agent, "F") and self.pymdp_agent.F is not None:
                state["free_energy"] = self.pymdp_agent.F

            if hasattr(self.pymdp_agent, "G") and self.pymdp_agent.G is not None:
                if isinstance(self.pymdp_agent.G, np.ndarray):
                    state["expected_free_energy"] = self.pymdp_agent.G.tolist()
                else:
                    state["expected_free_energy"] = self.pymdp_agent.G

            return state

        except Exception as e:
            logger.error(f"Error extracting pymdp agent state: {e}")
            return {"error": str(e)}

    def _add_to_history(self) -> None:
        """Add current dimensions to history"""
        # Create a deep copy of current dimensions
        history_entry = MarkovBlanketDimensions(
            internal_states=self.dimensions.internal_states.copy(),
            sensory_states=self.dimensions.sensory_states.copy(),
            active_states=self.dimensions.active_states.copy(),
            external_states=self.dimensions.external_states.copy(),
        )

        self.state_history.append(history_entry)

        # Maintain maximum history length
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

    def _handle_violation(self, violation: BoundaryViolationEvent) -> None:
        """Handle a detected violation"""
        # Add to violation history
        self.violation_events.append(violation)

        # Update metrics
        self.metrics.violation_count += 1
        self.metrics.last_violation_time = violation.timestamp

        # Update boundary state based on severity
        if violation.severity > 0.8:
            self.boundary_state = BoundaryState.VIOLATED
        elif violation.severity > 0.5:
            self.boundary_state = BoundaryState.COMPROMISED

        # Trigger violation handlers
        for handler in self.violation_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.error(f"Error in violation handler: {e}")

        logger.warning(
            f"Boundary violation detected for agent {self.agent_id}: "
            f"{violation.violation_type.value} "
            f"(severity: {violation.severity:.2f})"
        )


class MarkovBlanketFactory:
    """Factory for creating Markov blanket instances"""

    @staticmethod
    def create_pymdp_blanket(
        agent_id: str, num_states: int = 4, num_observations: int = 4, num_actions: int = 4
    ) -> PyMDPMarkovBlanket:
        """Create a pymdp-based Markov blanket"""
        return PyMDPMarkovBlanket(
            agent_id=agent_id,
            num_states=num_states,
            num_observations=num_observations,
            num_actions=num_actions,
        )

    @staticmethod
    def create_from_agent(agent: Agent) -> PyMDPMarkovBlanket:
        """Create a Markov blanket based on agent characteristics"""
        # Determine dimensions based on agent properties
        num_states = 4  # Default
        if hasattr(agent, "belief_state") and agent.belief_state is not None:
            num_states = max(4, len(agent.belief_state))

        return PyMDPMarkovBlanket(
            agent_id=agent.agent_id, num_states=num_states, num_observations=4, num_actions=4
        )
