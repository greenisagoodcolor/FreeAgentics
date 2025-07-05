"""Coalition Coordinator Agent implementation.

This agent specializes in coordinating multiple agents and forming coalitions
using Active Inference principles to optimize collective objectives.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.base_agent import PYMDP_AVAILABLE, ActiveInferenceAgent, safe_array_to_int
from agents.error_handling import (
    InferenceError,
    PyMDPError,
    safe_pymdp_operation,
    with_error_handling,
)
from agents.pymdp_error_handling import (
    PyMDPErrorHandler,
    safe_array_index,
    safe_numpy_conversion,
    validate_pymdp_matrices,
)

if PYMDP_AVAILABLE:
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent

logger = logging.getLogger(__name__)


class CoalitionCoordinatorAgent(ActiveInferenceAgent):
    """Agent that coordinates coalitions using Active Inference.

    This agent:
    - Monitors other agents in the environment
    - Identifies potential coalition opportunities
    - Coordinates multi-agent activities
    - Optimizes collective objectives through Active Inference
    - Manages coalition lifecycle (formation, maintenance, dissolution)
    """

    def __init__(self, agent_id: str, name: str, max_agents: int = 10):
        """Initialize coalition coordinator agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            max_agents: Maximum number of agents this coordinator can manage
        """
        # Set attributes needed for PyMDP initialization BEFORE calling super()
        self.max_agents = max_agents
        self.position = [0, 0]  # Coordinator position (conceptual)

        # Coalition state space for PyMDP
        # States represent different coalition configurations
        self.num_states = min(2**max_agents, 64)  # Limit state space for tractability
        self.num_obs = 8  # Coalition observations: empty, forming, active, conflict, success, failure, idle, unknown
        self.num_actions = (
            6  # Coalition actions: invite, exclude, merge, split, coordinate, dissolve
        )

        config = {
            "use_pymdp": PYMDP_AVAILABLE,
            "use_llm": True,  # Use LLM for coordination strategy
            "max_agents": max_agents,
            "agent_type": "coalition_coordinator",
        }
        # Coalition formation parameters - set BEFORE super().__init__
        self.coordination_range = 5.0  # Distance within which agents can coordinate
        self.min_coalition_size = 2
        self.max_coalition_size = min(max_agents // 2, 5)

        super().__init__(agent_id, name, config)

        # Coalition-specific state
        self.known_agents: Dict[str, Dict[str, Any]] = {}  # Other agents we know about
        self.active_coalitions: Dict[str, Dict[str, Any]] = {}  # Currently active coalitions
        self.coalition_history: List[Dict[str, Any]] = []  # Historical coalition performance

        # Action mapping for PyMDP
        self.action_map = {
            0: "invite",  # Invite agent to join coalition
            1: "exclude",  # Remove agent from coalition
            2: "merge",  # Merge two coalitions
            3: "split",  # Split coalition
            4: "coordinate",  # Coordinate existing coalition
            5: "dissolve",  # Dissolve coalition
        }

        # Coalition objectives and preferences
        self.coalition_objectives = {
            "efficiency": 1.0,  # Prefer efficient coalitions
            "diversity": 0.8,  # Value diverse skill sets
            "stability": 0.9,  # Prefer stable coalitions
            "scalability": 0.7,  # Consider growth potential
        }

        # Coordination metrics
        self.coordination_success_rate = 0.0
        self.average_coalition_lifetime = 0.0
        self.total_coordinated_tasks = 0

        # Initialize additional attributes needed by base class
        self.total_observations = 0
        self.total_actions = 0
        self.use_llm = config.get("use_llm", True)
        self.llm_manager = None  # Will be set by base class if available

        # Initialize error handling
        self.pymdp_error_handler = PyMDPErrorHandler(self.agent_id)

    @safe_pymdp_operation("pymdp_init", default_value=None)
    def _initialize_pymdp(self) -> None:
        """Initialize PyMDP agent for coalition coordination with comprehensive error handling."""
        if not PYMDP_AVAILABLE:
            return

        try:
            # A matrix: P(observation|state) - coalition state observations
            A = np.zeros((self.num_obs, self.num_states))

            for state in range(self.num_states):
                # Map coalition states to observation probabilities
                # State encoding: each bit represents whether an agent is in coalition
                coalition_size = bin(state).count("1")

                if coalition_size == 0:
                    A[0, state] = 0.8  # Empty
                    A[7, state] = 0.2  # Unknown
                elif coalition_size == 1:
                    A[6, state] = 0.7  # Idle
                    A[0, state] = 0.3  # Empty
                elif coalition_size <= self.max_coalition_size:
                    if coalition_size == 2:
                        A[1, state] = 0.6  # Forming
                        A[2, state] = 0.4  # Active
                    else:
                        A[2, state] = 0.7  # Active
                        A[4, state] = 0.2  # Success
                        A[3, state] = 0.1  # Conflict
                else:
                    # Over-sized coalitions are less stable
                    A[3, state] = 0.5  # Conflict
                    A[5, state] = 0.3  # Failure
                    A[2, state] = 0.2  # Active

            # Validate and normalize A matrix
            A = A + 1e-10  # Add epsilon for numerical stability
            A = utils.norm_dist(A)

            # B matrix: P(state_t+1|state_t, action) - coalition dynamics
            B = np.zeros((self.num_states, self.num_states, self.num_actions))

            for action in range(self.num_actions):
                for state in range(self.num_states):
                    if action == 0:  # invite
                        # Add agents to coalition (set more bits)
                        next_states = self._get_invite_transitions(state)
                    elif action == 1:  # exclude
                        # Remove agents from coalition (unset bits)
                        next_states = self._get_exclude_transitions(state)
                    elif action == 2:  # merge
                        # Merge coalitions (complex state changes)
                        next_states = self._get_merge_transitions(state)
                    elif action == 3:  # split
                        # Split coalition (reduce state complexity)
                        next_states = self._get_split_transitions(state)
                    elif action == 4:  # coordinate
                        # Maintain current coalition (high stay probability)
                        next_states = [(state, 0.9), (state, 0.1)]
                    else:  # dissolve
                        # Return to empty state
                        next_states = [(0, 0.8), (state, 0.2)]

                    # Set transition probabilities
                    for next_state, prob in next_states:
                        if 0 <= next_state < self.num_states:
                            B[next_state, state, action] = prob

            # Normalize B matrix with validation
            for action in range(self.num_actions):
                for state in range(self.num_states):
                    col_sum = B[:, state, action].sum()
                    if col_sum > 0:
                        B[:, state, action] /= col_sum
                    else:
                        # Handle zero columns
                        B[state, state, action] = 1.0

            # C matrix: Preferences over observations - prefer successful coalitions
            C = np.zeros(self.num_obs)
            C[0] = -0.2  # Slight penalty for empty
            C[1] = 0.5  # Moderate preference for forming
            C[2] = 1.5  # Strong preference for active coalitions
            C[3] = -2.0  # Strong aversion to conflict
            C[4] = 2.0  # Very strong preference for success
            C[5] = -3.0  # Very strong aversion to failure
            C[6] = -0.5  # Penalty for idle agents
            C[7] = -1.0  # Penalty for unknown states

            # D matrix: Initial belief state - start with uniform uncertainty
            D = utils.norm_dist(np.ones(self.num_states))

            # Validate matrices before creating PyMDP agent
            is_valid, validation_msg = validate_pymdp_matrices(A, B, C, D)
            if not is_valid:
                raise ValueError(f"PyMDP matrix validation failed: {validation_msg}")

            # Create PyMDP agent with coalition-specific configuration
            self.pymdp_agent = PyMDPAgent(
                A=A,
                B=B,
                C=C,
                D=D,
                inference_algo="VANILLA",
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False,
                inference_horizon=self.config.get("planning_horizon", 3),
            )

            logger.info(f"Initialized PyMDP for coalition coordinator {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize PyMDP: {e}")
            self.pymdp_agent = None

    def _get_invite_transitions(self, state: int) -> List[Tuple[int, float]]:
        """Get state transitions for invite action."""
        transitions = []

        # Find empty slots (unset bits) to invite agents
        for i in range(min(self.max_agents, 6)):  # Limit for tractability
            if not (state & (1 << i)):  # Bit i is not set
                new_state = state | (1 << i)  # Set bit i
                transitions.append((new_state, 0.7))

        if not transitions:
            transitions.append((state, 1.0))  # No change if coalition is full

        # Normalize probabilities
        total_prob = sum(prob for _, prob in transitions)
        if total_prob > 0:
            transitions = [(s, p / total_prob) for s, p in transitions]

        return transitions[:3]  # Limit transitions for efficiency

    def _get_exclude_transitions(self, state: int) -> List[Tuple[int, float]]:
        """Get state transitions for exclude action."""
        transitions = []

        # Find occupied slots (set bits) to exclude agents
        for i in range(min(self.max_agents, 6)):
            if state & (1 << i):  # Bit i is set
                new_state = state & ~(1 << i)  # Unset bit i
                transitions.append((new_state, 0.8))

        if not transitions:
            transitions.append((state, 1.0))  # No change if coalition is empty

        # Normalize probabilities
        total_prob = sum(prob for _, prob in transitions)
        if total_prob > 0:
            transitions = [(s, p / total_prob) for s, p in transitions]

        return transitions[:3]

    def _get_merge_transitions(self, state: int) -> List[Tuple[int, float]]:
        """Get state transitions for merge action."""
        # Simplified: merge increases coalition size
        coalition_size = bin(state).count("1")
        if coalition_size < self.max_coalition_size:
            # Add one more agent with high probability
            for i in range(min(self.max_agents, 6)):
                if not (state & (1 << i)):
                    new_state = state | (1 << i)
                    return [(new_state, 0.6), (state, 0.4)]

        return [(state, 1.0)]  # No change if can't merge

    def _get_split_transitions(self, state: int) -> List[Tuple[int, float]]:
        """Get state transitions for split action."""
        # Simplified: split reduces coalition size
        coalition_size = bin(state).count("1")
        if coalition_size > 1:
            # Remove roughly half the agents
            new_state = state
            removed = 0
            target_removals = coalition_size // 2

            for i in range(min(self.max_agents, 6)):
                if (state & (1 << i)) and removed < target_removals:
                    new_state &= ~(1 << i)
                    removed += 1

            return [(new_state, 0.7), (state, 0.3)]

        return [(0, 0.8), (state, 0.2)]  # Split single agent goes to empty

    def perceive(self, observation: Any) -> None:
        """Process observations about other agents and coalition opportunities.

        Args:
            observation: Environment observation with agent information
        """
        self.total_observations += 1

        # Extract agent information from observation
        if isinstance(observation, dict):
            # Update known agents
            if "visible_agents" in observation:
                self._update_agent_registry(observation["visible_agents"])

            # Update coalition status
            if "coalition_status" in observation:
                self._update_coalition_status(observation["coalition_status"])

            # Update coordinator position if provided
            if "position" in observation:
                self.position = observation["position"]

        # Use LLM for coalition strategy assessment
        if self.use_llm and self.llm_manager:
            observation = self._assess_coalition_opportunities(observation)

        # Update PyMDP beliefs with error handling
        if self.pymdp_agent and isinstance(observation, dict):
            obs_idx = self._observation_to_index(observation)
            if obs_idx is not None:
                # PyMDP expects observation as a list for multiple modalities
                success, _, error = self.pymdp_error_handler.safe_execute(
                    "state_inference",
                    lambda: self.pymdp_agent.infer_states([obs_idx]),
                    lambda: None,
                )
                if error:
                    logger.warning(f"State inference failed: {error}")

    def _update_agent_registry(self, visible_agents: List[Dict[str, Any]]):
        """Update registry of known agents."""
        current_time = datetime.now()

        for agent_info in visible_agents:
            agent_id = agent_info.get("id", "unknown")

            # Update or create agent record
            self.known_agents[agent_id] = {
                "id": agent_id,
                "position": agent_info.get("position", [0, 0]),
                "capabilities": agent_info.get("capabilities", []),
                "status": agent_info.get("status", "active"),
                "last_seen": current_time,
                "coalition_history": self.known_agents.get(agent_id, {}).get(
                    "coalition_history", []
                ),
                "performance_score": agent_info.get("performance_score", 0.5),
                "availability": agent_info.get("availability", True),
            }

    def _update_coalition_status(self, coalition_status: Dict[str, Any]):
        """Update status of active coalitions."""
        for coalition_id, status in coalition_status.items():
            if coalition_id in self.active_coalitions:
                self.active_coalitions[coalition_id].update(
                    {
                        "status": status.get("status", "unknown"),
                        "performance": status.get("performance", 0.0),
                        "last_update": datetime.now(),
                    }
                )

    def _assess_coalition_opportunities(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to assess coalition opportunities and strategies."""
        try:
            visible_agents = observation.get("visible_agents", [])
            current_coalitions = list(self.active_coalitions.keys())

            if visible_agents or current_coalitions:
                prompt = """As a coalition coordinator agent, analyze the current situation:

                Visible agents: {len(visible_agents)}
                Active coalitions: {len(current_coalitions)}
                Known agents: {len(self.known_agents)}
                Coordination success rate: {self.coordination_success_rate:.2f}

                Assess:
                1. Coalition formation opportunities
                2. Potential conflicts or inefficiencies
                3. Recommended coordination actions
                4. Strategic priorities

                Consider objectives: efficiency ({self.coalition_objectives['efficiency']}),
                diversity ({self.coalition_objectives['diversity']}),
                stability ({self.coalition_objectives['stability']})
                """

                response = self.llm_manager.generate(prompt, max_tokens=200)
                observation["coordination_analysis"] = response

        except Exception as e:
            logger.error(f"Coalition assessment failed: {e}")

        return observation

    @safe_pymdp_operation("action_selection", default_value="coordinate")
    def select_action(self) -> Optional[Any]:
        """Select coalition coordination action with comprehensive error handling.

        Returns:
            Selected coordination action
        """
        if not self.is_active:
            return None

        # Use PyMDP if available
        if self.pymdp_agent:
            try:
                # Infer policies with error handling
                success, q_pi_G, error = self.pymdp_error_handler.safe_execute(
                    "policy_inference",
                    lambda: self.pymdp_agent.infer_policies(),
                    lambda: (None, None),
                )

                if success:
                    # Sample action with safe conversion
                    success, action_idx, error = self.pymdp_error_handler.safe_execute(
                        "action_sampling",
                        lambda: self.pymdp_agent.sample_action(),
                        lambda: 4,  # Default to coordinate
                    )

                    if success:
                        # Convert numpy array to scalar for dictionary lookup
                        action_idx = safe_array_to_int(action_idx)
                        action = safe_array_index(self.action_map, action_idx, "coordinate")
                    else:
                        logger.warning(f"Action sampling failed: {error}")
                        action = self._fallback_action_selection()
                else:
                    logger.warning(f"Policy inference failed: {error}")
                    action = self._fallback_action_selection()

            except Exception as e:
                logger.error(f"PyMDP action selection failed: {e}")
                action = self._fallback_action_selection()
        else:
            action = self._fallback_action_selection()

        self.total_actions += 1
        self.last_action_at = datetime.now()

        # Update coordination metrics
        self._update_coordination_metrics(action)

        logger.debug(f"Coalition coordinator {self.agent_id} selected action: {action}")
        return action

    def _fallback_action_selection(self) -> str:
        """Fallback action selection without PyMDP."""
        # Simple coalition strategy

        # If we have no active coalitions and multiple known agents, try to form one
        if not self.active_coalitions and len(self.known_agents) >= self.min_coalition_size:
            return "invite"

        # If we have oversized coalitions, consider splitting
        for coalition in self.active_coalitions.values():
            if len(coalition.get("members", [])) > self.max_coalition_size:
                return "split"

        # If we have underperforming coalitions, consider dissolving
        for coalition in self.active_coalitions.values():
            if coalition.get("performance", 0.5) < 0.3:
                return "dissolve"

        # Default: coordinate existing coalitions
        if self.active_coalitions:
            return "coordinate"

        # If no coalitions exist, try to invite
        return "invite"

    @safe_pymdp_operation("belief_update", default_value=None)
    def update_beliefs(self) -> None:
        """Update beliefs about coalition states and opportunities with error handling."""
        # Update coordination success rate
        if self.total_actions > 0:
            successful_coordinations = sum(
                1 for c in self.active_coalitions.values() if c.get("performance", 0) > 0.6
            )
            self.coordination_success_rate = (
                successful_coordinations / len(self.active_coalitions)
                if self.active_coalitions
                else 0.0
            )

        # Update average coalition lifetime
        if self.coalition_history:
            lifetimes = [c.get("lifetime", 0) for c in self.coalition_history]
            self.average_coalition_lifetime = np.mean(lifetimes) if lifetimes else 0.0

        # Decay old agent information
        current_time = datetime.now()
        agents_to_remove = []

        for agent_id, agent_info in self.known_agents.items():
            last_seen = agent_info.get("last_seen", current_time)
            age = (current_time - last_seen).total_seconds()

            if age > 300:  # Remove agents not seen for 5 minutes
                agents_to_remove.append(agent_id)

        for agent_id in agents_to_remove:
            del self.known_agents[agent_id]

    def _update_coordination_metrics(self, action: str):
        """Update coordination performance metrics."""
        if action == "coordinate":
            self.metrics["coordination_attempts"] = self.metrics.get("coordination_attempts", 0) + 1
        elif action == "invite":
            self.metrics["invitations_sent"] = self.metrics.get("invitations_sent", 0) + 1
        elif action == "dissolve":
            self.metrics["coalitions_dissolved"] = self.metrics.get("coalitions_dissolved", 0) + 1

        # Update efficiency metrics
        if self.active_coalitions:
            avg_performance = np.mean(
                [c.get("performance", 0) for c in self.active_coalitions.values()]
            )
            self.metrics["avg_coalition_performance"] = avg_performance

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator agent status."""
        status = super().get_status()
        status.update(
            {
                "known_agents": len(self.known_agents),
                "active_coalitions": len(self.active_coalitions),
                "coordination_success_rate": round(self.coordination_success_rate, 3),
                "average_coalition_lifetime": round(self.average_coalition_lifetime, 2),
                "total_coordinated_tasks": self.total_coordinated_tasks,
                "coordination_range": self.coordination_range,
            }
        )
        return status

    def _observation_to_index(self, observation: Dict[str, Any]) -> Optional[int]:
        """Convert observation to PyMDP index."""
        # Determine coalition state based on active coalitions
        if not self.active_coalitions:
            return 0  # Empty
        elif len(self.active_coalitions) == 1:
            coalition = list(self.active_coalitions.values())[0]
            performance = coalition.get("performance", 0.5)
            if performance > 0.8:
                return 4  # Success
            elif performance > 0.6:
                return 2  # Active
            elif performance > 0.3:
                return 1  # Forming
            else:
                return 3  # Conflict
        else:
            # Multiple coalitions
            avg_performance = np.mean(
                [c.get("performance", 0) for c in self.active_coalitions.values()]
            )
            if avg_performance > 0.7:
                return 4  # Success
            elif avg_performance > 0.4:
                return 2  # Active
            else:
                return 3  # Conflict

    @safe_pymdp_operation("free_energy_computation", default_value={})
    def compute_free_energy(self) -> Dict[str, float]:
        """Compute free energy components for coalition coordination with error handling."""
        if not self.pymdp_agent:
            return {}

        try:
            # Use PyMDP's free energy computation
            fe_components = {}

            # Get belief entropy if available
            if hasattr(self.pymdp_agent, "qs") and self.pymdp_agent.qs is not None:
                qs = self.pymdp_agent.qs
                if isinstance(qs, list):
                    # Handle multiple factors
                    belief_entropy = 0
                    for factor in qs:
                        if hasattr(factor, "shape") and factor.size > 0:
                            belief_entropy += -np.sum(factor * np.log(factor + 1e-16))
                else:
                    # Single factor
                    belief_entropy = -np.sum(qs * np.log(qs + 1e-16))
                fe_components["belief_entropy"] = float(belief_entropy)

            # Expected free energy (simplified)
            if hasattr(self.pymdp_agent, "G") and self.pymdp_agent.G is not None:
                fe_components["expected_free_energy"] = float(np.mean(self.pymdp_agent.G))

            return fe_components

        except Exception as e:
            logger.error(f"Free energy computation failed: {e}")
            raise PyMDPError(f"Free energy computation failed: {e}")
