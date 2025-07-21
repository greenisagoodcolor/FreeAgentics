"""Resource Collector Agent implementation.

This agent specializes in finding and collecting resources in the environment
using Active Inference principles to optimize resource gathering efficiency.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import (
    PYMDP_AVAILABLE,
    ActiveInferenceAgent,
    safe_array_to_int,
)
from agents.error_handling import (
    PyMDPError,
    safe_pymdp_operation,
)
from agents.pymdp_error_handling import (
    PyMDPErrorHandler,
    safe_array_index,
    validate_pymdp_matrices,
)

if PYMDP_AVAILABLE:
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent

logger = logging.getLogger(__name__)


class ResourceCollectorAgent(ActiveInferenceAgent):
    """Agent that collects resources using Active Inference.

    This agent:
    - Searches for resources in the environment
    - Maintains beliefs about resource locations
    - Plans efficient collection routes
    - Balances exploration vs exploitation
    """

    def __init__(self, agent_id: str, name: str, grid_size: int = 10):
        """Initialize resource collector agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            grid_size: Size of the grid world
        """
        # Set attributes needed for PyMDP initialization BEFORE calling super()
        self.grid_size = grid_size
        self.position = [grid_size // 2, grid_size // 2]

        # PyMDP state space - must be set before super().__init__
        self.num_states = grid_size * grid_size
        self.num_obs = 6  # Empty, resource, depleted, obstacle, agent, goal
        self.num_actions = 6  # up, down, left, right, collect, return_to_base

        config = {
            "use_pymdp": PYMDP_AVAILABLE,
            "use_llm": True,  # Use LLM for resource value assessment
            "grid_size": grid_size,
            "agent_type": "resource_collector",
        }
        super().__init__(agent_id, name, config)

        # Resource-specific state
        self.collected_resources: Dict[str, int] = {}
        self.resource_memory: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.carrying_capacity = 10
        self.current_load = 0

        self.action_map = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "collect",
            5: "return_to_base",
        }

        # Resource value beliefs
        self.resource_values = {
            "energy": 1.0,
            "material": 0.8,
            "information": 1.2,
        }

        # Collection efficiency metrics
        self.collection_efficiency = 1.0
        self.energy_cost_per_move = 0.1
        self.energy_cost_per_collect = 0.2

        # Initialize additional attributes needed by base class
        self.total_observations = 0
        self.total_actions = 0

        # Initialize attributes from base class that are expected
        self.use_llm = config.get("use_llm", True)
        self.llm_manager = None  # Will be set by base class if available

        # Initialize error handling
        self.pymdp_error_handler = PyMDPErrorHandler(self.agent_id)

    @safe_pymdp_operation("pymdp_init", default_value=None)
    def _initialize_pymdp(self) -> None:
        """Initialize PyMDP agent for resource collection with comprehensive error handling."""
        if not PYMDP_AVAILABLE:
            return

        try:
            # A matrix: P(observation|state) - resource-focused
            A = np.zeros((self.num_obs, self.num_states))

            for state in range(self.num_states):
                # Initialize with base probabilities
                A[0, state] = 0.6  # Empty
                A[1, state] = 0.2  # Resource
                A[2, state] = 0.05  # Depleted
                A[3, state] = 0.05  # Obstacle
                A[4, state] = 0.05  # Agent
                A[5, state] = 0.05  # Goal/base

            # Validate and normalize A matrix
            A = A + 1e-10  # Add epsilon for numerical stability
            A = utils.norm_dist(A)

            # B matrix: P(state_t+1|state_t, action) - movement dynamics
            B = np.zeros((self.num_states, self.num_states, self.num_actions))

            for action in range(self.num_actions):
                for state in range(self.num_states):
                    x, y = state // self.grid_size, state % self.grid_size

                    if action < 4:  # Movement actions
                        next_x, next_y = self._get_next_position(x, y, action)
                        next_state = next_x * self.grid_size + next_y
                        B[next_state, state, action] = 0.9
                        B[state, state, action] = 0.1  # Stay probability
                    else:  # Non-movement actions
                        B[state, state, action] = 1.0

            # Normalize B matrix with validation
            for action in range(self.num_actions):
                for state in range(self.num_states):
                    col_sum = B[:, state, action].sum()
                    if col_sum > 0:
                        B[:, state, action] /= col_sum
                    else:
                        # Handle zero columns
                        B[state, state, action] = 1.0

            # C matrix: Preferences over observations - prefer resources
            C = np.zeros((self.num_obs,))
            C[0] = -0.1  # Slight penalty for empty
            C[1] = 2.0  # Strong preference for resources
            C[2] = -0.5  # Penalty for depleted
            C[3] = -2.0  # Avoid obstacles
            C[4] = 0.0  # Neutral to other agents
            C[5] = 1.0  # Preference for base when loaded

            # D matrix: Initial belief state
            D = utils.norm_dist(np.ones(self.num_states))

            # Validate matrices before creating PyMDP agent
            is_valid, validation_msg = validate_pymdp_matrices(A, B, C, D)
            if not is_valid:
                raise ValueError(f"PyMDP matrix validation failed: {validation_msg}")

            # Create PyMDP agent with simplified configuration
            self.pymdp_agent = PyMDPAgent(
                A=A,
                B=B,
                C=C,
                D=D,
                inference_algo="VANILLA",
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False,
                inference_horizon=3,
            )

            logger.info(f"Initialized PyMDP for resource collector {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize PyMDP: {e}")
            self.pymdp_agent = None

    def _generate_collection_policies(self) -> np.ndarray:
        """Generate resource collection policies."""
        policies = []

        # Exploration patterns
        for start_dir in range(4):
            policy = []
            for step in range(3):
                policy.append((start_dir + step) % 4)
            policies.append(policy)

        # Collection patterns (move then collect)
        for direction in range(4):
            policies.append([direction, 4])  # Move then collect
            policies.append([4, direction])  # Collect then move

        # Return patterns (collect and return)
        policies.append([4, 5])  # Collect then return
        policies.append([5])  # Just return

        # Convert to numpy array and pad to same length
        max_len = max(len(p) for p in policies)
        padded_policies = []
        for policy in policies:
            padded = policy + [policy[-1]] * (max_len - len(policy))  # Repeat last action
            padded_policies.append(padded)

        return np.array(padded_policies)

    def perceive(self, observation: Any) -> None:
        """Process observation focusing on resources.

        Args:
            observation: Environment observation
        """
        self.total_observations += 1

        # Extract resource information
        if isinstance(observation, dict):
            # Update position if provided
            if "position" in observation:
                self.position = observation["position"]

            # Process visible resources
            if "visible_cells" in observation:
                self._update_resource_memory(observation["visible_cells"])

            # Update resource state
            if "resources" in observation:
                self.collected_resources = observation["resources"]

            if "current_load" in observation:
                self.current_load = observation["current_load"]

        # Use LLM for resource value assessment
        if self.use_llm and self.llm_manager:
            observation = self._assess_resource_value(observation)

        # Update PyMDP beliefs with error handling
        if self.pymdp_agent and isinstance(observation, dict):
            obs_idx = self._observation_to_index(observation)
            if obs_idx is not None:
                success, _, error = self.pymdp_error_handler.safe_execute(
                    "state_inference",
                    lambda: self.pymdp_agent.infer_states([obs_idx]),
                    lambda: None,
                )
                if error:
                    logger.warning(f"State inference failed: {error}")

    def _update_resource_memory(self, visible_cells: List[Dict[str, Any]]):
        """Update memory of resource locations."""
        current_time = self.total_steps

        for cell in visible_cells:
            pos = (cell["x"], cell["y"])

            if cell["type"] == "resource":
                # Remember resource location and properties
                self.resource_memory[pos] = {
                    "type": cell.get("resource_type", "unknown"),
                    "amount": cell.get("amount", 1),
                    "last_seen": current_time,
                    "collected": False,
                }
            elif cell["type"] == "empty" and pos in self.resource_memory:
                # Mark as collected if previously had resource
                self.resource_memory[pos]["collected"] = True
                self.resource_memory[pos]["last_seen"] = current_time

    def _assess_resource_value(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to assess resource value and strategy."""
        try:
            visible_resources = [
                cell for cell in observation.get("visible_cells", []) if
                    cell["type"] == "resource"
            ]

            if visible_resources:
                prompt = """As a resource collector agent, analyze these resources:

                Visible resources: {visible_resources}
                Current load: {self.current_load}/{self.carrying_capacity}
                Collected so far: {self.collected_resources}

                Provide:
                1. Value assessment of each resource
                2. Collection priority order
                3. Recommended strategy
                """

                response = self.llm_manager.generate(prompt, max_tokens=200)
                observation["resource_analysis"] = response

        except Exception as e:
            logger.error(f"Resource assessment failed: {e}")

        return observation

    @safe_pymdp_operation("action_selection", default_value="stay")
    def select_action(self) -> Optional[Any]:
        """Select resource collection action with comprehensive error handling.

        Returns:
            Selected action
        """
        if not self.is_active:
            return None

        # Check if at capacity
        if self.current_load >= self.carrying_capacity:
            # Need to return to base
            return "return_to_base"

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
                    (
                        success,
                        action_idx,
                        error,
                    ) = self.pymdp_error_handler.safe_execute(
                        "action_sampling",
                        lambda: self.pymdp_agent.sample_action(),
                        lambda: 4,  # Default to collect
                    )

                    if success:
                        # Convert numpy array to scalar for dictionary lookup
                        action_idx = safe_array_to_int(action_idx)
                        action = safe_array_index(self.action_map, action_idx, "stay")
                    else:
                        logger.warning(f"Action sampling failed: {error}")
                        action = self._fallback_action_selection()
                else:
                    logger.warning(f"Policy inference failed: {error}")
                    action = self._fallback_action_selection()

                # Override if necessary
                if self.current_load >= self.carrying_capacity and action != "return_to_base":
                    action = "return_to_base"

            except Exception as e:
                logger.error(f"PyMDP action selection failed: {e}")
                action = self._fallback_action_selection()
        else:
            action = self._fallback_action_selection()

        self.total_actions += 1
        self.last_action_at = datetime.now()

        # Update metrics
        self._update_collection_metrics(action)

        logger.debug(f"Resource collector {self.agent_id} selected action: {action}")
        return action

    def _fallback_action_selection(self) -> str:
        """Fallback action selection without PyMDP."""
        # Check if standing on resource
        current_pos = tuple(self.position)
        if current_pos in self.resource_memory:
            resource = self.resource_memory[current_pos]
            if not resource["collected"] and self.current_load < self.carrying_capacity:
                return "collect"

        # Find nearest known resource
        nearest_resource = self._find_nearest_resource()
        if nearest_resource:
            return self._move_towards(nearest_resource)

        # Explore if no resources known
        return self._exploration_action()

    def _find_nearest_resource(self) -> Optional[Tuple[int, int]]:
        """Find nearest uncollected resource."""
        min_distance = float("inf")
        nearest = None

        for pos, resource in self.resource_memory.items():
            if not resource["collected"]:
                distance = abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest = pos

        return nearest

    def _move_towards(self, target: Tuple[int, int]) -> str:
        """Move towards target position."""
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        elif dy != 0:
            return "down" if dy > 0 else "up"
        else:
            return "stay"

    def _exploration_action(self) -> str:
        """Select exploration action."""
        # Prefer unexplored areas
        directions = ["up", "down", "left", "right"]
        return np.random.choice(directions)

    @safe_pymdp_operation("belief_update", default_value=None)
    def update_beliefs(self) -> None:
        """Update beliefs about resource locations and values with error handling."""
        # Decay old resource memories
        current_time = self.total_steps
        decay_threshold = 50  # Steps before memory fades

        positions_to_forget = []
        for pos, resource in self.resource_memory.items():
            age = current_time - resource["last_seen"]
            if age > decay_threshold:
                positions_to_forget.append(pos)

        for pos in positions_to_forget:
            del self.resource_memory[pos]

        # Update resource value estimates based on collection success
        if hasattr(self, "last_collected_resource"):
            resource_type = self.last_collected_resource
            # Simple learning: increase value of successfully collected resources
            self.resource_values[resource_type] *= 1.1

    def _update_collection_metrics(self, action: str):
        """Update collection efficiency metrics."""
        if action in ["up", "down", "left", "right"]:
            self.metrics["movement_cost"] = (
                self.metrics.get("movement_cost", 0) + self.energy_cost_per_move
            )
        elif action == "collect":
            self.metrics["collection_cost"] = (
                self.metrics.get("collection_cost", 0) + self.energy_cost_per_collect
            )
            self.metrics["collections"] = self.metrics.get("collections", 0) + 1

        # Calculate efficiency
        if self.metrics.get("collections", 0) > 0:
            total_cost = self.metrics.get("movement_cost", 0) + self.metrics.get(
                "collection_cost", 0
            )
            self.collection_efficiency = self.metrics["collections"] / max(total_cost,
                1.0)

    def get_status(self) -> Dict[str, Any]:
        """Get collector agent status."""
        status = super().get_status()
        status.update(
            {
                "collected_resources": self.collected_resources,
                "current_load": f"{self.current_load}/{self.carrying_capacity}",
                "known_resources": len(
                    [r for r in self.resource_memory.values() if not r["collected"]]
                ),
                "collection_efficiency": round(self.collection_efficiency, 3),
                "resource_memory_size": len(self.resource_memory),
            }
        )
        return status

    def _get_next_position(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Get next position based on action."""
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)

        return x, y

    def _observation_to_index(self, observation: Dict[str, Any]) -> Optional[int]:
        """Convert observation to PyMDP index."""
        if "cell_type" not in observation:
            return None

        cell_type = observation["cell_type"]
        mapping = {
            "empty": 0,
            "resource": 1,
            "depleted": 2,
            "obstacle": 3,
            "agent": 4,
            "goal": 5,
        }

        return mapping.get(cell_type)

    @safe_pymdp_operation("free_energy_computation", default_value={})
    def compute_free_energy(self) -> Dict[str, float]:
        """Compute free energy components for Active Inference with error handling."""
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
