"""Base Active Inference Agent implementation using PyMDP."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agents.error_handling import (
    ActionSelectionError,
    ErrorHandler,
    InferenceError,
    safe_pymdp_operation,
    validate_action,
    validate_observation,
    with_error_handling,
)
from agents.performance_optimizer import PerformanceOptimizer, performance_monitor
from agents.pymdp_error_handling import (
    PyMDPErrorHandler,
    safe_array_index,
    safe_numpy_conversion,
    validate_pymdp_matrices,
)

logger = logging.getLogger(__name__)


def safe_array_to_int(value):
    """Safely convert numpy array or scalar to integer.

    Handles various PyMDP return types that might be numpy arrays,
    scalars, or other array-like objects. Maintains backward compatibility
    by raising exceptions for invalid inputs.

    Args:
        value: Value to convert to integer

    Returns:
        Integer value

    Raises:
        ValueError: If value cannot be converted to integer
    """
    try:
        # Handle numpy arrays specifically
        if hasattr(value, "ndim"):
            # This is a numpy array
            if value.ndim == 0:
                # 0-dimensional array (scalar)
                return int(value.item())
            elif value.size == 0:
                raise ValueError("Empty array cannot be converted to integer")
            elif value.size == 1:
                # Single element array
                return int(value.item())
            else:
                # Multi-element array - take first element
                return int(value.flat[0])  # Use flat to handle any shape
        elif hasattr(value, "__len__") and hasattr(value, "__getitem__"):
            # Other array-like objects (lists, tuples, etc.)
            if len(value) == 0:
                raise ValueError("Empty array cannot be converted to integer")
            else:
                return int(value[0])
        elif hasattr(value, "item"):
            # numpy scalar
            return int(value.item())
        else:
            # regular scalar
            return int(value)
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError(f"Cannot convert {type(value)} value {value} to integer: {e}")


try:
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent

    PYMDP_AVAILABLE = True
except ImportError:
    logger.warning("PyMDP not available - using simplified implementation")
    PYMDP_AVAILABLE = False

# Import observability integration
try:
    from observability import (
        monitor_pymdp_inference,
        record_agent_lifecycle_event,
        record_belief_update,
    )
    from observability.belief_monitoring import monitor_belief_update

    OBSERVABILITY_AVAILABLE = True
    BELIEF_MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("Observability integration not available")
    OBSERVABILITY_AVAILABLE = False
    BELIEF_MONITORING_AVAILABLE = False

    # Mock observability functions
    def monitor_pymdp_inference(agent_id: str):
        def decorator(func):
            return func

        return decorator

    async def record_belief_update(
        agent_id: str, beliefs_before: dict, beliefs_after: dict, free_energy: float = None
    ):
        pass

    async def record_agent_lifecycle_event(agent_id: str, event: str, metadata: dict = None):
        pass

    async def monitor_belief_update(
        agent_id: str, beliefs: dict, free_energy: float = None, metadata: dict = None
    ):
        pass


# Import GMN for model specification
try:
    from inference.active.gmn_parser import GMNParser, parse_gmn_spec

    GMN_AVAILABLE = True
except ImportError:
    logger.warning("GMN parser not available")
    GMN_AVAILABLE = False

# Import LLM manager
try:
    from inference.llm.local_llm_manager import LocalLLMConfig, LocalLLMManager

    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLM manager not available")
    LLM_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for an Active Inference agent."""

    name: str
    use_pymdp: bool = True
    planning_horizon: int = 3
    precision: float = 1.0
    lr: float = 0.1
    gmn_spec: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None


class ActiveInferenceAgent(ABC):
    """Base class for Active Inference agents.

    This implements the core Active Inference loop:
    1. Perceive observations from environment
    2. Update beliefs about hidden states
    3. Select actions to minimize expected free energy
    4. Act in the environment
    """

    def __init__(self, agent_id: str, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize an Active Inference agent.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            config: Configuration parameters
        """
        self.agent_id = agent_id
        self.name = name
        self.config = config or {}

        # Agent state
        self.is_active = False
        self.created_at = datetime.now()
        self.last_action_at = None
        self.total_steps = 0

        # Performance optimization settings
        self.performance_mode = config.get(
            "performance_mode", "balanced"
        )  # "fast", "balanced", "accurate"
        self.selective_update_interval = config.get(
            "selective_update_interval", 1
        )  # Update beliefs every N steps
        self.matrix_cache = {}  # Cache for normalized matrices

        # Active Inference components
        self.pymdp_agent = None  # PyMDP agent instance
        self.beliefs = {}  # Beliefs about hidden states
        self.preferences = {}  # Preferred observations
        self.policies = []  # Available action policies

        # GMN and LLM integration
        self.gmn_spec = None  # GMN specification for the agent
        if LLM_AVAILABLE:
            llm_config_dict = self.config.get("llm_config", {})
            # Create LocalLLMConfig with default values and user overrides
            llm_config = LocalLLMConfig()
            for key, value in llm_config_dict.items():
                if hasattr(llm_config, key):
                    setattr(llm_config, key, value)
            self.llm_manager = LocalLLMManager(llm_config)
        else:
            self.llm_manager = None

        # Initialize PyMDP if available
        if PYMDP_AVAILABLE and self.config.get("use_pymdp", True):
            self._initialize_pymdp()

        # Metrics
        self.metrics = {
            "total_observations": 0,
            "total_actions": 0,
            "avg_free_energy": 0.0,
            "belief_entropy": 0.0,
        }

        # Observability integration
        self.observability_enabled = OBSERVABILITY_AVAILABLE and config.get(
            "enable_observability", True
        )
        
        # Belief monitoring integration
        self.belief_monitoring_enabled = BELIEF_MONITORING_AVAILABLE and config.get(
            "enable_belief_monitoring", True
        )

        # Error handling
        self.error_handler = ErrorHandler(self.agent_id)
        self.pymdp_error_handler = PyMDPErrorHandler(self.agent_id)

        # PERFORMANCE OPTIMIZATION: Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer()
        self.performance_metrics = {}

        logger.info(
            f"Created agent {self.agent_id} ({self.name}) - PyMDP: {PYMDP_AVAILABLE}, Performance Mode: {self.performance_mode}"
        )

        # Record agent creation in observability system
        if self.observability_enabled:
            import asyncio

            try:
                asyncio.create_task(
                    record_agent_lifecycle_event(
                        self.agent_id,
                        "created",
                        {
                            "name": self.name,
                            "pymdp_available": PYMDP_AVAILABLE,
                            "performance_mode": self.performance_mode,
                            "config_keys": list(self.config.keys()),
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to record agent creation event: {e}")

    @abstractmethod
    def perceive(self, observation: Any) -> None:
        """Process an observation from the environment.

        Args:
            observation: Observation from the environment
        """
        pass

    def _initialize_pymdp(self) -> None:
        """Initialize PyMDP agent with default configuration."""
        # This will be overridden by subclasses with specific models
        pass

    def _get_policy_length(self) -> int:
        """Get adaptive policy length based on performance mode."""
        performance_mode = getattr(self, "performance_mode", "balanced")
        if performance_mode == "fast":
            return 1  # Single-step planning for maximum speed
        elif performance_mode == "balanced":
            return 2  # Reduced planning for good speed/accuracy balance
        else:  # "accurate"
            return 3  # Full planning horizon for maximum accuracy

    def _get_param_info_gain(self) -> bool:
        """Get parameter info gain setting based on performance mode."""
        # Disable parameter learning in fast mode for speed
        performance_mode = getattr(self, "performance_mode", "balanced")
        return performance_mode != "fast"

    def _get_gamma(self) -> float:
        """Get adaptive gamma (policy precision) based on performance mode."""
        performance_mode = getattr(self, "performance_mode", "balanced")
        if performance_mode == "fast":
            return 8.0  # Lower precision for faster computation
        elif performance_mode == "balanced":
            return 12.0  # Moderate precision
        else:  # "accurate"
            return 16.0  # High precision for accuracy

    def _get_alpha(self) -> float:
        """Get adaptive alpha (action precision) based on performance mode."""
        performance_mode = getattr(self, "performance_mode", "balanced")
        if performance_mode == "fast":
            return 8.0  # Lower precision for faster computation
        elif performance_mode == "balanced":
            return 12.0  # Moderate precision
        else:  # "accurate"
            return 16.0  # High precision for accuracy

    def _should_update_beliefs(self) -> bool:
        """Determine if beliefs should be updated this step (selective updating)."""
        selective_update_interval = getattr(self, "selective_update_interval", 1)
        total_steps = getattr(self, "total_steps", 0)
        if selective_update_interval <= 1:
            return True
        return (total_steps % selective_update_interval) == 0

    def _get_cached_matrix(
        self, matrix_name: str, matrix_data: Any, normalization_func: callable
    ) -> Any:
        """Get cached normalized matrix or compute and cache it."""
        # Initialize matrix_cache if it doesn't exist (during construction)
        if not hasattr(self, "matrix_cache"):
            self.matrix_cache = {}

        cache_key = f"{matrix_name}_{hash(str(matrix_data))}"

        if cache_key not in self.matrix_cache:
            self.matrix_cache[cache_key] = normalization_func(matrix_data)
            logger.debug(f"Cached matrix {matrix_name} for agent {self.agent_id}")

        return self.matrix_cache[cache_key]

    def load_gmn_spec(self, gmn_spec: str) -> None:
        """Load a GMN specification for the agent.

        Args:
            gmn_spec: GMN specification string
        """
        if not GMN_AVAILABLE:
            logger.error("GMN parser not available")
            return

        try:
            self.gmn_spec = parse_gmn_spec(gmn_spec)
            logger.info(f"Loaded GMN spec for agent {self.agent_id}")

            # Reinitialize PyMDP with new spec
            if PYMDP_AVAILABLE:
                self._initialize_pymdp_from_gmn()
        except Exception as e:
            logger.error(f"Failed to parse GMN spec: {e}")

    def _initialize_pymdp_from_gmn(self) -> None:
        """Initialize PyMDP agent from GMN specification."""
        if not self.gmn_spec or not PYMDP_AVAILABLE:
            return

        try:
            # Extract PyMDP model components from GMN spec
            num_states = self.gmn_spec.get("num_states", [4])  # Default 4 states
            num_obs = self.gmn_spec.get("num_obs", [4])  # Default 4 observations
            num_actions = self.gmn_spec.get("num_actions", [4])  # Default 4 actions

            # Get matrices from GMN spec or use defaults
            A_matrices = self.gmn_spec.get("A", [])
            B_matrices = self.gmn_spec.get("B", [])
            C_vectors = self.gmn_spec.get("C", [])
            D_vectors = self.gmn_spec.get("D", [])

            # Create default matrices if not provided
            if not A_matrices:
                A = np.eye(num_obs[0], num_states[0])  # Default identity observation model
                A_matrices = [A]

            if not B_matrices:
                B = np.zeros((num_states[0], num_states[0], num_actions[0]))
                for a in range(num_actions[0]):
                    B[:, :, a] = np.eye(num_states[0])  # Default identity transition
                B_matrices = [B]

            if not C_vectors:
                C = np.zeros(num_obs[0])
                C[0] = 1.0  # Prefer first observation
                C_vectors = [C]

            if not D_vectors:
                D = utils.norm_dist(np.ones(num_states[0]))  # Uniform prior
                D_vectors = [D]

            # Create PyMDP agent with GMN-specified model
            self.pymdp_agent = PyMDPAgent(
                A=A_matrices,
                B=B_matrices,
                C=C_vectors,
                D=D_vectors,
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False,
                inference_horizon=self.config.get("planning_horizon", 3),
            )

            logger.info(f"Successfully initialized PyMDP agent from GMN spec for {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize PyMDP from GMN: {e}")
            self.pymdp_agent = None

    @abstractmethod
    def update_beliefs(self) -> None:
        """Update beliefs about hidden states using Active Inference."""
        pass

    @abstractmethod
    def select_action(self) -> Any:
        """Select action to minimize expected free energy.

        Returns:
            Selected action
        """
        pass

    @with_error_handling("agent_step", fallback_result="stay")
    def step(self, observation: Any) -> Any:
        """Execute one step of the Active Inference loop.

        Args:
            observation: Current observation from environment

        Returns:
            Selected action
        """
        # OBSERVABILITY: Monitor inference performance if enabled
        if self.observability_enabled and OBSERVABILITY_AVAILABLE:
            return self._step_with_monitoring(observation)
        else:
            return self._step_implementation(observation)

    def _step_with_monitoring(self, observation: Any) -> Any:
        """Step implementation with observability monitoring."""
        # For now, skip monitoring to avoid async issues in sync context
        # TODO: Implement proper async handling for monitoring
        return self._step_implementation(observation)

    def _step_implementation(self, observation: Any) -> Any:
        """Core step implementation."""
        if not self.is_active:
            raise RuntimeError(f"Agent {self.agent_id} is not active")

        try:
            # Validate and sanitize observation
            observation = validate_observation(observation)

            # Use LLM to process observation if available
            if self.llm_manager and self.config.get("use_llm", False):
                observation = self._process_observation_with_llm(observation)

            # Active Inference loop with error handling
            self.perceive(observation)
            self.update_beliefs()
            action = self.select_action()

            # Validate action
            if hasattr(self, "actions"):
                action = validate_action(action, self.actions)
            elif hasattr(self, "action_map"):
                action = validate_action(action, list(self.action_map.values()))

            # Compute and store free energy after belief update
            if self.pymdp_agent and PYMDP_AVAILABLE:
                fe_components = self.compute_free_energy()
                if fe_components and "total_free_energy" in fe_components:
                    self.metrics.update(fe_components)

            # Update metrics
            self.total_steps += 1
            self.last_action_at = datetime.now()
            self.metrics["total_observations"] += 1
            self.metrics["total_actions"] += 1

            logger.debug(f"Agent {self.agent_id} step {self.total_steps}: {action}")

            return action

        except Exception as e:
            # Log error and return safe fallback action
            logger.error(f"Agent {self.agent_id} step failed: {e}")
            self.error_handler.handle_error(e, "agent_step")
            return "stay"

    def _process_observation_with_llm(self, observation: Any) -> Any:
        """Process observation using LLM for enhanced understanding.

        Args:
            observation: Raw observation

        Returns:
            Processed observation
        """
        try:
            prompt = """As an Active Inference agent, interpret this observation:
            {observation}

            Provide structured analysis for belief updating."""

            response = self.llm_manager.generate(prompt, max_tokens=200)
            logger.debug(f"LLM interpretation: {response}")

            # Add LLM interpretation to observation
            if isinstance(observation, dict):
                observation["llm_interpretation"] = response

        except Exception as e:
            logger.error(f"LLM processing failed: {e}")

        return observation

    def start(self):
        """Start the agent."""
        # Initialize PyMDP if needed
        if hasattr(self, "_initialize_pymdp") and self.pymdp_agent is None:
            self._initialize_pymdp()
        self.is_active = True
        logger.info(f"Agent {self.agent_id} started")

    def stop(self):
        """Stop the agent."""
        self.is_active = False
        
        # Cleanup belief monitoring
        if self.belief_monitoring_enabled:
            try:
                from observability.belief_monitoring import belief_monitoring_hooks
                belief_monitoring_hooks.reset_agent_monitor(self.agent_id)
                logger.debug(f"Cleaned up belief monitoring for agent {self.agent_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup belief monitoring for agent {self.agent_id}: {e}")
        
        logger.info(f"Agent {self.agent_id} stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.

        Returns:
            Status dictionary
        """
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_action_at": self.last_action_at.isoformat() if self.last_action_at else None,
            "total_steps": self.total_steps,
            "metrics": self.metrics,
        }

        # Add error handling status
        if hasattr(self, "error_handler"):
            status["error_summary"] = self.error_handler.get_error_summary()

        # Add PyMDP-specific error handling status
        if hasattr(self, "pymdp_error_handler"):
            status["pymdp_error_report"] = self.pymdp_error_handler.get_error_report()

        return status
    
    def get_belief_monitoring_stats(self) -> Dict[str, Any]:
        """Get belief monitoring statistics for this agent.
        
        Returns:
            Dictionary containing belief monitoring statistics
        """
        if not self.belief_monitoring_enabled:
            return {"error": "Belief monitoring not enabled"}
        
        try:
            from observability.belief_monitoring import belief_monitoring_hooks
            return belief_monitoring_hooks.get_agent_statistics(self.agent_id)
        except Exception as e:
            logger.error(f"Failed to get belief monitoring stats for agent {self.agent_id}: {e}")
            return {"error": str(e)}


class BasicExplorerAgent(ActiveInferenceAgent):
    """Explorer agent using PyMDP Active Inference.

    This agent explores a 2D grid environment using Active Inference
    to minimize expected free energy and reduce uncertainty.
    """

    def __init__(self, agent_id: str, name: str, grid_size: int = 10):
        # Initialize attributes BEFORE calling parent constructor
        self.grid_size = grid_size
        self.position = [grid_size // 2, grid_size // 2]  # Start in center

        # State space for PyMDP
        self.num_states = grid_size * grid_size  # Each grid cell is a state
        self.num_obs = 5  # Empty, obstacle, goal, agent, out-of-bounds
        self.num_actions = 5  # up, down, left, right, stay

        # Map actions to indices
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}
        self.actions = list(self.action_map.values())

        # Set config for parent init
        config = {
            "use_pymdp": PYMDP_AVAILABLE,
            "use_llm": False,  # Can be enabled for enhanced perception
            "grid_size": grid_size,
            "performance_mode": "fast",  # Enable fast mode for better performance
            "selective_update_interval": 2,  # Update beliefs every 2 steps
        }

        # Now call parent constructor - this will call _initialize_pymdp()
        super().__init__(agent_id, name, config)

        # Simplified belief state (also maintained for non-PyMDP fallback)
        self.uncertainty_map = np.ones((grid_size, grid_size))
        self.uncertainty_map[self.position[0], self.position[1]] = 0

        # Preferences: prefer to reduce uncertainty and find goals
        self.exploration_rate = 0.3

    def _initialize_pymdp(self) -> None:
        """Initialize PyMDP agent for grid world exploration."""
        if not PYMDP_AVAILABLE:
            return

        try:
            # A matrix: P(observation|state) - likelihood mapping
            # Shape: (num_observations, num_states)
            A = np.zeros((self.num_obs, self.num_states))

            # For each state (grid position), define observation likelihood
            for state in range(self.num_states):
                # Most states observe "empty"
                A[0, state] = 0.8  # Empty
                A[1, state] = 0.05  # Obstacle
                A[2, state] = 0.05  # Goal
                A[3, state] = 0.05  # Agent
                A[4, state] = 0.05  # Out of bounds

            # Add noise to observation model for robustness
            A = A + 0.01  # Small epsilon for numerical stability

            # Normalize - utils.norm_dist normalizes along first dimension
            # PERFORMANCE OPTIMIZATION: Cache matrix normalization
            A = self._get_cached_matrix("A_observation", A, utils.norm_dist)

            # B matrix: P(next_state|state,action) - transition dynamics
            # Shape: (num_states, num_states, num_actions)
            # PERFORMANCE OPTIMIZATION: Pre-allocate with correct dtype
            B = np.zeros((self.num_states, self.num_states, self.num_actions), dtype=np.float32)

            # PERFORMANCE OPTIMIZATION: Vectorized transition matrix construction
            # Pre-compute state coordinates
            states_x = np.arange(self.num_states) // self.grid_size
            states_y = np.arange(self.num_states) % self.grid_size

            # Define action deltas for vectorized computation
            action_deltas = {
                0: (-1, 0),  # up
                1: (1, 0),  # down
                2: (0, -1),  # left
                3: (0, 1),  # right
                4: (0, 0),  # stay
            }

            # Vectorized transition computation
            for action_idx, (dx, dy) in action_deltas.items():
                # Calculate next states
                next_x = states_x + dx
                next_y = states_y + dy

                # Check bounds
                valid_mask = (
                    (next_x >= 0)
                    & (next_x < self.grid_size)
                    & (next_y >= 0)
                    & (next_y < self.grid_size)
                )

                # Set transitions
                for state in range(self.num_states):
                    if valid_mask[state]:
                        next_state = next_x[state] * self.grid_size + next_y[state]
                        B[next_state, state, action_idx] = 0.9
                        B[state, state, action_idx] = 0.1
                    else:
                        B[state, state, action_idx] = 1.0

            # PERFORMANCE OPTIMIZATION: Vectorized normalization
            # Use einsum for faster column normalization
            col_sums = np.sum(B, axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            B = B / col_sums

            # C matrix: P(observation) - observation preferences
            # These encode the agent's goals/desires in Active Inference
            # Higher values = more preferred observations
            C = np.zeros(self.num_obs)
            C[0] = 2.0  # Slight preference for empty spaces (exploration)
            C[1] = -4.0  # Strong aversion to obstacles
            C[2] = 8.0  # Strong preference for goals
            C[3] = 0.0  # Neutral about other agents
            C[4] = -4.0  # Aversion to out-of-bounds

            # D matrix: P(initial_state) - initial state distribution
            D = np.ones(self.num_states) / self.num_states  # Uniform prior

            # Add slight bias toward current position if known
            current_state = self.position[0] * self.grid_size + self.position[1]
            D[current_state] *= 2.0
            D = D / D.sum()  # Normalize

            # Validate matrices before creating PyMDP agent
            is_valid, validation_msg = validate_pymdp_matrices(A, B, C, D)
            if not is_valid:
                raise ValueError(f"PyMDP matrix validation failed: {validation_msg}")

            # PERFORMANCE OPTIMIZATION: Create PyMDP agent with optimized settings
            self.pymdp_agent = PyMDPAgent(
                A=A,
                B=B,
                C=C,
                D=D,
                policy_len=self._get_policy_length(),  # Adaptive planning horizon
                inference_algo="VANILLA",  # Standard variational inference
                use_states_info_gain=True,  # Epistemic value (curiosity)
                use_param_info_gain=self._get_param_info_gain(),  # Adaptive learning signal
                use_utility=True,  # Pragmatic value (goal-seeking)
                gamma=self._get_gamma(),  # Adaptive policy precision
                alpha=self._get_alpha(),  # Adaptive action precision
                inference_horizon=1,  # Single-step state inference
                control_fac_idx=None,  # All factors are controllable
                save_belief_hist=False,  # PERFORMANCE: Disable history for speed
            )

            logger.info(f"Initialized PyMDP agent for {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize PyMDP: {e}")
            self.pymdp_agent = None

    def perceive(self, observation: Dict[str, Any]) -> None:
        """Process observation about current position.

        This implements the perception step in Active Inference:
        1. Convert raw observations to categorical observations
        2. Update agent's internal observation history
        3. Prepare for belief update through variational inference

        Args:
            observation: Dict with 'position' and 'surroundings' keys
        """
        if "position" in observation:
            self.position = observation["position"]

        # Convert observation to PyMDP format if available
        if self.pymdp_agent and PYMDP_AVAILABLE:
            # Map surroundings to observation index
            surroundings = np.array(observation.get("surroundings", []))

            # Simplified mapping: categorize center cell observation
            if surroundings.size > 0:
                center_val = surroundings[1, 1] if surroundings.shape == (3, 3) else 0

                # Map to observation categories for Active Inference
                if center_val == -2:  # Out of bounds
                    obs_idx = 4
                elif center_val == -1:  # Obstacle
                    obs_idx = 1
                elif center_val == 1:  # Goal
                    obs_idx = 2
                elif center_val == 2:  # Other agent
                    obs_idx = 3
                else:  # Empty
                    obs_idx = 0

                # Store observation for belief update
                # PyMDP expects observations as a list for multiple modalities
                self.current_observation = [obs_idx]

                # Update metrics
                self.metrics["last_observation"] = obs_idx
        else:
            # Fallback: update uncertainty map
            x, y = self.position
            self.uncertainty_map[x, y] = 0

            # Slightly reduce uncertainty in neighboring cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.uncertainty_map[nx, ny] *= 0.8

        # Note: total_observations is incremented in the base step() method

    @performance_monitor("belief_update")
    @safe_pymdp_operation("belief_update", default_value=None)
    def update_beliefs(self) -> None:
        """PERFORMANCE OPTIMIZED: Update beliefs using Active Inference.

        Optimizations applied:
        - Selective belief updates based on update interval
        - Fast belief entropy computation
        - Cached belief conversion
        - Reduced free energy calculations
        """
        # PERFORMANCE OPTIMIZATION: Skip updates based on interval
        if not self._should_update_beliefs():
            return
        if self.pymdp_agent and PYMDP_AVAILABLE:
            try:
                # OBSERVABILITY: Record beliefs before update
                beliefs_before = {}
                if (
                    self.observability_enabled
                    and hasattr(self.pymdp_agent, "qs")
                    and self.pymdp_agent.qs is not None
                ):
                    try:
                        beliefs_before = {
                            "belief_entropy": self.metrics.get("belief_entropy", 0.0),
                            "avg_free_energy": self.metrics.get("avg_free_energy", 0.0),
                            "state_posterior_size": (
                                len(self.pymdp_agent.qs) if self.pymdp_agent.qs else 0
                            ),
                        }
                    except Exception as e:
                        logger.debug(f"Failed to capture beliefs before update: {e}")

                # Perform variational inference with error handling
                if hasattr(self, "current_observation"):
                    success, _, error = self.pymdp_error_handler.safe_execute(
                        "state_inference",
                        lambda: self.pymdp_agent.infer_states(self.current_observation),
                        lambda: None,  # No fallback for inference - will use existing beliefs
                    )

                    if not success and error:
                        logger.warning(f"State inference failed: {error}")

                # PERFORMANCE OPTIMIZATION: Fast belief processing
                if hasattr(self.pymdp_agent, "qs") and self.pymdp_agent.qs is not None:
                    qs = self.pymdp_agent.qs  # Posterior beliefs over states

                    # PERFORMANCE: Optimized entropy calculation using vectorized operations
                    try:
                        # Use numpy's built-in entropy calculation for speed
                        epsilon = 1e-10
                        if len(qs) == 1:  # Single factor case (common)
                            factor = qs[0]
                            entropy = -np.sum(factor * np.log(factor + epsilon))
                        else:  # Multiple factors
                            entropy = sum(
                                -np.sum(factor * np.log(factor + epsilon)) for factor in qs
                            )

                        self.metrics["belief_entropy"] = float(entropy)

                        # PERFORMANCE: Only store beliefs if explicitly needed (debug mode)
                        if self.config.get("debug_mode", False):
                            self.beliefs["state_posterior"] = [q.tolist() for q in qs]

                        # PERFORMANCE: Reduced free energy computation frequency
                        if self.total_steps % 10 == 0:  # Only every 10 steps
                            if hasattr(self.pymdp_agent, "F") and self.pymdp_agent.F is not None:
                                self.metrics["avg_free_energy"] = float(np.mean(self.pymdp_agent.F))

                        # OBSERVABILITY: Record beliefs after update
                        if self.observability_enabled and beliefs_before:
                            try:
                                beliefs_after = {
                                    "belief_entropy": self.metrics.get("belief_entropy", 0.0),
                                    "avg_free_energy": self.metrics.get("avg_free_energy", 0.0),
                                    "state_posterior_size": len(qs) if qs else 0,
                                }

                                # Record belief update asynchronously
                                import asyncio

                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        # Use new belief monitoring system if available
                                        if self.belief_monitoring_enabled:
                                            # Create comprehensive belief state for monitoring
                                            beliefs_state = {
                                                "qs": qs,
                                                "entropy": self.metrics.get("belief_entropy", 0.0),
                                                "state_posterior_size": len(qs) if qs else 0,
                                                "previous_entropy": beliefs_before.get("belief_entropy", 0.0),
                                            }
                                            
                                            # Monitor belief update with detailed tracking
                                            asyncio.create_task(
                                                monitor_belief_update(
                                                    self.agent_id,
                                                    beliefs_state,
                                                    beliefs_after.get("avg_free_energy"),
                                                    {
                                                        "step": self.total_steps,
                                                        "update_type": "pymdp_inference",
                                                        "beliefs_before": beliefs_before,
                                                        "beliefs_after": beliefs_after,
                                                    }
                                                )
                                            )
                                        
                                        # Also record basic metrics
                                        asyncio.create_task(
                                            record_belief_update(
                                                self.agent_id,
                                                beliefs_before,
                                                beliefs_after,
                                                beliefs_after.get("avg_free_energy"),
                                            )
                                        )
                                except RuntimeError:
                                    # No event loop running, skip async recording
                                    pass
                            except Exception as e:
                                logger.debug(f"Failed to record belief update: {e}")

                    except Exception as e:
                        logger.warning(f"Fast belief processing failed: {e}, using fallback")
                        self.metrics["belief_entropy"] = 0.0

            except Exception as e:
                logger.warning(f"PyMDP belief update failed: {e}, using fallback")
                raise InferenceError(f"PyMDP belief update failed: {e}")

        else:
            # Fallback: simple uncertainty update
            self._fallback_update_beliefs()

    def _fallback_update_beliefs(self) -> None:
        """Fallback belief update when PyMDP fails."""
        self.metrics["belief_entropy"] = np.mean(self.uncertainty_map)

        # Increase uncertainty over time (world changes)
        self.uncertainty_map *= 1.01
        self.uncertainty_map = np.clip(self.uncertainty_map, 0, 1)

    @performance_monitor("action_selection")
    @safe_pymdp_operation("action_selection", default_value="stay")
    def select_action(self) -> str:
        """Select action to minimize expected free energy.

        Uses Active Inference to select actions that:
        1. Maximize information gain (epistemic value)
        2. Achieve preferred outcomes (pragmatic value)

        Returns:
            Selected action (up, down, left, right, stay)
        """
        if self.pymdp_agent and PYMDP_AVAILABLE:
            try:
                # Use PyMDP to infer optimal policy
                q_pi, G = self.pymdp_agent.infer_policies()

                # Sample action from the posterior over policies
                action_idx = self.pymdp_agent.sample_action()

                # Use robust PyMDP error handling for action conversion
                try:
                    action_idx_converted = safe_array_to_int(action_idx)
                except Exception as e:
                    # Use PyMDP error handler for fallback
                    success, action_idx_converted, error = self.pymdp_error_handler.safe_execute(
                        "action_index_conversion",
                        lambda: safe_numpy_conversion(action_idx, int, 4),
                        lambda: 4,  # Default to "stay" action index
                    )
                    if error:
                        logger.warning(f"Action index conversion failed, using fallback: {error}")

                # Convert action index to string with safe indexing
                selected_action = safe_array_index(self.action_map, action_idx_converted, "stay")

                # Store expected free energy for analysis
                if G is not None:
                    # Expected free energy combines epistemic and pragmatic value:
                    # G = -E_q[log P(o|s)] + E_q[H[P(s|s',u)]] - E_q[log P(o)]
                    # where the first term is pragmatic (goal-seeking) value
                    # and the second term is epistemic (information-seeking) value
                    self.metrics["expected_free_energy"] = float(np.min(G))
                    self.metrics["policy_posterior"] = (
                        q_pi.tolist() if hasattr(q_pi, "tolist") else q_pi
                    )

                    # Track which policy was selected for analysis with error handling
                    success, best_policy_idx, error = self.pymdp_error_handler.safe_execute(
                        "policy_index_extraction",
                        lambda: (
                            safe_array_to_int(np.argmax(q_pi)) if hasattr(q_pi, "__len__") else 0
                        ),
                        lambda: 0,  # Default policy index
                    )

                    self.metrics["selected_policy"] = best_policy_idx if success else 0

                return selected_action

            except Exception as e:
                logger.warning(f"PyMDP action selection failed: {e}, using fallback")
                raise ActionSelectionError(f"PyMDP action selection failed: {e}")

        # This will trigger the fallback in the decorator
        return self._fallback_action_selection()

    def _fallback_select_action(self) -> str:
        """Fallback action selection when PyMDP fails (called by decorator)."""
        return self._fallback_action_selection()

    def _fallback_action_selection(self) -> str:
        """Fallback action selection when PyMDP fails."""
        # Fallback: simple uncertainty-based action selection
        x, y = self.position

        # Calculate expected uncertainty reduction for each action
        action_values = {}

        for action in self.actions:
            # Simulate action
            new_x, new_y = self._simulate_action(x, y, action)

            # Expected uncertainty reduction
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                # Combine uncertainty reduction with goal preference
                uncertainty_reduction = self.uncertainty_map[new_x, new_y]

                # Add preference for goals if we know about them
                goal_bonus = 0
                if hasattr(self, "known_goals"):
                    if (new_x, new_y) in self.known_goals:
                        goal_bonus = 0.5

                action_values[action] = uncertainty_reduction + goal_bonus
            else:
                action_values[action] = -1  # Invalid action

        # Select action with highest expected value
        # Add some exploration noise
        if np.random.random() < self.exploration_rate:
            valid_actions = [a for a, v in action_values.items() if v >= 0]
            selected_action = np.random.choice(valid_actions) if valid_actions else "stay"
        else:
            selected_action = max(action_values, key=action_values.get)

        return selected_action

    def _simulate_action(self, x: int, y: int, action: str) -> Tuple[int, int]:
        """Simulate the result of an action.

        Args:
            x, y: Current position
            action: Action to simulate

        Returns:
            New position after action
        """
        if action == "up":
            return x - 1, y
        elif action == "down":
            return x + 1, y
        elif action == "left":
            return x, y - 1
        elif action == "right":
            return x, y + 1
        else:
            return x, y

    def compute_free_energy(self) -> Dict[str, float]:
        """Compute components of variational free energy.

        Returns:
            Dictionary with free energy components:
            - total_free_energy: Total variational free energy
            - accuracy: Expected log likelihood (negative surprise)
            - complexity: KL divergence between posterior and prior
        """
        if not (self.pymdp_agent and PYMDP_AVAILABLE):
            return {"error": "PyMDP not available"}

        if not hasattr(self.pymdp_agent, "qs") or self.pymdp_agent.qs is None:
            return {"error": "No beliefs available"}

        try:
            # Get current beliefs and observation
            qs = self.pymdp_agent.qs[0]  # Single factor for grid world

            # Get observation model and current observation
            A = self.pymdp_agent.A[0]  # Single modality
            if hasattr(self, "current_observation"):
                obs_idx = self.current_observation[0]  # Extract from list
            else:
                obs_idx = 0  # Default to empty

            # Compute accuracy term: E_q[log P(o|s)]
            # This is the expected log likelihood under posterior beliefs
            likelihood = A[obs_idx, :]  # P(o|s) for current observation
            epsilon = 1e-10
            accuracy = np.sum(qs * np.log(likelihood + epsilon))

            # Get prior beliefs
            if hasattr(self.pymdp_agent, "D"):
                prior = self.pymdp_agent.D[0]
            else:
                prior = np.ones_like(qs) / len(qs)

            # Compute complexity term: KL[q(s)||p(s)]
            # This measures how much the posterior deviates from prior
            complexity = np.sum(qs * (np.log(qs + epsilon) - np.log(prior + epsilon)))

            # Total variational free energy
            # F = complexity - accuracy
            free_energy = complexity - accuracy

            return {
                "total_free_energy": float(free_energy),
                "accuracy": float(accuracy),
                "complexity": float(complexity),
                "surprise": float(
                    -np.log(likelihood[np.argmax(qs)] + epsilon)
                ),  # Negative log likelihood at MAP estimate
            }

        except Exception as e:
            logger.error(f"Failed to compute free energy: {e}")
            return {"error": str(e)}
