"""Active Inference Configuration Module (Task 44.1).

This module provides configuration and setup utilities for the PyMDP-based
active inference engine, implementing the Nemesis Committee's recommendations
for clean architecture and comprehensive validation.

Based on:
- Robert C. Martin: Clean configuration with single responsibility
- Jessica Kerr: Comprehensive observability configuration
- Sindre Sorhus: High-quality validation and error handling
- Charity Majors: Production-ready configuration management
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ActiveInferenceConfig:
    """Configuration for PyMDP Active Inference Engine.

    This configuration class encapsulates all parameters needed for active
    inference operation, following clean architecture principles.
    """

    # Core PyMDP parameters
    num_observations: List[int] = field(default_factory=lambda: [2])
    num_states: List[int] = field(default_factory=lambda: [2])
    num_controls: List[int] = field(default_factory=lambda: [2])

    # Planning parameters
    planning_horizon: int = 5  # 3-10 steps as specified in task
    policy_length: int = 1  # Single-step policies for simplicity

    # Inference parameters
    use_utility: bool = True  # Use preference-based planning
    use_states_info_gain: bool = True  # Information-seeking behavior
    use_param_info_gain: bool = False  # Parameter learning (advanced)

    # Numerical stability
    alpha: float = 16.0  # Action precision parameter
    beta: float = 1.0  # Policy precision parameter

    # Performance optimization
    save_belief_hist: bool = False  # Memory optimization
    use_sparse_matrices: bool = False  # For large state spaces

    # Observability settings
    enable_detailed_logging: bool = True
    log_belief_updates: bool = True
    log_policy_selection: bool = True
    track_inference_metrics: bool = True

    # Error handling
    max_inference_iterations: int = 100
    belief_convergence_threshold: float = 1e-6
    enable_fallback_policies: bool = True

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate dimensions
        if not all(n > 0 for n in self.num_observations):
            raise ValueError("All observation dimensions must be positive")
        if not all(n > 0 for n in self.num_states):
            raise ValueError("All state dimensions must be positive")
        if not all(n > 0 for n in self.num_controls):
            raise ValueError("All control dimensions must be positive")

        # Validate planning parameters
        if not (3 <= self.planning_horizon <= 10):
            raise ValueError("Planning horizon must be between 3 and 10 steps")
        if self.policy_length < 1:
            raise ValueError("Policy length must be at least 1")

        # Validate precision parameters
        if self.alpha <= 0:
            raise ValueError("Action precision (alpha) must be positive")
        if self.beta <= 0:
            raise ValueError("Policy precision (beta) must be positive")

        # Validate convergence parameters
        if self.max_inference_iterations <= 0:
            raise ValueError("Max inference iterations must be positive")
        if self.belief_convergence_threshold <= 0:
            raise ValueError("Belief convergence threshold must be positive")

        logger.info("Active inference configuration validated successfully")

    def to_pymdp_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to PyMDP Agent constructor arguments.

        Returns:
            Dictionary of PyMDP Agent constructor arguments
        """
        return {
            "num_controls": self.num_controls,
            "policy_len": self.policy_length,
            "inference_horizon": self.planning_horizon,
            "use_utility": self.use_utility,
            "use_states_info_gain": self.use_states_info_gain,
            "use_param_info_gain": self.use_param_info_gain,
            "alpha": self.alpha,
            "gamma": self.beta,  # PyMDP uses gamma for policy precision
            "save_belief_hist": self.save_belief_hist,
        }


class PyMDPSetup:
    """Utility class for PyMDP environment setup and validation.

    Implements the Nemesis Committee's recommendations for comprehensive
    validation and error handling in PyMDP setup.
    """

    @staticmethod
    def validate_matrices(
        A: List[NDArray[np.floating]],
        B: List[NDArray[np.floating]],
        C: Optional[List[NDArray[np.floating]]] = None,
        D: Optional[List[NDArray[np.floating]]] = None,
    ) -> None:
        """Validate PyMDP matrix structures.

        Args:
            A: Observation model matrices
            B: Transition model matrices
            C: Preference vectors (optional)
            D: Initial belief vectors (optional)

        Raises:
            ValueError: If matrix structures are invalid
        """
        if not A:
            raise ValueError("Observation model (A) cannot be empty")
        if not B:
            raise ValueError("Transition model (B) cannot be empty")

        # Validate A matrices (observation model)
        for i, a_matrix in enumerate(A):
            if not isinstance(a_matrix, np.ndarray):
                raise ValueError(f"A matrix {i} must be numpy array")
            if a_matrix.ndim != 2:
                raise ValueError(f"A matrix {i} must be 2-dimensional")
            if not np.allclose(a_matrix.sum(axis=0), 1.0, rtol=1e-5):
                raise ValueError(f"A matrix {i} columns must sum to 1 (not normalized)")

        # Validate B matrices (transition model)
        for i, b_matrix in enumerate(B):
            if not isinstance(b_matrix, np.ndarray):
                raise ValueError(f"B matrix {i} must be numpy array")
            if b_matrix.ndim != 3:
                raise ValueError(f"B matrix {i} must be 3-dimensional")
            # Check normalization: sum over next states (axis 1) should be 1
            if not np.allclose(b_matrix.sum(axis=1), 1.0, rtol=1e-5):
                raise ValueError(f"B matrix {i} not properly normalized")

        # Validate C vectors (preferences)
        if C is not None:
            for i, c_vector in enumerate(C):
                if not isinstance(c_vector, np.ndarray):
                    raise ValueError(f"C vector {i} must be numpy array")
                if c_vector.ndim != 1:
                    raise ValueError(f"C vector {i} must be 1-dimensional")

        # Validate D vectors (initial beliefs)
        if D is not None:
            for i, d_vector in enumerate(D):
                if not isinstance(d_vector, np.ndarray):
                    raise ValueError(f"D vector {i} must be numpy array")
                if d_vector.ndim != 1:
                    raise ValueError(f"D vector {i} must be 1-dimensional")
                if not np.allclose(d_vector.sum(), 1.0, rtol=1e-5):
                    raise ValueError(f"D vector {i} must sum to 1 (not normalized)")

        logger.info("PyMDP matrix validation completed successfully")

    @staticmethod
    def create_default_matrices(
        config: ActiveInferenceConfig,
    ) -> Tuple[
        List[NDArray[np.floating]],
        List[NDArray[np.floating]],
        List[NDArray[np.floating]],
        List[NDArray[np.floating]],
    ]:
        """Create default matrices for testing and development.

        Args:
            config: Active inference configuration

        Returns:
            Tuple of (A, B, C, D) matrices
        """
        A_matrices = []
        B_matrices = []
        C_vectors = []
        D_vectors = []

        for factor_idx in range(len(config.num_states)):
            num_obs = config.num_observations[factor_idx]
            num_states = config.num_states[factor_idx]
            num_controls = config.num_controls[factor_idx]

            # Create identity-like A matrix (perfect observation)
            A = np.eye(num_obs, num_states) + 0.01 * np.random.rand(num_obs, num_states)
            A = A / A.sum(axis=0)  # Normalize
            A_matrices.append(A)

            # Create controllable B matrix
            B = np.zeros((num_states, num_states, num_controls))
            for action in range(num_controls):
                # Action-dependent transition probabilities
                B[:, :, action] = np.eye(num_states) * 0.8 + 0.2 / num_states
                # Add some randomness
                B[:, :, action] += 0.1 * np.random.rand(num_states, num_states)
                B[:, :, action] = B[:, :, action] / B[:, :, action].sum(axis=0)
            B_matrices.append(B)

            # Create preference vector (slight preference for lower observations)
            C = np.ones(num_obs)
            C[0] = 2.0  # Prefer first observation
            C_vectors.append(C)

            # Create uniform initial beliefs
            D = np.ones(num_states) / num_states
            D_vectors.append(D)

        logger.info(f"Created default matrices for {len(config.num_states)} factors")
        return A_matrices, B_matrices, C_vectors, D_vectors

    @staticmethod
    def normalize_matrices(
        A: List[NDArray[np.floating]], B: List[NDArray[np.floating]]
    ) -> Tuple[List[NDArray[np.floating]], List[NDArray[np.floating]]]:
        """Normalize matrices to ensure proper probability distributions.

        Args:
            A: Observation model matrices to normalize
            B: Transition model matrices to normalize

        Returns:
            Tuple of normalized (A, B) matrices
        """
        A_normalized = []
        B_normalized = []

        # Normalize A matrices (columns should sum to 1)
        for a_matrix in A:
            a_norm = a_matrix.copy()
            col_sums = a_norm.sum(axis=0)
            col_sums[col_sums == 0] = 1.0  # Avoid division by zero
            a_norm = a_norm / col_sums[np.newaxis, :]
            A_normalized.append(a_norm)

        # Normalize B matrices (axis 1 should sum to 1 for each action)
        for b_matrix in B:
            b_norm = b_matrix.copy()
            for action in range(b_norm.shape[2]):
                col_sums = b_norm[:, :, action].sum(axis=0)
                col_sums[col_sums == 0] = 1.0  # Avoid division by zero
                b_norm[:, :, action] = b_norm[:, :, action] / col_sums[np.newaxis, :]
            B_normalized.append(b_norm)

        logger.info("Matrix normalization completed")
        return A_normalized, B_normalized

    @staticmethod
    def check_pymdp_compatibility() -> Dict[str, Any]:
        """Check PyMDP installation and compatibility.

        Returns:
            Dictionary with compatibility information
        """
        try:
            from pymdp.agent import Agent

            # Test basic functionality
            test_A = np.array([[0.9, 0.1], [0.1, 0.9]])
            test_B = np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])

            test_agent = Agent(A=test_A, B=test_B, num_controls=[2])

            # Test inference
            test_agent.infer_states([0])
            test_agent.infer_policies()
            action = test_agent.sample_action()

            compatibility_info = {
                "pymdp_available": True,
                "agent_creation": True,
                "inference_working": True,
                "action_sampling": isinstance(action, np.ndarray),
                "numpy_version": np.__version__,
                "test_action": action.tolist() if hasattr(action, "tolist") else str(action),
            }

            logger.info("PyMDP compatibility check passed")
            return compatibility_info

        except Exception as e:
            logger.error(f"PyMDP compatibility check failed: {e}")
            return {"pymdp_available": False, "error": str(e), "numpy_version": np.__version__}


def create_simple_environment_config() -> ActiveInferenceConfig:
    """Create configuration for simple 2x2 test environment.

    Returns:
        Configuration for simple binary choice environment
    """
    config = ActiveInferenceConfig(
        num_observations=[2],
        num_states=[2],
        num_controls=[2],
        planning_horizon=3,  # Short horizon for testing
        enable_detailed_logging=True,
        log_belief_updates=True,
        log_policy_selection=True,
    )

    config.validate()
    logger.info("Created simple environment configuration")
    return config


def create_complex_environment_config() -> ActiveInferenceConfig:
    """Create configuration for more complex multi-dimensional environment.

    Returns:
        Configuration for complex environment with multiple factors
    """
    config = ActiveInferenceConfig(
        num_observations=[4, 3],  # Two observation modalities
        num_states=[4, 3],  # Two state factors
        num_controls=[3, 2],  # Two control factors
        planning_horizon=5,  # Medium horizon
        policy_length=2,  # Multi-step policies
        use_param_info_gain=True,  # Enable learning
        enable_detailed_logging=True,
        track_inference_metrics=True,
    )

    config.validate()
    logger.info("Created complex environment configuration")
    return config
