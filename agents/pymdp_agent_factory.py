"""PyMDP Agent Factory for creating real PyMDP agents from GMN specifications.

This factory implements the core pipeline: GMN spec -> PyMDP model -> PyMDP Agent
Following TDD principles with minimal implementation and progressive enhancement.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
from numpy.typing import NDArray

# Import real PyMDP - no fallbacks allowed per Nemesis Committee guidance
try:
    from pymdp import utils as pymdp_utils
    from pymdp.agent import Agent as PyMDPAgent
    PYMDP_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        f"PyMDP is required for real agent factory. Install with: pip install inferactively-pymdp==0.0.7.1. "
        f"Original error: {e}"
    )

logger = logging.getLogger(__name__)


class PyMDPAgentCreationError(Exception):
    """Raised when PyMDP agent creation fails."""
    pass


class PyMDPAgentFactory:
    """Factory for creating real PyMDP agents from GMN specifications.
    
    This factory bridges the gap between high-level GMN specifications
    and low-level PyMDP agent instances. It handles:
    
    1. GMN specification validation
    2. Matrix construction and validation  
    3. PyMDP agent creation with proper error handling
    4. Performance optimization through caching
    5. Comprehensive observability and metrics
    """
    
    def __init__(self):
        """Initialize the PyMDP agent factory."""
        self._metrics = {
            "agents_created": 0,
            "creation_failures": 0,
            "avg_creation_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0
        }
        
        # Performance optimization: cache computed matrices
        self._matrix_cache: Dict[str, Any] = {}
        
        logger.info("PyMDP Agent Factory initialized with real PyMDP integration")

    def create_agent(self, gmn_spec: Dict[str, Any]) -> PyMDPAgent:
        """Create a real PyMDP agent from GMN specification.
        
        Args:
            gmn_spec: GMN specification dictionary with PyMDP model parameters
            
        Returns:
            PyMDPAgent: Real PyMDP agent instance
            
        Raises:
            PyMDPAgentCreationError: If agent creation fails
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate GMN specification
            if not self.validate_gmn_spec(gmn_spec):
                self._metrics["validation_failures"] += 1
                raise PyMDPAgentCreationError("GMN specification validation failed")
            
            # Step 2: Check cache for performance optimization
            spec_hash = self._compute_spec_hash(gmn_spec)
            cached_agent = self._get_cached_agent(spec_hash)
            if cached_agent is not None:
                self._metrics["cache_hits"] += 1
                return cached_agent
            
            self._metrics["cache_misses"] += 1
            
            # Step 3: Extract and validate matrices
            A, B, C, D = self._extract_matrices(gmn_spec)
            self._validate_matrices(A, B, C, D)
            
            # Step 4: Create PyMDP agent
            agent = self._create_pymdp_agent(A, B, C, D)
            
            # Step 5: Cache the result
            self._cache_agent(spec_hash, agent)
            
            # Update metrics
            creation_time = (time.time() - start_time) * 1000
            self._update_metrics(creation_time)
            
            logger.info(f"Created PyMDP agent in {creation_time:.2f}ms")
            return agent
            
        except Exception as e:
            self._metrics["creation_failures"] += 1
            creation_time = (time.time() - start_time) * 1000
            logger.error(f"PyMDP agent creation failed after {creation_time:.2f}ms: {e}")
            raise PyMDPAgentCreationError(f"Agent creation failed: {str(e)}") from e

    def validate_gmn_spec(self, gmn_spec: Dict[str, Any]) -> bool:
        """Validate GMN specification structure and content.
        
        Args:
            gmn_spec: GMN specification to validate
            
        Returns:
            bool: True if specification is valid
        """
        try:
            # Check required fields
            required_fields = ["num_states", "num_obs", "num_actions", "A", "B", "C", "D"]
            for field in required_fields:
                if field not in gmn_spec:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate dimensions consistency
            num_states = gmn_spec["num_states"]
            num_obs = gmn_spec["num_obs"] 
            num_actions = gmn_spec["num_actions"]
            
            if not isinstance(num_states, list) or not num_states:
                logger.error("num_states must be a non-empty list")
                return False
                
            if not isinstance(num_obs, list) or not num_obs:
                logger.error("num_obs must be a non-empty list")
                return False
                
            if not isinstance(num_actions, list) or not num_actions:
                logger.error("num_actions must be a non-empty list")
                return False
            
            # Validate matrix dimensions match specifications
            A_matrices = gmn_spec["A"]
            B_matrices = gmn_spec["B"]
            C_vectors = gmn_spec["C"]
            D_vectors = gmn_spec["D"]
            
            if len(A_matrices) != len(num_obs):
                logger.error(f"Number of A matrices ({len(A_matrices)}) must match number of observation modalities ({len(num_obs)})")
                return False
                
            if len(B_matrices) != len(num_states):
                logger.error(f"Number of B matrices ({len(B_matrices)}) must match number of state factors ({len(num_states)})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"GMN specification validation error: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get factory performance metrics.
        
        Returns:
            Dict containing performance and usage metrics
        """
        metrics = self._metrics.copy()
        
        # Add computed metrics
        total_requests = metrics["agents_created"] + metrics["creation_failures"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["agents_created"] / total_requests
            metrics["failure_rate"] = metrics["creation_failures"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            
        total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
        if total_cache_ops > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_ops
        else:
            metrics["cache_hit_rate"] = 0.0
            
        metrics["cache_size"] = len(self._matrix_cache)
        
        return metrics

    def _compute_spec_hash(self, gmn_spec: Dict[str, Any]) -> str:
        """Compute hash of GMN specification for caching."""
        # Create a deterministic string representation
        spec_str = str(sorted(gmn_spec.items()))
        return hashlib.md5(spec_str.encode()).hexdigest()

    def _get_cached_agent(self, spec_hash: str) -> Optional[PyMDPAgent]:
        """Get cached agent if available."""
        # For now, disable caching to ensure fresh agents
        # TODO: Implement proper agent caching with deep copying
        return None

    def _cache_agent(self, spec_hash: str, agent: PyMDPAgent) -> None:
        """Cache agent for future use."""
        # For now, disable caching to avoid memory issues
        # TODO: Implement LRU cache with size limits
        pass

    def _extract_matrices(self, gmn_spec: Dict[str, Any]) -> Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray]]:
        """Extract and normalize matrices from GMN specification."""
        A_matrices = gmn_spec["A"]
        B_matrices = gmn_spec["B"] 
        C_vectors = gmn_spec["C"]
        D_vectors = gmn_spec["D"]
        
        # Convert to numpy arrays and normalize
        A_normalized = []
        for A in A_matrices:
            A_array = np.array(A, dtype=np.float64)
            # Normalize A matrices to be proper probability distributions
            A_norm = pymdp_utils.norm_dist(A_array)
            A_normalized.append(A_norm)
        
        B_normalized = []
        for B in B_matrices:
            B_array = np.array(B, dtype=np.float64)
            # Normalize B matrices along the first axis (next state)
            B_norm = pymdp_utils.norm_dist(B_array)
            B_normalized.append(B_norm)
            
        C_normalized = []
        for C in C_vectors:
            C_array = np.array(C, dtype=np.float64)
            C_normalized.append(C_array)
            
        D_normalized = []
        for D in D_vectors:
            D_array = np.array(D, dtype=np.float64)
            # Normalize D vectors to be proper probability distributions
            D_norm = pymdp_utils.norm_dist(D_array)
            D_normalized.append(D_norm)
            
        return A_normalized, B_normalized, C_normalized, D_normalized

    def _validate_matrices(self, A: List[NDArray], B: List[NDArray], C: List[NDArray], D: List[NDArray]) -> None:
        """Validate matrices for PyMDP compatibility."""
        # Validate A matrices (observation model)
        for i, A_matrix in enumerate(A):
            if A_matrix.ndim < 2:
                raise PyMDPAgentCreationError(f"A matrix {i} must have at least 2 dimensions")
                
            # Check if probabilities sum to 1 along observation dimension
            if not np.allclose(np.sum(A_matrix, axis=0), 1.0, atol=1e-6):
                logger.warning(f"A matrix {i} columns do not sum to 1.0 - this may cause issues")
        
        # Validate B matrices (transition model)  
        for i, B_matrix in enumerate(B):
            if B_matrix.ndim != 3:
                raise PyMDPAgentCreationError(f"B matrix {i} must have exactly 3 dimensions (next_state, state, action)")
                
            # Check if probabilities sum to 1 along next state dimension
            for action in range(B_matrix.shape[2]):
                if not np.allclose(np.sum(B_matrix[:, :, action], axis=0), 1.0, atol=1e-6):
                    logger.warning(f"B matrix {i}, action {action} columns do not sum to 1.0")
        
        # Validate D vectors (prior beliefs)
        for i, D_vector in enumerate(D):
            if D_vector.ndim != 1:
                raise PyMDPAgentCreationError(f"D vector {i} must be 1-dimensional")
                
            if not np.allclose(np.sum(D_vector), 1.0, atol=1e-6):
                logger.warning(f"D vector {i} does not sum to 1.0")

    def _create_pymdp_agent(self, A: List[NDArray], B: List[NDArray], C: List[NDArray], D: List[NDArray]) -> PyMDPAgent:
        """Create PyMDP agent with validated matrices."""
        try:
            # Convert to PyMDP v0.0.7.1 format (single arrays for single-factor models)
            from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
            
            gmn_model = {
                "A": A,
                "B": B,
                "C": C, 
                "D": D
            }
            
            adapted_model = adapt_gmn_to_pymdp(gmn_model)
            
            # Create PyMDP agent with optimized settings
            agent = PyMDPAgent(
                A=adapted_model["A"],
                B=adapted_model["B"], 
                C=adapted_model.get("C"),
                D=adapted_model.get("D"),
                use_utility=True,  # Enable pragmatic value (goal-seeking)
                use_states_info_gain=True,  # Enable epistemic value (curiosity)
                use_param_info_gain=False,  # Disable parameter learning for performance
                policy_len=1,  # Single-step planning for performance
                inference_horizon=1,  # Single-step inference for performance
                inference_algo="VANILLA",  # Standard variational inference
                gamma=16.0,  # Policy precision
                alpha=16.0,  # Action precision
            )
            
            # Ensure F attribute exists for compatibility
            if not hasattr(agent, 'F'):
                agent.F = 0.0
                
            return agent
            
        except Exception as e:
            raise PyMDPAgentCreationError(f"Failed to create PyMDP agent: {str(e)}") from e

    def _update_metrics(self, creation_time_ms: float) -> None:
        """Update factory metrics."""
        self._metrics["agents_created"] += 1
        
        # Update rolling average creation time
        current_avg = self._metrics["avg_creation_time_ms"]
        total_created = self._metrics["agents_created"]
        
        if total_created <= 1:
            self._metrics["avg_creation_time_ms"] = creation_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self._metrics["avg_creation_time_ms"] = (alpha * creation_time_ms) + ((1 - alpha) * current_avg)