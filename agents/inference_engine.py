"""Agent Inference Engine for executing PyMDP decision-making.

This engine orchestrates the full PyMDP inference loop: observation processing,
belief updates, policy inference, and action selection. It provides real
Active Inference capabilities while maintaining clean interfaces and comprehensive
observability.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
from numpy.typing import NDArray

# Import real PyMDP - no fallbacks allowed per Committee guidance
try:
    from pymdp.agent import Agent as PyMDPAgent

    PYMDP_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        f"PyMDP is required for inference engine. Install with: pip install inferactively-pymdp==0.0.7.1. "
        f"Original error: {e}"
    )

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when inference operations fail."""

    pass


class CancellationToken:
    """Token for cancelling long-running inference operations."""

    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        """Cancel the operation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self._cancelled.is_set()


@dataclass
class InferenceResult:
    """Result of inference operation with action, beliefs, and metadata."""

    action: Optional[Union[int, NDArray]]
    beliefs: Dict[str, Any]
    free_energy: float
    confidence: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class InferenceEngine:
    """Engine for executing PyMDP agent inference operations.

    This engine implements the full Active Inference loop:
    1. Process observations into PyMDP format
    2. Update agent beliefs via infer_states()
    3. Infer optimal policies via infer_policies()
    4. Select actions via sample_action()
    5. Return structured results with observability

    Follows established factory patterns with comprehensive error handling,
    metrics collection, and production-ready features.
    """

    def __init__(self, max_workers: int = 4, default_timeout_ms: int = 5000):
        """Initialize the inference engine.

        Args:
            max_workers: Maximum concurrent inference operations
            default_timeout_ms: Default timeout for inference operations
        """
        self._metrics = {
            "inferences_completed": 0,
            "inference_failures": 0,
            "belief_update_failures": 0,
            "action_selection_failures": 0,
            "avg_inference_time_ms": 0.0,
            "timeout_failures": 0,
            "cancellation_requests": 0,
        }

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._default_timeout_ms = default_timeout_ms

        logger.info(
            f"InferenceEngine initialized with {max_workers} workers, {default_timeout_ms}ms default timeout"
        )

    def run_inference(
        self,
        agent: PyMDPAgent,
        observation: List[int],
        planning_horizon: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[InferenceResult]:
        """Execute single-step inference with PyMDP agent.

        Args:
            agent: PyMDP agent instance
            observation: Observation vector (list of observed states)
            planning_horizon: Planning horizon override (if None, uses agent default)
            timeout_ms: Timeout in milliseconds (if None, uses default)
            cancellation_token: Token for cancelling operation

        Returns:
            InferenceResult with action, beliefs, and metadata, or None if cancelled/timeout

        Raises:
            InferenceError: If inference operation fails
        """
        start_time = time.time()
        timeout = timeout_ms or self._default_timeout_ms

        try:
            # Check cancellation before starting
            if cancellation_token and cancellation_token.is_cancelled():
                self._metrics["cancellation_requests"] += 1
                return None

            # Validate observation format
            self._validate_observation(agent, observation)

            # Submit inference task to thread pool with timeout
            future = self._executor.submit(
                self._execute_inference_core,
                agent,
                observation,
                planning_horizon,
                cancellation_token,
            )

            try:
                result = future.result(timeout=timeout / 1000.0)  # Convert to seconds

                if result is None:  # Cancelled during execution
                    return None

                # Update metrics for successful inference
                inference_time = (time.time() - start_time) * 1000
                self._update_metrics(inference_time, success=True)

                # Add timing metadata
                result.metadata["inference_time_ms"] = inference_time
                result.metadata["timeout_ms"] = timeout

                logger.info(f"Inference completed in {inference_time:.2f}ms")
                return result

            except TimeoutError:
                future.cancel()
                self._metrics["timeout_failures"] += 1
                inference_time = (time.time() - start_time) * 1000
                logger.warning(f"Inference timeout after {inference_time:.2f}ms")
                return None

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000
            self._update_metrics(inference_time, success=False)
            logger.error(f"Inference failed after {inference_time:.2f}ms: {e}")
            raise InferenceError(f"Inference operation failed: {str(e)}") from e

    def run_batch_inference(
        self, agent: PyMDPAgent, observations: List[List[int]], timeout_ms: Optional[int] = None
    ) -> List[InferenceResult]:
        """Execute batch inference for multiple observations.

        Args:
            agent: PyMDP agent instance
            observations: List of observation vectors
            timeout_ms: Timeout per observation

        Returns:
            List of InferenceResult objects
        """
        results = []

        for i, observation in enumerate(observations):
            logger.debug(f"Processing batch observation {i+1}/{len(observations)}")

            result = self.run_inference(agent, observation, timeout_ms=timeout_ms)
            if result is not None:
                result.metadata["batch_index"] = i
                results.append(result)
            else:
                # Create empty result for failed/cancelled inferences
                results.append(
                    InferenceResult(
                        action=None,
                        beliefs={},
                        free_energy=float("inf"),
                        confidence=0.0,
                        metadata={"batch_index": i, "failed": True},
                    )
                )

        return results

    def create_cancellation_token(self) -> CancellationToken:
        """Create a cancellation token for inference operations."""
        return CancellationToken()

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics.

        Returns:
            Dictionary with performance and usage metrics
        """
        metrics = self._metrics.copy()

        # Add computed metrics
        total_requests = metrics["inferences_completed"] + metrics["inference_failures"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["inferences_completed"] / total_requests
            metrics["failure_rate"] = metrics["inference_failures"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0

        return metrics

    def _execute_inference_core(
        self,
        agent: PyMDPAgent,
        observation: List[int],
        planning_horizon: Optional[int],
        cancellation_token: Optional[CancellationToken],
    ) -> Optional[InferenceResult]:
        """Core inference execution (runs in thread pool).

        This method implements the PyMDP inference sequence:
        1. infer_states() - belief update
        2. infer_policies() - policy inference
        3. sample_action() - action selection
        """
        try:
            # Check cancellation
            if cancellation_token and cancellation_token.is_cancelled():
                return None

            # Step 1: Update beliefs with observation
            logger.debug(f"Updating beliefs with observation: {observation}")

            # Convert observation to numpy array format expected by PyMDP
            obs_array = np.array(observation, dtype=int)

            # Perform belief update (variational message passing)
            qs_current = agent.infer_states(obs_array)

            if cancellation_token and cancellation_token.is_cancelled():
                return None

            # Step 2: Infer optimal policies
            logger.debug("Inferring optimal policies")

            # Override planning horizon if specified
            if planning_horizon is not None:
                original_horizon = getattr(agent, "policy_len", 1)
                agent.policy_len = planning_horizon

            agent.infer_policies()

            if cancellation_token and cancellation_token.is_cancelled():
                return None

            # Step 3: Select action based on policies
            logger.debug("Selecting action")
            selected_action = agent.sample_action()

            # Step 4: Extract results and metadata
            beliefs = self._extract_beliefs(agent, qs_current)
            free_energy = self._extract_free_energy(agent)
            confidence = self._calculate_confidence(agent)

            metadata = {
                "pymdp_method": "variational_inference",
                "observation": observation,
                "policy_precision": getattr(agent, "gamma", 16.0),
                "action_precision": getattr(agent, "alpha", 16.0),
            }

            # Add planning information if used
            if planning_horizon is not None:
                metadata["planning_horizon"] = planning_horizon
                if hasattr(agent, "policies"):
                    metadata["num_policies"] = (
                        len(agent.policies) if agent.policies is not None else 0
                    )

            # Add policy sequence if available
            if hasattr(agent, "policies") and agent.policies is not None:
                try:
                    # Get the selected policy sequence
                    if hasattr(agent, "action") and agent.action is not None:
                        # Handle numpy array conversion properly
                        action_val = agent.action
                        if isinstance(action_val, np.ndarray):
                            if action_val.size == 1:
                                policy_idx = int(action_val.item())  # Extract scalar safely
                            else:
                                policy_idx = int(action_val[0])  # Use first element
                        else:
                            policy_idx = int(action_val)

                        if policy_idx < len(agent.policies):
                            policy_seq = agent.policies[policy_idx]
                            if hasattr(policy_seq, "tolist"):
                                metadata["policy_sequence"] = policy_seq.tolist()
                            else:
                                metadata["policy_sequence"] = policy_seq
                except (AttributeError, IndexError, TypeError):
                    pass  # Skip policy sequence if not available

            return InferenceResult(
                action=selected_action,
                beliefs=beliefs,
                free_energy=free_energy,
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "invalid observation" in error_msg
                or "out of bounds" in error_msg
                or "index" in error_msg
                and "out of bounds" in error_msg
            ):
                raise InferenceError(f"Invalid observation provided: {observation}")
            else:
                logger.error(f"Core inference failed: {e}")
                raise InferenceError(f"PyMDP inference failed: {str(e)}") from e

    def _validate_observation(self, agent: PyMDPAgent, observation: List[int]) -> None:
        """Validate observation format and values."""
        if not isinstance(observation, list):
            raise InferenceError("Observation must be a list")

        if len(observation) == 0:
            raise InferenceError("Observation cannot be empty")

        # For PyMDP v0.0.7.1, we need to be more flexible with validation
        # as the internal representation might differ from input format
        try:
            # Try to validate using num_obs if available
            if hasattr(agent, "num_obs") and agent.num_obs is not None:
                if isinstance(agent.num_obs, list) and len(agent.num_obs) > 0:
                    max_obs = agent.num_obs[0]  # First modality
                elif isinstance(agent.num_obs, int):
                    max_obs = agent.num_obs
                else:
                    return  # Can't validate

                for i, obs in enumerate(observation):
                    if not isinstance(obs, int) or obs < 0 or obs >= max_obs:
                        raise InferenceError(
                            f"Invalid observation value {obs} at index {i}. "
                            f"Must be integer in range [0, {max_obs-1}]"
                        )
                return

            # Fallback: try to determine from A matrix structure
            if hasattr(agent, "A") and agent.A is not None:
                A = agent.A
                max_obs = None

                if isinstance(A, list) and len(A) > 0:
                    # Multi-modality case - check first modality
                    first_A = A[0]
                    if isinstance(first_A, np.ndarray) and first_A.ndim >= 2:
                        max_obs = first_A.shape[0]  # Number of possible observations
                elif isinstance(A, np.ndarray):
                    if A.ndim >= 2:
                        max_obs = A.shape[0]
                    elif A.ndim == 1 and len(A) > 0:
                        # PyMDP might have converted to 1D - try to infer from length
                        # This is a heuristic - skip validation in this case
                        return

                # Only validate if we could determine observation space size
                if max_obs is not None and max_obs > 0:
                    for i, obs in enumerate(observation):
                        if not isinstance(obs, int) or obs < 0 or obs >= max_obs:
                            raise InferenceError(
                                f"Invalid observation value {obs} at index {i}. "
                                f"Must be integer in range [0, {max_obs-1}]"
                            )

        except Exception as e:
            # If validation fails due to PyMDP internal structure, just log and skip
            logger.debug(f"Could not validate observation bounds: {e}")
            pass

    def _extract_beliefs(self, agent: PyMDPAgent, qs_current: Any) -> Dict[str, Any]:
        """Extract belief state from PyMDP agent."""
        beliefs = {}

        try:
            # Extract current beliefs over states
            if qs_current is not None:
                if isinstance(qs_current, list):
                    # Multiple state factors - flatten for test compatibility
                    if len(qs_current) == 1 and isinstance(qs_current[0], np.ndarray):
                        # Single factor in list format
                        beliefs["states"] = qs_current[0].tolist()
                    else:
                        # Multiple factors
                        beliefs["states"] = [
                            q.tolist() if hasattr(q, "tolist") else q for q in qs_current
                        ]
                elif isinstance(qs_current, np.ndarray):
                    # Single state factor
                    beliefs["states"] = qs_current.tolist()
                else:
                    beliefs["states"] = qs_current

            # Add additional belief information if available
            if hasattr(agent, "qs_hist") and agent.qs_hist is not None:
                beliefs["history_length"] = len(agent.qs_hist)

        except Exception as e:
            logger.warning(f"Could not extract beliefs: {e}")
            beliefs = {"states": [], "error": str(e)}

        return beliefs

    def _extract_free_energy(self, agent: PyMDPAgent) -> float:
        """Extract free energy from PyMDP agent."""
        try:
            # Try to get free energy from agent
            if hasattr(agent, "F") and agent.F is not None:
                return float(agent.F)
            elif hasattr(agent, "free_energy") and agent.free_energy is not None:
                return float(agent.free_energy)
            else:
                return 0.0  # Default if not available

        except Exception as e:
            logger.warning(f"Could not extract free energy: {e}")
            return 0.0

    def _calculate_confidence(self, agent: PyMDPAgent) -> float:
        """Calculate confidence score based on belief entropy."""
        try:
            # Get current beliefs
            if hasattr(agent, "qs") and agent.qs is not None:
                qs = agent.qs
                if isinstance(qs, list) and len(qs) > 0:
                    qs = qs[0]  # Use first state factor

                if isinstance(qs, np.ndarray):
                    # Flatten if multi-dimensional and ensure 1D
                    qs_flat = qs.flatten()

                    # Ensure we're working with float64 to avoid numpy warnings
                    qs_safe = np.array(qs_flat, dtype=np.float64)

                    # Normalize to ensure it's a proper probability distribution
                    qs_sum = np.sum(qs_safe)
                    if qs_sum > 0:
                        qs_safe = qs_safe / qs_sum

                    # Calculate entropy and convert to confidence
                    entropy = -np.sum(qs_safe * np.log(qs_safe + 1e-10))  # Add small epsilon
                    max_entropy = np.log(len(qs_safe))

                    if max_entropy > 0:
                        # Convert entropy to confidence (0 = max entropy, 1 = min entropy)
                        confidence = 1.0 - (entropy / max_entropy)
                        return max(0.0, min(1.0, confidence))

            return 0.5  # Default moderate confidence

        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0.5

    def _update_metrics(self, inference_time_ms: float, success: bool) -> None:
        """Update engine metrics."""
        if success:
            self._metrics["inferences_completed"] += 1
        else:
            self._metrics["inference_failures"] += 1

        # Update rolling average inference time
        current_avg = self._metrics["avg_inference_time_ms"]
        total_completed = self._metrics["inferences_completed"]

        if success and total_completed <= 1:
            self._metrics["avg_inference_time_ms"] = inference_time_ms
        elif success:
            # Exponential moving average
            alpha = 0.1
            self._metrics["avg_inference_time_ms"] = (alpha * inference_time_ms) + (
                (1 - alpha) * current_avg
            )

    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors
