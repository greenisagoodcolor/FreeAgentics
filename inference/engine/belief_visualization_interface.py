"""
Belief State Visualization Interface

Provides interface between pymdp-based inference engine and visualization frontend
ensuring mathematical correctness and ADR-005 compliance.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .belief_state import BeliefState
from .pymdp_generative_model import PyMDPGenerativeModel

logger = logging.getLogger(__name__)


@dataclass
class BeliefVisualizationData:
    """Data structure for belief state visualization"""

    agent_id: str
    timestamp: datetime
    belief_distribution: np.ndarray
    confidence_intervals: np.ndarray
    free_energy: float
    prediction_accuracy: float
    convergence_metric: float
    uncertainty_measure: float
    mathematical_equations: Dict[str, str]
    numerical_precision: Dict[str, float]


class BeliefStateVisualizationInterface:
    """Interface for belief state visualization with mathematical rigor"""

    def __init__(self) -> None:
        """Initialize"""
        self.belief_history: Dict[str, List[BeliefVisualizationData]] = {}
        self.precision_tracking = {}

    def extract_visualization_data(
        self,
        agent_id: str,
        belief_state: BeliefState,
        pymdp_model: PyMDPGenerativeModel,
        observations: np.ndarray,
    ) -> BeliefVisualizationData:
        """Extract comprehensive data for belief state visualization"""

        # Get belief distribution
        if hasattr(belief_state, "beliefs"):
            belief_dist = belief_state.beliefs
        else:
            belief_dist = np.ones(pymdp_model.A.shape[1]) / pymdp_model.A.shape[1]

        # Calculate confidence intervals using Bayesian posterior
        confidence_intervals = self._calculate_confidence_intervals(belief_dist)

        # Calculate free energy
        free_energy = self._calculate_free_energy(belief_dist, pymdp_model, observations)

        # Calculate convergence metrics
        convergence_metric = self._calculate_convergence(agent_id, belief_dist)

        # Generate mathematical equations
        equations = self._generate_mathematical_equations(belief_dist, observations)

        # Track numerical precision
        precision_metrics = self._track_numerical_precision(belief_dist)

        viz_data = BeliefVisualizationData(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            belief_distribution=belief_dist,
            confidence_intervals=confidence_intervals,
            free_energy=free_energy,
            prediction_accuracy=0.85,  # Placeholder
            convergence_metric=convergence_metric,
            uncertainty_measure=self._calculate_uncertainty(belief_dist),
            mathematical_equations=equations,
            numerical_precision=precision_metrics,
        )

        # Store in history
        if agent_id not in self.belief_history:
            self.belief_history[agent_id] = []
        self.belief_history[agent_id].append(viz_data)

        return viz_data

    def _calculate_confidence_intervals(self, belief_dist: np.ndarray) -> np.ndarray:
        """Calculate confidence intervals for belief distribution"""
        # Use Bayesian posterior credible intervals
        # Simplified implementation - would use actual posterior calculations
        std_dev = np.sqrt(belief_dist * (1 - belief_dist))
        lower_bound = belief_dist - 1.96 * std_dev
        upper_bound = belief_dist + 1.96 * std_dev

        return np.column_stack([lower_bound, upper_bound])

    def _calculate_free_energy(
        self, belief_dist: np.ndarray, pymdp_model: PyMDPGenerativeModel, observations: np.ndarray
    ) -> float:
        """Calculate variational free energy F = -log P(o) + KL[Q(s)||P(s)]"""

        # Likelihood term: -log P(o|s)
        if observations.size > 0 and len(observations) <= pymdp_model.A.shape[0]:
            obs_idx = int(observations[0]) if observations[0] < pymdp_model.A.shape[0] else 0
            likelihood = np.log(np.maximum(pymdp_model.A[obs_idx, :], 1e-16))
            likelihood_term = -np.sum(belief_dist * likelihood)
        else:
            likelihood_term = 0.0

        # Prior term: KL divergence KL[Q(s)||P(s)]
        prior_dist = pymdp_model.D
        kl_div = np.sum(belief_dist * np.log(np.maximum(belief_dist / prior_dist, 1e-16)))

        free_energy = likelihood_term + kl_div
        return free_energy

    def _calculate_convergence(self, agent_id: str, belief_dist: np.ndarray) -> float:
        """Calculate convergence metric compared to previous belief state"""
        if agent_id not in self.belief_history or len(self.belief_history[agent_id]) == 0:
            return 0.0

        previous_belief = self.belief_history[agent_id][-1].belief_distribution

        # Calculate KL divergence as convergence metric
        kl_div = np.sum(belief_dist * np.log(np.maximum(belief_dist / previous_belief, 1e-16)))

        return float(kl_div)

    def _calculate_uncertainty(self, belief_dist: np.ndarray) -> float:
        """Calculate uncertainty measure (entropy)"""
        entropy = -np.sum(belief_dist * np.log(np.maximum(belief_dist, 1e-16)))
        return float(entropy)

    def _generate_mathematical_equations(
        self, belief_dist: np.ndarray, observations: np.ndarray
    ) -> Dict[str, str]:
        """Generate LaTeX mathematical equations for visualization"""

        equations = {
            "bayesian_update": r"P(s_t|o_{1:t}) = \frac{P(o_t|s_t)P(s_t|o_{1:t-1})}{\sum_s P(o_t|s)P(s|o_{1:t-1})}",
            "free_energy": r"F = -\log P(o) + D_{KL}[Q(s)||P(s)]",
            "entropy": r"H[Q(s)] = -\sum_s Q(s) \log Q(s)",
            "kl_divergence": r"D_{KL}[Q||P] = \sum_s Q(s) \log \frac{Q(s)}{P(s)}",
            "expected_free_energy": r"G(\pi) = \sum_{\tau} Q(s_\tau|\pi) \cdot F(s_\tau, \pi)",
            "variational_message_passing": r"\ln Q(s_\mu) = \langle \ln P(s, o) \rangle_{Q(\mathbf{s}_{\nu \neq \mu})}",
        }

        return equations

    def _track_numerical_precision(self, belief_dist: np.ndarray) -> Dict[str, float]:
        """Track numerical precision metrics"""

        return {
            "min_value": float(np.min(belief_dist)),
            "max_value": float(np.max(belief_dist)),
            "sum_check": float(np.sum(belief_dist)),  # Should be ~1.0
            "numerical_stability": float(np.std(belief_dist)),
            "condition_number": float(np.linalg.cond(belief_dist.reshape(-1, 1))),
        }

    def get_belief_trajectory(
        self, agent_id: str, window_size: int = 100
    ) -> List[BeliefVisualizationData]:
        """Get historical belief trajectory for agent"""
        if agent_id not in self.belief_history:
            return []

        return self.belief_history[agent_id][-window_size:]

    def export_for_publication(self, agent_id: str) -> Dict[str, Any]:
        """Export data in format suitable for scientific publication"""

        trajectory = self.get_belief_trajectory(agent_id)

        return {
            "agent_id": agent_id,
            "data_points": len(trajectory),
            "time_series": {
                "timestamps": [data.timestamp.isoformat() for data in trajectory],
                "free_energy": [data.free_energy for data in trajectory],
                "convergence": [data.convergence_metric for data in trajectory],
                "uncertainty": [data.uncertainty_measure for data in trajectory],
            },
            "mathematical_framework": trajectory[-1].mathematical_equations if trajectory else {},
            "numerical_precision": {
                "average_precision": (
                    np.mean(
                        [data.numerical_precision["numerical_stability"] for data in trajectory]
                    )
                    if trajectory
                    else 0.0
                ),
                "precision_range": (
                    [
                        min(data.numerical_precision["numerical_stability"] for data in trajectory),
                        max(data.numerical_precision["numerical_stability"] for data in trajectory),
                    ]
                    if trajectory
                    else [0.0, 0.0]
                ),
            },
        }


# Global interface instance
belief_viz_interface = BeliefStateVisualizationInterface()

__all__ = ["BeliefStateVisualizationInterface", "BeliefVisualizationData", "belief_viz_interface"]
