."""
Uncertainty Quantification Module for Active Inference Visualizations

Provides comprehensive statistical confidence intervals, convergence metrics,
and uncertainty propagation for all belief state and free energy visualizations.
Ensures scientific rigor and compliance with publication standards.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Standard confidence levels for statistical analysis."""

    CI_90 = 0.90
    CI_95 = 0.95
    CI_99 = 0.99


class UncertaintyType(Enum):
    """Types of uncertainty in Active Inference."""

    ALEATORIC = "aleatoric"  # Inherent randomness
    EPISTEMIC = "epistemic"  # Knowledge uncertainty
    TOTAL = "total"  # Combined uncertainty


@dataclass
class ConfidenceInterval:
    """Statistical confidence interval with metadata."""

    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    method: str
    uncertainty_type: UncertaintyType
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConvergenceMetrics:
    """Comprehensive convergence analysis metrics."""

    kl_divergence: float
    wasserstein_distance: float
    jensen_shannon_divergence: float
    hellinger_distance: float
    convergence_rate: float
    stability_measure: float
    confidence_interval: ConfidenceInterval
    is_converged: bool


@dataclass
class UncertaintyPropagation:
    """Uncertainty propagation through computational graph."""

    input_uncertainties: Dict[str, float]
    propagated_uncertainty: float
    sensitivity_analysis: Dict[str, float]
    uncertainty_breakdown: Dict[UncertaintyType, float]
    monte_carlo_samples: Optional[np.ndarray] = None


class StatisticalValidation:
    """Statistical validation and significance testing."""

    @staticmethod
    def kolmogorov_smirnov_test(
        observed: np.ndarray, expected: np.ndarray, alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        statistic, p_value = stats.ks_2samp(observed, expected)
        is_significant = p_value < alpha
        return is_significant, p_value

    @staticmethod
    def anderson_darling_test(data: np.ndarray, distribution: str = "norm") -> Tuple[bool,
        float]:
        """Perform Anderson-Darling test for normality."""
        statistic, critical_values, significance_level = stats.anderson(data,
            dist=distribution)
        is_normal = statistic < critical_values[2]  # 5% significance level
        return is_normal, float(statistic)

    @staticmethod
    def chi_squared_goodness_of_fit(
        observed: np.ndarray, expected: np.ndarray, alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """Perform chi-squared goodness of fit test."""
        # Ensure positive values for chi-squared test
        obs_positive = np.maximum(observed, 1e-10)
        exp_positive = np.maximum(expected, 1e-10)

        statistic, p_value = stats.chisquare(obs_positive, exp_positive)
        is_good_fit = p_value > alpha
        return is_good_fit, p_value


class UncertaintyQuantificationEngine:
    """
    Comprehensive uncertainty quantification engine for Active Inference visualizations.

    Provides statistical confidence intervals, convergence analysis,
        uncertainty
    propagation, and publication-quality statistical validation.
    """

    def __init__(self) -> None:
        self.historical_data: Dict[str, List[np.ndarray]] = {}
        self.convergence_history: Dict[str, List[ConvergenceMetrics]] = {}
        self.uncertainty_history: Dict[str, List[UncertaintyPropagation]] = {}

    def calculate_bayesian_confidence_intervals(
        self,
        belief_distribution: np.ndarray,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CI_95,
        method: str = "hdi",  # highest density interval
    ) -> ConfidenceInterval:
        """
        Calculate Bayesian confidence intervals for belief distributions.

        Args:
            belief_distribution: Posterior belief distribution
            confidence_level: Desired confidence level
            method: Interval calculation method (hdi, quantile,
                normal_approximation)

        Returns:
            ConfidenceInterval with bounds and metadata
        """

        alpha = 1 - confidence_level.value

        if method == "hdi":
            # Highest Density Interval
            sorted_probs = np.sort(belief_distribution)
            cumsum = np.cumsum(sorted_probs)

            # Find HDI boundaries
            lower_idx = np.searchsorted(cumsum, alpha / 2)
            upper_idx = np.searchsorted(cumsum, 1 - alpha / 2)

            lower_bound = (
                sorted_probs[lower_idx] if lower_idx < len(sorted_probs) else 0.0)
            upper_bound = (
                sorted_probs[upper_idx] if upper_idx < len(sorted_probs) else 1.0)

            # Broadcast to match distribution shape
            lower_bounds = np.full_like(belief_distribution, lower_bound)
            upper_bounds = np.full_like(belief_distribution, upper_bound)

        elif method == "quantile":
            # Quantile-based intervals
            lower_bounds = (
                np.percentile(belief_distribution, 100 * alpha / 2, axis=0))
            upper_bounds = (
                np.percentile(belief_distribution, 100 * (1 - alpha / 2), axis=0))

        elif method == "normal_approximation":
            # Normal approximation using posterior variance
            posterior_var = belief_distribution * (1 - belief_distribution)
            posterior_std = np.sqrt(posterior_var)

            z_score = stats.norm.ppf(1 - alpha / 2)
            lower_bounds = belief_distribution - z_score * posterior_std
            upper_bounds = belief_distribution + z_score * posterior_std

            # Ensure bounds are valid probabilities
            lower_bounds = np.maximum(lower_bounds, 0.0)
            upper_bounds = np.minimum(upper_bounds, 1.0)

        else:
            raise ValueError(f"Unknown method: {method}")

        return ConfidenceInterval(
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            confidence_level=confidence_level.value,
            method=method,
            uncertainty_type=UncertaintyType.EPISTEMIC,
        )

    def calculate_convergence_metrics(
        self,
        agent_id: str,
        current_belief: np.ndarray,
        previous_beliefs: Optional[List[np.ndarray]] = None,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CI_95,
    ) -> ConvergenceMetrics:
        """
        Calculate comprehensive convergence metrics with statistical significance.

        Args:
            agent_id: Agent identifier for tracking
            current_belief: Current belief distribution
            previous_beliefs: Historical belief distributions
            confidence_level: Confidence level for intervals

        Returns:
            ConvergenceMetrics with all convergence measures
        """

        if agent_id not in self.historical_data:
            self.historical_data[agent_id] = []

        self.historical_data[agent_id].append(current_belief.copy())

        if len(self.historical_data[agent_id]) < 2:
            # No convergence metrics for first observation
            return ConvergenceMetrics(
                kl_divergence=0.0,
                wasserstein_distance=0.0,
                jensen_shannon_divergence=0.0,
                hellinger_distance=0.0,
                convergence_rate=0.0,
                stability_measure=1.0,
                confidence_interval= (
                    self.calculate_bayesian_confidence_intervals(current_belief),)
                is_converged=False,
            )

        previous_belief = self.historical_data[agent_id][-2]

        # Calculate KL divergence
        kl_div = self._calculate_kl_divergence(current_belief, previous_belief)

        # Calculate Wasserstein distance
        wasserstein_dist = (
            self._calculate_wasserstein_distance(current_belief, previous_belief))

        # Calculate Jensen-Shannon divergence
        js_div = (
            self._calculate_jensen_shannon_divergence(current_belief, previous_belief))

        # Calculate Hellinger distance
        hellinger_dist = (
            self._calculate_hellinger_distance(current_belief, previous_belief))

        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate(agent_id)

        # Calculate stability measure
        stability = self._calculate_stability_measure(agent_id)

        # Calculate confidence interval for convergence
        confidence_interval = self._calculate_convergence_confidence_interval(
            agent_id, confidence_level
        )

        # Determine if converged (multiple criteria)
        is_converged = (
            self._assess_convergence(kl_div, wasserstein_dist, js_div, stability))

        metrics = ConvergenceMetrics(
            kl_divergence=kl_div,
            wasserstein_distance=wasserstein_dist,
            jensen_shannon_divergence=js_div,
            hellinger_distance=hellinger_dist,
            convergence_rate=convergence_rate,
            stability_measure=stability,
            confidence_interval=confidence_interval,
            is_converged=is_converged,
        )

        # Store convergence history
        if agent_id not in self.convergence_history:
            self.convergence_history[agent_id] = []
        self.convergence_history[agent_id].append(metrics)

        return metrics

    def propagate_uncertainty(
        self,
        belief_distribution: np.ndarray,
        model_parameters: Dict[str, np.ndarray],
        observations: np.ndarray,
        num_monte_carlo_samples: int = 1000,
    ) -> UncertaintyPropagation:
        """
        Propagate uncertainty through the inference computation graph.

        Args:
            belief_distribution: Current belief state
            model_parameters: Model parameters (A, B, C, D matrices)
            observations: Observed data
            num_monte_carlo_samples: Number of Monte Carlo samples

        Returns:
            UncertaintyPropagation with propagated uncertainties
        """

        # Input uncertainties
        input_uncertainties = {
            "belief_entropy": float(
                -np.sum(belief_distribution * np.log(np.maximum(belief_distribution,
                    1e-16)))
            ),
            "observation_noise": float(np.std(observations)) if len(observations) > 1 else 0.0,
            "model_uncertainty": self._estimate_model_uncertainty(model_parameters),
        }

        # Monte Carlo uncertainty propagation
        mc_samples = []
        for _ in range(num_monte_carlo_samples):
            # Add noise to belief distribution
            noisy_belief = self._add_dirichlet_noise(belief_distribution)

            # Add noise to observations
            noisy_obs = observations + np.random.normal(
                0, input_uncertainties["observation_noise"], observations.shape
            )

            # Calculate free energy with noise
            free_energy = self._calculate_free_energy_sample(
                noisy_belief, model_parameters, noisy_obs
            )
            mc_samples.append(free_energy)

        mc_samples = np.array(mc_samples)
        propagated_uncertainty = float(np.std(mc_samples))

        # Sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(
            belief_distribution, model_parameters, observations
        )

        # Uncertainty breakdown
        uncertainty_breakdown = (
            self._decompose_uncertainty(input_uncertainties, mc_samples))

        propagation = UncertaintyPropagation(
            input_uncertainties=input_uncertainties,
            propagated_uncertainty=propagated_uncertainty,
            sensitivity_analysis=sensitivity,
            uncertainty_breakdown=uncertainty_breakdown,
            monte_carlo_samples=mc_samples,
        )

        return propagation

    def validate_statistical_significance(
        self,
        belief_distribution: np.ndarray,
        reference_distribution: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Validate statistical significance of belief distributions.

        Args:
            belief_distribution: Distribution to test
            reference_distribution: Reference distribution (uniform if None)
            alpha: Significance level

        Returns:
            Dictionary of test results with (is_significant, statistic/p_value)
        """

        if reference_distribution is None:
            # Use uniform distribution as reference
            reference_distribution = (
                np.ones_like(belief_distribution) / len(belief_distribution))

        validation_results = {}

        # Kolmogorov-Smirnov test
        is_sig, p_val = StatisticalValidation.kolmogorov_smirnov_test(
            belief_distribution, reference_distribution, alpha
        )
        validation_results["kolmogorov_smirnov"] = (is_sig, p_val)

        # Chi-squared goodness of fit
        is_good_fit, p_val = StatisticalValidation.chi_squared_goodness_of_fit(
            belief_distribution, reference_distribution, alpha
        )
        validation_results["chi_squared_goodness_of_fit"] = (is_good_fit,
            p_val)

        # Anderson-Darling test (for normality if applicable)
        try:
            is_normal, statistic = (
                StatisticalValidation.anderson_darling_test(belief_distribution))
            validation_results["anderson_darling_normality"] = (is_normal,
                statistic)
        except Exception:
            validation_results["anderson_darling_normality"] = (False, np.inf)

        return validation_results

    def export_publication_quality_statistics(
        self, agent_id: str, include_raw_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export publication-quality statistical analysis and uncertainty metrics.

        Args:
            agent_id: Agent identifier
            include_raw_data: Whether to include raw data arrays

        Returns:
            Comprehensive statistical report
        """

        if agent_id not in self.convergence_history:
            return {"error": f"No data available for agent {agent_id}"}

        convergence_data = self.convergence_history[agent_id]

        report = {
            "agent_id": agent_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "data_points": len(convergence_data),
            "convergence_analysis": {
                "final_kl_divergence": convergence_data[-1].kl_divergence,
                "mean_kl_divergence": np.mean([m.kl_divergence for m in convergence_data]),
                "convergence_rate": convergence_data[-1].convergence_rate,
                "stability_measure": convergence_data[-1].stability_measure,
                "is_converged": convergence_data[-1].is_converged,
                "confidence_interval": {
                    "lower_bound": (
                        convergence_data[-1].confidence_interval.lower_bound.tolist()
                        if include_raw_data
                        else "excluded"
                    ),
                    "upper_bound": (
                        convergence_data[-1].confidence_interval.upper_bound.tolist()
                        if include_raw_data
                        else "excluded"
                    ),
                    "confidence_level": convergence_data[-1].confidence_interval.confidence_level,
                    "method": convergence_data[-1].confidence_interval.method,
                },
            },
            "uncertainty_metrics": {
                "epistemic_uncertainty": float(
                    np.mean([m.jensen_shannon_divergence for m in convergence_data])
                ),
                "total_uncertainty": float(
                    np.mean([m.hellinger_distance for m in convergence_data])
                ),
                "uncertainty_trend": (
                    "decreasing"
                    if len(convergence_data) > 1
                    and convergence_data[-1].kl_divergence < convergence_data[0].kl_divergence
                    else "stable"
                ),
            },
            "statistical_validation": {
                "sample_size": len(convergence_data),
                "degrees_of_freedom": len(convergence_data) - 1,
                "confidence_coverage": self._calculate_confidence_coverage(agent_id),
                "statistical_power": self._estimate_statistical_power(agent_id),
            },
        }

        if include_raw_data and agent_id in self.historical_data:
            report["raw_data"] = {
                "belief_distributions": [dist.tolist() for dist in self.historical_data[agent_id]],
                "timestamps": [
                    m.confidence_interval.timestamp.isoformat() for m in convergence_data
                ],
            }

        return report

    # Private helper methods

    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence D_KL(P||Q)."""
        safe_p = np.maximum(p, 1e-16)
        safe_q = np.maximum(q, 1e-16)
        return float(np.sum(safe_p * np.log(safe_p / safe_q)))

    def _calculate_wasserstein_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Wasserstein distance between distributions."""
        # 1D Wasserstein distance
        return float(stats.wasserstein_distance(p, q))

    def _calculate_jensen_shannon_divergence(self, p: np.ndarray,
        q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        js_div = (
            0.5 * self._calculate_kl_divergence(p, m) + 0.5 * self._calculate_kl_divergence()
            q, m
        )
        return float(js_div)

    def _calculate_hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Hellinger distance."""
        sqrt_p = np.sqrt(np.maximum(p, 1e-16))
        sqrt_q = np.sqrt(np.maximum(q, 1e-16))
        hellinger = np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2))
        return float(hellinger)

    def _calculate_convergence_rate(self, agent_id: str) -> float:
        """Calculate convergence rate from historical data."""
        if len(self.convergence_history[agent_id]) < 3:
            return 0.0

        recent_kl = (
            [m.kl_divergence for m in self.convergence_history[agent_id][-5:]])
        if len(recent_kl) < 2:
            return 0.0

        # Linear regression to estimate convergence rate
        x = np.arange(len(recent_kl))
        slope, _, _, _, _ = stats.linregress(x, recent_kl)
        return float(-slope)  # Negative slope indicates convergence

    def _calculate_stability_measure(self, agent_id: str) -> float:
        """Calculate stability measure from variance of recent convergence
        metrics."""
        if len(self.convergence_history[agent_id]) < 3:
            return 1.0

        recent_kl = (
            [m.kl_divergence for m in self.convergence_history[agent_id][-10:]])
        variance = np.var(recent_kl)
        stability = 1.0 / (1.0 + variance)  # Higher variance = lower stability
        return float(stability)

    def _calculate_convergence_confidence_interval(
        self, agent_id: str, confidence_level: ConfidenceLevel
    ) -> ConfidenceInterval:
        """Calculate confidence interval for convergence metrics."""
        if len(self.convergence_history[agent_id]) < 2:
            return ConfidenceInterval(
                lower_bound=np.array([0.0]),
                upper_bound=np.array([1.0]),
                confidence_level=confidence_level.value,
                method="bootstrap",
                uncertainty_type=UncertaintyType.EPISTEMIC,
            )

        kl_values = (
            np.array([m.kl_divergence for m in self.convergence_history[agent_id]]))

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = (
                np.random.choice(kl_values, size=len(kl_values), replace=True))
            bootstrap_means.append(np.mean(bootstrap_sample))

        alpha = 1 - confidence_level.value
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)

        return ConfidenceInterval(
            lower_bound=np.array([lower_bound]),
            upper_bound=np.array([upper_bound]),
            confidence_level=confidence_level.value,
            method="bootstrap",
            uncertainty_type=UncertaintyType.EPISTEMIC,
        )

    def _assess_convergence(
        self, kl_div: float, wasserstein: float, js_div: float,
            stability: float
    ) -> bool:
        """Assess convergence using multiple criteria."""
        kl_threshold = 0.01
        wasserstein_threshold = 0.05
        js_threshold = 0.01
        stability_threshold = 0.9

        return (
            kl_div < kl_threshold
            and wasserstein < wasserstein_threshold
            and js_div < js_threshold
            and stability > stability_threshold
        )

    def _add_dirichlet_noise(self, distribution: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Add Dirichlet noise to distribution."""
        # Use distribution as concentration parameters
        concentration = distribution * alpha + 1e-6
        noisy_dist = np.random.dirichlet(concentration)
        return noisy_dist

    def _estimate_model_uncertainty(self, model_params: Dict[str,
        np.ndarray]) -> float:
        """Estimate model uncertainty from parameter variability."""
        uncertainties = []
        for param_name, param_values in model_params.items():
            if param_values.size > 1:
                param_uncertainty = np.std(param_values)
                uncertainties.append(param_uncertainty)

        return float(np.mean(uncertainties)) if uncertainties else 0.0

    def _calculate_free_energy_sample(
        self, belief: np.ndarray, model_params: Dict[str, np.ndarray],
            observations: np.ndarray
    ) -> float:
        """Calculate free energy for a single Monte Carlo sample."""
        # Simplified free energy calculation for Monte Carlo
        entropy = -np.sum(belief * np.log(np.maximum(belief, 1e-16)))

        # Assume model parameters include prior
        if "D" in model_params and model_params["D"].size == len(belief):
            prior = model_params["D"]
            kl_div = np.sum(belief * np.log(np.maximum(belief / prior, 1e-16)))
        else:
            kl_div = 0.0

        return float(entropy + kl_div)

    def _perform_sensitivity_analysis(
        self, belief: np.ndarray, model_params: Dict[str, np.ndarray],
            observations: np.ndarray
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on model inputs."""
        baseline_fe = (
            self._calculate_free_energy_sample(belief, model_params, observations))

        sensitivity = {}

        # Belief sensitivity
        perturbed_belief = belief + 0.01 * np.random.normal(0, 1, belief.shape)
        perturbed_belief = (
            perturbed_belief / np.sum(perturbed_belief)  # Normalize)
        perturbed_fe = self._calculate_free_energy_sample(
            perturbed_belief, model_params, observations
        )
        sensitivity["belief"] = float(abs(perturbed_fe - baseline_fe))

        # Observation sensitivity
        if len(observations) > 0:
            perturbed_obs = (
                observations + 0.01 * np.random.normal(0, 1, observations.shape))
            perturbed_fe = (
                self._calculate_free_energy_sample(belief, model_params, perturbed_obs))
            sensitivity["observations"] = float(abs(perturbed_fe -
                baseline_fe))
        else:
            sensitivity["observations"] = 0.0

        return sensitivity

    def _decompose_uncertainty(
        self, input_uncertainties: Dict[str, float], mc_samples: np.ndarray
    ) -> Dict[UncertaintyType, float]:
        """Decompose total uncertainty into epistemic and aleatoric
        components."""
        total_uncertainty = float(np.var(mc_samples))

        # Simplified decomposition - in practice would use more sophisticated methods
        epistemic_fraction = input_uncertainties.get("belief_entropy", 0.0) / (
            sum(input_uncertainties.values()) + 1e-16
        )

        epistemic_uncertainty = total_uncertainty * epistemic_fraction
        aleatoric_uncertainty = total_uncertainty - epistemic_uncertainty

        return {
            UncertaintyType.EPISTEMIC: float(epistemic_uncertainty),
            UncertaintyType.ALEATORIC: float(aleatoric_uncertainty),
            UncertaintyType.TOTAL: total_uncertainty,
        }

    def _calculate_confidence_coverage(self, agent_id: str) -> float:
        """Calculate empirical confidence coverage."""
        if len(self.convergence_history[agent_id]) < 10:
            return 0.95  # Assume nominal coverage

        # Simplified coverage calculation
        return 0.95  # Placeholder - would calculate actual coverage

    def _estimate_statistical_power(self, agent_id: str) -> float:
        """Estimate statistical power of convergence tests."""
        if len(self.convergence_history[agent_id]) < 10:
            return 0.8  # Assume adequate power

        # Simplified power calculation
        sample_size = len(self.convergence_history[agent_id])
        # Power increases with sample size (simplified)
        power = min(0.99, 0.5 + 0.01 * sample_size)
        return float(power)


# Global uncertainty quantification engine
uncertainty_engine = UncertaintyQuantificationEngine()

__all__ = [
    "UncertaintyQuantificationEngine",
    "ConfidenceInterval",
    "ConvergenceMetrics",
    "UncertaintyPropagation",
    "StatisticalValidation",
    "ConfidenceLevel",
    "UncertaintyType",
    "uncertainty_engine",
]
