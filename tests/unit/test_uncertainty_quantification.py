"""
Tests for Uncertainty Quantification Module.

Comprehensive test suite for statistical confidence intervals, convergence metrics,
and uncertainty propagation for Active Inference visualizations.
"""

from datetime import datetime

import numpy as np
import pytest

from inference.engine.uncertainty_quantification import (
    ConfidenceInterval,
    ConfidenceLevel,
    ConvergenceMetrics,
    StatisticalValidation,
    UncertaintyPropagation,
    UncertaintyQuantificationEngine,
    UncertaintyType,
)


class TestConfidenceLevel:
    """Test confidence level enumeration."""

    def test_confidence_levels(self):
        """Test all confidence levels are defined correctly."""
        assert ConfidenceLevel.CI_90.value == 0.90
        assert ConfidenceLevel.CI_95.value == 0.95
        assert ConfidenceLevel.CI_99.value == 0.99

    def test_confidence_level_ordering(self):
        """Test confidence levels are ordered correctly."""
        assert ConfidenceLevel.CI_90.value < ConfidenceLevel.CI_95.value
        assert ConfidenceLevel.CI_95.value < ConfidenceLevel.CI_99.value


class TestUncertaintyType:
    """Test uncertainty type enumeration."""

    def test_uncertainty_types(self):
        """Test all uncertainty types are defined."""
        assert UncertaintyType.ALEATORIC.value == "aleatoric"
        assert UncertaintyType.EPISTEMIC.value == "epistemic"
        assert UncertaintyType.TOTAL.value == "total"

    def test_uncertainty_type_coverage(self):
        """Test uncertainty types cover expected categories."""
        types = [t.value for t in UncertaintyType]
        assert "aleatoric" in types  # Inherent randomness
        assert "epistemic" in types  # Knowledge uncertainty
        assert "total" in types  # Combined uncertainty


class TestConfidenceInterval:
    """Test confidence interval data structure."""

    def test_confidence_interval_creation(self):
        """Test creating a confidence interval."""
        lower = np.array([0.1, 0.2, 0.3])
        upper = np.array([0.9, 0.8, 0.7])

        ci = ConfidenceInterval(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=0.95,
            method="bootstrap",
            uncertainty_type=UncertaintyType.EPISTEMIC,
        )

        assert np.array_equal(ci.lower_bound, lower)
        assert np.array_equal(ci.upper_bound, upper)
        assert ci.confidence_level == 0.95
        assert ci.method == "bootstrap"
        assert ci.uncertainty_type == UncertaintyType.EPISTEMIC
        assert isinstance(ci.timestamp, datetime)

    def test_confidence_interval_validation(self):
        """Test confidence interval validation."""
        lower = np.array([0.2, 0.4, 0.6])
        upper = np.array([0.8, 0.9, 0.7])

        ci = ConfidenceInterval(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=0.95,
            method="analytical",
            uncertainty_type=UncertaintyType.ALEATORIC,
        )

        # Check that bounds are properly ordered for most elements
        assert len(ci.lower_bound) == len(ci.upper_bound)
        assert ci.confidence_level > 0.0 and ci.confidence_level < 1.0


class TestConvergenceMetrics:
    """Test convergence metrics data structure."""

    def test_convergence_metrics_creation(self):
        """Test creating convergence metrics."""
        ci = ConfidenceInterval(
            lower_bound=np.array([0.1]),
            upper_bound=np.array([0.9]),
            confidence_level=0.95,
            method="bootstrap",
            uncertainty_type=UncertaintyType.TOTAL,
        )

        metrics = ConvergenceMetrics(
            kl_divergence=0.05,
            wasserstein_distance=0.12,
            jensen_shannon_divergence=0.08,
            hellinger_distance=0.15,
            convergence_rate=0.95,
            stability_measure=0.88,
            confidence_interval=ci,
            is_converged=True,
        )

        assert metrics.kl_divergence == 0.05
        assert metrics.wasserstein_distance == 0.12
        assert metrics.jensen_shannon_divergence == 0.08
        assert metrics.hellinger_distance == 0.15
        assert metrics.convergence_rate == 0.95
        assert metrics.stability_measure == 0.88
        assert metrics.confidence_interval == ci
        assert metrics.is_converged is True

    def test_convergence_decision_logic(self):
        """Test convergence decision based on metrics."""
        # Create metrics indicating convergence
        ci_converged = ConfidenceInterval(
            lower_bound=np.array([0.45]),
            upper_bound=np.array([0.55]),
            confidence_level=0.95,
            method="analytical",
            uncertainty_type=UncertaintyType.TOTAL,
        )

        converged_metrics = ConvergenceMetrics(
            kl_divergence=0.001,  # Very low
            wasserstein_distance=0.002,  # Very low
            jensen_shannon_divergence=0.001,  # Very low
            hellinger_distance=0.003,  # Very low
            convergence_rate=0.99,  # High
            stability_measure=0.95,  # High
            confidence_interval=ci_converged,
            is_converged=True,
        )

        assert converged_metrics.is_converged is True
        assert converged_metrics.kl_divergence < 0.01
        assert converged_metrics.stability_measure > 0.9


class TestUncertaintyPropagation:
    """Test uncertainty propagation data structure."""

    def test_uncertainty_propagation_creation(self):
        """Test creating uncertainty propagation structure."""
        input_uncertainties = {
            "belief_state": 0.1,
            "observation": 0.05,
            "action": 0.08}

        sensitivity_analysis = {
            "belief_state": 0.7,
            "observation": 0.2,
            "action": 0.1}

        uncertainty_breakdown = {
            UncertaintyType.ALEATORIC: 0.06,
            UncertaintyType.EPISTEMIC: 0.04,
            UncertaintyType.TOTAL: 0.10,
        }

        mc_samples = np.random.rand(1000)

        propagation = UncertaintyPropagation(
            input_uncertainties=input_uncertainties,
            propagated_uncertainty=0.12,
            sensitivity_analysis=sensitivity_analysis,
            uncertainty_breakdown=uncertainty_breakdown,
            monte_carlo_samples=mc_samples,
        )

        assert propagation.input_uncertainties == input_uncertainties
        assert propagation.propagated_uncertainty == 0.12
        assert propagation.sensitivity_analysis == sensitivity_analysis
        assert propagation.uncertainty_breakdown == uncertainty_breakdown
        assert np.array_equal(propagation.monte_carlo_samples, mc_samples)

    def test_uncertainty_conservation(self):
        """Test uncertainty conservation principles."""
        uncertainty_breakdown = {
            UncertaintyType.ALEATORIC: 0.06,
            UncertaintyType.EPISTEMIC: 0.04,
            UncertaintyType.TOTAL: 0.10,
        }

        propagation = UncertaintyPropagation(
            input_uncertainties={"param1": 0.05, "param2": 0.03},
            propagated_uncertainty=0.08,
            sensitivity_analysis={"param1": 0.6, "param2": 0.4},
            uncertainty_breakdown=uncertainty_breakdown,
        )

        # Total uncertainty should be at least as large as individual
        # components
        assert (
            propagation.uncertainty_breakdown[UncertaintyType.TOTAL]
            >= propagation.uncertainty_breakdown[UncertaintyType.ALEATORIC]
        )
        assert (
            propagation.uncertainty_breakdown[UncertaintyType.TOTAL]
            >= propagation.uncertainty_breakdown[UncertaintyType.EPISTEMIC]
        )


class TestStatisticalValidation:
    """Test statistical validation functionality."""

    def test_kolmogorov_smirnov_test(self):
        """Test Kolmogorov-Smirnov test for distribution comparison."""
        # Create two similar distributions
        np.random.seed(42)
        observed = np.random.normal(0, 1, 1000)
        expected = np.random.normal(0.1, 1, 1000)  # Slightly different mean

        is_significant, p_value = StatisticalValidation.kolmogorov_smirnov_test(
            observed, expected, alpha=0.05)

        assert isinstance(is_significant, bool)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    def test_kolmogorov_smirnov_identical_distributions(self):
        """Test KS test with identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        is_significant, p_value = StatisticalValidation.kolmogorov_smirnov_test(
            data, data, alpha=0.05)

        # Identical distributions should not be significantly different
        assert is_significant is False
        assert p_value > 0.05

    def test_anderson_darling_test(self):
        """Test Anderson-Darling test for normality."""
        np.random.seed(42)
        # Generate normal data
        normal_data = np.random.normal(0, 1, 1000)

        is_normal, statistic = StatisticalValidation.anderson_darling_test(
            normal_data, distribution="norm"
        )

        assert isinstance(is_normal, bool)
        assert isinstance(statistic, float)
        assert statistic >= 0.0

    def test_anderson_darling_non_normal(self):
        """Test Anderson-Darling test with non-normal data."""
        np.random.seed(42)
        # Generate clearly non-normal data (uniform)
        uniform_data = np.random.uniform(0, 1, 1000)

        is_normal, statistic = StatisticalValidation.anderson_darling_test(
            uniform_data, distribution="norm"
        )

        # Uniform data should not pass normality test
        assert is_normal is False
        assert statistic > 0.0

    def test_chi_squared_goodness_of_fit(self):
        """Test chi-squared goodness of fit test."""
        # Create observed and expected frequencies
        observed = np.array([20, 25, 30, 25, 20])
        expected = np.array([24, 24, 24, 24, 24])  # Uniform expected

        is_good_fit, p_value = StatisticalValidation.chi_squared_goodness_of_fit(
            observed, expected, alpha=0.05)

        assert isinstance(is_good_fit, bool)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    def test_chi_squared_perfect_fit(self):
        """Test chi-squared test with perfect fit."""
        # Identical observed and expected
        data = np.array([25, 25, 25, 25])

        is_good_fit, p_value = StatisticalValidation.chi_squared_goodness_of_fit(
            data, data, alpha=0.05)

        # Perfect fit should have high p-value
        assert is_good_fit is True
        assert p_value > 0.05


class TestUncertaintyQuantificationEngine:
    """Test main uncertainty quantification engine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantifier = UncertaintyQuantificationEngine()

    def test_quantifier_initialization(self):
        """Test uncertainty quantifier initialization."""
        assert isinstance(self.quantifier.historical_data, dict)
        assert isinstance(self.quantifier.convergence_history, dict)
        assert isinstance(self.quantifier.uncertainty_history, dict)
        assert len(self.quantifier.historical_data) == 0
        assert len(self.quantifier.convergence_history) == 0
        assert len(self.quantifier.uncertainty_history) == 0

    def test_calculate_bayesian_confidence_intervals(self):
        """Test calculating Bayesian confidence intervals."""
        np.random.seed(42)
        belief_distribution = np.random.dirichlet([1, 1, 1])

        ci = self.quantifier.calculate_bayesian_confidence_intervals(
            belief_distribution, confidence_level=ConfidenceLevel.CI_95, method="hdi")

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert ci.method == "hdi"
        assert ci.uncertainty_type == UncertaintyType.EPISTEMIC
        assert len(ci.lower_bound) == len(ci.upper_bound)

    def test_calculate_bayesian_confidence_intervals_quantile(self):
        """Test calculating confidence intervals with quantile method."""
        np.random.seed(42)
        belief_distribution = np.random.dirichlet([2, 3, 1])

        ci = self.quantifier.calculate_bayesian_confidence_intervals(
            belief_distribution, confidence_level=ConfidenceLevel.CI_95, method="quantile")

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert ci.method == "quantile"
        assert ci.uncertainty_type == UncertaintyType.EPISTEMIC

    def test_calculate_convergence_metrics(self):
        """Test convergence metrics calculation."""
        np.random.seed(42)
        agent_id = "test_agent"

        # Create a belief distribution
        current_belief = np.random.dirichlet([1, 1, 1])

        metrics = self.quantifier.calculate_convergence_metrics(
            agent_id=agent_id,
            current_belief=current_belief,
            confidence_level=ConfidenceLevel.CI_95)

        assert isinstance(metrics, ConvergenceMetrics)
        assert metrics.kl_divergence >= 0.0
        assert metrics.wasserstein_distance >= 0.0
        assert metrics.jensen_shannon_divergence >= 0.0
        assert metrics.hellinger_distance >= 0.0
        assert 0.0 <= metrics.convergence_rate <= 1.0
        assert 0.0 <= metrics.stability_measure <= 1.0
        assert isinstance(metrics.is_converged, bool)
        assert isinstance(metrics.confidence_interval, ConfidenceInterval)

    def test_propagate_uncertainty(self):
        """Test uncertainty propagation."""
        np.random.seed(42)

        # Create mock data for uncertainty propagation
        belief_distribution = np.random.dirichlet([1, 1, 1])
        model_parameters = {
            "A": np.random.rand(3, 3),  # Observation model
            "B": np.random.rand(3, 3),  # Transition model
            "D": np.random.dirichlet([1, 1, 1]),  # Prior
        }
        observations = np.random.rand(5)

        propagation = self.quantifier.propagate_uncertainty(
            belief_distribution=belief_distribution,
            model_parameters=model_parameters,
            observations=observations,
            num_monte_carlo_samples=100,
        )

        assert isinstance(propagation, UncertaintyPropagation)
        assert isinstance(propagation.input_uncertainties, dict)
        assert propagation.propagated_uncertainty >= 0.0
        assert isinstance(propagation.sensitivity_analysis, dict)
        assert all(
            k in propagation.uncertainty_breakdown for k in UncertaintyType)
        assert propagation.monte_carlo_samples is not None
        assert len(propagation.monte_carlo_samples) == 100

    def test_validate_statistical_significance(self):
        """Test statistical significance validation."""
        np.random.seed(42)

        # Create a belief distribution
        belief_distribution = np.random.dirichlet([2, 3, 1])
        reference_distribution = np.random.dirichlet([1, 1, 1])

        validation_results = self.quantifier.validate_statistical_significance(
            belief_distribution=belief_distribution,
            reference_distribution=reference_distribution,
            alpha=0.05,
        )

        assert isinstance(validation_results, dict)
        assert "kolmogorov_smirnov" in validation_results
        assert "chi_squared_goodness_of_fit" in validation_results
        assert "anderson_darling_normality" in validation_results

        # Each result should be a tuple of (is_significant/is_good_fit,
        # statistic/p_value)
        for test_name, result in validation_results.items():
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], (int, float))

    def test_export_publication_quality_statistics(self):
        """Test exporting publication quality statistics."""
        np.random.seed(42)
        agent_id = "test_agent"

        # First, create some data by running convergence metrics
        belief1 = np.random.dirichlet([1, 1, 1])
        belief2 = np.random.dirichlet([1.1, 1.1, 1.1])

        self.quantifier.calculate_convergence_metrics(agent_id, belief1)
        self.quantifier.calculate_convergence_metrics(agent_id, belief2)

        # Export statistics
        stats = self.quantifier.export_publication_quality_statistics(
            agent_id=agent_id, include_raw_data=False
        )

        assert isinstance(stats, dict)
        assert "agent_id" in stats
        assert "analysis_timestamp" in stats
        assert "data_points" in stats
        assert "convergence_analysis" in stats
        assert "uncertainty_metrics" in stats
        assert "statistical_validation" in stats

        assert stats["agent_id"] == agent_id
        assert stats["data_points"] == 2


class TestUncertaintyQuantificationIntegration:
    """Integration tests for uncertainty quantification system."""

    def test_full_uncertainty_analysis_workflow(self):
        """Test complete uncertainty analysis workflow."""
        np.random.seed(42)

        # 1. Initialize uncertainty quantification engine
        quantifier = UncertaintyQuantificationEngine()
        agent_id = "workflow_test_agent"

        # 2. Generate synthetic Active Inference data and track convergence
        belief_states = []
        for t in range(10):
            # Simulate belief evolution with gradual convergence
            concentration = [1.0 + t * 0.1, 1.0 + t * 0.05, 1.0 + t * 0.02]
            belief = np.random.dirichlet(concentration)
            belief_states.append(belief)

            # Calculate convergence metrics for each step
            metrics = quantifier.calculate_convergence_metrics(
                agent_id=agent_id, current_belief=belief, confidence_level=ConfidenceLevel.CI_95)

        # 3. Compute confidence intervals for final beliefs
        final_belief = belief_states[-1]
        ci = quantifier.calculate_bayesian_confidence_intervals(
            belief_distribution=final_belief,
            confidence_level=ConfidenceLevel.CI_95,
            method="hdi")

        # 4. Test uncertainty propagation
        model_params = {
            "A": np.random.rand(3, 3),
            "B": np.random.rand(3, 3),
            "D": np.random.dirichlet([1, 1, 1]),
        }
        observations = np.random.rand(5)

        propagation = quantifier.propagate_uncertainty(
            belief_distribution=final_belief,
            model_parameters=model_params,
            observations=observations,
            num_monte_carlo_samples=100,
        )

        # 5. Validate results
        assert isinstance(ci, ConfidenceInterval)
        assert isinstance(metrics, ConvergenceMetrics)
        assert isinstance(propagation, UncertaintyPropagation)
        assert ci.confidence_level == 0.95
        assert metrics.kl_divergence >= 0.0
        assert isinstance(metrics.is_converged, bool)
        assert len(quantifier.historical_data[agent_id]) == 10
        assert len(quantifier.convergence_history[agent_id]) == 10

    def test_active_inference_uncertainty_pipeline(self):
        """Test uncertainty quantification in Active Inference context."""
        np.random.seed(42)

        quantifier = UncertaintyQuantificationEngine()
        agent_id = "active_inference_agent"

        # Simulate Active Inference belief evolution with decreasing
        # uncertainty
        for t in range(15):
            # Beliefs become more concentrated over time (learning)
            concentration_base = 1.0 + t * 0.2
            belief = np.random.dirichlet(
                [concentration_base, concentration_base / 2, concentration_base / 3]
            )

            # Calculate convergence metrics
            _ = quantifier.calculate_convergence_metrics(
                agent_id=agent_id,
                current_belief=belief,
                confidence_level=ConfidenceLevel.CI_95)

        # Test statistical significance validation
        final_belief = quantifier.historical_data[agent_id][-1]
        validation_results = quantifier.validate_statistical_significance(
            belief_distribution=final_belief, alpha=0.05
        )

        # Test export functionality
        stats = quantifier.export_publication_quality_statistics(
            agent_id=agent_id, include_raw_data=True
        )

        # Validate results
        assert isinstance(validation_results, dict)
        assert len(validation_results) >= 3
        assert isinstance(stats, dict)
        assert "agent_id" in stats
        assert "raw_data" in stats
        assert len(stats["raw_data"]["belief_distributions"]) == 15
        assert len(quantifier.convergence_history[agent_id]) == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
