"""
Test Suite for PyMDP Integration Layer.

This module provides comprehensive tests for the pymdp integration layer,
validating mathematical correctness, ADR-005 compliance, and expert
committee requirements.

Test Categories:
    - Mathematical Validation: Belief updates, free energy computation
    - PyMDP API Integration: Official library usage verification
    - Fallback Implementation: Graceful degradation testing
    - Expert Committee Criteria: Conor Heins, Alexander Tschantz validation

Expert Committee Requirements:
    - Unit tests comparing pymdp outputs to analytical solutions
    - Mathematical fidelity verification
    - Performance benchmarks for real-time deployment
"""

import unittest
from typing import Dict

import numpy as np
import numpy.testing as npt

# Import our template system
from .base_template import BeliefState, TemplateCategory, TemplateConfig
from .explorer_template import ExplorerTemplate
from .pymdp_integration import PyMDPAgentWrapper, create_pymdp_agent


# Mathematical validation functions
def entropy(p: np.ndarray, axis: int = -1) -> float:
    """Compute entropy with numerical stability"""
    return -np.sum(p * np.log(p + 1e-16), axis=axis)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence with numerical stability"""
    return np.sum(p * np.log((p + 1e-16) / (q + 1e-16)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax with numerical stability"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class TestPyMDPIntegration(unittest.TestCase):
    """Test suite for PyMDP integration layer"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Create test configuration
        self.config = TemplateConfig(
            template_id="test_explorer",
            category=TemplateCategory.EXPLORER,
            num_states=4,
            num_observations=3,
            num_policies=3,
            exploration_bonus=0.5,
            exploitation_weight=0.8,
            planning_horizon=2,
        )

        # Create test explorer template
        self.explorer_template = ExplorerTemplate()

        # Generate test generative model
        self.model_params = self.explorer_template.create_generative_model(self.config)

        # Create pymdp agent wrapper
        self.agent_wrapper = create_pymdp_agent(self.model_params, self.config)

    def test_generative_model_mathematical_constraints(self) -> None:
        """Test that generative model satisfies mathematical constraints"""
        # Test A matrix (observation model) stochasticity
        A_col_sums = np.sum(self.model_params.A, axis=0)
        npt.assert_allclose(A_col_sums, 1.0, atol=1e-10, err_msg="A matrix columns must sum to 1")

        # Test B tensor (transition models) stochasticity
        for policy in range(self.model_params.B.shape[2]):
            B_col_sums = np.sum(self.model_params.B[:, :, policy], axis=0)
            npt.assert_allclose(
                B_col_sums, 1.0, atol=1e-10, err_msg=f"B[:,:,{policy}] must be stochastic"
            )

        # Test D vector (prior) normalization
        D_sum = np.sum(self.model_params.D)
        npt.assert_allclose(D_sum, 1.0, atol=1e-10, err_msg="D vector must sum to 1")

        # Test non-negativity constraints
        self.assertTrue(np.all(self.model_params.A >= 0), "A matrix must be non-negative")
        self.assertTrue(np.all(self.model_params.B >= 0), "B tensor must be non-negative")
        self.assertTrue(np.all(self.model_params.D >= 0), "D vector must be non-negative")

    def test_belief_state_mathematical_properties(self) -> None:
        """Test that belief states satisfy mathematical properties"""
        # Initialize beliefs
        beliefs = self.explorer_template.initialize_beliefs(self.config)

        # Test probability simplex constraint
        belief_sum = np.sum(beliefs.beliefs)
        npt.assert_allclose(belief_sum, 1.0, atol=1e-10, err_msg="Beliefs must sum to 1")

        # Test non-negativity
        self.assertTrue(np.all(beliefs.beliefs >= 0), "Beliefs must be non-negative")

        # Test confidence bounds
        max_entropy = np.log(len(beliefs.beliefs))
        self.assertGreaterEqual(beliefs.confidence, 0, "Confidence must be non-negative")
        self.assertLessEqual(
            beliefs.confidence, max_entropy, "Confidence must be bounded by max entropy"
        )

        # Test entropy consistency
        expected_entropy = entropy(beliefs.beliefs)
        npt.assert_allclose(
            beliefs.confidence,
            expected_entropy,
            atol=1e-10,
            err_msg="Confidence must equal belief entropy",
        )

    def test_bayesian_belief_update_mathematical_correctness(self) -> None:
        """Test Bayesian belief update: P(s|o) ∝ P(o|s)P(s)"""
        # Get initial beliefs
        initial_beliefs = self.explorer_template.initialize_beliefs(self.config)

        # Test observation
        observation = 1  # Second observation

        # Perform belief update
        updated_beliefs = self.agent_wrapper.update_beliefs(observation)

        # Analytical Bayesian update for validation
        likelihood = self.model_params.A[observation, :]  # P(o|s)
        analytical_posterior = initial_beliefs.beliefs * likelihood  # P(o|s) * P(s)
        analytical_posterior = analytical_posterior / np.sum(analytical_posterior)  # Normalize

        # Compare numerical and analytical results
        npt.assert_allclose(
            updated_beliefs.beliefs,
            analytical_posterior,
            atol=1e-6,
            err_msg="Belief update must follow Bayes rule: P(s|o) ∝ P(o|s)P(s)",
        )

        # Test belief normalization
        npt.assert_allclose(
            np.sum(updated_beliefs.beliefs),
            1.0,
            atol=1e-10,
            err_msg="Updated beliefs must be normalized",
        )

    def test_free_energy_computation_mathematical_correctness(self) -> None:
        """Test free energy: F = D_KL[q(s)||P(s)] - E_q[ln P(o|s)]"""
        # Get beliefs and observation
        beliefs = self.explorer_template.initialize_beliefs(self.config)
        observation = 0

        # Compute free energy using wrapper
        computed_fe = self.agent_wrapper.compute_free_energy(beliefs, observation)

        # Analytical free energy computation
        # KL divergence from prior: D_KL[q(s)||P(s)]
        kl_prior = kl_divergence(beliefs.beliefs, self.model_params.D)

        # Expected log-likelihood: E_q[ln P(o|s)]
        log_likelihood = np.dot(
            beliefs.beliefs, np.log(self.model_params.A[observation, :] + 1e-16)
        )

        # Free energy = KL divergence - expected log-likelihood
        analytical_fe = kl_prior - log_likelihood

        # Compare results
        npt.assert_allclose(
            computed_fe,
            analytical_fe,
            atol=1e-6,
            err_msg="Free energy must equal D_KL[q(s)||P(s)] - E_q[ln P(o|s)]",
        )

    def test_epistemic_value_computation(self) -> None:
        """Test epistemic value computation for exploration"""
        # Create beliefs with some uncertainty
        beliefs = BeliefState.create_uniform(
            num_states=self.config.num_states, num_policies=self.config.num_policies
        )

        # Create test observation distribution
        obs_distribution = np.array([0.7, 0.2, 0.1])

        # Compute epistemic value
        epistemic_value = self.explorer_template.compute_epistemic_value(beliefs, obs_distribution)

        # Epistemic value should be positive for uncertain beliefs
        self.assertGreater(
            epistemic_value, 0, "Epistemic value should be positive for uncertain beliefs"
        )

        # Test with certain beliefs (low epistemic value)
        certain_beliefs = BeliefState(
            beliefs=np.array([0.95, 0.03, 0.01, 0.01]),
            policies=beliefs.policies,
            preferences=beliefs.preferences,
            timestamp=beliefs.timestamp,
            confidence=entropy(np.array([0.95, 0.03, 0.01, 0.01])),
        )

        certain_epistemic_value = self.explorer_template.compute_epistemic_value(
            certain_beliefs, obs_distribution
        )

        # Uncertain beliefs should have higher epistemic value
        self.assertGreater(
            epistemic_value,
            certain_epistemic_value,
            "Uncertain beliefs should have higher epistemic value",
        )

    def test_policy_inference_mathematical_properties(self) -> None:
        """Test policy inference using expected free energy minimization"""
        # Get initial beliefs
        beliefs = self.explorer_template.initialize_beliefs(self.config)

        # Infer policies
        policies = self.agent_wrapper.infer_policies(beliefs)

        # Test policy distribution properties
        npt.assert_allclose(
            np.sum(policies), 1.0, atol=1e-10, err_msg="Policies must form probability distribution"
        )

        self.assertTrue(np.all(policies >= 0), "Policy probabilities must be non-negative")

        self.assertEqual(
            len(policies), self.config.num_policies, "Policy vector must match number of policies"
        )

    def test_precision_parameter_integration(self) -> None:
        """Test ADR-005 precision parameter integration"""
        # Validate precision parameters are properly set
        summary = self.agent_wrapper.get_mathematical_summary()

        self.assertIn("precision_sensory", summary, "Must include sensory precision (γ)")
        self.assertIn("precision_policy", summary, "Must include policy precision (β)")
        self.assertIn("precision_state", summary, "Must include state precision (α)")

        # Test precision values are positive
        self.assertGreater(summary["precision_sensory"], 0, "Sensory precision must be positive")
        self.assertGreater(summary["precision_policy"], 0, "Policy precision must be positive")
        self.assertGreater(summary["precision_state"], 0, "State precision must be positive")

    def test_explorer_template_behavioral_properties(self) -> None:
        """Test explorer-specific behavioral properties"""
        # Test epistemic bonus
        self.assertGreater(
            self.explorer_template.epistemic_bonus, 0.5, "Explorer should have high epistemic bonus"
        )

        # Test exploitation weight
        self.assertLess(
            self.explorer_template.exploitation_weight,
            0.5,
            "Explorer should have low exploitation weight",
        )

        # Test behavioral description
        description = self.explorer_template.get_behavioral_description()
        self.assertIn("exploration", description.lower(), "Description must mention exploration")
        self.assertIn("epistemic", description.lower(), "Description must mention epistemic value")

    def test_mathematical_summary_completeness(self) -> None:
        """Test mathematical summary includes all required metrics"""
        summary = self.agent_wrapper.get_mathematical_summary()

        required_keys = [
            "belief_entropy",
            "belief_sum",
            "precision_sensory",
            "precision_policy",
            "precision_state",
            "model_dimensions",
            "pymdp_available",
            "agent_initialized",
        ]

        for key in required_keys:
            self.assertIn(key, summary, f"Mathematical summary must include {key}")

        # Test belief sum is approximately 1
        npt.assert_allclose(
            summary["belief_sum"], 1.0, atol=1e-10, err_msg="Belief sum must be 1.0"
        )

        # Test model dimensions match configuration
        dims = summary["model_dimensions"]
        self.assertEqual(dims["num_states"], self.config.num_states)
        self.assertEqual(dims["num_observations"], self.config.num_observations)
        self.assertEqual(dims["num_policies"], self.config.num_policies)

    def test_fallback_behavior_without_pymdp(self) -> None:
        """Test graceful fallback when pymdp is not available"""
        # This test validates that the system works even without pymdp
        # The fallback implementations should provide mathematically correct
        # results

        # Create wrapper that forces fallback mode
        fallback_wrapper = PyMDPAgentWrapper(self.model_params, self.config)
        fallback_wrapper.agent = None  # Force fallback mode

        # Test belief update still works
        self.explorer_template.initialize_beliefs(self.config)
        updated_beliefs = fallback_wrapper.update_beliefs(1)

        # Should still satisfy mathematical constraints
        npt.assert_allclose(
            np.sum(updated_beliefs.beliefs),
            1.0,
            atol=1e-10,
            err_msg="Fallback belief update must be normalized",
        )

        # Test free energy computation still works
        fe = fallback_wrapper.compute_free_energy(updated_beliefs, 1)
        self.assertIsInstance(fe, float, "Fallback free energy must return float")

    def test_integration_with_agent_data(self) -> None:
        """Test integration with AgentData model"""
        from ..base.data_model import Position

        # Create agent data using template
        position = Position(1.0, 2.0, 0.0)
        agent_data = self.explorer_template.create_agent_data(self.config, position)

        # Validate agent data properties
        self.assertEqual(agent_data.agent_type, "explorer")
        self.assertEqual(agent_data.position, position)

        # Check metadata includes template information
        metadata = agent_data.metadata
        self.assertEqual(metadata["template_id"], "explorer_v1")
        self.assertEqual(metadata["template_category"], "explorer")

        # Validate model dimensions in metadata
        model_dims = metadata["model_dimensions"]
        self.assertEqual(model_dims["num_states"], self.config.num_states)
        self.assertEqual(model_dims["num_observations"], self.config.num_observations)
        self.assertEqual(model_dims["num_policies"], self.config.num_policies)


class TestMathematicalValidation(unittest.TestCase):
    """Additional mathematical validation tests"""

    def test_belief_update_convergence(self) -> None:
        """Test belief update convergence properties"""
        # Create simple 2-state, 2-observation model
        config = TemplateConfig(
            template_id="test_convergence",
            category=TemplateCategory.EXPLORER,
            num_states=2,
            num_observations=2,
            num_policies=2,
        )

        explorer = ExplorerTemplate()
        model = explorer.create_generative_model(config)
        wrapper = create_pymdp_agent(model, config)

        # Start with uniform beliefs
        beliefs = explorer.initialize_beliefs(config)

        # Apply consistent observation multiple times
        observation = 0
        for _ in range(10):
            beliefs = wrapper.update_beliefs(observation)

        # Beliefs should converge to certainty about state that generates
        # observation
        max_belief = np.max(beliefs.beliefs)
        self.assertGreater(max_belief, 0.8, "Repeated observations should increase certainty")

    def test_free_energy_minimization_property(self) -> None:
        """Test that free energy decreases with informative observations"""
        config = TemplateConfig(
            template_id="test_fe_min",
            category=TemplateCategory.EXPLORER,
            num_states=3,
            num_observations=3,
            num_policies=2,
        )

        explorer = ExplorerTemplate()
        model = explorer.create_generative_model(config)
        wrapper = create_pymdp_agent(model, config)

        # Start with uniform beliefs (high uncertainty/entropy)
        initial_beliefs = explorer.initialize_beliefs(config)
        initial_fe = wrapper.compute_free_energy(initial_beliefs, 0)

        # Update with informative observation
        updated_beliefs = wrapper.update_beliefs(0)
        updated_fe = wrapper.compute_free_energy(updated_beliefs, 0)

        # Free energy should decrease (or at least not increase significantly)
        # This tests the principle that informative observations reduce free
        # energy
        self.assertLessEqual(
            updated_fe,
            initial_fe + 1e-6,
            "Free energy should not increase with informative observations",
        )


def run_mathematical_validation_suite() -> Dict[str, bool]:
    """
    Run complete mathematical validation suite.

    Returns:
        Dict: Test results summary
    """
    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(unittest.makeSuite(TestPyMDPIntegration))
    suite.addTest(unittest.makeSuite(TestMathematicalValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return summary
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
        "mathematical_correctness_validated": result.wasSuccessful(),
    }


if __name__ == "__main__":
    # Run mathematical validation
    print("🔬 Running Mathematical Validation Suite for PyMDP Integration")
    print("=" * 70)

    results = run_mathematical_validation_suite()

    print("\n" + "=" * 70)
    print("📊 VALIDATION RESULTS:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(
        f"Mathematical Correctness: {
            '✅ VALIDATED' if results['success'] else '❌ FAILED'}"
    )

    if results["success"]:
        print("\n🎉 All mathematical validations PASSED!")
        print("PyMDP integration meets Expert Committee standards.")
    else:
        print("\n⚠️  Some validations FAILED - review required.")

    print("=" * 70)
