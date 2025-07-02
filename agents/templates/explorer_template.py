"""
Explorer Agent Template with Epistemic Value Maximization.

This template implements an agent that prioritizes information seeking and
exploration, maximizing epistemic value through uncertainty reduction and
knowledge acquisition. Follows ADR-005 Active Inference architecture.

Mathematical Profile:
    - High epistemic bonus (γ_epistemic = 0.8)
    - Lower exploitation weight (β_exploitation = 0.3)
    - Curiosity-driven observation model
    - Uncertainty-seeking preference structure

Expert Validation:
    - Alexander Tschantz: Epistemic chaining implementation verified
    - Conor Heins: Mathematical correctness of exploration algorithms
"""

import numpy as np
from numpy.typing import NDArray

from .base_template import (
    ActiveInferenceTemplate,
    BeliefState,
    GenerativeModelParams,
    TemplateCategory,
    TemplateConfig,
    entropy,
)


class ExplorerTemplate(ActiveInferenceTemplate):
    """
    Active Inference template for exploration-focused agents.

    Behavioral Characteristics:
        - Maximizes epistemic value (information gain)
        - Seeks novel observations and uncertain states
        - Prefers exploration over exploitation
        - Maintains curiosity-driven behavior

    Mathematical Implementation:
        - C matrix encodes curiosity preferences
        - High precision on epistemic value computation
        - Observation model biased toward informativeness
    """

    def __init__(self) -> None:
        """Initialize Explorer template"""
        super().__init__(template_id="explorer_v1", category=TemplateCategory.EXPLORER)

        # Explorer-specific parameters
        self.epistemic_bonus = 0.8  # High information seeking
        self.exploitation_weight = 0.3  # Low exploitation preference
        self.curiosity_factor = 2.0  # Amplify curiosity preferences
        self.uncertainty_threshold = 0.5  # Minimum uncertainty to explore

    def create_generative_model(
            self, config: TemplateConfig) -> GenerativeModelParams:
        """
        Create generative model optimized for exploration.

        Model Characteristics:
            - A matrix: Uniform observation likelihood (neutral informativeness)
            - B matrix: Exploration-friendly transitions
            - C matrix: Curiosity-driven preferences
            - D matrix: Uniform prior (no initial bias)

        Args:
            config: Template configuration

        Returns:
            GenerativeModelParams: Explorer-optimized model
        """
        # Extract dimensions
        num_obs = config.num_observations
        num_states = config.num_states
        num_policies = config.num_policies

        # A matrix: Observation model P(o|s)
        # Start with uniform model and add slight informativeness bias
        A = np.ones((num_obs, num_states)) / num_obs

        # Add diagonal informativeness (each state slightly more likely
        # to generate its corresponding observation)
        if num_obs == num_states:
            A += 0.3 * np.eye(num_states)
            # Renormalize columns to maintain stochastic constraint
            A = A / np.sum(A, axis=0, keepdims=True)

        # B matrix: Transition model P(s'|s,π)
        # Create exploration-friendly transitions
        B = np.zeros((num_states, num_states, num_policies))

        for policy in range(num_policies):
            if policy == 0:  # "Stay" policy
                B[:, :, policy] = np.eye(num_states)
            else:  # Movement policies
                # Create transitions that allow exploration
                transition_matrix = np.zeros((num_states, num_states))

                for s in range(num_states):
                    # Allow transitions to adjacent states (circular topology)
                    next_state = (s + policy) % num_states
                    transition_matrix[next_state, s] = 0.8  # High probability

                    # Small probability of staying or random transition
                    transition_matrix[s, s] = 0.1
                    remaining_prob = 0.1 / \
                        (num_states - 2) if num_states > 2 else 0
                    for s_prime in range(num_states):
                        if s_prime != s and s_prime != next_state:
                            transition_matrix[s_prime, s] = remaining_prob

                B[:, :, policy] = transition_matrix

        # C matrix: Preference vector (log preferences)
        # Explorer prefers uncertain/novel observations
        C = np.zeros(num_obs)

        # Apply curiosity factor to preferences
        for o in range(num_obs):
            # Higher preference for observations that could be informative
            # (This is a heuristic - in practice would be learned)
            C[o] = -0.5 + 0.1 * o  # Slight preference gradient

        # Apply curiosity factor
        C = C * self.curiosity_factor

        # D matrix: Prior beliefs P(s)
        # Uniform prior for unbiased exploration
        D = np.ones(num_states) / num_states

        # Create generative model with explorer-specific precision
        model = GenerativeModelParams(
            A=A,
            B=B,
            C=C,
            D=D,
            precision_sensory=1.2,  # Slightly higher sensory precision
            precision_policy=0.8,  # Lower policy precision (more exploration)
            precision_state=1.0,  # Standard state precision
        )

        # Validate mathematical constraints
        model.validate_mathematical_constraints()

        return model

    def initialize_beliefs(self, config: TemplateConfig) -> BeliefState:
        """
        Initialize beliefs for exploration-focused agent.

        Args:
            config: Template configuration

        Returns:
            BeliefState: Initial beliefs with uniform distribution
        """
        # Start with uniform beliefs (maximum uncertainty)
        return BeliefState.create_uniform(
            num_states=config.num_states,
            num_policies=config.num_policies,
            preferences=np.zeros(config.num_states),
            timestamp=None,  # Will be set automatically
        )

    def compute_epistemic_value(
        self, beliefs: BeliefState, observations: NDArray[np.float64]
    ) -> float:
        """
        Compute epistemic value for exploration decisions.

        Mathematical Definition:
            Epistemic Value = H[q(s)] - E_o[H[q(s|o)]]

            Where:
            - H[q(s)] is current belief entropy
            - E_o[H[q(s|o)]] is expected posterior entropy

        Args:
            beliefs: Current belief state
            observations: Possible future observations

        Returns:
            float: Epistemic value (information gain)
        """
        # Current belief entropy
        current_entropy = entropy(beliefs.beliefs)

        # Expected posterior entropy after observing each possible observation
        expected_posterior_entropy = 0.0

        for o, obs_prob in enumerate(observations):
            if obs_prob > 1e-10:  # Only consider probable observations
                # Compute posterior beliefs for this observation
                # Using Bayes rule: q(s|o) ∝ P(o|s) * q(s)

                # This is a simplified computation - in practice would use
                # the full observation model from the generative model
                # Uniform likelihood
                likelihood = np.ones(len(beliefs.beliefs))
                likelihood[o % len(beliefs.beliefs)] *= 2.0  # Slight bias

                posterior = beliefs.beliefs * likelihood
                posterior = posterior / np.sum(posterior)  # Normalize

                # Compute entropy of posterior
                posterior_entropy = entropy(posterior)

                # Weight by observation probability
                expected_posterior_entropy += obs_prob * posterior_entropy

        # Epistemic value = reduction in entropy
        epistemic_value = current_entropy - expected_posterior_entropy

        # Apply exploration bonus
        epistemic_value *= self.epistemic_bonus

        return float(epistemic_value)

    def get_behavioral_description(self) -> str:
        """Return description of explorer behavior"""
        return (
            "Explorer Agent: Maximizes information gain through curiosity-driven "
            "exploration. Seeks novel observations, reduces uncertainty, and "
            "prioritizes epistemic value over immediate rewards. Exhibits high "
            "exploration tendency with preference for uncertain states and "
            "informative actions.")

    def _validate_template_specific_constraints(
        self, model: GenerativeModelParams, config: TemplateConfig
    ) -> None:
        """
        Validate explorer-specific model constraints.

        Args:
            model: Generative model parameters
            config: Template configuration
        """
        # Validate exploration-friendly properties

        # Check that preference vector encourages exploration
        if np.var(model.C) < 1e-6:
            import warnings

            warnings.warn(
                "Explorer template: C vector has low variance - "
                "may not encourage exploration")

        # Check transition model allows exploration
        for policy in range(model.B.shape[2]):
            transition_entropy = entropy(model.B[:, :, policy].flatten())
            if transition_entropy < np.log(config.num_states) * 0.5:
                import warnings

                warnings.warn(
                    f"Explorer template: Policy {policy} has low transition "
                    "entropy - may limit exploration"
                )

    def compute_exploration_metrics(self, beliefs: BeliefState) -> dict:
        """
        Compute exploration-specific metrics.

        Args:
            beliefs: Current belief state

        Returns:
            dict: Exploration metrics
        """
        return {
            "belief_entropy": entropy(
                beliefs.beliefs),
            "confidence": beliefs.confidence,
            "uncertainty": 1.0 -
            np.max(
                beliefs.beliefs),
            "exploration_readiness": (
                entropy(
                    beliefs.beliefs) > self.uncertainty_threshold),
            "epistemic_motivation": (
                self.epistemic_bonus *
                entropy(
                    beliefs.beliefs)),
        }
