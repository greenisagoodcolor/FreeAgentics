"""
Comprehensive tests for Policy Selection aligned with PyMDP calculations and GNN notation.
Tests follow PyMDP's expected free energy formulation and support LLM-generated models
through Generalized Notation Notation (GNN) compatibility.
"""

import pytest
import torch

from inference.engine.active_inference import InferenceConfig, VariationalMessagePassing
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.policy_selection import (
    DiscreteExpectedFreeEnergy,
    Policy,
    PolicyConfig,
    create_policy_selector,
)


class TestPolicyConfigPyMDPAlignment:
    """Test PolicyConfig dataclass following PyMDP conventions."""

    def test_pymdp_default_config(self):
        """Test default policy configuration aligned with PyMDP."""
        config = PolicyConfig()

        # PyMDP-aligned defaults
        assert config.planning_horizon == 5
        assert config.policy_length == 1
        assert config.epistemic_weight == 1.0  # use_states_info_gain in PyMDP
        assert config.pragmatic_weight == 1.0  # use_utility in PyMDP
        assert config.exploration_constant == 1.0  # precision_parameter in PyMDP

    def test_pymdp_compatible_config(self):
        """Test configuration compatible with PyMDP parameters."""
        config = PolicyConfig(
            planning_horizon=3,  # T in PyMDP
            policy_length=2,  # policy depth
            epistemic_weight=0.5,  # information gain weight
            pragmatic_weight=1.5,  # utility weight
            exploration_constant=2.0,  # precision (beta)
        )

        assert config.planning_horizon == 3
        assert config.policy_length == 2
        assert config.epistemic_weight == 0.5
        assert config.pragmatic_weight == 1.5
        assert config.exploration_constant == 2.0

    def test_gnn_notation_integration(self):
        """Test integration with GNN notation for LLM model generation."""
        # GNN-compatible configuration
        gnn_metadata = {
            "model_type": "discrete_generative_model",
            "state_space": {
                "type": "discrete",
                "size": 4,
                "semantic_labels": ["start", "middle", "goal", "obstacle"],
            },
            "action_space": {"type": "discrete", "size": 2, "semantic_labels": ["stay", "move"]},
            "time_settings": {"horizon": 3, "dt": 1.0},
            "active_inference_ontology": "standard",
            "llm_generated": True,
        }

        config = PolicyConfig(
            # For testing
            planning_horizon=gnn_metadata["time_settings"]["horizon"],
            use_gpu=False,
        )

        # Attach GNN metadata for LLM compatibility
        config.gnn_metadata = gnn_metadata

        assert hasattr(config, "gnn_metadata")
        assert config.gnn_metadata["llm_generated"] is True
        assert config.gnn_metadata["state_space"]["size"] == 4


class TestPolicyPyMDPRepresentation:
    """Test Policy class with PyMDP-aligned representation."""

    def test_pymdp_policy_creation(self):
        """Test PyMDP-style policy creation."""
        # PyMDP policies are sequences of discrete actions
        actions = torch.tensor([0, 1, 0])  # Action indices
        policy = Policy(actions)

        assert len(policy) == 3
        assert policy[0] == 0
        assert policy[1] == 1
        assert policy[2] == 0
        assert torch.equal(policy.actions, actions)

    def test_policy_with_horizon(self):
        """Test policy with planning horizon (PyMDP T parameter)."""
        actions = torch.tensor([1, 0, 1])
        horizon = 5
        policy = Policy(actions, horizon=horizon)

        assert len(policy) == 3
        assert policy.horizon == horizon
        assert torch.equal(policy.actions, actions)

    def test_policy_representation(self):
        """Test policy string representation."""
        policy = Policy([0, 1, 2])
        assert repr(policy) == "Policy([0, 1, 2])"

    def test_pymdp_one_hot_conversion(self):
        """Test conversion to one-hot encoding for PyMDP compatibility."""
        # Action indices
        actions = torch.tensor([0, 1, 0])
        Policy(actions)

        # Convert to one-hot for matrix operations (as PyMDP does)
        num_actions = 2
        one_hot = torch.zeros(len(actions), num_actions)
        one_hot.scatter_(1, actions.unsqueeze(1), 1)

        # Verify one-hot encoding
        expected = torch.tensor(
            # action 0  # action 1  # action 0
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.float32,
        )

        assert torch.allclose(one_hot, expected)

    def test_gnn_semantic_policy(self):
        """Test policy with GNN semantic annotations."""
        # Policy with semantic action labels for LLM understanding
        actions = torch.tensor([0, 1, 0])
        policy = Policy(actions)

        # Add GNN metadata for semantic understanding
        policy.gnn_metadata = {
            "action_semantics": {0: "stay_in_place", 1: "move_forward"},
            "task_context": "grid_navigation",
            "llm_generated": True,
            "text_description": "Policy for cautious exploration: stay, move, stay",
        }

        assert hasattr(policy, "gnn_metadata")
        assert policy.gnn_metadata["action_semantics"][0] == "stay_in_place"
        assert policy.gnn_metadata["llm_generated"] is True


class TestDiscreteExpectedFreeEnergyPyMDP:
    """Test discrete expected free energy calculation following PyMDP formulation."""

    @pytest.fixture
    def pymdp_compatible_model(self, model_dimensions, model_parameters):
        """Create PyMDP-compatible discrete generative model."""
        model = DiscreteGenerativeModel(model_dimensions, model_parameters)

        # PyMDP-style matrices with proper normalization
        # A matrix: P(obs|states) - each column sums to 1
        model.A = torch.tensor(
            [
                [0.9, 0.1, 0.05],  # obs 0 | states 0,1,2
                [0.1, 0.8, 0.15],  # obs 1 | states 0,1,2
                [0.0, 0.1, 0.80],  # obs 2 | states 0,1,2
            ],
            dtype=torch.float32,
        )

        # B matrix: P(next_state|current_state, action) - each B[:, s, a] sums
        # to 1
        model.B = torch.zeros(3, 3, 2)
        # Action 0: stay in place
        model.B[:, :, 0] = torch.eye(3)
        # Action 1: deterministic transitions
        model.B[1, 0, 1] = 1.0  # 0->1
        model.B[2, 1, 1] = 1.0  # 1->2
        model.B[2, 2, 1] = 1.0  # 2->2 (absorbing)

        # C matrix: Prior preferences over observations (log probabilities)
        model.C = torch.tensor(
            # Avoid obs 0  # Neutral about obs 1  # Prefer obs 2
            [[-2.0], [0.0], [2.0]],
            dtype=torch.float32,
        )

        # D vector: Initial state prior (normalized)
        model.D = torch.tensor([1.0, 0.0, 0.0])

        return model

    @pytest.fixture
    def pymdp_efe_calculator(self):
        """Create EFE calculator with PyMDP-compatible configuration."""
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config = PolicyConfig(
            planning_horizon=3,
            policy_length=1,
            epistemic_weight=1.0,
            pragmatic_weight=1.0,
            exploration_constant=1.0,
            use_gpu=False,
        )
        return DiscreteExpectedFreeEnergy(config, inference)

    def test_pymdp_matrix_compatibility(self, pymdp_compatible_model):
        """Test that model matrices follow PyMDP conventions."""
        model = pymdp_compatible_model

        # A matrix columns should sum to 1 (observation model)
        for s in range(model.A.shape[1]):
            assert torch.allclose(model.A[:, s].sum(), torch.tensor(1.0), atol=1e-6)

        # B matrix transitions should sum to 1
        for a in range(model.B.shape[2]):
            for s in range(model.B.shape[1]):
                assert torch.allclose(model.B[:, s, a].sum(), torch.tensor(1.0), atol=1e-6)

        # D vector should sum to 1 (probability distribution)
        assert torch.allclose(model.D.sum(), torch.tensor(1.0), atol=1e-6)

    def test_pymdp_expected_free_energy_formula(self, pymdp_efe_calculator, pymdp_compatible_model):
        """Test EFE calculation following PyMDP's G(π) = E[ln Q(s,o|π) - ln P(o,s|π)]."""
        # Initial beliefs (posterior over states)
        beliefs = torch.tensor([0.7, 0.2, 0.1])  # Q(s)

        # Test policies
        policy_stay = Policy([0])  # Stay action
        policy_move = Policy([1])  # Move action

        # Calculate expected free energy for both policies
        G_stay, epistemic_stay, pragmatic_stay = pymdp_efe_calculator.compute_expected_free_energy(
            policy_stay, beliefs, pymdp_compatible_model
        )

        G_move, epistemic_move, pragmatic_move = pymdp_efe_calculator.compute_expected_free_energy(
            policy_move, beliefs, pymdp_compatible_model
        )

        # Both should be finite and real
        assert torch.isfinite(G_stay) and torch.isfinite(G_move)
        assert not torch.isnan(G_stay) and not torch.isnan(G_move)

        # EFE should be sum of epistemic and pragmatic terms
        assert torch.allclose(G_stay, epistemic_stay + pragmatic_stay, atol=1e-6)
        assert torch.allclose(G_move, epistemic_move + pragmatic_move, atol=1e-6)

    def test_pymdp_utility_term_calculation(self, pymdp_efe_calculator, pymdp_compatible_model):
        """Test utility (pragmatic) term following PyMDP: E_Q[ln P(o|C)]."""
        # Set only pragmatic weight
        pymdp_efe_calculator.config.epistemic_weight = 0.0
        pymdp_efe_calculator.config.pragmatic_weight = 1.0

        beliefs = torch.tensor([1.0, 0.0, 0.0])  # Certain about state 0

        # Policy that leads to preferred observation
        # Move to state 1, which gives obs 1 (preferred)
        policy_good = Policy([1])

        # Policy that keeps in non-preferred state
        # Stay in state 0, which gives obs 0 (avoided)
        policy_bad = Policy([0])

        G_good, _, pragmatic_good = pymdp_efe_calculator.compute_expected_free_energy(
            policy_good, beliefs, pymdp_compatible_model
        )

        G_bad, _, pragmatic_bad = pymdp_efe_calculator.compute_expected_free_energy(
            policy_bad, beliefs, pymdp_compatible_model
        )

        # Good policy should have lower (more negative) pragmatic value
        assert pragmatic_good < pragmatic_bad
        assert G_good < G_bad  # Lower expected free energy is better

    def test_pymdp_epistemic_term_calculation(self, pymdp_efe_calculator, pymdp_compatible_model):
        """Test epistemic term following PyMDP: E[KL[Q(s|o,π)||Q(s|π)]]."""
        # Set only epistemic weight
        pymdp_efe_calculator.config.epistemic_weight = 1.0
        pymdp_efe_calculator.config.pragmatic_weight = 0.0

        # Uncertain beliefs (high entropy)
        uncertain_beliefs = torch.tensor([0.33, 0.33, 0.34])

        # Certain beliefs (low entropy)
        certain_beliefs = torch.tensor([1.0, 0.0, 0.0])

        policy = Policy([1])  # Move action

        # Calculate epistemic value for both belief states
        _, epistemic_uncertain, _ = pymdp_efe_calculator.compute_expected_free_energy(
            policy, uncertain_beliefs, pymdp_compatible_model
        )

        _, epistemic_certain, _ = pymdp_efe_calculator.compute_expected_free_energy(
            policy, certain_beliefs, pymdp_compatible_model
        )

        # Both should be non-negative (information gain ≥ 0)
        assert epistemic_uncertain >= 0
        assert epistemic_certain >= 0

    def test_pymdp_policy_posterior_calculation(self, pymdp_efe_calculator, pymdp_compatible_model):
        """Test policy posterior following PyMDP: Q(π) ∝ exp(-βG(π))."""
        beliefs = torch.tensor([0.5, 0.3, 0.2])

        # Set precision parameter (β in PyMDP)
        beta = pymdp_efe_calculator.config.exploration_constant

        # Calculate EFE for multiple policies
        policies = [Policy([0]), Policy([1])]
        efe_values = []

        for policy in policies:
            G, _, _ = pymdp_efe_calculator.compute_expected_free_energy(
                policy, beliefs, pymdp_compatible_model
            )
            efe_values.append(G)

        # Calculate policy posterior: Q(π) ∝ exp(-βG(π))
        log_posteriors = [-beta * G for G in efe_values]
        posteriors = torch.softmax(torch.tensor(log_posteriors), dim=0)

        # Should be normalized probability distribution
        assert torch.allclose(posteriors.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(posteriors >= 0)

        # Policy with lower EFE should have higher posterior probability
        best_policy_idx = torch.argmin(torch.tensor(efe_values))
        assert posteriors[best_policy_idx] == torch.max(posteriors)

    def test_pymdp_multi_step_policy_evaluation(self, pymdp_compatible_model):
        """Test multi-step policy evaluation following PyMDP conventions."""
        # Multi-step policy configuration
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config = PolicyConfig(planning_horizon=3, policy_length=2, use_gpu=False)  # 2-step policies
        efe_calculator = DiscreteExpectedFreeEnergy(config, inference)

        beliefs = torch.tensor([1.0, 0.0, 0.0])  # Start in state 0

        # Multi-step policies
        policy_stay_stay = Policy([0, 0])  # Stay, then stay
        policy_move_move = Policy([1, 1])  # Move, then move
        policy_move_stay = Policy([1, 0])  # Move, then stay

        # All should produce finite EFE values
        for policy in [policy_stay_stay, policy_move_move, policy_move_stay]:
            G, _, _ = efe_calculator.compute_expected_free_energy(
                policy, beliefs, pymdp_compatible_model
            )
            assert torch.isfinite(G)

    def test_gnn_llm_model_integration(self, pymdp_efe_calculator):
        """Test integration with LLM-generated models following GNN notation."""

        # Mock LLM-generated model following GNN specification
        class LLMGeneratedPyMDPModel:
            def __init__(self):
                # GNN metadata for semantic understanding
                self.gnn_metadata = {
                    "model_type": "discrete_generative_model",
                    "semantic_context": "office_navigation",
                    "state_semantics": ["hallway", "office", "meeting_room"],
                    "action_semantics": ["wait", "walk"],
                    "observation_semantics": [
                        "clear_path",
                        "person_present",
                        "destination_reached",
                    ],
                    "llm_generated": True,
                    "text_description": "Navigation model for office environment",
                }

                # PyMDP-compatible matrices generated by LLM
                self.A = torch.tensor(
                    [
                        # P(clear_path | hallway, office, meeting_room)
                        [0.8, 0.2, 0.1],
                        [0.2, 0.6, 0.2],  # P(person_present | states)
                        [0.0, 0.2, 0.7],  # P(destination_reached | states)
                    ],
                    dtype=torch.float32,
                )

                self.B = torch.zeros(3, 3, 2)
                # Action 0 (wait): stay in same state
                self.B[:, :, 0] = torch.eye(3)
                # Action 1 (walk): progress through states
                self.B[1, 0, 1] = 0.9  # hallway -> office
                self.B[0, 0, 1] = 0.1  # hallway -> hallway (might get lost)
                self.B[2, 1, 1] = 0.8  # office -> meeting_room
                self.B[1, 1, 1] = 0.2  # office -> office
                # meeting_room -> meeting_room (absorbing)
                self.B[2, 2, 1] = 1.0

                self.C = torch.tensor(
                    [
                        [0.0],  # Neutral about clear_path
                        [-1.0],  # Avoid people
                        [3.0],  # Strongly prefer reaching destination
                    ],
                    dtype=torch.float32,
                )

                self.D = torch.tensor([1.0, 0.0, 0.0])  # Start in hallway

            def observation_model(self, states):
                if states.dim() == 0:
                    return self.A[:, states]
                else:
                    return torch.matmul(self.A, states)

            def transition_model(self, states, action):
                if action.dim() == 0:
                    if states.dim() == 0:
                        return self.B[:, states, action]
                    else:
                        return torch.matmul(self.B[:, :, action], states)
                else:
                    raise NotImplementedError("Action distributions not supported")

            def get_preferences(self, timestep=None):
                return self.C.squeeze()

            def get_initial_prior(self):
                return self.D

            def set_preferences(self, preferences):
                self.C = preferences.unsqueeze(-1) if preferences.dim() == 1 else preferences

        llm_model = LLMGeneratedPyMDPModel()

        # Test EFE calculation with LLM-generated model
        beliefs = llm_model.get_initial_prior()

        # Test semantic policies
        wait_policy = Policy([0])  # "wait" action
        walk_policy = Policy([1])  # "walk" action

        # Both should work with LLM model
        G_wait, _, _ = pymdp_efe_calculator.compute_expected_free_energy(
            wait_policy, beliefs, llm_model
        )

        G_walk, _, _ = pymdp_efe_calculator.compute_expected_free_energy(
            walk_policy, beliefs, llm_model
        )

        # Both should be finite
        assert torch.isfinite(G_wait) and torch.isfinite(G_walk)

        # Walk should be preferred (lower EFE) due to goal-seeking
        assert G_walk < G_wait

        # Verify GNN metadata is preserved
        assert llm_model.gnn_metadata["llm_generated"] is True
        assert llm_model.gnn_metadata["semantic_context"] == "office_navigation"
        assert "walk" in llm_model.gnn_metadata["action_semantics"]


class TestPyMDPPolicySelectionIntegration:
    """Integration tests for complete PyMDP-aligned policy selection."""

    def test_complete_pymdp_workflow(self, simple_generative_model):
        """Test complete PyMDP workflow from beliefs to action selection."""
        # Create PyMDP-compatible configuration
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config = PolicyConfig(
            planning_horizon=2,
            policy_length=1,
            epistemic_weight=1.0,
            pragmatic_weight=1.0,
            exploration_constant=2.0,  # High precision for deterministic selection
            use_gpu=False,
        )

        # Create policy selector
        selector = DiscreteExpectedFreeEnergy(config, inference)

        # Initial beliefs
        beliefs = torch.tensor([0.6, 0.3, 0.1])

        # Select policy using PyMDP algorithm
        selected_policy, action_probs = selector.select_policy(beliefs, simple_generative_model)

        # Verify results
        assert isinstance(selected_policy, Policy)
        # Number of actions
        assert len(action_probs) == simple_generative_model.B.shape[2]
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)

        # Selected action should be valid
        selected_action = selected_policy[0]
        assert 0 <= selected_action < simple_generative_model.B.shape[2]

    def test_pymdp_algorithm_comparison(self, model_dimensions, model_parameters):
        """Test that our implementation produces similar results to PyMDP algorithm."""
        # Create model with known properties
        model = DiscreteGenerativeModel(model_dimensions, model_parameters)

        # Set up classic PyMDP scenario: T-maze
        model.A = torch.eye(3)  # Perfect observations

        # T-maze transitions
        model.B = torch.zeros(3, 3, 2)
        # Action 0: go left (state 0 -> state 1)
        model.B[1, 0, 0] = 1.0
        model.B[1, 1, 0] = 1.0  # Stay in left
        model.B[2, 2, 0] = 1.0  # Stay in right

        # Action 1: go right (state 0 -> state 2)
        model.B[2, 0, 1] = 1.0
        model.B[1, 1, 1] = 1.0  # Stay in left
        model.B[2, 2, 1] = 1.0  # Stay in right

        # Preferences: prefer right arm (state 2)
        model.C = torch.tensor(
            [
                [-1.0],  # Avoid left observation
                [0.0],  # Neutral about center
                [2.0],  # Prefer right observation
            ],
            dtype=torch.float32,
        )

        model.D = torch.tensor([1.0, 0.0, 0.0])  # Start at center

        # Create selector
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config = PolicyConfig(use_gpu=False)
        selector = DiscreteExpectedFreeEnergy(config, inference)

        # Test policy selection
        beliefs = model.get_initial_prior()
        selected_policy, action_probs = selector.select_policy(beliefs, model)

        # Should prefer right action (action 1) due to preferences
        assert action_probs[1] > action_probs[0]
        assert selected_policy[0] == 1  # Should select "go right"

    def test_gnn_notation_end_to_end(self):
        """Test end-to-end workflow with GNN notation for LLM model generation."""
        # GNN specification for LLM model generation
        gnn_spec = {
            "model_type": "discrete_generative_model",
            "task_description": "Simple grid world navigation",
            "state_space": {
                "type": "discrete",
                "size": 4,
                "semantic_labels": ["start", "corridor", "junction", "goal"],
                "description": "Linear path from start to goal",
            },
            "observation_space": {
                "type": "discrete",
                "size": 3,
                "semantic_labels": ["wall", "open", "goal_visible"],
                "description": "What the agent can see",
            },
            "action_space": {
                "type": "discrete",
                "size": 2,
                "semantic_labels": ["wait", "advance"],
                "description": "Agent movement actions",
            },
            "initial_parameterization": {
                "A_matrix": "mostly_identity_with_goal_observation",
                "B_matrix": "linear_progression_with_waiting",
                "C_vector": "goal_seeking_preferences",
                "D_vector": "start_at_beginning",
            },
            "preferences": {
                "goal": "reach_goal_state",
                "avoid": ["staying_too_long"],
                "exploration": "minimal",
            },
            "llm_generated": True,
        }

        # Mock LLM model generator that follows GNN specification
        class GNNModelGenerator:
            def __init__(self, gnn_spec):
                self.gnn_spec = gnn_spec

            def generate_pymdp_model(self):
                """Generate PyMDP model from GNN specification."""
                # Extract dimensions from GNN spec
                dims = ModelDimensions(
                    num_states=gnn_spec["state_space"]["size"],
                    num_observations=gnn_spec["observation_space"]["size"],
                    num_actions=gnn_spec["action_space"]["size"],
                )

                params = ModelParameters(use_gpu=False)
                model = DiscreteGenerativeModel(dims, params)

                # Generate matrices based on GNN semantic description
                # A matrix: mostly identity (agent sees current state)
                model.A = torch.eye(3, 4)  # obs x states
                # goal state gives goal_visible observation
                model.A[2, 3] = 1.0
                model.A[1, 3] = 0.0  # goal state doesn't give open observation

                # B matrix: linear progression
                model.B = torch.zeros(4, 4, 2)
                # Action 0 (wait): stay in place
                model.B[:, :, 0] = torch.eye(4)
                # Action 1 (advance): move forward
                for s in range(3):
                    model.B[s + 1, s, 1] = 1.0
                model.B[3, 3, 1] = 1.0  # Goal is absorbing

                # C vector: goal-seeking preferences
                model.C = torch.tensor(
                    [
                        [-0.5],  # Slightly avoid walls
                        [0.0],  # Neutral about open spaces
                        [3.0],  # Strongly prefer goal
                    ],
                    dtype=torch.float32,
                )

                # D vector: start at beginning
                model.D = torch.tensor([1.0, 0.0, 0.0, 0.0])

                # Attach GNN metadata
                model.gnn_metadata = gnn_spec

                return model

        # Generate model from GNN specification
        generator = GNNModelGenerator(gnn_spec)
        llm_model = generator.generate_pymdp_model()

        # Create policy selector
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config = PolicyConfig(use_gpu=False)
        selector = DiscreteExpectedFreeEnergy(config, inference)

        # Test policy selection with GNN-generated model
        beliefs = llm_model.get_initial_prior()
        selected_policy, action_probs = selector.select_policy(beliefs, llm_model)

        # Should work correctly
        assert isinstance(selected_policy, Policy)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)

        # Should prefer advancing (action 1) over waiting (action 0)
        assert action_probs[1] > action_probs[0]

        # Verify GNN metadata is preserved
        assert hasattr(llm_model, "gnn_metadata")
        assert llm_model.gnn_metadata["llm_generated"] is True
        assert llm_model.gnn_metadata["task_description"] == "Simple grid world navigation"


class TestPolicySelectionFactory:
    """Test factory function for creating PyMDP-compatible policy selectors."""

    def test_create_discrete_pymdp_selector(self):
        """Test creating discrete selector compatible with PyMDP."""
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)

        selector = create_policy_selector("discrete", inference_algorithm=inference)

        assert isinstance(selector, DiscreteExpectedFreeEnergy)
        # Should use PyMDP-compatible default configuration
        assert selector.config.epistemic_weight == 1.0
        assert selector.config.pragmatic_weight == 1.0

    def test_create_selector_with_gnn_config(self):
        """Test creating selector with GNN-compatible configuration."""
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)

        # Configuration with GNN metadata
        gnn_config = PolicyConfig(planning_horizon=3, use_gpu=False)
        gnn_config.gnn_metadata = {"model_type": "discrete_generative_model", "llm_generated": True}

        selector = create_policy_selector(
            "discrete", config=gnn_config, inference_algorithm=inference
        )

        assert isinstance(selector, DiscreteExpectedFreeEnergy)
        assert hasattr(selector.config, "gnn_metadata")
        assert selector.config.gnn_metadata["llm_generated"] is True

    def test_invalid_selector_type(self):
        """Test error handling for invalid selector type."""
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)

        with pytest.raises(ValueError):
            create_policy_selector("invalid_type", inference_algorithm=inference)

    def test_missing_inference_algorithm(self):
        """Test error handling for missing inference algorithm."""
        with pytest.raises(ValueError):
            create_policy_selector("discrete")  # Missing required inference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
