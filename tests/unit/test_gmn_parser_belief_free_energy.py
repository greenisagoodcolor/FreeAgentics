"""Comprehensive failing tests for GMN parser belief specification with free energy support.

This test suite defines expected behavior for parsing belief specifications with free energy:
- Complex belief structures with precision parameters
- Free energy calculations and optimization
- Variational inference mechanisms
- Belief propagation algorithms
- Hierarchical belief organization
- Dynamic belief updating with free energy minimization

Following strict TDD: These tests MUST fail initially and drive implementation.
NO graceful fallbacks or try-except blocks allowed.
"""

from inference.active.gmn_parser import (
    GMNParser,
    GMNSchemaValidator,
)


class TestBeliefSpecificationParsing:
    """Test parsing of complex belief specifications with free energy components."""

    def test_parse_variational_belief_with_free_energy(self):
        """Test parsing variational beliefs with free energy minimization."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "variational_belief",
                    "type": "belief",
                    "belief_type": "variational",
                    "about": "world_state",
                    "variational_parameters": {
                        "distribution_family": "mean_field_gaussian",
                        "natural_parameters": {
                            "eta_1": "natural_parameter_1",
                            "eta_2": "natural_parameter_2",
                        },
                        "sufficient_statistics": {"t_1": "first_moment", "t_2": "second_moment"},
                    },
                    "free_energy": {
                        "formulation": "variational_free_energy",
                        "components": {
                            "accuracy": {
                                "formula": "E_q[log p(y|x)]",
                                "description": "expected_log_likelihood",
                            },
                            "complexity": {
                                "formula": "KL(q(x) || p(x))",
                                "description": "kl_divergence_from_prior",
                            },
                        },
                        "optimization": {
                            "method": "natural_gradient_descent",
                            "learning_rate": 0.01,
                            "convergence_criterion": {
                                "type": "free_energy_change",
                                "threshold": 1e-6,
                                "patience": 10,
                            },
                        },
                    },
                    "precision_parameters": {
                        "likelihood_precision": {
                            "value": 2.0,
                            "learnable": True,
                            "prior": {"distribution": "gamma", "shape": 2.0, "rate": 1.0},
                        },
                        "prior_precision": {"value": 1.0, "learnable": False, "fixed": True},
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify variational belief structure
        belief = result["nodes"][0]
        assert belief["belief_type"] == "variational"
        assert belief["about"] == "world_state"

        # Variational parameters
        var_params = belief["variational_parameters"]
        assert var_params["distribution_family"] == "mean_field_gaussian"

        natural_params = var_params["natural_parameters"]
        assert "eta_1" in natural_params
        assert "eta_2" in natural_params

        sufficient_stats = var_params["sufficient_statistics"]
        assert sufficient_stats["t_1"] == "first_moment"
        assert sufficient_stats["t_2"] == "second_moment"

        # Free energy formulation
        free_energy = belief["free_energy"]
        assert free_energy["formulation"] == "variational_free_energy"

        components = free_energy["components"]
        assert components["accuracy"]["formula"] == "E_q[log p(y|x)]"
        assert components["complexity"]["formula"] == "KL(q(x) || p(x))"

        # Optimization
        optimization = free_energy["optimization"]
        assert optimization["method"] == "natural_gradient_descent"
        assert optimization["learning_rate"] == 0.01

        convergence = optimization["convergence_criterion"]
        assert convergence["type"] == "free_energy_change"
        assert convergence["threshold"] == 1e-6

        # Precision parameters
        precision_params = belief["precision_parameters"]
        likelihood_prec = precision_params["likelihood_precision"]
        assert likelihood_prec["value"] == 2.0
        assert likelihood_prec["learnable"] is True

        prior_info = likelihood_prec["prior"]
        assert prior_info["distribution"] == "gamma"
        assert prior_info["shape"] == 2.0

    def test_parse_hierarchical_belief_with_message_passing(self):
        """Test parsing hierarchical beliefs with message passing algorithms."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "hierarchical_belief_system",
                    "type": "belief",
                    "belief_type": "hierarchical_variational",
                    "hierarchy": {
                        "levels": [
                            {
                                "level": 1,
                                "name": "sensory_level",
                                "belief_dimensions": 16,
                                "temporal_depth": 1,
                                "update_frequency": "high",
                            },
                            {
                                "level": 2,
                                "name": "perceptual_level",
                                "belief_dimensions": 8,
                                "temporal_depth": 3,
                                "update_frequency": "medium",
                            },
                            {
                                "level": 3,
                                "name": "conceptual_level",
                                "belief_dimensions": 4,
                                "temporal_depth": 10,
                                "update_frequency": "low",
                            },
                        ],
                        "message_passing": {
                            "algorithm": "belief_propagation",
                            "message_types": {
                                "bottom_up": {
                                    "type": "prediction_error",
                                    "precision_weighting": True,
                                    "formula": "epsilon = y - g(mu)",
                                },
                                "top_down": {
                                    "type": "prediction",
                                    "precision_weighting": True,
                                    "formula": "mu_pred = g(mu_higher)",
                                },
                            },
                            "synchronization": {
                                "method": "asynchronous",
                                "update_order": "bottom_up_then_top_down",
                                "iteration_limit": 10,
                            },
                        },
                    },
                    "precision_dynamics": {
                        "precision_evolution": {
                            "method": "prediction_error_driven",
                            "adaptation_rate": 0.1,
                            "homeostatic_regulation": True,
                            "metaplasticity": {
                                "enabled": True,
                                "time_constant": 100,
                                "stability_bias": 0.8,
                            },
                        },
                        "cross_level_precision": {
                            "precision_propagation": "bidirectional",
                            "precision_coupling_strength": 0.3,
                            "precision_balance": "dynamic",
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify hierarchical belief system
        belief_system = result["nodes"][0]
        assert belief_system["belief_type"] == "hierarchical_variational"

        # Hierarchy structure
        hierarchy = belief_system["hierarchy"]
        levels = hierarchy["levels"]
        assert len(levels) == 3

        # Level 1 (sensory)
        sensory = levels[0]
        assert sensory["name"] == "sensory_level"
        assert sensory["belief_dimensions"] == 16
        assert sensory["temporal_depth"] == 1
        assert sensory["update_frequency"] == "high"

        # Level 3 (conceptual)
        conceptual = levels[2]
        assert conceptual["name"] == "conceptual_level"
        assert conceptual["temporal_depth"] == 10
        assert conceptual["update_frequency"] == "low"

        # Message passing
        message_passing = hierarchy["message_passing"]
        assert message_passing["algorithm"] == "belief_propagation"

        message_types = message_passing["message_types"]
        bottom_up = message_types["bottom_up"]
        assert bottom_up["type"] == "prediction_error"
        assert bottom_up["formula"] == "epsilon = y - g(mu)"

        top_down = message_types["top_down"]
        assert top_down["type"] == "prediction"
        assert top_down["precision_weighting"] is True

        # Synchronization
        sync = message_passing["synchronization"]
        assert sync["method"] == "asynchronous"
        assert sync["update_order"] == "bottom_up_then_top_down"

        # Precision dynamics
        precision_dynamics = belief_system["precision_dynamics"]
        precision_evolution = precision_dynamics["precision_evolution"]
        assert precision_evolution["method"] == "prediction_error_driven"
        assert precision_evolution["homeostatic_regulation"] is True

        metaplasticity = precision_evolution["metaplasticity"]
        assert metaplasticity["enabled"] is True
        assert metaplasticity["time_constant"] == 100

    def test_parse_active_inference_belief_dynamics(self):
        """Test parsing active inference belief dynamics with action-perception loops."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "active_inference_beliefs",
                    "type": "belief",
                    "belief_type": "active_inference",
                    "generative_model": {
                        "model_structure": "partially_observable_markov_decision_process",
                        "components": {
                            "likelihood_model": {
                                "matrix_name": "A",
                                "dimensions": {"observations": 16, "states": 8},
                                "learning": {
                                    "enabled": True,
                                    "learning_rate": 0.1,
                                    "prior_strength": 1.0,
                                },
                            },
                            "transition_model": {
                                "matrix_name": "B",
                                "dimensions": {"states": 8, "actions": 4},
                                "learning": {
                                    "enabled": True,
                                    "learning_rate": 0.05,
                                    "prior_strength": 2.0,
                                },
                            },
                            "preference_model": {
                                "vector_name": "C",
                                "dimensions": {"observations": 16},
                                "preferences": {
                                    "type": "goal_directed",
                                    "target_observations": [12, 13, 14, 15],
                                    "preference_strength": 2.0,
                                },
                            },
                            "prior_beliefs": {
                                "vector_name": "D",
                                "dimensions": {"states": 8},
                                "initial_distribution": "uniform",
                                "concentration": 1.0,
                            },
                        },
                    },
                    "inference_dynamics": {
                        "state_estimation": {
                            "method": "variational_message_passing",
                            "iterations": 16,
                            "convergence_threshold": 1e-4,
                            "belief_update_rule": "predictive_coding",
                        },
                        "policy_inference": {
                            "method": "expected_free_energy_minimization",
                            "planning_horizon": 5,
                            "policy_precision": 16.0,
                            "policy_prior": "uniform",
                        },
                        "action_selection": {
                            "method": "probabilistic_sampling",
                            "temperature": 1.0,
                            "exploration_bonus": 0.1,
                        },
                    },
                    "learning_dynamics": {
                        "model_learning": {
                            "method": "bayesian_model_averaging",
                            "update_schedule": "online",
                            "forgetting_factor": 0.99,
                        },
                        "precision_learning": {
                            "method": "empirical_bayes",
                            "precision_update_rate": 0.01,
                            "precision_bounds": [0.1, 10.0],
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify active inference beliefs
        ai_beliefs = result["nodes"][0]
        assert ai_beliefs["belief_type"] == "active_inference"

        # Generative model
        gen_model = ai_beliefs["generative_model"]
        assert gen_model["model_structure"] == "partially_observable_markov_decision_process"

        components = gen_model["components"]

        # Likelihood model (A matrix)
        likelihood = components["likelihood_model"]
        assert likelihood["matrix_name"] == "A"
        assert likelihood["dimensions"]["observations"] == 16
        assert likelihood["dimensions"]["states"] == 8

        likelihood_learning = likelihood["learning"]
        assert likelihood_learning["enabled"] is True
        assert likelihood_learning["learning_rate"] == 0.1

        # Transition model (B matrix)
        transition = components["transition_model"]
        assert transition["matrix_name"] == "B"
        assert transition["dimensions"]["actions"] == 4

        # Preference model (C vector)
        preferences = components["preference_model"]
        assert preferences["vector_name"] == "C"
        pref_details = preferences["preferences"]
        assert pref_details["type"] == "goal_directed"
        assert pref_details["target_observations"] == [12, 13, 14, 15]

        # Prior beliefs (D vector)
        priors = components["prior_beliefs"]
        assert priors["vector_name"] == "D"
        assert priors["initial_distribution"] == "uniform"

        # Inference dynamics
        inference = ai_beliefs["inference_dynamics"]

        # State estimation
        state_est = inference["state_estimation"]
        assert state_est["method"] == "variational_message_passing"
        assert state_est["iterations"] == 16
        assert state_est["belief_update_rule"] == "predictive_coding"

        # Policy inference
        policy_inf = inference["policy_inference"]
        assert policy_inf["method"] == "expected_free_energy_minimization"
        assert policy_inf["planning_horizon"] == 5
        assert policy_inf["policy_precision"] == 16.0

        # Action selection
        action_sel = inference["action_selection"]
        assert action_sel["method"] == "probabilistic_sampling"
        assert action_sel["exploration_bonus"] == 0.1

        # Learning dynamics
        learning = ai_beliefs["learning_dynamics"]

        # Model learning
        model_learning = learning["model_learning"]
        assert model_learning["method"] == "bayesian_model_averaging"
        assert model_learning["update_schedule"] == "online"

        # Precision learning
        precision_learning = learning["precision_learning"]
        assert precision_learning["method"] == "empirical_bayes"
        assert precision_learning["precision_bounds"] == [0.1, 10.0]


class TestFreeEnergyCalculationIntegration:
    """Test integration of free energy calculations with belief specifications."""

    def test_parse_expected_free_energy_with_belief_integration(self):
        """Test parsing expected free energy calculations integrated with beliefs."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "efe_belief_integration",
                    "type": "belief",
                    "belief_type": "action_oriented",
                    "expected_free_energy": {
                        "calculation_method": "path_integral",
                        "temporal_horizon": 5,
                        "components": {
                            "epistemic_value": {
                                "formula": "-sum_tau sum_s q(s_tau|pi) * H[p(o_tau|s_tau)]",
                                "description": "information_gain_over_time",
                                "weighting": {
                                    "epistemic_bonus": 1.0,
                                    "temporal_discount": 0.95,
                                    "uncertainty_threshold": 0.5,
                                },
                            },
                            "pragmatic_value": {
                                "formula": "sum_tau sum_o q(o_tau|pi) * log(C(o_tau))",
                                "description": "expected_reward_over_time",
                                "weighting": {
                                    "reward_scaling": 1.0,
                                    "temporal_discount": 0.95,
                                    "satiation_factor": 0.9,
                                },
                            },
                            "novelty_bonus": {
                                "formula": "-sum_tau KL(q(s_tau|pi) || q(s_tau))",
                                "description": "exploration_bonus_for_novel_states",
                                "weighting": {"novelty_strength": 0.2, "habituation_rate": 0.01},
                            },
                        },
                        "policy_evaluation": {
                            "method": "monte_carlo_tree_search",
                            "num_samples": 1000,
                            "tree_depth": 5,
                            "ucb_constant": 1.414,
                            "expansion_threshold": 10,
                        },
                    },
                    "belief_state_integration": {
                        "belief_propagation": {
                            "forward_messages": {
                                "alpha_recursion": "alpha_t+1(s) = sum_a pi(a) sum_s' alpha_t(s') B(s|s',a) A(o_t+1|s)",
                                "message_normalization": True,
                            },
                            "backward_messages": {
                                "beta_recursion": "beta_t(s) = sum_a pi(a) sum_s' B(s'|s,a) A(o_t+1|s') beta_t+1(s')",
                                "message_normalization": True,
                            },
                        },
                        "posterior_computation": {
                            "formula": "gamma_t(s) = alpha_t(s) * beta_t(s)",
                            "normalization": "sum_normalization",
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify action-oriented belief
        belief = result["nodes"][0]
        assert belief["belief_type"] == "action_oriented"

        # Expected free energy
        efe = belief["expected_free_energy"]
        assert efe["calculation_method"] == "path_integral"
        assert efe["temporal_horizon"] == 5

        components = efe["components"]

        # Epistemic value
        epistemic = components["epistemic_value"]
        assert "H[p(o_tau|s_tau)]" in epistemic["formula"]
        assert epistemic["description"] == "information_gain_over_time"

        epistemic_weighting = epistemic["weighting"]
        assert epistemic_weighting["epistemic_bonus"] == 1.0
        assert epistemic_weighting["temporal_discount"] == 0.95

        # Pragmatic value
        pragmatic = components["pragmatic_value"]
        assert "log(C(o_tau))" in pragmatic["formula"]

        # Novelty bonus
        novelty = components["novelty_bonus"]
        assert "KL(q(s_tau|pi) || q(s_tau))" in novelty["formula"]
        novelty_weighting = novelty["weighting"]
        assert novelty_weighting["novelty_strength"] == 0.2

        # Policy evaluation
        policy_eval = efe["policy_evaluation"]
        assert policy_eval["method"] == "monte_carlo_tree_search"
        assert policy_eval["num_samples"] == 1000
        assert policy_eval["ucb_constant"] == 1.414

        # Belief state integration
        integration = belief["belief_state_integration"]
        belief_prop = integration["belief_propagation"]

        # Forward messages
        forward = belief_prop["forward_messages"]
        assert "alpha_recursion" in forward
        assert "alpha_t+1(s)" in forward["alpha_recursion"]
        assert forward["message_normalization"] is True

        # Backward messages
        backward = belief_prop["backward_messages"]
        assert "beta_recursion" in backward
        assert "beta_t(s)" in backward["beta_recursion"]

        # Posterior computation
        posterior = integration["posterior_computation"]
        assert posterior["formula"] == "gamma_t(s) = alpha_t(s) * beta_t(s)"
        assert posterior["normalization"] == "sum_normalization"


class TestBeliefValidation:
    """Test validation of belief specifications with free energy constraints."""

    def test_validate_free_energy_mathematical_consistency(self):
        """Test validation of mathematical consistency in free energy formulations."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "inconsistent_free_energy",
                    "type": "belief",
                    "free_energy": {
                        "components": {
                            "accuracy": {
                                "formula": "E_q[log p(y|x)]",
                                "variables": {
                                    "q": "posterior_distribution",
                                    "p": "likelihood_function",
                                },
                            },
                            "complexity": {
                                "formula": "KL(q(z) || p(z))",  # Different variable names
                                "variables": {
                                    "q": "posterior_distribution",
                                    "p": "prior_distribution",  # Inconsistent with accuracy
                                },
                            },
                        }
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect inconsistent variable definitions
        assert is_valid is False
        consistency_errors = [
            e for e in errors if "consistency" in e.lower() or "variable" in e.lower()
        ]
        assert len(consistency_errors) > 0

    def test_validate_precision_parameter_constraints(self):
        """Test validation of precision parameter constraints."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "invalid_precision",
                    "type": "belief",
                    "precision_parameters": {
                        "likelihood_precision": {
                            "value": -2.0,  # Invalid: precision must be positive
                            "learnable": True,
                        },
                        "prior_precision": {
                            "value": 0.0,  # Invalid: precision cannot be zero
                            "learnable": False,
                        },
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect invalid precision values
        assert is_valid is False
        precision_errors = [
            e
            for e in errors
            if "precision" in e.lower() and ("positive" in e.lower() or "zero" in e.lower())
        ]
        assert len(precision_errors) > 0

    def test_validate_hierarchical_belief_level_consistency(self):
        """Test validation of hierarchical belief level consistency."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "inconsistent_hierarchy",
                    "type": "belief",
                    "belief_type": "hierarchical_variational",
                    "hierarchy": {
                        "levels": [
                            {"level": 1, "belief_dimensions": 16, "temporal_depth": 1},
                            {
                                "level": 3,  # Missing level 2
                                "belief_dimensions": 4,
                                "temporal_depth": 5,
                            },
                        ]
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect missing hierarchy level
        assert is_valid is False
        hierarchy_errors = [e for e in errors if "hierarchy" in e.lower() or "level" in e.lower()]
        assert len(hierarchy_errors) > 0
