"""Comprehensive failing tests for GMN parser state transitions support.

This test suite defines expected behavior for parsing state transition definitions:
- Complex transition matrices with constraints
- Action-dependent state transitions
- Temporal dynamics and time-dependent transitions
- Stochastic transition processes
- Hierarchical and multi-scale transitions
- Transition learning and adaptation

Following strict TDD: These tests MUST fail initially and drive implementation.
NO graceful fallbacks or try-except blocks allowed.
"""

from inference.active.gmn_parser import (
    GMNParser,
    GMNSchemaValidator,
)


class TestActionDependentTransitions:
    """Test parsing of action-dependent state transition definitions."""

    def test_parse_discrete_action_transitions(self):
        """Test parsing discrete action-dependent transition matrices."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "agent_location",
                    "type": "state",
                    "num_states": 9,
                    "state_space": {
                        "type": "discrete",
                        "structure": "grid_3x3",
                        "coordinates": {
                            "state_0": [0, 0],
                            "state_1": [0, 1],
                            "state_2": [0, 2],
                            "state_3": [1, 0],
                            "state_4": [1, 1],
                            "state_5": [1, 2],
                            "state_6": [2, 0],
                            "state_7": [2, 1],
                            "state_8": [2, 2],
                        },
                    },
                },
                {
                    "name": "movement_actions",
                    "type": "action",
                    "num_actions": 5,
                    "action_space": {
                        "type": "discrete",
                        "actions": ["up", "down", "left", "right", "stay"],
                        "action_effects": {
                            "up": {"coordinate_change": [0, -1]},
                            "down": {"coordinate_change": [0, 1]},
                            "left": {"coordinate_change": [-1, 0]},
                            "right": {"coordinate_change": [1, 0]},
                            "stay": {"coordinate_change": [0, 0]},
                        },
                    },
                },
                {
                    "name": "spatial_transitions",
                    "type": "transition",
                    "transition_type": "action_dependent",
                    "transition_matrices": {
                        "action_0": {  # up
                            "matrix_type": "sparse",
                            "deterministic_rules": [
                                {"from_state": 3, "to_state": 0, "probability": 0.9},
                                {"from_state": 4, "to_state": 1, "probability": 0.9},
                                {"from_state": 5, "to_state": 2, "probability": 0.9},
                                {"from_state": 6, "to_state": 3, "probability": 0.9},
                                {"from_state": 7, "to_state": 4, "probability": 0.9},
                                {"from_state": 8, "to_state": 5, "probability": 0.9},
                            ],
                            "boundary_behavior": "stay_in_place",
                            "noise_model": {
                                "type": "uniform",
                                "noise_probability": 0.1,
                                "distribution": "uniform_over_valid_transitions",
                            },
                        },
                        "action_4": {  # stay
                            "matrix_type": "diagonal",
                            "self_transition_probability": 0.95,
                            "noise_probability": 0.05,
                        },
                    },
                    "transition_constraints": {
                        "stochastic": True,
                        "column_normalized": True,
                        "non_negative": True,
                        "boundary_constraints": "reflective",
                    },
                },
            ],
            "edges": [
                {"from": "agent_location", "to": "spatial_transitions", "type": "depends_on"},
                {"from": "movement_actions", "to": "spatial_transitions", "type": "depends_on"},
            ],
        }

        result = parser.parse(spec)

        # Verify state space structure
        location_state = result["nodes"][0]
        assert location_state["state_space"]["structure"] == "grid_3x3"
        assert len(location_state["state_space"]["coordinates"]) == 9

        # Verify action space
        actions = result["nodes"][1]
        action_effects = actions["action_space"]["action_effects"]
        assert action_effects["up"]["coordinate_change"] == [0, -1]
        assert action_effects["stay"]["coordinate_change"] == [0, 0]

        # Verify transition structure
        transitions = result["nodes"][2]
        assert transitions["transition_type"] == "action_dependent"

        # Action 0 (up) transitions
        action_0_matrix = transitions["transition_matrices"]["action_0"]
        assert action_0_matrix["matrix_type"] == "sparse"
        rules = action_0_matrix["deterministic_rules"]
        assert len(rules) == 6  # Non-boundary upward transitions

        # Boundary behavior
        assert action_0_matrix["boundary_behavior"] == "stay_in_place"
        noise = action_0_matrix["noise_model"]
        assert noise["type"] == "uniform"
        assert noise["noise_probability"] == 0.1

        # Action 4 (stay) transitions
        action_4_matrix = transitions["transition_matrices"]["action_4"]
        assert action_4_matrix["matrix_type"] == "diagonal"
        assert action_4_matrix["self_transition_probability"] == 0.95

    def test_parse_continuous_action_transitions(self):
        """Test parsing continuous action-dependent transitions with interpolation."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "robot_pose",
                    "type": "state",
                    "num_states": 16,
                    "state_space": {
                        "type": "continuous_discretized",
                        "dimensions": 2,
                        "bounds": [[0, 10], [0, 10]],
                        "discretization": {"method": "uniform_grid", "resolution": [4, 4]},
                    },
                },
                {
                    "name": "velocity_control",
                    "type": "action",
                    "action_space": {
                        "type": "continuous",
                        "dimensions": 2,
                        "bounds": [[-1, 1], [-1, 1]],
                        "discretization": {
                            "method": "uniform_sampling",
                            "num_actions": 8,
                            "include_bounds": True,
                        },
                    },
                },
                {
                    "name": "dynamics_model",
                    "type": "transition",
                    "transition_type": "continuous_action_dependent",
                    "dynamics": {
                        "model_type": "linear_gaussian",
                        "state_equation": "x_{t+1} = A * x_t + B * u_t + w_t",
                        "parameters": {
                            "A": [[1.0, 0.0], [0.0, 1.0]],  # Identity dynamics
                            "B": [[0.1, 0.0], [0.0, 0.1]],  # Control matrix
                            "process_noise_covariance": [[0.01, 0.0], [0.0, 0.01]],
                        },
                        "discretization": {
                            "method": "gaussian_approximation",
                            "num_sigma_points": 3,
                            "approximation_order": 2,
                        },
                    },
                    "action_interpolation": {
                        "method": "bilinear",
                        "boundary_handling": "extrapolation",
                        "smoothness_constraint": True,
                    },
                },
            ],
            "edges": [
                {"from": "robot_pose", "to": "dynamics_model", "type": "depends_on"},
                {"from": "velocity_control", "to": "dynamics_model", "type": "depends_on"},
            ],
        }

        result = parser.parse(spec)

        # Verify continuous state space
        pose_state = result["nodes"][0]
        state_space = pose_state["state_space"]
        assert state_space["type"] == "continuous_discretized"
        assert state_space["bounds"] == [[0, 10], [0, 10]]

        # Verify continuous action space
        velocity_action = result["nodes"][1]
        action_space = velocity_action["action_space"]
        assert action_space["type"] == "continuous"
        assert action_space["bounds"] == [[-1, 1], [-1, 1]]

        # Verify dynamics model
        dynamics = result["nodes"][2]
        assert dynamics["transition_type"] == "continuous_action_dependent"

        dynamics_params = dynamics["dynamics"]
        assert dynamics_params["model_type"] == "linear_gaussian"
        assert dynamics_params["state_equation"] == "x_{t+1} = A * x_t + B * u_t + w_t"

        # Parameters
        params = dynamics_params["parameters"]
        assert params["A"] == [[1.0, 0.0], [0.0, 1.0]]
        assert params["B"] == [[0.1, 0.0], [0.0, 0.1]]

        # Discretization method
        discretization = dynamics_params["discretization"]
        assert discretization["method"] == "gaussian_approximation"
        assert discretization["num_sigma_points"] == 3

    def test_parse_stochastic_transition_processes(self):
        """Test parsing stochastic transition processes with uncertainty."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "environmental_state",
                    "type": "state",
                    "num_states": 5,
                    "state_labels": ["clear", "light_rain", "heavy_rain", "fog", "storm"],
                },
                {
                    "name": "weather_transitions",
                    "type": "transition",
                    "transition_type": "stochastic_process",
                    "stochastic_model": {
                        "type": "markov_chain",
                        "order": 1,
                        "transition_matrix": [
                            [0.7, 0.2, 0.05, 0.04, 0.01],  # from clear
                            [0.3, 0.4, 0.2, 0.08, 0.02],  # from light_rain
                            [0.1, 0.3, 0.5, 0.05, 0.05],  # from heavy_rain
                            [0.2, 0.1, 0.1, 0.5, 0.1],  # from fog
                            [0.05, 0.1, 0.3, 0.1, 0.45],  # from storm
                        ],
                        "stationary_distribution": [0.35, 0.25, 0.2, 0.1, 0.1],
                        "convergence_properties": {
                            "irreducible": True,
                            "aperiodic": True,
                            "ergodic": True,
                        },
                    },
                    "uncertainty_quantification": {
                        "parameter_uncertainty": {
                            "type": "dirichlet_prior",
                            "concentration_parameters": [
                                [7, 2, 0.5, 0.4, 0.1],
                                [3, 4, 2, 0.8, 0.2],
                                [1, 3, 5, 0.5, 0.5],
                                [2, 1, 1, 5, 1],
                                [0.5, 1, 3, 1, 4.5],
                            ],
                            "confidence_intervals": 0.95,
                        },
                        "model_uncertainty": {
                            "model_selection": [
                                "markov_order_1",
                                "markov_order_2",
                                "hidden_markov",
                            ],
                            "model_weights": [0.6, 0.3, 0.1],
                            "model_averaging": True,
                        },
                    },
                },
            ],
            "edges": [
                {"from": "environmental_state", "to": "weather_transitions", "type": "depends_on"}
            ],
        }

        result = parser.parse(spec)

        # Verify environmental state
        env_state = result["nodes"][0]
        assert len(env_state["state_labels"]) == 5
        assert "storm" in env_state["state_labels"]

        # Verify stochastic model
        weather_trans = result["nodes"][1]
        stochastic_model = weather_trans["stochastic_model"]
        assert stochastic_model["type"] == "markov_chain"
        assert stochastic_model["order"] == 1

        # Transition matrix
        trans_matrix = stochastic_model["transition_matrix"]
        assert len(trans_matrix) == 5
        assert len(trans_matrix[0]) == 5
        # Check row normalization
        for row in trans_matrix:
            assert abs(sum(row) - 1.0) < 1e-10

        # Stationary distribution
        stationary = stochastic_model["stationary_distribution"]
        assert len(stationary) == 5
        assert abs(sum(stationary) - 1.0) < 1e-10

        # Convergence properties
        convergence = stochastic_model["convergence_properties"]
        assert convergence["irreducible"] is True
        assert convergence["ergodic"] is True

        # Uncertainty quantification
        uncertainty = weather_trans["uncertainty_quantification"]
        param_unc = uncertainty["parameter_uncertainty"]
        assert param_unc["type"] == "dirichlet_prior"
        assert param_unc["confidence_intervals"] == 0.95

        model_unc = uncertainty["model_uncertainty"]
        assert len(model_unc["model_selection"]) == 3
        assert sum(model_unc["model_weights"]) == 1.0


class TestTemporalDynamics:
    """Test parsing of temporal dynamics and time-dependent transitions."""

    def test_parse_time_dependent_transitions(self):
        """Test parsing time-dependent transition matrices."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "seasonal_behavior",
                    "type": "state",
                    "num_states": 4,
                    "state_labels": ["foraging", "nesting", "migrating", "hibernating"],
                },
                {
                    "name": "temporal_transitions",
                    "type": "transition",
                    "transition_type": "time_dependent",
                    "temporal_structure": {
                        "time_discretization": {
                            "type": "periodic",
                            "period": 365,  # days
                            "resolution": "daily",
                            "phase_alignment": "calendar_year",
                        },
                        "transition_schedule": {
                            "spring": {
                                "time_range": [60, 151],  # March-May
                                "transition_matrix": [
                                    [0.6, 0.3, 0.05, 0.05],
                                    [0.2, 0.7, 0.05, 0.05],
                                    [0.1, 0.1, 0.8, 0.0],
                                    [0.4, 0.3, 0.3, 0.0],
                                ],
                            },
                            "summer": {
                                "time_range": [152, 243],  # June-August
                                "transition_matrix": [
                                    [0.8, 0.15, 0.03, 0.02],
                                    [0.1, 0.8, 0.05, 0.05],
                                    [0.05, 0.05, 0.9, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            },
                            "autumn": {
                                "time_range": [244, 334],  # September-November
                                "transition_matrix": [
                                    [0.4, 0.2, 0.3, 0.1],
                                    [0.1, 0.4, 0.4, 0.1],
                                    [0.05, 0.05, 0.8, 0.1],
                                    [0.0, 0.0, 0.2, 0.8],
                                ],
                            },
                            "winter": {
                                "time_range": [335, 59],  # December-February
                                "transition_matrix": [
                                    [0.1, 0.05, 0.05, 0.8],
                                    [0.05, 0.1, 0.05, 0.8],
                                    [0.0, 0.0, 0.1, 0.9],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            },
                        },
                        "interpolation": {
                            "method": "smooth_transition",
                            "transition_duration": 7,  # days
                            "interpolation_function": "sigmoid",
                        },
                    },
                },
            ],
            "edges": [
                {"from": "seasonal_behavior", "to": "temporal_transitions", "type": "depends_on"}
            ],
        }

        result = parser.parse(spec)

        # Verify seasonal behavior states
        behavior_state = result["nodes"][0]
        assert len(behavior_state["state_labels"]) == 4
        assert "hibernating" in behavior_state["state_labels"]

        # Verify temporal transition structure
        temporal_trans = result["nodes"][1]
        temporal_struct = temporal_trans["temporal_structure"]

        # Time discretization
        time_disc = temporal_struct["time_discretization"]
        assert time_disc["type"] == "periodic"
        assert time_disc["period"] == 365
        assert time_disc["resolution"] == "daily"

        # Transition schedule
        schedule = temporal_struct["transition_schedule"]
        assert len(schedule) == 4  # Four seasons

        # Spring transitions
        spring = schedule["spring"]
        assert spring["time_range"] == [60, 151]
        spring_matrix = spring["transition_matrix"]
        assert len(spring_matrix) == 4

        # Winter transitions (wraps around year)
        winter = schedule["winter"]
        assert winter["time_range"] == [335, 59]

        # Interpolation settings
        interpolation = temporal_struct["interpolation"]
        assert interpolation["method"] == "smooth_transition"
        assert interpolation["transition_duration"] == 7
        assert interpolation["interpolation_function"] == "sigmoid"

    def test_parse_multi_timescale_transitions(self):
        """Test parsing multi-timescale transition dynamics."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "hierarchical_behavior",
                    "type": "state",
                    "hierarchical": True,
                    "factors": [
                        {
                            "name": "immediate_action",
                            "num_states": 4,
                            "timescale": "short",
                            "temporal_resolution": 1.0,  # seconds
                        },
                        {
                            "name": "behavioral_mode",
                            "num_states": 3,
                            "timescale": "medium",
                            "temporal_resolution": 60.0,  # minutes
                        },
                        {
                            "name": "life_stage",
                            "num_states": 5,
                            "timescale": "long",
                            "temporal_resolution": 365.0,  # days
                        },
                    ],
                },
                {
                    "name": "multi_scale_transitions",
                    "type": "transition",
                    "transition_type": "multi_timescale",
                    "timescale_structure": {
                        "short_timescale": {
                            "factor": "immediate_action",
                            "transition_rate": "fast",
                            "update_frequency": 1.0,
                            "dynamics": {
                                "type": "reactive",
                                "response_time": 0.1,
                                "adaptation_speed": "high",
                            },
                        },
                        "medium_timescale": {
                            "factor": "behavioral_mode",
                            "transition_rate": "moderate",
                            "update_frequency": 0.017,  # 1/60
                            "dynamics": {
                                "type": "strategic",
                                "planning_horizon": 10,
                                "stability_bias": 0.8,
                            },
                        },
                        "long_timescale": {
                            "factor": "life_stage",
                            "transition_rate": "slow",
                            "update_frequency": 0.0027,  # 1/365
                            "dynamics": {
                                "type": "developmental",
                                "irreversibility": 0.95,
                                "maturation_constraints": True,
                            },
                        },
                    },
                    "cross_timescale_coupling": {
                        "short_to_medium": {
                            "influence_strength": 0.1,
                            "aggregation_method": "temporal_averaging",
                            "threshold_effects": {
                                "activation_threshold": 0.7,
                                "persistence_requirement": 30,  # time units
                            },
                        },
                        "medium_to_long": {
                            "influence_strength": 0.05,
                            "aggregation_method": "cumulative_evidence",
                            "threshold_effects": {
                                "activation_threshold": 0.9,
                                "persistence_requirement": 180,
                            },
                        },
                    },
                },
            ],
            "edges": [
                {
                    "from": "hierarchical_behavior",
                    "to": "multi_scale_transitions",
                    "type": "depends_on",
                }
            ],
        }

        result = parser.parse(spec)

        # Verify hierarchical behavior structure
        behavior_state = result["nodes"][0]
        assert behavior_state["hierarchical"] is True
        factors = behavior_state["factors"]
        assert len(factors) == 3

        # Check factor timescales
        immediate = factors[0]
        assert immediate["timescale"] == "short"
        assert immediate["temporal_resolution"] == 1.0

        behavioral = factors[1]
        assert behavioral["timescale"] == "medium"
        assert behavioral["temporal_resolution"] == 60.0

        life_stage = factors[2]
        assert life_stage["timescale"] == "long"
        assert life_stage["temporal_resolution"] == 365.0

        # Verify multi-scale transitions
        multi_trans = result["nodes"][1]
        timescale_struct = multi_trans["timescale_structure"]

        # Short timescale dynamics
        short = timescale_struct["short_timescale"]
        assert short["factor"] == "immediate_action"
        assert short["transition_rate"] == "fast"
        short_dynamics = short["dynamics"]
        assert short_dynamics["type"] == "reactive"
        assert short_dynamics["response_time"] == 0.1

        # Long timescale dynamics
        long_ts = timescale_struct["long_timescale"]
        long_dynamics = long_ts["dynamics"]
        assert long_dynamics["type"] == "developmental"
        assert long_dynamics["irreversibility"] == 0.95

        # Cross-timescale coupling
        coupling = multi_trans["cross_timescale_coupling"]
        short_to_medium = coupling["short_to_medium"]
        assert short_to_medium["influence_strength"] == 0.1
        assert short_to_medium["aggregation_method"] == "temporal_averaging"

        threshold = short_to_medium["threshold_effects"]
        assert threshold["activation_threshold"] == 0.7
        assert threshold["persistence_requirement"] == 30


class TestHierarchicalTransitions:
    """Test parsing of hierarchical and multi-scale transition structures."""

    def test_parse_nested_transition_hierarchies(self):
        """Test parsing nested transition hierarchies with different scales."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "multi_level_system",
                    "type": "state",
                    "hierarchical": True,
                    "levels": [
                        {
                            "level": 1,
                            "name": "micro_states",
                            "num_states": 16,
                            "spatial_scale": "local",
                            "interaction_range": 1,
                        },
                        {
                            "level": 2,
                            "name": "meso_states",
                            "num_states": 8,
                            "spatial_scale": "regional",
                            "interaction_range": 4,
                        },
                        {
                            "level": 3,
                            "name": "macro_states",
                            "num_states": 4,
                            "spatial_scale": "global",
                            "interaction_range": "infinite",
                        },
                    ],
                },
                {
                    "name": "hierarchical_transitions",
                    "type": "transition",
                    "transition_type": "hierarchical",
                    "hierarchy_structure": {
                        "level_1_transitions": {
                            "type": "local_dynamics",
                            "transition_speed": "fast",
                            "locality_constraint": True,
                            "interaction_matrix": {
                                "type": "nearest_neighbor",
                                "connectivity": "von_neumann",
                                "boundary_conditions": "periodic",
                            },
                        },
                        "level_2_transitions": {
                            "type": "emergent_dynamics",
                            "transition_speed": "medium",
                            "emergence_mechanism": {
                                "aggregation_method": "majority_vote",
                                "threshold": 0.6,
                                "hysteresis": 0.1,
                            },
                        },
                        "level_3_transitions": {
                            "type": "global_dynamics",
                            "transition_speed": "slow",
                            "global_constraints": {
                                "conservation_laws": ["total_mass", "total_energy"],
                                "symmetry_constraints": ["translational", "rotational"],
                            },
                        },
                    },
                    "cross_level_interactions": {
                        "upward_causation": {
                            "level_1_to_2": {
                                "mechanism": "statistical_aggregation",
                                "aggregation_function": "mean_field",
                                "time_averaging_window": 10,
                            },
                            "level_2_to_3": {
                                "mechanism": "pattern_formation",
                                "pattern_detection": "spatial_correlation",
                                "correlation_threshold": 0.8,
                            },
                        },
                        "downward_causation": {
                            "level_3_to_2": {
                                "mechanism": "parameter_modulation",
                                "modulated_parameters": [
                                    "transition_rates",
                                    "interaction_strengths",
                                ],
                                "modulation_strength": 0.3,
                            },
                            "level_2_to_1": {
                                "mechanism": "field_effects",
                                "field_type": "potential_field",
                                "field_strength": 0.2,
                            },
                        },
                    },
                },
            ],
            "edges": [
                {
                    "from": "multi_level_system",
                    "to": "hierarchical_transitions",
                    "type": "depends_on",
                }
            ],
        }

        result = parser.parse(spec)

        # Verify multi-level system
        multi_level = result["nodes"][0]
        assert multi_level["hierarchical"] is True
        levels = multi_level["levels"]
        assert len(levels) == 3

        # Micro level
        micro = levels[0]
        assert micro["spatial_scale"] == "local"
        assert micro["interaction_range"] == 1

        # Macro level
        macro = levels[2]
        assert macro["spatial_scale"] == "global"
        assert macro["interaction_range"] == "infinite"

        # Verify hierarchical transitions
        hier_trans = result["nodes"][1]
        hierarchy = hier_trans["hierarchy_structure"]

        # Level 1 local dynamics
        level_1 = hierarchy["level_1_transitions"]
        assert level_1["type"] == "local_dynamics"
        assert level_1["locality_constraint"] is True
        interaction = level_1["interaction_matrix"]
        assert interaction["type"] == "nearest_neighbor"
        assert interaction["connectivity"] == "von_neumann"

        # Level 2 emergent dynamics
        level_2 = hierarchy["level_2_transitions"]
        assert level_2["type"] == "emergent_dynamics"
        emergence = level_2["emergence_mechanism"]
        assert emergence["aggregation_method"] == "majority_vote"
        assert emergence["threshold"] == 0.6

        # Cross-level interactions
        cross_level = hier_trans["cross_level_interactions"]

        # Upward causation
        upward = cross_level["upward_causation"]
        level_1_to_2 = upward["level_1_to_2"]
        assert level_1_to_2["mechanism"] == "statistical_aggregation"
        assert level_1_to_2["aggregation_function"] == "mean_field"

        # Downward causation
        downward = cross_level["downward_causation"]
        level_3_to_2 = downward["level_3_to_2"]
        assert level_3_to_2["mechanism"] == "parameter_modulation"
        modulated = level_3_to_2["modulated_parameters"]
        assert "transition_rates" in modulated


class TestTransitionLearning:
    """Test parsing of transition learning and adaptation mechanisms."""

    def test_parse_adaptive_transition_learning(self):
        """Test parsing adaptive transition matrix learning."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "learnable_transitions",
                    "type": "transition",
                    "transition_type": "adaptive",
                    "learning_mechanism": {
                        "type": "bayesian_learning",
                        "prior_distribution": {
                            "type": "dirichlet",
                            "concentration_parameters": "uniform",
                            "initial_strength": 1.0,
                        },
                        "update_rule": {
                            "method": "conjugate_update",
                            "update_equation": "alpha_new = alpha_prior + observation_counts",
                            "learning_rate": "adaptive",
                            "forgetting_factor": 0.99,
                        },
                        "uncertainty_estimation": {
                            "method": "posterior_variance",
                            "confidence_intervals": True,
                            "credible_interval_level": 0.95,
                        },
                    },
                    "adaptation_dynamics": {
                        "adaptation_speed": {
                            "initial": "slow",
                            "asymptotic": "fast",
                            "acceleration_function": "exponential",
                            "adaptation_threshold": 100,  # observations
                        },
                        "plasticity_constraints": {
                            "structural_stability": True,
                            "stability_preservation": 0.8,
                            "catastrophic_forgetting_prevention": True,
                        },
                        "meta_learning": {
                            "enabled": True,
                            "meta_parameter_optimization": ["learning_rate", "forgetting_factor"],
                            "optimization_method": "gradient_ascent",
                            "meta_objective": "predictive_accuracy",
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify learnable transitions
        learnable_trans = result["nodes"][0]
        assert learnable_trans["transition_type"] == "adaptive"

        # Learning mechanism
        learning = learnable_trans["learning_mechanism"]
        assert learning["type"] == "bayesian_learning"

        # Prior distribution
        prior = learning["prior_distribution"]
        assert prior["type"] == "dirichlet"
        assert prior["concentration_parameters"] == "uniform"
        assert prior["initial_strength"] == 1.0

        # Update rule
        update_rule = learning["update_rule"]
        assert update_rule["method"] == "conjugate_update"
        assert update_rule["learning_rate"] == "adaptive"
        assert update_rule["forgetting_factor"] == 0.99

        # Uncertainty estimation
        uncertainty = learning["uncertainty_estimation"]
        assert uncertainty["method"] == "posterior_variance"
        assert uncertainty["confidence_intervals"] is True

        # Adaptation dynamics
        adaptation = learnable_trans["adaptation_dynamics"]

        # Adaptation speed
        speed = adaptation["adaptation_speed"]
        assert speed["initial"] == "slow"
        assert speed["asymptotic"] == "fast"
        assert speed["acceleration_function"] == "exponential"

        # Plasticity constraints
        plasticity = adaptation["plasticity_constraints"]
        assert plasticity["structural_stability"] is True
        assert plasticity["catastrophic_forgetting_prevention"] is True

        # Meta-learning
        meta_learning = adaptation["meta_learning"]
        assert meta_learning["enabled"] is True
        meta_params = meta_learning["meta_parameter_optimization"]
        assert "learning_rate" in meta_params
        assert "forgetting_factor" in meta_params

    def test_parse_context_dependent_transition_learning(self):
        """Test parsing context-dependent transition learning mechanisms."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "contextual_transitions",
                    "type": "transition",
                    "transition_type": "context_dependent_adaptive",
                    "context_structure": {
                        "context_variables": [
                            {
                                "name": "environmental_condition",
                                "type": "categorical",
                                "values": [
                                    "normal",
                                    "stress",
                                    "resource_abundant",
                                    "resource_scarce",
                                ],
                                "observation_method": "direct",
                            },
                            {
                                "name": "internal_state",
                                "type": "continuous",
                                "dimensions": 3,
                                "bounds": [[0, 1], [0, 1], [0, 1]],
                                "observation_method": "inferred",
                            },
                        ],
                        "context_representation": {
                            "method": "feature_embedding",
                            "embedding_dimension": 8,
                            "similarity_metric": "cosine_similarity",
                        },
                    },
                    "context_dependent_learning": {
                        "learning_rule": "context_modulated_bayesian",
                        "context_specificity": {
                            "generalization_kernel": "gaussian_rbf",
                            "kernel_bandwidth": 0.5,
                            "similarity_threshold": 0.7,
                        },
                        "transition_families": {
                            "normal_conditions": {
                                "base_transition_matrix": "learned_average",
                                "adaptation_rate": "standard",
                            },
                            "stress_conditions": {
                                "base_transition_matrix": "stress_specific",
                                "adaptation_rate": "accelerated",
                                "stress_modulation": {
                                    "stress_level_dependency": True,
                                    "adaptation_amplification": 2.0,
                                },
                            },
                        },
                        "context_transfer": {
                            "enabled": True,
                            "transfer_method": "weighted_combination",
                            "transfer_weights": "similarity_based",
                            "minimum_experience_threshold": 50,
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Verify contextual transitions
        contextual_trans = result["nodes"][0]
        assert contextual_trans["transition_type"] == "context_dependent_adaptive"

        # Context structure
        context_struct = contextual_trans["context_structure"]
        context_vars = context_struct["context_variables"]
        assert len(context_vars) == 2

        # Environmental condition context
        env_context = context_vars[0]
        assert env_context["type"] == "categorical"
        assert len(env_context["values"]) == 4
        assert "stress" in env_context["values"]

        # Internal state context
        internal_context = context_vars[1]
        assert internal_context["type"] == "continuous"
        assert internal_context["dimensions"] == 3
        assert internal_context["observation_method"] == "inferred"

        # Context representation
        context_repr = context_struct["context_representation"]
        assert context_repr["method"] == "feature_embedding"
        assert context_repr["embedding_dimension"] == 8

        # Context-dependent learning
        cd_learning = contextual_trans["context_dependent_learning"]
        assert cd_learning["learning_rule"] == "context_modulated_bayesian"

        # Context specificity
        specificity = cd_learning["context_specificity"]
        assert specificity["generalization_kernel"] == "gaussian_rbf"
        assert specificity["kernel_bandwidth"] == 0.5

        # Transition families
        families = cd_learning["transition_families"]
        assert "normal_conditions" in families
        assert "stress_conditions" in families

        stress_family = families["stress_conditions"]
        assert stress_family["adaptation_rate"] == "accelerated"
        stress_mod = stress_family["stress_modulation"]
        assert stress_mod["adaptation_amplification"] == 2.0

        # Context transfer
        transfer = cd_learning["context_transfer"]
        assert transfer["enabled"] is True
        assert transfer["transfer_method"] == "weighted_combination"
        assert transfer["minimum_experience_threshold"] == 50


class TestTransitionValidation:
    """Test validation of transition specifications and constraints."""

    def test_validate_transition_matrix_constraints(self):
        """Test validation of transition matrix mathematical constraints."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "invalid_transitions",
                    "type": "transition",
                    "transition_matrices": {
                        "action_0": {
                            "matrix": [
                                [0.5, 0.3, 0.4],  # Row sum > 1
                                [0.2, 0.8, 0.1],  # Row sum > 1
                                [0.1, 0.1, 0.8],  # Valid row
                            ]
                        }
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect invalid transition probabilities
        assert is_valid is False
        prob_errors = [e for e in errors if "probability" in e.lower() or "sum" in e.lower()]
        assert len(prob_errors) > 0

    def test_validate_temporal_consistency_constraints(self):
        """Test validation of temporal consistency in time-dependent transitions."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "temporal_inconsistency",
                    "type": "transition",
                    "transition_type": "time_dependent",
                    "temporal_structure": {
                        "time_discretization": {"period": 100, "resolution": "daily"},
                        "transition_schedule": {
                            "phase_1": {
                                "time_range": [0, 60],
                                "transition_matrix": "valid_matrix_1",
                            },
                            "phase_2": {
                                "time_range": [40, 80],  # Overlaps with phase_1
                                "transition_matrix": "valid_matrix_2",
                            },
                        },
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect temporal overlap
        assert is_valid is False
        temporal_errors = [e for e in errors if "temporal" in e.lower() or "overlap" in e.lower()]
        assert len(temporal_errors) > 0

    def test_validate_hierarchical_consistency_constraints(self):
        """Test validation of consistency in hierarchical transition structures."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "inconsistent_hierarchy",
                    "type": "transition",
                    "transition_type": "hierarchical",
                    "hierarchy_structure": {
                        "level_1": {"num_states": 8, "aggregation_ratio": 2},
                        "level_2": {
                            "num_states": 5,  # Inconsistent: should be 4 (8/2)
                            "aggregation_ratio": 2,
                        },
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect hierarchical inconsistency
        assert is_valid is False
        hierarchy_errors = [
            e for e in errors if "hierarchy" in e.lower() or "aggregation" in e.lower()
        ]
        assert len(hierarchy_errors) > 0
