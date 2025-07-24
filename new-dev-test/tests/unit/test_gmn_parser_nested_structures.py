"""Comprehensive failing tests for GMN parser nested structures support.

This test suite defines expected behavior for parsing nested GMN structures:
- Hierarchical state spaces
- Nested belief structures
- Multi-level observation mappings
- Complex factor graph representations
- Nested preferences and policies

Following strict TDD: These tests MUST fail initially and drive implementation.
NO graceful fallbacks or try-except blocks allowed.
"""

import numpy as np

from inference.active.gmn_parser import GMNParser, GMNSchemaValidator

# Import mocks for missing classes
from tests.unit.gmn_mocks import GMNToPyMDPConverter


class TestNestedStateSpaces:
    """Test parsing of hierarchical and nested state spaces."""

    def test_parse_hierarchical_state_with_factors(self):
        """Test parsing hierarchical state spaces with multiple factors."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "agent_state",
                    "type": "state",
                    "hierarchical": True,
                    "factors": [
                        {
                            "name": "location",
                            "type": "factor",
                            "num_states": 9,
                            "spatial_structure": "grid_3x3",
                        },
                        {
                            "name": "health",
                            "type": "factor",
                            "num_states": 3,
                            "states": ["healthy", "injured", "critical"],
                        },
                        {
                            "name": "resources",
                            "type": "factor",
                            "num_states": 5,
                            "range": [0, 100],
                            "discretization": "uniform",
                        },
                    ],
                    "factor_dependencies": [
                        {
                            "factor": "health",
                            "depends_on": ["location"],
                            "dependency_type": "conditional",
                        }
                    ],
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        # Should parse hierarchical structure correctly
        agent_state = result["nodes"][0]
        assert agent_state["hierarchical"] is True
        assert len(agent_state["factors"]) == 3

        # Location factor with spatial structure
        location_factor = agent_state["factors"][0]
        assert location_factor["name"] == "location"
        assert location_factor["num_states"] == 9
        assert location_factor["spatial_structure"] == "grid_3x3"

        # Health factor with named states
        health_factor = agent_state["factors"][1]
        assert health_factor["states"] == ["healthy", "injured", "critical"]

        # Resources factor with continuous discretization
        resources_factor = agent_state["factors"][2]
        assert resources_factor["range"] == [0, 100]
        assert resources_factor["discretization"] == "uniform"

        # Factor dependencies
        assert len(agent_state["factor_dependencies"]) == 1
        dep = agent_state["factor_dependencies"][0]
        assert dep["factor"] == "health"
        assert dep["depends_on"] == ["location"]

    def test_parse_nested_observation_modalities(self):
        """Test parsing nested observation modalities with different structures."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "sensory_input",
                    "type": "observation",
                    "modalities": [
                        {
                            "name": "visual",
                            "type": "modality",
                            "structure": "grid",
                            "dimensions": [8, 8],
                            "channels": 3,
                            "encoding": "categorical",
                        },
                        {
                            "name": "proprioceptive",
                            "type": "modality",
                            "structure": "vector",
                            "dimensions": [4],
                            "encoding": "continuous",
                            "discretization_bins": 10,
                        },
                        {
                            "name": "auditory",
                            "type": "modality",
                            "structure": "sequence",
                            "max_length": 16,
                            "vocabulary_size": 64,
                            "encoding": "categorical",
                        },
                    ],
                    "cross_modal_dependencies": [
                        {
                            "modality": "visual",
                            "influences": ["proprioceptive"],
                            "strength": 0.3,
                        }
                    ],
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        sensory_input = result["nodes"][0]
        assert len(sensory_input["modalities"]) == 3

        # Visual modality with grid structure
        visual = sensory_input["modalities"][0]
        assert visual["structure"] == "grid"
        assert visual["dimensions"] == [8, 8]
        assert visual["channels"] == 3

        # Proprioceptive with continuous encoding
        proprioceptive = sensory_input["modalities"][1]
        assert proprioceptive["encoding"] == "continuous"
        assert proprioceptive["discretization_bins"] == 10

        # Auditory with sequence structure
        auditory = sensory_input["modalities"][2]
        assert auditory["structure"] == "sequence"
        assert auditory["max_length"] == 16

        # Cross-modal dependencies
        cross_deps = sensory_input["cross_modal_dependencies"]
        assert len(cross_deps) == 1
        assert cross_deps[0]["modality"] == "visual"
        assert cross_deps[0]["influences"] == ["proprioceptive"]

    def test_parse_complex_nested_beliefs(self):
        """Test parsing complex nested belief structures with hierarchical organization."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "world_model",
                    "type": "belief",
                    "about": "world_state",
                    "hierarchical": True,
                    "levels": [
                        {
                            "level": 1,
                            "name": "local_beliefs",
                            "scope": "immediate_environment",
                            "factors": [
                                {
                                    "name": "local_position_belief",
                                    "about": "position",
                                    "uncertainty": "gaussian",
                                    "precision": 2.5,
                                    "initial_distribution": [0.7, 0.2, 0.1],
                                },
                                {
                                    "name": "local_hazard_belief",
                                    "about": "hazards",
                                    "uncertainty": "categorical",
                                    "precision": 1.0,
                                    "initial_distribution": [0.9, 0.05, 0.05],
                                },
                            ],
                        },
                        {
                            "level": 2,
                            "name": "global_beliefs",
                            "scope": "environment_structure",
                            "factors": [
                                {
                                    "name": "map_belief",
                                    "about": "environment_layout",
                                    "uncertainty": "dirichlet",
                                    "precision": 0.5,
                                    "prior_strength": 10.0,
                                }
                            ],
                        },
                    ],
                    "belief_propagation": {
                        "method": "message_passing",
                        "iterations": 10,
                        "convergence_threshold": 1e-6,
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        world_model = result["nodes"][0]
        assert world_model["hierarchical"] is True
        assert len(world_model["levels"]) == 2

        # Level 1 local beliefs
        local_level = world_model["levels"][0]
        assert local_level["level"] == 1
        assert local_level["scope"] == "immediate_environment"
        assert len(local_level["factors"]) == 2

        pos_belief = local_level["factors"][0]
        assert pos_belief["uncertainty"] == "gaussian"
        assert pos_belief["precision"] == 2.5

        # Level 2 global beliefs
        global_level = world_model["levels"][1]
        assert global_level["level"] == 2
        map_belief = global_level["factors"][0]
        assert map_belief["uncertainty"] == "dirichlet"
        assert map_belief["prior_strength"] == 10.0

        # Belief propagation parameters
        bp = world_model["belief_propagation"]
        assert bp["method"] == "message_passing"
        assert bp["iterations"] == 10


class TestNestedPolicyStructures:
    """Test parsing of nested policy structures with hierarchical organization."""

    def test_parse_hierarchical_policy_with_temporal_structure(self):
        """Test parsing hierarchical policies with temporal decomposition."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "behavioral_policy",
                    "type": "policy",
                    "hierarchical": True,
                    "temporal_structure": {
                        "horizon": 10,
                        "decomposition": "hierarchical",
                        "levels": [
                            {
                                "level": 1,
                                "name": "immediate_actions",
                                "time_scale": 1,
                                "action_space": {
                                    "type": "discrete",
                                    "actions": [
                                        "move_up",
                                        "move_down",
                                        "move_left",
                                        "move_right",
                                        "stay",
                                    ],
                                    "constraints": [
                                        {
                                            "type": "spatial",
                                            "condition": "boundary_check",
                                        }
                                    ],
                                },
                            },
                            {
                                "level": 2,
                                "name": "strategic_goals",
                                "time_scale": 5,
                                "action_space": {
                                    "type": "abstract",
                                    "goals": [
                                        "explore",
                                        "exploit",
                                        "avoid_danger",
                                    ],
                                    "goal_decomposition": {
                                        "explore": ["move_to_unknown"],
                                        "exploit": ["move_to_resource"],
                                        "avoid_danger": ["move_away_from_hazard"],
                                    },
                                },
                            },
                        ],
                    },
                    "policy_evaluation": {
                        "method": "expected_free_energy",
                        "parameters": {
                            "epistemic_weight": 1.0,
                            "pragmatic_weight": 1.0,
                            "temporal_discount": 0.95,
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        policy = result["nodes"][0]
        assert policy["hierarchical"] is True

        temp_struct = policy["temporal_structure"]
        assert temp_struct["horizon"] == 10
        assert temp_struct["decomposition"] == "hierarchical"
        assert len(temp_struct["levels"]) == 2

        # Level 1 immediate actions
        immediate = temp_struct["levels"][0]
        assert immediate["time_scale"] == 1
        action_space = immediate["action_space"]
        assert len(action_space["actions"]) == 5
        assert len(action_space["constraints"]) == 1

        # Level 2 strategic goals
        strategic = temp_struct["levels"][1]
        assert strategic["time_scale"] == 5
        goal_space = strategic["action_space"]
        assert goal_space["type"] == "abstract"
        decomp = goal_space["goal_decomposition"]
        assert "explore" in decomp

        # Policy evaluation
        eval_params = policy["policy_evaluation"]
        assert eval_params["method"] == "expected_free_energy"
        params = eval_params["parameters"]
        assert params["epistemic_weight"] == 1.0

    def test_parse_multi_agent_nested_policies(self):
        """Test parsing multi-agent policies with nested coordination structures."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "collective_policy",
                    "type": "policy",
                    "multi_agent": True,
                    "agents": [
                        {
                            "id": "agent_1",
                            "role": "explorer",
                            "individual_policy": {
                                "type": "individual",
                                "action_space": [
                                    "move",
                                    "observe",
                                    "communicate",
                                ],
                                "constraints": {
                                    "communication_range": 3,
                                    "energy_limit": 100,
                                },
                            },
                            "coordination_policy": {
                                "type": "message_passing",
                                "protocols": [
                                    "information_sharing",
                                    "task_allocation",
                                ],
                            },
                        },
                        {
                            "id": "agent_2",
                            "role": "collector",
                            "individual_policy": {
                                "type": "individual",
                                "action_space": [
                                    "move",
                                    "collect",
                                    "share_resources",
                                ],
                                "constraints": {
                                    "carrying_capacity": 10,
                                    "collection_range": 1,
                                },
                            },
                            "coordination_policy": {
                                "type": "negotiation",
                                "protocols": [
                                    "resource_exchange",
                                    "priority_resolution",
                                ],
                            },
                        },
                    ],
                    "global_coordination": {
                        "mechanism": "auction_based",
                        "objective": "collective_efficiency",
                        "communication_topology": "fully_connected",
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        collective_policy = result["nodes"][0]
        assert collective_policy["multi_agent"] is True
        assert len(collective_policy["agents"]) == 2

        # Agent 1 explorer
        agent1 = collective_policy["agents"][0]
        assert agent1["role"] == "explorer"
        assert len(agent1["individual_policy"]["action_space"]) == 3
        assert agent1["coordination_policy"]["type"] == "message_passing"

        # Agent 2 collector
        agent2 = collective_policy["agents"][1]
        assert agent2["role"] == "collector"
        constraints = agent2["individual_policy"]["constraints"]
        assert constraints["carrying_capacity"] == 10

        # Global coordination
        global_coord = collective_policy["global_coordination"]
        assert global_coord["mechanism"] == "auction_based"
        assert global_coord["communication_topology"] == "fully_connected"


class TestNestedValidationFramework:
    """Test validation of nested structures with complex constraints."""

    def test_validate_nested_structure_consistency(self):
        """Test validation of consistency across nested structure levels."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "nested_state",
                    "type": "state",
                    "hierarchical": True,
                    "factors": [
                        {"name": "factor_a", "num_states": 4},
                        {"name": "factor_b", "num_states": 3},
                    ],
                },
                {
                    "name": "nested_obs",
                    "type": "observation",
                    "modalities": [
                        {
                            "name": "modality_a",
                            "dimensions": [4],
                        },  # Should match factor_a
                        {
                            "name": "modality_b",
                            "dimensions": [5],
                        },  # Mismatch with factor_b
                    ],
                },
            ],
            "edges": [
                {
                    "from": "nested_state",
                    "to": "nested_obs",
                    "type": "generates",
                    "factor_mappings": [
                        {"factor": "factor_a", "modality": "modality_a"},
                        {"factor": "factor_b", "modality": "modality_b"},
                    ],
                }
            ],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect dimension mismatch in nested structures
        assert is_valid is False
        dimension_errors = [e for e in errors if "dimension" in e.lower()]
        assert len(dimension_errors) > 0

    def test_validate_hierarchical_belief_constraints(self):
        """Test validation of hierarchical belief structure constraints."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "hierarchical_belief",
                    "type": "belief",
                    "hierarchical": True,
                    "levels": [
                        {
                            "level": 1,
                            "factors": [
                                {
                                    "name": "level1_factor",
                                    "precision": 2.0,
                                    "initial_distribution": [
                                        0.5,
                                        0.3,
                                        0.2,
                                    ],  # Sum = 1.0
                                }
                            ],
                        },
                        {
                            "level": 2,
                            "factors": [
                                {
                                    "name": "level2_factor",
                                    "precision": -1.0,  # Invalid negative precision
                                    "initial_distribution": [
                                        0.7,
                                        0.5,
                                    ],  # Sum > 1.0
                                }
                            ],
                        },
                    ],
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect invalid precision and probability distribution
        assert is_valid is False
        precision_errors = [e for e in errors if "precision" in e.lower()]
        assert len(precision_errors) > 0

    def test_validate_policy_temporal_consistency(self):
        """Test validation of temporal consistency in hierarchical policies."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "temporal_policy",
                    "type": "policy",
                    "temporal_structure": {
                        "horizon": 5,
                        "levels": [
                            {
                                "level": 1,
                                "time_scale": 1,
                                "horizon": 5,  # Should be consistent with global
                            },
                            {
                                "level": 2,
                                "time_scale": 3,
                                "horizon": 10,
                            },  # Exceeds global horizon
                        ],
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect temporal inconsistency
        assert is_valid is False
        temporal_errors = [e for e in errors if "temporal" in e.lower() or "horizon" in e.lower()]
        assert len(temporal_errors) > 0


class TestNestedToPyMDPConversion:
    """Test conversion of nested structures to PyMDP format."""

    def test_convert_hierarchical_state_to_factorized_arrays(self):
        """Test conversion of hierarchical states to factorized PyMDP arrays."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {
                    "name": "hierarchical_state",
                    "type": "state",
                    "hierarchical": True,
                    "factors": [
                        {"name": "location", "num_states": 4},
                        {"name": "health", "num_states": 3},
                    ],
                },
                {
                    "name": "joint_observation",
                    "type": "observation",
                    "modalities": [
                        {"name": "location_obs", "dimensions": [4]},
                        {"name": "health_obs", "dimensions": [3]},
                    ],
                },
            ],
            "edges": [
                {
                    "from": "hierarchical_state",
                    "to": "joint_observation",
                    "type": "generates",
                    "factor_mappings": [
                        {"factor": "location", "modality": "location_obs"},
                        {"factor": "health", "modality": "health_obs"},
                    ],
                }
            ],
        }

        matrices = converter.convert_to_matrices(spec)

        # Should produce factorized A matrices
        assert "A" in matrices
        A_matrices = matrices["A"]

        # Should be a list/array of matrices for each modality
        assert isinstance(A_matrices, (list, np.ndarray))

        if isinstance(A_matrices, list):
            assert len(A_matrices) == 2  # Two modalities
            assert A_matrices[0].shape == (4, 4)  # Location observation
            assert A_matrices[1].shape == (3, 3)  # Health observation

    def test_convert_nested_beliefs_to_factorized_distributions(self):
        """Test conversion of nested beliefs to factorized PyMDP distributions."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {
                    "name": "factorized_belief",
                    "type": "belief",
                    "hierarchical": True,
                    "factors": [
                        {
                            "name": "position_belief",
                            "initial_distribution": [0.6, 0.2, 0.2],
                            "precision": 2.0,
                        },
                        {
                            "name": "status_belief",
                            "initial_distribution": [0.8, 0.15, 0.05],
                            "precision": 1.5,
                        },
                    ],
                }
            ],
            "edges": [],
        }

        matrices = converter.convert_to_matrices(spec)

        # Should produce factorized D vector
        assert "D" in matrices
        D_factors = matrices["D"]

        # Should be a list of factor distributions
        assert isinstance(D_factors, list)
        assert len(D_factors) == 2

        # Check distributions are properly normalized
        assert np.allclose(D_factors[0].sum(), 1.0)
        assert np.allclose(D_factors[1].sum(), 1.0)

        # Check precision information is preserved
        assert "belief_precisions" in matrices
        precisions = matrices["belief_precisions"]
        assert len(precisions) == 2

    def test_convert_hierarchical_policy_to_nested_matrices(self):
        """Test conversion of hierarchical policies to nested PyMDP matrices."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {
                    "name": "hierarchical_policy",
                    "type": "policy",
                    "temporal_structure": {
                        "horizon": 6,
                        "levels": [
                            {"level": 1, "time_scale": 1, "num_actions": 4},
                            {"level": 2, "time_scale": 3, "num_actions": 2},
                        ],
                    },
                }
            ],
            "edges": [],
        }

        matrices = converter.convert_to_matrices(spec)

        # Should produce hierarchical policy matrices
        assert "policies" in matrices
        policy_matrices = matrices["policies"]

        # Should contain policies for each level
        assert isinstance(policy_matrices, dict)
        assert "level_1" in policy_matrices
        assert "level_2" in policy_matrices

        # Level 1 policies (fine-grained, 6 timesteps)
        level1_policies = policy_matrices["level_1"]
        assert level1_policies.shape[1] == 6  # Horizon

        # Level 2 policies (coarse-grained, 2 timesteps for scale 3)
        level2_policies = policy_matrices["level_2"]
        assert level2_policies.shape[1] == 2  # Horizon / time_scale
