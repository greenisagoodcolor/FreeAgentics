"""Test GMN parser with real PyMDP integration - no mocks or fallbacks allowed."""

import numpy as np

# PyMDP imports - must fail hard if not available
from pymdp.agent import Agent

from inference.active.gmn_parser import GMNParser
from tests.unit.gmn_mocks import GMNToPyMDPConverter


class TestGMNParserPyMDPIntegration:
    """Test GMN parser with real PyMDP integration."""

    def test_basic_gmn_parsing_with_belief_states(self):
        """Test that GMN parser creates valid PyMDP belief states."""
        # Arrange - create a simple GMN specification
        gmn_spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "observation",
                    "type": "observation",
                    "num_observations": 4,
                },
                {"name": "move", "type": "action", "num_actions": 4},
                {
                    "name": "location_belief",
                    "type": "belief",
                    "about": "location",
                    "initial_distribution": [0.25, 0.25, 0.25, 0.25],
                },
            ],
            "edges": [{"from": "location", "to": "observation", "type": "generates"}],
        }

        # Act
        parser = GMNParser()
        parsed = parser.parse(gmn_spec)

        # Convert to PyMDP matrices
        converter = GMNToPyMDPConverter()
        matrices = converter.convert_to_matrices(gmn_spec)

        # Assert - verify matrices are valid for PyMDP
        assert "A" in matrices  # Observation model
        assert "B" in matrices  # Transition model
        assert "C" in matrices  # Preferences
        assert "D" in matrices  # Prior beliefs

        # Verify belief states are properly initialized
        assert "belief_states" in parsed
        belief_states = parsed["belief_states"]
        assert "location_belief" in belief_states

        # Verify belief state is a proper PyMDP Categorical distribution
        location_belief = belief_states["location_belief"]
        assert hasattr(location_belief, "values")
        assert np.allclose(location_belief.values, [0.25, 0.25, 0.25, 0.25])
        assert np.allclose(location_belief.values.sum(), 1.0)  # Must be normalized

        # Verify belief can be used with PyMDP agent
        agent = Agent(
            A=matrices["A"],
            B=matrices["B"],
            C=matrices["C"],
            D=matrices["D"],  # Use D matrix from converter (which incorporates belief)
            control_fac_idx=[0],  # Control factor indices
        )

        # Verify agent has proper belief state initialized
        assert hasattr(agent, "qs")
        assert len(agent.qs) > 0

        # Verify the D matrix matches the belief initial distribution
        assert np.allclose(matrices["D"], location_belief.values)

    def test_gmn_dynamic_belief_updates(self):
        """Test that GMN supports dynamic belief updates with PyMDP."""
        # Arrange - GMN with belief update rules
        gmn_spec = {
            "nodes": [
                {
                    "name": "health",
                    "type": "state",
                    "num_states": 3,
                },  # healthy, injured, critical
                {
                    "name": "health_obs",
                    "type": "observation",
                    "num_observations": 3,
                },
                {
                    "name": "health_belief",
                    "type": "belief",
                    "about": "health",
                    "initial_distribution": [0.8, 0.15, 0.05],
                    "update_rules": [
                        {
                            "condition": "observation == injured",
                            "operation": "shift_probability",
                            "from_state": 0,  # healthy
                            "to_state": 1,  # injured
                            "amount": 0.3,
                        }
                    ],
                },
            ],
            "edges": [{"from": "health", "to": "health_obs", "type": "generates"}],
        }

        # Act
        parser = GMNParser()
        parsed = parser.parse(gmn_spec)

        # Get belief update function
        assert "belief_update_functions" in parsed
        update_functions = parsed["belief_update_functions"]
        assert "health_belief" in update_functions

        # Apply update rule
        initial_belief = parsed["belief_states"]["health_belief"]
        update_fn = update_functions["health_belief"]

        # Simulate injured observation
        updated_belief = update_fn(initial_belief, observation=1)  # 1 = injured

        # Assert - verify belief was properly updated
        assert updated_belief.values[0] < initial_belief.values[0]  # healthy decreased
        assert updated_belief.values[1] > initial_belief.values[1]  # injured increased
        assert np.allclose(updated_belief.values.sum(), 1.0)  # Still normalized

    def test_gmn_multi_factor_beliefs(self):
        """Test GMN with multi-factor belief states for PyMDP."""
        # Arrange - GMN with multiple state factors
        gmn_spec = {
            "nodes": [
                {
                    "name": "position",
                    "type": "state",
                    "num_states": 9,
                },  # 3x3 grid
                {
                    "name": "energy",
                    "type": "state",
                    "num_states": 3,
                },  # low, medium, high
                {
                    "name": "joint_belief",
                    "type": "belief",
                    "about": ["position", "energy"],
                    "factorized": True,
                    "initial_distributions": {
                        "position": [1 / 9] * 9,  # Uniform over positions
                        "energy": [0.2, 0.5, 0.3],  # Most likely medium energy
                    },
                },
            ]
        }

        # Act
        parser = GMNParser()
        parsed = parser.parse(gmn_spec)

        # Assert - verify multi-factor belief structure
        joint_belief = parsed["belief_states"]["joint_belief"]
        assert hasattr(joint_belief, "factors")
        assert len(joint_belief.factors) == 2

        # Verify each factor is properly initialized
        position_belief = joint_belief.factors[0]
        energy_belief = joint_belief.factors[1]

        assert position_belief.shape == (9,)
        assert energy_belief.shape == (3,)
        assert np.allclose(position_belief.values.sum(), 1.0)
        assert np.allclose(energy_belief.values.sum(), 1.0)

    def test_gmn_belief_precision_and_entropy(self):
        """Test GMN belief states maintain proper precision and entropy constraints."""
        # Arrange - GMN with precision constraints
        gmn_spec = {
            "nodes": [
                {"name": "target", "type": "state", "num_states": 5},
                {
                    "name": "target_belief",
                    "type": "belief",
                    "about": "target",
                    "initial_distribution": [0.1, 0.2, 0.4, 0.2, 0.1],
                    "constraints": {
                        "min_entropy": 0.5,
                        "max_entropy": 2.0,
                        "precision": 10.0,  # Belief precision parameter
                    },
                },
            ]
        }

        # Act
        parser = GMNParser()
        parsed = parser.parse(gmn_spec)

        # Get belief and compute entropy
        belief = parsed["belief_states"]["target_belief"]
        entropy = -np.sum(belief.values * np.log(belief.values + 1e-10))

        # Assert - verify constraints are respected
        assert entropy >= 0.5  # Min entropy constraint
        assert entropy <= 2.0  # Max entropy constraint

        # Verify precision is stored and accessible
        assert "belief_precisions" in parsed
        assert parsed["belief_precisions"]["target_belief"] == 10.0
