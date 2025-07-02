"""
Tests for GNN Model Module.

Comprehensive test suite for GMN (Generalized Model Notation) and GNN model data structures
used in the FreeAgentics Active Inference implementation.
"""

import pytest

from inference.gnn.model import GMNModel, GNNModel


class TestGMNModel:
    """Test GMN (Generalized Model Notation) model functionality."""

    def test_gmn_model_initialization(self):
        """Test GMN model initialization."""
        model = GMNModel("test_model")

        assert model.name == "test_model"
        assert model.description == ""
        assert isinstance(model.state_space, dict)
        assert isinstance(model.observations, dict)
        assert isinstance(model.connections, list)
        assert isinstance(model.update_equations, dict)
        assert isinstance(model.preferences, dict)
        assert isinstance(model.metadata, dict)

        # Check all containers are empty initially
        assert len(model.state_space) == 0
        assert len(model.observations) == 0
        assert len(model.connections) == 0
        assert len(model.update_equations) == 0
        assert len(model.preferences) == 0
        assert len(model.metadata) == 0

    def test_gmn_model_attribute_setting(self):
        """Test setting GMN model attributes."""
        model = GMNModel("agent_model")

        # Set description
        model.description = "Active inference agent model"
        assert model.description == "Active inference agent model"

        # Set state space
        model.state_space = {
            "belief_state": {"dim": 3, "type": "categorical"},
            "hidden_state": {"dim": 5, "type": "continuous"},
        }
        assert len(model.state_space) == 2
        assert model.state_space["belief_state"]["dim"] == 3
        assert model.state_space["hidden_state"]["type"] == "continuous"

        # Set observations
        model.observations = {
            "visual": {"channels": 3, "resolution": [64, 64]},
            "proprioceptive": {"dim": 6},
        }
        assert len(model.observations) == 2
        assert model.observations["visual"]["channels"] == 3
        assert model.observations["proprioceptive"]["dim"] == 6

        # Set connections
        model.connections = [
            {"from": "belief_state", "to": "action", "type": "policy"},
            {"from": "observation", "to": "belief_state", "type": "update"},
        ]
        assert len(model.connections) == 2
        assert model.connections[0]["from"] == "belief_state"
        assert model.connections[1]["type"] == "update"

        # Set update equations
        model.update_equations = {
            "belief_update": "Q_s = softmax(log_prior + log_likelihood)",
            "action_selection": "a = argmax(G(pi))",
        }
        assert len(model.update_equations) == 2
        assert "belief_update" in model.update_equations
        assert "action_selection" in model.update_equations

        # Set preferences
        model.preferences = {
            "goal_state": [
                1,
                0,
                0],
            "precision": 1.0,
            "temperature": 0.1}
        assert len(model.preferences) == 3
        assert model.preferences["goal_state"] == [1, 0, 0]
        assert model.preferences["precision"] == 1.0

        # Set metadata
        model.metadata = {
            "created": "2024-01-01",
            "version": "1.0",
            "author": "FreeAgentics"}
        assert len(model.metadata) == 3
        assert model.metadata["version"] == "1.0"

    def test_gmn_model_execute(self):
        """Test GMN model execute method."""
        model = GMNModel("executable_model")

        # Execute should not raise any errors (it's a placeholder)
        model.execute()

        # Model should remain unchanged after execute
        assert model.name == "executable_model"
        assert len(model.state_space) == 0

    def test_gmn_model_with_complex_data(self):
        """Test GMN model with complex nested data structures."""
        model = GMNModel("complex_model")

        # Complex state space with nested structures
        model.state_space = {
            "hierarchical_beliefs": {
                "level_1": {"states": ["explore", "exploit"], "prior": [0.5, 0.5]},
                "level_2": {"states": ["left", "right", "forward"], "prior": [0.3, 0.3, 0.4]},
            },
            "continuous_states": {
                "position": {"dim": 2, "bounds": [[-10, 10], [-10, 10]]},
                "velocity": {"dim": 2, "bounds": [[-5, 5], [-5, 5]]},
            },
        }

        # Complex observations with multiple modalities
        model.observations = {
            "sensory": {
                "visual": {
                    "rgb": {"shape": [3, 224, 224], "range": [0, 255]},
                    "depth": {"shape": [1, 224, 224], "range": [0, 10]},
                },
                "audio": {"spectrogram": {"shape": [128, 256], "sampling_rate": 44100}},
            },
            "proprioceptive": {
                "joint_angles": {"dim": 7, "range": [-3.14, 3.14]},
                "joint_velocities": {"dim": 7, "range": [-10, 10]},
            },
        }

        # Verify complex data is stored correctly
        assert "hierarchical_beliefs" in model.state_space
        assert model.state_space["hierarchical_beliefs"]["level_1"]["prior"] == [
            0.5, 0.5]
        assert model.observations["sensory"]["visual"]["rgb"]["shape"] == [
            3, 224, 224]
        assert model.observations["proprioceptive"]["joint_angles"]["dim"] == 7

    def test_gmn_model_name_variations(self):
        """Test GMN model with different name types."""
        # Simple name
        model1 = GMNModel("simple")
        assert model1.name == "simple"

        # Name with spaces
        model2 = GMNModel("complex agent model")
        assert model2.name == "complex agent model"

        # Name with special characters
        model3 = GMNModel("agent_v2.1-beta")
        assert model3.name == "agent_v2.1-beta"

        # Empty name
        model4 = GMNModel("")
        assert model4.name == ""


class TestGNNModel:
    """Test GNN model data class functionality."""

    def test_gnn_model_default_initialization(self):
        """Test GNN model default initialization."""
        model = GNNModel()

        assert model.name == ""
        assert isinstance(model.state_space, dict)
        assert isinstance(model.observations, dict)
        assert isinstance(model.connections, list)
        assert isinstance(model.update_equations, dict)
        assert isinstance(model.preferences, dict)
        assert isinstance(model.metadata, dict)

        # Check all containers are empty initially
        assert len(model.state_space) == 0
        assert len(model.observations) == 0
        assert len(model.connections) == 0
        assert len(model.update_equations) == 0
        assert len(model.preferences) == 0
        assert len(model.metadata) == 0

    def test_gnn_model_with_name(self):
        """Test GNN model initialization with name."""
        model = GNNModel(name="neural_agent")

        assert model.name == "neural_agent"
        assert len(model.state_space) == 0
        assert len(model.observations) == 0

    def test_gnn_model_with_all_parameters(self):
        """Test GNN model with all parameters specified."""
        state_space = {
            "belief": {"type": "categorical", "size": 4},
            "hidden": {"type": "continuous", "size": 10},
        }

        observations = {
            "visual": {"channels": 3, "width": 64, "height": 64},
            "tactile": {"sensors": 16},
        }

        connections = [
            {"from": "input", "to": "hidden", "weight": 0.5},
            {"from": "hidden", "to": "output", "weight": 0.8},
        ]

        update_equations = {
            "forward": "h = tanh(Wx + b)",
            "backward": "dW = h.T @ delta"}

        preferences = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "regularization": 0.01}

        metadata = {
            "architecture": "feedforward",
            "layers": 3,
            "parameters": 1024}

        model = GNNModel(
            name="full_model",
            state_space=state_space,
            observations=observations,
            connections=connections,
            update_equations=update_equations,
            preferences=preferences,
            metadata=metadata,
        )

        assert model.name == "full_model"
        assert model.state_space == state_space
        assert model.observations == observations
        assert model.connections == connections
        assert model.update_equations == update_equations
        assert model.preferences == preferences
        assert model.metadata == metadata

        # Test specific nested values
        assert model.state_space["belief"]["type"] == "categorical"
        assert model.observations["visual"]["channels"] == 3
        assert model.connections[0]["weight"] == 0.5
        assert model.update_equations["forward"] == "h = tanh(Wx + b)"
        assert model.preferences["learning_rate"] == 0.001
        assert model.metadata["layers"] == 3

    def test_gnn_model_field_modification(self):
        """Test modifying GNN model fields after creation."""
        model = GNNModel(name="modifiable")

        # Add to state space
        model.state_space["new_state"] = {"dim": 5}
        assert "new_state" in model.state_space
        assert model.state_space["new_state"]["dim"] == 5

        # Add to observations
        model.observations["lidar"] = {"beams": 360, "range": 100}
        assert "lidar" in model.observations
        assert model.observations["lidar"]["beams"] == 360

        # Add to connections
        model.connections.append({"type": "attention", "heads": 8})
        assert len(model.connections) == 1
        assert model.connections[0]["heads"] == 8

        # Add to update equations
        model.update_equations["attention"] = "A = softmax(QK^T/sqrt(d))"
        assert "attention" in model.update_equations

        # Add to preferences
        model.preferences["dropout"] = 0.1
        assert model.preferences["dropout"] == 0.1

        # Add to metadata
        model.metadata["created_by"] = "test_suite"
        assert model.metadata["created_by"] == "test_suite"

    def test_gnn_model_dataclass_behavior(self):
        """Test GNN model dataclass specific behavior."""
        model1 = GNNModel(name="test")
        model2 = GNNModel(name="test")

        # Dataclasses with same values should be equal
        assert model1 == model2

        # Change one field
        model2.name = "different"
        assert model1 != model2

        # Test string representation contains class name
        repr_str = repr(model1)
        assert "GNNModel" in repr_str
        assert "name='test'" in repr_str

    def test_gnn_model_copy_semantics(self):
        """Test GNN model copying behavior."""
        import copy

        original = GNNModel(
            name="original",
            state_space={"state": {"value": 1}},
            connections=[{"connection": "test"}],
        )

        # Shallow copy
        shallow = copy.copy(original)
        assert shallow.name == original.name
        assert shallow.state_space is original.state_space  # Same reference

        # Deep copy
        deep = copy.deepcopy(original)
        assert deep.name == original.name
        assert deep.state_space is not original.state_space  # Different reference
        assert deep.state_space == original.state_space  # Same content

        # Modify deep copy
        deep.state_space["state"]["value"] = 2
        # Original unchanged
        assert original.state_space["state"]["value"] == 1
        assert deep.state_space["state"]["value"] == 2  # Deep copy changed


class TestModelIntegration:
    """Integration tests for GMN and GNN models."""

    def test_model_compatibility(self):
        """Test that GMN and GNN models are compatible."""
        # Create GMN model
        gmn = GMNModel("agent_gmn")
        gmn.state_space = {"belief": {"states": 3}}
        gmn.observations = {"vision": {"dim": 64}}

        # Create equivalent GNN model
        gnn = GNNModel(
            name="agent_gnn",
            state_space={"belief": {"states": 3}},
            observations={"vision": {"dim": 64}},
        )

        # Both should have compatible data structures
        assert gmn.state_space == gnn.state_space
        assert gmn.observations == gnn.observations

        # Names can be different
        assert gmn.name != gnn.name

    def test_model_conversion_workflow(self):
        """Test workflow for converting between model types."""
        # Start with GMN model
        gmn = GMNModel("prototype")
        gmn.state_space = {"q": {"dim": 4}, "s": {"dim": 3}}
        gmn.observations = {"o": {"modalities": 2}}
        gmn.preferences = {"goal": [1, 0, 0]}
        gmn.metadata = {"type": "active_inference"}

        # Convert to GNN model (manual conversion)
        gnn = GNNModel(
            name=f"converted_{gmn.name}",
            state_space=gmn.state_space.copy(),
            observations=gmn.observations.copy(),
            preferences=gmn.preferences.copy(),
            metadata=gmn.metadata.copy(),
        )

        # Verify conversion preserved data
        assert gnn.state_space["q"]["dim"] == 4
        assert gnn.observations["o"]["modalities"] == 2
        assert gnn.preferences["goal"] == [1, 0, 0]
        assert gnn.metadata["type"] == "active_inference"
        assert gnn.name == "converted_prototype"

    def test_model_serialization_compatibility(self):
        """Test that models can be serialized/deserialized."""
        import json

        # Create GNN model with serializable data
        original = GNNModel(
            name="serializable",
            state_space={"belief": {"dim": 3, "type": "categorical"}},
            observations={"visual": {"shape": [64, 64, 3]}},
            preferences={"learning_rate": 0.01, "batch_size": 32},
            metadata={"version": "1.0", "created": "2024-01-01"},
        )

        # Convert to dict (as if for JSON serialization)
        as_dict = {
            "name": original.name,
            "state_space": original.state_space,
            "observations": original.observations,
            "connections": original.connections,
            "update_equations": original.update_equations,
            "preferences": original.preferences,
            "metadata": original.metadata,
        }

        # Serialize to JSON string
        json_str = json.dumps(as_dict)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize back
        restored_dict = json.loads(json_str)

        # Create new model from restored data
        restored = GNNModel(**restored_dict)

        # Verify restoration
        assert restored == original
        assert restored.name == "serializable"
        assert restored.state_space["belief"]["dim"] == 3
        assert restored.preferences["learning_rate"] == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
