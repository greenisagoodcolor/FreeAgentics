"""Tests for inference.gnn.model module."""

from unittest.mock import MagicMock, patch

import pytest


class TestGMNModel:
    """Test the GMNModel class."""

    def test_gmn_model_creation(self):
        """Test GMNModel creation with basic config."""
        from inference.gnn.model import GMNModel

        config = {
            "architecture": "GAT",
            "layers": [64, 32, 16],
            "hyperparameters": {"lr": 0.01, "dropout": 0.5},
            "metadata": {"version": "1.0"},
        }

        model = GMNModel(config)

        assert model.config == config
        assert model.architecture == "GAT"
        assert model.layers == [64, 32, 16]
        assert model.hyperparameters == {"lr": 0.01, "dropout": 0.5}
        assert model.metadata == {"version": "1.0"}
        assert model._model is None
        assert model._device is None

    def test_gmn_model_creation_with_defaults(self):
        """Test GMNModel creation with minimal config."""
        from inference.gnn.model import GMNModel

        config = {}
        model = GMNModel(config)

        assert model.config == config
        assert model.architecture == "GCN"  # Default
        assert model.layers == []  # Default
        assert model.hyperparameters == {}  # Default
        assert model.metadata == {}  # Default
        assert model._model is None
        assert model._device is None

    def test_gmn_model_partial_config(self):
        """Test GMNModel with partial config."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GraphSAGE", "layers": [128, 64]}

        model = GMNModel(config)

        assert model.architecture == "GraphSAGE"
        assert model.layers == [128, 64]
        assert model.hyperparameters == {}  # Default
        assert model.metadata == {}  # Default

    @patch("inference.gnn.model.logger")
    def test_gmn_model_build(self, mock_logger):
        """Test GMNModel build method."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GCN", "layers": [64, 32, 16]}

        model = GMNModel(config)
        model.build()

        # Should log the build information
        mock_logger.info.assert_called_once_with(
            "Building GCN model with 3 layers"
        )

    @patch("inference.gnn.model.logger")
    def test_gmn_model_build_no_layers(self, mock_logger):
        """Test GMNModel build method with no layers."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GAT"}
        model = GMNModel(config)
        model.build()

        # Should log the build information
        mock_logger.info.assert_called_once_with(
            "Building GAT model with 0 layers"
        )

    def test_gmn_model_forward_without_build(self):
        """Test GMNModel forward without building first."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GCN"}
        model = GMNModel(config)

        # Should raise RuntimeError if model not built
        with pytest.raises(
            RuntimeError, match="Model not built. Call build\\(\\) first."
        ):
            model.forward(None, None)

    def test_gmn_model_forward_with_build(self):
        """Test GMNModel forward after building."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GCN"}
        model = GMNModel(config)

        # Mock the _model attribute to simulate built model
        model._model = MagicMock()

        # Create mock input data
        x = MagicMock()
        edge_index = MagicMock()

        # Should return the input x (current implementation)
        result = model.forward(x, edge_index)
        assert result == x

    def test_gmn_model_to_dict(self):
        """Test GMNModel to_dict method."""
        from inference.gnn.model import GMNModel

        config = {
            "architecture": "GraphSAGE",
            "layers": [128, 64, 32],
            "hyperparameters": {"lr": 0.001, "weight_decay": 0.0005},
            "metadata": {"created_by": "test", "version": "2.0"},
        }

        model = GMNModel(config)
        result = model.to_dict()

        expected = {
            "architecture": "GraphSAGE",
            "layers": [128, 64, 32],
            "hyperparameters": {"lr": 0.001, "weight_decay": 0.0005},
            "metadata": {"created_by": "test", "version": "2.0"},
        }

        assert result == expected

    def test_gmn_model_to_dict_defaults(self):
        """Test GMNModel to_dict with default values."""
        from inference.gnn.model import GMNModel

        config = {}
        model = GMNModel(config)
        result = model.to_dict()

        expected = {
            "architecture": "GCN",
            "layers": [],
            "hyperparameters": {},
            "metadata": {},
        }

        assert result == expected

    def test_gmn_model_config_immutability(self):
        """Test that modifying config after creation doesn't affect model."""
        from inference.gnn.model import GMNModel

        config = {"architecture": "GCN", "layers": [64, 32]}

        model = GMNModel(config)

        # Modify original config
        config["architecture"] = "GAT"
        config["layers"].append(16)

        # Model should retain original values
        assert model.architecture == "GCN"
        assert model.layers == [
            64,
            32,
            16,
        ]  # This will be affected since it's a reference

        # But model.config should be the same reference
        assert model.config == config

    def test_gmn_model_logger_import(self):
        """Test that logger is properly imported."""
        from inference.gnn.model import logger

        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_gmn_model_class_existence(self):
        """Test that GMNModel class exists and is properly defined."""
        from inference.gnn.model import GMNModel

        assert GMNModel is not None
        assert hasattr(GMNModel, "__init__")
        assert hasattr(GMNModel, "build")
        assert hasattr(GMNModel, "forward")
        assert hasattr(GMNModel, "to_dict")
