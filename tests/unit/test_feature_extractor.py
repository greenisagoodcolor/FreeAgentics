"""
Module for FreeAgentics Active Inference implementation.
"""

import numpy as np
import pytest

from inference.gnn.feature_extractor import (
    ExtractionResult,
    FeatureConfig,
    FeatureType,
    NodeFeatureExtractor,
    NormalizationType,
)


class TestFeatureConfig:
    ."""Test FeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = FeatureConfig(name="test_feature", type=FeatureType.NUMERICAL)
        assert config.name == "test_feature"
        assert config.type == FeatureType.NUMERICAL
        assert config.dimension is None
        assert config.normalization == NormalizationType.STANDARD
        assert config.default_value is None
        assert config.constraints == {}
        assert config.preprocessing_fn is None

    def test_custom_config(self) -> None:
        """Test custom configuration values"""
        config = FeatureConfig(
            name="position",
            type=FeatureType.SPATIAL,
            dimension=3,
            normalization=NormalizationType.MINMAX,
            default_value=0.0,
            constraints={"min": -1, "max": 1},
        )
        assert config.dimension == 3
        assert config.normalization == NormalizationType.MINMAX
        assert config.default_value == 0.0
        assert config.constraints["min"] == -1
        assert config.constraints["max"] == 1


class TestNodeFeatureExtractor:
    ."""Test NodeFeatureExtractor class."""

    def test_initialization(self) -> None:
        """Test extractor initialization"""
        configs = [
            FeatureConfig("x", FeatureType.NUMERICAL),
            FeatureConfig("y", FeatureType.NUMERICAL),
            FeatureConfig("status", FeatureType.CATEGORICAL),
        ]
        extractor = NodeFeatureExtractor(configs)
        assert len(extractor.feature_configs) == 3
        assert "x" in extractor.feature_configs
        assert "y" in extractor.feature_configs
        assert "status" in extractor.feature_configs

    def test_empty_nodes(self) -> None:
        """Test extraction with empty node list"""
        configs = [FeatureConfig("x", FeatureType.NUMERICAL)]
        extractor = NodeFeatureExtractor(configs)
        result = extractor.extract_features([])
        assert result.features.shape == (0,)
        assert len(result.feature_names) == 0
        assert len(result.feature_dims) == 0

    def test_numerical_features(self) -> None:
        """Test numerical feature extraction"""
        configs = [
            FeatureConfig(
                "energy",
                FeatureType.NUMERICAL,
                normalization=NormalizationType.MINMAX,
                constraints={"min": 0, "max": 1},
            )
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [{"energy": 0.5}, {"energy": 0.8}, {"energy": 0.2}]
        result = extractor.extract_features(nodes)
        assert result.features.shape == (3, 1)
        assert result.feature_names == ["energy"]
        assert "energy" in result.feature_dims
        assert np.all(result.features >= 0)
        assert np.all(result.features <= 1)

    def test_spatial_features(self) -> None:
        """Test spatial feature extraction"""
        configs = [
            FeatureConfig(
                "position",
                FeatureType.SPATIAL,
                dimension=2,
                normalization=NormalizationType.NONE,
            )
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [
            {"position": [0.5, 0.3]},
            {"position": [0.7, 0.9]},
            {"position": [0.1, 0.5]},
        ]
        result = extractor.extract_features(nodes)
        assert result.features.shape == (3, 2)
        assert result.feature_names == ["position_0", "position_1"]
        np.testing.assert_array_almost_equal(result.features[0], [0.5, 0.3])
        np.testing.assert_array_almost_equal(result.features[1], [0.7, 0.9])

    def test_temporal_features(self) -> None:
        """Test temporal feature extraction"""
        configs = [
            FeatureConfig(
                "timestamp",
                FeatureType.TEMPORAL,
                normalization=NormalizationType.STANDARD,
            )
        ]
        extractor = NodeFeatureExtractor(configs)
        timestamp1 = 1642000000
        timestamp2 = 1642003600
        nodes = [
            {"timestamp": timestamp1},
            {"timestamp": timestamp2},
            {"timestamp": "2022-01-12T17:00:00"},
        ]
        result = extractor.extract_features(nodes)
        assert result.features.shape == (3, 7)
        expected_names = [
            "timestamp_hour",
            "timestamp_day_of_week",
            "timestamp_day_of_month",
            "timestamp_month",
            "timestamp_year",
            "timestamp_is_weekend",
            "timestamp_timestamp_normalized",
        ]
        assert result.feature_names == expected_names

    def test_categorical_features(self) -> None:
        """Test categorical feature extraction with one-hot encoding"""
        configs = [FeatureConfig("status", FeatureType.CATEGORICAL)]
        extractor = NodeFeatureExtractor(configs)
        nodes = [
            {"status": "active"},
            {"status": "idle"},
            {"status": "active"},
            {"status": "blocked"},
        ]
        result = extractor.extract_features(nodes)
        assert result.features.shape == (4, 3)
        assert len(result.feature_names) == 3
        assert np.sum(result.features[0]) == 1.0
        assert np.sum(result.features, axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_embedding_features(self) -> None:
        """Test embedding feature extraction"""
        configs = [
            FeatureConfig(
                "agent_embedding",
                FeatureType.EMBEDDING,
                dimension=8,
                normalization=NormalizationType.NONE,
            )
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [
            {"agent_embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
            {"agent_embedding": None},
            {"agent_embedding": "agent_123"},
        ]
        result = extractor.extract_features(nodes)
        assert result.features.shape == (3, 8)
        assert len(result.feature_names) == 8
        np.testing.assert_array_almost_equal(
            result.features[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        np.testing.assert_array_equal(result.features[1], np.zeros(8))

    def test_missing_data_handling(self) -> None:
        """Test handling of missing data"""
        configs = [
            FeatureConfig("x", FeatureType.NUMERICAL),
            FeatureConfig("y", FeatureType.NUMERICAL),
            FeatureConfig("status", FeatureType.CATEGORICAL),
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [
            {"x": 1.0, "y": 2.0, "status": "active"},
            {"x": None, "y": 3.0, "status": "idle"},
            {"x": 2.0, "status": "active"},
            {},
        ]
        result = extractor.extract_features(nodes)
        assert result.missing_mask is not None
        assert result.missing_mask.shape == (4, 3)
        assert result.missing_mask[1, 0] == True
        assert result.missing_mask[2, 1] == True
        assert np.all(result.missing_mask[3, :] == True)

    def test_multiple_feature_types(self) -> None:
        """Test extraction with multiple feature types"""
        configs = [
            FeatureConfig("x", FeatureType.NUMERICAL),
            FeatureConfig("y", FeatureType.NUMERICAL),
            FeatureConfig("status", FeatureType.CATEGORICAL),
            FeatureConfig("timestamp", FeatureType.TEMPORAL),
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [
            {"x": 1.0, "y": 2.0, "status": "active", "timestamp": 1642000000},
            {"x": 3.0, "y": 4.0, "status": "idle", "timestamp": 1642003600},
        ]
        result = extractor.extract_features(nodes)
        assert result.features.shape[1] >= 10
        assert "x" in result.feature_dims
        assert "y" in result.feature_dims
        assert "status" in result.feature_dims
        assert "timestamp" in result.feature_dims

    def test_normalization_methods(self) -> None:
        """Test different normalization methods"""
        configs_minmax = [
            FeatureConfig("value", FeatureType.NUMERICAL, normalization=NormalizationType.MINMAX)
        ]
        extractor_minmax = NodeFeatureExtractor(configs_minmax)
        nodes = [{"value": 0}, {"value": 50}, {"value": 100}]
        result_minmax = extractor_minmax.extract_features(nodes)
        assert np.min(result_minmax.features) >= 0
        assert np.max(result_minmax.features) <= 1
        configs_standard = [
            FeatureConfig("value", FeatureType.NUMERICAL, normalization=NormalizationType.STANDARD)
        ]
        extractor_standard = NodeFeatureExtractor(configs_standard)
        result_standard = extractor_standard.extract_features(nodes)
        assert abs(np.mean(result_standard.features)) < 0.1
        assert abs(np.std(result_standard.features) - 1.0) < 0.1

    def test_handle_missing_data_strategies(self) -> None:
        """Test different missing data imputation strategies"""
        configs = [FeatureConfig("value", FeatureType.NUMERICAL)]
        extractor = NodeFeatureExtractor(configs)
        features = np.array([[1.0], [np.nan], [3.0], [np.nan], [5.0]])
        missing_mask = np.array([[False], [True], [False], [True], [False]])
        imputed_mean = extractor.handle_missing_data(features, missing_mask, strategy="mean")
        assert imputed_mean[1, 0] == 3.0
        assert imputed_mean[3, 0] == 3.0
        imputed_median = extractor.handle_missing_data(features, missing_mask, strategy="median")
        assert imputed_median[1, 0] == 3.0
        assert imputed_median[3, 0] == 3.0
        imputed_zero = extractor.handle_missing_data(features, missing_mask, strategy="zero")
        assert imputed_zero[1, 0] == 0.0
        assert imputed_zero[3, 0] == 0.0

    def test_extraction_metadata(self) -> None:
        """Test extraction result metadata"""
        configs = [
            FeatureConfig("x", FeatureType.NUMERICAL),
            FeatureConfig("y", FeatureType.NUMERICAL),
        ]
        extractor = NodeFeatureExtractor(configs)
        nodes = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        result = extractor.extract_features(nodes)
        assert result.metadata["num_nodes"] == 2
        assert result.metadata["num_features"] == 2
        assert "extraction_time" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
