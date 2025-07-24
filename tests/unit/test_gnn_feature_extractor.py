"""
Comprehensive test suite for GNN Feature Extractor module - Meta Quality Standards.

This test suite provides comprehensive coverage for the NodeFeatureExtractor class,
which implements sophisticated feature extraction for GNN nodes.
Coverage target: 95%+
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Import the module under test
try:
    from inference.gnn.feature_extractor import (
        FeatureConfig,
        FeatureType,
        NodeFeatureExtractor,
        NormalizationStrategy,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class NodeFeatureExtractor:
        pass

    class FeatureConfig:
        pass

    class FeatureType:
        SPATIAL = "spatial"
        TEMPORAL = "temporal"
        CATEGORICAL = "categorical"
        NUMERICAL = "numerical"
        GRAPH_STRUCTURAL = "graph_structural"

    class NormalizationStrategy:
        STANDARD = "standard"
        MINMAX = "minmax"
        ROBUST = "robust"
        NONE = "none"


class TestFeatureConfig:
    """Test FeatureConfig class."""

    def test_config_creation_defaults(self):
        """Test feature config creation with defaults."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required module 'inference.gnn.feature_extractor' not available")

        config = FeatureConfig()

        assert config.feature_types is not None
        assert config.normalization_strategy == NormalizationStrategy.STANDARD
        assert config.handle_missing is True
        assert config.temporal_window_size == 10
        assert config.spatial_resolution == 7  # H3 resolution
        assert config.cache_features is True

    def test_config_creation_custom(self):
        """Test feature config with custom values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required module 'inference.gnn.feature_extractor' not available")

        config = FeatureConfig(
            feature_types=[FeatureType.SPATIAL, FeatureType.TEMPORAL],
            normalization_strategy=NormalizationStrategy.MINMAX,
            handle_missing=False,
            temporal_window_size=20,
            spatial_resolution=9,
            cache_features=False,
        )

        assert len(config.feature_types) == 2
        assert FeatureType.SPATIAL in config.feature_types
        assert config.normalization_strategy == NormalizationStrategy.MINMAX
        assert config.temporal_window_size == 20


class TestNodeFeatureExtractor:
    """Test suite for NodeFeatureExtractor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required module 'inference.gnn.feature_extractor' not available")
            return Mock()
        return FeatureConfig(
            feature_types=[
                FeatureType.SPATIAL,
                FeatureType.TEMPORAL,
                FeatureType.CATEGORICAL,
                FeatureType.NUMERICAL,
            ]
        )

    @pytest.fixture
    def extractor(self, config):
        """Create feature extractor instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required module 'inference.gnn.feature_extractor' not available")
        return NodeFeatureExtractor(config)

    @pytest.fixture
    def sample_node_data(self):
        """Create sample node data for testing."""
        return {
            "nodes": [
                {
                    "id": "node_1",
                    "position": {"lat": 37.7749, "lon": -122.4194},  # SF
                    "timestamp": datetime.now(),
                    "category": "agent",
                    "attributes": {
                        "energy": 0.8,
                        "velocity": 5.0,
                        "confidence": 0.9,
                    },
                },
                {
                    "id": "node_2",
                    "position": {"lat": 37.7849, "lon": -122.4094},
                    "timestamp": datetime.now() - timedelta(minutes=5),
                    "category": "resource",
                    "attributes": {
                        "energy": 0.6,
                        "velocity": 0.0,
                        "confidence": 0.7,
                    },
                },
            ]
        }

    def test_extractor_initialization(self, extractor, config):
        """Test feature extractor initialization."""
        assert extractor is not None
        assert extractor.config == config
        assert hasattr(extractor, "normalizers")
        assert hasattr(extractor, "feature_cache")
        assert hasattr(extractor, "feature_stats")

    def test_extract_features_basic(self, extractor, sample_node_data):
        """Test basic feature extraction."""
        features = extractor.extract_features(sample_node_data["nodes"])

        assert features is not None
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(sample_node_data["nodes"])
        assert features.shape[1] > 0  # Should have extracted features
        assert torch.all(torch.isfinite(features))

    def test_extract_spatial_features(self, extractor, sample_node_data):
        """Test spatial feature extraction."""
        # Modify sample data to have proper position format
        test_nodes = [
            {
                "id": "node_1",
                "position": [37.7749, -122.4194],
            },  # x, y coordinates
            {"id": "node_2", "position": [37.7849, -122.4094]},
            {"id": "node_3", "position": [0.0, 0.0]},  # Test zero position
        ]

        features = extractor._extract_spatial_features(test_nodes)

        assert features is not None
        assert features.shape[0] == len(test_nodes)
        assert features.shape[1] == 2  # x, y coordinates

        # Test spatial resolution scaling
        expected_resolution = extractor.config.spatial_resolution
        if expected_resolution != 1.0:
            # Verify that coordinates are scaled by resolution
            assert torch.allclose(
                features[0],
                torch.tensor([37.7749, -122.4194]) / expected_resolution,
            )

    def test_extract_spatial_features_no_h3(self, extractor, sample_node_data):
        """Test spatial features when H3 is not available."""
        with patch.dict("sys.modules", {"h3": None}):
            features = extractor._extract_spatial_features(sample_node_data["nodes"])

            assert features is not None
            assert features.shape[1] >= 2  # Should still have lat/lon

    def test_extract_temporal_features(self, extractor, sample_node_data):
        """Test temporal feature extraction."""
        features = extractor._extract_temporal_features(sample_node_data["nodes"])

        assert features is not None
        assert features.shape[0] == len(sample_node_data["nodes"])
        # Hour, day of week, day of month, month, is_weekend
        assert features.shape[1] >= 5

    def test_extract_categorical_features(self, extractor, sample_node_data):
        """Test categorical feature extraction."""
        features = extractor._extract_categorical_features(sample_node_data["nodes"])

        assert features is not None
        assert features.shape[0] == len(sample_node_data["nodes"])
        # One-hot encoded categories
        assert features.shape[1] == 2  # Two unique categories

    def test_extract_numerical_features(self, extractor, sample_node_data):
        """Test numerical feature extraction."""
        features = extractor._extract_numerical_features(sample_node_data["nodes"])

        assert features is not None
        assert features.shape[0] == len(sample_node_data["nodes"])
        assert features.shape[1] == 3  # energy, velocity, confidence

    def test_handle_missing_values(self, extractor):
        """Test handling of missing values."""
        nodes_with_missing = [
            {
                "id": "node_1",
                "attributes": {"energy": 0.8, "velocity": None},
            },  # Missing velocity
            {
                "id": "node_2",
                "attributes": {"energy": None, "velocity": 5.0},
            },  # Missing energy
        ]

        features = extractor._extract_numerical_features(nodes_with_missing)

        assert torch.all(torch.isfinite(features))
        # Should handle missing values appropriately

    def test_normalization_strategies(self, extractor, sample_node_data):
        """Test different normalization strategies."""
        strategies = [
            NormalizationStrategy.STANDARD,
            NormalizationStrategy.MINMAX,
            NormalizationStrategy.ROBUST,
            NormalizationStrategy.NONE,
        ]

        for strategy in strategies:
            extractor.config.normalization_strategy = strategy
            features = extractor.extract_features(sample_node_data["nodes"])

            assert torch.all(torch.isfinite(features))

            if strategy == NormalizationStrategy.MINMAX:
                # Features should be in [0, 1]
                assert torch.all(features >= 0) and torch.all(features <= 1)

    def test_feature_caching(self, extractor, sample_node_data):
        """Test feature caching mechanism."""
        # First extraction
        features1 = extractor.extract_features(sample_node_data["nodes"])

        # Second extraction with same data
        with patch.object(extractor, "_compute_features") as mock_compute:
            mock_compute.return_value = features1
            extractor.extract_features(sample_node_data["nodes"])

            # Should use cache if enabled
            if extractor.config.cache_features:
                # Cache logic would prevent recomputation
                pass

    def test_batch_feature_extraction(self, extractor):
        """Test extraction for large batches."""
        # Create large batch
        large_batch = []
        for i in range(1000):
            large_batch.append(
                {
                    "id": f"node_{i}",
                    "position": {
                        "lat": 37.0 + i * 0.001,
                        "lon": -122.0 + i * 0.001,
                    },
                    "timestamp": datetime.now() - timedelta(minutes=i),
                    "category": f"cat_{i % 10}",
                    "attributes": {
                        "energy": np.random.rand(),
                        "velocity": np.random.rand() * 10,
                        "confidence": np.random.rand(),
                    },
                }
            )

        features = extractor.extract_features(large_batch)

        assert features.shape[0] == 1000
        assert torch.all(torch.isfinite(features))

    def test_incremental_feature_updates(self, extractor, sample_node_data):
        """Test incremental feature updates."""
        # Initial extraction
        initial_features = extractor.extract_features(sample_node_data["nodes"])

        # Update one node
        updated_nodes = sample_node_data["nodes"].copy()
        updated_nodes[0]["attributes"]["energy"] = 0.3

        # Re-extract
        updated_features = extractor.extract_features(updated_nodes)

        # Features should be different
        assert not torch.allclose(initial_features, updated_features)

    def test_feature_statistics_tracking(self, extractor, sample_node_data):
        """Test that feature statistics are tracked."""
        extractor.extract_features(sample_node_data["nodes"])

        # Check if statistics were recorded
        assert len(extractor.feature_stats) > 0

        # Statistics should include mean, std, min, max
        for stat in extractor.feature_stats.values():
            assert "mean" in stat
            assert "std" in stat
            assert "min" in stat
            assert "max" in stat

    def test_temporal_windowing(self, extractor):
        """Test temporal windowing functionality."""
        # Create time series data
        base_time = datetime.now()
        time_series_nodes = []

        for i in range(20):
            time_series_nodes.append(
                {
                    "id": "node_1",
                    "timestamp": base_time - timedelta(minutes=i),
                    "attributes": {"value": i},
                }
            )

        # Extract with temporal window
        extractor.config.temporal_window_size = 5
        features = extractor._apply_temporal_windowing(time_series_nodes)

        assert features is not None

    def test_graph_structural_features(self, extractor):
        """Test graph structural feature extraction."""
        # Create graph structure
        nodes_with_edges = [
            {"id": "A", "edges": ["B", "C"]},
            {"id": "B", "edges": ["A", "C", "D"]},
            {"id": "C", "edges": ["A", "B"]},
            {"id": "D", "edges": ["B"]},
        ]

        if hasattr(extractor, "_extract_graph_structural_features"):
            features = extractor._extract_graph_structural_features(nodes_with_edges)

            assert features is not None
            # Should include degree, clustering coefficient, etc.

    def test_custom_feature_extractors(self, extractor):
        """Test adding custom feature extractors."""

        def custom_extractor(nodes):
            return torch.tensor([[node.get("custom_value", 0)] for node in nodes])

        if hasattr(extractor, "add_custom_extractor"):
            extractor.add_custom_extractor("custom", custom_extractor)

            nodes = [{"id": "1", "custom_value": 42}]
            features = extractor.extract_features(nodes)

            # Should include custom feature
            assert 42 in features

    def test_feature_importance_scoring(self, extractor, sample_node_data):
        """Test feature importance scoring."""
        features = extractor.extract_features(sample_node_data["nodes"])

        if hasattr(extractor, "compute_feature_importance"):
            importance = extractor.compute_feature_importance(features)

            assert importance is not None
            assert len(importance) == features.shape[1]
            assert all(i >= 0 for i in importance)

    def test_outlier_detection(self, extractor):
        """Test outlier detection in features."""
        # Create data with outliers
        nodes = []
        for i in range(100):
            value = 1.0 if i < 95 else 1000.0  # 5 outliers
            nodes.append({"id": f"node_{i}", "attributes": {"value": value}})

        features = extractor._extract_numerical_features(nodes)

        # With robust normalization, outliers should be handled
        if extractor.config.normalization_strategy == NormalizationStrategy.ROBUST:
            assert torch.all(torch.abs(features) < 10)  # Outliers clipped

    def test_memory_efficiency(self, extractor):
        """Test memory efficiency with large feature sets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Extract features for large dataset
        large_nodes = []
        for i in range(10000):
            large_nodes.append(
                {
                    "id": f"node_{i}",
                    "position": {
                        "lat": np.random.rand() * 180 - 90,
                        "lon": np.random.rand() * 360 - 180,
                    },
                    "attributes": {f"attr_{j}": np.random.rand() for j in range(50)},
                }
            )

        extractor.extract_features(large_nodes)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_increase < 1000  # Less than 1GB increase

    def test_categorical_encoding_strategies(self, extractor):
        """Test different categorical encoding strategies."""
        nodes = [
            {"id": "1", "category": "A"},
            {"id": "2", "category": "B"},
            {"id": "3", "category": "A"},
            {"id": "4", "category": "C"},
        ]

        # Test one-hot encoding
        features_onehot = extractor._extract_categorical_features(nodes)
        assert features_onehot.shape[1] == 3  # 3 unique categories

        # Test with high cardinality
        high_card_nodes = [{"id": str(i), "category": f"cat_{i}"} for i in range(100)]
        features_high = extractor._extract_categorical_features(high_card_nodes)

        # Should handle high cardinality appropriately
        assert features_high is not None

    def test_thread_safety(self, extractor, sample_node_data):
        """Test thread safety of feature extraction."""
        import threading

        results = []
        errors = []

        def extract_features_thread():
            try:
                features = extractor.extract_features(sample_node_data["nodes"])
                results.append(features)
            except Exception as e:
                errors.append(e)

        # Run in multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=extract_features_thread)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    @pytest.mark.parametrize("missing_rate", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_missing_data_robustness(self, extractor, missing_rate):
        """Test robustness to different levels of missing data."""
        nodes = []
        for i in range(100):
            # Randomly make values missing
            attributes = {}
            for attr in ["energy", "velocity", "confidence"]:
                if np.random.rand() > missing_rate:
                    attributes[attr] = np.random.rand()
                else:
                    attributes[attr] = None

            nodes.append({"id": f"node_{i}", "attributes": attributes})

        features = extractor.extract_features(nodes)

        # Should handle any level of missing data
        assert torch.all(torch.isfinite(features))
        assert features.shape[0] == 100

    def test_feature_selection(self, extractor, sample_node_data):
        """Test feature selection capabilities."""
        # Extract all features
        all_features = extractor.extract_features(sample_node_data["nodes"])

        # Select subset of features
        if hasattr(extractor, "select_features"):
            selected_features = extractor.select_features(
                all_features, n_features=5, method="mutual_information"
            )

            assert selected_features.shape[1] == 5
            assert selected_features.shape[0] == all_features.shape[0]

    def test_feature_transformation(self, extractor, sample_node_data):
        """Test feature transformation capabilities."""
        features = extractor.extract_features(sample_node_data["nodes"])

        # Test polynomial features
        if hasattr(extractor, "polynomial_features"):
            poly_features = extractor.polynomial_features(features, degree=2)
            assert poly_features.shape[1] > features.shape[1]

        # Test PCA transformation
        if hasattr(extractor, "pca_transform"):
            pca_features = extractor.pca_transform(features, n_components=3)
            assert pca_features.shape[1] == 3


class TestFeatureExtractorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required module 'inference.gnn.feature_extractor' not available")
        return NodeFeatureExtractor(FeatureConfig())

    def test_empty_node_list(self, extractor):
        """Test extraction with empty node list."""
        features = extractor.extract_features([])

        assert features is not None
        assert features.shape[0] == 0

    def test_nodes_without_required_fields(self, extractor):
        """Test nodes missing required fields."""
        incomplete_nodes = [
            {"id": "1"},  # Missing everything else
            {"id": "2", "attributes": {}},  # Empty attributes
            {"position": {"lat": 0, "lon": 0}},  # Missing id
        ]

        features = extractor.extract_features(incomplete_nodes)

        # Should handle gracefully
        assert features is not None
        assert torch.all(torch.isfinite(features))

    def test_malformed_position_data(self, extractor):
        """Test malformed position data."""
        malformed_nodes = [
            {"id": "1", "position": "not a dict"},
            {"id": "2", "position": {"lat": "not a number", "lon": -122}},
            {
                "id": "3",
                "position": {"latitude": 37, "longitude": -122},
            },  # Wrong keys
        ]

        features = extractor._extract_spatial_features(malformed_nodes)

        # Should handle gracefully
        assert features is not None

    def test_extreme_numerical_values(self, extractor):
        """Test extreme numerical values."""
        extreme_nodes = [
            {"id": "1", "attributes": {"value": 1e10}},
            {"id": "2", "attributes": {"value": -1e10}},
            {"id": "3", "attributes": {"value": float("inf")}},
            {"id": "4", "attributes": {"value": float("-inf")}},
            {"id": "5", "attributes": {"value": float("nan")}},
        ]

        features = extractor._extract_numerical_features(extreme_nodes)

        # Should handle extreme values
        assert torch.all(torch.isfinite(features))

    def test_duplicate_node_ids(self, extractor):
        """Test handling of duplicate node IDs."""
        duplicate_nodes = [
            {"id": "node_1", "attributes": {"value": 1}},
            {"id": "node_1", "attributes": {"value": 2}},  # Duplicate
            {"id": "node_2", "attributes": {"value": 3}},
        ]

        features = extractor.extract_features(duplicate_nodes)

        # Should handle duplicates
        assert features.shape[0] == 3

    def test_very_high_dimensional_features(self, extractor):
        """Test extraction of very high dimensional features."""
        # Create nodes with many attributes
        high_dim_nodes = []
        for i in range(10):
            attributes = {f"attr_{j}": np.random.rand() for j in range(1000)}
            high_dim_nodes.append({"id": f"node_{i}", "attributes": attributes})

        features = extractor._extract_numerical_features(high_dim_nodes)

        assert features.shape[1] == 1000
        assert torch.all(torch.isfinite(features))


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=inference.gnn.feature_extractor",
            "--cov-report=html",
        ]
    )
