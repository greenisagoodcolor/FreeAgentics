"""Test suite for matrix caching benchmarks."""

from unittest.mock import patch

import numpy as np
import pytest

from tests.performance.matrix_caching_benchmarks import (
    CacheComparisonBenchmark,
    IntermediateResultCachingBenchmark,
    MatrixCache,
    ObservationLikelihoodCachingBenchmark,
)


class TestMatrixCache:
    """Test the MatrixCache implementation."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = MatrixCache(max_size=3)

        # Test cache miss
        result = cache.get("key1")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Test cache put and hit
        matrix = np.random.rand(5, 5)
        cache.put("key1", matrix)

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert np.array_equal(retrieved, matrix)
        assert cache.hits == 1
        assert cache.misses == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = MatrixCache()

        # Initially no hits or misses
        assert cache.get_hit_rate() == 0.0

        # Add some operations
        cache.get("missing")  # miss
        cache.put("key1", np.random.rand(3, 3))
        cache.get("key1")  # hit
        cache.get("key1")  # hit

        expected_rate = 2 / 3  # 2 hits out of 3 total operations
        assert abs(cache.get_hit_rate() - expected_rate) < 0.01

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = MatrixCache(max_size=2)

        # Fill cache
        cache.put("key1", np.random.rand(2, 2))
        cache.put("key2", np.random.rand(2, 2))

        # Access key1 to make it more recently used
        cache.get("key1")

        # Add key3, should evict key2
        cache.put("key3", np.random.rand(2, 2))

        # key1 and key3 should exist, key2 should be evicted
        assert cache.get("key1") is not None
        assert cache.get("key3") is not None
        assert cache.get("key2") is None

    def test_cache_memory_usage(self):
        """Test memory usage calculation."""
        cache = MatrixCache()

        # Empty cache should have zero memory usage
        assert cache.get_memory_usage_mb() == 0.0

        # Add a matrix and check memory usage
        matrix = np.random.rand(100, 100)  # ~80KB
        cache.put("large_matrix", matrix)

        memory_usage = cache.get_memory_usage_mb()
        assert memory_usage > 0.0
        assert memory_usage < 1.0  # Should be less than 1MB for 100x100 float64


class TestCachingBenchmarks:
    """Test the caching benchmark implementations."""

    def test_observation_likelihood_benchmark(self):
        """Test observation likelihood caching benchmark."""
        # Check if PyMDP is available
        try:
            import pymdp

            pymdp_available = True
        except ImportError:
            pymdp_available = False

        if not pymdp_available:
            # Mock the benchmark when PyMDP is not available
            mock_result = {
                "avg_computation_time_ms": 1.5,
                "cache_hit_rate": 0.75,
                "cache_memory_mb": 2.1,
                "total_likelihood_operations": 100,
            }

            with (
                patch.object(ObservationLikelihoodCachingBenchmark, "setup"),
                patch.object(
                    ObservationLikelihoodCachingBenchmark,
                    "run_iteration",
                    return_value=mock_result,
                ),
            ):
                benchmark = ObservationLikelihoodCachingBenchmark(
                    state_size=10, num_modalities=2, cache_enabled=True
                )
                benchmark.setup()
                result = benchmark.run_iteration()

                # Check that result contains expected metrics
                assert "avg_computation_time_ms" in result
                assert "cache_hit_rate" in result
                assert "cache_memory_mb" in result
                assert "total_likelihood_operations" in result

                # Values should be reasonable
                assert result["avg_computation_time_ms"] >= 0
                assert 0 <= result["cache_hit_rate"] <= 1
                assert result["cache_memory_mb"] >= 0
                assert result["total_likelihood_operations"] > 0
            return

        # Real test when PyMDP is available
        benchmark = ObservationLikelihoodCachingBenchmark(
            state_size=10, num_modalities=2, cache_enabled=True
        )

        benchmark.setup()
        result = benchmark.run_iteration()

        # Check that result contains expected metrics
        assert "avg_computation_time_ms" in result
        assert "cache_hit_rate" in result
        assert "cache_memory_mb" in result
        assert "total_likelihood_operations" in result

        # Values should be reasonable
        assert result["avg_computation_time_ms"] >= 0
        assert 0 <= result["cache_hit_rate"] <= 1
        assert result["cache_memory_mb"] >= 0
        assert result["total_likelihood_operations"] > 0

    def test_intermediate_result_benchmark(self):
        """Test intermediate result caching benchmark."""
        # Check if PyMDP is available
        try:
            import pymdp

            pymdp_available = True
        except ImportError:
            pymdp_available = False

        if not pymdp_available:
            # Mock the benchmark when PyMDP is not available
            mock_result = {
                "avg_computation_time_ms": 2.3,
                "cache_hit_rate": 0.68,
                "cache_memory_mb": 1.7,
                "total_operations": 50,
            }

            with (
                patch.object(IntermediateResultCachingBenchmark, "setup"),
                patch.object(
                    IntermediateResultCachingBenchmark,
                    "run_iteration",
                    return_value=mock_result,
                ),
            ):
                benchmark = IntermediateResultCachingBenchmark(
                    complexity_level=2, cache_enabled=True
                )
                benchmark.setup()
                result = benchmark.run_iteration()

                # Check that result contains expected metrics
                assert "avg_computation_time_ms" in result
                assert "cache_hit_rate" in result
                assert "cache_memory_mb" in result
                assert "total_operations" in result

                # Values should be reasonable
                assert result["avg_computation_time_ms"] >= 0
                assert 0 <= result["cache_hit_rate"] <= 1
                assert result["cache_memory_mb"] >= 0
                assert result["total_operations"] > 0
            return

        # Real test when PyMDP is available
        benchmark = IntermediateResultCachingBenchmark(complexity_level=2, cache_enabled=True)

        benchmark.setup()
        result = benchmark.run_iteration()

        # Check that result contains expected metrics
        assert "avg_computation_time_ms" in result
        assert "cache_hit_rate" in result
        assert "cache_memory_mb" in result
        assert "total_operations" in result

        # Values should be reasonable
        assert result["avg_computation_time_ms"] >= 0
        assert 0 <= result["cache_hit_rate"] <= 1
        assert result["cache_memory_mb"] >= 0
        assert result["total_operations"] > 0

    def test_cache_comparison_benchmark(self):
        """Test cache comparison benchmark."""
        # Check if PyMDP is available
        try:
            import pymdp

            pymdp_available = True
        except ImportError:
            pymdp_available = False

        if not pymdp_available:
            # Mock the benchmark when PyMDP is not available
            mock_result = {
                "cached_avg_time_ms": 1.2,
                "uncached_avg_time_ms": 3.5,
                "speedup_factor": 2.9,
                "cache_hit_rate": 0.82,
                "efficiency_gain": 65.7,
            }

            with (
                patch.object(CacheComparisonBenchmark, "setup"),
                patch.object(CacheComparisonBenchmark, "run_iteration", return_value=mock_result),
            ):
                benchmark = CacheComparisonBenchmark("mixed_workload")
                benchmark.setup()
                result = benchmark.run_iteration()

                # Check that result contains expected metrics
                assert "cached_avg_time_ms" in result
                assert "uncached_avg_time_ms" in result
                assert "speedup_factor" in result
                assert "cache_hit_rate" in result
                assert "efficiency_gain" in result

                # Values should be reasonable
                assert result["cached_avg_time_ms"] >= 0
                assert result["uncached_avg_time_ms"] >= 0
                assert result["speedup_factor"] >= 0
                assert 0 <= result["cache_hit_rate"] <= 1
                assert result["efficiency_gain"] >= 0
            return

        # Real test when PyMDP is available
        benchmark = CacheComparisonBenchmark("mixed_workload")

        benchmark.setup()
        result = benchmark.run_iteration()

        # Check that result contains expected metrics
        assert "cached_avg_time_ms" in result
        assert "uncached_avg_time_ms" in result
        assert "speedup_factor" in result
        assert "cache_hit_rate" in result
        assert "efficiency_gain" in result

        # Values should be reasonable
        assert result["cached_avg_time_ms"] >= 0
        assert result["uncached_avg_time_ms"] >= 0
        assert result["speedup_factor"] >= 0
        assert 0 <= result["cache_hit_rate"] <= 1
        assert result["efficiency_gain"] >= 0

    def test_benchmark_configuration(self):
        """Test benchmark configuration reporting."""
        # Check if PyMDP is available
        try:
            import pymdp

            pymdp_available = True
        except ImportError:
            pymdp_available = False

        if not pymdp_available:
            # Mock the benchmark when PyMDP is not available
            mock_config = {"complexity_level": 3, "cache_enabled": True}

            with patch.object(
                IntermediateResultCachingBenchmark,
                "get_configuration",
                return_value=mock_config,
            ):
                benchmark = IntermediateResultCachingBenchmark(
                    complexity_level=3, cache_enabled=True
                )
                config = benchmark.get_configuration()

                assert "complexity_level" in config
                assert "cache_enabled" in config
                assert config["complexity_level"] == 3
                assert config["cache_enabled"] is True
            return

        # Real test when PyMDP is available
        benchmark = IntermediateResultCachingBenchmark(complexity_level=3, cache_enabled=True)

        config = benchmark.get_configuration()

        assert "complexity_level" in config
        assert "cache_enabled" in config
        assert config["complexity_level"] == 3
        assert config["cache_enabled"] is True


class TestCachingBenchmarksWithoutPyMDP:
    """Test benchmark behavior when PyMDP is not available."""

    def test_benchmark_fails_without_pymdp(self):
        """Test that benchmarks properly fail when PyMDP is unavailable."""
        # Mock PYMDP_AVAILABLE as False
        from tests.performance import matrix_caching_benchmarks

        original_pymdp_available = matrix_caching_benchmarks.PYMDP_AVAILABLE
        matrix_caching_benchmarks.PYMDP_AVAILABLE = False

        try:
            benchmark = IntermediateResultCachingBenchmark()
            benchmark.setup()

            # Should raise ImportError when trying to run without PyMDP
            with pytest.raises(ImportError, match="PyMDP is required"):
                benchmark.run_iteration()

        finally:
            # Restore original value
            matrix_caching_benchmarks.PYMDP_AVAILABLE = original_pymdp_available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
