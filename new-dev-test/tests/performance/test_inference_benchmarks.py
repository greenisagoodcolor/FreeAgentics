"""Test the inference benchmarking functionality."""

import pytest

from tests.performance.inference_benchmarks import (
    BeliefPropagationBenchmark,
    InferenceProfilingBenchmark,
    MessagePassingBenchmark,
    VariationalInferenceBenchmark,
)


def test_variational_inference_benchmark():
    """Test variational inference benchmark runs."""
    benchmark = VariationalInferenceBenchmark([5, 5], num_iterations=5)
    benchmark.setup()

    # Run single iteration
    metrics = benchmark.run_iteration()

    # Should return metrics even if PyMDP not available
    assert isinstance(metrics, dict)

    # Run full benchmark with minimal iterations
    result = benchmark.run(iterations=5, warmup=1)

    assert result.name == "variational_inference_2D"
    assert result.iterations == 5
    assert result.mean_time_ms > 0


def test_belief_propagation_benchmark():
    """Test belief propagation benchmark."""
    benchmark = BeliefPropagationBenchmark(num_nodes=5, connectivity=0.4)
    benchmark.setup()

    metrics = benchmark.run_iteration()
    assert isinstance(metrics, dict)

    result = benchmark.run(iterations=5, warmup=1)
    assert result.name == "belief_propagation"
    assert result.mean_time_ms > 0


def test_message_passing_benchmark():
    """Test message passing with different schedules."""
    for schedule in ["sequential", "parallel", "random"]:
        benchmark = MessagePassingBenchmark(grid_size=3, schedule=schedule)
        benchmark.setup()

        metrics = benchmark.run_iteration()
        assert isinstance(metrics, dict)
        # Schedule is only returned when PyMDP is available
        if metrics:
            assert metrics.get("schedule") == schedule

        result = benchmark.run(iterations=5, warmup=1)
        assert result.name == f"message_passing_{schedule}"
        assert result.mean_time_ms > 0


def test_inference_profiling_benchmark():
    """Test inference profiling benchmark."""
    benchmark = InferenceProfilingBenchmark(state_size=10)
    benchmark.setup()

    metrics = benchmark.run_iteration()
    assert isinstance(metrics, dict)

    # Should have timing breakdowns
    if metrics:  # Only if PyMDP available
        for key in [
            "state_inference_ms",
            "policy_inference_ms",
            "action_selection_ms",
        ]:
            if key in metrics:
                assert metrics[key] >= 0

    result = benchmark.run(iterations=5, warmup=1)
    assert result.name == "inference_profiling"
    assert result.mean_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
