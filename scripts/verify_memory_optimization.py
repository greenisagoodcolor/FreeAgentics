#!/usr/bin/env python3
"""Verify memory optimization is working correctly."""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.memory_optimization.agent_memory_optimizer import (
    AgentMemoryOptimizer,
    OptimizedAgentMemory,
)
from agents.memory_optimization.efficient_structures import LazyBeliefArray


def verify_memory_reduction():
    """Verify that memory optimization actually reduces memory usage."""

    print("=== Memory Optimization Verification ===\n")

    # Create a mock agent with large memory footprint
    class MockAgent:
        def __init__(self):
            self.id = "test_agent"
            self.position = np.array([0.0, 0.0], dtype=np.float32)
            self.active = True
            # Large belief state (~30MB)
            self.beliefs = np.random.rand(2000, 4000).astype(np.float32)
            # Action history (~2MB)
            self.action_history = [f"action_{i}" * 100 for i in range(2000)]
            # Observation buffer (~1.5MB)
            self.observations = np.random.rand(500, 1000).astype(np.float32)
            # Transition matrix (~1MB)
            self.transition_matrix = np.eye(1000, dtype=np.float32)

    # Calculate original memory
    agent = MockAgent()
    original_beliefs_mb = agent.beliefs.nbytes / (1024 * 1024)
    original_history_mb = sum(len(s.encode()) for s in agent.action_history) / (
        1024 * 1024
    )
    original_obs_mb = agent.observations.nbytes / (1024 * 1024)
    original_trans_mb = agent.transition_matrix.nbytes / (1024 * 1024)
    original_total = (
        original_beliefs_mb + original_history_mb + original_obs_mb + original_trans_mb
    )

    print("Original Memory Usage:")
    print(f"  Beliefs: {original_beliefs_mb:.2f} MB")
    print(f"  History: {original_history_mb:.2f} MB")
    print(f"  Observations: {original_obs_mb:.2f} MB")
    print(f"  Transition Matrix: {original_trans_mb:.2f} MB")
    print(f"  Total: {original_total:.2f} MB\n")

    # Test LazyBeliefArray directly
    print("Testing LazyBeliefArray:")
    lazy_beliefs = LazyBeliefArray(
        shape=agent.beliefs.shape, dtype=np.float32, sparsity_threshold=0.9
    )
    lazy_beliefs.update(agent.beliefs)

    # Force conversion to sparse
    sparse_repr = lazy_beliefs.sparse
    print(f"  Original dense size: {agent.beliefs.nbytes / (1024 * 1024):.2f} MB")
    print(f"  Sparse data size: {sparse_repr.data.nbytes / (1024 * 1024):.2f} MB")
    print(f"  Sparse indices size: {sparse_repr.indices.nbytes / (1024 * 1024):.2f} MB")
    if hasattr(sparse_repr, "indptr"):
        print(
            f"  Sparse indptr size: {sparse_repr.indptr.nbytes / (1024 * 1024):.2f} MB"
        )

    total_sparse_mb = (
        sparse_repr.data.nbytes
        + sparse_repr.indices.nbytes
        + (sparse_repr.indptr.nbytes if hasattr(sparse_repr, "indptr") else 0)
    ) / (1024 * 1024)
    print(f"  Total sparse size: {total_sparse_mb:.2f} MB")
    print(
        f"  Compression ratio: {agent.beliefs.nbytes / (sparse_repr.data.nbytes + sparse_repr.indices.nbytes):.2f}x\n"
    )

    # Create optimized memory structure directly
    print("Creating OptimizedAgentMemory directly:")
    opt_memory = OptimizedAgentMemory(
        agent_id=agent.id, position=agent.position.copy(), active=agent.active
    )

    # Set up lazy beliefs
    opt_memory._beliefs = LazyBeliefArray(
        shape=agent.beliefs.shape, dtype=np.float32, sparsity_threshold=0.9
    )
    opt_memory._beliefs.update(agent.beliefs)
    _ = opt_memory._beliefs.sparse  # Force sparse conversion
    opt_memory._beliefs._dense_array = None  # Clear dense array

    # Calculate optimized memory
    optimized_mb = opt_memory.get_memory_usage_mb()
    print(f"  Optimized memory usage: {optimized_mb:.2f} MB")

    # Test with optimizer
    print("\nTesting with AgentMemoryOptimizer:")
    optimizer = AgentMemoryOptimizer(target_memory_per_agent_mb=10.0)

    # Create fresh agent
    agent2 = MockAgent()
    agent2.id = "test_agent_2"

    # Optimize
    opt_result = optimizer.optimize_agent(agent2)
    final_memory = opt_result.get_memory_usage_mb()

    print(f"  Final optimized memory: {final_memory:.2f} MB")
    print(
        f"  Reduction: {((original_total - final_memory) / original_total * 100):.1f}%"
    )
    print(f"  Target achieved: {'YES' if final_memory < 10.0 else 'NO'}")

    # Get optimization stats
    stats = optimizer.get_optimization_stats()
    print("\nOptimization Statistics:")
    print(f"  Agents optimized: {stats['agents_optimized']}")
    print(f"  Mean memory: {stats['actual_memory_mb']['mean']:.2f} MB")
    print(f"  Min memory: {stats['actual_memory_mb']['min']:.2f} MB")
    print(f"  Max memory: {stats['actual_memory_mb']['max']:.2f} MB")


if __name__ == "__main__":
    verify_memory_reduction()
