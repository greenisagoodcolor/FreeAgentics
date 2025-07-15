#!/usr/bin/env python3
"""Demonstration of matrix operation memory pooling.

This example shows how to use the matrix pooling system to optimize
memory usage in PyMDP operations.

Based on Task 5.4: Create matrix operation memory pooling
"""

import time

import numpy as np
import psutil

from agents.memory_optimization.matrix_pooling import (
    MatrixOperationPool,
    get_global_pool,
    pooled_dot,
    pooled_einsum,
    pooled_matmul,
    pooled_matrix,
)

# Try to import PyMDP
try:
    pass

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    print("PyMDP not available - will demonstrate with numpy operations only")


def print_memory_usage(label: str):
    """Print current memory usage."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {mem_mb:.1f} MB")


def demonstrate_basic_pooling():
    """Demonstrate basic matrix pooling operations."""
    print("\n=== Basic Matrix Pooling Demo ===")

    pool = get_global_pool()

    # 1. Simple matrix allocation
    print("\n1. Allocating matrices from pool:")
    with pooled_matrix((100, 100), np.float32) as matrix:
        print(f"   Allocated matrix shape: {matrix.shape}")
        matrix[0, 0] = 42.0
        print(f"   Modified matrix[0,0] = {matrix[0, 0]}")

    print("   Matrix returned to pool")

    # 2. Multiple allocations
    print("\n2. Multiple allocations:")
    matrices = []
    for i in range(5):
        with pooled_matrix((50, 50), np.float32) as m:
            m[0, 0] = i
            matrices.append(m.copy())

    print(f"   Allocated and released 5 matrices")

    # 3. Check pool statistics
    stats = pool.get_statistics()
    print("\n3. Pool Statistics:")
    print(f"   Total pools created: {stats['global']['total_pools']}")
    print(f"   Total matrices allocated: {stats['global']['total_matrices']}")
    print(f"   Total memory used: {stats['global']['total_memory_mb']:.2f} MB")

    for pool_key, pool_stats in stats["pools"].items():
        print(f"   Pool {pool_key}:")
        print(f"     - Hit rate: {pool_stats['hit_rate']:.1%}")
        print(f"     - Available: {pool_stats['stats']['current_available']}")
        print(f"     - In use: {pool_stats['stats']['current_in_use']}")


def demonstrate_matrix_operations():
    """Demonstrate pooled matrix operations."""
    print("\n=== Pooled Matrix Operations Demo ===")

    # Create test matrices
    a = np.random.rand(200, 300).astype(np.float32)
    b = np.random.rand(300, 400).astype(np.float32)
    c = np.random.rand(400, 200).astype(np.float32)

    print(f"\nMatrix shapes: A={a.shape}, B={b.shape}, C={c.shape}")

    # 1. Pooled dot product
    print("\n1. Pooled dot product:")
    start = time.time()
    result1 = pooled_dot(a, b)
    pool_time = time.time() - start

    start = time.time()
    result2 = np.dot(a, b)
    numpy_time = time.time() - start

    print(f"   Pooled dot time: {pool_time*1000:.2f} ms")
    print(f"   Numpy dot time: {numpy_time*1000:.2f} ms")
    print(f"   Results match: {np.allclose(result1, result2)}")

    # 2. Pooled matmul
    print("\n2. Pooled matrix multiplication:")
    result1 = pooled_matmul(b, c)
    result2 = np.matmul(b, c)
    print(f"   Results match: {np.allclose(result1, result2)}")

    # 3. Pooled einsum
    print("\n3. Pooled einsum (ABC chain):")
    start = time.time()
    result1 = pooled_einsum("ij,jk,ki->i", a, b, c)
    pool_time = time.time() - start

    start = time.time()
    result2 = np.einsum("ij,jk,ki->i", a, b, c)
    numpy_time = time.time() - start

    print(f"   Pooled einsum time: {pool_time*1000:.2f} ms")
    print(f"   Numpy einsum time: {numpy_time*1000:.2f} ms")
    print(f"   Results match: {np.allclose(result1, result2)}")


def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency gains."""
    print("\n=== Memory Efficiency Demo ===")

    print_memory_usage("Initial memory")

    # 1. Without pooling - repeated allocations
    print("\n1. Without pooling (100 iterations):")
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024

    for i in range(100):
        a = np.random.rand(500, 500).astype(np.float32)
        b = np.random.rand(500, 500).astype(np.float32)
        np.dot(a, b)

    print_memory_usage("   After non-pooled operations")
    no_pool_mem_increase = psutil.Process().memory_info().rss / 1024 / 1024 - start_mem

    # 2. With pooling
    print("\n2. With pooling (100 iterations):")
    pool = get_global_pool()
    pool.clear_all()  # Start fresh

    start_mem = psutil.Process().memory_info().rss / 1024 / 1024

    for i in range(100):
        a = np.random.rand(500, 500).astype(np.float32)
        b = np.random.rand(500, 500).astype(np.float32)
        pooled_dot(a, b)

    print_memory_usage("   After pooled operations")
    pool_mem_increase = psutil.Process().memory_info().rss / 1024 / 1024 - start_mem

    print(f"\nMemory increase comparison:")
    print(f"   Without pooling: +{no_pool_mem_increase:.1f} MB")
    print(f"   With pooling: +{pool_mem_increase:.1f} MB")
    print(f"   Savings: {no_pool_mem_increase - pool_mem_increase:.1f} MB")

    # Show pool statistics
    stats = pool.get_statistics()
    print(f"\nPool reuse statistics:")
    for pool_key, pool_stats in stats["pools"].items():
        if pool_stats["stats"]["total_requests"] > 0:
            print(f"   {pool_key}: {pool_stats['hit_rate']:.1%} hit rate")


def demonstrate_pymdp_integration():
    """Demonstrate integration with PyMDP operations."""
    if not PYMDP_AVAILABLE:
        print("\n=== PyMDP Integration Demo (Simulated) ===")
        print("PyMDP not available - showing simulated belief updates")
    else:
        print("\n=== PyMDP Integration Demo ===")

    # Simulate PyMDP-style belief updates
    num_states = 25
    num_obs = 5
    num_actions = 4

    # Create observation model (A matrix)
    A = np.random.rand(num_obs, num_states).astype(np.float32)
    A = A / A.sum(axis=0, keepdims=True)

    # Create transition model (B matrix)
    B = np.random.rand(num_states, num_states, num_actions).astype(np.float32)
    B = B / B.sum(axis=0, keepdims=True)

    # Initial belief
    belief = np.ones(num_states, dtype=np.float32) / num_states

    print(f"\nSimulating belief updates:")
    print(f"   State space: {num_states} states")
    print(f"   Observations: {num_obs}")
    print(f"   Actions: {num_actions}")

    # Run belief updates with pooling
    pool = get_global_pool()
    initial_stats = pool.get_statistics()

    num_steps = 50
    start = time.time()

    for step in range(num_steps):
        # Get random observation
        obs = np.random.randint(0, num_obs)

        # Belief update with pooled operations
        with pooled_matrix((num_states,), np.float32) as posterior:
            # P(s|o) ∝ P(o|s) * P(s)
            likelihood = A[obs, :]
            np.multiply(likelihood, belief, out=posterior)
            posterior /= posterior.sum()
            belief = posterior.copy()

        # Predict next belief for random action
        action = np.random.randint(0, num_actions)
        belief = pooled_dot(B[:, :, action], belief)

    elapsed = time.time() - start

    print(f"\nCompleted {num_steps} belief updates in {elapsed*1000:.1f} ms")
    print(f"Average time per update: {elapsed/num_steps*1000:.2f} ms")

    # Show pool efficiency
    final_stats = pool.get_statistics()

    print("\nPool efficiency for belief updates:")
    for pool_key, pool_stats in final_stats["pools"].items():
        initial_requests = (
            initial_stats["pools"].get(pool_key, {}).get("stats", {}).get("total_requests", 0)
        )
        final_requests = pool_stats["stats"]["total_requests"]

        if final_requests > initial_requests:
            print(f"   {pool_key}:")
            print(f"     - New requests: {final_requests - initial_requests}")
            print(f"     - Hit rate: {pool_stats['hit_rate']:.1%}")


def demonstrate_advanced_pooling():
    """Demonstrate advanced pooling strategies."""
    print("\n=== Advanced Pooling Strategies Demo ===")

    pool = MatrixOperationPool(enable_profiling=True)

    # 1. Einsum with multiple operands
    print("\n1. Complex einsum with pooled intermediates:")

    # Tensor contraction for policy optimization
    batch_size = 10
    state_dim = 20
    action_dim = 4
    hidden_dim = 32

    # Create tensors
    states = np.random.rand(batch_size, state_dim).astype(np.float32)
    W1 = np.random.rand(state_dim, hidden_dim).astype(np.float32)
    W2 = np.random.rand(hidden_dim, action_dim).astype(np.float32)

    # Manual computation with pooled intermediates
    with pool.allocate_matrix((batch_size, hidden_dim), np.float32) as hidden:
        # hidden = states @ W1
        np.dot(states, W1, out=hidden)

        with pool.allocate_matrix((batch_size, action_dim), np.float32) as output:
            # output = hidden @ W2
            np.dot(hidden, W2, out=output)
            result = output.copy()

    # Verify with direct computation
    expected = states @ W1 @ W2
    print(f"   Results match: {np.allclose(result, expected)}")

    # 2. Batch operations with shared pools
    print("\n2. Batch operations with shared pools:")

    # Process multiple belief updates in parallel
    num_agents = 5
    beliefs = [np.ones(state_dim) / state_dim for _ in range(num_agents)]

    # Process each agent's belief update
    for i, belief in enumerate(beliefs):
        # Get a transition matrix
        transition = np.random.rand(state_dim, state_dim).astype(np.float32)
        transition /= transition.sum(axis=0, keepdims=True)

        # Update belief using pooled operation
        beliefs[i] = pooled_dot(transition, belief)

    print(f"   Updated {num_agents} agent beliefs with shared pools")

    # Show profiling results
    stats = pool.get_statistics()
    print("\n3. Operation profiling:")
    for op, count in stats["global"]["operation_counts"].items():
        print(f"   {op}: {count} calls")


def main():
    """Run all demonstrations."""
    print("Matrix Operation Memory Pooling Demonstration")
    print("=" * 50)

    # Run demos
    demonstrate_basic_pooling()
    demonstrate_matrix_operations()
    demonstrate_memory_efficiency()
    demonstrate_pymdp_integration()
    demonstrate_advanced_pooling()

    # Final statistics
    pool = get_global_pool()
    stats = pool.get_statistics()

    print("\n" + "=" * 50)
    print("Final Global Pool Statistics:")
    print(f"Total pools created: {stats['global']['total_pools']}")
    print(f"Total matrices in pools: {stats['global']['total_matrices']}")
    print(f"Total memory allocated: {stats['global']['total_memory_mb']:.2f} MB")

    print("\nPool utilization:")
    for pool_key, pool_stats in stats["pools"].items():
        shape = pool_stats["shape"]
        dtype = pool_stats["dtype"]
        requests = pool_stats["stats"]["total_requests"]

        if requests > 0:
            print(f"  {shape} ({dtype}):")
            print(f"    - Requests: {requests}")
            print(f"    - Hit rate: {pool_stats['hit_rate']:.1%}")
            print(f"    - Memory: {pool_stats['stats']['total_memory_bytes'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
