#!/usr/bin/env python3
"""Demonstration of Memory-Efficient Data Structures.

This demo shows the functionality implemented for Task 5.6:
Implement memory-efficient data structures.

The demo showcases:
1. Lazy belief arrays with sparse storage
2. Memory-mapped buffers for large data
3. Compact action history with compression
4. Efficient temporal sequences with delta compression
5. Compact knowledge graphs with sparse features
6. Performance comparisons and memory usage analysis
"""

import logging
import time
from pathlib import Path

import numpy as np

from agents.memory_optimization.efficient_structures import (
    CompactActionHistory,
    CompactKnowledgeGraph,
    EfficientTemporalSequence,
    LazyBeliefArray,
    MemoryMappedBuffer,
    benchmark_data_structures,
    create_efficient_belief_buffer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_lazy_belief_arrays():
    """Demonstrate lazy belief arrays with sparse storage."""
    print("\n" + "=" * 60)
    print("DEMO 1: Lazy Belief Arrays with Sparse Storage")
    print("=" * 60)

    # Create belief array
    shape = (50, 50)
    belief_array = LazyBeliefArray(shape, dtype=np.float32, sparsity_threshold=0.8)

    print(f"\n1. Created LazyBeliefArray with shape {shape}")
    print(f"   Initial memory usage: {belief_array.memory_usage():.4f} MB")

    # Create sparse belief state (mostly zeros)
    print("\n2. Creating sparse belief state (10% non-zero)...")
    sparse_belief = np.zeros(shape, dtype=np.float32)

    # Add some concentrated activity
    sparse_belief[20:25, 20:25] = np.random.random((5, 5))
    sparse_belief[10:15, 35:40] = np.random.random((5, 5)) * 0.5

    # Update belief array
    belief_array.update(sparse_belief)
    print(f"   Memory usage after update: {belief_array.memory_usage():.4f} MB")

    # Test compression
    print("\n3. Testing compression...")
    compressed = belief_array.compress_if_beneficial()
    print(f"   Compression applied: {compressed}")
    print(f"   Memory usage after compression: {belief_array.memory_usage():.4f} MB")
    print(f"   Compression ratio: {belief_array.stats.compression_ratio:.2f}x")

    # Test partial updates
    print("\n4. Testing partial updates...")
    indices = np.array([100, 200, 300, 400])
    new_values = np.array([0.9, 0.8, 0.7, 0.6])
    belief_array.update(new_values, indices)

    # Verify values
    retrieved_values = belief_array.get_values(indices)
    print(f"   Updated values: {new_values}")
    print(f"   Retrieved values: {retrieved_values}")

    # Performance comparison
    print("\n5. Memory efficiency comparison:")
    dense_memory = np.prod(shape) * 4 / (1024 * 1024)  # float32
    actual_memory = belief_array.memory_usage()
    print(f"   Dense storage: {dense_memory:.4f} MB")
    print(f"   Sparse storage: {actual_memory:.4f} MB")
    print(f"   Space savings: {(1 - actual_memory / dense_memory) * 100:.1f}%")


def demo_memory_mapped_buffers():
    """Demonstrate memory-mapped buffers for large data."""
    print("\n" + "=" * 60)
    print("DEMO 2: Memory-Mapped Buffers for Large Data")
    print("=" * 60)

    # Create memory-mapped buffer
    shape = (100, 100)
    buffer = MemoryMappedBuffer(shape, dtype=np.float32)

    print(f"\n1. Created MemoryMappedBuffer with shape {shape}")
    print(f"   Virtual memory usage: {buffer.memory_usage():.2f} MB")
    print(f"   Backing file: {Path(buffer.filename).name}")

    # Fill with test data
    print("\n2. Writing test data...")
    test_pattern = np.random.random(shape).astype(np.float32)
    buffer.array[:] = test_pattern
    buffer.sync()  # Ensure written to disk

    # Verify data persistence
    print("   Data written and synced to disk")
    checksum_original = np.sum(buffer.array)
    print(f"   Original checksum: {checksum_original:.6f}")

    # Test resizing
    print("\n3. Testing buffer resizing...")
    new_shape = (150, 150)
    buffer.resize(new_shape)

    print(f"   Resized to {new_shape}")
    print(f"   New virtual memory usage: {buffer.memory_usage():.2f} MB")

    # Verify data preservation
    preserved_data = buffer.array[:100, :100]
    checksum_preserved = np.sum(preserved_data)
    print(f"   Preserved data checksum: {checksum_preserved:.6f}")
    print(
        f"   Data preserved correctly: {abs(checksum_original - checksum_preserved) < 1e-6}"
    )

    # Test random access
    print("\n4. Testing random access performance...")
    indices = np.random.randint(0, 150, size=(1000, 2))

    start_time = time.time()
    [buffer.array[i, j] for i, j in indices]
    access_time = time.time() - start_time

    print(f"   1000 random accesses in {access_time * 1000:.2f} ms")
    print(f"   Average access time: {access_time * 1000000 / 1000:.2f} μs")

    # Cleanup
    buffer._finalizer()
    print("   Buffer cleaned up")


def demo_compact_action_history():
    """Demonstrate compact action history with compression."""
    print("\n" + "=" * 60)
    print("DEMO 3: Compact Action History with Compression")
    print("=" * 60)

    # Create action history
    max_actions = 1000
    action_space_size = 8
    history = CompactActionHistory(max_actions, action_space_size)

    print("\n1. Created CompactActionHistory")
    print(f"   Max actions: {max_actions}")
    print(f"   Action space size: {action_space_size}")
    print(f"   Action data type: {history.action_dtype}")

    # Simulate agent actions
    print("\n2. Simulating agent actions...")
    start_time = time.time()

    for i in range(2000):  # More than max_actions
        action = np.random.randint(0, action_space_size)
        timestamp = start_time + i * 0.1
        reward = np.random.random() - 0.5  # Rewards between -0.5 and 0.5

        history.add_action(action, timestamp, reward)

    print("   Added 2000 actions (circular buffer)")
    print(f"   Current size: {history._size}")
    print(f"   Memory usage: {history.memory_usage_bytes()} bytes")

    # Get action statistics
    print("\n3. Action statistics:")
    stats = history.get_action_statistics()
    print(f"   Stored actions: {stats['count']}")
    print(f"   Action distribution: {stats['action_distribution']}")
    print(f"   Average reward: {stats['avg_reward']:.3f}")
    print(f"   Time span: {stats['time_span']:.1f} seconds")

    # Get recent actions
    print("\n4. Recent action analysis:")
    recent_actions, recent_times, recent_rewards = history.get_recent_actions(20)
    print(f"   Last 20 actions: {recent_actions}")
    print(f"   Recent avg reward: {np.mean(recent_rewards):.3f}")

    # Test compression
    print("\n5. Testing compression...")
    original_size = history._size
    history.compress_old_entries(keep_recent=200)

    new_stats = history.get_action_statistics()
    print(f"   Original size: {original_size}")
    print(f"   Compressed size: {new_stats['count']}")
    print(f"   Compression ratio: {original_size / new_stats['count']:.2f}x")

    # Performance analysis
    print("\n6. Performance analysis:")
    memory_per_action = history.memory_usage_bytes() / new_stats["count"]
    print(f"   Memory per action: {memory_per_action:.1f} bytes")
    print(f"   Total efficiency: {memory_per_action < 20} (target: <20 bytes/action)")


def demo_efficient_temporal_sequence():
    """Demonstrate efficient temporal sequences with delta compression."""
    print("\n" + "=" * 60)
    print("DEMO 4: Efficient Temporal Sequences with Delta Compression")
    print("=" * 60)

    # Create temporal sequence
    max_length = 500
    feature_dim = 32
    sequence = EfficientTemporalSequence(max_length, feature_dim)

    print("\n1. Created EfficientTemporalSequence")
    print(f"   Max length: {max_length}")
    print(f"   Feature dimension: {feature_dim}")

    # Simulate temporal data with smooth transitions
    print("\n2. Adding temporal sequence data...")
    base_state = np.random.random(feature_dim).astype(np.float32)

    for i in range(300):
        # Gradual changes with some noise
        change = 0.01 * np.random.randn(feature_dim) + 0.001 * i
        new_state = base_state + change
        new_state = np.clip(new_state, 0, 1)  # Keep in [0,1] range

        timestamp = time.time() + i * 0.1
        sequence.add_state(new_state, timestamp)

        base_state = new_state

    print("   Added 300 temporal states")

    # Analyze memory usage
    print("\n3. Memory usage analysis:")
    memory_stats = sequence.memory_usage_stats()

    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print(f"\n   Compression achieved: {memory_stats['compression_ratio']:.2f}x")
    print(f"   Space savings: {(1 - 1 / memory_stats['compression_ratio']) * 100:.1f}%")

    # Test state reconstruction
    print("\n4. Testing state reconstruction...")
    test_indices = [0, 50, 100, 150, 200, 250, 299]

    for idx in test_indices:
        reconstructed = sequence.get_state_at_index(idx)
        if reconstructed is not None:
            state_norm = np.linalg.norm(reconstructed)
            print(f"   State {idx}: norm = {state_norm:.4f}")

    # Get recent states
    print("\n5. Recent states analysis:")
    recent_states, recent_timestamps = sequence.get_recent_states(10)

    print(f"   Retrieved {len(recent_states)} recent states")
    if len(recent_states) > 1:
        # Calculate temporal stability
        diffs = [
            np.linalg.norm(recent_states[i + 1] - recent_states[i])
            for i in range(len(recent_states) - 1)
        ]
        avg_diff = np.mean(diffs)
        print(f"   Average state difference: {avg_diff:.6f}")
        print(f"   Temporal stability: {avg_diff < 0.1}")


def demo_compact_knowledge_graph():
    """Demonstrate compact knowledge graphs with sparse features."""
    print("\n" + "=" * 60)
    print("DEMO 5: Compact Knowledge Graphs with Sparse Features")
    print("=" * 60)

    # Create knowledge graph
    max_nodes = 200
    max_edges = 1000
    kg = CompactKnowledgeGraph(max_nodes, max_edges)

    print("\n1. Created CompactKnowledgeGraph")
    print(f"   Max nodes: {max_nodes}")
    print(f"   Max edges: {max_edges}")

    # Add nodes with sparse features
    print("\n2. Adding nodes with sparse features...")
    node_types = ["agent", "resource", "location", "goal", "obstacle"]

    for i in range(100):
        node_type = i % len(node_types)

        # Create sparse feature vector
        features = np.zeros(64, dtype=np.float32)
        # Only set a few random features (sparse)
        sparse_indices = np.random.choice(64, size=5, replace=False)
        features[sparse_indices] = np.random.random(5)

        success = kg.add_node(i, node_type=node_type, features=features)
        if not success:
            print(f"   Failed to add node {i}")
            break

    print(f"   Added {kg._num_nodes} nodes")

    # Add edges to create graph structure
    print("\n3. Building graph structure...")
    edges_added = 0

    for i in range(500):
        source = np.random.randint(0, kg._num_nodes)
        target = np.random.randint(0, kg._num_nodes)

        if source != target:
            edge_type = np.random.randint(0, 3)  # 3 edge types
            weight = np.random.random()

            success = kg.add_edge(source, target, edge_type, weight)
            if success:
                edges_added += 1

    print(f"   Added {edges_added} edges")

    # Analyze graph structure
    print("\n4. Graph structure analysis:")

    # Check node connectivity
    node_degrees = {}
    for node_id in range(kg._num_nodes):
        neighbors = kg.get_neighbors(node_id)
        node_degrees[node_id] = len(neighbors)

    avg_degree = np.mean(list(node_degrees.values()))
    max_degree = max(node_degrees.values())

    print(f"   Average node degree: {avg_degree:.2f}")
    print(f"   Maximum node degree: {max_degree}")
    print(
        f"   Graph density: {edges_added / (kg._num_nodes * (kg._num_nodes - 1)):.4f}"
    )

    # Test feature retrieval
    print("\n5. Feature retrieval test:")
    test_nodes = [0, 25, 50, 75, 99]

    for node_id in test_nodes:
        features = kg.get_node_features(node_id)
        if features is not None:
            non_zero_count = np.count_nonzero(features)
            print(f"   Node {node_id}: {non_zero_count}/64 non-zero features")

    # Memory usage analysis
    print("\n6. Memory usage analysis:")
    memory_stats = kg.memory_usage_stats()

    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # Test storage compaction
    print("\n7. Testing storage compaction...")
    original_memory = memory_stats["total_mb"]

    kg.compact_storage()

    compacted_stats = kg.memory_usage_stats()
    compacted_memory = compacted_stats["total_mb"]

    print(f"   Original memory: {original_memory:.4f} MB")
    print(f"   Compacted memory: {compacted_memory:.4f} MB")
    print(f"   Memory reduction: {(1 - compacted_memory / original_memory) * 100:.1f}%")


def demo_performance_benchmarking():
    """Demonstrate performance benchmarking of all structures."""
    print("\n" + "=" * 60)
    print("DEMO 6: Performance Benchmarking")
    print("=" * 60)

    print("\n1. Running comprehensive benchmarks...")
    results = benchmark_data_structures()

    print("\n2. Benchmark results:")

    for structure_name, metrics in results.items():
        print(f"\n   {structure_name.upper()}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if "mb" in metric_name.lower():
                    print(f"     {metric_name}: {value:.4f} MB")
                elif "ratio" in metric_name.lower():
                    print(f"     {metric_name}: {value:.2f}x")
                else:
                    print(f"     {metric_name}: {value:.4f}")
            else:
                print(f"     {metric_name}: {value}")

    # Calculate overall efficiency
    print("\n3. Overall efficiency analysis:")

    total_memory = 0
    total_compression = 0
    structure_count = 0

    for structure_name, metrics in results.items():
        if "memory_usage_mb" in metrics:
            total_memory += metrics["memory_usage_mb"]
        elif "total_mb" in metrics:
            total_memory += metrics["total_mb"]

        if "compression_ratio" in metrics:
            total_compression += metrics["compression_ratio"]
            structure_count += 1

    avg_compression = total_compression / max(1, structure_count)

    print(f"   Total memory usage: {total_memory:.4f} MB")
    print(f"   Average compression ratio: {avg_compression:.2f}x")
    print(f"   Overall memory efficiency: {avg_compression > 1.5}")


def demo_factory_functions():
    """Demonstrate factory functions for efficient data structures."""
    print("\n" + "=" * 60)
    print("DEMO 7: Factory Functions for Efficient Structures")
    print("=" * 60)

    # Test belief buffer factory
    print("\n1. Testing belief buffer factory...")

    # Small buffer (should use LazyBeliefArray)
    small_buffer = create_efficient_belief_buffer(
        (20, 20), buffer_size=5, use_memory_mapping=False
    )
    print(f"   Small buffer type: {type(small_buffer).__name__}")

    # Large buffer (should use MemoryMappedBuffer)
    large_buffer = create_efficient_belief_buffer(
        (100, 100), buffer_size=50, use_memory_mapping=True
    )
    print(f"   Large buffer type: {type(large_buffer).__name__}")

    # Test usage
    print("\n2. Testing buffer usage...")

    # Use small buffer
    test_belief = np.random.random((20, 20)).astype(np.float32)
    if hasattr(small_buffer, "update"):
        small_buffer.update(test_belief)
        memory_usage = small_buffer.memory_usage()
        print(f"   Small buffer memory: {memory_usage:.4f} MB")

    # Use large buffer
    if hasattr(large_buffer, "array"):
        large_buffer.array[10:15, 10:15] = 1.0
        memory_usage = large_buffer.memory_usage()
        print(f"   Large buffer memory: {memory_usage:.2f} MB")

        # Cleanup
        large_buffer._finalizer()

    print("\n3. Factory function recommendations:")
    print("   - Use LazyBeliefArray for sparse, small-to-medium data")
    print("   - Use MemoryMappedBuffer for large, dense data")
    print("   - Use CompactActionHistory for sequential action data")
    print("   - Use EfficientTemporalSequence for time-series with smooth transitions")
    print("   - Use CompactKnowledgeGraph for sparse graph structures")


def main():
    """Run all demonstrations."""
    print("Memory-Efficient Data Structures Demo")
    print("Implementing Task 5.6: Implement memory-efficient data structures")

    try:
        # Run demonstrations
        demo_lazy_belief_arrays()
        demo_memory_mapped_buffers()
        demo_compact_action_history()
        demo_efficient_temporal_sequence()
        demo_compact_knowledge_graph()
        demo_performance_benchmarking()
        demo_factory_functions()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✓ Lazy belief arrays with automatic sparse compression")
        print("✓ Memory-mapped buffers for efficient large data storage")
        print("✓ Compact action history with temporal compression")
        print("✓ Efficient temporal sequences with delta compression")
        print("✓ Compact knowledge graphs with sparse feature storage")
        print("✓ Factory functions for automatic structure selection")
        print("✓ Performance benchmarking and memory analysis")
        print("✓ Space-efficient data types and storage strategies")
        print("✓ Automatic compression and compaction techniques")
        print("✓ Memory usage monitoring and optimization")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
