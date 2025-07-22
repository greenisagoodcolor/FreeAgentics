#!/usr/bin/env python3
"""Tests for memory-efficient data structures.

Tests for Task 5.6: Implement memory-efficient data structures
"""

import logging
import time
import unittest

import numpy as np

from agents.memory_optimization.efficient_structures import (
    CompactActionHistory,
    CompactKnowledgeGraph,
    EfficientTemporalSequence,
    LazyBeliefArray,
    MemoryMappedBuffer,
    MemoryStats,
    benchmark_data_structures,
    create_efficient_belief_buffer,
)


class TestMemoryStats(unittest.TestCase):
    """Test MemoryStats functionality."""

    def test_stats_initialization(self):
        """Test memory stats initialization."""
        stats = MemoryStats()

        self.assertEqual(stats.dense_memory_mb, 0.0)
        self.assertEqual(stats.sparse_memory_mb, 0.0)
        self.assertEqual(stats.total_memory_mb, 0.0)
        self.assertEqual(stats.compression_ratio, 1.0)
        self.assertEqual(stats.access_count, 0)

    def test_stats_update(self):
        """Test memory stats updates."""
        stats = MemoryStats()

        stats.update_stats(100.0, 25.0)

        self.assertEqual(stats.dense_memory_mb, 100.0)
        self.assertEqual(stats.total_memory_mb, 25.0)
        self.assertEqual(stats.compression_ratio, 4.0)  # 100/25
        self.assertEqual(stats.access_count, 1)


class TestLazyBeliefArray(unittest.TestCase):
    """Test LazyBeliefArray functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.shape = (20, 20)
        self.dtype = np.float32
        self.belief_array = LazyBeliefArray(self.shape, self.dtype)

    def test_initialization(self):
        """Test lazy belief array initialization."""
        self.assertEqual(self.belief_array.shape, self.shape)
        self.assertEqual(self.belief_array.dtype, self.dtype)
        self.assertIsNone(self.belief_array._dense_array)
        self.assertIsNone(self.belief_array._sparse_representation)

    def test_dense_property(self):
        """Test dense property access."""
        # First access should create zero array
        dense = self.belief_array.dense

        self.assertEqual(dense.shape, self.shape)
        self.assertEqual(dense.dtype, self.dtype)
        self.assertTrue(np.all(dense == 0))
        self.assertIsNotNone(self.belief_array._dense_array)

    def test_sparse_property(self):
        """Test sparse property access."""
        # Create some data first
        test_data = np.zeros(self.shape, dtype=self.dtype)
        test_data[5:10, 5:10] = 1.0
        self.belief_array.update(test_data)

        # Access sparse representation
        sparse = self.belief_array.sparse

        self.assertEqual(sparse.shape, self.shape)
        self.assertEqual(sparse.dtype, self.dtype)
        self.assertTrue(sparse.nnz > 0)

    def test_update_full(self):
        """Test full array update."""
        test_data = np.random.random(self.shape).astype(self.dtype)

        self.belief_array.update(test_data)

        # Check that data was stored correctly
        np.testing.assert_array_equal(self.belief_array.dense, test_data)
        self.assertTrue(self.belief_array._is_dirty)

    def test_update_partial(self):
        """Test partial array update."""
        # Initialize with some data
        initial_data = np.ones(self.shape, dtype=self.dtype)
        self.belief_array.update(initial_data)

        # Update specific indices
        indices = np.array([5, 15, 25, 35])
        new_values = np.array([2.0, 3.0, 4.0, 5.0], dtype=self.dtype)

        self.belief_array.update(new_values, indices)

        # Check that updates were applied
        dense = self.belief_array.dense
        self.assertEqual(dense.flat[5], 2.0)
        self.assertEqual(dense.flat[15], 3.0)
        self.assertEqual(dense.flat[25], 4.0)
        self.assertEqual(dense.flat[35], 5.0)

    def test_get_values(self):
        """Test getting values at specific indices."""
        # Set up test data
        test_data = np.arange(np.prod(self.shape), dtype=self.dtype).reshape(self.shape)
        self.belief_array.update(test_data)

        # Get specific values
        indices = np.array([0, 10, 50, 100])
        values = self.belief_array.get_values(indices)

        expected = test_data.flat[indices]
        np.testing.assert_array_equal(values, expected)

    def test_compression(self):
        """Test beneficial compression."""
        # Create sparse data (mostly zeros)
        sparse_data = np.zeros(self.shape, dtype=self.dtype)
        sparse_data[2:5, 2:5] = 1.0  # Only 9 non-zero elements out of 400

        self.belief_array.update(sparse_data)

        # Should compress successfully
        compressed = self.belief_array.compress_if_beneficial()
        self.assertTrue(compressed)

        # Dense array should be freed
        self.assertIsNone(self.belief_array._dense_array)

        # But we should still be able to access data
        recovered = self.belief_array.dense
        np.testing.assert_array_almost_equal(recovered, sparse_data)

    def test_memory_usage(self):
        """Test memory usage calculation."""
        # Initially should be minimal
        initial_usage = self.belief_array.memory_usage()
        self.assertGreaterEqual(initial_usage, 0.0)

        # Add data and check usage increases
        test_data = np.random.random(self.shape).astype(self.dtype)
        self.belief_array.update(test_data)

        usage_with_data = self.belief_array.memory_usage()
        self.assertGreater(usage_with_data, initial_usage)


class TestMemoryMappedBuffer(unittest.TestCase):
    """Test MemoryMappedBuffer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.shape = (10, 10)
        self.dtype = np.float32

    def test_initialization(self):
        """Test memory-mapped buffer initialization."""
        buffer = MemoryMappedBuffer(self.shape, self.dtype)

        self.assertEqual(buffer.shape, self.shape)
        self.assertEqual(buffer.dtype, self.dtype)
        self.assertEqual(buffer.array.shape, self.shape)
        self.assertEqual(buffer.array.dtype, self.dtype)

        # Should be initialized to zeros
        self.assertTrue(np.all(buffer.array == 0))

        # Cleanup
        buffer._finalizer()

    def test_array_access(self):
        """Test array access and modification."""
        buffer = MemoryMappedBuffer(self.shape, self.dtype)

        # Modify data
        test_data = np.random.random(self.shape).astype(self.dtype)
        buffer.array[:] = test_data

        # Check data persists
        np.testing.assert_array_equal(buffer.array, test_data)

        # Cleanup
        buffer._finalizer()

    def test_sync(self):
        """Test synchronization to disk."""
        buffer = MemoryMappedBuffer(self.shape, self.dtype)

        # Modify data
        buffer.array[5, 5] = 42.0

        # Sync should not raise error
        buffer.sync()
        self.assertEqual(buffer.array[5, 5], 42.0)

        # Cleanup
        buffer._finalizer()

    def test_resize(self):
        """Test buffer resizing."""
        buffer = MemoryMappedBuffer((5, 5), self.dtype)

        # Set some initial data
        buffer.array[2, 2] = 10.0

        # Resize to larger
        new_shape = (8, 8)
        buffer.resize(new_shape)

        self.assertEqual(buffer.shape, new_shape)
        self.assertEqual(buffer.array.shape, new_shape)

        # Original data should be preserved
        self.assertEqual(buffer.array[2, 2], 10.0)

        # Cleanup
        buffer._finalizer()

    def test_memory_usage(self):
        """Test memory usage calculation."""
        buffer = MemoryMappedBuffer(self.shape, self.dtype)

        expected_mb = np.prod(self.shape) * np.dtype(self.dtype).itemsize / (1024 * 1024)
        actual_mb = buffer.memory_usage()

        self.assertAlmostEqual(actual_mb, expected_mb, places=6)

        # Cleanup
        buffer._finalizer()


class TestCompactActionHistory(unittest.TestCase):
    """Test CompactActionHistory functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_actions = 100
        self.action_space_size = 8
        self.history = CompactActionHistory(self.max_actions, self.action_space_size)

    def test_initialization(self):
        """Test action history initialization."""
        self.assertEqual(self.history.max_actions, self.max_actions)
        self.assertEqual(self.history.action_space_size, self.action_space_size)
        self.assertEqual(self.history._size, 0)
        self.assertEqual(self.history._head, 0)

        # Should use uint8 for small action space
        self.assertEqual(self.history.action_dtype, np.uint8)

    def test_add_action(self):
        """Test adding actions."""
        timestamp = time.time()

        self.history.add_action(3, timestamp, 0.5)

        self.assertEqual(self.history._size, 1)
        self.assertEqual(self.history._head, 1)

        # Check stored values
        self.assertEqual(self.history._actions[0], 3)
        self.assertEqual(self.history._timestamps[0], timestamp)
        self.assertAlmostEqual(self.history._rewards[0], 0.5, places=2)

    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        # Fill beyond capacity
        for i in range(self.max_actions + 10):
            self.history.add_action(i % self.action_space_size, time.time() + i, i * 0.1)

        # Should have max_actions items
        self.assertEqual(self.history._size, self.max_actions)

        # Head should wrap around
        self.assertEqual(self.history._head, 10)  # 110 % 100

    def test_get_recent_actions(self):
        """Test getting recent actions."""
        # Add some actions
        timestamps = []
        for i in range(20):
            timestamp = time.time() + i
            timestamps.append(timestamp)
            self.history.add_action(i % self.action_space_size, timestamp, i * 0.1)

        # Get recent actions
        actions, times, rewards = self.history.get_recent_actions(5)

        self.assertEqual(len(actions), 5)
        self.assertEqual(len(times), 5)
        self.assertEqual(len(rewards), 5)

        # Should be most recent
        expected_actions = np.array([15, 16, 17, 18, 19]) % self.action_space_size
        np.testing.assert_array_equal(actions, expected_actions)

    def test_action_statistics(self):
        """Test action statistics."""
        # Add diverse actions
        for i in range(50):
            action = i % 4  # Use only actions 0, 1, 2, 3
            self.history.add_action(action, time.time() + i, np.random.random())

        stats = self.history.get_action_statistics()

        self.assertEqual(stats["count"], 50)
        self.assertIn("action_distribution", stats)
        self.assertIn("avg_reward", stats)
        self.assertIn("time_span", stats)

        # Check action distribution
        dist = stats["action_distribution"]
        # Should have roughly equal distribution for actions 0-3
        for action in range(4):
            self.assertIn(action, dist)
            self.assertGreaterEqual(dist[action], 10)  # At least 10 occurrences

    def test_memory_usage(self):
        """Test memory usage calculation."""
        # Add some actions
        for i in range(50):
            self.history.add_action(i % self.action_space_size, time.time() + i, i * 0.1)

        memory_bytes = self.history.memory_usage_bytes()

        # Should be reasonable
        self.assertGreater(memory_bytes, 0)
        self.assertLess(memory_bytes, 10000)  # Should be small

    def test_compression(self):
        """Test compression of old entries."""
        # Fill with many actions
        for i in range(200):
            self.history.add_action(i % self.action_space_size, time.time() + i, i * 0.01)

        original_count = self.history._size

        # Compress, keeping only 30 recent entries
        self.history.compress_old_entries(keep_recent=30)

        # Should have fewer total entries but still include recent ones
        self.assertLessEqual(self.history._size, original_count)
        self.assertGreaterEqual(self.history._size, 30)

        # Recent actions should still be accessible
        actions, _, _ = self.history.get_recent_actions(10)
        self.assertEqual(len(actions), 10)


class TestEfficientTemporalSequence(unittest.TestCase):
    """Test EfficientTemporalSequence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_length = 100
        self.feature_dim = 16
        self.sequence = EfficientTemporalSequence(self.max_length, self.feature_dim)

    def test_initialization(self):
        """Test temporal sequence initialization."""
        self.assertEqual(self.sequence.max_length, self.max_length)
        self.assertEqual(self.sequence.feature_dim, self.feature_dim)
        self.assertEqual(len(self.sequence._deltas), 0)
        self.assertEqual(len(self.sequence._timestamps), 0)
        self.assertEqual(len(self.sequence._base_states), 0)

    def test_add_state(self):
        """Test adding states."""
        state1 = np.random.random(self.feature_dim).astype(np.float32)
        timestamp1 = time.time()

        self.sequence.add_state(state1, timestamp1)

        self.assertEqual(len(self.sequence._deltas), 1)
        self.assertEqual(len(self.sequence._timestamps), 1)
        self.assertEqual(len(self.sequence._base_states), 1)

        # First state should be stored as base state
        base_index, base_state, base_timestamp = self.sequence._base_states[0]
        self.assertEqual(base_index, 0)
        np.testing.assert_array_equal(base_state, state1)
        self.assertEqual(base_timestamp, timestamp1)

    def test_delta_compression(self):
        """Test delta compression between states."""
        # Create sequence of similar states
        base_state = np.ones(self.feature_dim, dtype=np.float32)

        for i in range(20):
            # Small changes each time
            state = base_state + 0.01 * i
            self.sequence.add_state(state, time.time() + i)

        # Should have base states and deltas
        self.assertGreater(len(self.sequence._base_states), 0)
        self.assertEqual(len(self.sequence._deltas), 20)

        # Most deltas should be small
        delta_list = list(self.sequence._deltas)
        delta_magnitudes = [np.linalg.norm(delta) for delta in delta_list[1:]]
        avg_delta_magnitude = np.mean(delta_magnitudes)
        self.assertLess(avg_delta_magnitude, 1.0)  # Should be small

    def test_state_reconstruction(self):
        """Test state reconstruction from deltas."""
        # Add a sequence of states with known pattern
        states = []
        for i in range(15):
            state = np.full(self.feature_dim, i, dtype=np.float32)
            states.append(state)
            self.sequence.add_state(state, time.time() + i)

        # Reconstruct states and verify
        for i in range(len(states)):
            reconstructed = self.sequence.get_state_at_index(i)
            self.assertIsNotNone(reconstructed)
            np.testing.assert_array_almost_equal(reconstructed, states[i], decimal=5)

    def test_get_recent_states(self):
        """Test getting recent states."""
        # Add several states
        original_states = []
        timestamps = []
        for i in range(25):
            state = np.full(self.feature_dim, i, dtype=np.float32)
            timestamp = time.time() + i
            original_states.append(state)
            timestamps.append(timestamp)
            self.sequence.add_state(state, timestamp)

        # Get recent states
        recent_states, recent_timestamps = self.sequence.get_recent_states(5)

        self.assertEqual(len(recent_states), 5)
        self.assertEqual(len(recent_timestamps), 5)

        # Should match last 5 states
        for i in range(5):
            np.testing.assert_array_almost_equal(
                recent_states[i], original_states[-5 + i], decimal=5
            )

    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        # Add states
        for i in range(50):
            state = np.random.random(self.feature_dim).astype(np.float32)
            self.sequence.add_state(state, time.time() + i)

        stats = self.sequence.memory_usage_stats()

        self.assertIn("raw_memory_mb", stats)
        self.assertIn("total_mb", stats)
        self.assertIn("compression_ratio", stats)
        self.assertIn("sequence_length", stats)

        # Should achieve some compression (or at least not use significantly more)
        self.assertGreater(stats["compression_ratio"], 0.5)
        self.assertEqual(stats["sequence_length"], 50)


class TestCompactKnowledgeGraph(unittest.TestCase):
    """Test CompactKnowledgeGraph functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_nodes = 50
        self.max_edges = 200
        self.kg = CompactKnowledgeGraph(self.max_nodes, self.max_edges)

    def test_initialization(self):
        """Test knowledge graph initialization."""
        self.assertEqual(self.kg.max_nodes, self.max_nodes)
        self.assertEqual(self.kg.max_edges, self.max_edges)
        self.assertEqual(self.kg._num_nodes, 0)
        self.assertEqual(self.kg._num_edges, 0)
        self.assertEqual(len(self.kg._node_index), 0)

    def test_add_node(self):
        """Test adding nodes."""
        node_id = 1
        node_type = 2
        features = np.array([1.0, 0.0, 2.5, 0.0, 1.5])

        success = self.kg.add_node(node_id, node_type, features)

        self.assertTrue(success)
        self.assertEqual(self.kg._num_nodes, 1)
        self.assertIn(node_id, self.kg._node_index)
        self.assertEqual(self.kg._node_types[0], node_type)

    def test_add_duplicate_node(self):
        """Test adding duplicate node."""
        node_id = 1

        # Add node first time
        success1 = self.kg.add_node(node_id)
        self.assertTrue(success1)

        # Try to add same node again
        success2 = self.kg.add_node(node_id)
        self.assertFalse(success2)

        # Should still have only one node
        self.assertEqual(self.kg._num_nodes, 1)

    def test_add_edge(self):
        """Test adding edges."""
        # Add nodes first
        self.kg.add_node(1)
        self.kg.add_node(2)

        # Add edge
        success = self.kg.add_edge(1, 2, edge_type=1, weight=0.8)

        self.assertTrue(success)
        self.assertEqual(self.kg._num_edges, 1)
        self.assertEqual(self.kg._edge_sources[0], 1)
        self.assertEqual(self.kg._edge_targets[0], 2)
        self.assertEqual(self.kg._edge_types[0], 1)
        self.assertAlmostEqual(self.kg._edge_weights[0], 0.8, places=2)

    def test_add_edge_invalid_nodes(self):
        """Test adding edge with invalid nodes."""
        # Try to add edge without nodes
        success = self.kg.add_edge(1, 2)
        self.assertFalse(success)
        self.assertEqual(self.kg._num_edges, 0)

    def test_get_neighbors(self):
        """Test getting node neighbors."""
        # Build small graph
        self.kg.add_node(1)
        self.kg.add_node(2)
        self.kg.add_node(3)

        self.kg.add_edge(1, 2, edge_type=0, weight=0.5)
        self.kg.add_edge(1, 3, edge_type=1, weight=0.8)

        # Get neighbors of node 1
        neighbors = self.kg.get_neighbors(1)

        self.assertEqual(len(neighbors), 2)

        # Check neighbor details
        neighbor_ids = [n[0] for n in neighbors]
        self.assertIn(2, neighbor_ids)
        self.assertIn(3, neighbor_ids)

    def test_get_node_features(self):
        """Test getting node features."""
        node_id = 1
        features = np.array([1.0, 2.0, 0.0, 3.0])

        # Add node with features
        self.kg.add_node(node_id, features=features)

        # Retrieve features
        retrieved_features = self.kg.get_node_features(node_id)

        self.assertIsNotNone(retrieved_features)
        # Note: features are stored as float16 and may have precision loss
        np.testing.assert_array_almost_equal(retrieved_features[:4], features, decimal=2)

    def test_compact_storage(self):
        """Test storage compaction."""
        # Add some nodes and edges
        for i in range(10):
            self.kg.add_node(i)

        for i in range(5):
            self.kg.add_edge(i, i + 1)

        self.kg.max_nodes
        self.kg.max_edges

        # Compact storage
        self.kg.compact_storage()

        # Limits should be reduced to actual usage
        self.assertEqual(self.kg.max_nodes, 10)
        self.assertEqual(self.kg.max_edges, 5)

        # Functionality should still work
        neighbors = self.kg.get_neighbors(0)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], 1)

    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        # Add data
        for i in range(20):
            features = np.random.random(10).astype(np.float32)
            self.kg.add_node(i, features=features)

        for i in range(30):
            source = np.random.randint(0, 20)
            target = np.random.randint(0, 20)
            if source != target:
                self.kg.add_edge(source, target, weight=np.random.random())

        stats = self.kg.memory_usage_stats()

        self.assertIn("nodes_mb", stats)
        self.assertIn("edges_mb", stats)
        self.assertIn("total_mb", stats)
        self.assertIn("num_nodes", stats)
        self.assertIn("num_edges", stats)
        self.assertIn("utilization", stats)

        self.assertEqual(stats["num_nodes"], 20)
        self.assertGreater(stats["total_mb"], 0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_efficient_belief_buffer_small(self):
        """Test creating small efficient belief buffer."""
        shape = (10, 10)
        buffer_size = 5

        buffer = create_efficient_belief_buffer(shape, buffer_size, use_memory_mapping=False)

        # Should return LazyBeliefArray for small buffers
        self.assertIsInstance(buffer, LazyBeliefArray)
        self.assertEqual(buffer.shape, shape)

    def test_create_efficient_belief_buffer_large(self):
        """Test creating large efficient belief buffer."""
        shape = (100, 100)
        buffer_size = 50

        buffer = create_efficient_belief_buffer(shape, buffer_size, use_memory_mapping=True)

        # Should return MemoryMappedBuffer for large buffers
        self.assertIsInstance(buffer, MemoryMappedBuffer)
        expected_shape = (buffer_size,) + shape
        self.assertEqual(buffer.shape, expected_shape)

        # Cleanup
        buffer._finalizer()

    def test_benchmark_data_structures(self):
        """Test benchmarking function."""
        # This is mainly a smoke test to ensure it runs without error
        results = benchmark_data_structures()

        self.assertIsInstance(results, dict)
        self.assertIn("lazy_belief", results)
        self.assertIn("action_history", results)
        self.assertIn("temporal_sequence", results)
        self.assertIn("knowledge_graph", results)

        # Each result should have relevant metrics
        for structure_name, metrics in results.items():
            self.assertIsInstance(metrics, dict)
            self.assertTrue(len(metrics) > 0)


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
