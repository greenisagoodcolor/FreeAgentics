#!/usr/bin/env python3
"""Memory-efficient data structures for agent operations.

This module implements memory-efficient data structures for Task 5.6:
Implement memory-efficient data structures.

Key features:
- Sparse belief state arrays with lazy evaluation
- Memory-mapped observation buffers
- Compact action history storage
- Efficient temporal sequence storage
- Memory-efficient knowledge representations
- Optimized agent state serialization
"""

import logging
import mmap
import tempfile
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from .belief_compression import BeliefCompressor, SparseBeliefState

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics for efficient structures."""

    dense_memory_mb: float = 0.0
    sparse_memory_mb: float = 0.0
    compressed_memory_mb: float = 0.0
    total_memory_mb: float = 0.0
    compression_ratio: float = 1.0
    access_count: int = 0

    def update_stats(self, dense_mb: float, efficient_mb: float):
        """Update memory statistics."""
        self.dense_memory_mb = dense_mb
        self.total_memory_mb = efficient_mb
        if dense_mb > 0:
            self.compression_ratio = dense_mb / efficient_mb
        self.access_count += 1


class LazyBeliefArray:
    """Lazy-evaluated sparse belief array for memory efficiency."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        sparsity_threshold: float = 0.9,
    ):
        """Initialize lazy belief array.

        Args:
            shape: Array shape
            dtype: Data type
            sparsity_threshold: Sparsity threshold for compression
        """
        self.shape = shape
        self.dtype = dtype
        self.sparsity_threshold = sparsity_threshold

        # Lazy storage
        self._dense_array: Optional[np.ndarray] = None
        self._sparse_representation: Optional[SparseBeliefState] = None
        self._is_dirty = False
        self._compressor = BeliefCompressor(sparsity_threshold)

        # Memory tracking
        self.stats = MemoryStats()
        self._lock = threading.RLock()

    @property
    def dense(self) -> np.ndarray:
        """Get dense representation, computing if necessary."""
        with self._lock:
            if self._dense_array is None:
                if self._sparse_representation is not None:
                    self._dense_array = self._compressor.decompress(
                        self._sparse_representation
                    )
                else:
                    self._dense_array = np.zeros(self.shape, dtype=self.dtype)
            return self._dense_array

    @property
    def sparse(self) -> SparseBeliefState:
        """Get sparse representation, computing if necessary."""
        with self._lock:
            if self._sparse_representation is None or self._is_dirty:
                if self._dense_array is not None:
                    self._sparse_representation = self._compressor.compress(
                        self._dense_array, self.dtype
                    )
                    self._is_dirty = False
                else:
                    # Create empty sparse representation
                    self._sparse_representation = SparseBeliefState(
                        data=np.array([], dtype=self.dtype),
                        indices=np.array([], dtype=np.int32),
                        shape=self.shape,
                        dtype=self.dtype,
                    )
            return self._sparse_representation

    def update(self, values: np.ndarray, indices: Optional[np.ndarray] = None):
        """Update belief values efficiently.

        Args:
            values: New values
            indices: Indices to update (if None, update all)
        """
        with self._lock:
            if indices is None:
                # Full update
                self._dense_array = values.astype(self.dtype)
                self._sparse_representation = None
            else:
                # Partial update - ensure dense array exists
                dense = self.dense  # This will create if needed
                dense.flat[indices] = values
                self._sparse_representation = None

            self._is_dirty = True
            self._update_memory_stats()

    def get_values(self, indices: np.ndarray) -> np.ndarray:
        """Get values at specific indices efficiently.

        Args:
            indices: Indices to retrieve

        Returns:
            Values at indices
        """
        with self._lock:
            # Try to use sparse representation if available and efficient
            if (
                self._sparse_representation is not None
                and len(indices) < self._sparse_representation.nnz * 0.1
            ):
                # Use sparse lookup for small number of indices
                sparse_rep = self.sparse
                result = np.zeros(len(indices), dtype=self.dtype)

                # Find which requested indices exist in sparse representation
                for i, idx in enumerate(indices):
                    sparse_idx_pos = np.where(sparse_rep.indices == idx)[0]
                    if len(sparse_idx_pos) > 0:
                        result[i] = sparse_rep.data[sparse_idx_pos[0]]

                return result
            else:
                # Use dense lookup
                return self.dense.flat[indices]

    def compress_if_beneficial(self) -> bool:
        """Compress to sparse format if beneficial.

        Returns:
            True if compression was applied
        """
        with self._lock:
            if self._dense_array is None:
                return False

            # Calculate sparsity
            sparsity = 1 - (
                np.count_nonzero(self._dense_array) / self._dense_array.size
            )

            if sparsity > self.sparsity_threshold:
                # Force compression and free dense array
                self.sparse  # Compute sparse representation
                self._dense_array = None
                self._update_memory_stats()
                return True

            return False

    def memory_usage(self) -> float:
        """Get current memory usage in MB."""
        with self._lock:
            total_bytes = 0

            if self._dense_array is not None:
                total_bytes += self._dense_array.nbytes

            if self._sparse_representation is not None:
                total_bytes += self._sparse_representation.memory_usage()

            return total_bytes / (1024 * 1024)

    def _update_memory_stats(self):
        """Update memory statistics."""
        dense_mb = np.prod(self.shape) * np.dtype(self.dtype).itemsize / (1024 * 1024)
        current_mb = self.memory_usage()
        self.stats.update_stats(dense_mb, current_mb)


class MemoryMappedBuffer:
    """Memory-mapped buffer for efficient large data storage."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        mode: str = "r+",
        filename: Optional[str] = None,
    ):
        """Initialize memory-mapped buffer.

        Args:
            shape: Buffer shape
            dtype: Data type
            mode: File access mode
            filename: Optional filename (temp file if None)
        """
        self.shape = shape
        self.dtype = dtype
        self.mode = mode

        # Create temporary file if no filename provided
        if filename is None:
            self._temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.filename = self._temp_file.name
            self._temp_file.close()
        else:
            self.filename = filename
            self._temp_file = None

        # Calculate buffer size
        self.itemsize = np.dtype(dtype).itemsize
        self.total_items = np.prod(shape)
        self.buffer_size = self.total_items * self.itemsize

        # Initialize file
        self._initialize_file()

        # Create memory map
        self._file = open(self.filename, "r+b")
        self._mmap = mmap.mmap(
            self._file.fileno(), self.buffer_size, access=mmap.ACCESS_WRITE
        )
        self._array = np.frombuffer(self._mmap, dtype=dtype).reshape(shape)

        # Thread safety
        self._lock = threading.RLock()

        # Register for cleanup
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._mmap, self._file, self.filename
        )

    def _initialize_file(self):
        """Initialize the backing file."""
        with open(self.filename, "wb") as f:
            # Write zeros to allocate space
            f.write(b"\x00" * self.buffer_size)

    @property
    def array(self) -> np.ndarray:
        """Get the memory-mapped array."""
        return self._array

    def sync(self):
        """Synchronize changes to disk."""
        with self._lock:
            self._mmap.flush()

    def resize(self, new_shape: Tuple[int, ...]):
        """Resize the buffer (creates new file).

        Args:
            new_shape: New buffer shape
        """
        with self._lock:
            # Save current data
            old_data = self._array.copy()

            # Close current mapping (need to clear array reference first)
            del self._array
            self._mmap.close()
            self._file.close()

            # Update shape and size
            self.shape = new_shape
            self.total_items = np.prod(new_shape)
            self.buffer_size = self.total_items * self.itemsize

            # Create new file
            self._initialize_file()

            # Create new mapping
            self._file = open(self.filename, "r+b")
            self._mmap = mmap.mmap(
                self._file.fileno(), self.buffer_size, access=mmap.ACCESS_WRITE
            )
            self._array = np.frombuffer(self._mmap, dtype=self.dtype).reshape(new_shape)

            # Copy old data with proper 2D indexing
            if len(old_data.shape) == 2 and len(new_shape) == 2:
                # 2D to 2D copy - preserve spatial relationships
                old_h, old_w = old_data.shape
                new_h, new_w = new_shape
                copy_h = min(old_h, new_h)
                copy_w = min(old_w, new_w)
                self._array[:copy_h, :copy_w] = old_data[:copy_h, :copy_w]
            else:
                # Fallback to flat copy
                old_flat = old_data.flatten()
                new_flat = self._array.flatten()
                copy_size = min(len(old_flat), len(new_flat))
                new_flat[:copy_size] = old_flat[:copy_size]

            # Sync changes to ensure data is written
            self._mmap.flush()

    def memory_usage(self) -> float:
        """Get memory usage in MB (virtual, not physical)."""
        return float(self.buffer_size) / (1024 * 1024)

    @staticmethod
    def _cleanup(mmap_obj, file_obj, filename: str):
        """Cleanup function for finalizer."""
        try:
            if mmap_obj:
                mmap_obj.close()
            if file_obj:
                file_obj.close()
            Path(filename).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Error cleaning up memory-mapped file {filename}: {e}")


class CompactActionHistory:
    """Compact storage for action history with compression."""

    def __init__(self, max_actions: int = 1000, action_space_size: int = 10):
        """Initialize compact action history.

        Args:
            max_actions: Maximum number of actions to store
            action_space_size: Size of action space
        """
        self.max_actions = max_actions
        self.action_space_size = action_space_size

        # Use minimal data types
        if action_space_size <= 256:
            self.action_dtype = np.uint8
        elif action_space_size <= 65536:
            self.action_dtype = np.uint16
        else:
            self.action_dtype = np.uint32

        # Storage
        self._actions = np.zeros(max_actions, dtype=self.action_dtype)
        self._timestamps = np.zeros(max_actions, dtype=np.float32)
        self._rewards = np.zeros(
            max_actions, dtype=np.float16
        )  # Half precision for rewards

        # Circular buffer state
        self._head = 0
        self._size = 0
        self._lock = threading.RLock()

    def add_action(self, action: int, timestamp: float, reward: float = 0.0):
        """Add an action to the history.

        Args:
            action: Action taken
            timestamp: Timestamp of action
            reward: Reward received
        """
        with self._lock:
            self._actions[self._head] = action
            self._timestamps[self._head] = timestamp
            self._rewards[self._head] = reward

            self._head = (self._head + 1) % self.max_actions
            self._size = min(self._size + 1, self.max_actions)

    def get_recent_actions(
        self, count: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get recent actions.

        Args:
            count: Number of recent actions to retrieve

        Returns:
            Tuple of (actions, timestamps, rewards)
        """
        with self._lock:
            if self._size == 0:
                return np.array([]), np.array([]), np.array([])

            count = min(count, self._size)

            if self._size < self.max_actions:
                # Buffer not full yet
                start_idx = max(0, self._head - count)
                end_idx = self._head
                indices = np.arange(start_idx, end_idx)
            else:
                # Circular buffer
                if count <= self._head:
                    indices = np.arange(self._head - count, self._head)
                else:
                    # Wrap around
                    part1 = np.arange(
                        self.max_actions - (count - self._head),
                        self.max_actions,
                    )
                    part2 = np.arange(0, self._head)
                    indices = np.concatenate([part1, part2])

            return (
                self._actions[indices].copy(),
                self._timestamps[indices].copy(),
                self._rewards[indices].copy(),
            )

    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action history.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if self._size == 0:
                return {"count": 0, "action_distribution": {}}

            (
                recent_actions,
                recent_timestamps,
                recent_rewards,
            ) = self.get_recent_actions(self._size)

            # Action distribution
            unique_actions, counts = np.unique(recent_actions, return_counts=True)
            action_dist = dict(zip(unique_actions.tolist(), counts.tolist()))

            return {
                "count": self._size,
                "action_distribution": action_dist,
                "avg_reward": float(np.mean(recent_rewards)),
                "time_span": (
                    float(recent_timestamps[-1] - recent_timestamps[0])
                    if len(recent_timestamps) > 1
                    else 0.0
                ),
                "memory_usage_bytes": self.memory_usage_bytes(),
            }

    def memory_usage_bytes(self) -> int:
        """Get memory usage in bytes."""
        return int(
            self._actions.nbytes + self._timestamps.nbytes + self._rewards.nbytes
        )

    def compress_old_entries(self, keep_recent: int = 100):
        """Compress old entries by downsampling.

        Args:
            keep_recent: Number of recent entries to keep uncompressed
        """
        with self._lock:
            if self._size <= keep_recent:
                return

            # Get all current data
            actions, timestamps, rewards = self.get_recent_actions(self._size)

            # Keep recent entries and downsample old ones
            recent_actions = actions[-keep_recent:]
            recent_timestamps = timestamps[-keep_recent:]
            recent_rewards = rewards[-keep_recent:]

            old_actions = actions[:-keep_recent]
            old_timestamps = timestamps[:-keep_recent]
            old_rewards = rewards[:-keep_recent]

            # Downsample old entries (keep every nth entry)
            downsample_factor = max(
                2, len(old_actions) // (self.max_actions - keep_recent)
            )
            downsampled_indices = np.arange(0, len(old_actions), downsample_factor)

            downsampled_actions = old_actions[downsampled_indices]
            downsampled_timestamps = old_timestamps[downsampled_indices]
            downsampled_rewards = old_rewards[downsampled_indices]

            # Combine downsampled old + recent
            combined_actions = np.concatenate([downsampled_actions, recent_actions])
            combined_timestamps = np.concatenate(
                [downsampled_timestamps, recent_timestamps]
            )
            combined_rewards = np.concatenate([downsampled_rewards, recent_rewards])

            # Reset buffer and add combined data
            self._head = 0
            self._size = 0

            for action, timestamp, reward in zip(
                combined_actions, combined_timestamps, combined_rewards
            ):
                self.add_action(int(action), float(timestamp), float(reward))


class EfficientTemporalSequence:
    """Memory-efficient storage for temporal sequences with delta compression."""

    def __init__(
        self,
        max_length: int = 1000,
        feature_dim: int = 32,
        compression_ratio: float = 0.1,
    ):
        """Initialize efficient temporal sequence.

        Args:
            max_length: Maximum sequence length
            feature_dim: Feature dimensionality
            compression_ratio: Target compression ratio for deltas
        """
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.compression_ratio = compression_ratio

        # Storage for base states and deltas
        self._base_states: Deque[np.ndarray] = deque(
            maxlen=max_length // 10
        )  # Store periodic base states
        self._deltas: Deque[np.ndarray] = deque(maxlen=max_length)
        self._timestamps: Deque[float] = deque(maxlen=max_length)

        # Compression settings
        self._base_interval = 10  # Store base state every N steps
        self._delta_threshold = 0.01  # Minimum delta to store

        # Current state tracking
        self._current_state = np.zeros(feature_dim, dtype=np.float32)
        self._last_base_index = -1
        self._lock = threading.RLock()

    def add_state(self, state: np.ndarray, timestamp: float):
        """Add a state to the sequence.

        Args:
            state: State vector
            timestamp: Timestamp
        """
        with self._lock:
            state = state.astype(np.float32)

            # Check if we should store a new base state
            should_store_base = (
                len(self._deltas) % self._base_interval == 0
                or len(self._base_states) == 0
            )

            if should_store_base:
                # Store as base state
                self._base_states.append((len(self._deltas), state.copy(), timestamp))
                self._last_base_index = len(self._deltas)
                delta = np.zeros_like(state)
            else:
                # Store as delta from last state
                delta = state - self._current_state

                # Zero out small deltas
                delta[np.abs(delta) < self._delta_threshold] = 0

            # Store delta (may be zero for base states)
            self._deltas.append(delta)
            self._timestamps.append(timestamp)
            self._current_state = state.copy()

    def get_state_at_index(self, index: int) -> Optional[np.ndarray]:
        """Reconstruct state at given index.

        Args:
            index: Index in sequence

        Returns:
            Reconstructed state or None if invalid index
        """
        with self._lock:
            if index < 0 or index >= len(self._deltas):
                return None

            # Find nearest base state
            base_index = -1
            base_state = None

            for stored_index, stored_state, _ in self._base_states:
                if stored_index <= index:
                    base_index = stored_index
                    base_state = stored_state.copy()
                else:
                    break

            if base_state is None:
                # No base state found, start from zero
                base_state = np.zeros(self.feature_dim, dtype=np.float32)
                base_index = -1

            # Apply deltas from base to target index
            current_state = base_state
            for i in range(base_index + 1, index + 1):
                if i < len(self._deltas):
                    current_state += self._deltas[i]

            return current_state

    def get_recent_states(self, count: int) -> Tuple[List[np.ndarray], List[float]]:
        """Get recent states efficiently.

        Args:
            count: Number of recent states

        Returns:
            Tuple of (states, timestamps)
        """
        with self._lock:
            if len(self._deltas) == 0:
                return [], []

            count = min(count, len(self._deltas))
            start_index = len(self._deltas) - count

            states = []
            timestamps = []

            for i in range(start_index, len(self._deltas)):
                state = self.get_state_at_index(i)
                if state is not None:
                    states.append(state)
                    timestamps.append(self._timestamps[i])

            return states, timestamps

    def memory_usage_stats(self) -> Dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Memory usage breakdown in MB
        """
        with self._lock:
            # Calculate raw memory usage
            raw_states_mb = len(self._deltas) * self.feature_dim * 4 / (1024 * 1024)

            # Calculate actual memory usage
            base_states_mb = (
                len(self._base_states) * self.feature_dim * 4 / (1024 * 1024)
            )

            deltas_bytes = sum(delta.nbytes for delta in self._deltas)
            deltas_mb = deltas_bytes / (1024 * 1024)

            timestamps_mb = len(self._timestamps) * 4 / (1024 * 1024)

            total_mb = base_states_mb + deltas_mb + timestamps_mb

            return {
                "raw_memory_mb": raw_states_mb,
                "base_states_mb": base_states_mb,
                "deltas_mb": deltas_mb,
                "timestamps_mb": timestamps_mb,
                "total_mb": total_mb,
                "compression_ratio": raw_states_mb / total_mb if total_mb > 0 else 1.0,
                "sequence_length": len(self._deltas),
            }


class CompactKnowledgeGraph:
    """Memory-efficient knowledge graph with compressed storage."""

    def __init__(self, max_nodes: int = 1000, max_edges: int = 5000):
        """Initialize compact knowledge graph.

        Args:
            max_nodes: Maximum number of nodes
            max_edges: Maximum number of edges
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        # Node storage (compact)
        self._node_ids = np.zeros(max_nodes, dtype=np.uint32)
        self._node_types = np.zeros(max_nodes, dtype=np.uint8)  # Support 256 node types
        self._node_features = sparse.csr_matrix(
            (max_nodes, 64), dtype=np.float32
        )  # Sparse features

        # Edge storage (sparse)
        self._edge_sources = np.zeros(max_edges, dtype=np.uint32)
        self._edge_targets = np.zeros(max_edges, dtype=np.uint32)
        self._edge_types = np.zeros(max_edges, dtype=np.uint8)  # Support 256 edge types
        self._edge_weights = np.zeros(max_edges, dtype=np.float16)

        # Active counts
        self._num_nodes = 0
        self._num_edges = 0

        # Index for fast lookup
        self._node_index: Dict[str, int] = {}  # node_id -> index
        self._lock = threading.RLock()

    def add_node(
        self,
        node_id: int,
        node_type: int = 0,
        features: Optional[np.ndarray] = None,
    ) -> bool:
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Node type (0-255)
            features: Optional sparse feature vector

        Returns:
            True if added successfully
        """
        with self._lock:
            if self._num_nodes >= self.max_nodes:
                return False

            if node_id in self._node_index:
                return False  # Node already exists

            # Add node
            index = self._num_nodes
            self._node_ids[index] = node_id
            self._node_types[index] = min(node_type, 255)

            if features is not None:
                # Store sparse features
                features = features.astype(np.float32)
                if len(features) > 64:
                    features = features[:64]  # Truncate if too long

                # Set non-zero features
                for i, value in enumerate(features):
                    if value != 0 and i < 64:
                        self._node_features[index, i] = value

            self._node_index[node_id] = index
            self._num_nodes += 1

            return True

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: int = 0,
        weight: float = 1.0,
    ) -> bool:
        """Add an edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type (0-255)
            weight: Edge weight

        Returns:
            True if added successfully
        """
        with self._lock:
            if self._num_edges >= self.max_edges:
                return False

            # Check that nodes exist
            if source_id not in self._node_index or target_id not in self._node_index:
                return False

            # Add edge
            index = self._num_edges
            self._edge_sources[index] = source_id
            self._edge_targets[index] = target_id
            self._edge_types[index] = min(edge_type, 255)
            self._edge_weights[index] = np.float16(weight)

            self._num_edges += 1

            return True

    def get_neighbors(self, node_id: int) -> List[Tuple[int, int, float]]:
        """Get neighbors of a node.

        Args:
            node_id: Node ID

        Returns:
            List of (neighbor_id, edge_type, weight) tuples
        """
        with self._lock:
            neighbors = []

            # Find outgoing edges
            for i in range(self._num_edges):
                if self._edge_sources[i] == node_id:
                    neighbors.append(
                        (
                            int(self._edge_targets[i]),
                            int(self._edge_types[i]),
                            float(self._edge_weights[i]),
                        )
                    )

            return neighbors

    def get_node_features(self, node_id: int) -> Optional[np.ndarray]:
        """Get features for a node.

        Args:
            node_id: Node ID

        Returns:
            Feature vector or None if not found
        """
        with self._lock:
            if node_id not in self._node_index:
                return None

            index = self._node_index[node_id]
            return self._node_features[index].toarray().flatten()

    def compact_storage(self):
        """Compact storage by removing unused space."""
        with self._lock:
            if self._num_nodes == self.max_nodes and self._num_edges == self.max_edges:
                return  # Already at capacity

            # Compact node storage
            if self._num_nodes < self.max_nodes:
                self._node_ids = self._node_ids[: self._num_nodes].copy()
                self._node_types = self._node_types[: self._num_nodes].copy()
                self._node_features = self._node_features[: self._num_nodes].copy()
                self.max_nodes = self._num_nodes

            # Compact edge storage
            if self._num_edges < self.max_edges:
                self._edge_sources = self._edge_sources[: self._num_edges].copy()
                self._edge_targets = self._edge_targets[: self._num_edges].copy()
                self._edge_types = self._edge_types[: self._num_edges].copy()
                self._edge_weights = self._edge_weights[: self._num_edges].copy()
                self.max_edges = self._num_edges

    def memory_usage_stats(self) -> Dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Memory usage breakdown in MB
        """
        with self._lock:
            nodes_mb = (
                self._node_ids.nbytes
                + self._node_types.nbytes
                + self._node_features.data.nbytes
            ) / (1024 * 1024)

            edges_mb = (
                self._edge_sources.nbytes
                + self._edge_targets.nbytes
                + self._edge_types.nbytes
                + self._edge_weights.nbytes
            ) / (1024 * 1024)

            total_mb = nodes_mb + edges_mb

            # Calculate efficiency
            utilization = (
                self._num_nodes / self.max_nodes + self._num_edges / self.max_edges
            ) / 2

            return {
                "nodes_mb": nodes_mb,
                "edges_mb": edges_mb,
                "total_mb": total_mb,
                "num_nodes": self._num_nodes,
                "num_edges": self._num_edges,
                "utilization": utilization,
                "avg_node_size_bytes": nodes_mb * 1024 * 1024 / max(1, self._num_nodes),
                "avg_edge_size_bytes": edges_mb * 1024 * 1024 / max(1, self._num_edges),
            }


def create_efficient_belief_buffer(
    shape: Tuple[int, ...],
    buffer_size: int = 100,
    use_memory_mapping: bool = False,
) -> Union[LazyBeliefArray, MemoryMappedBuffer]:
    """Create efficient belief buffer using factory pattern.

    Args:
        shape: Belief state shape
        buffer_size: Number of belief states to buffer
        use_memory_mapping: Whether to use memory mapping for large buffers

    Returns:
        Efficient belief buffer
    """
    total_size_mb = np.prod(shape) * buffer_size * 4 / (1024 * 1024)  # Assume float32

    if (
        use_memory_mapping or total_size_mb > 100
    ):  # Use memory mapping for large buffers
        buffer_shape = (buffer_size,) + shape
        return MemoryMappedBuffer(buffer_shape, dtype=np.float32)
    else:
        # Use lazy belief array for smaller buffers
        return LazyBeliefArray(shape, dtype=np.float32)


def benchmark_data_structures():
    """Benchmark memory-efficient data structures."""
    logger.info("Benchmarking memory-efficient data structures...")

    results = {}

    # Benchmark LazyBeliefArray
    logger.info("Testing LazyBeliefArray...")
    belief_shape = (50, 50)
    lazy_belief = LazyBeliefArray(belief_shape, dtype=np.float32)

    # Create sparse belief
    sparse_belief = np.zeros(belief_shape, dtype=np.float32)
    sparse_belief[10:15, 10:15] = np.random.random((5, 5))
    lazy_belief.update(sparse_belief)

    # Test compression
    compressed = lazy_belief.compress_if_beneficial()

    results["lazy_belief"] = {
        "memory_usage_mb": lazy_belief.memory_usage(),
        "compression_ratio": lazy_belief.stats.compression_ratio,
        "compressed": compressed,
    }

    # Benchmark CompactActionHistory
    logger.info("Testing CompactActionHistory...")
    action_history = CompactActionHistory(max_actions=1000, action_space_size=10)

    import time

    for i in range(500):
        action_history.add_action(i % 10, time.time() + i, np.random.random())

    stats = action_history.get_action_statistics()
    results["action_history"] = {
        "memory_usage_bytes": action_history.memory_usage_bytes(),
        "stored_actions": stats["count"],
        "action_distribution": stats["action_distribution"],
    }

    # Benchmark EfficientTemporalSequence
    logger.info("Testing EfficientTemporalSequence...")
    temporal_seq = EfficientTemporalSequence(max_length=1000, feature_dim=32)

    for i in range(200):
        state = np.random.random(32).astype(np.float32)
        temporal_seq.add_state(state, time.time() + i)

    memory_stats = temporal_seq.memory_usage_stats()
    results["temporal_sequence"] = memory_stats

    # Benchmark CompactKnowledgeGraph
    logger.info("Testing CompactKnowledgeGraph...")
    kg = CompactKnowledgeGraph(max_nodes=100, max_edges=500)

    # Add nodes and edges
    for i in range(50):
        features = np.random.random(64).astype(np.float32)
        features[features < 0.8] = 0  # Make sparse
        kg.add_node(i, node_type=i % 5, features=features)

    for i in range(200):
        source = np.random.randint(0, 50)
        target = np.random.randint(0, 50)
        kg.add_edge(source, target, edge_type=i % 3, weight=np.random.random())

    kg_stats = kg.memory_usage_stats()
    results["knowledge_graph"] = kg_stats

    logger.info(f"Benchmark results: {results}")
    return results
