#!/usr/bin/env python3
"""Belief state compression strategies for memory optimization.

This module implements compression techniques identified in Task 5.2:
- Sparse matrix representation for beliefs
- Incremental belief updates
- Belief state pooling
- Shared belief components
- Adaptive precision
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# Module-level constants to avoid B008 mutable default arguments
DEFAULT_DTYPE = np.dtype(np.float32)


@dataclass
class SparseBeliefState:
    """Compressed sparse representation of belief state."""

    data: np.ndarray  # Non-zero values
    indices: np.ndarray  # Flat indices of non-zero values
    shape: Tuple[int, ...]  # Original shape
    dtype: np.dtype = field(default_factory=lambda: DEFAULT_DTYPE)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.data)

    def memory_usage(self) -> int:
        """Calculate memory usage in bytes."""
        # Data array + indices array + overhead
        data_bytes = self.data.nbytes
        indices_bytes = self.indices.nbytes
        overhead = 64  # Approximate overhead for object and attributes
        return data_bytes + indices_bytes + overhead

    def to_dense(self) -> np.ndarray:
        """Convert back to dense format."""
        dense = np.zeros(self.shape, dtype=self.dtype)
        flat_dense = dense.ravel()
        flat_dense[self.indices] = self.data
        return dense

    def to_scipy_sparse(self) -> sparse.csr_matrix:
        """Convert to scipy sparse matrix."""
        # Convert flat indices to 2D indices for matrix
        if len(self.shape) == 2:
            rows = self.indices // self.shape[1]
            cols = self.indices % self.shape[1]
            return sparse.csr_matrix(
                (self.data, (rows, cols)), shape=self.shape, dtype=self.dtype
            )
        else:
            raise ValueError("Conversion to scipy sparse only supports 2D arrays")


class BeliefCompressor:
    """Compressor for belief states using various strategies."""

    def __init__(self, sparsity_threshold: float = 0.9):
        """Initialize compressor.

        Args:
            sparsity_threshold: Minimum sparsity ratio to use compression (default 0.9 = 90% zeros)
        """
        self.sparsity_threshold = sparsity_threshold
        self._component_cache: Dict[str, Any] = {}

    def compress(
        self, belief: np.ndarray, dtype: np.dtype = DEFAULT_DTYPE
    ) -> SparseBeliefState:
        """Compress a belief state to sparse format.

        Args:
            belief: Dense belief array
            dtype: Data type for compressed values (default float32)

        Returns:
            Compressed sparse belief state
        """
        # Convert to specified dtype
        belief = belief.astype(dtype)

        # Find non-zero elements
        sparsity = 1 - (np.count_nonzero(belief) / belief.size)

        if sparsity < self.sparsity_threshold:
            logger.debug(
                f"Belief sparsity {sparsity:.1%} below threshold, using dense storage"
            )
            # For non-sparse beliefs, store all values
            data = belief.ravel()
            indices = np.arange(belief.size, dtype=np.int32)
        else:
            # Extract non-zero values and their indices
            nonzero_indices = np.where(belief.ravel() != 0)[0]
            data = belief.ravel()[nonzero_indices]
            indices = nonzero_indices.astype(np.int32)

        return SparseBeliefState(
            data=data, indices=indices, shape=belief.shape, dtype=dtype
        )

    def decompress(self, sparse_belief: SparseBeliefState) -> np.ndarray:
        """Decompress a sparse belief state to dense format.

        Args:
            sparse_belief: Compressed sparse belief

        Returns:
            Dense belief array
        """
        return sparse_belief.to_dense()

    def incremental_update(
        self,
        sparse_belief: SparseBeliefState,
        update: np.ndarray,
        learning_rate: float = 0.1,
    ) -> SparseBeliefState:
        """Apply incremental update to compressed belief.

        Args:
            sparse_belief: Current compressed belief
            update: Dense update array
            learning_rate: Update learning rate

        Returns:
            Updated compressed belief
        """
        # Convert to dense for update (can be optimized further)
        dense_belief = self.decompress(sparse_belief)

        # Apply incremental update
        updated_belief = (1 - learning_rate) * dense_belief + learning_rate * update

        # Normalize if it's a probability distribution
        if np.abs(updated_belief.sum() - 1.0) < 0.1:  # Likely a probability
            updated_belief = updated_belief / updated_belief.sum()

        # Re-compress
        return self.compress(updated_belief, dtype=sparse_belief.dtype)

    def compress_with_sharing(
        self,
        belief: np.ndarray,
        base_components: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SparseBeliefState, Dict[str, Any]]:
        """Compress belief with component sharing.

        Args:
            belief: Dense belief array
            base_components: Shared components from similar beliefs

        Returns:
            Tuple of (compressed belief, shareable components)
        """
        # Identify structure (non-zero positions)
        nonzero_mask = belief != 0
        structure_key = hash(nonzero_mask.tobytes())

        if base_components and "structure" in base_components:
            # Check if structure matches
            base_structure = base_components["structure"]
            if np.array_equal(nonzero_mask, base_structure):
                # Reuse structure, only store values
                indices = base_components["indices"]
                data = belief.ravel()[indices]
                # Reuse the exact same structure object
                components = {
                    "structure": base_structure,  # Reuse existing
                    "indices": indices,
                    "structure_key": base_components["structure_key"],
                }
            else:
                # Different structure, compress normally
                sparse_belief = self.compress(belief)
                indices = sparse_belief.indices
                data = sparse_belief.data
                components = {
                    "structure": nonzero_mask,
                    "indices": indices,
                    "structure_key": structure_key,
                }
        else:
            # No base components, compress normally
            sparse_belief = self.compress(belief)
            indices = sparse_belief.indices
            data = sparse_belief.data
            components = {
                "structure": nonzero_mask,
                "indices": indices,
                "structure_key": structure_key,
            }

        sparse_belief = SparseBeliefState(
            data=data, indices=indices, shape=belief.shape, dtype=belief.dtype
        )

        return sparse_belief, components

    def compress_batch(self, beliefs: List[np.ndarray]) -> List[SparseBeliefState]:
        """Compress multiple beliefs efficiently.

        Args:
            beliefs: List of dense belief arrays

        Returns:
            List of compressed beliefs
        """
        compressed = []

        # Group by shape for potential optimization
        shape_groups: Dict[Tuple[int, ...], List[Tuple[int, np.ndarray]]] = {}
        for i, belief in enumerate(beliefs):
            shape = belief.shape
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append((i, belief))

        # Compress each group
        for _shape, group in shape_groups.items():
            # Could implement further optimizations here
            for idx, belief in group:
                compressed.append((idx, self.compress(belief)))

        # Sort by original index and return
        compressed.sort(key=lambda x: x[0])
        return [comp for _, comp in compressed]


class CompressedBeliefPool:
    """Object pool for compressed belief states."""

    def __init__(
        self,
        pool_size: int,
        belief_shape: Tuple[int, ...],
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        """Initialize belief pool.

        Args:
            pool_size: Maximum number of pooled beliefs
            belief_shape: Shape of belief arrays
            dtype: Data type for beliefs
        """
        self.pool_size = pool_size
        self.belief_shape = belief_shape
        self.dtype = dtype
        self.available: deque = deque(maxlen=pool_size)
        self.in_use: Set[int] = set()

        # Pre-allocate pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Pre-allocate belief states in pool."""
        for _ in range(self.pool_size):
            # Create empty sparse belief
            sparse_belief = SparseBeliefState(
                data=np.array([], dtype=self.dtype),
                indices=np.array([], dtype=np.int32),
                shape=self.belief_shape,
                dtype=self.dtype,
            )
            self.available.append(sparse_belief)

    def acquire(self) -> SparseBeliefState:
        """Get a belief state from the pool.

        Returns:
            Available sparse belief state
        """
        if self.available:
            belief: SparseBeliefState = self.available.popleft()
        else:
            # Pool exhausted, create new one
            logger.warning("Belief pool exhausted, creating new belief")
            belief = SparseBeliefState(
                data=np.array([], dtype=self.dtype),
                indices=np.array([], dtype=np.int32),
                shape=self.belief_shape,
                dtype=self.dtype,
            )

        self.in_use.add(id(belief))
        return belief

    def release(self, belief: SparseBeliefState) -> None:
        """Return a belief state to the pool.

        Args:
            belief: Sparse belief to return
        """
        belief_id = id(belief)
        if belief_id in self.in_use:
            self.in_use.remove(belief_id)
            # Reset belief data
            belief.data = np.array([], dtype=self.dtype)
            belief.indices = np.array([], dtype=np.int32)
            # Return to pool if not full
            if len(self.available) < self.pool_size:
                self.available.append(belief)
        else:
            logger.warning("Attempting to release belief not from this pool")

    def clear(self):
        """Clear the pool."""
        self.available.clear()
        self.in_use.clear()
        self._initialize_pool()

    @property
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "available": len(self.available),
            "in_use": len(self.in_use),
            "total": len(self.available) + len(self.in_use),
            "pool_size": self.pool_size,
        }


def calculate_compression_stats(
    original: np.ndarray, compressed: SparseBeliefState
) -> Dict[str, float]:
    """Calculate compression statistics.

    Args:
        original: Original dense array
        compressed: Compressed sparse representation

    Returns:
        Dictionary of compression statistics
    """
    original_memory = original.nbytes
    compressed_memory = compressed.memory_usage()

    return {
        "original_memory_mb": original_memory / 1024 / 1024,
        "compressed_memory_mb": compressed_memory / 1024 / 1024,
        "compression_ratio": original_memory / compressed_memory,
        "space_savings_percent": (1 - compressed_memory / original_memory) * 100,
        "sparsity_percent": (1 - compressed.nnz / original.size) * 100,
    }
