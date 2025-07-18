#!/usr/bin/env python3
"""Matrix operation memory pooling for PyMDP operations.

This module implements memory pooling strategies for matrix operations
to reduce memory allocation overhead and improve performance.

Based on Task 5.4: Create matrix operation memory pooling
"""

import logging
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PooledMatrix:
    """Container for a pooled matrix with metadata."""

    data: NDArray[np.float32]
    shape: Tuple[int, ...]
    dtype: np.dtype
    pool_id: str
    in_use: bool = False
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        """Validate matrix properties."""
        if self.data.shape != self.shape:
            raise ValueError(
                f"Data shape {self.data.shape} != declared shape {self.shape}"
            )
        if self.data.dtype != self.dtype:
            raise ValueError(
                f"Data dtype {self.data.dtype} != declared dtype {self.dtype}"
            )


@dataclass
class PoolStatistics:
    """Statistics for a matrix pool."""

    total_allocated: int = 0
    current_available: int = 0
    current_in_use: int = 0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class MatrixPool:
    """Memory pool for matrices of a specific shape and dtype."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        initial_size: int = 5,
        max_size: int = 50,
    ):
        """Initialize matrix pool.

        Args:
            shape: Shape of matrices in this pool
            dtype: Data type of matrices
            initial_size: Number of matrices to pre-allocate
            max_size: Maximum number of matrices to keep in pool
        """
        self.shape = shape
        self.dtype = dtype
        self.initial_size = initial_size
        self.max_size = max_size

        # Thread safety
        self._lock = threading.RLock()

        # Pool storage
        self._available: deque[PooledMatrix] = deque()
        self._in_use: Dict[int, PooledMatrix] = {}

        # Statistics
        self.stats = PoolStatistics()

        # Pre-allocate initial matrices
        self._preallocate()

    def _preallocate(self):
        """Pre-allocate initial pool of matrices."""
        for _ in range(self.initial_size):
            matrix = self._create_matrix()
            self._available.append(matrix)
            self.stats.total_allocated += 1
            self.stats.current_available += 1

    def _create_matrix(self) -> PooledMatrix:
        """Create a new pooled matrix."""
        import time

        data = np.zeros(self.shape, dtype=self.dtype)
        pool_id = f"{id(data)}_{time.time()}"

        matrix = PooledMatrix(
            data=data, shape=self.shape, dtype=self.dtype, pool_id=pool_id
        )

        self.stats.total_memory_bytes += data.nbytes
        return matrix

    def acquire(self) -> PooledMatrix:
        """Get a matrix from the pool.

        Returns:
            PooledMatrix ready for use
        """
        with self._lock:
            self.stats.total_requests += 1

            if self._available:
                # Reuse existing matrix
                matrix = self._available.popleft()
                self.stats.cache_hits += 1
                self.stats.current_available -= 1
            else:
                # Create new matrix if under limit
                if self.stats.total_allocated < self.max_size:
                    matrix = self._create_matrix()
                    self.stats.total_allocated += 1
                    self.stats.cache_misses += 1
                    logger.debug(f"Created new matrix for pool {self.shape}")
                else:
                    # Pool exhausted, log warning
                    logger.warning(
                        f"Matrix pool exhausted for shape {self.shape}"
                    )
                    matrix = self._create_matrix()  # Emergency allocation
                    self.stats.cache_misses += 1

            # Mark as in use
            matrix.in_use = True
            matrix.access_count += 1
            matrix.last_accessed = self._get_time()
            self._in_use[id(matrix)] = matrix
            self.stats.current_in_use += 1

            return matrix

    def release(self, matrix: PooledMatrix):
        """Return a matrix to the pool.

        Args:
            matrix: Matrix to return to pool
        """
        with self._lock:
            matrix_id = id(matrix)

            if matrix_id not in self._in_use:
                logger.warning(
                    f"Attempting to release matrix not from this pool"
                )
                return

            # Remove from in-use tracking
            del self._in_use[matrix_id]
            self.stats.current_in_use -= 1

            # Reset matrix data to zeros (clear sensitive data)
            matrix.data.fill(0)
            matrix.in_use = False

            # Return to pool if not over limit
            if len(self._available) < self.max_size:
                self._available.append(matrix)
                self.stats.current_available += 1
            else:
                # Pool is full, let garbage collector handle it
                self.stats.total_allocated -= 1
                self.stats.total_memory_bytes -= matrix.data.nbytes
                logger.debug(f"Discarded excess matrix from pool {self.shape}")

    def clear(self):
        """Clear all matrices from the pool."""
        with self._lock:
            self._available.clear()
            self._in_use.clear()
            self.stats = PoolStatistics()

    @staticmethod
    def _get_time() -> float:
        """Get current time for tracking."""
        import time

        return time.time()


class MatrixOperationPool:
    """Central manager for matrix operation memory pooling."""

    def __init__(self, enable_profiling: bool = False):
        """Initialize the matrix operation pool.

        Args:
            enable_profiling: Whether to enable detailed profiling
        """
        self._pools: Dict[Tuple[Tuple[int, ...], np.dtype], MatrixPool] = {}
        self._lock = threading.RLock()
        self.enable_profiling = enable_profiling

        # Operation-specific temporary storage
        self._temp_matrices: Dict[str, deque] = defaultdict(deque)

        # Global statistics
        self.global_stats = {
            "total_pools": 0,
            "total_matrices": 0,
            "total_memory_mb": 0.0,
            "operation_counts": defaultdict(int),
        }

    def get_pool(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.float32
    ) -> MatrixPool:
        """Get or create a pool for specific shape and dtype.

        Args:
            shape: Matrix shape
            dtype: Matrix data type

        Returns:
            MatrixPool for the specified configuration
        """
        with self._lock:
            key = (shape, dtype)

            if key not in self._pools:
                # Determine pool size based on matrix size
                matrix_size_mb = (
                    np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
                )

                if matrix_size_mb < 1:  # Small matrices
                    initial_size = 10
                    max_size = 100
                elif matrix_size_mb < 10:  # Medium matrices
                    initial_size = 5
                    max_size = 50
                else:  # Large matrices
                    initial_size = 2
                    max_size = 10

                self._pools[key] = MatrixPool(
                    shape, dtype, initial_size, max_size
                )
                self.global_stats["total_pools"] += 1
                logger.info(
                    f"Created matrix pool for shape {shape}, dtype {dtype}"
                )

            return self._pools[key]

    @contextmanager
    def allocate_matrix(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.float32
    ):
        """Context manager for temporary matrix allocation.

        Args:
            shape: Matrix shape
            dtype: Matrix data type

        Yields:
            Numpy array from pool
        """
        pool = self.get_pool(shape, dtype)
        matrix = pool.acquire()

        try:
            yield matrix.data
        finally:
            pool.release(matrix)

    @contextmanager
    def allocate_einsum_operands(
        self, *shapes: Tuple[int, ...], dtype: np.dtype = np.float32
    ):
        """Allocate multiple matrices for einsum operations.

        Args:
            shapes: Variable number of matrix shapes
            dtype: Data type for all matrices

        Yields:
            Tuple of numpy arrays
        """
        matrices = []
        pools = []

        try:
            # Acquire all matrices
            for shape in shapes:
                pool = self.get_pool(shape, dtype)
                matrix = pool.acquire()
                matrices.append(matrix)
                pools.append(pool)

            # Yield just the data arrays
            yield tuple(m.data for m in matrices)

        finally:
            # Release all matrices
            for matrix, pool in zip(matrices, pools):
                pool.release(matrix)

    def optimize_matrix_operation(
        self,
        operation: str,
        *operands: NDArray,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> NDArray:
        """Perform matrix operation with pooled memory.

        Args:
            operation: Operation type ('dot', 'matmul', 'einsum', etc.)
            operands: Input matrices
            out_shape: Expected output shape (if known)

        Returns:
            Result array (may be from pool)
        """
        if self.enable_profiling:
            self.global_stats["operation_counts"][operation] += 1

        if operation == "dot":
            if len(operands) != 2:
                raise ValueError("dot requires exactly 2 operands")

            a, b = operands

            # Handle different shapes for dot product
            if a.ndim == 1 and b.ndim == 1:
                # Dot product of two 1D arrays returns scalar
                return np.dot(a, b)
            elif a.ndim == 2 and b.ndim == 1:
                # Matrix-vector multiplication
                out_shape = out_shape or (a.shape[0],)
            elif a.ndim == 1 and b.ndim == 2:
                # Vector-matrix multiplication
                out_shape = out_shape or (b.shape[1],)
            else:
                # Matrix-matrix multiplication
                out_shape = out_shape or (a.shape[0], b.shape[1])

            with self.allocate_matrix(out_shape, a.dtype) as out:
                np.dot(a, b, out=out)
                return out.copy()  # Return copy to avoid pool corruption

        elif operation == "matmul":
            if len(operands) != 2:
                raise ValueError("matmul requires exactly 2 operands")

            a, b = operands
            # Determine output shape for matmul
            if a.ndim == 1 and b.ndim == 1:
                # Dot product
                return np.dot(a, b)
            elif a.ndim == 2 and b.ndim == 2:
                out_shape = out_shape or (a.shape[0], b.shape[1])
            else:
                # Let numpy figure out broadcasting
                result = np.matmul(a, b)
                return result

            with self.allocate_matrix(out_shape, a.dtype) as out:
                np.matmul(a, b, out=out)
                return out.copy()

        elif operation == "einsum":
            # For einsum, we need the subscripts
            if len(operands) < 2:
                raise ValueError(
                    "einsum requires subscripts and at least one operand"
                )

            subscripts = operands[0]
            arrays = operands[1:]

            # Compute result to get shape if not provided
            if out_shape is None:
                result = np.einsum(subscripts, *arrays)
                out_shape = result.shape
                return result
            else:
                with self.allocate_matrix(out_shape, arrays[0].dtype) as out:
                    np.einsum(subscripts, *arrays, out=out)
                    return out.copy()

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = {"global": dict(self.global_stats), "pools": {}}

            total_memory = 0
            total_matrices = 0

            for (shape, dtype), pool in self._pools.items():
                pool_key = f"{shape}_{dtype}"
                stats["pools"][pool_key] = {
                    "shape": shape,
                    "dtype": str(dtype),
                    "stats": pool.stats.__dict__,
                    "hit_rate": pool.stats.hit_rate,
                }

                total_memory += pool.stats.total_memory_bytes
                total_matrices += pool.stats.total_allocated

            stats["global"]["total_memory_mb"] = total_memory / (1024 * 1024)
            stats["global"]["total_matrices"] = total_matrices

            return stats

    def clear_all(self):
        """Clear all matrix pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            self._pools.clear()
            self._temp_matrices.clear()

            self.global_stats = {
                "total_pools": 0,
                "total_matrices": 0,
                "total_memory_mb": 0.0,
                "operation_counts": defaultdict(int),
            }


# Global instance for convenience
_global_pool = None


def get_global_pool() -> MatrixOperationPool:
    """Get the global matrix operation pool instance.

    Returns:
        Global MatrixOperationPool instance
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = MatrixOperationPool(enable_profiling=True)
    return _global_pool


@contextmanager
def pooled_matrix(shape: Tuple[int, ...], dtype: np.dtype = np.float32):
    """Convenience function for allocating pooled matrix.

    Args:
        shape: Matrix shape
        dtype: Matrix data type

    Yields:
        Numpy array from pool
    """
    pool = get_global_pool()
    with pool.allocate_matrix(shape, dtype) as matrix:
        yield matrix


def pooled_dot(a: NDArray, b: NDArray) -> NDArray:
    """Compute dot product using pooled memory.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Dot product result
    """
    pool = get_global_pool()
    return pool.optimize_matrix_operation("dot", a, b)


def pooled_matmul(a: NDArray, b: NDArray) -> NDArray:
    """Compute matrix multiplication using pooled memory.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Matrix multiplication result
    """
    pool = get_global_pool()
    return pool.optimize_matrix_operation("matmul", a, b)


def pooled_einsum(subscripts: str, *operands: NDArray) -> NDArray:
    """Compute einsum using pooled memory.

    Args:
        subscripts: Einsum subscripts
        operands: Input arrays

    Returns:
        Einsum result
    """
    pool = get_global_pool()
    return pool.optimize_matrix_operation("einsum", subscripts, *operands)
