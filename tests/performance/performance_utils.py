"""Utilities for performance testing that use real computation instead of sleep."""

import time
import numpy as np
from typing import Optional


def cpu_work(duration_seconds: float, intensity: str = "medium") -> None:
    """
    Perform actual CPU work for approximately the specified duration.
    
    Args:
        duration_seconds: Approximate duration to perform work
        intensity: Work intensity - "light", "medium", or "heavy"
    """
    start_time = time.time()
    
    # Adjust iterations based on intensity
    intensity_multipliers = {
        "light": 1000,
        "medium": 5000,
        "heavy": 20000
    }
    
    base_iterations = intensity_multipliers.get(intensity, 5000)
    
    # Perform computation until duration is reached
    while time.time() - start_time < duration_seconds:
        # Matrix operations to consume CPU
        size = min(50, int(base_iterations * duration_seconds))
        if size > 0:
            matrix = np.random.rand(size, size)
            _ = np.linalg.svd(matrix, compute_uv=False)  # Singular value decomposition
        else:
            # For very short durations, do simple arithmetic
            _ = sum(i**2 for i in range(100))


def memory_work(size_mb: float = 1.0) -> None:
    """
    Allocate and perform operations on memory.
    
    Args:
        size_mb: Size of memory to allocate in megabytes
    """
    # Allocate memory
    elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
    data = np.random.rand(elements)
    
    # Perform operations to ensure memory is actually used
    _ = np.fft.fft(data[:1000]).real.sum()
    _ = data.mean()
    _ = data.std()


def io_work(operations: int = 100) -> None:
    """
    Simulate I/O operations using in-memory operations.
    
    Args:
        operations: Number of I/O-like operations to perform
    """
    # Simulate I/O with string operations and data structure manipulations
    data = []
    for i in range(operations):
        # String concatenation (simulates write)
        s = f"Operation {i}: " + "x" * 100
        data.append(s)
        
        # Dictionary operations (simulates key-value store)
        d = {f"key_{j}": f"value_{j}" for j in range(10)}
        _ = d.get(f"key_{i % 10}")
    
    # Sort to simulate index operations
    data.sort()


def adaptive_cpu_work(target_duration_seconds: float, max_iterations: int = 1000000) -> float:
    """
    Perform CPU work that adapts to the system's speed to match target duration.
    
    Args:
        target_duration_seconds: Target duration for the work
        max_iterations: Maximum iterations to prevent infinite loops
        
    Returns:
        Actual duration taken
    """
    start_time = time.time()
    iterations = 0
    
    # Start with a small workload and increase if needed
    chunk_size = 1000
    
    while time.time() - start_time < target_duration_seconds and iterations < max_iterations:
        # Perform some CPU-intensive work
        _ = sum(i**2 for i in range(chunk_size))
        iterations += chunk_size
        
        # Adapt chunk size based on remaining time
        elapsed = time.time() - start_time
        if elapsed > 0:
            remaining = target_duration_seconds - elapsed
            rate = iterations / elapsed
            chunk_size = min(10000, max(100, int(rate * remaining / 10)))
    
    return time.time() - start_time


def replace_sleep(duration_seconds: float, work_type: str = "cpu") -> None:
    """
    Replace sleep operations with actual work.
    
    Args:
        duration_seconds: Duration to simulate
        work_type: Type of work - "cpu", "memory", "io", or "mixed"
    """
    if work_type == "cpu":
        cpu_work(duration_seconds)
    elif work_type == "memory":
        memory_work(duration_seconds * 10)  # Scale memory based on duration
    elif work_type == "io":
        io_work(int(duration_seconds * 1000))  # Scale operations based on duration
    elif work_type == "mixed":
        # Split time between different work types
        third = duration_seconds / 3
        cpu_work(third, "light")
        memory_work(third * 10)
        io_work(int(third * 1000))
    else:
        # Default to CPU work
        cpu_work(duration_seconds)