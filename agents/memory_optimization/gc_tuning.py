#!/usr/bin/env python3
"""Garbage Collection Tuning for Multi-Agent Memory Optimization.

This module implements advanced garbage collection strategies and tuning
for the multi-agent system to optimize memory usage and reduce GC overhead.

Key features:
- Adaptive GC thresholds based on memory pressure
- Generation-specific GC tuning
- GC pause minimization for real-time performance
- Memory usage monitoring and automatic adjustment
- Agent-aware GC scheduling
"""

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class GCStats:
    """Garbage collection statistics."""

    gen0_collections: int = 0
    gen1_collections: int = 0
    gen2_collections: int = 0
    total_gc_time_ms: float = 0.0
    avg_gc_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    gc_effectiveness: float = 0.0  # Memory freed / time spent


class AdaptiveGCTuner:
    """Adaptive garbage collection tuner for multi-agent systems."""

    def __init__(
        self,
        base_threshold0: int = 700,  # Generation 0 threshold
        base_threshold1: int = 10,  # Generation 1 threshold
        base_threshold2: int = 10,  # Generation 2 threshold
        enable_auto_tuning: bool = True,
        target_gc_overhead: float = 0.05,  # 5% max GC overhead
        min_gc_interval_ms: float = 100,  # Min 100ms between GCs
    ):
        """Initialize the adaptive GC tuner.

        Args:
            base_threshold0: Base threshold for generation 0
            base_threshold1: Base threshold for generation 1
            base_threshold2: Base threshold for generation 2
            enable_auto_tuning: Enable automatic threshold adjustment
            target_gc_overhead: Target GC overhead as fraction of runtime
            min_gc_interval_ms: Minimum interval between GC runs
        """
        self.base_thresholds = (
            base_threshold0,
            base_threshold1,
            base_threshold2,
        )
        self.current_thresholds = list(self.base_thresholds)
        self.enable_auto_tuning = enable_auto_tuning
        self.target_gc_overhead = target_gc_overhead
        self.min_gc_interval_ms = min_gc_interval_ms

        # GC statistics tracking
        self.stats = GCStats()
        self._gc_history: List[Tuple[float, int, float]] = []  # (timestamp, gen, duration)
        self._last_gc_time = 0.0
        self._lock = threading.RLock()

        # Memory pressure tracking
        self._memory_pressure = 0.0
        self._agent_count = 0

        # Apply initial configuration
        self._apply_gc_settings()

        # Install GC callback for monitoring
        gc.callbacks.append(self._gc_callback)

        logger.info(
            f"Initialized adaptive GC tuner with thresholds:"
            f" {self.current_thresholds}"
        )

    def _apply_gc_settings(self):
        """Apply current GC threshold settings."""
        gc.set_threshold(*self.current_thresholds)

        # Enable automatic garbage collection
        gc.enable()

        # Configure GC debug flags for monitoring (in debug mode only)
        if logger.isEnabledFor(logging.DEBUG):
            gc.set_debug(gc.DEBUG_STATS)

    def _gc_callback(self, phase: str, info: Dict[str, Any]):
        """Handle garbage collector callback.

        Args:
            phase: GC phase ('start' or 'stop')
            info: GC information dict
        """
        if phase == "start":
            self._gc_start_time = time.time()
        elif phase == "stop":
            duration = (time.time() - self._gc_start_time) * 1000
            generation = info.get("generation", 0)

            with self._lock:
                self._gc_history.append((time.time(), generation, duration))
                self._update_stats(generation, duration)

                # Auto-tune if enabled
                if self.enable_auto_tuning:
                    self._auto_tune_thresholds()

    def _update_stats(self, generation: int, duration_ms: float):
        """Update GC statistics.

        Args:
            generation: GC generation (0, 1, or 2)
            duration_ms: GC duration in milliseconds
        """
        if generation == 0:
            self.stats.gen0_collections += 1
        elif generation == 1:
            self.stats.gen1_collections += 1
        else:
            self.stats.gen2_collections += 1

        self.stats.total_gc_time_ms += duration_ms

        total_collections = (
            self.stats.gen0_collections + self.stats.gen1_collections +
                self.stats.gen2_collections
        )

        if total_collections > 0:
            self.stats.avg_gc_time_ms = self.stats.total_gc_time_ms / total_collections

        # Update memory stats
        if psutil:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.stats.current_memory_mb = memory_info.rss / (1024 * 1024)
            self.stats.peak_memory_mb = max(self.stats.peak_memory_mb,
                self.stats.current_memory_mb)

    def _auto_tune_thresholds(self):
        """Automatically tune GC thresholds based on performance metrics."""
        # Calculate recent GC overhead
        recent_window = 60.0  # 60 seconds
        current_time = time.time()
        recent_gcs = [
            (gen, duration)
            for ts, gen, duration in self._gc_history
            if current_time - ts < recent_window
        ]

        if not recent_gcs:
            return

        # Calculate overhead by generation
        gen_overheads = [0.0, 0.0, 0.0]
        gen_counts = [0, 0, 0]

        for gen, duration in recent_gcs:
            gen_overheads[gen] += duration
            gen_counts[gen] += 1

        total_gc_time = sum(gen_overheads)
        gc_overhead = total_gc_time / (recent_window * 1000)  # Fraction of time in GC

        # Adjust thresholds based on overhead and memory pressure
        if gc_overhead > self.target_gc_overhead:
            # Too much GC overhead - increase thresholds
            self._increase_thresholds()
        elif gc_overhead < self.target_gc_overhead * 0.5 and
            self._memory_pressure > 0.7:
            # Low GC overhead but high memory pressure - decrease thresholds
            self._decrease_thresholds()

        # Agent-aware adjustments
        if self._agent_count > 20:
            # Many agents - be more aggressive with gen0
            self.current_thresholds[0] = int(
                max(
                    self.base_thresholds[0] // 2,
                    min(self.current_thresholds[0], self.base_thresholds[0]),
                )
            )

        # Apply new settings
        self._apply_gc_settings()

    def _increase_thresholds(self):
        """Increase GC thresholds to reduce overhead."""
        with self._lock:
            for i in range(3):
                self.current_thresholds[i] = int(
                    min(
                        self.current_thresholds[i] * 1.2,
                        self.base_thresholds[i] * 3,
                    )
                )

            logger.debug(f"Increased GC thresholds to: {self.current_thresholds}")

    def _decrease_thresholds(self):
        """Decrease GC thresholds to free memory more aggressively."""
        with self._lock:
            for i in range(3):
                self.current_thresholds[i] = int(
                    max(
                        self.current_thresholds[i] * 0.8,
                        self.base_thresholds[i] // 2,
                    )
                )

            logger.debug(f"Decreased GC thresholds to: {self.current_thresholds}")

    def update_memory_pressure(self, pressure: float):
        """Update memory pressure metric (0.0 to 1.0).

        Args:
            pressure: Current memory pressure
        """
        with self._lock:
            self._memory_pressure = max(0.0, min(1.0, pressure))

    def update_agent_count(self, count: int):
        """Update active agent count for tuning.

        Args:
            count: Number of active agents
        """
        with self._lock:
            self._agent_count = count

    def force_collection(self, generation: int = 2) -> float:
        """Force a garbage collection with timing.

        Args:
            generation: Generation to collect (0, 1, or 2)

        Returns:
            Collection duration in milliseconds
        """
        start_time = time.time()

        # Check minimum interval
        if start_time - self._last_gc_time < self.min_gc_interval_ms / 1000:
            logger.debug("Skipping GC due to minimum interval")
            return 0.0

        # Disable GC during forced collection to avoid recursion
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            collected = gc.collect(generation)
            duration = (time.time() - start_time) * 1000

            logger.debug(
                f"Forced GC gen{generation}: collected {collected} objects in {duration:.1f}ms"
            )

            self._last_gc_time = time.time()
            return duration

        finally:
            if gc_was_enabled:
                gc.enable()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive GC statistics.

        Returns:
            Dictionary of GC statistics
        """
        with self._lock:
            gc_stats = gc.get_stats()

            return {
                "thresholds": {
                    "current": list(self.current_thresholds),
                    "base": list(self.base_thresholds),
                },
                "collections": {
                    "gen0": self.stats.gen0_collections,
                    "gen1": self.stats.gen1_collections,
                    "gen2": self.stats.gen2_collections,
                    "total": (
                        self.stats.gen0_collections
                        + self.stats.gen1_collections
                        + self.stats.gen2_collections
                    ),
                },
                "timing": {
                    "total_gc_time_ms": self.stats.total_gc_time_ms,
                    "avg_gc_time_ms": self.stats.avg_gc_time_ms,
                },
                "memory": {
                    "current_mb": self.stats.current_memory_mb,
                    "peak_mb": self.stats.peak_memory_mb,
                },
                "gc_stats": gc_stats,
                "memory_pressure": self._memory_pressure,
                "agent_count": self._agent_count,
            }

    def shutdown(self):
        """Shutdown the GC tuner and restore defaults."""
        # Remove callback
        try:
            gc.callbacks.remove(self._gc_callback)
        except ValueError:
            pass

        # Restore default thresholds
        gc.set_threshold(700, 10, 10)

        logger.info("GC tuner shutdown complete")


class GCContextManager:
    """Context manager for optimized GC behavior during critical operations."""

    def __init__(self, tuner: AdaptiveGCTuner):
        """Initialize GC context manager.

        Args:
            tuner: The adaptive GC tuner instance
        """
        self.tuner = tuner

    @contextmanager
    def batch_operation(self, disable_gc: bool = True):
        """Context for batch operations with optional GC disable.

        Args:
            disable_gc: Whether to disable GC during operation

        Yields:
            None
        """
        gc_was_enabled = gc.isenabled()

        if disable_gc:
            gc.disable()

        try:
            yield
        finally:
            if disable_gc and gc_was_enabled:
                gc.enable()
                # Force collection after batch
                self.tuner.force_collection(0)

    @contextmanager
    def low_latency(self):
        """Context for low-latency operations with deferred GC.

        Yields:
            None
        """
        # Temporarily increase thresholds
        old_thresholds = list(self.tuner.current_thresholds)

        with self.tuner._lock:
            # Double thresholds for low latency
            for i in range(3):
                self.tuner.current_thresholds[i] *= 2
            self.tuner._apply_gc_settings()

        try:
            yield
        finally:
            # Restore thresholds
            with self.tuner._lock:
                self.tuner.current_thresholds = old_thresholds
                self.tuner._apply_gc_settings()


# Global GC tuner instance
_global_gc_tuner: Optional[AdaptiveGCTuner] = None


def get_gc_tuner() -> AdaptiveGCTuner:
    """Get the global GC tuner instance.

    Returns:
        Global AdaptiveGCTuner instance
    """
    global _global_gc_tuner
    if _global_gc_tuner is None:
        _global_gc_tuner = AdaptiveGCTuner()
    return _global_gc_tuner


def optimize_gc_for_agents(agent_count: int, memory_limit_mb: float):
    """Optimize GC settings for a specific agent configuration.

    Args:
        agent_count: Number of agents
        memory_limit_mb: Total memory limit in MB
    """
    tuner = get_gc_tuner()

    # Calculate memory pressure
    if psutil:
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024)
        pressure = current_memory / memory_limit_mb
    else:
        pressure = 0.5  # Default moderate pressure

    tuner.update_agent_count(agent_count)
    tuner.update_memory_pressure(pressure)

    # Force initial tuning
    if tuner.enable_auto_tuning:
        tuner._auto_tune_thresholds()

    logger.info(f"Optimized GC for {agent_count} agents with {memory_limit_mb}MB limit")
