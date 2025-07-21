#!/usr/bin/env python3
"""Agent Memory Lifecycle Management for efficient resource utilization.

This module implements comprehensive lifecycle management for agent memory
as part of Task 5.5: Design agent memory lifecycle management.

Key features:
- Agent memory lifecycle tracking
- Resource cleanup protocols
- Memory limits per agent
- Agent recycling mechanisms
- Memory-aware agent pool management
"""

import gc
import logging
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from .belief_compression import BeliefCompressor, CompressedBeliefPool
from .matrix_pooling import get_global_pool

logger = logging.getLogger(__name__)


class AgentLifecycleState(Enum):
    """Agent lifecycle states for memory management."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    HIBERNATING = "hibernating"
    RECYCLING = "recycling"
    TERMINATED = "terminated"


@dataclass
class MemoryUsageSnapshot:
    """Snapshot of agent memory usage at a point in time."""

    timestamp: float
    belief_memory_mb: float
    matrix_memory_mb: float
    total_memory_mb: float
    peak_memory_mb: float
    gc_collections: int
    state: AgentLifecycleState


@dataclass
class AgentMemoryProfile:
    """Memory profile for an agent with lifecycle tracking."""

    agent_id: str
    created_at: float
    last_accessed: float
    state: AgentLifecycleState = AgentLifecycleState.INITIALIZING

    # Memory limits and tracking
    memory_limit_mb: float = 50.0  # Default limit per agent
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Resource references
    belief_compressor: Optional[BeliefCompressor] = None
    belief_pool: Optional[CompressedBeliefPool] = None
    matrix_pool_refs: Set[int] = field(default_factory=set)

    # Usage statistics
    access_count: int = 0
    memory_snapshots: deque = field(default_factory=lambda: deque(maxlen=100))

    # Lifecycle events
    lifecycle_events: List[Tuple[float, str, Dict]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize additional fields after dataclass creation."""
        if self.belief_compressor is None:
            self.belief_compressor = BeliefCompressor()

        self.record_lifecycle_event("created",
            {"initial_limit_mb": self.memory_limit_mb})

    def record_lifecycle_event(self, event: str, metadata: Optional[Dict] = None):
        """Record a lifecycle event with timestamp."""
        self.lifecycle_events.append((time.time(), event, metadata or {}))
        if len(self.lifecycle_events) > 1000:  # Keep only recent events
            self.lifecycle_events = self.lifecycle_events[-500:]

    def update_memory_usage(
        self,
        belief_mb: float = 0.0,
        matrix_mb: float = 0.0,
        other_mb: float = 0.0,
    ):
        """Update memory usage tracking."""
        total_mb = belief_mb + matrix_mb + other_mb
        self.current_memory_mb = total_mb
        self.peak_memory_mb = max(self.peak_memory_mb, total_mb)
        self.last_accessed = time.time()
        self.access_count += 1

        # Take memory snapshot
        snapshot = MemoryUsageSnapshot(
            timestamp=time.time(),
            belief_memory_mb=belief_mb,
            matrix_memory_mb=matrix_mb,
            total_memory_mb=total_mb,
            peak_memory_mb=self.peak_memory_mb,
            gc_collections=len(gc.get_stats()),
            state=self.state,
        )
        self.memory_snapshots.append(snapshot)

    def check_memory_limit(self) -> bool:
        """Check if agent is within memory limits."""
        return self.current_memory_mb <= self.memory_limit_mb

    def get_memory_efficiency(self) -> float:
        """Calculate memory efficiency ratio (0-1, higher is better)."""
        if self.memory_limit_mb == 0:
            return 1.0
        return max(0.0, 1.0 - (self.current_memory_mb / self.memory_limit_mb))


class AgentMemoryLifecycleManager:
    """Central manager for agent memory lifecycles."""

    def __init__(
        self,
        global_memory_limit_mb: float = 1024.0,
        cleanup_threshold: float = 0.8,
        hibernation_idle_seconds: float = 300.0,
        recycling_idle_seconds: float = 600.0,
    ):
        """Initialize the lifecycle manager.

        Args:
            global_memory_limit_mb: Total memory limit for all agents
            cleanup_threshold: Memory usage ratio to trigger cleanup (0.8 = 80%)
            hibernation_idle_seconds: Seconds of inactivity before hibernation
            recycling_idle_seconds: Seconds of inactivity before recycling
        """
        self.global_memory_limit_mb = global_memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.hibernation_idle_seconds = hibernation_idle_seconds
        self.recycling_idle_seconds = recycling_idle_seconds

        # Agent tracking
        self._profiles: Dict[str, AgentMemoryProfile] = {}
        self._agent_refs: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()

        # Resource pools
        self._belief_pools: Dict[Tuple, CompressedBeliefPool] = {}
        self._matrix_pool = get_global_pool()

        # Cleanup and monitoring
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._monitoring_enabled = True

        # Statistics
        self.stats = {
            "total_agents_created": 0,
            "total_agents_recycled": 0,
            "total_agents_hibernated": 0,
            "total_memory_cleaned_mb": 0.0,
            "cleanup_cycles": 0,
            "peak_global_memory_mb": 0.0,
        }

        # Start background cleanup
        self._start_cleanup_thread()

    def register_agent(
        self,
        agent_id: str,
        memory_limit_mb: Optional[float] = None,
        agent_ref: Any = None,
    ) -> AgentMemoryProfile:
        """Register a new agent for lifecycle management.

        Args:
            agent_id: Unique identifier for the agent
            memory_limit_mb: Memory limit for this agent (uses default if None)
            agent_ref: Weak reference to the actual agent object

        Returns:
            AgentMemoryProfile for the registered agent
        """
        with self._lock:
            if agent_id in self._profiles:
                logger.warning(
                    f"Agent {agent_id} already registered, returning existing"
                    f" profile"
                )
                return self._profiles[agent_id]

            # Determine memory limit
            if memory_limit_mb is None:
                # Adaptive memory limit based on available resources
                memory_limit_mb = self._calculate_agent_memory_limit()

            # Create profile
            profile = AgentMemoryProfile(
                agent_id=agent_id,
                created_at=time.time(),
                last_accessed=time.time(),
                memory_limit_mb=memory_limit_mb,
            )

            # Setup resource pools for this agent
            self._setup_agent_resources(profile)

            self._profiles[agent_id] = profile

            # Track agent reference if provided
            if agent_ref is not None:
                self._agent_refs.add(agent_ref)

            self.stats["total_agents_created"] += 1

            logger.info(
                f"Registered agent {agent_id} with {memory_limit_mb:.1f}MB"
                f" limit"
            )
            profile.record_lifecycle_event(
                "registered",
                {
                    "memory_limit_mb": memory_limit_mb,
                    "total_agents": len(self._profiles),
                },
            )

            return profile

    def unregister_agent(self, agent_id: str, force_cleanup: bool = True):
        """Unregister an agent and cleanup its resources.

        Args:
            agent_id: Agent identifier to unregister
            force_cleanup: Whether to immediately cleanup resources
        """
        with self._lock:
            if agent_id not in self._profiles:
                logger.warning(f"Agent {agent_id} not found for unregistration")
                return

            profile = self._profiles[agent_id]
            profile.record_lifecycle_event("unregistering",
                {"force_cleanup": force_cleanup})

            if force_cleanup:
                self._cleanup_agent_resources(profile)

            profile.state = AgentLifecycleState.TERMINATED
            profile.record_lifecycle_event("terminated")

            # Remove from tracking
            del self._profiles[agent_id]

            logger.info(f"Unregistered agent {agent_id}")

    def get_agent_profile(self, agent_id: str) -> Optional[AgentMemoryProfile]:
        """Get the memory profile for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentMemoryProfile if found, None otherwise
        """
        with self._lock:
            return self._profiles.get(agent_id)

    def update_agent_memory(
        self,
        agent_id: str,
        belief_memory_mb: float = 0.0,
        matrix_memory_mb: float = 0.0,
        other_memory_mb: float = 0.0,
    ) -> bool:
        """Update memory usage for an agent.

        Args:
            agent_id: Agent identifier
            belief_memory_mb: Memory used by belief states
            matrix_memory_mb: Memory used by matrix operations
            other_memory_mb: Other memory usage

        Returns:
            True if within limits, False if over limit
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                logger.warning(f"Agent {agent_id} not found for memory update")
                return False

            profile.update_memory_usage(belief_memory_mb, matrix_memory_mb,
                other_memory_mb)

            # Check if agent needs state transition
            self._check_agent_state_transition(profile)

            # Update global peak memory
            total_memory = self.get_total_memory_usage()
            self.stats["peak_global_memory_mb"] = max(
                self.stats["peak_global_memory_mb"], total_memory
            )

            return profile.check_memory_limit()

    @contextmanager
    def agent_memory_context(self, agent_id: str):
        """Context manager for agent memory operations.

        Args:
            agent_id: Agent identifier

        Yields:
            AgentMemoryProfile for memory operations
        """
        profile = self.get_agent_profile(agent_id)
        if profile is None:
            raise ValueError(f"Agent {agent_id} not registered")

        profile.record_lifecycle_event("memory_context_enter")

        try:
            yield profile
        except Exception as e:
            profile.record_lifecycle_event(
                "memory_context_error",
                {"error": str(e), "error_type": type(e).__name__},
            )
            raise
        finally:
            profile.record_lifecycle_event("memory_context_exit")

    def hibernate_agent(self, agent_id: str) -> bool:
        """Hibernate an agent to reduce memory usage.

        Args:
            agent_id: Agent identifier

        Returns:
            True if hibernation successful
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                return False

            if profile.state in [
                AgentLifecycleState.HIBERNATING,
                AgentLifecycleState.TERMINATED,
            ]:
                return True

            # Compress belief states if not already compressed
            if profile.belief_compressor and profile.belief_pool:
                # Further compress by reducing pool size
                original_size = len(profile.belief_pool.available)
                profile.belief_pool.clear()
                freed_mb = original_size * 0.1  # Rough estimate
                self.stats["total_memory_cleaned_mb"] += freed_mb

            # Release matrix pool references
            self._release_matrix_pool_refs(profile)

            # Update state
            profile.state = AgentLifecycleState.HIBERNATING
            profile.record_lifecycle_event(
                "hibernated", {"memory_before_mb": profile.current_memory_mb}
            )

            # Force garbage collection
            gc.collect()

            self.stats["total_agents_hibernated"] += 1
            logger.info(f"Hibernated agent {agent_id}")
            return True

    def wake_agent(self, agent_id: str) -> bool:
        """Wake an agent from hibernation.

        Args:
            agent_id: Agent identifier

        Returns:
            True if wake successful
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                return False

            if profile.state != AgentLifecycleState.HIBERNATING:
                return True

            # Restore resources
            self._setup_agent_resources(profile)

            profile.state = AgentLifecycleState.ACTIVE
            profile.last_accessed = time.time()
            profile.record_lifecycle_event("awakened")

            logger.info(f"Awakened agent {agent_id}")
            return True

    def recycle_agent(self, agent_id: str) -> bool:
        """Recycle an agent for reuse with different configuration.

        Args:
            agent_id: Agent identifier

        Returns:
            True if recycling successful
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                return False

            # Clean up current resources
            self._cleanup_agent_resources(profile)

            # Reset memory tracking
            freed_memory = profile.current_memory_mb
            profile.current_memory_mb = 0.0
            profile.access_count = 0
            profile.memory_snapshots.clear()

            # Update state
            profile.state = AgentLifecycleState.RECYCLING
            profile.record_lifecycle_event("recycled",
                {"freed_memory_mb": freed_memory})

            self.stats["total_agents_recycled"] += 1
            self.stats["total_memory_cleaned_mb"] += freed_memory

            logger.info(f"Recycled agent {agent_id}, freed {freed_memory:.1f}MB")
            return True

    def cleanup_idle_agents(self) -> Dict[str, Union[int, float]]:
        """Cleanup idle agents based on lifecycle policies.

        Returns:
            Dictionary with cleanup statistics
        """
        current_time = time.time()
        cleanup_stats = {
            "hibernated": 0,
            "recycled": 0,
            "memory_freed_mb": 0.0,
        }

        with self._lock:
            agents_to_hibernate = []
            agents_to_recycle = []

            for agent_id, profile in self._profiles.items():
                idle_time = current_time - profile.last_accessed

                if (
                    profile.state == AgentLifecycleState.ACTIVE
                    and idle_time > self.hibernation_idle_seconds
                ):
                    agents_to_hibernate.append(agent_id)
                elif (
                    profile.state
                    in [
                        AgentLifecycleState.IDLE,
                        AgentLifecycleState.HIBERNATING,
                    ]
                    and idle_time > self.recycling_idle_seconds
                ):
                    agents_to_recycle.append(agent_id)

            # Perform cleanup outside of profile iteration
            for agent_id in agents_to_hibernate:
                if self.hibernate_agent(agent_id):
                    cleanup_stats["hibernated"] += 1

            for agent_id in agents_to_recycle:
                if self.recycle_agent(agent_id):
                    cleanup_stats["recycled"] += 1

        if cleanup_stats["hibernated"] > 0 or cleanup_stats["recycled"] > 0:
            cleanup_stats["memory_freed_mb"] = self.stats["total_memory_cleaned_mb"]
            logger.info(
                f"Cleanup cycle: hibernated {cleanup_stats['hibernated']}, "
                f"recycled {cleanup_stats['recycled']} agents"
            )

        return cleanup_stats

    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all agents.

        Returns:
            Total memory usage in MB
        """
        with self._lock:
            return sum(profile.current_memory_mb for profile in self._profiles.values())

    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0-1, higher means more pressure).

        Returns:
            Memory pressure ratio
        """
        total_memory = self.get_total_memory_usage()
        return min(1.0, total_memory / self.global_memory_limit_mb)

    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics.

        Returns:
            Dictionary with lifecycle and memory statistics
        """
        with self._lock:
            total_memory = self.get_total_memory_usage()
            memory_pressure = self.get_memory_pressure()

            state_counts: Dict[str, int] = defaultdict(int)
            memory_by_state: Dict[str, float] = defaultdict(float)

            for profile in self._profiles.values():
                state_counts[profile.state.value] += 1
                memory_by_state[profile.state.value] += profile.current_memory_mb

            return {
                "global": {
                    "total_agents": len(self._profiles),
                    "total_memory_mb": total_memory,
                    "memory_limit_mb": self.global_memory_limit_mb,
                    "memory_pressure": memory_pressure,
                    "cleanup_threshold": self.cleanup_threshold,
                },
                "agent_states": dict(state_counts),
                "memory_by_state": dict(memory_by_state),
                "lifecycle_stats": dict(self.stats),
                "pool_stats": {
                    "belief_pools": len(self._belief_pools),
                    "matrix_pool_stats": self._matrix_pool.get_statistics(),
                },
            }

    def force_cleanup(self) -> Dict[str, Any]:
        """Force immediate cleanup of all agents.

        Returns:
            Cleanup statistics
        """
        logger.info("Forcing immediate cleanup of all agents")

        cleanup_stats = self.cleanup_idle_agents()

        # Additional forced cleanup if still over threshold
        if self.get_memory_pressure() > self.cleanup_threshold:
            with self._lock:
                # Sort agents by last access time and hibernate oldest
                sorted_agents = sorted(self._profiles.items(),
                    key=lambda x: x[1].last_accessed)

                forced_hibernations = 0
                for agent_id, profile in sorted_agents:
                    if (
                        profile.state == AgentLifecycleState.ACTIVE
                        and self.get_memory_pressure() > self.cleanup_threshold
                    ):
                        if self.hibernate_agent(agent_id):
                            forced_hibernations += 1

                cleanup_stats["forced_hibernations"] = forced_hibernations

        # Force garbage collection
        collected = gc.collect()
        cleanup_stats["gc_collected"] = collected

        self.stats["cleanup_cycles"] += 1

        return cleanup_stats

    def shutdown(self):
        """Shutdown the lifecycle manager and cleanup all resources."""
        logger.info("Shutting down agent memory lifecycle manager")

        # Stop cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)

        # Cleanup all agents
        with self._lock:
            agent_ids = list(self._profiles.keys())
            for agent_id in agent_ids:
                self.unregister_agent(agent_id, force_cleanup=True)

        # Cleanup pools
        for pool in self._belief_pools.values():
            pool.clear()
        self._belief_pools.clear()

        if self._matrix_pool:
            self._matrix_pool.clear_all()

        logger.info("Agent memory lifecycle manager shutdown complete")

    def _calculate_agent_memory_limit(self) -> float:
        """Calculate appropriate memory limit for a new agent."""
        # Base limit
        base_limit = 20.0  # MB

        # Adjust based on current memory pressure
        pressure = self.get_memory_pressure()
        if pressure > 0.8:
            base_limit *= 0.5  # Reduce limit under high pressure
        elif pressure < 0.3:
            base_limit *= 1.5  # Increase limit under low pressure

        # Adjust based on number of agents
        num_agents = len(self._profiles)
        if num_agents > 10:
            base_limit *= 0.8  # Reduce per-agent limit with many agents

        return min(base_limit, self.global_memory_limit_mb / max(1, num_agents))

    def _setup_agent_resources(self, profile: AgentMemoryProfile):
        """Set up resource pools for an agent."""
        # Create belief pool if needed
        belief_shape = (20, 20)  # Default shape, could be customized
        pool_key = (belief_shape, np.float32)

        if pool_key not in self._belief_pools:
            self._belief_pools[pool_key] = CompressedBeliefPool(
                pool_size=10, belief_shape=belief_shape, dtype=np.float32
            )

        profile.belief_pool = self._belief_pools[pool_key]

    def _cleanup_agent_resources(self, profile: AgentMemoryProfile):
        """Cleanup resources for an agent."""
        # Release belief pool
        if profile.belief_pool:
            profile.belief_pool.clear()

        # Release matrix pool references
        self._release_matrix_pool_refs(profile)

        # Clear memory snapshots (keep only recent ones)
        if len(profile.memory_snapshots) > 10:
            recent = list(profile.memory_snapshots)[-10:]
            profile.memory_snapshots.clear()
            profile.memory_snapshots.extend(recent)

    def _release_matrix_pool_refs(self, profile: AgentMemoryProfile):
        """Release matrix pool references for an agent."""
        profile.matrix_pool_refs.clear()

    def _check_agent_state_transition(self, profile: AgentMemoryProfile):
        """Check if agent needs state transition based on usage patterns."""
        current_time = time.time()
        idle_time = current_time - profile.last_accessed

        # Transition to idle if not accessed recently
        if profile.state == AgentLifecycleState.ACTIVE and
            idle_time > 60.0:  # 1 minute idle
            profile.state = AgentLifecycleState.IDLE
            profile.record_lifecycle_event("transitioned_to_idle")

        # Transition back to active on access
        elif profile.state == AgentLifecycleState.IDLE and
            idle_time < 10.0:  # Recent access
            profile.state = AgentLifecycleState.ACTIVE
            profile.record_lifecycle_event("transitioned_to_active")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""

        def cleanup_worker():
            while not self._stop_cleanup.wait(30.0):  # Check every 30 seconds
                try:
                    if self.get_memory_pressure() > self.cleanup_threshold:
                        self.cleanup_idle_agents()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()


# Global instance for convenience
_global_lifecycle_manager: Optional[AgentMemoryLifecycleManager] = None


def get_global_lifecycle_manager() -> AgentMemoryLifecycleManager:
    """Get the global agent memory lifecycle manager.

    Returns:
        Global AgentMemoryLifecycleManager instance
    """
    global _global_lifecycle_manager
    if _global_lifecycle_manager is None:
        _global_lifecycle_manager = AgentMemoryLifecycleManager()
    return _global_lifecycle_manager


@contextmanager
def managed_agent_memory(agent_id: str, memory_limit_mb: Optional[float] = None):
    """Context manager for managed agent memory lifecycle.

    Args:
        agent_id: Agent identifier
        memory_limit_mb: Optional memory limit for the agent

    Yields:
        AgentMemoryProfile for the managed agent
    """
    manager = get_global_lifecycle_manager()

    # Register agent
    manager.register_agent(agent_id, memory_limit_mb)

    try:
        with manager.agent_memory_context(agent_id) as context_profile:
            yield context_profile
    finally:
        # Cleanup but don't unregister (allow reuse)
        manager.hibernate_agent(agent_id)


# Convenience functions
def register_agent_memory(
    agent_id: str, memory_limit_mb: Optional[float] = None
) -> AgentMemoryProfile:
    """Register an agent for memory lifecycle management."""
    return get_global_lifecycle_manager().register_agent(agent_id, memory_limit_mb)


def update_agent_memory_usage(
    agent_id: str,
    belief_mb: float = 0.0,
    matrix_mb: float = 0.0,
    other_mb: float = 0.0,
) -> bool:
    """Update memory usage for an agent."""
    return get_global_lifecycle_manager().update_agent_memory(
        agent_id, belief_mb, matrix_mb, other_mb
    )


def cleanup_agent_memory() -> Dict[str, Any]:
    """Force cleanup of agent memory."""
    return get_global_lifecycle_manager().force_cleanup()


def get_memory_statistics() -> Dict[str, Any]:
    """Get memory lifecycle statistics."""
    return get_global_lifecycle_manager().get_lifecycle_statistics()
