"""
WebSocket Resource Manager for Agent Lifecycle

Manages resource allocation, tracking, and cleanup for agents using
pooled WebSocket connections. Provides:
- Resource allocation and lifecycle management
- Connection sharing between agents
- Resource limits enforcement
- Automatic cleanup of stale resources
- Metrics collection
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from websocket.connection_pool import PooledConnection, WebSocketConnectionPool

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """States for agent resources."""

    ALLOCATED = "allocated"
    ACTIVE = "active"
    IDLE = "idle"
    RELEASED = "released"
    ERROR = "error"


class ResourceAllocationError(Exception):
    """Raised when resource allocation fails."""

    pass


class ResourceNotFoundError(Exception):
    """Raised when a resource is not found."""

    pass


class ResourceLimitExceededError(ResourceAllocationError):
    """Raised when resource limits are exceeded."""

    pass


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    max_agents_per_connection: int = 10
    max_memory_per_agent: int = 100 * 1024 * 1024  # 100MB in bytes
    max_cpu_per_agent: float = 1.0  # 1 CPU core
    agent_timeout: float = 3600.0  # 1 hour
    cleanup_interval: float = 60.0  # 1 minute
    enable_resource_limits: bool = True
    connection_reuse_strategy: str = "least_loaded"  # least_loaded, round_robin, affinity

    def __post_init__(self):
        """Validate configuration."""
        if self.max_agents_per_connection < 1:
            raise ValueError("max_agents_per_connection must be at least 1")
        if self.max_memory_per_agent < 0:
            raise ValueError("max_memory_per_agent must be non-negative")
        if self.max_cpu_per_agent < 0:
            raise ValueError("max_cpu_per_agent must be non-negative")
        if self.agent_timeout < 0:
            raise ValueError("agent_timeout must be non-negative")


@dataclass
class ResourceLimits:
    """Resource limits for an agent."""

    max_memory: int
    max_cpu: float
    timeout: float


class AgentResource:
    """Represents resources allocated to an agent."""

    def __init__(self, agent_id: str, connection_id: str, allocated_at: datetime):
        self.agent_id = agent_id
        self.connection_id = connection_id
        self.allocated_at = allocated_at
        self.activated_at: Optional[datetime] = None
        self.released_at: Optional[datetime] = None
        self.state = ResourceState.ALLOCATED
        self.memory_usage: int = 0  # bytes
        self.cpu_usage: float = 0.0  # cores
        self.metadata: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def mark_active(self):
        """Mark resource as active."""
        self.state = ResourceState.ACTIVE
        self.activated_at = datetime.utcnow()

    def mark_idle(self):
        """Mark resource as idle."""
        self.state = ResourceState.IDLE

    def mark_released(self):
        """Mark resource as released."""
        self.state = ResourceState.RELEASED
        self.released_at = datetime.utcnow()

    def mark_error(self):
        """Mark resource as in error state."""
        self.state = ResourceState.ERROR

    def update_usage(self, memory: Optional[int] = None, cpu: Optional[float] = None):
        """Update resource usage metrics."""
        if memory is not None:
            self.memory_usage = memory
        if cpu is not None:
            self.cpu_usage = cpu

    def set_metadata(self, key: str, value: Any):
        """Set metadata for the resource."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata value."""
        return self.metadata.get(key)

    def is_timed_out(self, timeout: float) -> bool:
        """Check if resource has timed out."""
        if self.state == ResourceState.RELEASED:
            return False
        elapsed = (datetime.utcnow() - self.allocated_at).total_seconds()
        return elapsed > timeout


class ResourceMetrics:
    """Tracks metrics for resource management."""

    def __init__(self):
        self.total_allocations = 0
        self.total_releases = 0
        self.allocation_failures = 0
        self.active_agents = 0
        self.peak_agents = 0
        self._lock = asyncio.Lock()

    def record_allocation(self):
        """Record a successful allocation."""
        self.total_allocations += 1
        self.active_agents += 1
        if self.active_agents > self.peak_agents:
            self.peak_agents = self.active_agents

    def record_release(self):
        """Record a resource release."""
        self.total_releases += 1
        self.active_agents = max(0, self.active_agents - 1)

    def record_allocation_failure(self):
        """Record an allocation failure."""
        self.allocation_failures += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        total_attempts = self.total_allocations + self.allocation_failures
        success_rate = self.total_allocations / total_attempts if total_attempts > 0 else 0.0

        return {
            "total_allocations": self.total_allocations,
            "total_releases": self.total_releases,
            "allocation_failures": self.allocation_failures,
            "active_agents": self.active_agents,
            "peak_agents": self.peak_agents,
            "success_rate": round(success_rate, 2),
        }


class AgentResourceManager:
    """Manages WebSocket resources for agent lifecycle."""

    def __init__(self, pool: WebSocketConnectionPool, config: Optional[ResourceConfig] = None):
        self.pool = pool
        self.config = config or ResourceConfig()
        self._resources: Dict[str, AgentResource] = {}  # agent_id -> resource
        self._connection_agents: Dict[str, Set[str]] = defaultdict(
            set
        )  # connection_id -> agent_ids
        self._metrics = ResourceMetrics()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self):
        """Start the resource manager."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Agent resource manager started")

    async def stop(self):
        """Stop the resource manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Release all resources
        await self.force_release_all()
        logger.info("Agent resource manager stopped")

    async def allocate_resource(
        self, agent_id: str, prefer_metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResource:
        """
        Allocate resources for an agent.

        Args:
            agent_id: Unique identifier for the agent
            prefer_metadata: Preferred connection metadata for affinity

        Returns:
            AgentResource object

        Raises:
            ResourceAllocationError: If allocation fails
        """
        async with self._lock:
            # Check if agent already has resources
            if agent_id in self._resources:
                raise ResourceAllocationError(f"Agent {agent_id} already has allocated resources")

            try:
                # Find or acquire a connection
                connection = await self._find_or_acquire_connection(prefer_metadata)

                # Cache the connection for reuse
                if not hasattr(self, "_connection_cache"):
                    self._connection_cache = {}
                self._connection_cache[connection.connection_id] = connection

                # Create resource
                resource = AgentResource(
                    agent_id=agent_id,
                    connection_id=connection.connection_id,
                    allocated_at=datetime.utcnow(),
                )

                # Track resource
                self._resources[agent_id] = resource
                self._connection_agents[connection.connection_id].add(agent_id)

                # Update connection metadata
                connection.set_metadata(
                    "agent_count", len(self._connection_agents[connection.connection_id])
                )
                connection.set_metadata(
                    "agents", list(self._connection_agents[connection.connection_id])
                )

                self._metrics.record_allocation()
                logger.info(
                    f"Allocated resources for agent {agent_id} on connection {connection.connection_id}"
                )

                return resource

            except Exception as e:
                self._metrics.record_allocation_failure()
                logger.error(f"Failed to allocate resources for agent {agent_id}: {e}")
                raise ResourceAllocationError(f"Failed to allocate resources: {e}")

    async def _find_or_acquire_connection(
        self, prefer_metadata: Optional[Dict[str, Any]] = None
    ) -> PooledConnection:
        """Find an existing connection with capacity or acquire a new one."""
        # Initialize cache if needed
        if not hasattr(self, "_connection_cache"):
            self._connection_cache = {}

        # Try to find existing connection with capacity
        for conn_id, agent_ids in self._connection_agents.items():
            if len(agent_ids) < self.config.max_agents_per_connection:
                # Return the cached connection if available
                if conn_id in self._connection_cache:
                    return self._connection_cache[conn_id]

        # No existing connection with capacity, acquire new one
        return await self.pool.acquire(prefer_metadata=prefer_metadata)

    async def activate_resource(self, agent_id: str):
        """Activate an allocated resource."""
        async with self._lock:
            resource = self._resources.get(agent_id)
            if not resource:
                raise ResourceNotFoundError(f"No resource found for agent {agent_id}")

            resource.mark_active()
            logger.debug(f"Activated resource for agent {agent_id}")

    async def release_resource(self, agent_id: str):
        """
        Release resources for an agent.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            resource = self._resources.get(agent_id)
            if not resource:
                logger.warning(f"No resource found for agent {agent_id}")
                return

            # Mark as released
            resource.mark_released()

            # Remove from tracking
            del self._resources[agent_id]
            self._connection_agents[resource.connection_id].discard(agent_id)

            # Update connection metadata
            connection_id = resource.connection_id
            remaining_agents = self._connection_agents[connection_id]

            # If no more agents on this connection, release it
            if not remaining_agents:
                del self._connection_agents[connection_id]
                await self.pool.release(connection_id)
                logger.info(f"Released connection {connection_id} (no more agents)")

            self._metrics.record_release()
            logger.info(f"Released resources for agent {agent_id}")

    async def update_resource_usage(
        self, agent_id: str, memory: Optional[int] = None, cpu: Optional[float] = None
    ):
        """
        Update resource usage for an agent.

        Args:
            agent_id: Agent identifier
            memory: Memory usage in bytes
            cpu: CPU usage in cores

        Raises:
            ResourceLimitExceededError: If limits are exceeded
        """
        async with self._lock:
            resource = self._resources.get(agent_id)
            if not resource:
                raise ResourceNotFoundError(f"No resource found for agent {agent_id}")

            # Check limits if enabled
            if self.config.enable_resource_limits:
                if memory is not None and memory > self.config.max_memory_per_agent:
                    raise ResourceLimitExceededError(
                        f"Memory usage {memory} exceeds limit {self.config.max_memory_per_agent}"
                    )
                if cpu is not None and cpu > self.config.max_cpu_per_agent:
                    raise ResourceLimitExceededError(
                        f"CPU usage {cpu} exceeds limit {self.config.max_cpu_per_agent}"
                    )

            resource.update_usage(memory=memory, cpu=cpu)
            logger.debug(f"Updated usage for agent {agent_id}: memory={memory}, cpu={cpu}")

    async def get_agent_connection(self, agent_id: str) -> Optional[PooledConnection]:
        """Get the connection assigned to an agent."""
        async with self._lock:
            resource = self._resources.get(agent_id)
            if not resource:
                return None

            # Return the cached connection
            if (
                hasattr(self, "_connection_cache")
                and resource.connection_id in self._connection_cache
            ):
                return self._connection_cache[resource.connection_id]

            # Fallback - this shouldn't happen in normal operation
            return None

    async def get_agent_resource(self, agent_id: str) -> Optional[AgentResource]:
        """Get resource information for an agent."""
        return self._resources.get(agent_id)

    async def _cleanup_loop(self):
        """Background task to cleanup stale resources."""
        while self._running:
            try:
                await self._cleanup_stale_resources()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _cleanup_stale_resources(self):
        """Clean up timed-out or stale resources."""
        async with self._lock:
            agents_to_release = []

            for agent_id, resource in self._resources.items():
                if resource.is_timed_out(self.config.agent_timeout):
                    agents_to_release.append(agent_id)
                    logger.warning(f"Agent {agent_id} timed out after {self.config.agent_timeout}s")

            # Release timed-out agents
            for agent_id in agents_to_release:
                await self.release_resource(agent_id)

    async def force_release_all(self):
        """Force release all resources."""
        async with self._lock:
            agent_ids = list(self._resources.keys())

            for agent_id in agent_ids:
                await self.release_resource(agent_id)

            logger.info(f"Force released {len(agent_ids)} agents")

    def get_metrics(self) -> Dict[str, Any]:
        """Get resource manager metrics."""
        total_memory = sum(r.memory_usage for r in self._resources.values())
        total_cpu = sum(r.cpu_usage for r in self._resources.values())

        active_agents = sum(1 for r in self._resources.values() if r.state == ResourceState.ACTIVE)

        metrics = self._metrics.get_summary()
        metrics.update(
            {
                "total_agents": len(self._resources),
                "active_agents": active_agents,
                "total_memory_usage": total_memory,
                "total_cpu_usage": total_cpu,
                "connections_in_use": len(self._connection_agents),
                "avg_agents_per_connection": (
                    len(self._resources) / len(self._connection_agents)
                    if self._connection_agents
                    else 0
                ),
            }
        )

        return metrics

    def get_resource_info(self) -> List[Dict[str, Any]]:
        """Get information about all resources."""
        info = []
        for resource in self._resources.values():
            info.append(
                {
                    "agent_id": resource.agent_id,
                    "connection_id": resource.connection_id,
                    "state": resource.state.value,
                    "allocated_at": resource.allocated_at.isoformat(),
                    "activated_at": (
                        resource.activated_at.isoformat() if resource.activated_at else None
                    ),
                    "memory_usage": resource.memory_usage,
                    "cpu_usage": resource.cpu_usage,
                    "metadata": resource.metadata,
                }
            )
        return info
