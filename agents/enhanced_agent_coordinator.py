"""Enhanced Agent Coordinator with Connection Pooling and Resource Monitoring.

Integrates multi-agent coordination with connection pooling, circuit breaker patterns,
and comprehensive resource monitoring for optimal performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from agents.connection_pool_manager import (
    CircuitBreaker,
    ConnectionPoolConfig,
    EnhancedConnectionPoolManager,
    ResourceMonitor,
)
from agents.optimized_threadpool_manager import OptimizedThreadPoolManager
from database.enhanced_connection_manager import get_enhanced_db_manager

logger = logging.getLogger(__name__)


@dataclass
class CoordinationMetrics:
    """Metrics for agent coordination performance."""

    total_agents: int = 0
    active_agents: int = 0
    coordination_efficiency: float = 0.0
    avg_response_time_ms: float = 0.0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    circuit_breaker_triggers: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    coordination_times: List[float] = field(default_factory=list)
    agent_throughput: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)


class EnhancedAgentCoordinator:
    """Enhanced agent coordinator with connection pooling and resource monitoring."""

    def __init__(
        self,
        config: Optional[ConnectionPoolConfig] = None,
        database_url: Optional[str] = None,
    ):
        """Initialize enhanced agent coordinator."""
        self.config = config or ConnectionPoolConfig()

        # Initialize connection pool manager
        self.pool_manager = EnhancedConnectionPoolManager(self.config, database_url or "")

        # Initialize optimized thread pool manager
        self.thread_pool_manager = OptimizedThreadPoolManager(
            initial_workers=self.config.agent_pool_size,
            max_workers=self.config.agent_max_concurrent,
            scaling_threshold=0.8,
        )

        # Enhanced database connection manager
        self.db_manager = get_enhanced_db_manager(database_url) if database_url else None

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(self.config)

        # Circuit breaker for coordination failures
        self.coordination_circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            recovery_timeout=self.config.circuit_breaker_recovery_timeout,
            half_open_max_calls=self.config.circuit_breaker_half_open_max_calls,
        )

        # Agent registry and coordination state
        self.registered_agents: Dict[str, Any] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.coordination_metrics = CoordinationMetrics()

        # Thread safety
        self._agents_lock = Lock()
        self._metrics_lock = Lock()

        # Active coordination tasks
        self.active_coordinations: Set[str] = set()
        self._coordinations_lock = Lock()

        logger.info("Enhanced agent coordinator initialized")

    async def initialize(self):
        """Initialize all components."""
        await self.pool_manager.initialize()

        if self.db_manager:
            await self.db_manager.initialize()

        logger.info("Enhanced agent coordinator fully initialized")

    def register_agent(self, agent_id: str, agent: Any, metadata: Optional[Dict] = None) -> bool:
        """Register an agent with enhanced coordination."""
        try:
            with self._agents_lock:
                self.registered_agents[agent_id] = agent
                self.agent_states[agent_id] = {
                    "status": "idle",
                    "last_activity": time.time(),
                    "metadata": metadata or {},
                    "performance_metrics": {
                        "total_tasks": 0,
                        "successful_tasks": 0,
                        "failed_tasks": 0,
                        "avg_response_time": 0.0,
                    },
                }

            # Register with thread pool manager
            self.thread_pool_manager.register_agent(agent_id, agent)

            with self._metrics_lock:
                self.coordination_metrics.total_agents += 1
                self.coordination_metrics.active_agents += 1

            logger.info(f"Agent {agent_id} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        try:
            with self._agents_lock:
                if agent_id in self.registered_agents:
                    del self.registered_agents[agent_id]
                    del self.agent_states[agent_id]

            # Unregister from thread pool manager
            self.thread_pool_manager.unregister_agent(agent_id)

            with self._metrics_lock:
                self.coordination_metrics.total_agents -= 1
                if self.coordination_metrics.active_agents > 0:
                    self.coordination_metrics.active_agents -= 1

            logger.info(f"Agent {agent_id} unregistered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents with enhanced resource management."""
        coordination_id = coordination_request.get("coordination_id", f"coord_{int(time.time())}")
        agent_ids = coordination_request.get("agent_ids", [])
        operation = coordination_request.get("operation", "step")
        coordination_data = coordination_request.get("data", {})

        start_time = time.time()

        try:
            # Use circuit breaker for coordination
            return await self.coordination_circuit_breaker.call(
                self._execute_coordination,
                coordination_id,
                agent_ids,
                operation,
                coordination_data,
            )

        except Exception as e:
            logger.error(f"Coordination {coordination_id} failed: {e}")
            with self._metrics_lock:
                self.coordination_metrics.failed_coordinations += 1
            raise

        finally:
            coordination_time = time.time() - start_time
            with self._metrics_lock:
                self.coordination_metrics.coordination_times.append(coordination_time)
                # Keep only last 100 coordination times
                if len(self.coordination_metrics.coordination_times) > 100:
                    self.coordination_metrics.coordination_times.pop(0)

    async def _execute_coordination(
        self,
        coordination_id: str,
        agent_ids: List[str],
        operation: str,
        coordination_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute coordination with resource monitoring."""
        with self._coordinations_lock:
            self.active_coordinations.add(coordination_id)

        try:
            # Get current resource metrics
            resource_metrics = self.resource_monitor.get_current_metrics()

            # Check if we have enough resources
            if resource_metrics.get("avg_cpu_60s", 0) > 90:
                logger.warning(
                    f"High CPU usage ({resource_metrics['avg_cpu_60s']}%), "
                    "throttling coordination"
                )
                await asyncio.sleep(0.1)  # Brief throttling

            # Validate agents are registered
            valid_agents = []
            with self._agents_lock:
                for agent_id in agent_ids:
                    if agent_id in self.registered_agents:
                        valid_agents.append(agent_id)
                        self.agent_states[agent_id]["status"] = "coordinating"
                        self.agent_states[agent_id]["last_activity"] = time.time()
                    else:
                        logger.warning(f"Agent {agent_id} not registered for coordination")

            if not valid_agents:
                raise ValueError("No valid agents for coordination")

            # Submit tasks to thread pool
            futures = []
            for agent_id in valid_agents:
                future = self.thread_pool_manager.submit_task(
                    agent_id=agent_id,
                    operation=operation,
                    data=coordination_data,
                    priority=coordination_data.get("priority", 0),
                )
                futures.append((agent_id, future))

            # Collect results with timeout
            results = {}
            timeout = coordination_data.get("timeout", self.config.agent_task_timeout)

            for agent_id, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results[agent_id] = {
                        "success": True,
                        "result": result,
                        "agent_id": agent_id,
                    }

                    # Update agent performance metrics
                    with self._agents_lock:
                        if agent_id in self.agent_states:
                            self.agent_states[agent_id]["performance_metrics"][
                                "successful_tasks"
                            ] += 1
                            self.agent_states[agent_id]["status"] = "idle"

                except Exception as e:
                    results[agent_id] = {
                        "success": False,
                        "error": str(e),
                        "agent_id": agent_id,
                    }

                    # Update agent performance metrics
                    with self._agents_lock:
                        if agent_id in self.agent_states:
                            self.agent_states[agent_id]["performance_metrics"]["failed_tasks"] += 1
                            self.agent_states[agent_id]["status"] = "error"

                    logger.error(f"Agent {agent_id} coordination failed: {e}")

            # Calculate coordination efficiency
            successful_agents = sum(1 for r in results.values() if r["success"])
            coordination_efficiency = (
                (successful_agents / len(valid_agents)) * 100 if valid_agents else 0
            )

            # Update metrics
            with self._metrics_lock:
                self.coordination_metrics.successful_coordinations += 1
                self.coordination_metrics.coordination_efficiency = coordination_efficiency

            return {
                "coordination_id": coordination_id,
                "success": True,
                "results": results,
                "coordination_efficiency": coordination_efficiency,
                "resource_metrics": resource_metrics,
                "agents_coordinated": len(valid_agents),
                "successful_agents": successful_agents,
            }

        finally:
            with self._coordinations_lock:
                self.active_coordinations.discard(coordination_id)

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed status for a specific agent."""
        with self._agents_lock:
            if agent_id not in self.agent_states:
                return {"error": f"Agent {agent_id} not found"}

            agent_state = self.agent_states[agent_id].copy()

        # Get performance stats from thread pool manager
        thread_pool_stats = self.thread_pool_manager.get_performance_stats()
        agent_stats = thread_pool_stats.get(agent_id, {})

        return {
            "agent_id": agent_id,
            "state": agent_state,
            "thread_pool_stats": agent_stats,
            "registered": agent_id in self.registered_agents,
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        # Get pool manager metrics
        pool_metrics = self.pool_manager.get_system_metrics()

        # Get resource metrics
        resource_metrics = self.resource_monitor.get_current_metrics()

        # Get database metrics if available
        db_metrics = {}
        if self.db_manager:
            db_metrics = self.db_manager.get_connection_metrics()

        # Get thread pool metrics
        thread_pool_stats = self.thread_pool_manager.get_performance_stats()

        # Calculate coordination metrics
        with self._metrics_lock:
            coordination_metrics = {
                "total_agents": self.coordination_metrics.total_agents,
                "active_agents": self.coordination_metrics.active_agents,
                "coordination_efficiency": self.coordination_metrics.coordination_efficiency,
                "successful_coordinations": self.coordination_metrics.successful_coordinations,
                "failed_coordinations": self.coordination_metrics.failed_coordinations,
                "circuit_breaker_triggers": self.coordination_metrics.circuit_breaker_triggers,
                "avg_coordination_time": (
                    (
                        sum(self.coordination_metrics.coordination_times)
                        / len(self.coordination_metrics.coordination_times)
                    )
                    if self.coordination_metrics.coordination_times
                    else 0
                ),
            }

        return {
            "coordination_metrics": coordination_metrics,
            "pool_metrics": pool_metrics,
            "resource_metrics": resource_metrics,
            "database_metrics": db_metrics,
            "thread_pool_stats": thread_pool_stats,
            "circuit_breaker_state": self.coordination_circuit_breaker.state,
            "active_coordinations": len(self.active_coordinations),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "coordinator": True,
            "pool_manager": False,
            "database": False,
            "resource_monitor": False,
            "issues": [],
        }

        # Check pool manager health
        try:
            pool_metrics = self.pool_manager.get_system_metrics()
            health_status["pool_manager"] = bool(pool_metrics)
        except Exception as e:
            health_status["issues"].append(f"Pool manager error: {e}")

        # Check database health
        if self.db_manager:
            try:
                db_health = await self.db_manager.health_check()
                health_status["database"] = db_health.get("database_connection", False)
                if not health_status["database"]:
                    health_status["issues"].append("Database connection failed")
            except Exception as e:
                health_status["issues"].append(f"Database health check error: {e}")

        # Check resource monitor
        try:
            resource_metrics = self.resource_monitor.get_current_metrics()
            health_status["resource_monitor"] = bool(resource_metrics)
        except Exception as e:
            health_status["issues"].append(f"Resource monitor error: {e}")

        # Determine overall health
        if health_status["issues"]:
            health_status["status"] = (
                "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
            )

        return health_status

    async def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("Initiating enhanced agent coordinator shutdown...")

        # Stop accepting new coordinations
        with self._coordinations_lock:
            logger.info(
                f"Waiting for {len(self.active_coordinations)} active coordinations to complete..."
            )

        # Wait for active coordinations to complete (with timeout)
        timeout = 30  # 30 second timeout
        start_time = time.time()

        while self.active_coordinations and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self.active_coordinations:
            logger.warning(
                f"Shutdown timeout: {len(self.active_coordinations)} coordinations still active"
            )

        # Shutdown thread pool manager
        self.thread_pool_manager.shutdown()

        # Close database manager
        if self.db_manager:
            await self.db_manager.close()

        # Close pool manager
        await self.pool_manager.close()

        logger.info("Enhanced agent coordinator shutdown complete")


# Global instance
_global_coordinator: Optional[EnhancedAgentCoordinator] = None


def get_enhanced_coordinator(
    config: Optional[ConnectionPoolConfig] = None,
    database_url: Optional[str] = None,
) -> EnhancedAgentCoordinator:
    """Get global enhanced agent coordinator."""
    global _global_coordinator

    if _global_coordinator is None:
        _global_coordinator = EnhancedAgentCoordinator(config, database_url)

    return _global_coordinator


async def initialize_global_coordinator(
    config: Optional[ConnectionPoolConfig] = None,
    database_url: Optional[str] = None,
):
    """Initialize global enhanced agent coordinator."""
    coordinator = get_enhanced_coordinator(config, database_url)
    await coordinator.initialize()
    logger.info("Global enhanced agent coordinator initialized")


async def shutdown_global_coordinator():
    """Shutdown global enhanced agent coordinator."""
    global _global_coordinator

    if _global_coordinator:
        await _global_coordinator.shutdown()
        _global_coordinator = None
