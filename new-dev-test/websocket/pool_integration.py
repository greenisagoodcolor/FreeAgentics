"""
WebSocket Connection Pool Integration

Integrates the connection pool, resource manager, and monitoring
with the existing WebSocket system.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from api.v1.websocket import manager as websocket_manager
from fastapi import FastAPI

from websocket.connection_pool import PoolConfig, WebSocketConnectionPool
from websocket.monitoring import initialize_monitor
from websocket.monitoring import router as monitoring_router
from websocket.resource_manager import AgentResourceManager, ResourceConfig

logger = logging.getLogger(__name__)


class WebSocketPooledConnectionManager:
    """Enhanced connection manager using connection pooling."""

    def __init__(
        self,
        pool_config: Optional[PoolConfig] = None,
        resource_config: Optional[ResourceConfig] = None,
    ):
        self.pool_config = pool_config or PoolConfig()
        self.resource_config = resource_config or ResourceConfig()

        # Core components
        self.pool: Optional[WebSocketConnectionPool] = None
        self.resource_manager: Optional[AgentResourceManager] = None

        # Integration with existing websocket manager
        self.websocket_manager = websocket_manager

        self._initialized = False

    async def initialize(self, websocket_url: str = "ws://localhost:8000/ws"):
        """Initialize the pooled connection system."""
        if self._initialized:
            return

        logger.info("Initializing WebSocket connection pool system")

        # Create connection pool
        self.pool = WebSocketConnectionPool(self.pool_config)
        await self.pool.initialize(websocket_url)

        # Create resource manager
        self.resource_manager = AgentResourceManager(self.pool, self.resource_config)
        await self.resource_manager.start()

        # Initialize monitoring
        initialize_monitor(self.pool, self.resource_manager)

        self._initialized = True
        logger.info("WebSocket connection pool system initialized")

    async def shutdown(self):
        """Shutdown the pooled connection system."""
        logger.info("Shutting down WebSocket connection pool system")

        if self.resource_manager:
            await self.resource_manager.stop()

        if self.pool:
            await self.pool.shutdown(graceful=True)

        self._initialized = False
        logger.info("WebSocket connection pool system shutdown complete")

    async def allocate_agent_connection(
        self, agent_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Allocate a pooled connection for an agent.

        Returns:
            Connection ID that can be used for communication
        """
        if not self._initialized:
            raise RuntimeError("Connection pool not initialized")

        # Allocate resources
        resource = await self.resource_manager.allocate_resource(agent_id, prefer_metadata=metadata)

        # Activate the resource
        await self.resource_manager.activate_resource(agent_id)

        logger.info(f"Allocated connection {resource.connection_id} for agent {agent_id}")
        return resource.connection_id

    async def release_agent_connection(self, agent_id: str):
        """Release the connection allocated to an agent."""
        if not self._initialized:
            return

        await self.resource_manager.release_resource(agent_id)
        logger.info(f"Released connection for agent {agent_id}")

    async def send_agent_message(self, agent_id: str, message: Dict[str, Any]):
        """Send a message through the agent's allocated connection."""
        # Get the connection for this agent
        conn = await self.resource_manager.get_agent_connection(agent_id)
        if not conn:
            raise ValueError(f"No connection found for agent {agent_id}")

        # In real implementation, would send through actual WebSocket
        # For now, we can broadcast through the existing manager
        await self.websocket_manager.broadcast(message, event_type=f"agent:{agent_id}")

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        if not self.pool:
            return {}
        return self.pool.get_metrics()

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource manager metrics."""
        if not self.resource_manager:
            return {}
        return self.resource_manager.get_metrics()


# Global instance
pooled_connection_manager = WebSocketPooledConnectionManager()


# Lifespan management for FastAPI
@asynccontextmanager
async def websocket_pool_lifespan(app: FastAPI):
    """Manage WebSocket pool lifecycle with FastAPI."""
    # Startup
    await pooled_connection_manager.initialize()

    yield

    # Shutdown
    await pooled_connection_manager.shutdown()


def setup_websocket_pool(
    app: FastAPI,
    pool_config: Optional[PoolConfig] = None,
    resource_config: Optional[ResourceConfig] = None,
):
    """
    Setup WebSocket connection pooling for a FastAPI application.

    Args:
        app: FastAPI application instance
        pool_config: Configuration for connection pool
        resource_config: Configuration for resource management
    """
    # Update configuration if provided
    if pool_config:
        pooled_connection_manager.pool_config = pool_config
    if resource_config:
        pooled_connection_manager.resource_config = resource_config

    # Include monitoring router
    app.include_router(monitoring_router)

    # Add startup/shutdown handlers
    @app.on_event("startup")
    async def startup_event():
        await pooled_connection_manager.initialize()

    @app.on_event("shutdown")
    async def shutdown_event():
        await pooled_connection_manager.shutdown()

    logger.info("WebSocket connection pool setup complete")


# Example usage functions


async def example_agent_lifecycle():
    """Example of using pooled connections for agent lifecycle."""
    agent_id = "example-agent-123"

    try:
        # Allocate connection for agent
        conn_id = await pooled_connection_manager.allocate_agent_connection(
            agent_id, metadata={"type": "inference", "region": "us-east"}
        )

        logger.info(f"Agent {agent_id} allocated connection {conn_id}")

        # Send messages
        for i in range(10):
            await pooled_connection_manager.send_agent_message(
                agent_id,
                {
                    "type": "agent_update",
                    "data": {"status": "processing", "step": i},
                },
            )
            await asyncio.sleep(1)

        # Update resource usage
        await pooled_connection_manager.resource_manager.update_resource_usage(
            agent_id,
            memory=50 * 1024 * 1024,
            cpu=0.5,  # 50MB  # 0.5 cores
        )

    finally:
        # Always release resources
        await pooled_connection_manager.release_agent_connection(agent_id)
        logger.info(f"Agent {agent_id} resources released")


async def example_multi_agent_scenario():
    """Example of multiple agents sharing connections."""
    agent_ids = [f"agent-{i}" for i in range(20)]

    # Allocate connections for all agents
    tasks = []
    for agent_id in agent_ids:
        tasks.append(pooled_connection_manager.allocate_agent_connection(agent_id))

    await asyncio.gather(*tasks)
    logger.info(f"Allocated connections for {len(agent_ids)} agents")

    # Check metrics
    pool_metrics = pooled_connection_manager.get_pool_metrics()
    resource_metrics = pooled_connection_manager.get_resource_metrics()

    logger.info(f"Pool size: {pool_metrics.get('pool_size')}")
    logger.info(f"Connections in use: {resource_metrics.get('connections_in_use')}")
    logger.info(f"Agents per connection: {resource_metrics.get('avg_agents_per_connection')}")

    # Simulate work
    await asyncio.sleep(5)

    # Release all agents
    release_tasks = []
    for agent_id in agent_ids:
        release_tasks.append(pooled_connection_manager.release_agent_connection(agent_id))

    await asyncio.gather(*release_tasks)
    logger.info("All agents released")


# Performance comparison functions


async def benchmark_without_pooling(num_agents: int, duration: int):
    """Benchmark performance without connection pooling."""
    start_time = asyncio.get_event_loop().time()
    connections_created = 0
    messages_sent = 0
    errors = 0

    async def agent_work(agent_id: str):
        nonlocal connections_created, messages_sent, errors

        try:
            # Simulate creating new connection each time
            await asyncio.sleep(0.1)  # Connection creation time
            connections_created += 1

            # Send messages
            work_start = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - work_start) < duration:
                await asyncio.sleep(0.1)  # Message send time
                messages_sent += 1

        except Exception:
            errors += 1

    # Run agents
    tasks = [agent_work(f"agent-{i}") for i in range(num_agents)]
    await asyncio.gather(*tasks, return_exceptions=True)

    total_time = asyncio.get_event_loop().time() - start_time

    return {
        "approach": "without_pooling",
        "num_agents": num_agents,
        "duration": duration,
        "total_time": total_time,
        "connections_created": connections_created,
        "messages_sent": messages_sent,
        "errors": errors,
        "avg_messages_per_second": messages_sent / total_time if total_time > 0 else 0,
    }


async def benchmark_with_pooling(num_agents: int, duration: int):
    """Benchmark performance with connection pooling."""
    start_time = asyncio.get_event_loop().time()
    messages_sent = 0
    errors = 0

    # Get initial pool size
    initial_pool_size = pooled_connection_manager.pool.size if pooled_connection_manager.pool else 0

    async def agent_work(agent_id: str):
        nonlocal messages_sent, errors

        try:
            # Allocate from pool (much faster)
            await pooled_connection_manager.allocate_agent_connection(agent_id)

            # Send messages
            work_start = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - work_start) < duration:
                await pooled_connection_manager.send_agent_message(
                    agent_id,
                    {
                        "type": "test",
                        "data": {"timestamp": asyncio.get_event_loop().time()},
                    },
                )
                messages_sent += 1
                await asyncio.sleep(0.1)

            # Release
            await pooled_connection_manager.release_agent_connection(agent_id)

        except Exception as e:
            errors += 1
            logger.error(f"Error in agent {agent_id}: {e}")

    # Run agents
    tasks = [agent_work(f"agent-{i}") for i in range(num_agents)]
    await asyncio.gather(*tasks, return_exceptions=True)

    total_time = asyncio.get_event_loop().time() - start_time

    # Get final pool size
    final_pool_size = pooled_connection_manager.pool.size if pooled_connection_manager.pool else 0
    connections_created = final_pool_size - initial_pool_size

    # Get pool metrics
    pool_metrics = pooled_connection_manager.get_pool_metrics()

    return {
        "approach": "with_pooling",
        "num_agents": num_agents,
        "duration": duration,
        "total_time": total_time,
        "connections_created": connections_created,
        "connections_reused": pool_metrics.get("total_acquisitions", 0) - connections_created,
        "messages_sent": messages_sent,
        "errors": errors,
        "avg_messages_per_second": messages_sent / total_time if total_time > 0 else 0,
        "avg_acquisition_time": pool_metrics.get("average_wait_time", 0),
        "pool_utilization": pool_metrics.get("utilization", 0),
    }


async def run_performance_comparison(num_agents: int = 50, duration: int = 30):
    """Run performance comparison between pooled and non-pooled approaches."""
    logger.info(f"Running performance comparison with {num_agents} agents for {duration}s")

    # Run without pooling
    without_pooling = await benchmark_without_pooling(num_agents, duration)

    # Run with pooling
    with_pooling = await benchmark_with_pooling(num_agents, duration)

    # Calculate improvements
    connection_reduction = (
        (without_pooling["connections_created"] - with_pooling["connections_created"])
        / without_pooling["connections_created"]
        * 100
        if without_pooling["connections_created"] > 0
        else 0
    )

    throughput_improvement = (
        (with_pooling["avg_messages_per_second"] - without_pooling["avg_messages_per_second"])
        / without_pooling["avg_messages_per_second"]
        * 100
        if without_pooling["avg_messages_per_second"] > 0
        else 0
    )

    comparison = {
        "without_pooling": without_pooling,
        "with_pooling": with_pooling,
        "improvements": {
            "connection_reduction_percent": round(connection_reduction, 1),
            "throughput_improvement_percent": round(throughput_improvement, 1),
            "connections_saved": without_pooling["connections_created"]
            - with_pooling["connections_created"],
        },
    }

    logger.info("Performance comparison complete:")
    logger.info(f"  Connection reduction: {connection_reduction:.1f}%")
    logger.info(f"  Throughput improvement: {throughput_improvement:.1f}%")

    return comparison
