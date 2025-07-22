"""Asynchronous Agent Manager for Multi-Agent Scaling.

This module provides async/await based multi-agent processing with thread pools
for improved concurrent agent operations and I/O coordination.
Key features:
1. Thread pool for concurrent PyMDP operations
2. Async coordination layer for I/O and coordination
3. Batched processing for efficiency
4. Better resource utilization than single-threaded processing
"""

import asyncio
import logging
import multiprocessing as mp
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentOperation:
    """Represents a single agent operation to be processed."""

    agent_id: str
    operation_type: str  # 'perceive', 'update_beliefs', 'select_action', 'step'
    data: Dict[str, Any]
    timestamp: float
    operation_id: str = None

    def __post_init__(self):
        """Initialize operation ID if not provided."""
        if self.operation_id is None:
            self.operation_id = str(uuid.uuid4())


@dataclass
class AgentOperationResult:
    """Result of an agent operation."""

    operation_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time_ms: float


def _process_agent_operation(
    agent_config: Dict[str, Any], operation: AgentOperation
) -> AgentOperationResult:
    """Process a single agent operation in a thread pool.

    This function runs in a thread pool to improve concurrency for I/O-bound operations
    while sharing the same environment and dependencies.
    """
    start_time = time.time()

    try:
        # Import here to avoid pickling issues in thread pool
        from agents.base_agent import BasicExplorerAgent

        # Create agent from config
        agent = BasicExplorerAgent(
            agent_config["agent_id"],
            agent_config["name"],
            agent_config.get("grid_size", 5),
        )

        # Set agent state if provided
        if "position" in agent_config:
            agent.position = agent_config["position"]
        if "total_steps" in agent_config:
            agent.total_steps = agent_config["total_steps"]
        if "is_active" in agent_config:
            if agent_config["is_active"]:
                agent.start()

        # Execute the operation
        result = None
        if operation.operation_type == "perceive":
            agent.perceive(operation.data["observation"])
            result = {"status": "perceived"}

        elif operation.operation_type == "update_beliefs":
            agent.update_beliefs()
            result = {
                "belief_entropy": agent.metrics.get("belief_entropy", 0),
                "avg_free_energy": agent.metrics.get("avg_free_energy", 0),
            }

        elif operation.operation_type == "select_action":
            action = agent.select_action()
            result = {"action": action}

        elif operation.operation_type == "step":
            action = agent.step(operation.data["observation"])
            result = {
                "action": action,
                "position": agent.position,
                "total_steps": agent.total_steps,
                "metrics": dict(agent.metrics),
            }

        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")

        execution_time = (time.time() - start_time) * 1000

        return AgentOperationResult(
            operation_id=operation.operation_id,
            agent_id=operation.agent_id,
            success=True,
            result=result,
            error=None,
            execution_time_ms=execution_time,
        )

    except Exception as e:
        import traceback

        execution_time = (time.time() - start_time) * 1000
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Process operation failed: {error_details}")  # Debug output
        return AgentOperationResult(
            operation_id=operation.operation_id,
            agent_id=operation.agent_id,
            success=False,
            result=None,
            error=error_details,
            execution_time_ms=execution_time,
        )


class AsyncAgentManager:
    """
    Asynchronous manager for multi-agent operations.

    Uses thread pools and async coordination to improve concurrent
    agent processing and I/O operations.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize async agent manager.

        Args:
            max_workers: Maximum number of worker processes. Defaults to CPU count.
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.process_pool = None
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.operation_queue = asyncio.Queue()
        self.result_callbacks: Dict[str, asyncio.Future] = {}
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0,
            "avg_execution_time": 0,
        }

        logger.info(f"Initialized AsyncAgentManager with {self.max_workers} workers")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the async agent manager."""
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor for dependency compatibility
        self.process_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Start background worker task
        self.worker_task = asyncio.create_task(self._process_operations())

        logger.info("AsyncAgentManager started")

    async def stop(self):
        """Stop the async agent manager."""
        if hasattr(self, "worker_task"):
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

        logger.info("AsyncAgentManager stopped")

    def register_agent(self, agent_id: str, agent_config: Dict[str, Any]):
        """Register an agent for async processing."""
        self.agent_configs[agent_id] = {
            "agent_id": agent_id,
            "name": agent_config.get("name", f"Agent-{agent_id}"),
            "grid_size": agent_config.get("grid_size", 5),
            "position": agent_config.get("position", [2, 2]),
            "total_steps": agent_config.get("total_steps", 0),
            "is_active": agent_config.get("is_active", False),
        }
        logger.debug(f"Registered agent {agent_id}")

    async def _process_operations(self):
        """Background task to process operations from the queue."""
        while True:
            try:
                # Get operation from queue
                operation = await self.operation_queue.get()

                # Get agent config
                agent_config = self.agent_configs.get(operation.agent_id)
                if not agent_config:
                    # Create error result
                    result = AgentOperationResult(
                        operation_id=operation.operation_id,
                        agent_id=operation.agent_id,
                        success=False,
                        result=None,
                        error=f"Agent {operation.agent_id} not registered",
                        execution_time_ms=0,
                    )
                else:
                    # Process operation in separate process
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.process_pool,
                        _process_agent_operation,
                        agent_config,
                        operation,
                    )

                # Update stats
                self.stats["total_operations"] += 1
                if result.success:
                    self.stats["successful_operations"] += 1
                else:
                    self.stats["failed_operations"] += 1

                self.stats["total_execution_time"] += result.execution_time_ms
                self.stats["avg_execution_time"] = (
                    self.stats["total_execution_time"] / self.stats["total_operations"]
                )

                # Resolve the future if it exists
                if operation.operation_id in self.result_callbacks:
                    future = self.result_callbacks.pop(operation.operation_id)
                    if not future.cancelled():
                        future.set_result(result)

                # Mark queue task as done
                self.operation_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing operation: {e}")

    async def execute_operation(
        self,
        agent_id: str,
        operation_type: str,
        data: Dict[str, Any],
        timeout: float = 10.0,
    ) -> AgentOperationResult:
        """
        Execute a single agent operation asynchronously.

        Args:
            agent_id: ID of the agent
            operation_type: Type of operation ('perceive', 'update_beliefs',
                'select_action', 'step')
            data: Operation data
            timeout: Timeout in seconds

        Returns:
            AgentOperationResult
        """
        operation = AgentOperation(
            agent_id=agent_id,
            operation_type=operation_type,
            data=data,
            timestamp=time.time(),
        )

        # Create future for result
        future = asyncio.Future()
        self.result_callbacks[operation.operation_id] = future

        # Queue operation
        await self.operation_queue.put(operation)

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Clean up
            self.result_callbacks.pop(operation.operation_id, None)
            return AgentOperationResult(
                operation_id=operation.operation_id,
                agent_id=agent_id,
                success=False,
                result=None,
                error="Operation timed out",
                execution_time_ms=timeout * 1000,
            )

    async def execute_batch_operations(
        self,
        operations: List[Tuple[str, str, Dict[str, Any]]],
        timeout: float = 10.0,
    ) -> List[AgentOperationResult]:
        """
        Execute multiple operations concurrently.

        Args:
            operations: List of (agent_id, operation_type, data) tuples
            timeout: Timeout in seconds

        Returns:
            List of AgentOperationResults
        """
        tasks = []
        for agent_id, operation_type, data in operations:
            task = self.execute_operation(agent_id, operation_type, data, timeout)
            tasks.append(task)

        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_id, operation_type, data = operations[i]
                processed_results.append(
                    AgentOperationResult(
                        operation_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        success=False,
                        result=None,
                        error=str(result),
                        execution_time_ms=0,
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def step_all_agents(
        self, observations: Dict[str, Dict[str, Any]], timeout: float = 10.0
    ) -> Dict[str, AgentOperationResult]:
        """
        Execute step operation for all registered agents.

        Args:
            observations: Dict mapping agent_id to observation
            timeout: Timeout in seconds

        Returns:
            Dict mapping agent_id to AgentOperationResult
        """
        operations = []
        for agent_id in self.agent_configs:
            if agent_id in observations:
                operations.append((agent_id, "step", {"observation": observations[agent_id]}))

        results = await self.execute_batch_operations(operations, timeout)

        # Return as dict
        return {result.agent_id: result for result in results}

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return dict(self.stats)

    def get_agent_count(self) -> int:
        """Get number of registered agents."""
        return len(self.agent_configs)


async def benchmark_async_multi_agent_performance():
    """Benchmark the async multi-agent system performance."""
    print("=" * 60)
    print("ASYNC MULTI-AGENT PERFORMANCE BENCHMARK")
    print("=" * 60)

    results = {}

    # Test different agent counts
    agent_counts = [1, 5, 10, 20, 30, 50]

    for num_agents in agent_counts:
        print(f"\nTesting {num_agents} agents with async processing...")

        async with AsyncAgentManager(max_workers=mp.cpu_count()) as manager:
            # Register agents
            for i in range(num_agents):
                manager.register_agent(
                    f"async-agent-{i}",
                    {
                        "name": f"AsyncAgent-{i}",
                        "grid_size": 5,
                        "is_active": True,
                    },
                )

            # Create observations for all agents
            observations = {}
            for i in range(num_agents):
                observations[f"async-agent-{i}"] = {
                    "position": [2, 2],
                    "surroundings": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                }

            # Benchmark multiple rounds
            rounds = 5
            total_start = time.time()

            for round_num in range(rounds):
                round_start = time.time()

                # Step all agents concurrently
                round_results = await manager.step_all_agents(observations, timeout=30.0)

                round_time = time.time() - round_start

                # Count successful operations
                successful = sum(1 for r in round_results.values() if r.success)
                failed = len(round_results) - successful

                # Print first error for debugging
                if failed > 0:
                    first_error = next(r.error for r in round_results.values() if not r.success)
                    error_msg = first_error.split("\n")[0] if first_error else "Unknown error"
                    print(
                        f"  Round {round_num + 1}: {round_time:.3f}s, "
                        f"{successful} success, {failed} failed (error: {error_msg[:50]}...)"
                    )
                else:
                    print(
                        f"  Round {round_num + 1}: {round_time:.3f}s, "
                        f"{successful} success, {failed} failed"
                    )

            total_time = time.time() - total_start
            total_operations = rounds * num_agents

            # Get final stats
            stats = manager.get_stats()

            results[num_agents] = {
                "num_agents": num_agents,
                "rounds": rounds,
                "total_operations": total_operations,
                "total_time": total_time,
                "avg_time_per_operation": (total_time / total_operations) * 1000,  # ms
                "throughput_ops_per_sec": total_operations / total_time,
                "success_rate": stats["successful_operations"] / stats["total_operations"],
                "avg_execution_time_ms": stats["avg_execution_time"],
            }

            print(
                f"  Summary: {results[num_agents]['avg_time_per_operation']:.1f}ms/op, "
                f"{results[num_agents]['throughput_ops_per_sec']:.1f} ops/sec"
            )

    return results


if __name__ == "__main__":
    # Run the async benchmark
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    results = asyncio.run(benchmark_async_multi_agent_performance())

    print("\n" + "=" * 60)
    print("ASYNC PERFORMANCE ANALYSIS")
    print("=" * 60)

    baseline_ms = 370  # Original baseline
    single_agent = results[1]
    max_agents = max(results.keys())
    max_agent_result = results[max_agents]

    print(f"Single agent async performance:" f" {single_agent['avg_time_per_operation']:.1f}ms")
    print(f"Max agents tested: {max_agents}")
    print(f"Max agent performance: {max_agent_result['avg_time_per_operation']:.1f}ms")
    print(f"Throughput scaling: {max_agent_result['throughput_ops_per_sec']:.1f} ops/sec")

    # Calculate scaling efficiency
    theoretical_max = single_agent["throughput_ops_per_sec"] * max_agents
    actual_max = max_agent_result["throughput_ops_per_sec"]
    scaling_efficiency = actual_max / theoretical_max

    print(f"Scaling efficiency: {scaling_efficiency:.1%}")

    if scaling_efficiency > 0.7:
        print("🎯 EXCELLENT: Async thread processing enables near-linear scaling")
    elif scaling_efficiency > 0.5:
        print("✅ GOOD: Significant improvement in multi-agent scaling via async threads")
    elif scaling_efficiency > 0.2:
        print("✅ MODERATE: Meaningful improvement over sequential processing")
    else:
        print("⚠️ LIMITED: Marginal improvement, need further optimization")
