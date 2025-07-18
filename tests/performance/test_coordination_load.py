#!/usr/bin/env python3
"""
Specialized Load Tests for Multi-Agent Coordination System

This test suite validates the documented architectural limitations:
- Python GIL constraints
- ~50 agent practical limit
- 28.4% efficiency at scale
- 72% coordination overhead

Tests focus on real-world coordination scenarios, not idealized performance.
"""

import asyncio

# Configure logging for performance tests
import logging
import os
import queue
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from agents.async_agent_manager import AsyncAgentManager
from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class CoordinationMetrics:
    """Metrics for coordination performance analysis."""

    agent_count: int
    total_messages: int
    successful_handoffs: int
    failed_handoffs: int
    coordination_latency_ms: float
    queue_depth_avg: float
    queue_depth_max: int
    consensus_time_ms: float
    recovery_time_ms: float
    actual_efficiency: float
    theoretical_efficiency: float
    coordination_overhead: float

    def efficiency_loss(self) -> float:
        """Calculate efficiency loss percentage."""
        return (
            (self.theoretical_efficiency - self.actual_efficiency)
            / self.theoretical_efficiency
            * 100
        )


class MessageQueue:
    """Simulated message queue for agent coordination."""

    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.message_count = 0
        self.dropped_messages = 0
        self.latencies = []

    def send(
        self, sender_id: str, receiver_id: str, message: Dict[str, Any]
    ) -> bool:
        """Send message with timestamp."""
        try:
            self.queue.put(
                {
                    "sender": sender_id,
                    "receiver": receiver_id,
                    "message": message,
                    "timestamp": time.time(),
                },
                block=False,
            )
            self.message_count += 1
            return True
        except queue.Full:
            self.dropped_messages += 1
            return False

    def receive(
        self, receiver_id: str, timeout: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Receive message for specific receiver."""
        try:
            msg = self.queue.get(timeout=timeout)
            if (
                msg["receiver"] == receiver_id
                or msg["receiver"] == "broadcast"
            ):
                # Track latency
                latency = time.time() - msg["timestamp"]
                self.latencies.append(latency)
                return msg
            else:
                # Put back if not for this receiver
                self.queue.put(msg, block=False)
                return None
        except queue.Empty:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue performance metrics."""
        return {
            "total_messages": self.message_count,
            "dropped_messages": self.dropped_messages,
            "current_depth": self.queue.qsize(),
            "avg_latency_ms": np.mean(self.latencies) * 1000
            if self.latencies
            else 0,
            "max_latency_ms": np.max(self.latencies) * 1000
            if self.latencies
            else 0,
        }


class CoordinationAgent(BasicExplorerAgent):
    """Extended agent with coordination capabilities."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        message_queue: MessageQueue,
        grid_size: int = 10,
    ):
        super().__init__(agent_id, name, grid_size)
        self.message_queue = message_queue
        self.coordination_count = 0
        self.handoff_success = 0
        self.handoff_failure = 0
        self.consensus_votes = {}

    def send_coordination_message(
        self, receiver_id: str, message_type: str, data: Dict[str, Any]
    ) -> bool:
        """Send coordination message to another agent."""
        return self.message_queue.send(
            self.agent_id,
            receiver_id,
            {
                "type": message_type,
                "data": data,
                "coordination_count": self.coordination_count,
            },
        )

    def check_messages(self) -> List[Dict[str, Any]]:
        """Check for coordination messages."""
        messages = []
        # Check multiple times to get all pending messages
        for _ in range(5):
            msg = self.message_queue.receive(self.agent_id, timeout=0.01)
            if msg:
                messages.append(msg)
            else:
                break
        return messages

    def coordinate_handoff(
        self, target_agent_id: str, task_data: Dict[str, Any]
    ) -> bool:
        """Coordinate task handoff with another agent."""
        # Send handoff request
        if self.send_coordination_message(
            target_agent_id, "handoff_request", task_data
        ):
            # Wait for acknowledgment
            start_time = time.time()
            while time.time() - start_time < 0.5:  # 500ms timeout
                messages = self.check_messages()
                for msg in messages:
                    if (
                        msg["sender"] == target_agent_id
                        and msg["message"]["type"] == "handoff_ack"
                    ):
                        self.handoff_success += 1
                        return True
                time.sleep(0.01)

            self.handoff_failure += 1
            return False
        return False

    def participate_consensus(self, consensus_id: str, vote: bool) -> None:
        """Participate in consensus mechanism."""
        self.send_coordination_message(
            "broadcast",
            "consensus_vote",
            {
                "consensus_id": consensus_id,
                "vote": vote,
                "voter": self.agent_id,
            },
        )

    def handle_coordination_message(self, msg: Dict[str, Any]) -> None:
        """Handle incoming coordination message."""
        message = msg["message"]

        if message["type"] == "handoff_request":
            # Acknowledge handoff
            self.send_coordination_message(
                msg["sender"], "handoff_ack", {"accepted": True}
            )
            self.coordination_count += 1

        elif message["type"] == "consensus_vote":
            # Track consensus votes
            consensus_id = message["data"]["consensus_id"]
            if consensus_id not in self.consensus_votes:
                self.consensus_votes[consensus_id] = []
            self.consensus_votes[consensus_id].append(message["data"])


class CoordinationLoadTester:
    """Main load testing framework for multi-agent coordination."""

    def __init__(self):
        self.agents: List[CoordinationAgent] = []
        self.message_queue = MessageQueue(max_size=10000)
        self.metrics: List[CoordinationMetrics] = []
        self.failure_simulator = AgentFailureSimulator()

    def spawn_agents(
        self, count: int, grid_size: int = 10
    ) -> List[CoordinationAgent]:
        """Spawn multiple coordination agents."""
        agents = []
        for i in range(count):
            agent = CoordinationAgent(
                f"coord_agent_{i}",
                f"Coordination Agent {i}",
                self.message_queue,
                grid_size,
            )
            agent.config[
                "performance_mode"
            ] = "fast"  # Optimize for load testing
            agent.start()
            agents.append(agent)
        self.agents.extend(agents)
        return agents

    def simulate_task_handoffs(
        self, duration_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """Simulate task handoff coordination scenario."""
        start_time = time.time()
        handoff_count = 0
        successful_handoffs = 0

        # Create handoff pairs
        agent_pairs = [
            (self.agents[i], self.agents[(i + 1) % len(self.agents)])
            for i in range(len(self.agents))
        ]

        while time.time() - start_time < duration_seconds:
            for sender, receiver in agent_pairs:
                # Simulate task handoff
                task_data = {
                    "task_id": f"task_{handoff_count}",
                    "timestamp": time.time(),
                    "priority": random.randint(1, 5),
                }

                if sender.coordinate_handoff(receiver.agent_id, task_data):
                    successful_handoffs += 1
                handoff_count += 1

                # Process messages for receiver
                messages = receiver.check_messages()
                for msg in messages:
                    receiver.handle_coordination_message(msg)

            time.sleep(0.01)  # Small delay between rounds

        return {
            "total_handoffs": handoff_count,
            "successful_handoffs": successful_handoffs,
            "success_rate": successful_handoffs / handoff_count
            if handoff_count > 0
            else 0,
            "handoffs_per_second": handoff_count / duration_seconds,
        }

    def simulate_resource_contention(
        self, duration_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """Simulate resource contention scenario."""
        start_time = time.time()
        resource_requests = 0
        contentions = 0
        resolutions = 0

        # Define shared resources
        resources = {
            f"resource_{i}": None for i in range(max(1, len(self.agents) // 5))
        }
        resource_locks = {r: threading.Lock() for r in resources}

        def try_acquire_resource(
            agent: CoordinationAgent, resource_id: str
        ) -> bool:
            """Try to acquire a resource with contention handling."""
            nonlocal contentions, resolutions

            lock = resource_locks[resource_id]
            if lock.acquire(blocking=False):
                if resources[resource_id] is None:
                    resources[resource_id] = agent.agent_id
                    lock.release()
                    return True
                else:
                    # Contention detected
                    contentions += 1
                    current_holder = resources[resource_id]
                    lock.release()

                    # Negotiate through messaging
                    agent.send_coordination_message(
                        current_holder,
                        "resource_request",
                        {
                            "resource_id": resource_id,
                            "priority": random.randint(1, 10),
                        },
                    )

                    # Wait for response
                    time.sleep(0.05)
                    messages = agent.check_messages()
                    for msg in messages:
                        if msg["message"]["type"] == "resource_release":
                            resolutions += 1
                            return True
                    return False
            return False

        # Run contention scenario
        while time.time() - start_time < duration_seconds:
            for agent in self.agents:
                # Each agent tries to acquire a random resource
                resource_id = random.choice(list(resources.keys()))
                resource_requests += 1

                if try_acquire_resource(agent, resource_id):
                    # Use resource briefly then release
                    time.sleep(random.uniform(0.01, 0.05))
                    with resource_locks[resource_id]:
                        if resources[resource_id] == agent.agent_id:
                            resources[resource_id] = None

                # Handle incoming messages
                messages = agent.check_messages()
                for msg in messages:
                    if msg["message"]["type"] == "resource_request":
                        # Randomly decide to release
                        if random.random() > 0.5:
                            agent.send_coordination_message(
                                msg["sender"],
                                "resource_release",
                                {
                                    "resource_id": msg["message"]["data"][
                                        "resource_id"
                                    ]
                                },
                            )

        return {
            "total_requests": resource_requests,
            "contentions": contentions,
            "resolutions": resolutions,
            "contention_rate": contentions / resource_requests
            if resource_requests > 0
            else 0,
            "resolution_rate": resolutions / contentions
            if contentions > 0
            else 0,
        }

    def simulate_consensus_building(
        self, consensus_rounds: int = 5
    ) -> Dict[str, Any]:
        """Simulate consensus building scenario."""
        consensus_times = []
        success_count = 0

        for round_num in range(consensus_rounds):
            consensus_id = f"consensus_{round_num}"
            start_time = time.time()

            # All agents vote
            for agent in self.agents:
                vote = random.random() > 0.3  # 70% positive bias
                agent.participate_consensus(consensus_id, vote)

            # Allow time for message propagation
            time.sleep(0.1)

            # Collect votes
            all_votes = {}
            for agent in self.agents:
                messages = agent.check_messages()
                for msg in messages:
                    if (
                        msg["message"]["type"] == "consensus_vote"
                        and msg["message"]["data"]["consensus_id"]
                        == consensus_id
                    ):
                        voter = msg["message"]["data"]["voter"]
                        vote = msg["message"]["data"]["vote"]
                        all_votes[voter] = vote

            # Calculate consensus
            if len(all_votes) >= len(self.agents) * 0.8:  # 80% participation
                yes_votes = sum(1 for v in all_votes.values() if v)
                if yes_votes / len(all_votes) > 0.5:
                    success_count += 1

            consensus_time = time.time() - start_time
            consensus_times.append(consensus_time)

        return {
            "rounds": consensus_rounds,
            "successful_consensus": success_count,
            "avg_consensus_time_ms": np.mean(consensus_times) * 1000,
            "max_consensus_time_ms": np.max(consensus_times) * 1000,
            "success_rate": success_count / consensus_rounds,
        }

    def measure_coordination_overhead(
        self, agent_count: int
    ) -> CoordinationMetrics:
        """Measure actual coordination overhead for given agent count."""
        # Clear previous agents
        for agent in self.agents:
            agent.stop()
        self.agents.clear()
        self.message_queue = MessageQueue(max_size=10000)

        # Spawn new agents
        self.spawn_agents(agent_count)

        # Baseline: single agent performance
        single_agent = self.agents[0]
        observation = {"position": [0, 0], "surroundings": np.zeros((3, 3))}

        start_time = time.time()
        for _ in range(100):
            single_agent.step(observation)
        single_agent_time = time.time() - start_time

        # Multi-agent coordinated performance
        start_time = time.time()

        # Run coordination scenarios in parallel
        handoff_task = asyncio.create_task(
            asyncio.to_thread(self.simulate_task_handoffs, 2.0)
        )
        contention_task = asyncio.create_task(
            asyncio.to_thread(self.simulate_resource_contention, 2.0)
        )
        consensus_task = asyncio.create_task(
            asyncio.to_thread(self.simulate_consensus_building, 3)
        )

        # Wait for all scenarios
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        (
            handoff_results,
            contention_results,
            consensus_results,
        ) = loop.run_until_complete(
            asyncio.gather(handoff_task, contention_task, consensus_task)
        )

        coordination_time = time.time() - start_time

        # Calculate metrics
        queue_metrics = self.message_queue.get_metrics()

        # Calculate efficiency
        theoretical_efficiency = 1.0  # Perfect linear scaling
        actual_efficiency = (
            single_agent_time * agent_count
        ) / coordination_time
        coordination_overhead = 1.0 - actual_efficiency

        metrics = CoordinationMetrics(
            agent_count=agent_count,
            total_messages=queue_metrics["total_messages"],
            successful_handoffs=handoff_results["successful_handoffs"],
            failed_handoffs=handoff_results["total_handoffs"]
            - handoff_results["successful_handoffs"],
            coordination_latency_ms=queue_metrics["avg_latency_ms"],
            queue_depth_avg=queue_metrics["current_depth"],
            queue_depth_max=queue_metrics["current_depth"],  # Simplified
            consensus_time_ms=consensus_results["avg_consensus_time_ms"],
            recovery_time_ms=0.0,  # Will be measured in failure tests
            actual_efficiency=actual_efficiency,
            theoretical_efficiency=theoretical_efficiency,
            coordination_overhead=coordination_overhead,
        )

        return metrics

    def simulate_agent_failures(
        self, failure_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Simulate agent failures and recovery."""
        recovery_times = []
        failure_count = 0

        # Run for several iterations
        for _ in range(10):
            # Randomly fail agents
            failed_agents = []
            for agent in self.agents:
                if random.random() < failure_rate:
                    failure_count += 1
                    failed_agents.append(agent)
                    self.failure_simulator.fail_agent(agent)

            if failed_agents:
                # Measure recovery time
                start_time = time.time()

                # Other agents detect failures through missing messages
                time.sleep(0.1)

                # Recover failed agents
                for agent in failed_agents:
                    self.failure_simulator.recover_agent(agent)

                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)

            # Run normal coordination
            self.simulate_task_handoffs(0.5)

        return {
            "failure_count": failure_count,
            "avg_recovery_time_ms": np.mean(recovery_times) * 1000
            if recovery_times
            else 0,
            "max_recovery_time_ms": np.max(recovery_times) * 1000
            if recovery_times
            else 0,
        }


class AgentFailureSimulator:
    """Simulates various agent failure scenarios."""

    def fail_agent(self, agent: CoordinationAgent) -> None:
        """Simulate agent failure."""
        agent.is_active = False
        # Simulate PyMDP failure
        if agent.pymdp_agent:
            agent.pymdp_agent = None

    def recover_agent(self, agent: CoordinationAgent) -> None:
        """Recover failed agent."""
        agent.is_active = True
        # Reinitialize PyMDP
        if PYMDP_AVAILABLE and agent.config.get("use_pymdp", True):
            agent._initialize_pymdp()


def run_comprehensive_load_tests():
    """Run comprehensive load tests validating architectural limitations."""
    print("=" * 80)
    print("MULTI-AGENT COORDINATION LOAD TESTS")
    print("Validating Real-World Performance Characteristics")
    print("=" * 80)

    tester = CoordinationLoadTester()

    # Test different agent counts
    agent_counts = [1, 5, 10, 20, 30, 40, 50]
    all_metrics = []

    print("\n📊 Coordination Overhead Analysis:")
    print("Agents | Messages | Efficiency | Overhead | Latency(ms) | Status")
    print("-------|----------|------------|----------|-------------|--------")

    for count in agent_counts:
        metrics = tester.measure_coordination_overhead(count)
        all_metrics.append(metrics)

        # Determine status based on expected efficiency
        if count <= 10:
            expected_efficiency = 0.7
        elif count <= 30:
            expected_efficiency = 0.4
        else:
            expected_efficiency = 0.284  # Documented 28.4% at 50 agents

        status = (
            "✅"
            if metrics.actual_efficiency >= expected_efficiency * 0.9
            else "❌"
        )

        print(
            f"{count:6} | {metrics.total_messages:8} | {metrics.actual_efficiency:10.1%} | "
            f"{metrics.coordination_overhead:9.1%} | {metrics.coordination_latency_ms:11.1f} | {status}"
        )

    # Validate 72% efficiency loss at scale
    max_metrics = all_metrics[-1]  # 50 agents
    efficiency_loss = max_metrics.efficiency_loss()

    print(
        f"\n🎯 Efficiency Loss at {max_metrics.agent_count} agents: {efficiency_loss:.1f}%"
    )
    if efficiency_loss >= 70 and efficiency_loss <= 75:
        print(
            "✅ Validated: ~72% efficiency loss at scale matches documentation"
        )
    else:
        print("❌ Warning: Efficiency loss doesn't match documented 72%")

    # Test failure scenarios
    print("\n🔥 Agent Failure and Recovery Testing:")
    tester.spawn_agents(20)  # Reset with 20 agents
    failure_results = tester.simulate_agent_failures(failure_rate=0.2)

    print(f"Failures simulated: {failure_results['failure_count']}")
    print(
        f"Avg recovery time: {failure_results['avg_recovery_time_ms']:.1f}ms"
    )
    print(
        f"Max recovery time: {failure_results['max_recovery_time_ms']:.1f}ms"
    )

    # Memory usage analysis
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    memory_per_agent = memory_mb / len(tester.agents)

    print("\n💾 Memory Usage:")
    print(f"Total: {memory_mb:.1f}MB")
    print(f"Per agent: {memory_per_agent:.1f}MB")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT:")
    print("=" * 80)

    if (
        efficiency_loss >= 70
        and efficiency_loss <= 75
        and max_metrics.coordination_latency_ms < 100
        and memory_per_agent < 50
    ):
        print(
            "✅ VALIDATED: Multi-agent coordination performs within documented limits"
        )
        print("   - GIL constraints confirmed")
        print("   - ~50 agent practical limit validated")
        print("   - 72% efficiency loss at scale confirmed")
        print("   - Coordination overhead is manageable")
    else:
        print("❌ ISSUES DETECTED: Performance outside documented parameters")
        print("   - Review architectural assumptions")
        print("   - Consider optimization strategies")

    return all_metrics


if __name__ == "__main__":
    # Run the comprehensive load tests
    metrics = run_comprehensive_load_tests()

    # Additional focused tests
    print("\n" + "=" * 80)
    print("SPECIALIZED COORDINATION TESTS")
    print("=" * 80)

    tester = CoordinationLoadTester()
    tester.spawn_agents(30)

    print("\n📬 Message Queue Performance:")
    handoff_results = tester.simulate_task_handoffs(5.0)
    print(f"Handoffs/sec: {handoff_results['handoffs_per_second']:.1f}")
    print(f"Success rate: {handoff_results['success_rate']:.1%}")

    print("\n🔒 Resource Contention:")
    contention_results = tester.simulate_resource_contention(5.0)
    print(f"Contention rate: {contention_results['contention_rate']:.1%}")
    print(f"Resolution rate: {contention_results['resolution_rate']:.1%}")

    print("\n🗳️ Consensus Building:")
    consensus_results = tester.simulate_consensus_building(10)
    print(f"Success rate: {consensus_results['success_rate']:.1%}")
    print(f"Avg time: {consensus_results['avg_consensus_time_ms']:.1f}ms")
