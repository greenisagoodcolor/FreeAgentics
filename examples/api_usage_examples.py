#!/usr/bin/env python3
"""
FreeAgentics API Usage Examples.

Comprehensive examples demonstrating how to use the FreeAgentics API
for Active Inference multi-agent systems.
"""

import asyncio
import json
import time
from typing import Dict, List

import aiohttp
import websockets


class FreeAgenticsAPIClient:
    """Client for interacting with FreeAgentics API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the FreeAgentics API client."""
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def create_agent(self, agent_data: Dict) -> Dict:
        """Create a new agent."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents", json=agent_data
        ) as response:
            return await response.json()

    async def get_agent(self, agent_id: str) -> Dict:
        """Get agent information."""
        async with self.session.get(
            f"{self.base_url}/api/v1/agents/{agent_id}"
        ) as response:
            return await response.json()

    async def list_agents(self) -> List[Dict]:
        """List all agents."""
        async with self.session.get(f"{self.base_url}/api/v1/agents") as response:
            return await response.json()

    async def update_agent(self, agent_id: str, updates: Dict) -> Dict:
        """Update agent configuration."""
        async with self.session.put(
            f"{self.base_url}/api/v1/agents/{agent_id}", json=updates
        ) as response:
            return await response.json()

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        async with self.session.delete(
            f"{self.base_url}/api/v1/agents/{agent_id}"
        ) as response:
            return response.status == 200

    async def start_agent(self, agent_id: str) -> Dict:
        """Start an agent."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents/{agent_id}/start"
        ) as response:
            return await response.json()

    async def stop_agent(self, agent_id: str) -> Dict:
        """Stop an agent."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents/{agent_id}/stop"
        ) as response:
            return await response.json()

    async def agent_step(self, agent_id: str, observation: Dict) -> Dict:
        """Send observation to agent and get action."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents/{agent_id}/step",
            json={"observation": observation},
        ) as response:
            return await response.json()

    async def get_agent_metrics(self, agent_id: str) -> Dict:
        """Get agent performance metrics."""
        async with self.session.get(
            f"{self.base_url}/api/v1/agents/{agent_id}/metrics"
        ) as response:
            return await response.json()

    async def create_coalition(self, coalition_data: Dict) -> Dict:
        """Create a coalition."""
        async with self.session.post(
            f"{self.base_url}/api/v1/coalitions", json=coalition_data
        ) as response:
            return await response.json()

    async def add_agent_to_coalition(self, coalition_id: str, agent_id: str) -> Dict:
        """Add agent to coalition."""
        async with self.session.post(
            f"{self.base_url}/api/v1/coalitions/{coalition_id}/agents/{agent_id}"
        ) as response:
            return await response.json()

    async def get_system_metrics(
        self, metric_type: str, duration: float = 60.0
    ) -> Dict:
        """Get system metrics."""
        params = {"duration": duration}
        async with self.session.get(
            f"{self.base_url}/api/v1/metrics/{metric_type}", params=params
        ) as response:
            return await response.json()


async def example_1_basic_agent_lifecycle():
    """Example 1: Basic agent creation, start, and interaction."""
    print("=== Example 1: Basic Agent Lifecycle ===")

    async with FreeAgenticsAPIClient() as client:
        # Create a new explorer agent
        agent_data = {
            "name": "Explorer-1",
            "agent_type": "explorer",
            "config": {
                "grid_size": 10,
                "use_pymdp": True,
                "exploration_rate": 0.3,
            },
        }

        print("Creating agent...")
        agent = await client.create_agent(agent_data)
        agent_id = agent["agent_id"]
        print(f"Created agent: {agent_id}")

        # Start the agent
        print("Starting agent...")
        start_result = await client.start_agent(agent_id)
        print(f"Agent started: {start_result}")

        # Send some observations and get actions
        print("Sending observations...")
        for i in range(5):
            observation = {
                "position": [i, i],
                "surroundings": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }

            action_result = await client.agent_step(agent_id, observation)
            print(f"Step {i}: Action = {action_result.get('action')}")

            # Small delay
            await asyncio.sleep(0.1)

        # Get agent metrics
        print("Getting agent metrics...")
        metrics = await client.get_agent_metrics(agent_id)
        print(f"Agent metrics: {json.dumps(metrics, indent=2)}")

        # Stop and clean up
        print("Stopping agent...")
        await client.stop_agent(agent_id)
        await client.delete_agent(agent_id)
        print("Agent cleaned up")


async def example_2_gmn_agent_creation():
    """Example 2: Creating agent from GMN specification."""
    print("\n=== Example 2: GMN Agent Creation ===")

    async with FreeAgenticsAPIClient() as client:
        # GMN specification for a simple 2x2 grid world
        gmn_spec = {
            "num_states": [4],  # 2x2 grid = 4 states
            "num_obs": [4],  # 4 possible observations
            "num_actions": [4],  # 4 actions: up, down, left, right
            "A": [
                # Observation model - identity mapping
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            "C": [
                # Preferences - prefer state 3 (goal)
                [0.0, 0.0, 0.0, 2.0]
            ],
        }

        agent_data = {
            "name": "GMN-Agent",
            "gmn_spec": json.dumps(gmn_spec),
            "config": {"use_pymdp": True, "planning_horizon": 3},
        }

        print("Creating GMN agent...")
        agent = await client.create_agent(agent_data)
        agent_id = agent["agent_id"]
        print(f"Created GMN agent: {agent_id}")

        # Start and test the agent
        await client.start_agent(agent_id)

        # Test with observations
        observations = [
            {"state_index": 0},  # Start state
            {"state_index": 1},  # Move
            {"state_index": 2},  # Move
            {"state_index": 3},  # Goal state
        ]

        for i, obs in enumerate(observations):
            result = await client.agent_step(agent_id, obs)
            print(f"GMN Step {i}: {obs} -> {result.get('action')}")

        # Clean up
        await client.stop_agent(agent_id)
        await client.delete_agent(agent_id)


async def example_3_multi_agent_coalition():
    """Example 3: Multi-agent coalition formation."""
    print("\n=== Example 3: Multi-Agent Coalition ===")

    async with FreeAgenticsAPIClient() as client:
        # Create multiple agents
        agents = []
        for i in range(3):
            agent_data = {
                "name": f"Coalition-Agent-{i}",
                "agent_type": "resource_collector",
                "config": {
                    "grid_size": 15,
                    "efficiency_factor": 0.8 + i * 0.1,
                },
            }

            agent = await client.create_agent(agent_data)
            agents.append(agent)
            await client.start_agent(agent["agent_id"])
            print(f"Created and started agent: {agent['agent_id']}")

        # Create a coalition
        coalition_data = {
            "name": "Resource Collection Coalition",
            "objectives": {
                "primary": "maximize_resource_collection",
                "secondary": "minimize_energy_consumption",
            },
            "strategy": "distributed_search",
        }

        print("Creating coalition...")
        coalition = await client.create_coalition(coalition_data)
        coalition_id = coalition["coalition_id"]
        print(f"Created coalition: {coalition_id}")

        # Add agents to coalition
        for agent in agents:
            result = await client.add_agent_to_coalition(
                coalition_id, agent["agent_id"]
            )
            print(f"Added agent {agent['agent_id']} to coalition")

        # Simulate coordinated actions
        print("Simulating coordinated actions...")
        for step in range(5):
            for i, agent in enumerate(agents):
                # Each agent gets different area to explore
                observation = {
                    "position": [i * 5 + step, step],
                    "resources_visible": (
                        [{"type": "energy", "amount": 10}] if step % 2 == i % 2 else []
                    ),
                    "coalition_members": [a["agent_id"] for a in agents if a != agent],
                }

                result = await client.agent_step(agent["agent_id"], observation)
                print(f"Coalition step {step}, Agent {i}: {result.get('action')}")

        # Clean up
        for agent in agents:
            await client.stop_agent(agent["agent_id"])
            await client.delete_agent(agent["agent_id"])


async def example_4_realtime_monitoring():
    """Example 4: Real-time monitoring with WebSocket."""
    print("\n=== Example 4: Real-time Monitoring ===")

    # Start monitoring via WebSocket
    uri = "ws://localhost:8000/api/v1/ws/monitor/example-client"

    try:
        async with websockets.connect(uri) as websocket:
            # Start monitoring session
            monitor_config = {
                "type": "start_monitoring",
                "config": {
                    "metrics": ["cpu_usage", "memory_usage", "agent_count"],
                    "sample_rate": 1.0,
                },
            }

            await websocket.send(json.dumps(monitor_config))

            # Receive monitoring data
            print("Monitoring system metrics for 10 seconds...")
            start_time = time.time()

            while time.time() - start_time < 10:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)

                    if data.get("type") == "metrics_update":
                        metrics = data.get("metrics", {})
                        print(
                            f"Metrics: CPU={metrics.get('cpu_usage', 0):.1f}%, "
                            f"Memory={metrics.get('memory_usage', 0):.1f}%, "
                            f"Agents={metrics.get('agent_count', 0)}"
                        )

                except asyncio.TimeoutError:
                    continue

            # Stop monitoring
            stop_config = {"type": "stop_monitoring"}
            await websocket.send(json.dumps(stop_config))

    except Exception as e:
        print(f"Monitoring connection failed: {e}")
        print("Make sure the FreeAgentics server is running")


async def example_5_batch_operations():
    """Example 5: Batch operations and bulk management."""
    print("\n=== Example 5: Batch Operations ===")

    async with FreeAgenticsAPIClient() as client:
        # Create multiple agents in batch
        agent_configs = [
            {
                "name": f"Batch-Explorer-{i}",
                "agent_type": "explorer",
                "config": {"grid_size": 8, "exploration_rate": 0.2 + i * 0.1},
            }
            for i in range(5)
        ]

        print("Creating agents in batch...")
        agents = []
        for config in agent_configs:
            agent = await client.create_agent(config)
            agents.append(agent)

        # Start all agents
        print("Starting all agents...")
        for agent in agents:
            await client.start_agent(agent["agent_id"])

        # Get all agent statuses
        print("Getting agent statuses...")
        all_agents = await client.list_agents()
        for agent_info in all_agents:
            if agent_info["name"].startswith("Batch-"):
                print(
                    f"Agent {agent_info['name']}: {agent_info.get('status', 'unknown')}"
                )

        # Bulk observation sending
        print("Sending bulk observations...")
        observation = {
            "position": [5, 5],
            "surroundings": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        }

        # Send same observation to all agents simultaneously
        tasks = [client.agent_step(agent["agent_id"], observation) for agent in agents]

        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"Batch Agent {i}: {result.get('action')}")

        # Clean up all agents
        print("Cleaning up batch agents...")
        for agent in agents:
            await client.stop_agent(agent["agent_id"])
            await client.delete_agent(agent["agent_id"])


async def example_6_performance_testing():
    """Example 6: Performance testing and metrics analysis."""
    print("\n=== Example 6: Performance Testing ===")

    async with FreeAgenticsAPIClient() as client:
        # Create a high-performance agent
        agent_data = {
            "name": "Performance-Test-Agent",
            "agent_type": "explorer",
            "config": {
                "grid_size": 20,
                "use_pymdp": True,
                "planning_horizon": 5,
            },
        }

        print("Creating performance test agent...")
        agent = await client.create_agent(agent_data)
        agent_id = agent["agent_id"]
        await client.start_agent(agent_id)

        # Performance test: rapid observations
        print("Running performance test (100 rapid observations)...")
        start_time = time.time()

        for i in range(100):
            observation = {
                "position": [i % 20, (i // 20) % 20],
                "surroundings": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }

            await client.agent_step(agent_id, observation)

            if i % 20 == 0:
                print(f"Completed {i} observations...")

        end_time = time.time()
        duration = end_time - start_time
        rate = 100 / duration

        print("Performance test completed:")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Rate: {rate:.2f} observations/second")

        # Get detailed metrics
        metrics = await client.get_agent_metrics(agent_id)
        print(f"  - Total observations: {metrics.get('total_observations', 0)}")
        print(f"  - Total actions: {metrics.get('total_actions', 0)}")
        print(f"  - Average free energy: {metrics.get('avg_free_energy', 0):.3f}")
        print(f"  - Belief entropy: {metrics.get('belief_entropy', 0):.3f}")

        # Get system metrics during the test
        print("\nSystem metrics during test:")
        system_metrics = await client.get_system_metrics("cpu_usage")
        if system_metrics.get("summary"):
            summary = system_metrics["summary"]
            print(
                f"  - CPU usage: {summary.get('latest', 0):.1f}% (max: {summary.get('max', 0):.1f}%)"
            )

        memory_metrics = await client.get_system_metrics("memory_usage")
        if memory_metrics.get("summary"):
            summary = memory_metrics["summary"]
            print(
                f"  - Memory usage: {summary.get('latest', 0):.1f}% (max: {summary.get('max', 0):.1f}%)"
            )

        # Clean up
        await client.stop_agent(agent_id)
        await client.delete_agent(agent_id)


async def main():
    """Run all examples."""
    print("FreeAgentics API Usage Examples")
    print("=" * 50)

    examples = [
        example_1_basic_agent_lifecycle,
        example_2_gmn_agent_creation,
        example_3_multi_agent_coalition,
        example_4_realtime_monitoring,
        example_5_batch_operations,
        example_6_performance_testing,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed: {e}")
            print(
                "Make sure the FreeAgentics server is running on http://localhost:8000"
            )

        # Small delay between examples
        await asyncio.sleep(1)

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    # Install dependencies:
    # pip install aiohttp websockets

    asyncio.run(main())
