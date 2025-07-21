#!/usr/bin/env python3
"""
Agent Simulation Framework for FreeAgentics Load Testing

This framework provides realistic agent spawning and lifecycle management
for testing multi-agent coordination at scale.
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import ActiveInferenceAgent, BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent
from tests.performance.performance_utils import replace_sleep
from world.grid_world import GridWorld, GridWorldConfig


class AgentType(Enum):
    """Types of agents that can be spawned."""

    EXPLORER = "explorer"
    COLLECTOR = "collector"
    COORDINATOR = "coordinator"


@dataclass
class AgentSpawnConfig:
    """Configuration for spawning agents."""

    agent_type: AgentType
    count: int
    grid_size: int = 10
    performance_mode: str = "fast"
    spawn_delay_ms: int = 100
    initial_positions: Optional[List[Tuple[int, int]]] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""

    duration_seconds: float
    tick_rate_hz: float = 10.0
    enable_visualization: bool = False
    enable_metrics: bool = True
    enable_coordination: bool = True
    failure_injection_rate: float = 0.0
    network_latency_ms: float = 10.0
    cpu_affinity: bool = False  # Pin agents to CPU cores


class AgentLifecycleManager:
    """Manages agent lifecycle including spawning, monitoring, and cleanup."""

    def __init__(self, max_agents: int = 100):
        self.max_agents = max_agents
        self.agents: Dict[str, ActiveInferenceAgent] = {}
        self.spawn_times: Dict[str, float] = {}
        self.agent_threads: Dict[str, threading.Thread] = {}
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._spawn_counter = 0

    def spawn_agent(self, agent_type: AgentType, config: Dict[str, Any]) -> ActiveInferenceAgent:
        """Spawn a single agent with given configuration."""
        with self._lock:
            if len(self.agents) >= self.max_agents:
                raise RuntimeError(f"Maximum agent limit ({self.max_agents}) reached")

            self._spawn_counter += 1
            agent_id = f"{agent_type.value}_{self._spawn_counter}"
            name = f"{agent_type.value.capitalize()} {self._spawn_counter}"

        # Create agent based on type
        if agent_type == AgentType.EXPLORER:
            agent = BasicExplorerAgent(agent_id, name, grid_size=config.get("grid_size", 10))
        elif agent_type == AgentType.COLLECTOR:
            agent = ResourceCollectorAgent(agent_id, name, grid_size=config.get("grid_size", 10))
        elif agent_type == AgentType.COORDINATOR:
            agent = CoalitionCoordinatorAgent(
                agent_id, name, max_agents=config.get("max_agents", 10)
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Apply configuration overrides
        for key, value in config.items():
            if key != "grid_size":  # Already handled in constructor
                agent.config[key] = value

        # Initialize agent
        agent.start()

        # Track agent
        with self._lock:
            self.agents[agent_id] = agent
            self.spawn_times[agent_id] = time.time()
            self.agent_metrics[agent_id] = {
                "spawn_time": time.time(),
                "steps": 0,
                "errors": 0,
                "coordination_events": 0,
            }

        return agent

    def spawn_batch(self, spawn_config: AgentSpawnConfig) -> List[ActiveInferenceAgent]:
        """Spawn a batch of agents with staggered timing."""
        agents = []

        for i in range(spawn_config.count):
            # Prepare config
            config = {
                "grid_size": spawn_config.grid_size,
                "performance_mode": spawn_config.performance_mode,
                **spawn_config.config_overrides,
            }

            # Set initial position if provided
            if spawn_config.initial_positions and i < len(spawn_config.initial_positions):
                config["initial_position"] = spawn_config.initial_positions[i]

            # Spawn agent
            try:
                agent = self.spawn_agent(spawn_config.agent_type, config)
                agents.append(agent)

                # Stagger spawning to avoid thundering herd
                if spawn_config.spawn_delay_ms > 0 and i < spawn_config.count - 1:
                    replace_sleep(spawn_config.spawn_delay_ms / 1000.0)

            except Exception as e:
                print(f"Failed to spawn agent {i}: {e}")

        return agents

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        with self._lock:
            if agent_id not in self.agents:
                return {}

            agent = self.agents[agent_id]
            metrics = self.agent_metrics[agent_id].copy()

            # Add current agent stats
            metrics.update(
                {
                    "is_active": agent.is_active,
                    "total_steps": agent.total_steps,
                    "uptime_seconds": time.time() - self.spawn_times[agent_id],
                    "current_metrics": agent.metrics.copy(),
                }
            )

            return metrics

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all agents."""
        with self._lock:
            return {agent_id: self.get_agent_metrics(agent_id) for agent_id in self.agents.keys()}

    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate a specific agent."""
        with self._lock:
            if agent_id not in self.agents:
                return False

            agent = self.agents[agent_id]
            agent.stop()

            # Clean up
            del self.agents[agent_id]
            del self.spawn_times[agent_id]
            if agent_id in self.agent_threads:
                del self.agent_threads[agent_id]

        return True

    def terminate_all(self) -> int:
        """Terminate all agents."""
        count = 0
        agent_ids = list(self.agents.keys())

        for agent_id in agent_ids:
            if self.terminate_agent(agent_id):
                count += 1

        return count


class SimulationEnvironment:
    """Manages the simulation environment including world and agent interactions."""

    def __init__(self, world_size: int = 20):
        self.world = GridWorld(GridWorldConfig(width=world_size, height=world_size))
        self.lifecycle_manager = AgentLifecycleManager()
        self.tick_count = 0
        self.start_time = None
        self.running = False
        self._executor = ThreadPoolExecutor(max_workers=32)

    def add_resources(self, count: int) -> None:
        """Add random resources to the world."""
        for _ in range(count):
            random.randint(0, self.world.width - 1)
            random.randint(0, self.world.height - 1)
            random.choice(["energy", "material", "information"])
            random.randint(5, 20)

            # Add resource at position (simplified - would need proper world API)
            # self.world.add_resource(x, y, resource_type, amount)

    def step_agent(self, agent: ActiveInferenceAgent) -> Dict[str, Any]:
        """Execute one step for an agent with error handling."""
        try:
            # Generate observation based on agent type and position
            observation = self.generate_observation(agent)

            # Agent processes observation and selects action
            action = agent.step(observation)

            # Update metrics
            with self.lifecycle_manager._lock:
                metrics = self.lifecycle_manager.agent_metrics.get(agent.agent_id, {})
                metrics["steps"] = metrics.get("steps", 0) + 1

            return {
                "agent_id": agent.agent_id,
                "action": action,
                "success": True,
            }

        except Exception as e:
            # Track errors
            with self.lifecycle_manager._lock:
                metrics = self.lifecycle_manager.agent_metrics.get(agent.agent_id, {})
                metrics["errors"] = metrics.get("errors", 0) + 1

            return {
                "agent_id": agent.agent_id,
                "action": "stay",
                "success": False,
                "error": str(e),
            }

    def generate_observation(self, agent: ActiveInferenceAgent) -> Dict[str, Any]:
        """Generate observation for agent based on type and world state."""
        base_observation = {
            "position": getattr(agent, "position", [0, 0]),
            "tick": self.tick_count,
            "timestamp": time.time(),
        }

        if isinstance(agent, BasicExplorerAgent):
            # Explorer sees surroundings
            base_observation["surroundings"] = np.random.randint(0, 3, (3, 3))

        elif isinstance(agent, ResourceCollectorAgent):
            # Collector sees resources
            base_observation["visible_cells"] = []
            base_observation["current_load"] = random.randint(0, 10)

        elif isinstance(agent, CoalitionCoordinatorAgent):
            # Coordinator sees other agents
            visible_agents = []
            for other_id, other_agent in self.lifecycle_manager.agents.items():
                if other_id != agent.agent_id and random.random() > 0.5:
                    visible_agents.append(
                        {
                            "id": other_id,
                            "position": getattr(other_agent, "position", [0, 0]),
                            "status": "active" if other_agent.is_active else "inactive",
                        }
                    )
            base_observation["visible_agents"] = visible_agents
            base_observation["coalition_status"] = {}

        return base_observation

    def run_tick(self) -> Dict[str, Any]:
        """Execute one simulation tick for all agents."""
        self.tick_count += 1

        # Get all active agents
        agents = list(self.lifecycle_manager.agents.values())

        # Submit all agent steps to thread pool
        futures = []
        for agent in agents:
            if agent.is_active:
                future = self._executor.submit(self.step_agent, agent)
                futures.append(future)

        # Collect results
        results = []
        errors = 0

        for future in as_completed(futures, timeout=1.0):
            try:
                result = future.result()
                results.append(result)
                if not result["success"]:
                    errors += 1
            except Exception:
                errors += 1

        return {
            "tick": self.tick_count,
            "agents_processed": len(results),
            "errors": errors,
            "tick_duration_ms": 0,  # Would measure actual duration
        }

    def run_simulation(self, config: SimulationConfig) -> Dict[str, Any]:
        """Run full simulation with given configuration."""
        self.running = True
        self.start_time = time.time()
        tick_interval = 1.0 / config.tick_rate_hz

        results = {
            "ticks": [],
            "total_ticks": 0,
            "total_errors": 0,
            "average_tick_ms": 0,
        }

        try:
            while self.running and (time.time() - self.start_time) < config.duration_seconds:
                tick_start = time.time()

                # Run one tick
                tick_result = self.run_tick()
                results["ticks"].append(tick_result)
                results["total_errors"] += tick_result["errors"]

                # Maintain tick rate
                tick_duration = time.time() - tick_start
                sleep_time = max(0, tick_interval - tick_duration)
                if sleep_time > 0:
                    replace_sleep(sleep_time)

                # Update tick duration in result
                tick_result["tick_duration_ms"] = tick_duration * 1000

        finally:
            self.running = False
            results["total_ticks"] = self.tick_count
            results["duration_seconds"] = time.time() - self.start_time

            if results["ticks"]:
                avg_tick_ms = np.mean([t["tick_duration_ms"] for t in results["ticks"]])
                results["average_tick_ms"] = avg_tick_ms

        return results

    def inject_failure(self, agent_id: str, failure_type: str = "crash") -> bool:
        """Inject a failure into a specific agent."""
        if agent_id not in self.lifecycle_manager.agents:
            return False

        agent = self.lifecycle_manager.agents[agent_id]

        if failure_type == "crash":
            # Simulate crash by making agent inactive
            agent.is_active = False
        elif failure_type == "slowdown":
            # Simulate performance degradation
            agent.config["performance_mode"] = "accurate"  # Slower mode
        elif failure_type == "memory_leak":
            # Simulate memory leak by adding large data
            agent._leak_data = np.zeros((1000, 1000))  # ~8MB

        return True


class LoadTestScenario:
    """Base class for load test scenarios."""

    def __init__(self, name: str, environment: SimulationEnvironment):
        self.name = name
        self.environment = environment
        self.results = {}

    def setup(self) -> None:
        """Setup the scenario."""
        pass

    def run(self, duration_seconds: float) -> Dict[str, Any]:
        """Run the scenario."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Clean up after scenario."""
        self.environment.lifecycle_manager.terminate_all()


class ScalingTestScenario(LoadTestScenario):
    """Test scaling behavior with increasing agent counts."""

    def __init__(self, environment: SimulationEnvironment, max_agents: int = 50):
        super().__init__("Scaling Test", environment)
        self.max_agents = max_agents

    def run(self, duration_seconds: float) -> Dict[str, Any]:
        """Run scaling test with progressive agent spawning."""
        results = {"scenario": self.name, "phases": []}

        # Test different agent counts
        for agent_count in [1, 5, 10, 20, 30, 40, 50]:
            if agent_count > self.max_agents:
                break

            print(f"\nTesting with {agent_count} agents...")

            # Spawn agents
            spawn_config = AgentSpawnConfig(
                agent_type=AgentType.EXPLORER,
                count=agent_count,
                performance_mode="fast",
                spawn_delay_ms=50,
            )

            self.environment.lifecycle_manager.spawn_batch(spawn_config)

            # Run simulation
            sim_config = SimulationConfig(
                duration_seconds=min(duration_seconds, 10.0),
                tick_rate_hz=10.0,
                enable_metrics=True,
            )

            phase_results = self.environment.run_simulation(sim_config)

            # Calculate metrics
            total_agent_steps = sum(
                m["steps"] for m in self.environment.lifecycle_manager.get_all_metrics().values()
            )

            phase_results.update(
                {
                    "agent_count": agent_count,
                    "total_agent_steps": total_agent_steps,
                    "steps_per_second": total_agent_steps / phase_results["duration_seconds"],
                    "avg_steps_per_agent": total_agent_steps / agent_count,
                }
            )

            results["phases"].append(phase_results)

            # Clean up for next phase
            self.environment.lifecycle_manager.terminate_all()
            replace_sleep(0.5)  # Brief pause between phases

        return results


class MixedWorkloadScenario(LoadTestScenario):
    """Test with mixed agent types and workloads."""

    def run(self, duration_seconds: float) -> Dict[str, Any]:
        """Run mixed workload scenario."""
        # Spawn different agent types
        explorers = self.environment.lifecycle_manager.spawn_batch(
            AgentSpawnConfig(AgentType.EXPLORER, count=10)
        )

        collectors = self.environment.lifecycle_manager.spawn_batch(
            AgentSpawnConfig(AgentType.COLLECTOR, count=10)
        )

        coordinators = self.environment.lifecycle_manager.spawn_batch(
            AgentSpawnConfig(AgentType.COORDINATOR, count=5)
        )

        # Add resources to world
        self.environment.add_resources(20)

        # Run simulation
        sim_config = SimulationConfig(
            duration_seconds=duration_seconds,
            tick_rate_hz=10.0,
            enable_coordination=True,
            failure_injection_rate=0.05,  # 5% failure rate
        )

        results = self.environment.run_simulation(sim_config)

        # Add scenario-specific metrics
        metrics = self.environment.lifecycle_manager.get_all_metrics()

        results["agent_distribution"] = {
            "explorers": len(explorers),
            "collectors": len(collectors),
            "coordinators": len(coordinators),
        }

        results["coordination_events"] = sum(
            m.get("coordination_events", 0) for m in metrics.values()
        )

        return results


def run_agent_simulation_tests():
    """Run comprehensive agent simulation tests."""
    print("=" * 80)
    print("AGENT SIMULATION FRAMEWORK TESTS")
    print("=" * 80)

    # Create environment
    environment = SimulationEnvironment(world_size=20)

    # Test 1: Basic spawning and lifecycle
    print("\n1. Testing agent spawning and lifecycle...")
    lifecycle_mgr = environment.lifecycle_manager

    # Spawn different agent types
    lifecycle_mgr.spawn_agent(AgentType.EXPLORER, {"grid_size": 10})
    lifecycle_mgr.spawn_agent(AgentType.COLLECTOR, {"grid_size": 10})
    lifecycle_mgr.spawn_agent(AgentType.COORDINATOR, {"max_agents": 5})

    print(f"✅ Spawned agents: {list(lifecycle_mgr.agents.keys())}")

    # Test 2: Batch spawning
    print("\n2. Testing batch spawning...")
    batch_config = AgentSpawnConfig(agent_type=AgentType.EXPLORER, count=10, spawn_delay_ms=50)

    start_time = time.time()
    batch_agents = lifecycle_mgr.spawn_batch(batch_config)
    spawn_duration = time.time() - start_time

    print(f"✅ Spawned {len(batch_agents)} agents in {spawn_duration:.2f}s")
    print(f"   Average spawn time: {spawn_duration/len(batch_agents)*1000:.1f}ms per agent")

    # Test 3: Scaling scenario
    print("\n3. Running scaling test scenario...")
    lifecycle_mgr.terminate_all()  # Clean slate

    scaling_scenario = ScalingTestScenario(environment, max_agents=30)
    scaling_results = scaling_scenario.run(duration_seconds=5.0)

    print("\nScaling Results:")
    print("Agents | Steps/sec | Avg ms/step")
    print("-------|-----------|------------")

    for phase in scaling_results["phases"]:
        agents = phase["agent_count"]
        steps_per_sec = phase["steps_per_second"]
        ms_per_step = 1000 / steps_per_sec if steps_per_sec > 0 else 0
        print(f"{agents:6} | {steps_per_sec:9.1f} | {ms_per_step:11.1f}")

    # Test 4: Mixed workload
    print("\n4. Running mixed workload scenario...")
    lifecycle_mgr.terminate_all()

    mixed_scenario = MixedWorkloadScenario("Mixed Workload", environment)
    mixed_results = mixed_scenario.run(duration_seconds=10.0)

    print("\nMixed Workload Results:")
    print(f"Total ticks: {mixed_results['total_ticks']}")
    print(f"Total errors: {mixed_results['total_errors']}")
    print(f"Average tick time: {mixed_results['average_tick_ms']:.1f}ms")
    print(f"Agent distribution: {mixed_results['agent_distribution']}")

    # Cleanup
    lifecycle_mgr.terminate_all()
    environment._executor.shutdown()

    print("\n✅ All simulation framework tests completed!")


if __name__ == "__main__":
    run_agent_simulation_tests()
