#!/usr/bin/env python3
"""Memory profiler for PyMDP agents to identify memory usage patterns and hotspots.

This script profiles memory usage per agent component to support Task 5.1:
Profile current memory usage per agent component.
"""

import gc
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import psutil

# Optional memory_profiler import
try:
    from memory_profiler import profile
except ImportError:
    # Define a no-op decorator if memory_profiler is not available
    def profile(func):
        return func


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import ActiveInferenceAgent, BasicExplorerAgent
from world.grid_world import GridWorld, GridWorldConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Profile memory usage of PyMDP agents and components."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0
        self.measurements = []

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def set_baseline(self):
        """Set baseline memory usage."""
        gc.collect()  # Force garbage collection
        mem = self.get_memory_usage()
        self.baseline_memory = mem["rss_mb"]
        logger.info(f"Baseline memory: {self.baseline_memory:.2f} MB")

    def measure(self, label: str) -> Dict[str, float]:
        """Measure memory usage with label."""
        gc.collect()
        mem = self.get_memory_usage()
        delta = mem["rss_mb"] - self.baseline_memory

        measurement = {
            "label": label,
            "rss_mb": mem["rss_mb"],
            "delta_mb": delta,
            "percent": mem["percent"],
            "timestamp": datetime.now().isoformat(),
        }

        self.measurements.append(measurement)
        logger.info(f"{label}: {mem['rss_mb']:.2f} MB (delta: {delta:.2f} MB)")

        return measurement

    def profile_agent_creation(
        self, n_agents: int = 10
    ) -> List[Dict[str, float]]:
        """Profile memory usage during agent creation."""
        logger.info(f"\n=== Profiling Agent Creation ({n_agents} agents) ===")

        self.set_baseline()
        agents = []

        for i in range(n_agents):
            try:
                # Create minimal agent
                agent = self._create_test_agent(f"agent_{i}")
                agents.append(agent)

                # Measure after each agent
                self.measure(f"After creating agent {i+1}")

            except Exception as e:
                logger.error(f"Failed to create agent {i}: {e}")

        # Calculate per-agent memory
        if agents:
            total_delta = self.measurements[-1]["delta_mb"]
            per_agent = total_delta / len(agents)
            logger.info(f"\nAverage memory per agent: {per_agent:.2f} MB")

        return self.measurements

    def profile_agent_components(self) -> Dict[str, float]:
        """Profile memory usage of individual agent components."""
        logger.info("\n=== Profiling Agent Components ===")

        self.set_baseline()
        component_memory = {}

        # Profile belief states
        self._create_belief_states()
        mem = self.measure("Belief states created")
        component_memory["beliefs"] = mem["delta_mb"]

        # Profile transition matrices
        self._create_transition_matrices()
        mem = self.measure("Transition matrices created")
        component_memory["transitions"] = (
            mem["delta_mb"] - component_memory["beliefs"]
        )

        # Profile observation matrices
        self._create_observation_matrices()
        mem = self.measure("Observation matrices created")
        component_memory["observations"] = mem["delta_mb"] - sum(
            component_memory.values()
        )

        # Profile preference matrices
        self._create_preference_matrices()
        mem = self.measure("Preference matrices created")
        component_memory["preferences"] = mem["delta_mb"] - sum(
            component_memory.values()
        )

        # Profile full PyMDP agent
        if self._pymdp_available():
            self._create_pymdp_agent()
            mem = self.measure("PyMDP agent created")
            component_memory["pymdp_agent"] = mem["delta_mb"] - sum(
                component_memory.values()
            )

        logger.info("\n=== Component Memory Usage ===")
        for component, memory in component_memory.items():
            logger.info(f"{component}: {memory:.2f} MB")

        return component_memory

    def profile_agent_operations(self, n_steps: int = 100) -> Dict[str, float]:
        """Profile memory usage during agent operations."""
        logger.info(f"\n=== Profiling Agent Operations ({n_steps} steps) ===")

        self.set_baseline()

        # Create test agent
        agent = self._create_test_agent("test_agent")
        self.measure("Agent created")

        # Profile inference operations
        operation_memory = {}

        # Perception updates
        start_mem = self.measurements[-1]["rss_mb"]
        for i in range(n_steps):
            observation = np.random.randint(0, 5)  # Random observation
            agent.perceive(observation)

        mem = self.measure(f"After {n_steps} perception updates")
        operation_memory["perception"] = mem["rss_mb"] - start_mem

        # Belief updates
        start_mem = mem["rss_mb"]
        for i in range(n_steps):
            agent.update_beliefs()

        mem = self.measure(f"After {n_steps} belief updates")
        operation_memory["belief_updates"] = mem["rss_mb"] - start_mem

        # Action selection
        start_mem = mem["rss_mb"]
        for i in range(n_steps):
            agent.select_action()

        mem = self.measure(f"After {n_steps} action selections")
        operation_memory["action_selection"] = mem["rss_mb"] - start_mem

        logger.info("\n=== Operation Memory Usage ===")
        for operation, memory in operation_memory.items():
            logger.info(
                f"{operation}: {memory:.2f} MB total, {memory/n_steps*1000:.2f} KB per operation"
            )

        return operation_memory

    def identify_memory_hotspots(self) -> Dict[str, Any]:
        """Identify memory hotspots in PyMDP operations."""
        logger.info("\n=== Identifying Memory Hotspots ===")

        hotspots = {
            "large_arrays": [],
            "memory_leaks": [],
            "inefficient_operations": [],
        }

        # Check for large numpy arrays
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / 1024 / 1024
                if size_mb > 1.0:  # Arrays larger than 1MB
                    hotspots["large_arrays"].append(
                        {
                            "shape": obj.shape,
                            "dtype": obj.dtype,
                            "size_mb": size_mb,
                        }
                    )

        # Sort by size
        hotspots["large_arrays"].sort(key=lambda x: x["size_mb"], reverse=True)

        # Log top memory consumers
        logger.info("\nTop memory-consuming arrays:")
        for i, array in enumerate(hotspots["large_arrays"][:10]):
            logger.info(
                f"{i+1}. Shape: {array['shape']}, Type: {array['dtype']}, Size: {array['size_mb']:.2f} MB"
            )

        return hotspots

    def generate_report(self) -> str:
        """Generate comprehensive memory profiling report."""
        report = ["=" * 80]
        report.append("PYMDP AGENT MEMORY PROFILING REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)

        # Summary statistics
        if self.measurements:
            max_mem = max(m["rss_mb"] for m in self.measurements)
            final_mem = self.measurements[-1]["rss_mb"]
            total_delta = final_mem - self.baseline_memory

            report.append("\nSUMMARY:")
            report.append(f"- Baseline memory: {self.baseline_memory:.2f} MB")
            report.append(f"- Peak memory: {max_mem:.2f} MB")
            report.append(f"- Final memory: {final_mem:.2f} MB")
            report.append(f"- Total increase: {total_delta:.2f} MB")

        # Measurements timeline
        report.append("\nMEMORY TIMELINE:")
        for m in self.measurements:
            report.append(
                f"- {m['label']}: {m['rss_mb']:.2f} MB (Δ{m['delta_mb']:+.2f} MB)"
            )

        return "\n".join(report)

    def _create_test_agent(self, agent_id: str) -> ActiveInferenceAgent:
        """Create a test agent for profiling."""
        config = GridWorldConfig(width=10, height=10)
        world = GridWorld(config)

        # Use BasicExplorerAgent which is a concrete implementation
        agent = BasicExplorerAgent(
            agent_id=agent_id, initial_position=(0, 0), world=world
        )

        return agent

    def _pymdp_available(self) -> bool:
        """Check if PyMDP is available."""
        try:
            pass

            return True
        except ImportError:
            return False

    def _create_pymdp_agent(self):
        """Create a PyMDP agent if available."""
        if not self._pymdp_available():
            return None

        from pymdp import utils
        from pymdp.agent import Agent as PyMDPAgent

        # Create minimal PyMDP agent with proper matrices
        num_states = [10, 10]  # 10x10 grid
        num_obs = [10, 10]
        num_factors = 2
        num_controls = [4, 1]  # 4 movement actions

        # Create observation model (A matrices)
        A = utils.obj_array_zeros(
            [[num_obs[f], num_states[f]] for f in range(num_factors)]
        )
        for f in range(num_factors):
            A[f] = np.eye(num_obs[f], num_states[f])  # Identity mapping

        # Create transition model (B matrices)
        B = utils.obj_array_zeros(
            [
                [num_states[f], num_states[f], num_controls[f]]
                for f in range(num_factors)
            ]
        )
        for f in range(num_factors):
            for a in range(num_controls[f]):
                B[f][:, :, a] = np.eye(num_states[f])  # Identity transitions

        # Create preference model (C vectors)
        C = utils.obj_array_zeros([num_obs[f] for f in range(num_factors)])

        # Create initial state distribution (D vectors)
        D = utils.obj_array_uniform(
            [num_states[f] for f in range(num_factors)]
        )

        agent = PyMDPAgent(A=A, B=B, C=C, D=D)

        return agent

    def _create_belief_states(self) -> np.ndarray:
        """Create typical belief state arrays."""
        # Typical belief states for 10x10 grid
        beliefs = []
        for _ in range(10):  # 10 agents
            belief = np.random.rand(10, 10)
            belief = belief / belief.sum()  # Normalize
            beliefs.append(belief)
        return np.array(beliefs)

    def _create_transition_matrices(self) -> np.ndarray:
        """Create typical transition matrices."""
        # State transition matrices for grid world
        # Shape: (num_actions, num_states, num_states)
        num_actions = 4
        num_states = 100  # 10x10 grid

        transitions = np.zeros((num_actions, num_states, num_states))

        # Simple transitions (sparse matrix)
        for a in range(num_actions):
            for s in range(num_states):
                # Each state transitions to at most 4 neighbors
                transitions[a, s, s] = 0.9  # Stay in place
                if s + 1 < num_states:
                    transitions[a, s, s + 1] = 0.1  # Move to neighbor

        return transitions

    def _create_observation_matrices(self) -> np.ndarray:
        """Create typical observation matrices."""
        # Observation matrices
        # Shape: (num_states, num_observations)
        num_states = 100
        num_obs = 10

        obs_matrix = np.eye(num_states, num_obs)  # Simple observation model
        return obs_matrix

    def _create_preference_matrices(self) -> np.ndarray:
        """Create typical preference matrices."""
        # Preference/reward matrices
        num_states = 100
        preferences = np.random.rand(num_states)
        return preferences


def main():
    """Run comprehensive memory profiling."""
    profiler = MemoryProfiler()

    # Profile agent creation
    profiler.profile_agent_creation(n_agents=10)

    # Profile individual components
    component_memory = profiler.profile_agent_components()

    # Profile operations
    operation_memory = profiler.profile_agent_operations(n_steps=100)

    # Identify hotspots
    hotspots = profiler.identify_memory_hotspots()

    # Generate and save report
    report = profiler.generate_report()

    # Save to file
    report_path = "memory_profiling_report_pymdp.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nMemory profiling complete. Report saved to: {report_path}")

    # Save detailed measurements as JSON
    import json

    measurements_path = "memory_profiling_measurements.json"
    with open(measurements_path, "w") as f:
        json.dump(
            {
                "measurements": profiler.measurements,
                "component_memory": component_memory,
                "operation_memory": operation_memory,
                "hotspots": hotspots,
            },
            f,
            indent=2,
        )

    print(f"Detailed measurements saved to: {measurements_path}")


if __name__ == "__main__":
    main()
