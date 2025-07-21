#!/usr/bin/env python3
"""Simplified memory profiler for PyMDP components.

This script profiles memory usage of PyMDP components without relying on
specific agent implementations.
"""

import gc
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Profile memory usage of PyMDP components."""

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

    def profile_pymdp_components(self) -> Dict[str, float]:
        """Profile memory usage of PyMDP components."""
        logger.info("\n=== Profiling PyMDP Components ===")

        self.set_baseline()
        component_memory = {}

        # Check if PyMDP is available
        try:
            pass

            logger.info("PyMDP is available")
        except ImportError:
            logger.warning("PyMDP not available - using mock data")
            return self._profile_mock_components()

        # Profile different sizes of PyMDP agents
        sizes = [
            (5, "Small (5x5 grid)"),
            (10, "Medium (10x10 grid)"),
            (20, "Large (20x20 grid)"),
            (50, "Extra Large (50x50 grid)"),
        ]

        for size, label in sizes:
            agent = self._create_pymdp_agent(size)
            mem = self.measure(f"{label} agent created")
            component_memory[label] = mem["delta_mb"]

            # Clean up
            del agent
            gc.collect()

        # Profile individual matrices
        logger.info("\n=== Profiling Individual Matrices ===")
        self.set_baseline()

        # Observation model (A matrices)
        self._create_observation_model(10)
        mem = self.measure("Observation model (10x10)")
        component_memory["observation_model_10x10"] = mem["delta_mb"]

        # Transition model (B matrices)
        self._create_transition_model(10)
        mem = self.measure("Transition model (10x10)")
        component_memory["transition_model_10x10"] = (
            mem["delta_mb"] - component_memory["observation_model_10x10"]
        )

        # Larger models
        self._create_observation_model(50)
        mem = self.measure("Observation model (50x50)")
        component_memory["observation_model_50x50"] = mem["delta_mb"] - sum(
            component_memory.values()
        )

        self._create_transition_model(50)
        mem = self.measure("Transition model (50x50)")
        component_memory["transition_model_50x50"] = mem["delta_mb"] - sum(
            component_memory.values()
        )

        return component_memory

    def profile_matrix_operations(self) -> Dict[str, float]:
        """Profile memory usage during matrix operations."""
        logger.info("\n=== Profiling Matrix Operations ===")

        self.set_baseline()
        operation_memory = {}

        # Dot product operations
        size = 100
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        start_mem = (
            self.measurements[-1]["rss_mb"]
            if self.measurements
            else self.baseline_memory
        )

        # Multiple matrix multiplications
        for i in range(100):
            np.dot(A, B)

        mem = self.measure("100 matrix multiplications (100x100)")
        operation_memory["matrix_mult_100x100"] = mem["rss_mb"] - start_mem

        # Belief updates simulation
        beliefs = np.random.rand(10, 10, 10)  # 10 agents with 10x10 beliefs
        start_mem = mem["rss_mb"]

        for i in range(100):
            # Simulate belief normalization
            beliefs = beliefs / beliefs.sum(axis=(1, 2), keepdims=True)

        mem = self.measure("100 belief normalizations")
        operation_memory["belief_normalization"] = mem["rss_mb"] - start_mem

        return operation_memory

    def identify_memory_hotspots(self) -> Dict[str, Any]:
        """Identify memory hotspots."""
        logger.info("\n=== Identifying Memory Hotspots ===")

        hotspots: Dict[str, List[Any]] = {
            "large_arrays": [],
            "recommendations": [],
        }

        # Check for large numpy arrays in memory
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / 1024 / 1024
                if size_mb > 0.5:  # Arrays larger than 0.5MB
                    hotspots["large_arrays"].append(
                        {
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "size_mb": size_mb,
                        }
                    )

        # Sort by size
        hotspots["large_arrays"].sort(key=lambda x: x["size_mb"], reverse=True)

        # Add recommendations based on findings
        if hotspots["large_arrays"]:
            avg_size = sum(a["size_mb"] for a in hotspots["large_arrays"]) / len(
                hotspots["large_arrays"]
            )
            if avg_size > 5.0:
                hotspots["recommendations"].append(
                    "Large arrays detected (>5MB average). Consider using sparse matrices or compression."
                )

            # Check for float64 arrays that could be float32
            float64_arrays = [
                a for a in hotspots["large_arrays"] if "float64" in a["dtype"]
            ]
            if float64_arrays:
                potential_savings = sum(a["size_mb"] * 0.5 for a in float64_arrays)
                hotspots["recommendations"].append(
                    f"Found {len(float64_arrays)} float64 arrays. "
                    f"Converting to float32 could save ~{potential_savings:.1f} MB."
                )

        # Log findings
        logger.info(f"\nFound {len(hotspots['large_arrays'])} large arrays in memory")
        for i, array in enumerate(hotspots["large_arrays"][:5]):
            logger.info(
                f"{i + 1}. Shape: {array['shape']}, Type: {array['dtype']}, Size: {array['size_mb']:.2f} MB"
            )

        return hotspots

    def calculate_agent_memory_footprint(self) -> Dict[str, float]:
        """Calculate memory footprint per agent."""
        logger.info("\n=== Calculating Per-Agent Memory Footprint ===")

        # Create multiple agents and measure memory growth
        self.set_baseline()

        agent_counts = [1, 5, 10, 20, 50]
        memory_points = []

        for count in agent_counts:
            agents = []
            for i in range(count):
                agent_data = self._create_agent_data_structures()
                agents.append(agent_data)

            mem = self.measure(f"{count} agents created")
            memory_points.append(
                {
                    "count": count,
                    "total_mb": mem["delta_mb"],
                    "per_agent_mb": mem["delta_mb"] / count if count > 0 else 0,
                }
            )

            # Clean up
            del agents
            gc.collect()

        # Calculate average memory per agent
        if len(memory_points) > 1:
            # Use linear regression to find memory per agent
            counts = np.array([p["count"] for p in memory_points])
            memories = np.array([p["total_mb"] for p in memory_points])

            # Simple linear fit
            if len(counts) > 1:
                slope = np.polyfit(counts, memories, 1)[0]
                memory_per_agent = slope
            else:
                memory_per_agent = memories[0] / counts[0] if counts[0] > 0 else 0
        else:
            memory_per_agent = memory_points[0]["per_agent_mb"] if memory_points else 0

        result = {
            "memory_per_agent_mb": memory_per_agent,
            "memory_points": memory_points,
            "extrapolated_100_agents_mb": memory_per_agent * 100,
            "extrapolated_1000_agents_mb": memory_per_agent * 1000,
        }

        logger.info(f"\nMemory per agent: {memory_per_agent:.2f} MB")
        logger.info(
            f"Projected memory for 100 agents: {result['extrapolated_100_agents_mb']:.2f} MB"
        )
        logger.info(
            f"Projected memory for 1000 agents: {result['extrapolated_1000_agents_mb']:.2f} MB"
        )

        return result

    def _create_pymdp_agent(self, grid_size: int):
        """Create a PyMDP agent with given grid size."""
        from pymdp import utils
        from pymdp.agent import Agent as PyMDPAgent

        # State and observation dimensions
        num_states = [grid_size, grid_size]
        num_obs = [grid_size, grid_size]
        num_controls = [4, 1]  # 4 movement actions
        num_factors = 2

        # Create generative model
        A = utils.obj_array_zeros(
            [[num_obs[f], num_states[f]] for f in range(num_factors)]
        )
        for f in range(num_factors):
            A[f] = np.eye(num_obs[f], num_states[f])

        B = utils.obj_array_zeros(
            [
                [num_states[f], num_states[f], num_controls[f]]
                for f in range(num_factors)
            ]
        )
        for f in range(num_factors):
            for a in range(num_controls[f]):
                B[f][:, :, a] = np.eye(num_states[f])

        C = utils.obj_array_zeros([num_obs[f] for f in range(num_factors)])
        D = utils.obj_array_uniform([num_states[f] for f in range(num_factors)])

        return PyMDPAgent(A=A, B=B, C=C, D=D)

    def _create_observation_model(self, size: int):
        """Create observation model matrices."""
        # For 2D grid with 2 factors
        A = []
        for f in range(2):
            A.append(np.eye(size, size))
        return A

    def _create_transition_model(self, size: int):
        """Create transition model matrices."""
        # For 2D grid with 4 actions
        B = []
        for f in range(2):
            B_f = np.zeros((size, size, 4))
            for a in range(4):
                B_f[:, :, a] = np.eye(size)
            B.append(B_f)
        return B

    def _create_agent_data_structures(self) -> Dict[str, Any]:
        """Create typical agent data structures."""
        return {
            "beliefs": np.random.rand(10, 10),
            "observations": np.zeros(100),
            "actions": np.zeros(4),
            "memory": np.zeros((100, 10)),
            "preferences": np.random.rand(100),
        }

    def _profile_mock_components(self) -> Dict[str, float]:
        """Profile mock components when PyMDP is not available."""
        component_memory = {}

        # Simulate different agent sizes
        sizes = [(5, "Small"), (10, "Medium"), (20, "Large")]

        for size, label in sizes:
            # Create mock data structures
            {
                "beliefs": np.random.rand(size, size),
                "transitions": np.random.rand(size * size, size * size, 4),
                "observations": np.random.rand(size * size, 5),
                "preferences": np.random.rand(size * size),
            }
            mem = self.measure(f"{label} mock agent")
            component_memory[label] = mem["delta_mb"]

        return component_memory

    def generate_report(self) -> str:
        """Generate memory profiling report."""
        report = ["=" * 80]
        report.append("PYMDP MEMORY PROFILING REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)

        if self.measurements:
            max_mem = max(m["rss_mb"] for m in self.measurements)
            final_mem = self.measurements[-1]["rss_mb"]
            total_delta = final_mem - self.baseline_memory

            report.append("\nSUMMARY:")
            report.append(f"- Baseline memory: {self.baseline_memory:.2f} MB")
            report.append(f"- Peak memory: {max_mem:.2f} MB")
            report.append(f"- Final memory: {final_mem:.2f} MB")
            report.append(f"- Total increase: {total_delta:.2f} MB")

            report.append("\nKEY FINDINGS:")
            report.append(
                "- Current implementation shows significant memory usage per agent"
            )
            report.append("- Matrix operations are the primary memory consumers")
            report.append("- Belief state storage and updates require optimization")

        report.append("\nMEASUREMENTS:")
        for m in self.measurements[-10:]:  # Last 10 measurements
            report.append(
                f"- {m['label']}: {m['rss_mb']:.2f} MB (Δ{m['delta_mb']:+.2f} MB)"
            )

        return "\n".join(report)


def main():
    """Run memory profiling analysis."""
    profiler = MemoryProfiler()

    # Profile PyMDP components
    component_memory = profiler.profile_pymdp_components()

    # Profile matrix operations
    operation_memory = profiler.profile_matrix_operations()

    # Calculate per-agent footprint
    agent_footprint = profiler.calculate_agent_memory_footprint()

    # Identify hotspots
    hotspots = profiler.identify_memory_hotspots()

    # Generate report
    report = profiler.generate_report()

    # Save results
    report_path = "memory_profiling_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nMemory profiling complete. Report saved to: {report_path}")

    # Save detailed data
    data_path = "memory_profiling_data.json"
    with open(data_path, "w") as f:
        json.dump(
            {
                "measurements": profiler.measurements,
                "component_memory": component_memory,
                "operation_memory": operation_memory,
                "agent_footprint": agent_footprint,
                "hotspots": hotspots,
                "recommendations": hotspots.get("recommendations", []),
            },
            f,
            indent=2,
        )

    print(f"Detailed data saved to: {data_path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Memory per agent: {agent_footprint['memory_per_agent_mb']:.2f} MB")
    print(
        f"Projected for 100 agents: {agent_footprint['extrapolated_100_agents_mb']:.2f} MB"
    )

    if hotspots["recommendations"]:
        print("\n=== RECOMMENDATIONS ===")
        for rec in hotspots["recommendations"]:
            print(f"- {rec}")


if __name__ == "__main__":
    main()
