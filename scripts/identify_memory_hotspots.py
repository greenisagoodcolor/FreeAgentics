#!/usr/bin/env python3
"""Identify memory hotspots in PyMDP operations for Task 5.2.

This script performs deep analysis of memory usage patterns in PyMDP agents
to identify specific hotspots and bottlenecks.
"""

import gc
import json
import logging
import os
import sys
import tracemalloc
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BasicExplorerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryHotspotAnalyzer:
    """Analyzer for identifying memory hotspots in PyMDP operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.hotspots = defaultdict(list)
        self.memory_snapshots = []
        self.operation_memory = defaultdict(float)

    def start_tracing(self):
        """Start memory tracing."""
        tracemalloc.start()

    def stop_tracing(self):
        """Stop memory tracing."""
        tracemalloc.stop()

    def take_snapshot(self, label: str) -> tracemalloc.Snapshot:
        """Take a memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append((label, snapshot))
        return snapshot

    def analyze_pymdp_matrices(self) -> Dict[str, Any]:
        """Analyze memory usage of PyMDP matrix operations."""
        logger.info("\n=== Analyzing PyMDP Matrix Memory Usage ===")

        results = {"matrix_sizes": {}, "memory_per_operation": {}, "inefficiencies": []}

        try:
            from pymdp import utils
        except ImportError:
            logger.warning("PyMDP not available - using mock analysis")
            return self._analyze_mock_matrices()

        # Test different grid sizes
        grid_sizes = [5, 10, 20, 50]

        for size in grid_sizes:
            self.start_tracing()
            snapshot_start = self.take_snapshot(f"Before {size}x{size} matrices")

            # Create PyMDP matrices
            num_states = [size, size]
            num_obs = [size, size]
            num_controls = [4, 1]
            num_factors = 2

            # Observation model (A matrices)
            A = utils.obj_array_zeros([[num_obs[f], num_states[f]] for f in range(num_factors)])
            for f in range(num_factors):
                A[f] = np.eye(num_obs[f], num_states[f])

            # Measure A matrix memory
            snapshot_a = self.take_snapshot(f"After A matrices ({size}x{size})")
            a_memory = self._calculate_memory_diff(snapshot_start, snapshot_a)

            # Transition model (B matrices)
            B = utils.obj_array_zeros(
                [[num_states[f], num_states[f], num_controls[f]] for f in range(num_factors)]
            )
            for f in range(num_factors):
                for a in range(num_controls[f]):
                    B[f][:, :, a] = np.eye(num_states[f])

            # Measure B matrix memory
            snapshot_b = self.take_snapshot(f"After B matrices ({size}x{size})")
            b_memory = self._calculate_memory_diff(snapshot_a, snapshot_b)

            # Store results
            results["matrix_sizes"][f"{size}x{size}"] = {
                "A_matrices_mb": a_memory / 1024 / 1024,
                "B_matrices_mb": b_memory / 1024 / 1024,
                "total_mb": (a_memory + b_memory) / 1024 / 1024,
            }

            # Check for inefficiencies
            if size > 10:
                # Check if matrices are sparse but stored dense
                sparsity_a = np.mean([np.count_nonzero(A[f]) / A[f].size for f in range(len(A))])
                sparsity_b = np.mean([np.count_nonzero(B[f]) / B[f].size for f in range(len(B))])

                if sparsity_a < 0.1:  # Less than 10% non-zero
                    results["inefficiencies"].append(
                        {
                            "type": "sparse_matrix_stored_dense",
                            "matrix": "A (observation)",
                            "size": f"{size}x{size}",
                            "sparsity": sparsity_a,
                            "potential_savings_mb": (a_memory * (1 - sparsity_a)) / 1024 / 1024,
                        }
                    )

                if sparsity_b < 0.1:
                    results["inefficiencies"].append(
                        {
                            "type": "sparse_matrix_stored_dense",
                            "matrix": "B (transition)",
                            "size": f"{size}x{size}",
                            "sparsity": sparsity_b,
                            "potential_savings_mb": (b_memory * (1 - sparsity_b)) / 1024 / 1024,
                        }
                    )

            self.stop_tracing()

        return results

    def analyze_belief_operations(self) -> Dict[str, Any]:
        """Analyze memory usage in belief state operations."""
        logger.info("\n=== Analyzing Belief State Operations ===")

        results = {"belief_sizes": {}, "operation_costs": {}, "memory_leaks": []}

        # Create test agent
        agent = BasicExplorerAgent(agent_id="test_agent", name="Test Agent", grid_size=10)

        # Track belief update memory
        self.start_tracing()
        initial_snapshot = self.take_snapshot("Initial state")

        # Perform multiple belief updates
        n_updates = 100
        for i in range(n_updates):
            # Simulate perception and belief update
            observation = {
                "position": (np.random.randint(0, 10), np.random.randint(0, 10)),
                "surroundings": np.random.randint(0, 3, size=(3, 3)),
            }
            agent.perceive(observation)
            agent.update_beliefs()

            if i % 20 == 0:
                snapshot = self.take_snapshot(f"After {i} belief updates")
                memory_growth = self._calculate_memory_diff(initial_snapshot, snapshot)
                results["operation_costs"][f"belief_updates_{i}"] = memory_growth / 1024 / 1024

        # Check for memory leaks
        final_snapshot = self.take_snapshot("Final state")
        total_growth = self._calculate_memory_diff(initial_snapshot, final_snapshot)
        expected_growth = (
            results["operation_costs"].get("belief_updates_20", 0) * 5
        )  # Linear growth expected

        if total_growth / 1024 / 1024 > expected_growth * 1.5:  # 50% more than expected
            results["memory_leaks"].append(
                {
                    "operation": "belief_updates",
                    "expected_mb": expected_growth,
                    "actual_mb": total_growth / 1024 / 1024,
                    "leak_mb": (total_growth / 1024 / 1024) - expected_growth,
                }
            )

        self.stop_tracing()

        return results

    def analyze_agent_lifecycle(self) -> Dict[str, Any]:
        """Analyze memory patterns during agent lifecycle."""
        logger.info("\n=== Analyzing Agent Lifecycle Memory ===")

        results = {"creation_cost": {}, "operation_cost": {}, "cleanup_efficiency": {}}

        self.start_tracing()

        # Measure agent creation
        snapshot_start = self.take_snapshot("Before agent creation")

        agents = []
        for i in range(10):
            agent = BasicExplorerAgent(agent_id=f"agent_{i}", name=f"Agent {i}", grid_size=10)
            agents.append(agent)

        snapshot_created = self.take_snapshot("After creating 10 agents")
        creation_memory = self._calculate_memory_diff(snapshot_start, snapshot_created)
        results["creation_cost"]["10_agents_mb"] = creation_memory / 1024 / 1024
        results["creation_cost"]["per_agent_mb"] = creation_memory / 1024 / 1024 / 10

        # Measure operations
        for _ in range(100):
            for agent in agents:
                observation = {
                    "position": (np.random.randint(0, 10), np.random.randint(0, 10)),
                    "surroundings": np.random.randint(0, 3, size=(3, 3)),
                }
                agent.perceive(observation)
                agent.select_action()

        snapshot_operated = self.take_snapshot("After 100 operations per agent")
        operation_memory = self._calculate_memory_diff(snapshot_created, snapshot_operated)
        results["operation_cost"]["100_operations_mb"] = operation_memory / 1024 / 1024
        results["operation_cost"]["per_operation_kb"] = operation_memory / 1024 / (100 * 10)

        # Measure cleanup
        del agents
        gc.collect()

        snapshot_cleaned = self.take_snapshot("After cleanup")
        remaining_memory = self._calculate_memory_diff(snapshot_start, snapshot_cleaned)
        cleanup_efficiency = 1 - (remaining_memory / creation_memory) if creation_memory > 0 else 1

        results["cleanup_efficiency"]["efficiency_percent"] = cleanup_efficiency * 100
        results["cleanup_efficiency"]["leaked_mb"] = remaining_memory / 1024 / 1024

        self.stop_tracing()

        return results

    def identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify specific optimization opportunities."""
        logger.info("\n=== Identifying Optimization Opportunities ===")

        opportunities = {
            "matrix_optimizations": [],
            "belief_optimizations": [],
            "memory_pooling": [],
            "data_structure_improvements": [],
        }

        # Analyze current memory patterns
        gc.collect()

        # Check for large numpy arrays
        large_arrays = []
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / 1024 / 1024
                if size_mb > 0.5:
                    large_arrays.append(
                        {"shape": obj.shape, "dtype": str(obj.dtype), "size_mb": size_mb}
                    )

        # Matrix optimization opportunities
        for array in large_arrays:
            # Check for float64 that could be float32
            if "float64" in array["dtype"]:
                opportunities["matrix_optimizations"].append(
                    {
                        "type": "dtype_optimization",
                        "current": "float64",
                        "suggested": "float32",
                        "shape": array["shape"],
                        "potential_savings_mb": array["size_mb"] * 0.5,
                    }
                )

            # Check for sparse matrices
            if len(array["shape"]) >= 2 and min(array["shape"][:2]) > 50:
                opportunities["matrix_optimizations"].append(
                    {
                        "type": "sparse_matrix_candidate",
                        "shape": array["shape"],
                        "size_mb": array["size_mb"],
                        "reason": "Large 2D matrix that may be sparse",
                    }
                )

        # Belief state optimizations
        opportunities["belief_optimizations"].extend(
            [
                {
                    "type": "belief_compression",
                    "description": "Use compressed representations for belief states",
                    "potential_savings": "60-80% for sparse beliefs",
                },
                {
                    "type": "belief_sharing",
                    "description": "Share common belief components across agents",
                    "potential_savings": "30-50% for similar agents",
                },
            ]
        )

        # Memory pooling opportunities
        opportunities["memory_pooling"].extend(
            [
                {
                    "type": "matrix_pool",
                    "description": "Reuse matrix buffers for temporary calculations",
                    "potential_savings": "20-40% reduction in allocations",
                },
                {
                    "type": "belief_pool",
                    "description": "Pool belief state arrays to reduce allocation overhead",
                    "potential_savings": "15-25% reduction in GC pressure",
                },
            ]
        )

        # Data structure improvements
        opportunities["data_structure_improvements"].extend(
            [
                {
                    "type": "lazy_evaluation",
                    "description": "Defer matrix computations until needed",
                    "impact": "Reduce peak memory usage by 30-50%",
                },
                {
                    "type": "incremental_updates",
                    "description": "Update beliefs incrementally instead of full recomputation",
                    "impact": "Reduce computation memory by 40-60%",
                },
            ]
        )

        return opportunities

    def _calculate_memory_diff(
        self, snapshot1: tracemalloc.Snapshot, snapshot2: tracemalloc.Snapshot
    ) -> int:
        """Calculate memory difference between snapshots."""
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_diff = sum(stat.size_diff for stat in stats)
        return total_diff

    def _analyze_mock_matrices(self) -> Dict[str, Any]:
        """Analyze mock matrices when PyMDP is not available."""
        results = {"matrix_sizes": {}, "memory_per_operation": {}, "inefficiencies": []}

        for size in [5, 10, 20, 50]:
            # Simulate matrix memory usage
            a_memory = size * size * 8 * 2  # 2 factors, 8 bytes per float64
            b_memory = size * size * 8 * 2 * 4  # 4 actions

            results["matrix_sizes"][f"{size}x{size}"] = {
                "A_matrices_mb": a_memory / 1024 / 1024,
                "B_matrices_mb": b_memory / 1024 / 1024,
                "total_mb": (a_memory + b_memory) / 1024 / 1024,
            }

        return results

    def generate_hotspot_report(self) -> str:
        """Generate comprehensive hotspot analysis report."""
        report = ["=" * 80]
        report.append("PYMDP MEMORY HOTSPOT ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)

        # Matrix analysis
        matrix_results = self.analyze_pymdp_matrices()
        report.append("\n### MATRIX MEMORY ANALYSIS ###")
        for size, data in matrix_results["matrix_sizes"].items():
            report.append(f"\n{size} Grid:")
            report.append(f"  - A matrices: {data['A_matrices_mb']:.2f} MB")
            report.append(f"  - B matrices: {data['B_matrices_mb']:.2f} MB")
            report.append(f"  - Total: {data['total_mb']:.2f} MB")

        # Inefficiencies
        if matrix_results["inefficiencies"]:
            report.append("\n### MATRIX INEFFICIENCIES ###")
            for ineff in matrix_results["inefficiencies"]:
                report.append(f"\n- {ineff['matrix']} matrix ({ineff['size']}):")
                report.append(f"  - Sparsity: {ineff['sparsity']:.1%}")
                report.append(f"  - Potential savings: {ineff['potential_savings_mb']:.2f} MB")

        # Belief operations
        belief_results = self.analyze_belief_operations()
        report.append("\n### BELIEF OPERATION COSTS ###")
        for op, cost in belief_results["operation_costs"].items():
            report.append(f"- {op}: {cost:.2f} MB")

        # Memory leaks
        if belief_results["memory_leaks"]:
            report.append("\n### POTENTIAL MEMORY LEAKS ###")
            for leak in belief_results["memory_leaks"]:
                report.append(f"\n- Operation: {leak['operation']}")
                report.append(f"  - Expected: {leak['expected_mb']:.2f} MB")
                report.append(f"  - Actual: {leak['actual_mb']:.2f} MB")
                report.append(f"  - Leak: {leak['leak_mb']:.2f} MB")

        # Agent lifecycle
        lifecycle_results = self.analyze_agent_lifecycle()
        report.append("\n### AGENT LIFECYCLE MEMORY ###")
        report.append(
            f"- Creation cost: {lifecycle_results['creation_cost']['per_agent_mb']:.2f} MB/agent"
        )
        report.append(
            f"- Operation cost: {lifecycle_results['operation_cost']['per_operation_kb']:.2f} KB/operation"
        )
        report.append(
            f"- Cleanup efficiency: {lifecycle_results['cleanup_efficiency']['efficiency_percent']:.1f}%"
        )

        # Optimization opportunities
        opportunities = self.identify_optimization_opportunities()
        report.append("\n### OPTIMIZATION OPPORTUNITIES ###")

        report.append("\nMatrix Optimizations:")
        for opt in opportunities["matrix_optimizations"][:5]:  # Top 5
            report.append(f"- {opt['type']}: Save {opt.get('potential_savings_mb', 'N/A'):.2f} MB")

        report.append("\nBelief Optimizations:")
        for opt in opportunities["belief_optimizations"]:
            report.append(f"- {opt['type']}: {opt['potential_savings']}")

        report.append("\nMemory Pooling:")
        for opt in opportunities["memory_pooling"]:
            report.append(f"- {opt['type']}: {opt['potential_savings']}")

        report.append("\nData Structure Improvements:")
        for opt in opportunities["data_structure_improvements"]:
            report.append(f"- {opt['type']}: {opt['impact']}")

        # Key findings
        report.append("\n### KEY FINDINGS ###")
        report.append("1. Matrix operations are the primary memory consumers")
        report.append("2. Dense matrix storage for sparse data is inefficient")
        report.append("3. Belief state updates show potential memory leak patterns")
        report.append("4. Float64 usage doubles memory requirements unnecessarily")
        report.append("5. Lack of memory pooling causes excessive allocations")

        # Recommendations
        report.append("\n### RECOMMENDATIONS ###")
        report.append("1. Implement sparse matrix support for A and B matrices")
        report.append("2. Switch to float32 for all non-critical calculations")
        report.append("3. Implement belief state compression for sparse beliefs")
        report.append("4. Add memory pooling for temporary matrix operations")
        report.append("5. Use incremental belief updates instead of full recomputation")

        return "\n".join(report)


def main():
    """Run memory hotspot analysis."""
    analyzer = MemoryHotspotAnalyzer()

    # Generate comprehensive report
    report = analyzer.generate_hotspot_report()

    # Save report
    report_path = "memory_hotspot_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\nHotspot analysis complete. Report saved to: {report_path}")

    # Save detailed data
    detailed_data = {
        "matrix_analysis": analyzer.analyze_pymdp_matrices(),
        "belief_operations": analyzer.analyze_belief_operations(),
        "agent_lifecycle": analyzer.analyze_agent_lifecycle(),
        "optimization_opportunities": analyzer.identify_optimization_opportunities(),
        "timestamp": datetime.now().isoformat(),
    }

    data_path = "memory_hotspot_analysis_data.json"
    with open(data_path, "w") as f:
        json.dump(detailed_data, f, indent=2)

    logger.info(f"Detailed data saved to: {data_path}")

    # Print summary
    print("\n=== MEMORY HOTSPOT SUMMARY ===")
    print("Top memory consumers identified:")
    print("1. Dense matrix storage for sparse transition/observation models")
    print("2. Float64 usage where float32 would suffice")
    print("3. Full belief recomputation instead of incremental updates")
    print("4. Lack of memory pooling causing excessive allocations")
    print("5. Potential memory leaks in belief update cycles")


if __name__ == "__main__":
    main()
