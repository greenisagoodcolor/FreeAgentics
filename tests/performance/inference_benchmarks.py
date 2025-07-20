"""Detailed inference benchmarking for PyMDP algorithms.

Implements specific benchmarks for variational inference, belief propagation,
and message passing with profiling hooks and parameterized tests.
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import PyMDP - REQUIRED for inference benchmarks
import pymdp
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent
from pymdp.maths import softmax, spm_dot
from pymdp.maths import spm_log_single as log_stable

from tests.performance.pymdp_benchmarks import BenchmarkResult, PyMDPBenchmark

PYMDP_AVAILABLE = True


@contextmanager
def profile_code(sort_by="cumulative"):
    """Context manager for profiling code execution."""
    pr = cProfile.Profile()
    pr.enable()

    yield pr

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())


class VariationalInferenceBenchmark(PyMDPBenchmark):
    """Benchmark variational inference algorithms."""

    def __init__(self, state_dims: List[int], num_iterations: int = 16):
        super().__init__(f"variational_inference_{len(state_dims)}D")
        self.state_dims = state_dims
        self.num_iterations = num_iterations
        self.qs = None
        self.A = None
        self.obs = None

    def setup(self):
        """Initialize test environment for variational inference."""
        # Initialize belief states (uniform)
        self.qs = utils.obj_array_uniform(self.state_dims)

        # Create observation model
        num_obs = [d for d in self.state_dims]  # Same size as states
        self.A = utils.random_A_matrix(num_obs, self.state_dims)

        # Pre-generate observations
        self.obs = [np.random.randint(0, d) for d in num_obs]

    def run_iteration(self) -> Dict[str, Any]:
        """Run variational inference iteration."""
        # Perform variational inference
        start_vfe = self._calculate_vfe(self.qs, self.A, self.obs)

        # Fixed-point iteration
        for it in range(self.num_iterations):
            qs_prev = utils.obj_array_from_list([q.copy() for q in self.qs])

            # Update each factor
            for f in range(len(self.qs)):
                # Calculate messages from other factors
                messages = self._compute_messages(f, self.obs)

                # Update belief
                self.qs[f] = softmax(messages)

            # Check convergence
            delta = self._compute_delta(qs_prev, self.qs)
            if delta < 1e-6:
                break

        end_vfe = self._calculate_vfe(self.qs, self.A, self.obs)

        return {
            "iterations": it + 1,
            "vfe_reduction": float(start_vfe - end_vfe),
            "final_entropy": float(self._calculate_entropy(self.qs)),
        }

    def _calculate_vfe(self, qs, A, obs):
        """Calculate variational free energy."""
        if not PYMDP_AVAILABLE:
            return 0.0

        # Simplified VFE calculation
        log_likelihood = 0
        for m, o in enumerate(obs):
            likelihood = spm_dot(A[m], qs)
            log_likelihood += log_stable(likelihood[o])

        entropy = self._calculate_entropy(qs)
        return -log_likelihood + entropy

    def _calculate_entropy(self, qs):
        """Calculate entropy of beliefs."""
        if not PYMDP_AVAILABLE:
            return 0.0

        entropy = 0
        for q in qs:
            entropy -= np.sum(q * log_stable(q))
        return entropy

    def _compute_messages(self, factor_idx, obs):
        """Compute messages for belief update."""
        if not PYMDP_AVAILABLE:
            return np.ones(self.state_dims[factor_idx])

        # Simplified message computation
        messages = np.ones(self.state_dims[factor_idx])

        for m, o in enumerate(obs):
            # Get likelihood for this observation
            likelihood = self.A[m][o]

            # Marginalize over other factors
            marginal_dims = list(range(len(self.state_dims)))
            marginal_dims.remove(factor_idx)

            if marginal_dims:
                # Simple approximation - would use proper marginalization in real code
                messages *= np.mean(likelihood, axis=tuple(marginal_dims))
            else:
                messages *= likelihood

        return log_stable(messages)

    def _compute_delta(self, qs_prev, qs_curr):
        """Compute change in beliefs."""
        delta = 0
        for q_prev, q_curr in zip(qs_prev, qs_curr):
            delta += np.sum(np.abs(q_prev - q_curr))
        return delta

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_dims": self.state_dims,
            "num_factors": len(self.state_dims),
            "total_states": np.prod(self.state_dims),
            "num_iterations": self.num_iterations,
        }


class BeliefPropagationBenchmark(PyMDPBenchmark):
    """Benchmark belief propagation in factor graphs."""

    def __init__(self, num_nodes: int = 10, connectivity: float = 0.3):
        super().__init__("belief_propagation")
        self.num_nodes = num_nodes
        self.connectivity = connectivity
        self.factor_graph = None
        self.messages = None

    def setup(self):
        """Initialize factor graph for belief propagation."""
        if not PYMDP_AVAILABLE:
            return

        # Create random factor graph structure
        self.factor_graph = self._create_factor_graph()

        # Initialize messages
        self.messages = {}
        for edge in self.factor_graph["edges"]:
            self.messages[edge] = np.random.rand(
                self.factor_graph["node_dims"][edge[1]]
            )
            self.messages[edge] = self.messages[edge] / np.sum(
                self.messages[edge]
            )

    def run_iteration(self) -> Dict[str, Any]:
        """Run belief propagation iteration."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        num_updates = 0
        max_delta = 0

        # Update messages
        for edge in self.factor_graph["edges"]:
            old_message = self.messages[edge].copy()

            # Compute new message
            new_message = self._compute_bp_message(edge)
            self.messages[edge] = new_message

            # Track convergence
            delta = np.max(np.abs(old_message - new_message))
            max_delta = max(max_delta, delta)
            num_updates += 1

        # Compute beliefs
        beliefs = self._compute_beliefs()

        return {
            "num_message_updates": num_updates,
            "max_message_delta": float(max_delta),
            "avg_belief_entropy": float(
                np.mean([self._entropy(b) for b in beliefs.values()])
            ),
        }

    def _create_factor_graph(self):
        """Create a random factor graph."""
        nodes = list(range(self.num_nodes))
        edges = []
        node_dims = {}

        # Assign random dimensions to nodes
        for node in nodes:
            node_dims[node] = np.random.randint(2, 6)

        # Create edges based on connectivity
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.rand() < self.connectivity:
                    edges.append((i, j))
                    edges.append((j, i))  # Bidirectional

        return {"nodes": nodes, "edges": edges, "node_dims": node_dims}

    def _compute_bp_message(self, edge):
        """Compute belief propagation message along edge."""
        from_node, to_node = edge
        dim = self.factor_graph["node_dims"][to_node]

        # Collect incoming messages
        incoming = []
        for other_edge in self.factor_graph["edges"]:
            if other_edge[1] == from_node and other_edge[0] != to_node:
                incoming.append(self.messages[other_edge])

        # Combine messages (simplified)
        if incoming:
            combined = np.prod(incoming, axis=0)
        else:
            combined = np.ones(self.factor_graph["node_dims"][from_node])

        # Project to target dimension (simplified)
        message = np.random.rand(
            dim
        )  # Would use actual factor in real implementation
        message = message * np.sum(combined)

        # Normalize
        return message / np.sum(message)

    def _compute_beliefs(self):
        """Compute node beliefs from messages."""
        beliefs = {}

        for node in self.factor_graph["nodes"]:
            # Collect incoming messages
            incoming = []
            for edge in self.factor_graph["edges"]:
                if edge[1] == node:
                    incoming.append(self.messages[edge])

            # Combine messages
            if incoming:
                belief = np.prod(incoming, axis=0)
                belief = belief / np.sum(belief)
            else:
                belief = np.ones(self.factor_graph["node_dims"][node])
                belief = belief / np.sum(belief)

            beliefs[node] = belief

        return beliefs

    def _entropy(self, dist):
        """Calculate entropy of distribution."""
        return -np.sum(dist * np.log(dist + 1e-10))

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "connectivity": self.connectivity,
            "num_edges": len(self.factor_graph["edges"])
            if self.factor_graph
            else 0,
        }


class MessagePassingBenchmark(PyMDPBenchmark):
    """Benchmark message passing with different schedules."""

    def __init__(self, grid_size: int = 5, schedule: str = "sequential"):
        super().__init__(f"message_passing_{schedule}")
        self.grid_size = grid_size
        self.schedule = schedule  # sequential, parallel, random
        self.grid_beliefs = None

    def setup(self):
        """Initialize grid for message passing."""
        if not PYMDP_AVAILABLE:
            return

        # Initialize beliefs on grid
        self.grid_beliefs = np.random.rand(
            self.grid_size, self.grid_size, 4
        )  # 4 states per cell
        # Normalize
        self.grid_beliefs = self.grid_beliefs / np.sum(
            self.grid_beliefs, axis=2, keepdims=True
        )

    def run_iteration(self) -> Dict[str, Any]:
        """Run message passing iteration."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        old_beliefs = self.grid_beliefs.copy()

        if self.schedule == "sequential":
            updates = self._sequential_update()
        elif self.schedule == "parallel":
            updates = self._parallel_update()
        else:  # random
            updates = self._random_update()

        # Compute convergence metric
        max_delta = np.max(np.abs(self.grid_beliefs - old_beliefs))
        avg_entropy = np.mean(
            [
                self._entropy(self.grid_beliefs[i, j])
                for i in range(self.grid_size)
                for j in range(self.grid_size)
            ]
        )

        return {
            "num_updates": updates,
            "max_belief_delta": float(max_delta),
            "avg_entropy": float(avg_entropy),
            "schedule": self.schedule,
        }

    def _sequential_update(self):
        """Update beliefs sequentially."""
        updates = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self._update_cell(i, j)
                updates += 1
        return updates

    def _parallel_update(self):
        """Update beliefs in parallel (simulated)."""
        new_beliefs = self.grid_beliefs.copy()
        updates = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                new_beliefs[i, j] = self._compute_new_belief(i, j)
                updates += 1

        self.grid_beliefs = new_beliefs
        return updates

    def _random_update(self):
        """Update beliefs in random order."""
        updates = 0
        cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
        ]
        np.random.shuffle(cells)

        for i, j in cells:
            self._update_cell(i, j)
            updates += 1

        return updates

    def _update_cell(self, i, j):
        """Update belief for a single cell."""
        self.grid_beliefs[i, j] = self._compute_new_belief(i, j)

    def _compute_new_belief(self, i, j):
        """Compute new belief for cell based on neighbors."""
        # Get neighbor beliefs
        neighbors = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append(self.grid_beliefs[ni, nj])

        if neighbors:
            # Average neighbor beliefs (simplified message passing)
            avg_neighbor = np.mean(neighbors, axis=0)
            # Combine with current belief
            new_belief = 0.7 * self.grid_beliefs[i, j] + 0.3 * avg_neighbor
        else:
            new_belief = self.grid_beliefs[i, j]

        # Normalize
        return new_belief / np.sum(new_belief)

    def _entropy(self, dist):
        """Calculate entropy of distribution."""
        return -np.sum(dist * np.log(dist + 1e-10))

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "total_cells": self.grid_size**2,
            "schedule": self.schedule,
        }


class InferenceProfilingBenchmark(PyMDPBenchmark):
    """Benchmark with detailed profiling of inference operations."""

    def __init__(self, state_size: int = 25):
        super().__init__("inference_profiling")
        self.state_size = state_size
        self.agent = None
        self.profile_data = []

    def setup(self):
        """Initialize agent for profiling."""
        if not PYMDP_AVAILABLE:
            return

        # Create agent with moderate complexity
        num_states = [self.state_size, self.state_size // 2]
        num_obs = [self.state_size, self.state_size // 2]
        num_actions = 4

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)
        C = utils.obj_array_uniform(num_obs)

        self.agent = PyMDPAgent(A, B, C=C)

    def run_iteration(self) -> Dict[str, Any]:
        """Run inference with profiling."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        obs = [
            np.random.randint(0, self.state_size),
            np.random.randint(0, self.state_size // 2),
        ]

        # Profile different stages
        timings = {}

        # State inference
        start = time.perf_counter()
        self.agent.infer_states(obs)
        timings["state_inference"] = (time.perf_counter() - start) * 1000

        # Policy inference
        start = time.perf_counter()
        self.agent.infer_policies()
        timings["policy_inference"] = (time.perf_counter() - start) * 1000

        # Action selection
        start = time.perf_counter()
        action = self.agent.sample_action()
        timings["action_selection"] = (time.perf_counter() - start) * 1000

        # Total time
        timings["total"] = sum(timings.values())

        return {
            "state_inference_ms": timings["state_inference"],
            "policy_inference_ms": timings["policy_inference"],
            "action_selection_ms": timings["action_selection"],
            "total_ms": timings["total"],
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {"state_size": self.state_size, "num_factors": 2}


def run_inference_benchmarks():
    """Run comprehensive inference benchmarks."""
    from tests.performance.pymdp_benchmarks import BenchmarkSuite

    suite = BenchmarkSuite()

    # Variational inference with different dimensions
    suite.add_benchmark(VariationalInferenceBenchmark([5, 5]))
    suite.add_benchmark(VariationalInferenceBenchmark([10, 10, 5]))
    suite.add_benchmark(VariationalInferenceBenchmark([20, 20]))

    # Belief propagation
    suite.add_benchmark(BeliefPropagationBenchmark(num_nodes=10))
    suite.add_benchmark(
        BeliefPropagationBenchmark(num_nodes=20, connectivity=0.2)
    )

    # Message passing schedules
    suite.add_benchmark(
        MessagePassingBenchmark(grid_size=5, schedule="sequential")
    )
    suite.add_benchmark(
        MessagePassingBenchmark(grid_size=5, schedule="parallel")
    )
    suite.add_benchmark(
        MessagePassingBenchmark(grid_size=5, schedule="random")
    )

    # Profiling
    suite.add_benchmark(InferenceProfilingBenchmark(state_size=25))

    # Run benchmarks
    results = suite.run_all(iterations=30)

    # Save results
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(f"inference_benchmark_results_{timestamp}.json")

    # Analyze bottlenecks
    print("\n" + "=" * 60)
    print("INFERENCE PERFORMANCE ANALYSIS")
    print("=" * 60)

    for result in results:
        if "profiling" in result.name:
            metrics = result.additional_metrics
            if metrics:
                print("\nInference Stage Breakdown:")
                total = metrics.get("total_ms", 1)
                for stage in [
                    "state_inference",
                    "policy_inference",
                    "action_selection",
                ]:
                    stage_ms = metrics.get(f"{stage}_ms", 0)
                    percentage = (stage_ms / total) * 100 if total > 0 else 0
                    print(f"  {stage}: {stage_ms:.2f} ms ({percentage:.1f}%)")

    return results


if __name__ == "__main__":
    run_inference_benchmarks()
