"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import logging
import time
from collections import defaultdict
from collections import deque
from collections import deque as DequeType
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation

"""
Diagnostic and Visualization Tools for Active Inference
This module provides comprehensive diagnostic tools and visualization capabilities
for understanding, debugging, and analyzing Active Inference agents.
."""
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic tools"""

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: Path = Path("logs/active_inference")
    # Visualization
    figure_size: Tuple[int, int] = (10, 8)
    save_figures: bool = True
    figure_dir: Path = Path("figures/active_inference")
    figure_format: str = "png"
    dpi: int = 150
    # Real-time monitoring
    enable_realtime: bool = True
    update_frequency: int = 10  # Updates per second
    buffer_size: int = 1000
    # Analysis
    track_gradients: bool = True
    track_activations: bool = True
    track_beliefs: bool = True
    track_free_energy: bool = True
    # Performance
    profile_performance: bool = True
    memory_tracking: bool = True


class BeliefTracker:
    """
    Tracks and visualizes belief states over time.
    """

    def __init__(
        self,
        config: DiagnosticConfig,
        num_states: int,
        state_labels: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.num_states = num_states
        self.state_labels = (
            state_labels or [f"State {i}" for i in range(num_states)])
        # History buffers
        self.belief_history: DequeType[torch.Tensor] = deque(maxlen=config.buffer_size)
        self.timestamp_history: DequeType[float] = deque(maxlen=config.buffer_size)
        self.entropy_history: DequeType[float] = deque(maxlen=config.buffer_size)
        # Statistics
        self.total_updates = 0
        self.start_time = time.time()

    def record_belief(self, belief: torch.Tensor, timestamp: Optional[float] = None) -> None:
        """Record belief state"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        # Store belief
        belief_tensor = belief.detach().cpu()
        self.belief_history.append(belief_tensor)
        self.timestamp_history.append(timestamp)
        # Compute and store entropy
        entropy = -torch.sum(belief * torch.log(belief + 1e-16))
        self.entropy_history.append(entropy.item())
        self.total_updates += 1

    def plot_belief_evolution(self, save_path: Optional[Path] = None) -> Optional[plt.Figure]:
        """Plot belief evolution over time"""
        if len(self.belief_history) == 0:
            logger.warning("No belief data to plot")
            return None
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, sharex=True)
        # Convert history to array
        beliefs = np.array(self.belief_history)
        timestamps = np.array(self.timestamp_history)
        # Plot belief trajectories
        for i in range(self.num_states):
            ax1.plot(timestamps, beliefs[:, i], label=self.state_labels[i])
        ax1.set_ylabel("Belief Probability")
        ax1.set_title("Belief State Evolution")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)
        # Plot entropy
        ax2.plot(timestamps, self.entropy_history, "k-", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Belief Entropy Over Time")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path or self.config.save_figures:
            save_path = save_path or self.config.figure_dir / "belief_evolution.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
        return fig

    def plot_belief_heatmap(self, save_path: Optional[Path] = None) -> Optional[plt.Figure]:
        """Plot belief states as heatmap"""
        if len(self.belief_history) == 0:
            return None
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        beliefs = np.array(self.belief_history).T
        # Create heatmap
        sns.heatmap(
            beliefs,
            xticklabels=False,
            yticklabels=self.state_labels,
            cmap="viridis",
            cbar_kws={"label": "Belief Probability"},
            ax=ax,
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State")
        ax.set_title("Belief State Heatmap")
        if save_path or self.config.save_figures:
            save_path = save_path or self.config.figure_dir / "belief_heatmap.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi)
        return fig

    def get_statistics(self) -> Dict[str, Any]:
        """Get belief statistics"""
        if len(self.belief_history) == 0:
            return {}
        beliefs = np.array(self.belief_history)
        stats = {
            "total_updates": self.total_updates,
            "duration": time.time() - self.start_time,
            "mean_entropy": np.mean(self.entropy_history),
            "std_entropy": np.std(self.entropy_history),
            "min_entropy": np.min(self.entropy_history),
            "max_entropy": np.max(self.entropy_history),
            "dominant_states": [],
        }
        # Find dominant states over time
        dominant = np.argmax(beliefs, axis=1)
        unique, counts = np.unique(dominant, return_counts=True)
        for state, count in zip(unique, counts):
            stats["dominant_states"].append(
                {"state": self.state_labels[state], "frequency": count / len(dominant)}
            )
        return stats


class FreeEnergyMonitor:
    """
    Monitors and visualizes free energy components.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self.config = config
        # History buffers
        self.vfe_history: DequeType[float] = deque(maxlen=config.buffer_size)
        self.efe_history: DequeType[float] = deque(maxlen=config.buffer_size)
        self.accuracy_history: DequeType[float] = deque(maxlen=config.buffer_size)
        self.complexity_history: DequeType[float] = deque(maxlen=config.buffer_size)
        self.timestamps: DequeType[float] = deque(maxlen=config.buffer_size)
        # Action-specific EFE
        self.action_efe_history: DefaultDict[str, DequeType[float]] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )
        self.start_time = time.time()

    def record_vfe(
        self, accuracy: float, complexity: float, timestamp: Optional[float] = None
    ) -> None:
        """Record variational free energy components"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        vfe = accuracy + complexity
        self.vfe_history.append(vfe)
        self.accuracy_history.append(accuracy)
        self.complexity_history.append(complexity)
        self.timestamps.append(timestamp)

    def record_efe(
        self,
        efe_values: torch.Tensor,
        action_labels: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record expected free energy values"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        # Total EFE (minimum across actions)
        min_efe = torch.min(efe_values).item()
        self.efe_history.append(min_efe)
        # Per-action EFE
        if action_labels is None:
            action_labels = [f"Action {i}" for i in range(len(efe_values))]
        for i, (action, efe) in enumerate(zip(action_labels, efe_values)):
            self.action_efe_history[action].append(efe.item())

    def plot_free_energy_components(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot free energy components over time"""
        fig, axes = (
            plt.subplots(3, 1, figsize=self.config.figure_size, sharex=True))
        timestamps = np.array(self.timestamps)
        # VFE components
        if len(self.vfe_history) > 0:
            axes[0].plot(timestamps, self.vfe_history, "k-", linewidth=2, label="Total VFE")
            axes[0].plot(timestamps, self.accuracy_history, "r--", label="Accuracy")
            axes[0].plot(timestamps, self.complexity_history, "b--", label="Complexity")
            axes[0].set_ylabel("VFE")
            axes[0].set_title("Variational Free Energy Components")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        # EFE evolution
        if len(self.efe_history) > 0:
            axes[1].plot(timestamps[: len(self.efe_history)], self.efe_history, "g-", linewidth=2)
            axes[1].set_ylabel("EFE")
            axes[1].set_title("Expected Free Energy (Minimum)")
            axes[1].grid(True, alpha=0.3)
        # Action-specific EFE
        if self.action_efe_history:
            for action, efe_vals in self.action_efe_history.items():
                axes[2].plot(timestamps[: len(efe_vals)], efe_vals, label=action)
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("EFE")
            axes[2].set_title("Action-Specific Expected Free Energy")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path or self.config.save_figures:
            save_path = save_path or self.config.figure_dir / "free_energy_components.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi)
        return fig


class GradientAnalyzer:
    """
    Analyzes gradients during learning.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self.config = config
        # Gradient statistics
        self.gradient_norms: DefaultDict[str, DequeType[float]] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )
        self.gradient_means: DefaultDict[str, DequeType[float]] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )
        self.gradient_stds: DefaultDict[str, DequeType[float]] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )
        # Gradient flow
        self.layer_gradients: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
        self.update_count = 0

    def analyze_gradients(self, model: nn.Module) -> None:
        """Analyze gradients in model"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                # Compute statistics
                norm = torch.norm(grad).item()
                mean = torch.mean(grad).item()
                std = torch.std(grad).item()
                self.gradient_norms[name].append(norm)
                self.gradient_means[name].append(mean)
                self.gradient_stds[name].append(std)
        self.update_count += 1

    def plot_gradient_flow(self, save_path: Optional[Path] = None) -> Optional[plt.Figure]:
        """Plot gradient flow through layers"""
        if not self.gradient_norms:
            return None
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        # Get average gradient norms
        layer_names = []
        avg_norms = []
        for name, norms in self.gradient_norms.items():
            layer_names.append(name)
            avg_norms.append(np.mean(norms))
        # Create bar plot
        y_pos = np.arange(len(layer_names))
        ax.barh(y_pos, avg_norms)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names)
        ax.set_xlabel("Average Gradient Norm")
        ax.set_title("Gradient Flow Analysis")
        ax.grid(True, alpha=0.3, axis="x")
        if save_path or self.config.save_figures:
            save_path = save_path or self.config.figure_dir / "gradient_flow.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
        return fig

    def check_gradient_health(self) -> Dict[str, Any]:
        """Check for gradient issues"""
        issues: Dict[str, List[Dict[str, Any]]] = {
            "vanishing_gradients": [],
            "exploding_gradients": [],
            "dead_neurons": [],
        }
        for name, norms in self.gradient_norms.items():
            recent_norms = list(norms)[-100:]  # Last 100 updates
            avg_norm = np.mean(recent_norms)
            # Check for vanishing gradients
            if avg_norm < 1e-6:
                issues["vanishing_gradients"].append({"layer": name, "avg_norm": avg_norm})
            # Check for exploding gradients
            elif avg_norm > 100:
                issues["exploding_gradients"].append({"layer": name, "avg_norm": avg_norm})
            # Check for dead neurons (very low variance)
            if np.std(recent_norms) < 1e-8:
                issues["dead_neurons"].append({"layer": name, "std": np.std(recent_norms)})
        return issues


class InferenceVisualizer:
    """
    Visualizes the inference process.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self.config = config
        # Colors for visualization
        self.state_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # type: ignore[attr-defined]
        self.action_colors = plt.cm.tab20(np.linspace(0, 1, 20))  # type: ignore[attr-defined]

    def visualize_inference_graph(
        self,
        states: List[str],
        observations: List[str],
        A_matrix: torch.Tensor,
        B_matrix: torch.Tensor,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Visualize inference as a graph"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        # Create observation model graph
        G_obs = nx.Graph()
        # Add nodes
        for s in states:
            G_obs.add_node(f"S_{s}", node_type="state")
        for o in observations:
            G_obs.add_node(f"O_{o}", node_type="observation")
        # Add edges based on A matrix
        A_np = A_matrix.detach().cpu().numpy()
        for i, o in enumerate(observations):
            for j, s in enumerate(states):
                if A_np[i, j] > 0.01:  # Threshold for visibility
                    G_obs.add_edge(f"S_{s}", f"O_{o}", weight=A_np[i, j])
        # Layout and draw
        pos = nx.spring_layout(G_obs)
        # Draw nodes
        state_nodes = [n for n in G_obs.nodes() if G_obs.nodes[n].get("node_type") == "state"]
        obs_nodes = [n for n in G_obs.nodes() if G_obs.nodes[n].get("node_type") == "observation"]
        nx.draw_networkx_nodes(
            G_obs,
            pos,
            nodelist=state_nodes,
            node_color="lightblue",
            node_size=500,
            ax=ax1,
        )
        nx.draw_networkx_nodes(
            G_obs,
            pos,
            nodelist=obs_nodes,
            node_color="lightcoral",
            node_size=500,
            ax=ax1,
        )
        # Draw edges with weights
        edges = G_obs.edges(data=True)
        weights = [e[2]["weight"] for e in edges]
        nx.draw_networkx_edges(G_obs, pos, width=np.array(weights) * 5, alpha=0.6, ax=ax1)
        nx.draw_networkx_labels(G_obs, pos, ax=ax1)
        ax1.set_title("Observation Model (A Matrix)")
        ax1.axis("off")
        # Create transition model visualization
        if B_matrix.dim() == 3:
            # Show transition for first action
            B_action0 = B_matrix[:, :, 0].detach().cpu().numpy()
            im = ax2.imshow(B_action0, cmap="viridis", aspect="auto")
            ax2.set_xlabel("Current State")
            ax2.set_ylabel("Next State")
            ax2.set_title("Transition Model (B Matrix - Action 0)")
            ax2.set_xticks(range(len(states)))
            ax2.set_yticks(range(len(states)))
            ax2.set_xticklabels(states)
            ax2.set_yticklabels(states)
            # Add colorbar
            plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        if save_path or self.config.save_figures:
            save_path = save_path or self.config.figure_dir / "inference_graph.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi)
        return fig

    def create_realtime_dashboard(
        self, belief_tracker: BeliefTracker, fe_monitor: FreeEnergyMonitor
    ) -> FuncAnimation:
        """Create real-time dashboard animation"""
        fig = plt.figure(figsize=(15, 10))
        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Belief evolution
        ax2 = fig.add_subplot(gs[1, 0])  # Current belief
        ax3 = fig.add_subplot(gs[1, 1])  # Entropy
        ax4 = fig.add_subplot(gs[2, :])  # Free energy
        # Initialize plots
        lines: Dict[str, Any] = {}

        def init() -> List[Any]:
            ax1.set_xlim(0, 60)  # 60 seconds window
            ax1.set_ylim(0, 1)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Belief Probability")
            ax1.set_title("Belief State Evolution")
            ax1.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("State")
            ax2.set_ylabel("Probability")
            ax2.set_title("Current Belief State")
            ax3.set_xlim(0, 60)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Entropy")
            ax3.set_title("Belief Entropy")
            ax3.grid(True, alpha=0.3)
            ax4.set_xlim(0, 60)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Free Energy")
            ax4.set_title("Free Energy Components")
            ax4.grid(True, alpha=0.3)
            return []

        def update(frame: int) -> List[Any]:
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            # Reinitialize
            init()
            # Update belief evolution
            if len(belief_tracker.belief_history) > 0:
                beliefs = np.array(list(belief_tracker.belief_history))
                timestamps = np.array(list(belief_tracker.timestamp_history))
                # Plot trajectories
                for i in range(belief_tracker.num_states):
                    ax1.plot(timestamps, beliefs[:, i], label=belief_tracker.state_labels[i])
                ax1.legend(loc="upper right")
                # Current belief bar chart
                current_belief = beliefs[-1]
                ax2.bar(range(len(current_belief)), current_belief)
                ax2.set_xticks(range(len(current_belief)))
                ax2.set_xticklabels(belief_tracker.state_labels, rotation=45)
                # Entropy
                ax3.plot(timestamps, list(belief_tracker.entropy_history), "k-")
            # Update free energy
            if len(fe_monitor.vfe_history) > 0:
                timestamps = np.array(list(fe_monitor.timestamps))
                ax4.plot(
                    timestamps,
                    list(fe_monitor.vfe_history),
                    "k-",
                    label="VFE",
                    linewidth=2,
                )
                ax4.plot(
                    timestamps,
                    list(fe_monitor.accuracy_history),
                    "r--",
                    label="Accuracy",
                )
                ax4.plot(
                    timestamps,
                    list(fe_monitor.complexity_history),
                    "b--",
                    label="Complexity",
                )
                ax4.legend()
            return []

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            interval=1000 // self.config.update_frequency,
            blit=False,
        )
        return anim


class DiagnosticSuite:
    """
    Complete diagnostic suite for Active Inference.
    """

    def __init__(self, config: Optional[DiagnosticConfig] = None) -> None:
        self.config = config or DiagnosticConfig()
        # Create directories
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)
        # Initialize components
        self.belief_trackers: Dict[str, BeliefTracker] = {}
        self.fe_monitor = FreeEnergyMonitor(self.config)
        self.gradient_analyzer = GradientAnalyzer(self.config)
        self.visualizer = InferenceVisualizer(self.config)
        # Performance tracking
        self.performance_stats: DefaultDict[str, List[float]] = defaultdict(list)
        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup diagnostic logging"""
        log_file = self.config.log_dir / f"diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def create_belief_tracker(
        self, name: str, num_states: int, state_labels: Optional[List[str]] = None
    ) -> BeliefTracker:
        ."""Create named belief tracker."""
        tracker = BeliefTracker(self.config, num_states, state_labels)
        self.belief_trackers[name] = tracker
        return tracker

    def log_inference_step(self, step_data: Dict[str, Any]) -> None:
        ."""Log complete inference step."""
        logger.info(f"Inference step: {json.dumps(step_data, indent=2)}")
        # Track performance metrics
        if "computation_time" in step_data:
            self.performance_stats["inference_time"].append(step_data["computation_time"])

    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "belief_statistics": {},
            "free_energy_statistics": {},
            "gradient_health": {},
            "performance_metrics": {},
        }
        # Belief statistics
        belief_stats: Dict[str, Any] = {}
        for name, tracker in self.belief_trackers.items():
            belief_stats[name] = tracker.get_statistics()
        report["belief_statistics"] = belief_stats
        # Free energy statistics
        if len(self.fe_monitor.vfe_history) > 0:
            report["free_energy_statistics"] = {
                "mean_vfe": np.mean(list(self.fe_monitor.vfe_history)),
                "std_vfe": np.std(list(self.fe_monitor.vfe_history)),
                "mean_accuracy": np.mean(list(self.fe_monitor.accuracy_history)),
                "mean_complexity": np.mean(list(self.fe_monitor.complexity_history)),
            }
        # Gradient health
        if self.config.track_gradients:
            report["gradient_health"] = self.gradient_analyzer.check_gradient_health()
        # Performance metrics
        perf_metrics: Dict[str, Any] = {}
        for metric, values in self.performance_stats.items():
            if values:
                perf_metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        report["performance_metrics"] = perf_metrics
        # Save report
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
        return report

    def create_summary_plots(self) -> Dict[str, Optional[plt.Figure]]:
        """Create all summary plots"""
        plots = {}
        # Belief evolution plots
        for name, tracker in self.belief_trackers.items():
            plots[f"{name}_evolution"] = tracker.plot_belief_evolution()
            plots[f"{name}_heatmap"] = tracker.plot_belief_heatmap()
        # Free energy plots
        plots["free_energy"] = self.fe_monitor.plot_free_energy_components()
        # Gradient flow
        if self.config.track_gradients:
            plots["gradient_flow"] = self.gradient_analyzer.plot_gradient_flow()
        return plots


# Add VFEMonitor alias for backward compatibility
VFEMonitor = FreeEnergyMonitor
# Example usage
if __name__ == "__main__":
    # Configuration
    config = DiagnosticConfig(enable_realtime=True, track_gradients=True, save_figures=True)
    # Create diagnostic suite
    diagnostics = DiagnosticSuite(config)
    # Create belief tracker
    belief_tracker = diagnostics.create_belief_tracker(
        "main_agent", num_states=4, state_labels=["Explore", "Exploit", "Rest", "Flee"]
    )
    # Simulate some data
    for t in range(100):
        # Random belief
        belief = torch.rand(4)
        belief = belief / belief.sum()
        belief_tracker.record_belief(belief, timestamp=t * 0.1)
        # Random free energy
        diagnostics.fe_monitor.record_vfe(
            accuracy=np.random.randn() * 0.1,
            complexity=np.random.randn() * 0.05 + 0.5,
            timestamp=t * 0.1,
        )
        # Random EFE
        efe = torch.randn(3)
        diagnostics.fe_monitor.record_efe(
            efe, action_labels=["Move", "Stay", "Turn"], timestamp=t * 0.1
        )
    # Generate plots
    plots = diagnostics.create_summary_plots()
    # Generate report
    report = diagnostics.generate_report(save_path=Path("diagnostics_report.json"))
    print(f"Diagnostic report: {json.dumps(report, indent=2)}")
