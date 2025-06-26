import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from inference.engine.diagnostics import (
    BeliefTracker,
    DiagnosticConfig,
    DiagnosticSuite,
    FreeEnergyMonitor,
    GradientAnalyzer,
    InferenceVisualizer,
)


class TestDiagnosticConfig:
    def test_default_config(self) -> None:
        """Test default diagnostic configuration"""
        config = DiagnosticConfig()
        assert config.log_level == "INFO"
        assert config.figure_size == (10, 8)
        assert config.buffer_size == 1000
        assert config.track_beliefs is True
        assert config.enable_realtime is True

    def test_custom_config(self) -> None:
        """Test custom diagnostic configuration"""
        config = DiagnosticConfig(
            log_level="DEBUG",
            buffer_size=500,
            figure_size=(12, 10),
            track_gradients=False,
        )
        assert config.log_level == "DEBUG"
        assert config.buffer_size == 500
        assert config.figure_size == (12, 10)
        assert config.track_gradients is False


class TestBeliefTracker:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = DiagnosticConfig(save_figures=False)
        self.tracker = BeliefTracker(self.config, num_states=4, state_labels=["A", "B", "C", "D"])

    def test_initialization(self) -> None:
        """Test belief tracker initialization"""
        assert self.tracker.num_states == 4
        assert len(self.tracker.state_labels) == 4
        assert self.tracker.state_labels[0] == "A"
        assert self.tracker.total_updates == 0

    def test_record_belief(self) -> None:
        """Test recording belief states"""
        belief = torch.tensor([0.4, 0.3, 0.2, 0.1])
        self.tracker.record_belief(belief)
        assert len(self.tracker.belief_history) == 1
        assert len(self.tracker.timestamp_history) == 1
        assert len(self.tracker.entropy_history) == 1
        assert self.tracker.total_updates == 1
        assert np.allclose(self.tracker.belief_history[0], belief.numpy())

    def test_entropy_calculation(self) -> None:
        """Test entropy calculation"""
        uniform_belief = torch.ones(4) / 4
        self.tracker.record_belief(uniform_belief)
        peaked_belief = torch.tensor([0.9, 0.05, 0.03, 0.02])
        self.tracker.record_belief(peaked_belief)
        assert self.tracker.entropy_history[0] > self.tracker.entropy_history[1]

    def test_plot_belief_evolution(self) -> None:
        """Test belief evolution plotting"""
        for i in range(10):
            belief = torch.rand(4)
            belief = belief / belief.sum()
            self.tracker.record_belief(belief, timestamp=i * 0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            fig = self.tracker.plot_belief_evolution(save_path=save_path)
            assert fig is not None
            assert save_path.exists()
            plt.close(fig)

    def test_plot_belief_heatmap(self) -> None:
        """Test belief heatmap plotting"""
        for i in range(20):
            belief = torch.rand(4)
            belief = belief / belief.sum()
            self.tracker.record_belief(belief)
        fig = self.tracker.plot_belief_heatmap()
        assert fig is not None
        plt.close(fig)

    def test_get_statistics(self) -> None:
        """Test statistics computation"""
        beliefs = [
            torch.tensor([0.9, 0.05, 0.03, 0.02]),
            torch.tensor([0.1, 0.8, 0.05, 0.05]),
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor([0.05, 0.05, 0.05, 0.85]),
        ]
        for belief in beliefs:
            self.tracker.record_belief(belief)
        stats = self.tracker.get_statistics()
        assert stats["total_updates"] == 4
        assert "mean_entropy" in stats
        assert "std_entropy" in stats
        assert "dominant_states" in stats
        assert len(stats["dominant_states"]) > 0


class TestFreeEnergyMonitor:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = DiagnosticConfig(save_figures=False)
        self.monitor = FreeEnergyMonitor(self.config)

    def test_record_vfe(self) -> None:
        """Test recording variational free energy"""
        accuracy = 0.5
        complexity = 0.3
        self.monitor.record_vfe(accuracy, complexity)
        assert len(self.monitor.vfe_history) == 1
        assert len(self.monitor.accuracy_history) == 1
        assert len(self.monitor.complexity_history) == 1
        assert self.monitor.vfe_history[0] == accuracy + complexity

    def test_record_efe(self) -> None:
        """Test recording expected free energy"""
        efe_values = torch.tensor([1.2, 0.8, 1.5])
        action_labels = ["Left", "Right", "Stay"]
        self.monitor.record_efe(efe_values, action_labels)
        assert len(self.monitor.efe_history) == 1
        assert abs(self.monitor.efe_history[0] - 0.8) < 1e-6
        assert len(self.monitor.action_efe_history) == 3
        assert abs(self.monitor.action_efe_history["Right"][0] - 0.8) < 1e-6

    def test_plot_free_energy_components(self) -> None:
        """Test free energy plotting"""
        for i in range(15):
            self.monitor.record_vfe(
                accuracy=np.random.randn() * 0.1,
                complexity=0.5 + np.random.randn() * 0.05,
                timestamp=i * 0.1,
            )
            efe = torch.randn(3)
            self.monitor.record_efe(efe, timestamp=i * 0.1)
        fig = self.monitor.plot_free_energy_components()
        assert fig is not None
        plt.close(fig)


class TestGradientAnalyzer:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = DiagnosticConfig()
        self.analyzer = GradientAnalyzer(self.config)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        )

    def test_analyze_gradients(self) -> None:
        """Test gradient analysis"""
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)
        self.analyzer.analyze_gradients(self.model)
        assert self.analyzer.update_count == 1
        assert len(self.analyzer.gradient_norms) > 0
        for name, param in self.model.named_parameters():
            assert name in self.analyzer.gradient_norms
            assert len(self.analyzer.gradient_norms[name]) == 1

    def test_gradient_health_check(self) -> None:
        """Test gradient health checking"""
        for _ in range(10):
            for param in self.model.parameters():
                param.grad = torch.zeros_like(param) + 1e-08
            self.analyzer.analyze_gradients(self.model)
        issues = self.analyzer.check_gradient_health()
        assert len(issues["vanishing_gradients"]) > 0

    def test_plot_gradient_flow(self) -> None:
        """Test gradient flow plotting"""
        for _ in range(5):
            for param in self.model.parameters():
                param.grad = torch.randn_like(param)
            self.analyzer.analyze_gradients(self.model)
        fig = self.analyzer.plot_gradient_flow()
        assert fig is not None
        plt.close(fig)


class TestInferenceVisualizer:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = DiagnosticConfig(save_figures=False)
        self.visualizer = InferenceVisualizer(self.config)

    def test_visualize_inference_graph(self) -> None:
        """Test inference graph visualization"""
        states = ["S1", "S2", "S3"]
        observations = ["O1", "O2"]
        A = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]])
        B = torch.rand(3, 3, 2)
        B = B / B.sum(dim=0, keepdim=True)
        fig = self.visualizer.visualize_inference_graph(states, observations, A, B)
        assert fig is not None
        plt.close(fig)


class TestDiagnosticSuite:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DiagnosticConfig(
            log_dir=Path(self.temp_dir) / "logs",
            figure_dir=Path(self.temp_dir) / "figures",
            save_figures=False,
        )
        self.suite = DiagnosticSuite(self.config)

    def teardown_method(self):
        """Cleanup after tests"""
        shutil.rmtree(self.temp_dir)

    def test_create_belief_tracker(self) -> None:
        """Test creating belief tracker"""
        tracker = self.suite.create_belief_tracker(
            "agent1", num_states=3, state_labels=["A", "B", "C"]
        )
        assert "agent1" in self.suite.belief_trackers
        assert tracker.num_states == 3

    def test_log_inference_step(self) -> None:
        """Test logging inference step"""
        step_data = {
            "timestep": 10,
            "action": "move_left",
            "belief": [0.3, 0.4, 0.3],
            "computation_time": 0.015,
        }
        self.suite.log_inference_step(step_data)
        assert len(self.suite.performance_stats["inference_time"]) == 1
        assert self.suite.performance_stats["inference_time"][0] == 0.015

    def test_generate_report(self) -> None:
        """Test report generation"""
        tracker = self.suite.create_belief_tracker("test", num_states=2)
        for i in range(5):
            belief = torch.rand(2)
            belief = belief / belief.sum()
            tracker.record_belief(belief)
        self.suite.fe_monitor.record_vfe(0.5, 0.3)
        report = self.suite.generate_report()
        assert "timestamp" in report
        assert "belief_statistics" in report
        assert "test" in report["belief_statistics"]
        assert report["belief_statistics"]["test"]["total_updates"] == 5

    def test_create_summary_plots(self) -> None:
        """Test summary plot creation"""
        tracker = self.suite.create_belief_tracker("agent", num_states=3)
        for i in range(10):
            belief = torch.rand(3)
            belief = belief / belief.sum()
            tracker.record_belief(belief)
        for i in range(10):
            self.suite.fe_monitor.record_vfe(
                accuracy=0.5 + np.random.randn() * 0.1,
                complexity=0.3 + np.random.randn() * 0.05,
            )
        plots = self.suite.create_summary_plots()
        assert "agent_evolution" in plots
        assert "agent_heatmap" in plots
        assert "free_energy" in plots
        for fig in plots.values():
            if fig is not None:
                plt.close(fig)

    def test_integration(self) -> None:
        """Test integrated diagnostic workflow"""
        tracker = self.suite.create_belief_tracker(
            "main", num_states=4, state_labels=["Explore", "Exploit", "Rest", "Flee"]
        )
        for t in range(20):
            belief = torch.rand(4)
            belief = belief / belief.sum()
            tracker.record_belief(belief)
            self.suite.fe_monitor.record_vfe(
                accuracy=-0.5 + np.random.randn() * 0.1,
                complexity=1.0 + np.random.randn() * 0.05,
            )
            self.suite.log_inference_step(
                {
                    "timestep": t,
                    "belief": belief.tolist(),
                    "computation_time": 0.01 + np.random.rand() * 0.01,
                }
            )
        plots = self.suite.create_summary_plots()
        report = self.suite.generate_report()
        assert len(plots) > 0
        assert report["belief_statistics"]["main"]["total_updates"] == 20
        assert len(report["performance_metrics"]["inference_time"]) > 0
        for fig in plots.values():
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
