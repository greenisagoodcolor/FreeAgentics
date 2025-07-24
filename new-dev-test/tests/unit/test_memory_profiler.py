"""Unit tests for PyMDP memory profiler."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from scripts.memory_profiler_pymdp import MemoryProfiler


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler()

    def test_memory_measurement(self):
        """Test basic memory measurement."""
        # Get initial memory
        mem1 = self.profiler.get_memory_usage()

        # Allocate some memory
        np.zeros((1000, 1000))  # ~8MB

        # Get memory after allocation
        mem2 = self.profiler.get_memory_usage()

        # Memory should have increased
        self.assertGreater(mem2["rss_mb"], mem1["rss_mb"])

    def test_baseline_setting(self):
        """Test baseline memory setting."""
        self.profiler.set_baseline()

        # Baseline should be set
        self.assertGreater(self.profiler.baseline_memory, 0)

    def test_measurement_recording(self):
        """Test measurement recording."""
        self.profiler.set_baseline()

        # Make a measurement
        measurement = self.profiler.measure("Test measurement")

        # Check measurement structure
        self.assertIn("label", measurement)
        self.assertIn("rss_mb", measurement)
        self.assertIn("delta_mb", measurement)
        self.assertIn("timestamp", measurement)

        # Check measurement was recorded
        self.assertEqual(len(self.profiler.measurements), 1)
        self.assertEqual(self.profiler.measurements[0]["label"], "Test measurement")

    def test_component_memory_profiling(self):
        """Test component memory profiling."""
        component_memory = self.profiler.profile_agent_components()

        # Check that components were profiled
        self.assertIsInstance(component_memory, dict)
        self.assertIn("beliefs", component_memory)
        self.assertIn("transitions", component_memory)
        self.assertIn("observations", component_memory)
        self.assertIn("preferences", component_memory)

    def test_operation_memory_profiling(self):
        """Test operation memory profiling."""
        # Use fewer steps for testing
        operation_memory = self.profiler.profile_agent_operations(n_steps=10)

        # Check that operations were profiled
        self.assertIsInstance(operation_memory, dict)
        self.assertIn("perception", operation_memory)
        self.assertIn("belief_updates", operation_memory)
        self.assertIn("action_selection", operation_memory)

    def test_hotspot_identification(self):
        """Test memory hotspot identification."""
        # Create some large arrays
        np.zeros((1000, 1000))  # ~8MB
        np.ones((500, 500))  # ~2MB

        hotspots = self.profiler.identify_memory_hotspots()

        # Check hotspot structure
        self.assertIn("large_arrays", hotspots)
        self.assertIn("memory_leaks", hotspots)
        self.assertIn("inefficient_operations", hotspots)

        # Should find our large arrays
        self.assertGreater(len(hotspots["large_arrays"]), 0)

    def test_report_generation(self):
        """Test report generation."""
        # Make some measurements
        self.profiler.set_baseline()
        self.profiler.measure("Test 1")
        self.profiler.measure("Test 2")

        # Generate report
        report = self.profiler.generate_report()

        # Check report content
        self.assertIn("PYMDP AGENT MEMORY PROFILING REPORT", report)
        self.assertIn("SUMMARY:", report)
        self.assertIn("MEMORY TIMELINE:", report)
        self.assertIn("Test 1", report)
        self.assertIn("Test 2", report)

    @patch("scripts.memory_profiler_pymdp.BaseAgent")
    def test_agent_creation_profiling(self, mock_base_agent):
        """Test agent creation profiling."""
        # Mock agent creation
        mock_base_agent.return_value = MagicMock()

        # Profile agent creation
        measurements = self.profiler.profile_agent_creation(n_agents=3)

        # Should have measurements for each agent
        self.assertGreater(len(measurements), 0)

    def test_array_creation_functions(self):
        """Test array creation helper functions."""
        # Test belief states
        beliefs = self.profiler._create_belief_states()
        self.assertEqual(beliefs.shape[0], 10)  # 10 agents
        self.assertAlmostEqual(beliefs[0].sum(), 1.0, places=5)  # Normalized

        # Test transition matrices
        transitions = self.profiler._create_transition_matrices()
        self.assertEqual(transitions.shape[0], 4)  # 4 actions

        # Test observation matrices
        obs = self.profiler._create_observation_matrices()
        self.assertEqual(len(obs.shape), 2)

        # Test preferences
        prefs = self.profiler._create_preference_matrices()
        self.assertEqual(len(prefs.shape), 1)


if __name__ == "__main__":
    unittest.main()
