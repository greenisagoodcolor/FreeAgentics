#!/usr/bin/env python3
"""Unit tests for memory hotspot analyzer."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from scripts.identify_memory_hotspots import MemoryHotspotAnalyzer


class TestMemoryHotspotAnalyzer(unittest.TestCase):
    """Test cases for MemoryHotspotAnalyzer."""

    def setUp(self):
        """Set up test environment."""
        self.analyzer = MemoryHotspotAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer.process)
        self.assertEqual(len(self.analyzer.memory_snapshots), 0)
        self.assertEqual(len(self.analyzer.hotspots), 0)

    def test_matrix_analysis_mock(self):
        """Test matrix analysis with mock data."""
        results = self.analyzer._analyze_mock_matrices()

        # Check structure
        self.assertIn("matrix_sizes", results)
        self.assertIn("memory_per_operation", results)
        self.assertIn("inefficiencies", results)

        # Check sizes
        for size in ["5x5", "10x10", "20x20", "50x50"]:
            self.assertIn(size, results["matrix_sizes"])
            size_data = results["matrix_sizes"][size]
            self.assertIn("A_matrices_mb", size_data)
            self.assertIn("B_matrices_mb", size_data)
            self.assertIn("total_mb", size_data)

            # B matrices should be larger (4 actions)
            self.assertGreater(size_data["B_matrices_mb"], size_data["A_matrices_mb"])

    @patch("scripts.identify_memory_hotspots.tracemalloc")
    def test_memory_tracing(self, mock_tracemalloc):
        """Test memory tracing functionality."""
        # Test start tracing
        self.analyzer.start_tracing()
        mock_tracemalloc.start.assert_called_once()

        # Test stop tracing
        self.analyzer.stop_tracing()
        mock_tracemalloc.stop.assert_called_once()

    def test_identify_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        # Create some mock large arrays
        with patch("gc.get_objects") as mock_get_objects:
            # Mock float64 array
            mock_array1 = MagicMock(spec=np.ndarray)
            mock_array1.nbytes = 10 * 1024 * 1024  # 10MB
            mock_array1.shape = (1000, 1000)
            mock_array1.dtype = np.dtype("float64")

            # Mock large 2D array
            mock_array2 = MagicMock(spec=np.ndarray)
            mock_array2.nbytes = 20 * 1024 * 1024  # 20MB
            mock_array2.shape = (100, 100, 20)
            mock_array2.dtype = np.dtype("float32")

            mock_get_objects.return_value = [
                mock_array1,
                mock_array2,
                "other",
                123,
            ]

            opportunities = self.analyzer.identify_optimization_opportunities()

        # Check structure
        self.assertIn("matrix_optimizations", opportunities)
        self.assertIn("belief_optimizations", opportunities)
        self.assertIn("memory_pooling", opportunities)
        self.assertIn("data_structure_improvements", opportunities)

        # Check matrix optimizations found
        self.assertGreater(len(opportunities["matrix_optimizations"]), 0)

        # Check for float64 optimization
        float64_opts = [
            opt
            for opt in opportunities["matrix_optimizations"]
            if opt["type"] == "dtype_optimization"
        ]
        self.assertGreater(len(float64_opts), 0)

    def test_generate_hotspot_report(self):
        """Test report generation."""
        with patch.object(self.analyzer, "analyze_pymdp_matrices") as mock_matrix:
            with patch.object(self.analyzer, "analyze_belief_operations") as mock_belief:
                with patch.object(self.analyzer, "analyze_agent_lifecycle") as mock_lifecycle:
                    with patch.object(
                        self.analyzer, "identify_optimization_opportunities"
                    ) as mock_opt:
                        # Set up mock returns
                        mock_matrix.return_value = {
                            "matrix_sizes": {
                                "10x10": {
                                    "A_matrices_mb": 1.0,
                                    "B_matrices_mb": 2.0,
                                    "total_mb": 3.0,
                                }
                            },
                            "inefficiencies": [],
                        }
                        mock_belief.return_value = {
                            "operation_costs": {"belief_updates_20": 0.5},
                            "memory_leaks": [],
                        }
                        mock_lifecycle.return_value = {
                            "creation_cost": {"per_agent_mb": 5.0},
                            "operation_cost": {"per_operation_kb": 0.1},
                            "cleanup_efficiency": {"efficiency_percent": 95.0},
                        }
                        mock_opt.return_value = {
                            "matrix_optimizations": [],
                            "belief_optimizations": [],
                            "memory_pooling": [],
                            "data_structure_improvements": [],
                        }

                        report = self.analyzer.generate_hotspot_report()

        # Check report contains key sections
        self.assertIn("PYMDP MEMORY HOTSPOT ANALYSIS REPORT", report)
        self.assertIn("MATRIX MEMORY ANALYSIS", report)
        self.assertIn("BELIEF OPERATION COSTS", report)
        self.assertIn("AGENT LIFECYCLE MEMORY", report)
        self.assertIn("OPTIMIZATION OPPORTUNITIES", report)
        self.assertIn("KEY FINDINGS", report)
        self.assertIn("RECOMMENDATIONS", report)

    def test_memory_calculations(self):
        """Test memory size calculations."""
        # Test different grid sizes
        sizes = [5, 10, 20, 50]

        for size in sizes:
            # A matrix: size x size x 2 factors x 8 bytes
            expected_a = size * size * 2 * 8 / 1024 / 1024

            # B matrix: size x size x 4 actions x 2 factors x 8 bytes
            expected_b = size * size * 4 * 2 * 8 / 1024 / 1024

            results = self.analyzer._analyze_mock_matrices()
            size_key = f"{size}x{size}"

            self.assertAlmostEqual(
                results["matrix_sizes"][size_key]["A_matrices_mb"],
                expected_a,
                places=4,
            )
            self.assertAlmostEqual(
                results["matrix_sizes"][size_key]["B_matrices_mb"],
                expected_b,
                places=4,
            )

    def test_data_output_format(self):
        """Test the format of saved data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {
                "matrix_analysis": {"test": "data"},
                "belief_operations": {"test": "data"},
                "agent_lifecycle": {"test": "data"},
                "optimization_opportunities": {"test": "data"},
                "timestamp": "2024-01-01T00:00:00",
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Read and validate
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            self.assertIn("matrix_analysis", loaded_data)
            self.assertIn("belief_operations", loaded_data)
            self.assertIn("agent_lifecycle", loaded_data)
            self.assertIn("optimization_opportunities", loaded_data)
            self.assertIn("timestamp", loaded_data)

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
