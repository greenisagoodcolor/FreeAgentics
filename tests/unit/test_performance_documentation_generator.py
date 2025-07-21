#!/usr/bin/env python3
"""
Tests for Performance Documentation Generator

Following TDD principles, these tests define the expected behavior
for generating comprehensive performance documentation with charts,
analysis, and recommendations.
"""

import json
import os
import tempfile
import unittest

import pytest

# Test the documentation generator module (to be created)
from tools.performance_documentation_generator import (
    PerformanceAnalyzer,
    PerformanceChartGenerator,
    PerformanceDocumentationGenerator,
    PerformanceMetrics,
)


@pytest.mark.slow
class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics data structures."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics from raw data."""
        metrics = PerformanceMetrics(
            agent_count=50,
            efficiency=28.4,
            coordination_overhead=72.0,
            memory_per_agent_mb=34.5,
            inference_time_ms=50.0,
            throughput_ops_per_sec=250.0,
        )

        self.assertEqual(metrics.agent_count, 50)
        self.assertEqual(metrics.efficiency, 28.4)
        self.assertEqual(metrics.coordination_overhead, 72.0)
        self.assertEqual(metrics.memory_per_agent_mb, 34.5)
        self.assertEqual(metrics.inference_time_ms, 50.0)
        self.assertEqual(metrics.throughput_ops_per_sec, 250.0)

    def test_efficiency_loss_calculation(self):
        """Test calculating efficiency loss from metrics."""
        metrics = PerformanceMetrics(
            agent_count=50,
            efficiency=28.4,
            coordination_overhead=72.0,
            memory_per_agent_mb=34.5,
            inference_time_ms=50.0,
            throughput_ops_per_sec=250.0,
        )

        # Efficiency loss should be 100 - efficiency
        expected_loss = 100 - 28.4
        self.assertAlmostEqual(metrics.efficiency_loss(), expected_loss, places=1)

    def test_total_memory_calculation(self):
        """Test calculating total memory for agent count."""
        metrics = PerformanceMetrics(
            agent_count=50,
            efficiency=28.4,
            coordination_overhead=72.0,
            memory_per_agent_mb=34.5,
            inference_time_ms=50.0,
            throughput_ops_per_sec=250.0,
        )

        expected_total = 50 * 34.5
        self.assertEqual(metrics.total_memory_mb(), expected_total)


@pytest.mark.slow
class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analysis functionality."""

    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "agent_counts": [1, 5, 10, 20, 30, 50],
            "efficiencies": [95.0, 78.5, 65.2, 48.3, 35.7, 28.4],
            "throughputs": [680.5, 540.0, 408.0, 340.0, 272.0, 250.0],
            "memory_usage": [34.5, 172.5, 345.0, 690.0, 1035.0, 1725.0],
            "inference_times": [1.4, 12.5, 25.0, 35.0, 42.0, 50.0],
        }
        self.analyzer = PerformanceAnalyzer()

    def test_identify_bottlenecks(self):
        """Test identifying performance bottlenecks."""
        bottlenecks = self.analyzer.identify_bottlenecks(self.test_data)

        # Should identify key bottlenecks
        self.assertIn("coordination_overhead", bottlenecks)
        self.assertIn("memory_scaling", bottlenecks)
        self.assertIn("gil_contention", bottlenecks)

        # Check bottleneck details
        coord_bottleneck = bottlenecks["coordination_overhead"]
        self.assertEqual(coord_bottleneck["severity"], "critical")
        self.assertEqual(coord_bottleneck["efficiency_loss_at_50_agents"], 71.6)

    def test_root_cause_analysis(self):
        """Test root cause analysis of performance issues."""
        root_causes = self.analyzer.analyze_root_causes(self.test_data)

        # Should identify Python GIL as major cause
        self.assertIn("python_gil", root_causes)
        gil_cause = root_causes["python_gil"]
        self.assertEqual(gil_cause["impact"], "high")
        self.assertIn("thread_serialization", gil_cause["effects"])

        # Should identify async coordination overhead
        self.assertIn("async_coordination", root_causes)
        async_cause = root_causes["async_coordination"]
        self.assertEqual(async_cause["impact"], "high")
        self.assertIn("context_switching", async_cause["effects"])

    def test_generate_recommendations(self):
        """Test generating optimization recommendations."""
        recommendations = self.analyzer.generate_recommendations(self.test_data)

        # Should have immediate, medium-term, and long-term recommendations
        self.assertIn("immediate", recommendations)
        self.assertIn("medium_term", recommendations)
        self.assertIn("long_term", recommendations)

        # Check immediate recommendations
        immediate = recommendations["immediate"]
        self.assertGreater(len(immediate), 0)
        self.assertIn("impact", immediate[0])
        self.assertIn("effort", immediate[0])
        self.assertIn("description", immediate[0])


@pytest.mark.slow
class TestPerformanceChartGenerator(unittest.TestCase):
    """Test performance chart generation."""

    def setUp(self):
        """Set up test data and temp directory."""
        self.test_data = {
            "agent_counts": [1, 5, 10, 20, 30, 50],
            "efficiencies": [95.0, 78.5, 65.2, 48.3, 35.7, 28.4],
            "throughputs": [680.5, 540.0, 408.0, 340.0, 272.0, 250.0],
            "memory_usage": [34.5, 172.5, 345.0, 690.0, 1035.0, 1725.0],
        }
        self.temp_dir = tempfile.mkdtemp()
        self.chart_generator = PerformanceChartGenerator(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_generate_efficiency_chart(self):
        """Test generating efficiency vs agent count chart."""
        chart_path = self.chart_generator.generate_efficiency_chart(self.test_data)

        # Chart file should be created
        self.assertTrue(os.path.exists(chart_path))
        self.assertTrue(chart_path.endswith(".png"))

        # File should have content
        file_size = os.path.getsize(chart_path)
        self.assertGreater(file_size, 0)

    def test_generate_throughput_chart(self):
        """Test generating throughput comparison chart."""
        chart_path = self.chart_generator.generate_throughput_chart(self.test_data)

        self.assertTrue(os.path.exists(chart_path))
        self.assertTrue(chart_path.endswith(".png"))

    def test_generate_memory_scaling_chart(self):
        """Test generating memory scaling chart."""
        chart_path = self.chart_generator.generate_memory_scaling_chart(self.test_data)

        self.assertTrue(os.path.exists(chart_path))
        self.assertTrue(chart_path.endswith(".png"))

    def test_generate_bottleneck_heatmap(self):
        """Test generating bottleneck analysis heatmap."""
        bottleneck_data = {
            "factors": ["GIL", "Memory", "Coordination", "I/O"],
            "agent_counts": [1, 10, 50],
            "impact_matrix": [
                [10, 20, 80],  # GIL impact
                [5, 30, 60],  # Memory impact
                [15, 40, 72],  # Coordination impact
                [5, 10, 15],  # I/O impact
            ],
        }

        chart_path = self.chart_generator.generate_bottleneck_heatmap(bottleneck_data)
        self.assertTrue(os.path.exists(chart_path))


@pytest.mark.slow
class TestPerformanceDocumentationGenerator(unittest.TestCase):
    """Test complete documentation generation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.doc_generator = PerformanceDocumentationGenerator(output_dir=self.temp_dir)

        # Load test data from actual performance results
        self.test_results = {
            "coordination_load_test": {
                "agent_counts": [1, 5, 10, 20, 30, 50],
                "efficiencies": [95.0, 78.5, 65.2, 48.3, 35.7, 28.4],
                "throughputs": [680.5, 540.0, 408.0, 340.0, 272.0, 250.0],
                "memory_usage": [34.5, 172.5, 345.0, 690.0, 1035.0, 1725.0],
                "inference_times": [1.4, 12.5, 25.0, 35.0, 42.0, 50.0],
            },
            "memory_analysis": {
                "per_agent_mb": 34.5,
                "growth_rate": "linear",
                "optimization_potential": 84.0,
            },
            "threading_benchmark": {
                "vs_multiprocessing": {
                    "speedup_single": 49.35,
                    "speedup_5_agents": 4.0,
                    "speedup_10_agents": 3.09,
                }
            },
        }

    def tearDown(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_generate_comprehensive_documentation(self):
        """Test generating complete performance documentation."""
        doc_path = self.doc_generator.generate_comprehensive_documentation(self.test_results)

        # Documentation file should be created
        self.assertTrue(os.path.exists(doc_path))
        self.assertTrue(doc_path.endswith(".md"))

        # Read and verify content
        with open(doc_path, "r") as f:
            content = f.read()

        # Should contain all major sections
        self.assertIn("# Multi-Agent Coordination Performance Limits", content)
        self.assertIn("## Executive Summary", content)
        self.assertIn("## Performance Analysis Charts", content)
        self.assertIn("## Root Cause Analysis", content)
        self.assertIn("## Benchmarking Methodology", content)
        self.assertIn("## Performance Bottlenecks", content)
        self.assertIn("## Optimization Recommendations", content)

        # Should contain specific metrics
        self.assertIn("28.4% efficiency", content)
        self.assertIn("72% efficiency loss", content)
        self.assertIn("34.5 MB per agent", content)

    def test_generate_benchmarking_methodology(self):
        """Test generating benchmarking methodology documentation."""
        methodology_doc = self.doc_generator.generate_benchmarking_methodology()

        # Should describe test setup
        self.assertIn("Test Environment", methodology_doc)
        self.assertIn("Test Scenarios", methodology_doc)
        self.assertIn("Measurement Techniques", methodology_doc)
        self.assertIn("Statistical Analysis", methodology_doc)

    def test_generate_html_report(self):
        """Test generating HTML performance report."""
        html_path = self.doc_generator.generate_html_report(self.test_results)

        # HTML file should be created
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(html_path.endswith(".html"))

        # Read and verify HTML structure
        with open(html_path, "r") as f:
            html_content = f.read()

        self.assertIn("<html>", html_content)
        self.assertIn(
            "<h1>Multi-Agent Coordination Performance Analysis</h1>",
            html_content,
        )
        self.assertIn("efficiency-chart.png", html_content)
        self.assertIn("memory-scaling-chart.png", html_content)

    def test_export_performance_data(self):
        """Test exporting performance data in various formats."""
        # Export as JSON
        json_path = self.doc_generator.export_performance_data(self.test_results, format="json")
        self.assertTrue(os.path.exists(json_path))

        # Verify JSON content
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("coordination_load_test", data)
        self.assertIn("memory_analysis", data)

        # Export as CSV
        csv_paths = self.doc_generator.export_performance_data(self.test_results, format="csv")
        self.assertIsInstance(csv_paths, list)
        self.assertGreater(len(csv_paths), 0)

        for csv_path in csv_paths:
            self.assertTrue(os.path.exists(csv_path))


if __name__ == "__main__":
    unittest.main()
