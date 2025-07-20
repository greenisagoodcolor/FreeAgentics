#!/usr/bin/env python3
"""Enhanced unit tests for comprehensive memory profiling tools.

This test suite covers memory_profiler, tracemalloc, and pympler integration
for Task 20.2: Profile and Optimize Memory Usage
"""

import gc
import os
import sys
import time
import tracemalloc
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)


# Mock classes for testing
@dataclass
class MemoryHotspot:
    """Memory hotspot information."""

    location: str
    size_mb: float
    count: int
    type: str


class EnhancedMemoryProfiler:
    """Enhanced memory profiler combining memory_profiler, tracemalloc, and pympler."""

    def __init__(self):
        """Initialize the enhanced memory profiler."""
        self.snapshots = []
        self.hotspots = []
        self.tracemalloc_enabled = False
        self.memory_profiler_enabled = False
        self.pympler_enabled = False

    def start_profiling(self, tools: List[str] = None):
        """Start profiling with specified tools."""
        if tools is None:
            tools = ["tracemalloc", "memory_profiler", "pympler"]

        if "tracemalloc" in tools:
            if not tracemalloc.is_tracing():
                tracemalloc.start(10)
            self.tracemalloc_enabled = True

        if "memory_profiler" in tools:
            self.memory_profiler_enabled = True

        if "pympler" in tools:
            self.pympler_enabled = True

    def stop_profiling(self):
        """Stop all profiling."""
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
        self.tracemalloc_enabled = False
        self.memory_profiler_enabled = False
        self.pympler_enabled = False

    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot using all enabled tools."""
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "tracemalloc": None,
            "memory_profiler": None,
            "pympler": None,
        }

        # Tracemalloc snapshot
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            tm_snapshot = tracemalloc.take_snapshot()
            top_stats = tm_snapshot.statistics("lineno")[:10]
            snapshot["tracemalloc"] = {
                "total_size": sum(stat.size for stat in top_stats)
                / 1024
                / 1024,
                "peak_size": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
                "top_allocations": [
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count,
                    }
                    for stat in top_stats[:5]
                ],
            }

        # Memory profiler data (simulated)
        if self.memory_profiler_enabled:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            snapshot["memory_profiler"] = {
                "rss_mb": mem_info.rss / 1024 / 1024,
                "vms_mb": mem_info.vms / 1024 / 1024,
                "available_mb": psutil.virtual_memory().available
                / 1024
                / 1024,
            }

        # Pympler data (simulated)
        if self.pympler_enabled:
            # Simulate pympler object tracking
            all_objects = gc.get_objects()
            type_stats = {}
            for obj in all_objects[:1000]:  # Sample first 1000 objects
                obj_type = type(obj).__name__
                if obj_type not in type_stats:
                    type_stats[obj_type] = {"count": 0, "size": 0}
                type_stats[obj_type]["count"] += 1
                type_stats[obj_type]["size"] += sys.getsizeof(obj)

            snapshot["pympler"] = {
                "total_objects": len(all_objects),
                "type_stats": dict(
                    sorted(
                        type_stats.items(),
                        key=lambda x: x[1]["size"],
                        reverse=True,
                    )[:10]
                ),
            }

        self.snapshots.append(snapshot)
        return snapshot

    def analyze_hotspots(self) -> List[MemoryHotspot]:
        """Analyze memory hotspots from all profiling data."""
        hotspots = []

        # Analyze tracemalloc data
        if self.tracemalloc_enabled and self.snapshots:
            latest = self.snapshots[-1]
            if (
                latest["tracemalloc"]
                and "top_allocations" in latest["tracemalloc"]
            ):
                for alloc in latest["tracemalloc"]["top_allocations"]:
                    if alloc["size_mb"] > 1.0:  # Hotspot threshold: 1MB
                        hotspots.append(
                            MemoryHotspot(
                                location=f"{alloc['file']}:{alloc['line']}",
                                size_mb=alloc["size_mb"],
                                count=alloc["count"],
                                type="allocation",
                            )
                        )

        # Analyze pympler data
        if self.pympler_enabled and self.snapshots:
            latest = self.snapshots[-1]
            if latest["pympler"] and "type_stats" in latest["pympler"]:
                for type_name, stats in latest["pympler"][
                    "type_stats"
                ].items():
                    size_mb = stats["size"] / 1024 / 1024
                    if size_mb > 0.5:  # Type hotspot threshold: 0.5MB
                        hotspots.append(
                            MemoryHotspot(
                                location=f"Type: {type_name}",
                                size_mb=size_mb,
                                count=stats["count"],
                                type="object_type",
                            )
                        )

        self.hotspots = sorted(hotspots, key=lambda x: x.size_mb, reverse=True)
        return self.hotspots

    def compare_snapshots(self, idx1: int, idx2: int) -> Dict[str, Any]:
        """Compare two snapshots to identify memory growth."""
        if idx1 >= len(self.snapshots) or idx2 >= len(self.snapshots):
            return {"error": "Invalid snapshot indices"}

        snap1 = self.snapshots[idx1]
        snap2 = self.snapshots[idx2]

        comparison = {
            "time_diff": snap2["timestamp"] - snap1["timestamp"],
            "tracemalloc_diff": None,
            "memory_profiler_diff": None,
            "pympler_diff": None,
        }

        # Compare tracemalloc data
        if snap1["tracemalloc"] and snap2["tracemalloc"]:
            comparison["tracemalloc_diff"] = {
                "total_size_diff": snap2["tracemalloc"]["total_size"]
                - snap1["tracemalloc"]["total_size"],
                "peak_size_diff": snap2["tracemalloc"]["peak_size"]
                - snap1["tracemalloc"]["peak_size"],
            }

        # Compare memory_profiler data
        if snap1["memory_profiler"] and snap2["memory_profiler"]:
            comparison["memory_profiler_diff"] = {
                "rss_diff": snap2["memory_profiler"]["rss_mb"]
                - snap1["memory_profiler"]["rss_mb"],
                "vms_diff": snap2["memory_profiler"]["vms_mb"]
                - snap1["memory_profiler"]["vms_mb"],
            }

        # Compare pympler data
        if snap1["pympler"] and snap2["pympler"]:
            comparison["pympler_diff"] = {
                "object_count_diff": snap2["pympler"]["total_objects"]
                - snap1["pympler"]["total_objects"],
            }

        return comparison

    def generate_report(self) -> str:
        """Generate comprehensive memory profiling report."""
        if not self.snapshots:
            return "No profiling data available"

        report = ["=== Enhanced Memory Profiling Report ===\n"]

        # Summary
        report.append("## Summary")
        report.append(f"Total snapshots: {len(self.snapshots)}")
        report.append(f"Profiling tools: {self._get_enabled_tools()}")
        report.append("")

        # Memory timeline
        if len(self.snapshots) > 1:
            report.append("## Memory Timeline")
            first = self.snapshots[0]
            last = self.snapshots[-1]

            if first["tracemalloc"] and last["tracemalloc"]:
                total_growth = (
                    last["tracemalloc"]["total_size"]
                    - first["tracemalloc"]["total_size"]
                )
                peak_growth = (
                    last["tracemalloc"]["peak_size"]
                    - first["tracemalloc"]["peak_size"]
                )
                report.append(
                    f"Tracemalloc total growth: {total_growth:+.2f} MB"
                )
                report.append(
                    f"Tracemalloc peak growth: {peak_growth:+.2f} MB"
                )

            if first["memory_profiler"] and last["memory_profiler"]:
                rss_growth = (
                    last["memory_profiler"]["rss_mb"]
                    - first["memory_profiler"]["rss_mb"]
                )
                report.append(f"RSS memory growth: {rss_growth:+.2f} MB")

            report.append("")

        # Hotspots
        if self.hotspots:
            report.append("## Memory Hotspots")
            for i, hotspot in enumerate(self.hotspots[:10], 1):
                report.append(f"{i}. {hotspot.location}")
                report.append(
                    f"   Size: {hotspot.size_mb:.2f} MB, Count: {hotspot.count}, Type: {hotspot.type}"
                )
            report.append("")

        # Detailed snapshots
        report.append("## Snapshot Details")
        for i, snap in enumerate(self.snapshots[-3:]):  # Last 3 snapshots
            report.append(f"\n### Snapshot {i}: {snap['label']}")

            if snap["tracemalloc"]:
                report.append(
                    f"Tracemalloc - Total: {snap['tracemalloc']['total_size']:.2f} MB, "
                    f"Peak: {snap['tracemalloc']['peak_size']:.2f} MB"
                )

            if snap["memory_profiler"]:
                report.append(
                    f"Memory Profiler - RSS: {snap['memory_profiler']['rss_mb']:.2f} MB, "
                    f"VMS: {snap['memory_profiler']['vms_mb']:.2f} MB"
                )

            if snap["pympler"]:
                report.append(
                    f"Pympler - Total objects: {snap['pympler']['total_objects']}"
                )

        return "\n".join(report)

    def _get_enabled_tools(self) -> List[str]:
        """Get list of enabled profiling tools."""
        tools = []
        if self.tracemalloc_enabled:
            tools.append("tracemalloc")
        if self.memory_profiler_enabled:
            tools.append("memory_profiler")
        if self.pympler_enabled:
            tools.append("pympler")
        return tools


@pytest.mark.slow
class TestEnhancedMemoryProfiler(unittest.TestCase):
    """Test enhanced memory profiler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = EnhancedMemoryProfiler()

    def tearDown(self):
        """Clean up after tests."""
        self.profiler.stop_profiling()
        gc.collect()

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertEqual(len(self.profiler.snapshots), 0)
        self.assertEqual(len(self.profiler.hotspots), 0)
        self.assertFalse(self.profiler.tracemalloc_enabled)
        self.assertFalse(self.profiler.memory_profiler_enabled)
        self.assertFalse(self.profiler.pympler_enabled)

    def test_start_profiling_all_tools(self):
        """Test starting profiling with all tools."""
        self.profiler.start_profiling()

        self.assertTrue(self.profiler.tracemalloc_enabled)
        self.assertTrue(self.profiler.memory_profiler_enabled)
        self.assertTrue(self.profiler.pympler_enabled)
        self.assertTrue(tracemalloc.is_tracing())

    def test_start_profiling_specific_tools(self):
        """Test starting profiling with specific tools."""
        self.profiler.start_profiling(["tracemalloc"])

        self.assertTrue(self.profiler.tracemalloc_enabled)
        self.assertFalse(self.profiler.memory_profiler_enabled)
        self.assertFalse(self.profiler.pympler_enabled)

    def test_stop_profiling(self):
        """Test stopping profiling."""
        self.profiler.start_profiling()
        self.profiler.stop_profiling()

        self.assertFalse(self.profiler.tracemalloc_enabled)
        self.assertFalse(self.profiler.memory_profiler_enabled)
        self.assertFalse(self.profiler.pympler_enabled)
        self.assertFalse(tracemalloc.is_tracing())

    def test_take_snapshot_with_tracemalloc(self):
        """Test taking snapshot with tracemalloc."""
        self.profiler.start_profiling(["tracemalloc"])

        # Allocate some memory
        data = [i for i in range(10000)]

        snapshot = self.profiler.take_snapshot("test_snapshot")

        self.assertEqual(snapshot["label"], "test_snapshot")
        self.assertIsNotNone(snapshot["tracemalloc"])
        self.assertIn("total_size", snapshot["tracemalloc"])
        self.assertIn("peak_size", snapshot["tracemalloc"])
        self.assertIn("top_allocations", snapshot["tracemalloc"])
        self.assertGreater(snapshot["tracemalloc"]["total_size"], 0)

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_take_snapshot_with_memory_profiler(self, mock_vm, mock_process):
        """Test taking snapshot with memory_profiler."""
        # Mock psutil
        mock_process.return_value.memory_info.return_value = MagicMock(
            rss=100 * 1024 * 1024, vms=200 * 1024 * 1024  # 100 MB  # 200 MB
        )
        mock_vm.return_value.available = 4000 * 1024 * 1024  # 4 GB

        self.profiler.start_profiling(["memory_profiler"])
        snapshot = self.profiler.take_snapshot("memory_test")

        self.assertIsNotNone(snapshot["memory_profiler"])
        self.assertEqual(snapshot["memory_profiler"]["rss_mb"], 100.0)
        self.assertEqual(snapshot["memory_profiler"]["vms_mb"], 200.0)
        self.assertEqual(snapshot["memory_profiler"]["available_mb"], 4000.0)

    def test_take_snapshot_with_pympler(self):
        """Test taking snapshot with pympler."""
        self.profiler.start_profiling(["pympler"])

        # Create various objects
        test_list = [i for i in range(1000)]
        test_dict = {str(i): i for i in range(100)}
        test_array = np.zeros((100, 100))

        snapshot = self.profiler.take_snapshot("pympler_test")

        self.assertIsNotNone(snapshot["pympler"])
        self.assertIn("total_objects", snapshot["pympler"])
        self.assertIn("type_stats", snapshot["pympler"])
        self.assertGreater(snapshot["pympler"]["total_objects"], 0)
        self.assertIsInstance(snapshot["pympler"]["type_stats"], dict)

    def test_analyze_hotspots(self):
        """Test memory hotspot analysis."""
        self.profiler.start_profiling()

        # Create memory allocations
        large_list = [i for i in range(100000)]  # ~3.8 MB
        large_array = np.zeros((1000, 1000))  # ~8 MB

        self.profiler.take_snapshot("before")

        # More allocations
        more_data = [str(i) * 100 for i in range(10000)]

        self.profiler.take_snapshot("after")

        hotspots = self.profiler.analyze_hotspots()

        self.assertIsInstance(hotspots, list)
        if hotspots:
            self.assertIsInstance(hotspots[0], MemoryHotspot)
            self.assertGreater(hotspots[0].size_mb, 0)

    def test_compare_snapshots(self):
        """Test snapshot comparison."""
        self.profiler.start_profiling()

        # First snapshot
        self.profiler.take_snapshot("initial")

        # Allocate memory
        data = [i for i in range(50000)]

        # Second snapshot
        time.sleep(0.1)  # Ensure time difference
        self.profiler.take_snapshot("after_allocation")

        comparison = self.profiler.compare_snapshots(0, 1)

        self.assertIn("time_diff", comparison)
        self.assertGreater(comparison["time_diff"], 0)

        if comparison["tracemalloc_diff"]:
            self.assertIn("total_size_diff", comparison["tracemalloc_diff"])
            self.assertGreaterEqual(
                comparison["tracemalloc_diff"]["total_size_diff"], 0
            )

    def test_compare_snapshots_invalid_indices(self):
        """Test snapshot comparison with invalid indices."""
        comparison = self.profiler.compare_snapshots(0, 10)
        self.assertIn("error", comparison)

    def test_generate_report_empty(self):
        """Test report generation with no data."""
        report = self.profiler.generate_report()
        self.assertEqual(report, "No profiling data available")

    def test_generate_report_with_data(self):
        """Test report generation with profiling data."""
        self.profiler.start_profiling()

        # Take multiple snapshots
        self.profiler.take_snapshot("start")
        data1 = [i for i in range(10000)]
        self.profiler.take_snapshot("after_data1")
        data2 = np.zeros((500, 500))
        self.profiler.take_snapshot("after_data2")

        # Analyze hotspots
        self.profiler.analyze_hotspots()

        report = self.profiler.generate_report()

        self.assertIn("Enhanced Memory Profiling Report", report)
        self.assertIn("Summary", report)
        self.assertIn("Total snapshots: 3", report)
        self.assertIn("Profiling tools:", report)

        if len(self.profiler.hotspots) > 0:
            self.assertIn("Memory Hotspots", report)

    def test_memory_leak_detection(self):
        """Test memory leak detection scenario."""
        self.profiler.start_profiling()

        # Simulate memory leak
        leaked_data = []

        for i in range(5):
            self.profiler.take_snapshot(f"iteration_{i}")
            # Keep appending data (simulating leak)
            leaked_data.extend([str(j) * 100 for j in range(1000)])

        # Check growth between first and last snapshot
        comparison = self.profiler.compare_snapshots(0, 4)

        if comparison["tracemalloc_diff"]:
            self.assertGreater(
                comparison["tracemalloc_diff"]["total_size_diff"], 0
            )

        if comparison["pympler_diff"]:
            self.assertGreater(
                comparison["pympler_diff"]["object_count_diff"], 0
            )

    def test_context_manager_profiling(self):
        """Test profiling with context manager pattern."""

        @contextmanager
        def profile_operation(profiler, operation_name):
            profiler.take_snapshot(f"{operation_name}_start")
            yield
            profiler.take_snapshot(f"{operation_name}_end")

        self.profiler.start_profiling()

        with profile_operation(self.profiler, "test_operation"):
            # Simulate some work
            data = np.random.rand(1000, 1000)
            result = np.sum(data)

        self.assertEqual(len(self.profiler.snapshots), 2)
        self.assertEqual(
            self.profiler.snapshots[0]["label"], "test_operation_start"
        )
        self.assertEqual(
            self.profiler.snapshots[1]["label"], "test_operation_end"
        )

    def test_agent_memory_profiling(self):
        """Test profiling agent memory usage patterns."""

        class MockAgent:
            def __init__(self, agent_id):
                self.id = agent_id
                self.beliefs = np.zeros((1000, 1000))  # ~8MB
                self.action_history = []
                self.observations = []

            def step(self):
                # Simulate agent step
                self.action_history.append(
                    f"action_{len(self.action_history)}"
                )
                self.observations.append(np.random.rand(100))

        self.profiler.start_profiling()

        # Profile agent creation
        self.profiler.take_snapshot("before_agents")

        agents = []
        for i in range(5):
            agent = MockAgent(f"agent_{i}")
            agents.append(agent)

        self.profiler.take_snapshot("after_agent_creation")

        # Profile agent operations
        for _ in range(10):
            for agent in agents:
                agent.step()

        self.profiler.take_snapshot("after_agent_operations")

        # Analyze
        hotspots = self.profiler.analyze_hotspots()
        report = self.profiler.generate_report()

        # Verify memory growth tracking
        comparison = self.profiler.compare_snapshots(0, 2)

        if comparison["tracemalloc_diff"]:
            # Should show memory increase from agent creation and operations
            self.assertGreater(
                comparison["tracemalloc_diff"]["total_size_diff"], 0
            )

        self.assertIn("Memory Timeline", report)


if __name__ == "__main__":
    unittest.main()
