"""
Comprehensive tests for Resource Management module
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import psutil
import pytest

from infrastructure.hardware.resource_manager import (
    ResourceAllocation,
    ResourceManager,
    ResourceMonitor,
    ResourcePriority,
)


class TestResourcePriority:
    """Test ResourcePriority enum"""

    def test_priority_values(self):
        """Test priority enum values"""
        assert ResourcePriority.LOW.value == 1
        assert ResourcePriority.MEDIUM.value == 2
        assert ResourcePriority.HIGH.value == 3
        assert ResourcePriority.CRITICAL.value == 4

    def test_priority_ordering(self):
        """Test priority ordering"""
        assert ResourcePriority.LOW.value < ResourcePriority.MEDIUM.value
        assert ResourcePriority.MEDIUM.value < ResourcePriority.HIGH.value
        assert ResourcePriority.HIGH.value < ResourcePriority.CRITICAL.value


class TestResourceAllocation:
    """Test ResourceAllocation dataclass"""

    def test_default_allocation(self):
        """Test default resource allocation"""
        allocation = ResourceAllocation()
        assert allocation.cpu_cores is None
        assert allocation.memory_mb is None
        assert allocation.gpu_memory_mb is None
        assert allocation.disk_space_mb is None
        assert allocation.priority == ResourcePriority.MEDIUM
        assert allocation.max_duration_seconds is None

    def test_custom_allocation(self):
        """Test custom resource allocation"""
        allocation = ResourceAllocation(
            cpu_cores=4,
            memory_mb=8192,
            gpu_memory_mb=4096,
            disk_space_mb=10240,
            priority=ResourcePriority.HIGH,
            max_duration_seconds=3600,
        )
        assert allocation.cpu_cores == 4
        assert allocation.memory_mb == 8192
        assert allocation.gpu_memory_mb == 4096
        assert allocation.disk_space_mb == 10240
        assert allocation.priority == ResourcePriority.HIGH
        assert allocation.max_duration_seconds == 3600

    def test_partial_allocation(self):
        """Test partial resource allocation"""
        allocation = ResourceAllocation(cpu_cores=2, memory_mb=4096)
        assert allocation.cpu_cores == 2
        assert allocation.memory_mb == 4096
        assert allocation.gpu_memory_mb is None
        assert allocation.disk_space_mb is None


class TestResourceMonitor:
    """Test ResourceMonitor class"""

    def test_initialization(self):
        """Test monitor initialization"""
        monitor = ResourceMonitor()
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None
        assert monitor._stats == {}

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.getloadavg")
    def test_get_current_usage(self, mock_loadavg, mock_disk, mock_memory, mock_cpu):
        """Test getting current resource usage"""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(percent=60.0, available=4 * 1024 * 1024 * 1024)  # 4GB
        mock_disk.return_value = Mock(percent=75.0)
        mock_loadavg.return_value = (1.5, 2.0, 1.8)

        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()

        assert usage["cpu_percent"] == 45.5
        assert usage["memory_percent"] == 60.0
        assert usage["memory_available_mb"] == 4096
        assert usage["disk_percent"] == 75.0
        assert usage["load_average"] == 1.5

    @patch("psutil.disk_usage")
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    def test_get_current_usage_no_loadavg(self, mock_cpu, mock_memory, mock_disk):
        """Test getting usage on systems without getloadavg"""
        mock_cpu.return_value = 30.0
        mock_memory.return_value = Mock(percent=40.0, available=8 * 1024 * 1024 * 1024)
        mock_disk.return_value = Mock(percent=50.0)

        # Temporarily remove getloadavg to simulate systems without it
        original_getloadavg = getattr(psutil, "getloadavg", None)
        if hasattr(psutil, "getloadavg"):
            delattr(psutil, "getloadavg")

        try:
            monitor = ResourceMonitor()
            usage = monitor.get_current_usage()
            assert usage["load_average"] == 0.0  # Fallback value
        finally:
            # Restore original attribute
            if original_getloadavg is not None:
                setattr(psutil, "getloadavg", original_getloadavg)

    def test_start_monitoring(self):
        """Test starting monitoring"""
        monitor = ResourceMonitor()
        monitor.start_monitoring()

        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.daemon is True
        assert monitor._monitor_thread.is_alive()

        # Stop monitoring for cleanup
        monitor.stop_monitoring()

    def test_start_monitoring_idempotent(self):
        """Test that start_monitoring is idempotent"""
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        thread1 = monitor._monitor_thread

        # Starting again should not create new thread
        monitor.start_monitoring()
        thread2 = monitor._monitor_thread

        assert thread1 is thread2
        monitor.stop_monitoring()

    def test_stop_monitoring(self):
        """Test stopping monitoring"""
        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Give thread time to start
        time.sleep(0.1)

        monitor.stop_monitoring()
        assert monitor._monitoring is False

        # Thread should stop within timeout
        time.sleep(1.2)
        assert not monitor._monitor_thread.is_alive()

    def test_stop_monitoring_no_thread(self):
        """Test stopping when not monitoring"""
        monitor = ResourceMonitor()
        monitor.stop_monitoring()  # Should not raise
        assert monitor._monitoring is False

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_monitor_loop(self, mock_disk, mock_memory, mock_cpu):
        """Test monitoring loop updates stats"""
        mock_cpu.return_value = 25.0
        mock_memory.return_value = Mock(percent=35.0, available=16 * 1024 * 1024 * 1024)
        mock_disk.return_value = Mock(percent=45.0)

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Wait for at least one update
        time.sleep(1.5)

        # Stats should be updated
        assert monitor._stats != {}
        assert "cpu_percent" in monitor._stats

        monitor.stop_monitoring()

    @patch("infrastructure.hardware.resource_manager.ResourceMonitor.get_current_usage")
    def test_monitor_loop_exception_handling(self, mock_usage):
        """Test monitor loop handles exceptions"""
        mock_usage.side_effect = Exception("Test error")

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Should continue running despite exception
        time.sleep(0.5)
        assert monitor._monitoring is True

        monitor.stop_monitoring()


class TestResourceManager:
    """Test ResourceManager class"""

    def test_initialization(self):
        """Test manager initialization"""
        manager = ResourceManager()
        assert isinstance(manager.monitor, ResourceMonitor)
        assert manager._allocations == {}
        assert manager._active_processes == {}

    @patch("psutil.cpu_count")
    @patch.object(ResourceMonitor, "get_current_usage")
    def test_allocate_resources_success(self, mock_usage, mock_cpu_count):
        """Test successful resource allocation"""
        mock_cpu_count.return_value = 8
        mock_usage.return_value = {"memory_available_mb": 16384}  # 16GB available

        manager = ResourceManager()
        allocation = ResourceAllocation(cpu_cores=4, memory_mb=8192)  # Request 8GB

        result = manager.allocate_resources("task1", allocation)
        assert result is True
        assert "task1" in manager._allocations
        assert manager._allocations["task1"] == allocation

    @patch("psutil.cpu_count")
    @patch.object(ResourceMonitor, "get_current_usage")
    def test_allocate_resources_insufficient_memory(self, mock_usage, mock_cpu_count):
        """Test allocation failure due to insufficient memory"""
        mock_cpu_count.return_value = 8
        mock_usage.return_value = {"memory_available_mb": 1024}  # Only 1GB available

        manager = ResourceManager()
        allocation = ResourceAllocation(memory_mb=2048)  # Request 2GB (more than 80% of available)

        result = manager.allocate_resources("task1", allocation)
        assert result is False
        assert "task1" not in manager._allocations

    @patch("psutil.cpu_count")
    @patch.object(ResourceMonitor, "get_current_usage")
    def test_allocate_resources_too_many_cores(self, mock_usage, mock_cpu_count):
        """Test allocation failure due to too many CPU cores"""
        mock_cpu_count.return_value = 4
        mock_usage.return_value = {"memory_available_mb": 16384}

        manager = ResourceManager()
        allocation = ResourceAllocation(cpu_cores=8)  # More than available

        result = manager.allocate_resources("task1", allocation)
        assert result is False
        assert "task1" not in manager._allocations

    @patch.object(ResourceMonitor, "get_current_usage")
    def test_allocate_resources_no_constraints(self, mock_usage):
        """Test allocation with no specific constraints"""
        mock_usage.return_value = {"memory_available_mb": 8192}

        manager = ResourceManager()
        allocation = ResourceAllocation()  # No specific requirements

        result = manager.allocate_resources("task1", allocation)
        assert result is True
        assert "task1" in manager._allocations

    @patch.object(ResourceMonitor, "get_current_usage")
    def test_allocate_resources_exception_handling(self, mock_usage):
        """Test allocation handles exceptions gracefully"""
        mock_usage.side_effect = Exception("Test error")

        manager = ResourceManager()
        allocation = ResourceAllocation(memory_mb=1024)

        result = manager.allocate_resources("task1", allocation)
        assert result is False
        assert "task1" not in manager._allocations

    def test_release_resources(self):
        """Test releasing resources"""
        manager = ResourceManager()
        manager._allocations["task1"] = ResourceAllocation(cpu_cores=2)
        manager._active_processes["task1"] = 12345

        manager.release_resources("task1")

        assert "task1" not in manager._allocations
        assert "task1" not in manager._active_processes

    def test_release_resources_not_allocated(self):
        """Test releasing non-existent resources"""
        manager = ResourceManager()
        # Should not raise error
        manager.release_resources("nonexistent")
        assert len(manager._allocations) == 0

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch.object(ResourceMonitor, "get_current_usage")
    def test_get_available_resources(self, mock_usage, mock_memory, mock_cpu_count):
        """Test getting available resources"""
        mock_cpu_count.return_value = 16
        mock_memory.return_value = Mock(total=32 * 1024 * 1024 * 1024)  # 32GB total
        mock_usage.return_value = {"memory_percent": 25.0}  # 75% available

        manager = ResourceManager()
        available = manager.get_available_resources()

        assert available.cpu_cores == 16
        assert available.memory_mb == 24576  # 75% of 32GB
        assert available.priority == ResourcePriority.MEDIUM

    def test_start_stop(self):
        """Test starting and stopping manager"""
        manager = ResourceManager()

        # Initially monitoring should be off
        assert manager.monitor._monitoring is False

        manager.start()
        assert manager.monitor._monitoring is True

        manager.stop()
        assert manager.monitor._monitoring is False

    def test_get_allocation_status_empty(self):
        """Test getting status with no allocations"""
        manager = ResourceManager()
        status = manager.get_allocation_status()
        assert status == {}

    def test_get_allocation_status_with_allocations(self):
        """Test getting status with active allocations"""
        manager = ResourceManager()

        allocation1 = ResourceAllocation(cpu_cores=2, memory_mb=4096)
        allocation2 = ResourceAllocation(cpu_cores=4, priority=ResourcePriority.HIGH)

        manager._allocations["task1"] = allocation1
        manager._allocations["task2"] = allocation2
        manager._active_processes["task1"] = 12345

        status = manager.get_allocation_status()

        assert len(status) == 2
        assert status["task1"]["allocation"] == allocation1
        assert status["task1"]["status"] == "active"
        assert status["task2"]["allocation"] == allocation2
        assert status["task2"]["status"] == "allocated"

    def test_multiple_allocations(self):
        """Test managing multiple allocations"""
        manager = ResourceManager()

        # Allocate resources for multiple tasks
        alloc1 = ResourceAllocation(cpu_cores=2, memory_mb=2048)
        alloc2 = ResourceAllocation(cpu_cores=1, memory_mb=1024)
        alloc3 = ResourceAllocation(memory_mb=512, priority=ResourcePriority.LOW)

        with patch.object(ResourceMonitor, "get_current_usage") as mock_usage:
            mock_usage.return_value = {"memory_available_mb": 16384}

            assert manager.allocate_resources("task1", alloc1) is True
            assert manager.allocate_resources("task2", alloc2) is True
            assert manager.allocate_resources("task3", alloc3) is True

        # Check all are allocated
        assert len(manager._allocations) == 3

        # Release one
        manager.release_resources("task2")
        assert len(manager._allocations) == 2
        assert "task2" not in manager._allocations
        assert "task1" in manager._allocations
        assert "task3" in manager._allocations

    def test_priority_allocation(self):
        """Test allocation with different priorities"""
        manager = ResourceManager()

        high_priority = ResourceAllocation(
            cpu_cores=4, memory_mb=8192, priority=ResourcePriority.CRITICAL
        )

        low_priority = ResourceAllocation(
            cpu_cores=2, memory_mb=4096, priority=ResourcePriority.LOW
        )

        with patch.object(ResourceMonitor, "get_current_usage") as mock_usage:
            mock_usage.return_value = {"memory_available_mb": 16384}

            assert manager.allocate_resources("critical_task", high_priority) is True
            assert manager.allocate_resources("low_task", low_priority) is True

        status = manager.get_allocation_status()
        assert status["critical_task"]["allocation"].priority == ResourcePriority.CRITICAL
        assert status["low_task"]["allocation"].priority == ResourcePriority.LOW


class TestResourceManagerIntegration:
    """Integration tests for complete resource management workflow"""

    def test_full_resource_lifecycle(self):
        """Test complete resource allocation lifecycle"""
        manager = ResourceManager()
        manager.start()

        try:
            # Initial state
            assert len(manager._allocations) == 0
            available = manager.get_available_resources()
            assert available.cpu_cores > 0
            assert available.memory_mb > 0

            # Allocate resources
            allocation = ResourceAllocation(
                cpu_cores=2, memory_mb=1024, priority=ResourcePriority.HIGH
            )

            with patch.object(ResourceMonitor, "get_current_usage") as mock_usage:
                mock_usage.return_value = {"memory_available_mb": 8192}
                assert manager.allocate_resources("test_task", allocation) is True

            # Check allocation
            status = manager.get_allocation_status()
            assert len(status) == 1
            assert "test_task" in status

            # Update allocation
            manager._active_processes["test_task"] = 54321
            status = manager.get_allocation_status()
            assert status["test_task"]["status"] == "active"

            # Release resources
            manager.release_resources("test_task")
            assert len(manager.get_allocation_status()) == 0

        finally:
            manager.stop()

    def test_concurrent_allocations(self):
        """Test concurrent resource allocations"""
        manager = ResourceManager()
        results = []

        def allocate_task(task_id):
            allocation = ResourceAllocation(cpu_cores=1, memory_mb=512)
            with patch.object(ResourceMonitor, "get_current_usage") as mock_usage:
                mock_usage.return_value = {"memory_available_mb": 16384}
                result = manager.allocate_resources(task_id, allocation)
                results.append((task_id, result))

        # Create threads for concurrent allocation
        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_task, args=(f"concurrent_task_{i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert all(result[1] for result in results)
        assert len(manager._allocations) == 5

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_monitoring_during_allocation(self, mock_disk, mock_memory, mock_cpu):
        """Test that monitoring works during allocation"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(
            percent=40.0, available=8 * 1024 * 1024 * 1024, total=16 * 1024 * 1024 * 1024
        )
        mock_disk.return_value = Mock(percent=60.0)

        manager = ResourceManager()
        manager.start()

        try:
            # Let monitoring collect some data
            time.sleep(1.5)

            # Allocate resources while monitoring is active
            allocation = ResourceAllocation(memory_mb=2048)
            result = manager.allocate_resources("monitored_task", allocation)
            assert result is True

            # Check that monitoring stats are available
            current_usage = manager.monitor.get_current_usage()
            assert current_usage["cpu_percent"] == 50.0
            assert current_usage["memory_percent"] == 40.0

        finally:
            manager.stop()
