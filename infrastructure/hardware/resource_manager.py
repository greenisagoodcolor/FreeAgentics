"""
Resource Management for Hardware Infrastructure
Manages allocation and monitoring of system resources
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import psutil


class ResourcePriority(Enum):
    """Priority levels for resource allocation"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceAllocation:
    """Resource allocation configuration"""

    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_memory_mb: Optional[int] = None
    disk_space_mb: Optional[int] = None
    priority: ResourcePriority = ResourcePriority.MEDIUM
    max_duration_seconds: Optional[int] = None


class ResourceMonitor:
    """Monitor system resource usage"""

    def __init__(self):
        self._monitoring = False
        self._monitor_thread = None
        self._stats = {}

    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def get_current_usage(self) -> Dict[str, Union[float, int]]:
        """Get current resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
            "disk_percent": psutil.disk_usage("/").percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0,
        }

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._stats = self.get_current_usage()
                time.sleep(1.0)
            except Exception:
                pass  # Continue monitoring even if individual readings fail


class ResourceManager:
    """Manage system resource allocation and monitoring"""

    def __init__(self):
        self.monitor = ResourceMonitor()
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._active_processes: Dict[str, int] = {}

    def allocate_resources(self, task_id: str, allocation: ResourceAllocation) -> bool:
        """Allocate resources for a task"""
        try:
            current_usage = self.monitor.get_current_usage()

            # Check if allocation is feasible
            if allocation.memory_mb:
                available_mb = current_usage["memory_available_mb"]
                if allocation.memory_mb > available_mb * 0.8:  # Leave 20% buffer
                    return False

            if allocation.cpu_cores:
                if allocation.cpu_cores > psutil.cpu_count():
                    return False

            # Store allocation
            self._allocations[task_id] = allocation
            return True

        except Exception:
            return False

    def release_resources(self, task_id: str) -> None:
        """Release resources for a task"""
        self._allocations.pop(task_id, None)
        self._active_processes.pop(task_id, None)

    def get_available_resources(self) -> ResourceAllocation:
        """Get currently available resources"""
        usage = self.monitor.get_current_usage()

        # Calculate available resources
        total_memory = psutil.virtual_memory().total // (1024 * 1024)
        available_memory = int(total_memory * (100 - usage["memory_percent"]) / 100)

        return ResourceAllocation(
            cpu_cores=psutil.cpu_count(),
            memory_mb=available_memory,
            priority=ResourcePriority.MEDIUM,
        )

    def start(self) -> None:
        """Start resource management"""
        self.monitor.start_monitoring()

    def stop(self) -> None:
        """Stop resource management"""
        self.monitor.stop_monitoring()

    def get_allocation_status(self) -> Dict[str, Dict]:
        """Get status of all current allocations"""
        return {
            task_id: {
                "allocation": allocation,
                "status": "active" if task_id in self._active_processes else "allocated",
            }
            for task_id, allocation in self._allocations.items()
        }
