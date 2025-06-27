."""
Hardware Abstraction Layer (HAL) for FreeAgentics

This module provides a unified interface for hardware operations across different
platforms and devices, enabling deployment on edge devices with resource constraints.
"""

from .device_discovery import (
    DeviceDiscovery,
    DeviceInfo,
    DeviceStatus,
    DeviceType)
from .hal_core import (
    DeviceCapabilities,
    HardwareAbstractionLayer,
    HardwareInterface,
    ResourceConstraints,
)
from .offline_capabilities import (
    OfflineManager,
    StatePersistence,
    SyncManager,
    WorkQueue)
from .resource_manager import (
    ResourceAllocation,
    ResourceManager,
    ResourceMonitor,
    ResourcePriority)

__all__ = [
    # Core HAL
    "HardwareAbstractionLayer",
    "DeviceCapabilities",
    "ResourceConstraints",
    "HardwareInterface",
    # Device Discovery
    "DeviceDiscovery",
    "DeviceInfo",
    "DeviceType",
    "DeviceStatus",
    # Resource Management
    "ResourceManager",
    "ResourceAllocation",
    "ResourcePriority",
    "ResourceMonitor",
    # Offline Operations
    "OfflineManager",
    "StatePersistence",
    "WorkQueue",
    "SyncManager",
]

# Module version
__version__ = "1.0.0"
