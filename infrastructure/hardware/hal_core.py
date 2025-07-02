"""
Core Hardware Abstraction Layer Architecture

Defines the fundamental interfaces and architectural components for hardware abstraction.
"""

import abc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Types of hardware components"""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    SENSOR = "sensor"
    ACCELERATOR = "accelerator"
    CUSTOM = "custom"


class OperationStatus(Enum):
    """Status of hardware operations"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_AVAILABLE = "not_available"
    BUSY = "busy"
    DEGRADED = "degraded"


@dataclass
class DeviceCapabilities:
    """Represents the capabilities of a hardware device"""

    device_id: str
    device_type: HardwareType
    model_name: str
    vendor: str

    # Performance characteristics
    compute_units: int = 0
    clock_speed_mhz: int = 0
    memory_size_mb: int = 0
    bandwidth_gbps: float = 0.0

    # Feature support
    features: Dict[str, bool] = field(default_factory=dict)
    supported_operations: List[str] = field(default_factory=list)
    supported_data_types: List[str] = field(default_factory=list)

    # Constraints
    max_power_watts: float = 0.0
    thermal_limit_celsius: float = 0.0

    # Current status
    is_available: bool = True
    health_status: str = "healthy"

    def supports_operation(self, operation: str) -> bool:
        """Check if device supports a specific operation"""
        return operation in self.supported_operations

    def has_feature(self, feature: str) -> bool:
        """Check if device has a specific feature"""
        return self.features.get(feature, False)


@dataclass
class ResourceConstraints:
    """Defines resource constraints for operations"""

    max_memory_mb: int
    max_compute_percent: float = 100.0
    max_bandwidth_mbps: float = float("inf")
    max_power_watts: float = float("inf")
    timeout_seconds: float = 60.0
    priority: int = 5  # 1-10, higher is more important

    def is_within_limits(self, usage: Dict[str, float]) -> bool:
        """Check if resource usage is within constraints"""
        if usage.get("memory_mb", 0) > self.max_memory_mb:
            return False
        if usage.get("compute_percent", 0) > self.max_compute_percent:
            return False
        if usage.get("bandwidth_mbps", 0) > self.max_bandwidth_mbps:
            return False
        if usage.get("power_watts", 0) > self.max_power_watts:
            return False
        return True


@runtime_checkable
class HardwareInterface(Protocol):
    """Protocol defining the interface for hardware operations"""

    def initialize(self) -> bool:
        """Initialize the hardware device"""
        ...

    def shutdown(self) -> bool:
        """Shutdown the hardware device"""
        ...

    def execute_operation(
        self,
        operation: str,
        data: Any,
        constraints: Optional[ResourceConstraints] = None,
    ) -> Tuple[OperationStatus, Any]:
        """Execute an operation on the hardware"""
        ...

    def get_capabilities(self) -> DeviceCapabilities:
        """Get device capabilities"""
        ...

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        ...

    def health_check(self) -> Tuple[bool, str]:
        """Perform health check on the device"""
        ...


class BaseHardwareAdapter(abc.ABC):
    """Base class for hardware adapters implementing the HardwareInterface"""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._initialized = False
        self._lock = threading.Lock()
        self._resource_usage = {
            "memory_mb": 0,
            "compute_percent": 0,
            "bandwidth_mbps": 0,
            "power_watts": 0,
        }

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the hardware device"""
        pass

    @abc.abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the hardware device"""
        pass

    @abc.abstractmethod
    def execute_operation(
        self,
        operation: str,
        data: Any,
        constraints: Optional[ResourceConstraints] = None,
    ) -> Tuple[OperationStatus, Any]:
        """Execute an operation on the hardware"""
        pass

    @abc.abstractmethod
    def get_capabilities(self) -> DeviceCapabilities:
        """Get device capabilities"""
        pass

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        with self._lock:
            return self._resource_usage.copy()

    def health_check(self) -> Tuple[bool, str]:
        """Perform health check on the device"""
        if not self._initialized:
            return False, "Device not initialized"

        # Basic health check - can be overridden
        try:
            usage = self.get_resource_usage()
            if usage.get("compute_percent", 0) > 95:
                return False, "Device overloaded"
            if usage.get(
                "memory_mb",
                    0) > self.get_capabilities().memory_size_mb * 0.95:
                return False, "Memory nearly exhausted"
            return True, "Healthy"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    def _update_resource_usage(self, **kwargs):
        """Update resource usage metrics"""
        with self._lock:
            self._resource_usage.update(kwargs)


class HardwareAbstractionLayer:
    """
    Main HAL class that manages hardware devices and provides unified interface.

    This class implements the architectural pattern where application code
    interacts with hardware through a consistent API regardless of the
    underlying hardware implementation.
    """

    def __init__(self) -> None:
        """Initialize the Hardware Abstraction Layer"""
        self._devices: Dict[str, HardwareInterface] = {}
        self._device_registry: Dict[HardwareType, List[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.RLock()

        # Operation routing
        self._operation_routes: Dict[str, List[HardwareType]] = {
            "compute": [HardwareType.GPU, HardwareType.TPU, HardwareType.CPU],
            "store": [HardwareType.STORAGE, HardwareType.MEMORY],
            "network": [HardwareType.NETWORK],
            "sense": [HardwareType.SENSOR],
        }

        logger.info("Hardware Abstraction Layer initialized")

    def register_device(
        self, device_id: str, device: HardwareInterface, device_type: HardwareType
    ) -> bool:
        """Register a hardware device with the HAL"""
        with self._lock:
            if device_id in self._devices:
                logger.warning(f"Device {device_id} already registered")
                return False

            # Initialize the device
            if not device.initialize():
                logger.error(f"Failed to initialize device {device_id}")
                return False

            # Register device
            self._devices[device_id] = device

            # Update registry
            if device_type not in self._device_registry:
                self._device_registry[device_type] = []
            self._device_registry[device_type].append(device_id)

            logger.info(f"Registered device {device_id} of type {device_type}")
            return True

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a hardware device"""
        with self._lock:
            if device_id not in self._devices:
                logger.warning(f"Device {device_id} not found")
                return False

            device = self._devices[device_id]

            # Shutdown device
            device.shutdown()

            # Remove from registry
            for device_type, devices in self._device_registry.items():
                if device_id in devices:
                    devices.remove(device_id)

            # Remove device
            del self._devices[device_id]

            logger.info(f"Unregistered device {device_id}")
            return True

    def get_device(self, device_id: str) -> Optional[HardwareInterface]:
        """Get a specific device by ID"""
        return self._devices.get(device_id)

    def get_devices_by_type(self, device_type: HardwareType) -> List[HardwareInterface]:
        """Get all devices of a specific type"""
        device_ids = self._device_registry.get(device_type, [])
        return [self._devices[did] for did in device_ids if did in self._devices]

    def execute_operation(
        self,
        operation: str,
        data: Any,
        device_id: Optional[str] = None,
        constraints: Optional[ResourceConstraints] = None,
    ) -> Tuple[OperationStatus, Any]:
        """
        Execute an operation on hardware.

        If device_id is specified, use that device. Otherwise, automatically
        select the best available device based on operation type and
            constraints.
        """
        if device_id:
            # Use specific device
            device = self.get_device(device_id)
            if not device:
                return OperationStatus.NOT_AVAILABLE, f"Device {device_id} not found"

            return self._execute_on_device(device, operation, data, constraints)

        # Auto-select device
        device = self._select_best_device(operation, constraints)
        if not device:
            return OperationStatus.NOT_AVAILABLE, "No suitable device available"

        return self._execute_on_device(device, operation, data, constraints)

    def _select_best_device(
        self, operation: str, constraints: Optional[ResourceConstraints]
    ) -> Optional[HardwareInterface]:
        """Select the best device for an operation based on availability and
        constraints"""
        # Determine device types that can handle the operation
        operation_category = self._categorize_operation(operation)
        candidate_types = self._operation_routes.get(
            operation_category, [HardwareType.CPU])

        best_device = None
        best_score = -1

        for device_type in candidate_types:
            devices = self.get_devices_by_type(device_type)

            for device in devices:
                # Check if device supports operation
                caps = device.get_capabilities()
                if not caps.supports_operation(operation):
                    continue

                # Check health
                healthy, _ = device.health_check()
                if not healthy:
                    continue

                # Check resource constraints
                if constraints:
                    usage = device.get_resource_usage()
                    if not constraints.is_within_limits(usage):
                        continue

                # Calculate device score
                score = self._calculate_device_score(device, operation, constraints)

                if score > best_score:
                    best_score = score
                    best_device = device

        return best_device

    def _calculate_device_score(
        self,
        device: HardwareInterface,
        operation: str,
        constraints: Optional[ResourceConstraints],
    ) -> float:
        """Calculate a score for how suitable a device is for an operation"""
        score = 100.0

        caps = device.get_capabilities()
        usage = device.get_resource_usage()

        # Penalize based on current usage
        score -= usage.get("compute_percent", 0) * 0.5
        score -= (usage.get("memory_mb", 0) / caps.memory_size_mb * 100) * 0.3

        # Bonus for specialized hardware
        if caps.device_type in [HardwareType.GPU, HardwareType.TPU]:
            if "compute" in operation.lower():
                score += 20

        # Consider priority if constraints given
        if constraints:
            score += constraints.priority * 2

        return max(0, score)

    def _execute_on_device(
        self,
        device: HardwareInterface,
        operation: str,
        data: Any,
        constraints: Optional[ResourceConstraints],
    ) -> Tuple[OperationStatus, Any]:
        """Execute operation on a specific device"""
        try:
            # Submit to executor for async execution
            future = self._executor.submit(
                device.execute_operation, operation, data, constraints)

            # Wait with timeout
            timeout = constraints.timeout_seconds if constraints else 60.0
            status, result = future.result(timeout=timeout)

            return status, result

        except TimeoutError:
            logger.error(f"Operation {operation} timed out")
            return OperationStatus.TIMEOUT, "Operation timed out"
        except Exception as e:
            logger.error(f"Operation {operation} failed: {str(e)}")
            return OperationStatus.FAILED, str(e)

    def _categorize_operation(self, operation: str) -> str:
        """Categorize an operation for routing"""
        operation_lower = operation.lower()

        if any(
            term in operation_lower for term in [
                "compute",
                "calculate",
                "process",
                "infer"]):
            return "compute"
        elif any(term in operation_lower for term in ["store", "save", "write", "cache"]):
            return "store"
        elif any(term in operation_lower for term in ["network", "send", "receive", "transfer"]):
            return "network"
        elif any(term in operation_lower for term in ["sense", "measure", "detect"]):
            return "sense"
        else:
            return "compute"  # Default to compute

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status including all devices"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "devices": {},
            "summary": {
                "total_devices": len(self._devices),
                "healthy_devices": 0,
                "by_type": {},
            },
        }

        for device_id, device in self._devices.items():
            caps = device.get_capabilities()
            healthy, health_msg = device.health_check()
            usage = device.get_resource_usage()

            status["devices"][device_id] = {
                "type": caps.device_type.value,
                "model": caps.model_name,
                "healthy": healthy,
                "health_message": health_msg,
                "usage": usage,
                "capabilities": {
                    "compute_units": caps.compute_units,
                    "memory_mb": caps.memory_size_mb,
                    "features": caps.features,
                },
            }

            if healthy:
                status["summary"]["healthy_devices"] += 1

            # Count by type
            device_type = caps.device_type.value
            if device_type not in status["summary"]["by_type"]:
                status["summary"]["by_type"][device_type] = 0
            status["summary"]["by_type"][device_type] += 1

        return status

    def shutdown(self):
        """Shutdown the HAL and all registered devices"""
        logger.info("Shutting down Hardware Abstraction Layer")

        # Shutdown all devices
        with self._lock:
            for device_id in list(self._devices.keys()):
                self.unregister_device(device_id)

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("Hardware Abstraction Layer shutdown complete")
