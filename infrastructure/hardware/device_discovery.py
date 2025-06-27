"""
Device Discovery Mechanism

Automatically detects and enumerates available hardware devices and
    peripherals.
"""

import json
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

try:
    import GPUtil

    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

try:

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of discoverable devices"""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    USB = "usb"
    SENSOR = "sensor"
    CAMERA = "camera"
    AUDIO = "audio"
    ACCELERATOR = "accelerator"


class DeviceStatus(Enum):
    """Status of discovered devices"""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class DeviceInfo:
    """Information about a discovered device"""

    device_id: str
    device_type: DeviceType
    name: str
    vendor: str = "Unknown"
    model: str = "Unknown"

    # Connection info
    bus_info: str = ""
    driver: str = ""
    driver_version: str = ""

    # Capabilities
    capabilities: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_seen: datetime = field(default_factory=datetime.now)

    # Performance metrics
    performance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "name": self.name,
            "vendor": self.vendor,
            "model": self.model,
            "bus_info": self.bus_info,
            "driver": self.driver,
            "driver_version": self.driver_version,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat(),
            "performance_score": self.performance_score,
        }


class DeviceDiscovery:
    """
    Main device discovery system that detects hardware across platforms.
    """

    def __init__(self) -> None:
        """Initialize device discovery system"""
        self._devices: Dict[str, DeviceInfo] = {}
        self._discovery_methods: Dict[DeviceType, List[Callable]] = {
            DeviceType.CPU: [self._discover_cpu],
            DeviceType.GPU: [
                self._discover_gpu_nvidia,
                self._discover_gpu_amd,
                self._discover_gpu_intel,
            ],
            DeviceType.MEMORY: [self._discover_memory],
            DeviceType.STORAGE: [self._discover_storage],
            DeviceType.NETWORK: [self._discover_network],
            DeviceType.USB: [self._discover_usb],
            DeviceType.TPU: [self._discover_tpu],
        }
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[DeviceInfo, str], None]] = []

        logger.info("Device discovery system initialized")

    def discover_all(self) -> Dict[str, DeviceInfo]:
        """Discover all available devices"""
        logger.info("Starting full device discovery")

        discovered_devices = {}

        for device_type, methods in self._discovery_methods.items():
            for method in methods:
                try:
                    devices = method()
                    for device in devices:
                        discovered_devices[device.device_id] = device
                        logger.debug(f"Discovered {device_type.value}: {device.name}")
                except Exception as e:
                    logger.error(f"Error in {method.__name__}: {str(e)}")

        # Update internal registry
        self._devices = discovered_devices

        logger.info(f"Discovery complete. Found {len(discovered_devices)} devices")
        return discovered_devices

    def discover_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Discover devices of a specific type"""
        devices = []

        methods = self._discovery_methods.get(device_type, [])
        for method in methods:
            try:
                devices.extend(method())
            except Exception as e:
                logger.error(f"Error discovering {device_type.value}: {str(e)}")

        return devices

    def _discover_cpu(self) -> List[DeviceInfo]:
        """Discover CPU information"""
        devices = []

        try:
            # Get CPU info
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                "architecture": platform.machine(),
            }

            # Platform-specific CPU detection
            cpu_model = "Unknown CPU"
            vendor = "Unknown"

            if platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_model = line.split(":")[1].strip()
                            elif "vendor_id" in line:
                                vendor = line.split(":")[1].strip()
                                break
                except Exception:
                    pass

            elif platform.system() == "Darwin":  # macOS
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        cpu_model = result.stdout.strip()
                        if "Intel" in cpu_model:
                            vendor = "Intel"
                        elif "Apple" in cpu_model:
                            vendor = "Apple"
                except Exception:
                    pass

            elif platform.system() == "Windows":
                try:
                    import wmi  # type: ignore[import-not-found]

                    c = wmi.WMI()
                    for processor in c.Win32_Processor():
                        cpu_model = processor.Name
                        vendor = processor.Manufacturer
                        break
                except Exception:
                    pass

            device = DeviceInfo(
                device_id="cpu_0",
                device_type=DeviceType.CPU,
                name=cpu_model,
                vendor=vendor,
                model=cpu_model,
                capabilities=cpu_info,
                status=DeviceStatus.AVAILABLE,
                performance_score=cpu_info["logical_cores"] * (cpu_info["max_frequency"] / 1000),
            )

            devices.append(device)

        except Exception as e:
            logger.error(f"CPU discovery error: {str(e)}")

        return devices

    def _discover_gpu_nvidia(self) -> List[DeviceInfo]:
        """Discover NVIDIA GPUs"""
        devices = []

        if platform.system() == "Windows":
            nvidia_smi = "nvidia-smi.exe"
        else:
            nvidia_smi = "nvidia-smi"

        try:
            # Use nvidia-smi to get GPU info
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            idx, name, memory, driver = parts[:4]

                            device = DeviceInfo(
                                device_id=f"gpu_nvidia_{idx}",
                                device_type=DeviceType.GPU,
                                name=name,
                                vendor="NVIDIA",
                                model=name,
                                driver="nvidia",
                                driver_version=driver,
                                capabilities={
                                    "memory_mb": int(memory),
                                    "cuda_capable": True,
                                },
                                status=DeviceStatus.AVAILABLE,
                                performance_score=int(memory)
                                / 1000,  # Simple score based on memory
                            )
                            devices.append(device)

        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except Exception as e:
            logger.debug(f"NVIDIA GPU discovery error: {str(e)}")

        # Alternative: Use GPUtil if available
        if GPU_UTIL_AVAILABLE and not devices:
            try:
                gpus = GPUtil.getGPUs()
                for gpu_idx, gpu in enumerate(gpus):
                    device = DeviceInfo(
                        device_id=f"gpu_nvidia_{gpu_idx}",
                        device_type=DeviceType.GPU,
                        name=gpu.name,
                        vendor="NVIDIA",
                        model=gpu.name,
                        capabilities={
                            "memory_mb": int(gpu.memoryTotal),
                            "cuda_capable": True,
                        },
                        status=DeviceStatus.AVAILABLE,
                        performance_score=gpu.memoryTotal / 1000,
                    )
                    devices.append(device)
            except Exception:
                pass

        return devices

    def _discover_gpu_amd(self) -> List[DeviceInfo]:
        """Discover AMD GPUs"""
        devices: List[DeviceInfo] = []

        if platform.system() != "Linux":
            return devices

        try:
            # Check for AMD GPUs using rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True
            )

            if result.returncode == 0:
                # Parse output to get GPU info
                # This is simplified - real parsing would be more complex
                device = DeviceInfo(
                    device_id="gpu_amd_0",
                    device_type=DeviceType.GPU,
                    name="AMD GPU",
                    vendor="AMD",
                    driver="amdgpu",
                    capabilities={"rocm_capable": True},
                    status=DeviceStatus.AVAILABLE,
                )
                devices.append(device)

        except FileNotFoundError:
            logger.debug("rocm-smi not found")
        except Exception as e:
            logger.debug(f"AMD GPU discovery error: {str(e)}")

        return devices

    def _discover_gpu_intel(self) -> List[DeviceInfo]:
        """Discover Intel integrated GPUs"""
        devices = []

        # Platform-specific Intel GPU detection
        if platform.system() == "Linux":
            try:
                # Check for Intel GPU in /sys
                intel_gpu_path = Path("/sys/class/drm/card0/device/vendor")
                if intel_gpu_path.exists():
                    with open(intel_gpu_path) as f:
                        vendor_id = f.read().strip()
                        if vendor_id == "0x8086":  # Intel vendor ID
                            device = DeviceInfo(
                                device_id="gpu_intel_0",
                                device_type=DeviceType.GPU,
                                name="Intel Integrated Graphics",
                                vendor="Intel",
                                driver="i915",
                                capabilities={"integrated": True},
                                status=DeviceStatus.AVAILABLE,
                            )
                            devices.append(device)
            except Exception:
                pass

        return devices

    def _discover_memory(self) -> List[DeviceInfo]:
        """Discover memory information"""
        devices = []

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Main memory
            device = DeviceInfo(
                device_id="memory_main",
                device_type=DeviceType.MEMORY,
                name="System Memory",
                vendor="System",
                capabilities={
                    "total_mb": mem.total // (1024 * 1024),
                    "available_mb": mem.available // (1024 * 1024),
                    "speed_mhz": self._get_memory_speed(),
                },
                status=DeviceStatus.AVAILABLE,
                performance_score=mem.total // (1024 * 1024 * 1000),  # GB as score
            )
            devices.append(device)

            # Swap memory
            if swap.total > 0:
                swap_device = DeviceInfo(
                    device_id="memory_swap",
                    device_type=DeviceType.MEMORY,
                    name="Swap Memory",
                    vendor="System",
                    capabilities={
                        "total_mb": swap.total // (1024 * 1024),
                        "available_mb": swap.free // (1024 * 1024),
                    },
                    status=DeviceStatus.AVAILABLE,
                )
                devices.append(swap_device)

        except Exception as e:
            logger.error(f"Memory discovery error: {str(e)}")

        return devices

    def _discover_storage(self) -> List[DeviceInfo]:
        """Discover storage devices"""
        devices = []

        try:
            partitions = psutil.disk_partitions()

            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)

                    device = DeviceInfo(
                        device_id=f"storage_{partition.device.replace('/', '_')}",
                        device_type=DeviceType.STORAGE,
                        name=partition.device,
                        capabilities={
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total_gb": usage.total // (1024**3),
                            "free_gb": usage.free // (1024**3),
                            "opts": partition.opts,
                        },
                        status=DeviceStatus.AVAILABLE,
                        performance_score=self._estimate_storage_performance(partition),
                    )
                    devices.append(device)

                except PermissionError:
                    continue

        except Exception as e:
            logger.error(f"Storage discovery error: {str(e)}")

        return devices

    def _discover_network(self) -> List[DeviceInfo]:
        """Discover network interfaces"""
        devices = []

        try:
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            for iface_name, addrs in interfaces.items():
                if iface_name in stats:
                    stat = stats[iface_name]

                    # Get primary address
                    primary_addr = None
                    for addr in addrs:
                        if addr.family == 2:  # AF_INET (IPv4)
                            primary_addr = addr.address
                            break

                    device = DeviceInfo(
                        device_id=f"network_{iface_name}",
                        device_type=DeviceType.NETWORK,
                        name=iface_name,
                        capabilities={
                            "speed_mbps": stat.speed,
                            "mtu": stat.mtu,
                            "is_up": stat.isup,
                            "address": primary_addr,
                        },
                        status=(DeviceStatus.AVAILABLE if stat.isup else DeviceStatus.OFFLINE),
                        performance_score=stat.speed / 1000 if stat.speed else 0,
                    )
                    devices.append(device)

        except Exception as e:
            logger.error(f"Network discovery error: {str(e)}")

        return devices

    def _discover_usb(self) -> List[DeviceInfo]:
        """Discover USB devices"""
        devices = []

        if platform.system() == "Linux":
            try:
                result = subprocess.run(["lsusb"], capture_output=True, text=True)

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line and "Hub" not in line:  # Skip USB hubs
                            # Parse lsusb output
                            parts = line.split(" ", 6)
                            if len(parts) >= 7:
                                bus = parts[1]
                                device_num = parts[3].rstrip(":")
                                vendor_device = parts[5]
                                name = parts[6] if len(parts) > 6 else "Unknown USB Device"

                                device = DeviceInfo(
                                    device_id=f"usb_{bus}_{device_num}",
                                    device_type=DeviceType.USB,
                                    name=name,
                                    bus_info=f"Bus {bus} Device {device_num}",
                                    capabilities={"vendor_device": vendor_device},
                                    status=DeviceStatus.AVAILABLE,
                                )
                                devices.append(device)

            except Exception as e:
                logger.debug(f"USB discovery error: {str(e)}")

        return devices

    def _discover_tpu(self) -> List[DeviceInfo]:
        """Discover TPU devices (Google Coral, etc)"""
        devices = []

        # Check for Coral USB Accelerator
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["lsusb", "-d", "1a6e:089a"],  # Google Coral USB ID
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0 and result.stdout:
                    device = DeviceInfo(
                        device_id="tpu_coral_usb",
                        device_type=DeviceType.TPU,
                        name="Google Coral USB Accelerator",
                        vendor="Google",
                        model="Coral USB Accelerator",
                        capabilities={
                            "ops_per_second": 4e12,  # 4 TOPS
                            "edge_tpu": True,
                        },
                        status=DeviceStatus.AVAILABLE,
                        performance_score=4000,  # 4 TOPS
                    )
                    devices.append(device)

            except Exception:
                pass

        # Check for PCIe TPU
        pcie_tpu_path = Path("/dev/apex_0")
        if pcie_tpu_path.exists():
            device = DeviceInfo(
                device_id="tpu_coral_pcie",
                device_type=DeviceType.TPU,
                name="Google Coral PCIe Accelerator",
                vendor="Google",
                model="Coral PCIe Accelerator",
                capabilities={"ops_per_second": 4e12, "edge_tpu": True, "pcie": True},
                status=DeviceStatus.AVAILABLE,
                performance_score=4000,
            )
            devices.append(device)

        return devices

    def _get_memory_speed(self) -> int:
        """Get memory speed in MHz"""
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["sudo", "dmidecode", "-t", "memory"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Speed:" in line and "MHz" in line:
                            speed = line.split(":")[1].strip().split()[0]
                            return int(speed)
            except Exception:
                pass

        return 0  # Unknown

    def _estimate_storage_performance(self, partition) -> float:
        """Estimate storage device performance score"""
        score = 100.0  # Base score

        # SSD detection heuristics
        if "nvme" in partition.device.lower():
            score = 1000.0  # NVMe SSD
        elif "ssd" in partition.device.lower():
            score = 500.0  # SATA SSD
        elif partition.device.startswith("/dev/sd"):
            score = 100.0  # HDD

        # Adjust based on filesystem
        if partition.fstype in ["ext4", "xfs", "btrfs"]:
            score *= 1.1
        elif partition.fstype in ["ntfs", "fat32"]:
            score *= 0.9

        return score

    def start_monitoring(self, interval: float = 5.0):
        """Start monitoring for device changes"""
        if self._monitoring:
            logger.warning("Device monitoring already active")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started device monitoring with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop device monitoring"""
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped device monitoring")

    def _monitor_loop(self, interval: float):
        """Monitor loop for device changes"""
        previous_devices = self._devices.copy()

        while self._monitoring:
            try:
                # Discover devices
                current_devices = self.discover_all()

                # Check for changes
                for device_id, device in current_devices.items():
                    if device_id not in previous_devices:
                        # New device
                        self._notify_callbacks(device, "added")
                    elif previous_devices[device_id].status != device.status:
                        # Status change
                        self._notify_callbacks(device, "status_changed")

                for device_id in previous_devices:
                    if device_id not in current_devices:
                        # Device removed
                        self._notify_callbacks(previous_devices[device_id], "removed")

                previous_devices = current_devices

            except Exception as e:
                logger.error(f"Monitor loop error: {str(e)}")

            time.sleep(interval)

    def register_callback(self, callback: Callable[[DeviceInfo, str], None]) -> None:
        """Register callback for device events"""
        self._callbacks.append(callback)

    def _notify_callbacks(self, device: DeviceInfo, event: str):
        """Notify registered callbacks of device events"""
        for callback in self._callbacks:
            try:
                callback(device, event)
            except Exception as e:
                logger.error(f"Callback error: {str(e)}")

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get specific device by ID"""
        return self._devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Get all devices of specific type"""
        return [d for d in self._devices.values() if d.device_type == device_type]

    def export_inventory(self, filepath: str):
        """Export device inventory to JSON file"""
        inventory = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "devices": [device.to_dict() for device in self._devices.values()],
        }

        with open(filepath, "w") as f:
            json.dump(inventory, f, indent=2)

        logger.info(f"Exported device inventory to {filepath}")
