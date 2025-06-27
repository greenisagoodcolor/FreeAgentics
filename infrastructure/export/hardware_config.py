"""
Hardware Configuration Management

Manages hardware-specific configurations and optimizations for different deployment targets.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities"""

    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    storage_total_gb: float
    storage_available_gb: float
    gpu_available: bool
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    tpu_available: bool = False
    accelerators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores,
                "threads": self.cpu_threads,
            },
            "memory": {
                "total_gb": self.ram_total_gb,
                "available_gb": self.ram_available_gb,
            },
            "storage": {
                "total_gb": self.storage_total_gb,
                "available_gb": self.storage_available_gb,
            },
            "gpu": {
                "available": self.gpu_available,
                "model": self.gpu_model,
                "memory_gb": self.gpu_memory_gb,
            },
            "accelerators": self.accelerators,
        }


@dataclass
class OptimizationProfile:
    """Hardware-specific optimization settings"""

    name: str
    description: str

    # CPU settings
    cpu_threads: int

    # Memory settings
    memory_limit_mb: int

    # CPU settings with defaults
    cpu_affinity: Optional[List[int]] = None

    # Memory settings with defaults
    memory_growth_allowed: bool = True

    # Model settings
    batch_size: int = 1
    use_mixed_precision: bool = False
    quantization_bits: int = 32

    # Inference settings
    inference_threads: int = 4
    inference_timeout_ms: int = 1000

    # Power settings
    power_mode: str = "balanced"  # performance, balanced, efficiency

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "cpu": {"threads": self.cpu_threads, "affinity": self.cpu_affinity},
            "memory": {
                "limit_mb": self.memory_limit_mb,
                "growth_allowed": self.memory_growth_allowed,
            },
            "model": {
                "batch_size": self.batch_size,
                "mixed_precision": self.use_mixed_precision,
                "quantization_bits": self.quantization_bits,
            },
            "inference": {
                "threads": self.inference_threads,
                "timeout_ms": self.inference_timeout_ms,
            },
            "power_mode": self.power_mode,
        }


class HardwareDetector:
    """Detects hardware capabilities of the system"""

    @staticmethod
    def detect_capabilities() -> HardwareCapabilities:
        """Detect current system hardware capabilities"""
        caps = HardwareCapabilities(
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            ram_total_gb=psutil.virtual_memory().total / (1024**3),
            ram_available_gb=psutil.virtual_memory().available / (1024**3),
            storage_total_gb=0.0,
            storage_available_gb=0.0,
            gpu_available=False,
        )

        # Detect storage
        try:
            disk_usage = psutil.disk_usage("/")
            caps.storage_total_gb = disk_usage.total / (1024**3)
            caps.storage_available_gb = disk_usage.free / (1024**3)
        except Exception as e:
            logger.warning(f"Could not detect storage: {e}")

        # Detect GPU
        caps.gpu_available, caps.gpu_model, caps.gpu_memory_gb = HardwareDetector._detect_gpu()
        if caps.gpu_available and caps.gpu_model:
            if "nvidia" in caps.gpu_model.lower():
                caps.accelerators.append("cuda")
            elif "amd" in caps.gpu_model.lower():
                caps.accelerators.append("rocm")
            elif "apple" in caps.gpu_model.lower() or platform.system() == "Darwin":
                caps.accelerators.append("metal")

        # Detect TPU
        if HardwareDetector._detect_tpu():
            caps.tpu_available = True
            caps.accelerators.append("coral_tpu")

        # Platform-specific detections
        if platform.system() == "Linux":
            # Check for Raspberry Pi
            if HardwareDetector._is_raspberry_pi():
                caps.accelerators.append("videocore")

            # Check for Jetson
            if HardwareDetector._is_jetson():
                caps.accelerators.append("tensorrt")

        return caps

    @staticmethod
    def _detect_gpu() -> tuple[bool, Optional[str], Optional[float]]:
        """Detect GPU availability and model"""
        # Try NVIDIA
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    memory_mb = float(parts[1].replace(" MiB", ""))
                    return True, gpu_name, memory_mb / 1024
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try macOS Metal
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "Chipset Model:" in result.stdout:
                    for line in result.stdout.split("\n"):
                        if "Chipset Model:" in line:
                            gpu_name = line.split(":", 1)[1].strip()
                            # Estimate memory for Apple Silicon
                            if "M1" in gpu_name:
                                return True, gpu_name, 8.0
                            elif "M2" in gpu_name:
                                return True, gpu_name, 10.0
                            elif "M3" in gpu_name:
                                return True, gpu_name, 18.0
                            else:
                                return True, gpu_name, None
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        return False, None, None

    @staticmethod
    def _detect_tpu() -> bool:
        """Detect Coral TPU availability"""
        try:
            # Check for Coral USB accelerator
            result = subprocess.run(["lsusb"], capture_output=True, text=True, check=True)
            if "Google" in result.stdout and "Coral" in result.stdout:
                return True

            # Check for PCIe accelerator
            if Path("/dev/apex_0").exists():
                return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return False

    @staticmethod
    def _is_raspberry_pi() -> bool:
        """Check if running on Raspberry Pi"""
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read()
                return "Raspberry Pi" in model
        except Exception:
            return False

    @staticmethod
    def _is_jetson() -> bool:
        """Check if running on NVIDIA Jetson"""
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read()
                return "NVIDIA Jetson" in model or "Jetson" in model
        except Exception:
            return False


class HardwareOptimizer:
    """Generates optimal configurations for specific hardware"""

    def __init__(self) -> None:
        """Initialize optimizer with predefined profiles"""
        self.profiles = {
            "raspberry_pi_4b": self._raspberry_pi_profile(),
            "jetson_nano": self._jetson_nano_profile(),
            "mac_mini_m2": self._mac_mini_profile(),
            "generic_low": self._generic_low_profile(),
            "generic_mid": self._generic_mid_profile(),
            "generic_high": self._generic_high_profile(),
        }

    def get_optimal_profile(self, capabilities: HardwareCapabilities) -> OptimizationProfile:
        """Get optimal profile based on detected capabilities"""
        # Check for specific hardware
        if self._is_raspberry_pi_hardware(capabilities):
            return self.profiles["raspberry_pi_4b"]
        elif self._is_jetson_hardware(capabilities):
            return self.profiles["jetson_nano"]
        elif self._is_mac_hardware(capabilities):
            return self.profiles["mac_mini_m2"]

        # Generic profiles based on resources
        if capabilities.ram_total_gb < 4:
            return self.profiles["generic_low"]
        elif capabilities.ram_total_gb < 16:
            return self.profiles["generic_mid"]
        else:
            return self.profiles["generic_high"]

    def customize_profile(
        self, base_profile: OptimizationProfile, capabilities: HardwareCapabilities
    ) -> OptimizationProfile:
        """Customize profile based on actual capabilities"""
        profile = OptimizationProfile(
            name=f"{base_profile.name}_custom",
            description=f"Customized {base_profile.description}",
            cpu_threads=min(base_profile.cpu_threads, capabilities.cpu_threads),
            memory_limit_mb=int(
                min(
                    base_profile.memory_limit_mb,
                    capabilities.ram_available_gb * 1024 * 0.8,
                    # Use 80% of available
                )
            ),
            batch_size=base_profile.batch_size,
            use_mixed_precision=base_profile.use_mixed_precision,
            quantization_bits=base_profile.quantization_bits,
            inference_threads=min(base_profile.inference_threads, capabilities.cpu_threads // 2),
            inference_timeout_ms=base_profile.inference_timeout_ms,
            power_mode=base_profile.power_mode,
        )

        # Adjust for GPU
        if capabilities.gpu_available:
            profile.use_mixed_precision = True
            if capabilities.gpu_memory_gb and capabilities.gpu_memory_gb < 4:
                profile.batch_size = 1

        # Adjust for low memory
        if capabilities.ram_available_gb < 2:
            profile.quantization_bits = 8
            profile.memory_growth_allowed = False

        return profile

    def _raspberry_pi_profile(self) -> OptimizationProfile:
        """Profile for Raspberry Pi 4B"""
        return OptimizationProfile(
            name="raspberry_pi_4b",
            description="Optimized for Raspberry Pi 4B with 8GB RAM",
            cpu_threads=4,
            cpu_affinity=[0, 1, 2, 3],
            memory_limit_mb=6144,  # 6GB
            memory_growth_allowed=False,
            batch_size=1,
            use_mixed_precision=False,
            quantization_bits=8,
            inference_threads=2,
            inference_timeout_ms=2000,
            power_mode="efficiency",
        )

    def _jetson_nano_profile(self) -> OptimizationProfile:
        """Profile for NVIDIA Jetson Nano"""
        return OptimizationProfile(
            name="jetson_nano",
            description="Optimized for Jetson Nano with CUDA",
            cpu_threads=4,
            memory_limit_mb=3072,  # 3GB
            memory_growth_allowed=False,
            batch_size=1,
            use_mixed_precision=True,
            quantization_bits=8,
            inference_threads=2,
            inference_timeout_ms=1500,
            power_mode="balanced",
        )

    def _mac_mini_profile(self) -> OptimizationProfile:
        """Profile for Mac Mini M2"""
        return OptimizationProfile(
            name="mac_mini_m2",
            description="Optimized for Mac Mini M2 with Metal",
            cpu_threads=8,
            memory_limit_mb=6144,  # 6GB
            memory_growth_allowed=True,
            batch_size=4,
            use_mixed_precision=True,
            quantization_bits=16,
            inference_threads=4,
            inference_timeout_ms=500,
            power_mode="performance",
        )

    def _generic_low_profile(self) -> OptimizationProfile:
        """Generic profile for low-end hardware"""
        return OptimizationProfile(
            name="generic_low",
            description="Generic profile for low-resource systems",
            cpu_threads=2,
            memory_limit_mb=2048,
            memory_growth_allowed=False,
            batch_size=1,
            use_mixed_precision=False,
            quantization_bits=8,
            inference_threads=1,
            inference_timeout_ms=3000,
            power_mode="efficiency",
        )

    def _generic_mid_profile(self) -> OptimizationProfile:
        """Generic profile for mid-range hardware"""
        return OptimizationProfile(
            name="generic_mid",
            description="Generic profile for mid-range systems",
            cpu_threads=4,
            memory_limit_mb=4096,
            memory_growth_allowed=True,
            batch_size=2,
            use_mixed_precision=False,
            quantization_bits=16,
            inference_threads=2,
            inference_timeout_ms=1000,
            power_mode="balanced",
        )

    def _generic_high_profile(self) -> OptimizationProfile:
        """Generic profile for high-end hardware"""
        return OptimizationProfile(
            name="generic_high",
            description="Generic profile for high-performance systems",
            cpu_threads=8,
            memory_limit_mb=8192,
            memory_growth_allowed=True,
            batch_size=8,
            use_mixed_precision=True,
            quantization_bits=32,
            inference_threads=4,
            inference_timeout_ms=500,
            power_mode="performance",
        )

    def _is_raspberry_pi_hardware(self, caps: HardwareCapabilities) -> bool:
        """Check if capabilities match Raspberry Pi"""
        return (
            "arm" in platform.machine().lower()
            and caps.cpu_cores == 4
            and 4 <= caps.ram_total_gb <= 8
            and "videocore" in caps.accelerators
        )

    def _is_jetson_hardware(self, caps: HardwareCapabilities) -> bool:
        """Check if capabilities match Jetson"""
        return (
            "aarch64" in platform.machine().lower()
            and "cuda" in caps.accelerators
            and caps.ram_total_gb <= 8
        )

    def _is_mac_hardware(self, caps: HardwareCapabilities) -> bool:
        """Check if capabilities match Mac"""
        return platform.system() == "Darwin" and "metal" in caps.accelerators


class RuntimeConfigurator:
    """Generates runtime configurations for deployment"""

    @staticmethod
    def generate_runtime_config(
        profile: OptimizationProfile,
        capabilities: HardwareCapabilities,
        agent_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate complete runtime configuration"""
        config = {
            "version": "1.0.0",
            "agent": agent_config,
            "hardware": {
                "detected": capabilities.to_dict(),
                "profile": profile.to_dict(),
            },
            "runtime": {
                "auto_start": True,
                "restart_on_failure": True,
                "max_restarts": 3,
                "health_check_interval": 60,
                "telemetry": {
                    "enabled": True,
                    "interval": 300,
                    "include_performance": True,
                    "include_errors": True,
                },
                "logging": {
                    "level": "INFO",
                    "max_file_size_mb": 100,
                    "max_files": 5,
                    "include_timestamps": True,
                },
                "updates": {
                    "check_enabled": True,
                    "check_interval": 86400,  # Daily
                    "auto_update": False,
                },
            },
            "resource_limits": {
                "cpu": {
                    "threads": profile.cpu_threads,
                    "affinity": profile.cpu_affinity,
                    "max_percent": 80,
                },
                "memory": {
                    "limit_mb": profile.memory_limit_mb,
                    "growth_allowed": profile.memory_growth_allowed,
                    "oom_score_adj": 200,  # Make it more likely to be killed if OOM
                },
                "storage": {
                    "max_cache_mb": 500,
                    "max_logs_mb": 500,
                    "checkpoint_interval": 3600,
                },
            },
            "inference": {
                "model": {
                    "batch_size": profile.batch_size,
                    "mixed_precision": profile.use_mixed_precision,
                    "quantization_bits": profile.quantization_bits,
                },
                "execution": {
                    "threads": profile.inference_threads,
                    "timeout_ms": profile.inference_timeout_ms,
                    "max_queue_size": 100,
                },
                "optimization": {
                    "cache_size_mb": 100,
                    "preload_model": True,
                    "use_mmap": capabilities.ram_total_gb < 8,
                },
            },
            "power": {
                "mode": profile.power_mode,
                "sleep_when_idle": profile.power_mode == "efficiency",
                "wake_on_message": True,
                "battery_saver_threshold": 20,  # Percent
            },
        }

        # Add accelerator-specific settings
        if "cuda" in capabilities.accelerators:
            config["inference"]["cuda"] = {
                "enabled": True,
                "device_id": 0,
                "allow_growth": True,
                "per_process_memory_fraction": 0.8,
            }
        elif "metal" in capabilities.accelerators:
            config["inference"]["metal"] = {
                "enabled": True,
                "max_working_set_size": int(profile.memory_limit_mb * 0.5),
            }
        elif "coral_tpu" in capabilities.accelerators:
            config["inference"]["tpu"] = {
                "enabled": True,
                "device_path": "/dev/apex_0",
                "clock_hz": 500000000,  # 500MHz default
            }

        return config

    @staticmethod
    def validate_config(config: Dict[str, Any], capabilities: HardwareCapabilities) -> List[str]:
        """Validate runtime configuration against hardware capabilities"""
        warnings = []

        # Check memory limits
        if "resource_limits" in config:
            memory_limit = config["resource_limits"]["memory"]["limit_mb"]
            available_mb = capabilities.ram_available_gb * 1024

            if memory_limit > available_mb * 0.9:
                warnings.append(
                    f"Memory limit ({memory_limit}MB) exceeds 90% of available "
                    f"memory ({available_mb:.0f}MB)"
                )

        # Check CPU threads
        if "resource_limits" in config:
            cpu_threads = config["resource_limits"]["cpu"]["threads"]
            if cpu_threads > capabilities.cpu_threads:
                warnings.append(
                    f"Configured CPU threads ({cpu_threads}) exceeds "
                    f"available threads ({capabilities.cpu_threads})"
                )

        # Check accelerator configuration
        if "inference" in config:
            if config["inference"].get("cuda", {}).get("enabled", False):
                if "cuda" not in capabilities.accelerators:
                    warnings.append("CUDA enabled but no CUDA device detected")

            if config["inference"].get("metal", {}).get("enabled", False):
                if "metal" not in capabilities.accelerators:
                    warnings.append("Metal enabled but no Metal device detected")

        return warnings
